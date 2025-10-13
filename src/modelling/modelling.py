import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

class Modeling:
    '''
    Class containing all modelling steps for the classification models using rf and svm
    '''
    def __init__(self, config): 
        self.logger = logging.getLogger('ModellingModeller')
        self.config = config
        self.raster_path = config.stacked_raster
        self.training_path = config.training_samples
        self.processed = config.processed_data_dir

    def load_data(self) -> None:
        '''
        Loads all datasets required in the modelling. These are:
        - raster containing stacked bands
        - a shapefile containg the training labels
        - counts the number of bands in the raster and gets their respective names
        - and relevant directories where the results of the steps will be saved
        '''
        self.logger.info("Loading raster and training data...")
        self.raster = rasterio.open(self.raster_path)
        self.training = gpd.read_file(self.training_path).to_crs(self.raster.crs)
        self.bands = self.raster.count
        #we will need the band names in printing out the importances
        with rasterio.open(self.raster_path) as src:
            # Option 1: Get from band descriptions
            band_names = list(src.descriptions)
            # Option 2 (fallback): Get from metadata tags
            if not all(band_names):
                tags = src.tags()
                if "band_names" in tags:
                    band_names = tags["band_names"].split(",")
        self.feature_names = band_names
        self.logger.info(f"Loaded raster with {self.bands} bands.")

    def sample_training_data(self) -> None:
        '''
        Processed the training labels to find relevant labels and then counts how many points we have per class. This step also recodes all the non tree classes to 0
        '''
        self.logger.info("Sampling training data...")
        X, y = [], []
        for _, row in tqdm(self.training.iterrows(), total=self.training.shape[0]):
            geom = [row.geometry.__geo_interface__]
            try:
                out_image, _ = rasterio.mask.mask(self.raster, geom, crop=True)
                data = out_image.reshape(self.bands, -1).T
                data = data[~np.any(np.isnan(data), axis=1)]
                # Recode to binary (1 = tree, 0 = non-tree)
                class_val = 1 if row['class_id'] == 1 else 0
                labels = np.full(data.shape[0], class_val)
                X.append(data)
                y.append(labels)
            except Exception as e:
                self.logger.info(f"Skipping geometry: {e}")


        self.X = np.vstack(X)
        self.y = np.hstack(y)

        self.logger.info(f"Total samples: {len(self.y)}")
        self.logger.info(pd.Series(self.y).value_counts())

    def split_data(self) -> None:
        '''
        Our usual train test splitting for model performance evaluation later
        '''
        self.logger.info("Splitting into train/test...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

    def train_rf(self, tuned: bool=False) -> None:
        """Train Random Forest classifier and manage the training pipeline."""
        self.logger.info(f"Training Random Forest ({'tuned' if tuned else 'default'})...")
        rf = self._initialize_rf(tuned)
        self._evaluate_model(rf)
        band_names = getattr(self, 'feature_names', [f'Band {i+1}' for i in range(self.X_train.shape[1])])
        self._analyze_feature_importance(rf, band_names)
        self._analyze_permutation_importance(rf, band_names)
        self._save_model(rf, tuned)


    # --- 1. Train/initialize the model ---
    def _initialize_rf(self, tuned) -> RandomForestClassifier:
        rf = RandomForestClassifier(random_state=42)
        if tuned:
            params = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
            }
            grid = GridSearchCV(rf, params, cv=3, n_jobs=-1, verbose=1)
            grid.fit(self.X_train, self.y_train)
            rf = grid.best_estimator_
        else:
            rf.fit(self.X_train, self.y_train)
        return rf


    # --- 2. Evaluate model performance ---
    def _evaluate_model(self, rf) -> None:
        preds = rf.predict(self.X_test)
        report = classification_report(
            self.y_test, preds,
            target_names=["Non-tree", "Tree"],
            digits=3, zero_division=0
        )
        cm = confusion_matrix(self.y_test, preds)
        self.logger.info(f"\nClassification Report:\n{report}")
        self.logger.info(f"Confusion Matrix:\n{cm}")

        # Optional visualization
        ConfusionMatrixDisplay.from_estimator(rf, self.X_test, self.y_test)
        plt.title("Confusion Matrix - RF")
        plt.show()


    # --- 3. Standard feature importance ---
    def _analyze_feature_importance(self, rf, band_names) -> None:
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        self.logger.info("\nRandom Forest Feature Importances (descending):")
        for i in indices:
            self.logger.info(f"{band_names[i]}: {importances[i]:.4f}")


    # --- 4. Permutation importance and drop decision ---
    def _analyze_permutation_importance(self, rf, band_names)-> None:
        self.logger.info("\nCalculating permutation importance...")
        result = permutation_importance(rf, self.X_test, self.y_test, n_repeats=10, random_state=42)
        importances_mean = result.importances_mean
        importances_std = result.importances_std
        sorted_idx = importances_mean.argsort()[::-1]

        self.logger.info("Permutation Importances (descending):")
        for i in sorted_idx:
            self.logger.info(f"{band_names[i]}: {importances_mean[i]:.4f} ± {importances_std[i]:.4f}")

        threshold = 0.01 * importances_mean.max()
        low_features = [band_names[i] for i, v in enumerate(importances_mean) if v < threshold or v < 0]

        if low_features:
            self.logger.info(f"\nFeatures suggested for removal (below {threshold:.4f}):")
            for f in low_features:
                self.logger.info(f" - {f}")
        else:
            self.logger.info("\nNo features suggested for removal.")

        self._save_drop_suggestions(low_features)

        # Plot
        plt.figure(figsize=(8, 5))
        plt.title("Permutation Feature Importance")
        plt.bar(range(len(sorted_idx)), importances_mean[sorted_idx], yerr=importances_std[sorted_idx])
        plt.xticks(range(len(sorted_idx)), [band_names[i] for i in sorted_idx], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


    # --- 5. Save model and drop config ---
    def _save_model(self, rf, tuned) ->None:
        model_path = f"{self.processed}/models/rf_{'tuned' if tuned else 'base'}.pkl"
        joblib.dump(rf, model_path)
        self.logger.info(f"Model saved at: {model_path}")


    def _save_drop_suggestions(self, low_features) -> None:
        import json
        config_path = "features_to_drop.json"
        with open(config_path, "w") as f:
            json.dump(low_features, f, indent=2)
        self.logger.info(f"Saved drop suggestions to {config_path}")



#-----------------------------------SVM Model-----------------------------------------
    def train_svm(self, tuned=True) -> None:
        '''
        Trains a support vector machine classifier to rival the random forest and then compare which model would be best for our usecase. 
        Also gives the confusion matrix and classification report to determine the performance of the model.
        :param tuned: This parameter determines whether to tune the hyperparameters or use default base model.
        :returns: Saves the svm model to directory
        '''
        self.logger.info(f"Training SVM ({'tuned' if tuned else 'default'})...")
        svm = SVC(kernel='rbf', probability=True)
        if tuned:
            params = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 1],
            }
            grid = GridSearchCV(svm, params, cv=3, n_jobs=-1, verbose=1)
            grid.fit(self.X_train, self.y_train)
            svm = grid.best_estimator_
        else:
            svm.fit(self.X_train, self.y_train)

        self.svm_model = svm
        preds = svm.predict(self.X_test)
        report = classification_report(
            self.y_test,
            preds,
            target_names=["Non-tree", "Tree"],
            digits=3,
            zero_division=0
        )
        cm = confusion_matrix(self.y_test, preds)

        self.logger.info(f"\n + {report}")
        self.logger.info(f"Confusion Matrix:\n{cm}")
        joblib.dump(svm, f"{self.processed}/models/svm_{'tuned' if tuned else 'base'}.pkl")


#-------------------------------Run Everything----------------------------------------
    def classify_raster(self, model_name='rf_base') -> None:
        '''
        This is the entry point to running the operations above. Here is where the trained model is used to classify the image raster.
        :param model_name: The name of the saved model. Defaults to rf_base model. Could be changed to rf_tuned or svm_base/ svm_tuned
        :returns: A classified image tif and saves to memory with name
        >>> classify_raster(model_name='rf_base')
        Classification saved at rf_base_classified.tif
        '''
        self.logger.info(f"Classifying raster with {model_name} model...")
        model = joblib.load(f"{self.processed}/models/{model_name}.pkl")
        #raster dimensions 
        img = self.raster.read().reshape(self.bands, -1).T
        img = np.nan_to_num(img)
        preds = model.predict(img)
        #read meta for saving
        classified = preds.reshape(self.raster.height, self.raster.width)
        meta = self.raster.meta
        meta.update(count=1, dtype='int16')
        #write classified
        out_path = f"{self.processed}/classified/{model_name}_classified.tif"
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(classified, 1)

        self.logger.info(f"Classification saved at {out_path}")
