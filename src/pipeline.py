import logging
from .config import Config

#modules
from .data.preprocessing import Preprocessor
from .data.indices import Indices
from .modelling.modelling import Modeling
from .validate.validation import Validate

class CanopyCoverPipeline:
    '''
    Main orchestration class for the whole canopy cover pipeline
    '''
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = Preprocessor(config)
        self.indices = Indices(config)
        self.modeler = Modeling(config)
        self.validater = Validate(config)
        logging.basicConfig(level=self.config.log_level)
        self.logger = logging.getLogger('CanopyCoverPipeline')
        self.logger.info('Starting pipeline run...')
        
        
    def run_preprocessing(self):
        '''
        Runs the prerocessing pipeline alone
        '''
        self.logger.info('Preprocessing raster files...')
        self.preprocessor.run_preprocessing()
        self.logger.info('Preprocessing done!')

    def run_indices(self):
        '''
        Runs the calculation of indices pipeline alone
        '''
        self.logger.info('Calculating indices...')
        self.indices.run_indices()
        self.logger.info('Indices calculation complete!')
    
    def run_model(self):
        '''
        Runs the modelling pipeline RF or SVM
        '''
        self.logger.info('Starting modelling for rasters...')
        self.modeler.load_data()
        self.modeler.sample_training_data()
        self.modeler.split_data()
        self.modeler.train_rf()
        # self.modeler.train_svm()
        self.modeler.classify_raster()
        
    def run_validate(self):
        '''
        Runs the validation pipeline
        '''
        self.logger.info('Validating outputs...')
        self.validater.validation_run()
        self.logger.info('Validation complete, hope you were happy with results.')
        
    def run_full_pipeline(self):
        '''
        Runs the whole analysis pipeline
        '''
        self.run_preprocessing()
        self.run_indices()
        self.run_model()