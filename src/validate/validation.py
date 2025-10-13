from rasterio.merge import merge
from rasterio.plot import show
import glob
import os
import logging
import numpy as np
import rasterio
from ..utils.io_utils import coregister_raster, mask

class Validate:
    '''
    Compare canopy height model to the classified tif file to establish agreement and accuracy
    '''
    def __init__(self, config) -> None:
        self.config = config
        self.logger = logging.getLogger('Validation')
        self.classified = config.classified_raster
        self.validation = config.validation_dir
        self.chm_merged = os.path.join(self.validation/'chm_merged.tif')
        self.chm_aligned = os.path.join(self.validation/'chm_aligned.tif')
        self.plots_path = config.plots_path
        self.clipped_chm = os.path.join(self.validation/'chm_clipped.tif')
        self.crs = config.target_crs
            
    def align_rasters(self):
        '''
        Aligns the chm raster to match the classified raster so that they can be comparable in resolution and alignment
        '''
        chm_raster = self.chm_merged
        classified_raster = self.classified
        dest_path = os.path.join(self.validation/'chm_aligned.tif')
        
        #coregister chm
        coregister_raster(chm_raster, dest_path, classified_raster)
        
        #clip the chm to the 239 plot boundaries
        plots = self.plots_path
        out_clipped = os.path.join(self.validation, 'chm_clipped.tif')
        mask(dest_path, plots, out_clipped)
        
    def validation_metrics(self):
        '''
        Validates the classification against the canopy height model, now that the rasters are aligned to same grid
        '''
        with rasterio.open(self.classified) as clf, rasterio.open(self.clipped_chm) as chm:
            clf_data = clf.read(1)
            chm_data = chm.read(1)

        # Mask invalid values
        mask = (clf_data != clf.nodata) & (chm_data != chm.nodata)

        # Define tree condition (e.g., CHM > 2m)
        tree_mask_chm = (chm_data > 1.5)
        tree_mask_clf = (clf_data == 1)  # assuming 1 = tree

        # Intersection
        both_tree = np.sum(tree_mask_chm & tree_mask_clf)
        only_clf_tree = np.sum(tree_mask_clf)
        only_chm_tree = np.sum(tree_mask_chm)

        percent_agreement = (both_tree / only_clf_tree) * 100
        self.logger.info(f"Agreement (classified trees confirmed by CHM): {percent_agreement:.2f}%")

        # Correlation (optional)
        from scipy.stats import pearsonr
        corr, _ = pearsonr(chm_data[mask].flatten(), clf_data[mask].flatten())
        self.logger.info(f"Pearson correlation: {corr:.3f}")

    def validation_run(self):
        '''
        Entry point for validation processes
        '''
        self.align_rasters()
        self.validation_metrics()