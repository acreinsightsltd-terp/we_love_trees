import os
import rasterio
import logging
import rioxarray as rio
import numpy as np

class Indices:
    '''
    The main class for calculating indices on processed sentinel2 bands
    - NDVI
    - NDWI
    - MNDWI
    - BSI
    '''
    def __init__(self, config):
        self.logger = logging.getLogger('IndicesCalculator')
        self.config = config
        self.processed = config.processed_data_dir
        
    def find_band(self, target_band: str):
        '''
        Finds a specific band in a directory containing sentinel 2 bands
        :param target_band: this is a string containing the band desc to target
        :returns: path of the band matching the regex
        >>> find_band('b2')
        data/processed/sentinel2_b2.tif
        '''
        for file in os.listdir(self.processed):
            if target_band in file.lower() and file.lower().endswith('.tif'):
                band_path = os.path.join(self.processed, file)
                self.logger.info(f'Found band {band_path}')
                return band_path
        raise FileNotFoundError(f'Could not find any specific band matching the {target_band}.')
        
    def normalized_difference(self, first: str, next: str):
        '''
        This is a function to calculate the normalized difference between bands
        :param first: The first band to be used in equation- key in a desc('b2')
        :param next: The second band to be used in equation- key in a desc('b8')
        :returns: Normalized difference tif
        '''
        first_path = self.find_band(first)
        next_path = self.find_band(next)
        if first == 'b8' and next == 'b4':
            out_path = os.path.join(self.processed, 'ndvi.tif')
        elif first == 'b3' and next == 'b8':
            out_path = os.path.join(self.processed, 'ndwi.tif')
        elif first == 'b3' and next == 'b11':
            out_path = os.path.join(self.processed, 'mndwi.tif')
        else:
            raise ValueError('Invalid index inputs.')
        
        with rasterio.open(first_path) as band, rasterio.open(next_path) as band0:
            if band.shape != band0.shape or band.crs !=band0.crs:
                raise ValueError('Ensure crs and shape align for rasters')
            red = band.read(1).astype('float32')
            nir = band0.read(1).astype('float32')
            # Avoid division by zero
            np.seterr(divide='ignore', invalid='ignore')
            ndvi = (red - nir) / (red + nir)
            ndvi = np.clip(ndvi, -1, 1)
            meta = band.meta.copy()
            meta.update({
            "count": 1,
            "dtype": "float32",
            "driver": "GTiff",
            "nodata": -9999
            })
            self.logger.info(f'Writing the index to disk: {out_path}')
            with rasterio.open(out_path, 'w', **meta) as dest:
                dest.write(ndvi, 1)
        return out_path
            
                
    def bsi(self):
        '''
        This is function to calculate the bare soil index
        '''
        out_path = os.path.join(self.processed, 'bsi.tif')
        swir = self.find_band('b11')
        red = self.find_band('b4')
        nir = self.find_band('b8')
        blue = self.find_band('b2')
        
        with rasterio.open(swir) as swir_src, rasterio.open(red) as red_src, rasterio.open(nir) as nir_src, rasterio.open(blue) as blue_src:
            if swir_src.shape != red_src.shape != nir_src.shape != blue_src.shape  or swir_src.crs != red_src.crs != nir_src.crs != blue_src.crs:
                raise ValueError('Ensure crs and shape align for rasters')
            swir = swir_src.read(1).astype('float32')
            red = red_src.read(1).astype('float32')
            nir = nir_src.read(1).astype('float32')
            blue = blue_src.read(1).astype('float32')
            np.seterr(divide='ignore', invalid='ignore')
            bsi = ((swir + red) - (nir +  blue))/((swir + red) + (nir + blue))
            bsi = np.clip(bsi, -1, 1)
            meta = swir_src.meta.copy()
            meta.update({
                "count": 1,
                "dtype": "float32",
                "driver": "GTiff",
                "nodata": -9999
            })
            self.logger.info(f'Writing bsi to disk: {out_path}')
            with rasterio.open(out_path, 'w', **meta) as dest:
                dest.write(bsi, 1)
                
            
 
    def run_indices(self):
        '''
        Main entry point for the indices pipeline
        '''
        #calculate ndwi, ndvi, mndwi, bsi
        self.normalized_difference('b8', 'b4', )
        self.normalized_difference('b3', 'b8')
        self.normalized_difference('b3', 'b11')
        self.bsi()
        
            
        