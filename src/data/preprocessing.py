import rasterio
from rasterio.warp import reproject, calculate_default_transform, Resampling
from rasterio.mask import mask
import geopandas as gpd
from ..utils.io_utils import is_multiband, explode_bands
from rasterio.io import MemoryFile
import logging
import numpy as np
import json
import shutil
import os

class Preprocessor:
    '''
    Main class for preprocessing raster files
    - resamples the rasters to the resolution of one reference raster
    - coregisters all the rasters to the grid of reference raster
    '''
    def __init__(self, config):
        self.logger = logging.getLogger('Preprocessor')
        self.config = config
        self.plots_path = config.plots_path
        self.raw_dir = config.raw_data_dir
        self.bands_to_drop = config.drop_bands
        self.processed = config.processed_data_dir
        self.target_crs = config.target_crs
        self.stacked_raster_output = os.path.join(self.processed, 'stacked.tif')
        
    def get_reference_raster(self) -> str:
        '''
        Find the Planet canopy cover raster to be used as reference raster
        '''
        for file in os.listdir(self.processed/'exploded'):
            if 'canopy_cover' in file.lower() and file.lower().endswith('.tif'):
                file_path = os.path.join(self.processed, 'exploded', file)
                self.logger.info('Found the reference raster.')
                return file_path
        raise FileNotFoundError(f'Could not find the reference raster in {self.processed}')
    
    def coregister_raster(self, src_path: str, dest_path: str, ref_path: str) -> None:
        """Reproject and align raster to match reference raster."""
        #open the raster to be aligned
        with rasterio.open(src_path) as src:
            src_transform = src.transform
            #open the reference raster/ raster with target dimensions and resolution
            with rasterio.open(ref_path) as ref:
                dst_crs = ref.crs
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs,    
                    dst_crs,    
                    ref.width,   
                    ref.height,  
                    *ref.bounds,
                )
                dst_kwargs = src.meta.copy()
                dst_kwargs.update({
                                "crs": dst_crs,
                                "transform": dst_transform,
                                "width": dst_width,
                                "height": dst_height,
                                "nodata": 0,
                                "dtype": src.meta["dtype"],})
                #write the aligned raster
                with rasterio.open(dest_path, 'w', **dst_kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest#nearest resampling avoids overstretching rasters, esp given we are moving from 10m to around 3m
                        )

        self.logger.info(f"Coregistered: {os.path.basename(src_path)} to {dest_path}")
        
    def mask(self, output_path: str) -> None:
        '''
        Clips a raster using the overlay shapefile and saves to memory
        '''
        with rasterio.open(output_path) as src:
            crs = src.crs
            gdf = gpd.read_file(self.plots_path).to_crs(crs)
            geoms = [feature["geometry"] for feature in gdf.__geo_interface__["features"]]
            out_image, out_transform = rasterio.mask.mask(src, geoms, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # Keep the same band descriptions & tags
            band_descs = list(src.descriptions)
            tags = src.tags()

        out_clipped = os.path.join(self.processed, 'stacked_clip.tif')
        #write to directory
        with rasterio.open(out_clipped, 'w', **out_meta) as dest:
            dest.write(out_image)
            # Restore band descriptions
            for i, desc in enumerate(band_descs, start=1):
                if desc:
                    dest.set_band_description(i, desc)
            # Restore global tags
            dest.update_tags(**tags)

        self.logger.info(f"Clipped raster saved with {len(band_descs)} bands.")
        #confirmation if the clipped raster was written with band descriptions
        with rasterio.open(out_clipped) as src_check:
            self.logger.info(f"Band descriptions after clipping: {src_check.descriptions}")
        self.logger.info('Clipping complete.')
            
    def filter_rasters(self, all_rasters: list[str]) -> list[str]:
        """Filter out rasters listed in config.drop_bands."""
        return [
            r for r in all_rasters
            if not any(b in os.path.basename(r) for b in self.bands_to_drop)
        ]

            
    def stack_rasters(self, output_path: str):
        """
        Stack all rasters in the processed directory into one GeoTIFF,
        automatically excluding unwanted layers defined in config.drop_bands.
        """
        #iterate over directory to find all bands
        all_rasters = [
            os.path.join(self.processed, f)
            for f in os.listdir(self.processed)
            if f.lower().endswith('.tif') and 'stacked' not in f.lower()
        ]

        # --- Filter out dropped bands ---
        rasters = self.filter_rasters(all_rasters)

        dropped = [os.path.basename(r) for r in all_rasters if r not in rasters]

        self.logger.info(f"Found {len(all_rasters)} rasters, dropping {len(dropped)} based on config.")
        if dropped:
            self.logger.info(f"Dropped rasters: {', '.join(dropped)}")

        if not rasters:
            self.logger.error("No rasters left after filtering. Check your config.drop_bands.")
            return
        #stacking
        with rasterio.open(rasters[0]) as src0:
            meta = src0.meta.copy()
            meta.update(count=len(rasters), dtype='float32')
        # --- Stack rasters ---
        # --- Prepare output directory ---
        with rasterio.open(output_path, 'w', **meta) as dst:
            band_names = []
            for idx, path in enumerate(rasters, start=1):
                with rasterio.open(path) as src:
                    data = src.read(1).astype('float32')
                    dst.write(data, idx)

                    band_name = os.path.splitext(os.path.basename(path))[0]
                    band_names.append(band_name)

                    # Force GDAL-level description
                    dst.set_band_description(idx, band_name)  # underscore used intentionally

            dst.update_tags(band_names=",".join(band_names))
        self.logger.info(f'Stacking of rasters done.')
        

    def run_preprocessing(self) -> None:
        '''
        Main pipeline entry point for preprocessing the rasters
        '''
        temp_explode_dir = os.path.join(self.processed, "exploded")
        os.makedirs(temp_explode_dir, exist_ok=True)
        
        all_rasters = []
        
        #explode multiband files
        for file in os.listdir(self.raw_dir):
            if not file.lower().endswith(('tif', 'tiff')):
                continue
            
            full_path = os.path.join(self.raw_dir, file)
            if is_multiband(full_path):
                explode_bands(full_path, temp_explode_dir, rerun_explode=True)
                self.logger.info(f'Exploding multiband raster {file}')
            else:
                all_rasters.append(full_path)
            
        for fname in os.listdir(temp_explode_dir):
            if fname.lower().endswith('.tif'):
                all_rasters.append(os.path.join(temp_explode_dir, fname))
        #now delete the exploded temp directory
        shutil.rmtree(temp_explode_dir)
                    
        #coregister rasters
        ref_path = self.get_reference_raster()
        coregistered_dir = os.path.join(self.processed, "coregistered")
        os.makedirs(coregistered_dir, exist_ok=True)
        for file in all_rasters:
            out_path = os.path.join(coregistered_dir, os.path.basename(file))
            self.coregister_raster(file, out_path, ref_path)
            
        #stack rasters and then clip them
        self.stack_rasters(self.stacked_raster_output)
        self.mask(self.stacked_raster_output)
        
        self.logger.info('Preprocessing complete!')
        