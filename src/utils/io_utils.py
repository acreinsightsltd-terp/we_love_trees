import geopandas as gpd
import rasterio
import os
import logging
from rasterio.warp import reproject, calculate_default_transform, Resampling

def load_aoi(filepath: str):
    '''
    Loads the shapefile for the area of interest and converts to earth engine geometry
    :param filepath: path to the shapefile
    
    :return: gpd.GeoDataframe
    
    >>> load_aoi('data/shapefiles/aoi.shp')
    gdf
    '''
    gdf = gpd.read_file(filepath)
    return gdf.to_crs(epsg=32737)

def is_multiband(raster_path):
        """Check if raster has more than one band."""
        with rasterio.open(raster_path) as src:
            return src.count > 1
        
        
def explode_bands(input_path, output_dir, rerun_explode= False):
    '''
    Explodes a multi-band raster into individual single-band rasters
    :param input_path: path to the multi-band raster
    :param output_dir: directory to save the individual band rasters
    
    :return: list of paths to the individual band rasters
    
    >>> explode_bands('data/sentinel.tif')
    ['data/bands/band_1.tif', 'data/bands/band_2.tif', ...]
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        meta.update(count=1)  # Update meta to reflect single band
        
        # Try to get band descriptions (names)
        band_names = src.descriptions if any(src.descriptions) else [f"band{i}" for i in range(1, src.count + 1)]

        for i in range(1, src.count + 1):
            band_data = src.read(i)
            band_meta = meta.copy()

            # Clean up the name (in case of spaces or weird chars)
            band_name = band_names[i - 1] or f"band{i}"
            band_name = band_name.strip().replace(" ", "_").lower()

            band_filename = os.path.join(
                output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_{band_name}.tif"
            )

            with rasterio.open(band_filename, "w", **band_meta) as dst:
                dst.write(band_data, 1)
                
                
def coregister_raster(src_path, dest_path, ref_path):
    """Reproject and align raster to match reference raster."""
    with rasterio.open(src_path) as src:
        src_transform = src.transform

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

            with rasterio.open(dest_path, 'w', **dst_kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )

    logging.info(f"Coregistered: {os.path.basename(src_path)} to {dest_path}")
    
def mask(src_file, plots_path, output_path):
    with rasterio.open(src_file) as src:
        crs = src.crs
        gdf = gpd.read_file(plots_path).to_crs('EPSG:4326')
        geoms = [feature["geometry"] for feature in gdf.__geo_interface__["features"]]
        out_image, out_transform = rasterio.mask.mask(src, geoms, crop=True)
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform})
    with rasterio.open(output_path, 'w', **out_meta) as dest:
        dest.write(out_image)
    logging.info('Clipping complete.')