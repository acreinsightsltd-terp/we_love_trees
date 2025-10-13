import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Config:
    '''
    The main class for project configurables
    '''
    #----------------------------------------------------------------------
    # PROJECT DETAILS
    #----------------------------------------------------------------------
    project_name: str = 'CanopyCoverAnalysis'
    author: str = 'Kosonei Kipruto Elkana'
    
    #----------------------------------------------------------------------
    # PROJECT DIRECTORIES
    #----------------------------------------------------------------------
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    validation_dir: Path = field(init=False)
    geometry_dir: Path = field(init=False)
    plots_path: Path = field(init=False)
    planet_layer: Path = field(init=False)
    dem_layer: Path = field(init=False)
    sentinel2_layer: Path = field(init=False)
    sentinel1_vv: Path = field(init=False)
    sentinel1_vh: Path = field(init=False)
    sentinel1_vh_vv: Path = field(init=False)
    bsi_layer: Path = field(init=False)
    ndvi_layer: Path = field(init=False)
    ndwi_layer: Path = field(init=False)
    mndwi_layer: Path = field(init=False)
    stacked_raster: Path = field(init=False)
    training_samples: Path = field(init=False)
    classified_raster: Path = field(init=False)
    
    #----------------------------------------------------------------------
    # ANALYSIS AREA
    #----------------------------------------------------------------------
    aoi_path: Path = field(init=False)
    target_crs = "EPSG:32727" #UTM Zone 37S
    drop_bands = [
                    "dem",
  "planet_30m_aboveground_live_carbon_density",
  "planet_30m_canopy_cover_uncertainty_lower_bound",
  "planet_30m_canopy_cover_uncertainty_upper_bound",
  "planet_30m_canopy_height",
  "planet_30m_canopy_height_uncertainty_lower_bound",
  "planet_30m_observation_quality",
  "planet_3m_aboveground_live_carbon_density",
  "planet_3m_aboveground_live_carbon_density_uncertainty_lower_bound",
  "planet_3m_canopy_cover",
  "planet_3m_canopy_cover_uncertainty_lower_bound",
  "planet_3m_canopy_cover_uncertainty_upper_bound",
  "planet_3m_canopy_height_uncertainty_lower_bound",
  "planet_3m_canopy_height_uncertainty_upper_bound",
  "sentinel1_vh",
  "sentinel1_vv",
  "sentinel2_b1",
  "sentinel2_b11",
  "sentinel2_b2",
  "sentinel2_b4",
  "sentinel2_b5",
  "sentinel2_b7",
  "sentinel2_b8",
  "sentinel2_b8a"
    ]
  
    #----------------------------------------------------------------------
    # LOGGING
    #----------------------------------------------------------------------
    log_level: str = 'INFO'
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_prefix: str = field(init=False)
    
    def __post_init__(self):
        self.data_dir = self.base_dir/ 'data'
        self.raw_data_dir = self.data_dir/ 'raw'
        self.processed_data_dir = self.data_dir/ 'preprocessed'
        self.results_dir = self.data_dir/ 'results'
        self.geometry_dir = self.data_dir/ 'geometry'
        self.validation_dir = self.data_dir/ 'validation'
        
        self.aoi_path = self.geometry_dir/ 'aoi.shp'
        self.plots_path = self.geometry_dir/ '239_QC_Green.gpkg'
        self.planet_layer = self.raw_data_dir/ 'planet_3m.nc'
        self.sentinel1_vv = self.raw_data_dir/ 'sentinel1_vv.tif'
        self.sentinel1_vh = self.raw_data_dir/ 'sentinel1_vh.tif'
        self.sentinel1_vh_vv = self.raw_data_dir/ 'sentinel1_vh_vv.tif'
        self.sentinel2_layer = self.raw_data_dir/ 'sentinel2.tif'
        self.dem_layer = self.raw_data_dir/ 'dem.tif'
        self.bsi_layer = self.raw_data_dir/ 'bsi.tif'
        self.ndvi_layer = self.raw_data_dir/ 'ndvi.tif'
        self.mndwi = self.raw_data_dir/ 'mndwi.tif'
        self.ndwi_layer = self.raw_data_dir/ 'ndwi.tif'
        self.bsi_layer = self.raw_data_dir/ 'bsi.tif'
        self.stacked_raster = self.processed_data_dir/ 'stacked_clip.tif'
        self.training_samples = self.geometry_dir/ 'samples.shp'
        self.classified_raster = self.processed_data_dir/'classified'/'rf_base_classified.tif'
        self.output_prefix = f"{self.project_name}_{self.timestamp}"
        