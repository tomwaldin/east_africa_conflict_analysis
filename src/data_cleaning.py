"""
This file loads and cleans the relevant datasets for analysis. 
Each dataset will need to be reprojected into a common projection.
"""

import pandas as pd
import rasterio as rio
import geopandas as gpd
import osmnx as ox

# Define a common crs
COMMON_CRS = "EPSG:32637"

def load_data():
    """
    Loads data of various types from a local directory and structures it as a nested dictionary.
    
    Args:
        None (note that data loading has been hardcoded in this function)
        
    Returns:
        data: A nested dictionary of the data
    """

    # Initialise data dict with country codes
    data = {}
    data['eth'] = {}
    data['ken'] = {}
    data['ssd'] = {}
    data['uga'] = {}

    # ACLED conflict data
    acled_df = pd.read_csv("data/Africa_aggregated_data_up_to-2025-08-23.csv")
    # Convert to gdf
    acled_gdf = gpd.GeoDataFrame(
        acled_df,
        geometry=gpd.points_from_xy(
            acled_df['CENTROID_LONGITUDE'], 
            acled_df['CENTROID_LATITUDE']
        ),
        crs=COMMON_CRS
        )
    data['eth']['acled'] = acled_gdf[acled_gdf['COUNTRY'] == 'Ethiopia']
    data['ken']['acled'] = acled_gdf[acled_gdf['COUNTRY'] == 'Kenya']
    data['ssd']['acled'] = acled_gdf[acled_gdf['COUNTRY'] == 'South Sudan']
    data['uga']['acled'] = acled_gdf[acled_gdf['COUNTRY'] == 'Uganda']

    # Population data
    data['eth']['population'] = rio.open("data/eth_pop_2020_CN_100m_R2025A_v1.tif")
    data['ken']['population'] = rio.open("data/ken_pop_2020_CN_100m_R2025A_v1.tif")
    data['ssd']['population'] = rio.open("data/ssd_pop_2020_CN_100m_R2025A_v1.tif")
    data['uga']['population'] = rio.open("data/uga_pop_2020_CN_100m_R2025A_v1.tif")

    # Road data
    #data['eth']['roads'] = ox.features_from_place("Ethiopia", tags={"highway": True})
    #data['ken']['roads'] = ox.features_from_place("Kenya", tags={"highway": True})
    #data['ssd']['roads'] = ox.features_from_place("South Sudan", tags={"highway": True})
    #data['uga']['roads'] = ox.features_from_place("Uganda", tags={"highway": True})

    # Country and administritive boundaries
    data['eth']['bounds'] = gpd.read_file("data/gadm41_ETH_shp")
    data['ken']['bounds'] = gpd.read_file("data/gadm41_KEN_shp")
    data['ssd']['bounds'] = gpd.read_file("data/gadm41_SSD_shp")
    data['uga']['bounds'] = gpd.read_file("data/gadm41_UGA_shp")

    return data

def reproject(data):
    """
    Reprojects all data to a commmon projection

    Args:
        data: A nested dictionary of the data

    Returns:
        data_clean: A nested dictionary of the data, reprojected
    """
    data_clean = data

    return data_clean

def handle_missing_values(data):
    """
    Handles missing values
    
    Args:
        data: A nested dictionary of the data

    Returns:
        data_clean: A nested dictionary of the data with missing values removed or imputed
    """
    data_clean = data

    return data_clean

def handle_outliers(data):
    """
    Handles outiers
    
    Args:
        data: A nested dictionary of the data

    Returns:
        data_clean: A nested dictionary of the data with outliers handled
    """
    data_clean = data

    return data_clean

"""
handle_missing_values:

ACLED: Remove rows with missing lat/lon
Boundaries: Usually complete, maybe check for invalid geometries
Population: Rasters don't have "missing values" in the same way

handle_outliers:

ACLED: Check for impossible coordinates, extreme casualty numbers
Boundaries: Check for invalid geometries
Population: Maybe clip extreme values
"""


def clean_data_pipeline():
    """
    Complete data cleaning pipeline.
    
    Args:
        None
        
    Returns:
        data: A nested dictionary of the cleaned data
    """
    print("Starting data cleaning pipeline...")
    
    data = load_data()
    data = reproject(data)
    data = handle_missing_values(data)
    data = handle_outliers(data)
    
    print(f"Data cleaning complete.")
    return data