"""
This file loads and cleans the relevant datasets for analysis. 
Each dataset will need to be reprojected into a common projection.
"""

import pandas as pd
import rioxarray as rxr
import geopandas as gpd
import osmnx as ox
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COMMON_CRS = "EPSG:4326"

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
    acled_df = pd.read_csv(DATA_DIR / "Africa_aggregated_data_up_to-2025-08-23.csv")
    # Convert to gdf
    acled_gdf = gpd.GeoDataFrame(
        acled_df,
        geometry=gpd.points_from_xy(
            acled_df['CENTROID_LONGITUDE'], 
            acled_df['CENTROID_LATITUDE']
        ),
        crs=COMMON_CRS
        )
    #data['eth']['acled'] = acled_gdf[acled_gdf['COUNTRY'] == 'Ethiopia']
    data['ken']['acled'] = acled_gdf[acled_gdf['COUNTRY'] == 'Kenya']
    #data['ssd']['acled'] = acled_gdf[acled_gdf['COUNTRY'] == 'South Sudan']
    #data['uga']['acled'] = acled_gdf[acled_gdf['COUNTRY'] == 'Uganda']

    # Population data
    #data['eth']['population'] = rio.open("data/eth_pop_2020_CN_100m_R2025A_v1.tif")
    data['ken']['population'] = rxr.open_rasterio(DATA_DIR / "ken_pop_2020_CN_100m_R2025A_v1.tif")
    print(type(data['ken']['population'])) 
    #data['ssd']['population'] = rio.open("data/ssd_pop_2020_CN_100m_R2025A_v1.tif")
    #data['uga']['population'] = rio.open("data/uga_pop_2020_CN_100m_R2025A_v1.tif")

    # Country and administritive boundaries
    #data['eth']['bounds'] = gpd.read_file("data/gadm41_ETH_shp")
    data['ken']['bounds'] = gpd.read_file(DATA_DIR / "gadm41_KEN_shp")
    #data['ssd']['bounds'] = gpd.read_file("data/gadm41_SSD_shp")
    #data['uga']['bounds'] = gpd.read_file("data/gadm41_UGA_shp")

    # Define path relative to src directory
    police_file = DATA_DIR / "kenya_police.parquet"
    # Check if file exists, if not download and save
    if not os.path.exists(police_file):
        print("Downloading Kenya police data from OSM...")
        kenya_police = ox.features_from_place("Kenya", tags={
            "amenity": ["police"]
        })
        kenya_police.to_parquet(police_file)
        print(f"Saved {police_file}")
    else:
        print(f"Using existing {police_file}")
    # Load the police data
    data['ken']['police'] = gpd.read_parquet(police_file)
        

    return data

def reproject(data):
    """
    Reprojects all data to a commmon projection

    Args:
        data: A nested dictionary of the data

    Returns:
        data_clean: A nested dictionary of the data, reprojected
    """
    data_clean = {}
    
    for country, datasets in data.items():
        data_clean[country] = {}
        
        for dataset_name, dataset in datasets.items():
            # Check if it's a rioxarray dataset (has rio accessor)
            if hasattr(dataset, 'rio'):
                current_crs = dataset.rio.crs
                if current_crs is None:
                    print(f"Reprojection required: {country} - {dataset_name} (No CRS set)")
                elif str(current_crs) != COMMON_CRS:
                    print(f"Reprojection required: {country} - {dataset_name} (Current: {current_crs})")
                # For now, just pass through without reprojecting
                data_clean[country][dataset_name] = dataset
            
            # Check if it's a GeoDataFrame (has crs attribute)
            elif hasattr(dataset, 'crs'):
                current_crs = dataset.crs
                if current_crs is None:
                    print(f"Reprojection required: {country} - {dataset_name} (No CRS set)")
                elif str(current_crs) != COMMON_CRS:
                    print(f"Reprojection required: {country} - {dataset_name} (Current: {current_crs})")
                # For now, just pass through without reprojecting
                data_clean[country][dataset_name] = dataset
            
            else:
                print(f"Warning: {country} - {dataset_name} has no recognizable CRS attribute")
                data_clean[country][dataset_name] = dataset
    
    return data_clean

def handle_missing_values(data):
    """
    Handles missing values
    
    Args:
        data: A nested dictionary of the data

    Returns:
        data_clean: A nested dictionary of the data with missing values removed or imputed
    """
    data_clean = {}

    for country, datasets in data.items():
        data_clean[country] = {}
        
        for dataset_name, dataset in datasets.items():
            # Only handle acled and police data
            if dataset_name in ['acled', 'police']:
                # Remove rows where geometry (lat/long) is missing
                # Check if it's a GeoDataFrame
                if hasattr(dataset, 'geometry'):
                    original_len = len(dataset)
                    dataset_cleaned = dataset[dataset.geometry.notna()].copy()
                    removed = original_len - len(dataset_cleaned)
                    
                    if removed > 0:
                        print(f"Removed {removed} rows with missing geometry from {country} - {dataset_name}")
                    
                    data_clean[country][dataset_name] = dataset_cleaned
                else:
                    print(f"Warning: {country} - {dataset_name} has no geometry attribute")
                    data_clean[country][dataset_name] = dataset
            
            # Handle population data
            elif dataset_name == 'population':
                # Replace -99999 values with 0
                dataset_cleaned = dataset.where(dataset != -99999.0, 0)
                replaced = (dataset == -99999.0).sum().values
                
                if replaced > 0:
                    print(f"Replaced {replaced} missing values (-99999.0) with 0 in {country} - {dataset_name}")
                
                data_clean[country][dataset_name] = dataset_cleaned
            
            else:
                # For other datasets (bounds), pass through unchanged
                data_clean[country][dataset_name] = dataset
    
    return data_clean

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

    print(f"Data cleaning complete.")
    return data