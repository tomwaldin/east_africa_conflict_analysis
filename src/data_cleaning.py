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
COMMON_CRS = "EPSG:32637"   # UTM zone 37N (Kenya)

def load_data():
    """
    Loads data of various types from a local directory and structures it as a nested dictionary.
    
    Args:
        None (note that data loading has been hardcoded in this function)
        
    Returns:
        data: A nested dictionary of the data
    """
    
    # Define country configurations
    countries = {
        'eth': {'name': 'Ethiopia', 'code': 'ETH'},
        'ken': {'name': 'Kenya', 'code': 'KEN'},
        'ssd': {'name': 'South Sudan', 'code': 'SSD'},
        'uga': {'name': 'Uganda', 'code': 'UGA'}
    }

    # Define OSM features to download
    osm_features = {
        'police': {'amenity': ['police']},
        'hospitals': {'amenity': ['hospital', 'clinic']},
        'schools': {'amenity': ['school', 'university', 'college']},
        'roads': {'highway': ['motorway', 'trunk', 'primary', 'secondary']}
    }
    
    # Initialize data dict
    data = {code: {} for code in countries.keys()}
    
    # Load ACLED conflict data (shared across countries)
    acled_df = pd.read_csv(DATA_DIR / "Africa_aggregated_data_up_to-2025-08-23.csv")
    acled_gdf = gpd.GeoDataFrame(
        acled_df,
        geometry=gpd.points_from_xy(
            acled_df['CENTROID_LONGITUDE'], 
            acled_df['CENTROID_LATITUDE']
        ),
        crs="EPSG:4326"
    )
    
    # Load country-specific data
    for code, info in countries.items():
        country_name = info['name']
        country_code = info['code']
        
        # ACLED conflict data
        data[code]['acled'] = acled_gdf[acled_gdf['COUNTRY'] == country_name]
        
        # Population data
        pop_file = DATA_DIR / f"{code}_pop_2020_CN_100m_R2025A_v1.tif"
        if pop_file.exists():
            data[code]['population'] = rxr.open_rasterio(pop_file)
        
        # Country and administrative boundaries
        bounds_file = DATA_DIR / f"gadm41_{country_code}_shp" / f"gadm41_{country_code}_2.shp"
        if bounds_file.exists():
            data[code]['bounds'] = gpd.read_file(bounds_file)
        
        # OSM feature data
        for feature_name, tags in osm_features.items():
            feature_file = DATA_DIR / f"{code}_{feature_name}.parquet"
            
            if not feature_file.exists():
                print(f"Downloading {country_name} {feature_name} data from OSM...")
                try:
                    feature_data = ox.features_from_place(country_name, tags=tags)
                    feature_data.to_parquet(feature_file)
                    print(f"Saved {feature_file}")
                except Exception as e:
                    print(f"Failed to download {feature_name} data for {country_name}: {e}")
                    continue
            else:
                print(f"Using existing {feature_file}")
            
            try:
                data[code][feature_name] = gpd.read_parquet(feature_file)
            except Exception as e:
                print(f"Failed to load {feature_name} data for {country_name}: {e}")
        
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
                    # assign EPSG:4326 or whatever the rasters originally come in
                    dataset = dataset.rio.write_crs("EPSG:4326", inplace=True)
                if str(dataset.rio.crs) != COMMON_CRS:
                    dataset = dataset.rio.reproject(COMMON_CRS)
                data_clean[country][dataset_name] = dataset
            
            # Check if it's a GeoDataFrame (has crs attribute)
            elif hasattr(dataset, 'crs'):
                current_crs = dataset.crs
                if current_crs is None:
                    # assume it's EPSG:4326 if missing
                    dataset = dataset.set_crs("EPSG:4326")
                if str(dataset.crs) != COMMON_CRS:
                    dataset = dataset.to_crs(COMMON_CRS)
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