"""
Data Cleaning
Thomas Waldin

This file loads and cleans the relevant datasets for analysis. Each dataset will need to be reprojected into a common projection.
"""

# --- Import required libraries ---

import pandas as pd
import rasterio
import geopandas as gpd
import osmnx as ox
print("OSMnx version:", ox.__version__)
print("OSMnx location:", ox.__file__)


# --- Load datasets from various sources ---

# ACLED conflict data
acled = pd.read_csv("Africa_aggregated_data_up_to-2025-08-23.csv")

# Population data
eth_pop = rasterio.open("eth_pop_2020_CN_100m_R2025A_v1.tif")
ken_pop = rasterio.open("ken_pop_2020_CN_100m_R2025A_v1.tif")
ssd_pop = rasterio.open("ssd_pop_2020_CN_100m_R2025A_v1.tif")
uga_pop = rasterio.open("uga_pop_2020_CN_100m_R2025A_v1.tif")

# Road data
ken_roads = ox.features_from_place(
    "Kenya",
    tags={"highway": True}
)
print(ken_roads)

# Country and administritive boundaries
eth_bounds = gpd.read_file("gadm41_ETH_shp")
ken_bounds = gpd.read_file("gadm41_KEN_shp")
ssd_bounds = gpd.read_file("gadm41_SSD_shp")
uga_bounds = gpd.read_file("gadm41_UGA_shp")

# --- Clean data ---

# --- Reproject data to XX ---