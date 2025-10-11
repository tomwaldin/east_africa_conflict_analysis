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