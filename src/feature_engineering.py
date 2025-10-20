"""
This file aggregates cleaned inputs into model-ready features for analysis.
"""

from pathlib import Path
from typing import Dict, Union
import tempfile
import numpy as np
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import rasterio
from rasterstats import zonal_stats

#PROJECT_ROOT = Path(__file__).parent.parent
#DATA_DIR = PROJECT_ROOT / "data"

def aggregate_to_units(
    data: Dict,
    country_key: str,
    feature_keys: list,
    unit_id_col: str = "GID_2"
) -> Dict:
    """
    Aggregates conflict (ACLED) and multiple OSM features to admin units.

    Parameters
    ----------
    data : dict
        Nested dict, e.g. data['ken']['acled'], data['ken']['police'], ...
    country_key : str
    feature_keys : list of str
        Keys in data[country_key] corresponding to OSM-derived point or line features
        e.g. ["police", "schools", "hospitals"]
    unit_id_col : str
        Column in admin bounds uniquely identifying each unit

    Returns
    -------
    data : dict with
        data[country_key]["features"] = GeoDataFrame with conflict_count + feature counts
    """
    # Get needed layers
    acled_gdf = data[country_key].get("acled")
    units_gdf = data[country_key].get("bounds")

    if acled_gdf is None or units_gdf is None:
        raise ValueError(f"Missing 'acled' or 'bounds' in data['{country_key}'].")

    # Make a copy of the units to store new columns
    units = units_gdf.copy()

    # Aggregate ACLED conflict events
    print("  - Spatially joining ACLED events to admin units...")
    acled_join = gpd.sjoin(acled_gdf, units, how="left", predicate="within")
    acled_counts = acled_join.groupby(unit_id_col).size()
    units["conflict_count"] = units[unit_id_col].map(acled_counts).fillna(0).astype(int)

    # Loop through OSM-derived features
    for key in feature_keys:
        feature_gdf = data[country_key].get(key)
        if feature_gdf is None:
            print(f"  WARNING: data['{country_key}']['{key}'] not found. Skipping.")
            continue

        print(f"  - Spatially joining {key} to admin units...")
        joined = gpd.sjoin(feature_gdf, units, how="left", predicate="intersects")

        # Count number of features per unit
        counts = joined.groupby(unit_id_col).size()

        new_col = f"{key}_count"
        units[new_col] = units[unit_id_col].map(counts).fillna(0).astype(int)

    # Store back to data structure
    data[country_key]["features"] = units
    print("  -> Aggregation complete: conflict_count + feature counts added.")
    return data



def _ensure_raster_on_disk(raster_obj: Union[str, xr.DataArray], tmp_dir: Path) -> str:
    """
    Ensure the population raster is available as a GeoTIFF on disk.
    Accepts either:
      - a file path (str)
      - a rioxarray.DataArray
    Returns path to GeoTIFF.
    """
    if isinstance(raster_obj, str):
        return raster_obj

    # assume rioxarray.DataArray or xarray-like with rio accessor
    if hasattr(raster_obj, "rio"):
        tmp_file = tmp_dir / "tmp_population.tif"
        # write to disk (overwrite if exists)
        raster_obj.rio.to_raster(str(tmp_file))
        return str(tmp_file)

    raise ValueError("population must be either a file path or a rioxarray.DataArray")


def add_population_from_raster(
    data: Dict,
    country_key: str,
    unit_id_col: str = "GID_2",
    pop_raster_obj: Union[None, str] = None
) -> Dict:
    """
    Summarize population raster to admin units and add 'population' column to features.

    - If pop_raster_obj is None, uses data[country_key]['population'].
    - Uses rasterstats.zonal_stats (sum).
    """
    print("Adding population from raster to admin units...")

    # get features (must already exist)
    features = data[country_key].get("features")
    if features is None:
        raise ValueError("Call aggregate_conflict_and_police first to create data[country_key]['features'].")

    # get population raster
    pop_obj = pop_raster_obj if pop_raster_obj is not None else data[country_key].get("population")
    if pop_obj is None:
        raise ValueError(f"No population raster found for {country_key} (data['{country_key}']['population']).")

    # ensure CRS alignment: rasterstats expects geometries in same CRS as raster
    # Write raster to disk if necessary, then check CRS and reproject features if needed
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        pop_path = _ensure_raster_on_disk(pop_obj, tmp_dir)

        # inspect raster CRS
        with rasterio.open(pop_path) as src:
            raster_crs = src.crs
            print(f"  - Population raster CRS: {raster_crs}")

        # reproject features to raster CRS if needed
        features_for_zonal = features.to_crs(raster_crs.to_string()) if raster_crs is not None else features

        # run zonal stats (sum)
        print("  - Running zonal statistics (sum of population per unit). This may take a moment...")
        stats = zonal_stats(
            features_for_zonal,
            pop_path,
            stats=["sum"],
            nodata=None,
            all_touched=False,
            raster_out=False
        )

    # attach population sums (ensure int)
    pop_sums = [s.get("sum", 0) if s is not None else 0 for s in stats]
    features["population"] = np.array(pop_sums).astype(float)  # keep float in case of decimals

    # Protect against zero population: set min to small positive so rates don't divide by zero
    features["population"] = features["population"].replace({0: np.nan})

    data[country_key]["features"] = features
    print("  -> Population added to features (NaN for zero-pop units).")
    return data

def create_rates_and_densities(
    data: Dict,
    country_key: str,
    unit_id_col: str = "GID_2",
    per_capita_base: int = 10000  # 10k for general features, conflict uses 100k
) -> Dict:
    """
    Generalized version:
    - Computes area_km2 (for reference, not used in rates)
    - For conflict_count: creates conflict_rate_per_100k + log
    - For ALL *_count columns (including police, schools, hospitals, etc.):
        -> creates feature_per_10k
        -> creates log_feature_density
    - Assumes population exists; if not, falls back to per km²
    """
    print("Creating rates and densities...")

    features = data[country_key].get("features")
    if features is None:
        raise ValueError("Call aggregate_to_units + add_population_from_raster first.")

    gdf = features.copy()

    # Ensure CRS
    if gdf.crs is None:
        raise ValueError("features GeoDataFrame has no CRS. Reproject first.")

    # Warn if geographic CRS
    if "degree" in str(gdf.crs).lower() or str(gdf.crs).upper().startswith("EPSG:4326"):
        print("  WARNING: CRS is geographic. Area approximation may be inaccurate. Projected CRS is preferable.")

    # Area (not used for rates, but useful to keep)
    gdf["area_km2"] = gdf.geometry.area / 1e6

    # Check population
    has_pop = "population" in gdf.columns and gdf["population"].notna().sum() > 0
    if not has_pop:
        print("  WARNING: No valid population. Falling back to per km² densities.")

    # Small constant for log
    eps = 1e-6

    # 1. Conflict rate per 100k people (special case)
    if has_pop:
        gdf["conflict_rate_per_100k"] = (gdf["conflict_count"] / gdf["population"]) * 100_000
    else:
        gdf["conflict_rate_per_100k"] = gdf["conflict_count"] / gdf["area_km2"]

    gdf["log_conflict_rate"] = np.log(gdf["conflict_rate_per_100k"].fillna(0) + eps)

    # 2. Loop through ALL *other* *_count columns and compute per-capita + log
    count_cols = [col for col in gdf.columns 
                  if col.endswith("_count") and col != "conflict_count"]

    for col in count_cols:
        base_name = col.replace("_count", "")  # e.g. "police", "schools"

        if has_pop:
            rate_col = f"{base_name}_per_{per_capita_base//1000}k"
            gdf[rate_col] = (gdf[col] / gdf["population"]) * per_capita_base
        else:
            rate_col = f"{base_name}_per_km2"
            gdf[rate_col] = gdf[col] / gdf["area_km2"]

        # Log transform
        log_col = f"log_{base_name}_density"
        gdf[log_col] = np.log(gdf[rate_col].fillna(0) + eps)

    data[country_key]["features"] = gdf
    print("  -> Rates & log-transforms created for all feature counts.")
    return data

def feature_engineering_pipeline(
    data: Dict,
    feature_keys: list,
    country_key: str = "ken",
    unit_id_col: str = "GID_2"
) -> Dict:
    """
    Master pipeline to create model-ready features for the specified country.

    Steps:
      1. Aggregate conflict + selected OSM features (e.g., police, schools, hospitals)
      2. Add population via zonal stats from raster
      3. Create rates, densities, log transforms

    Parameters
    ----------
    data : dict
        Nested dictionary of cleaned and loaded data.
    feature_keys : list of str
        Keys in data[country_key] for OSM-derived GeoDataFrames to aggregate.
        e.g. ["police", "schools", "hospitals"]
    country_key : str
        Default "ken", but can be any loaded country.
    unit_id_col : str
        Unique identifier of the admin units, e.g. "GID_2"

    Returns
    -------
    data : dict
        Updated dict with data[country_key]['features'] ready for analysis.
    """

    print(f"Starting feature engineering pipeline for: {country_key}")

    # 1. Aggregate conflict and multiple OSM features
    data = aggregate_to_units(
        data=data,
        country_key=country_key,
        feature_keys=feature_keys,
        unit_id_col=unit_id_col
    )

    # 2. Add population
    data = add_population_from_raster(
        data=data,
        country_key=country_key,
        unit_id_col=unit_id_col
    )

    # 3. Create rates / densities / log transforms
    data = create_rates_and_densities(
        data=data,
        country_key=country_key,
        unit_id_col=unit_id_col
    )

    print("Feature engineering pipeline complete.")
    return data
