"""
feature_engineering.py

Aggregates cleaned inputs into model-ready features for analysis.

Assumptions
-----------
- Working with Kenya for now: data['ken']
- data['ken']['acled']     -> GeoDataFrame of ACLED points
- data['ken']['police']    -> GeoDataFrame of police station points
- data['ken']['bounds']    -> GeoDataFrame of GADM level-2 polygons (unique id: GID_2)
- data['ken']['population'] -> rioxarray DataArray (WorldPop) OR path to GeoTIFF

Main entrypoint: feature_engineering_pipeline(data)
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

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def aggregate_conflict_and_police(data: Dict, country_key: str, unit_id_col: str = "GID_2") -> Dict:
    """
    Spatially aggregates ACLED events and police points to the provided admin units.

    Adds:
      data[country_key]['features'] : GeoDataFrame (copy of bounds + counts)
    """
    acled_gdf = data[country_key].get("acled")
    police_gdf = data[country_key].get("police")
    units_gdf = data[country_key].get("bounds")

    if acled_gdf is None or police_gdf is None or units_gdf is None:
        raise ValueError(f"Missing one of acled/police/bounds in data['{country_key}'].")

    # ensure index is not ambiguous
    units = units_gdf.copy()

    # Spatial join ACLED -> units
    print("  - Spatially joining ACLED events to admin units...")
    acled_join = gpd.sjoin(acled_gdf, units, how="left", predicate="within")
    acled_counts = acled_join.groupby(unit_id_col).size()
    units["conflict_count"] = units[unit_id_col].map(acled_counts).fillna(0).astype(int)

    # Spatial join police -> units
    print("  - Spatially joining police stations to admin units...")
    police_join = gpd.sjoin(police_gdf, units, how="left", predicate="within")
    police_counts = police_join.groupby(unit_id_col).size()
    units["police_count"] = units[unit_id_col].map(police_counts).fillna(0).astype(int)

    # store results
    data[country_key]["features"] = units
    print("  -> Aggregation complete: conflict_count & police_count added.")
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

    # get population raster (either DataArray or path)
    pop_obj = pop_raster_obj if pop_raster_obj is not None else data[country_key].get("population")
    if pop_obj is None:
        raise ValueError(f"No population raster found for {country_key} (data['{country_key}']['population']).")

    # ensure CRS alignment: rasterstats expects geometries in same CRS as raster
    # We'll write raster to disk if necessary, then check CRS and reproject features if needed
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


def create_rates_and_densities(data: Dict, country_key: str, unit_id_col: str = "GID_2") -> Dict:
    """
    From the aggregated counts and population, create:
      - area_km2
      - conflict_rate (per 100k)
      - police_density (per 10k people)  *or* per km2 if population missing
      - log transforms for both (small constant added)
    """
    print("Creating rates and densities...")

    features = data[country_key].get("features")
    if features is None:
        raise ValueError("Call aggregate_conflict_and_police and add_population_from_raster first.")

    gdf = features.copy()

    # Ensure geometry is projected in meters for area calc; if not, compute based on current CRS (warn)
    if gdf.crs is None:
        raise ValueError("features GeoDataFrame has no CRS. Reproject to a projected CRS before proceeding.")
    # area in km^2 (CRS must be projected in metres; if EPSG:4326 this will be degrees -> small error)
    # best practice: user has reprojected earlier in data_cleaning; we still compute and warn if not projected
    if "degree" in str(gdf.crs).lower() or str(gdf.crs).upper().startswith("EPSG:4326"):
        print("  WARNING: features CRS appears geographic (degrees). Area will be approximate. Prefer projected CRS.")

    gdf["area_km2"] = gdf.geometry.to_crs(gdf.crs).area / 1e6

    # Conflict rate per 100k population
    if "population" in gdf.columns and gdf["population"].notna().sum() > 0:
        # population currently is raw counts; compute per-100k
        gdf["conflict_rate_per_100k"] = (gdf["conflict_count"] / gdf["population"]) * 100_000
        # police density per 10k people
        gdf["police_per_10k"] = (gdf["police_count"] / gdf["population"]) * 10_000
    else:
        print("  WARNING: population column missing or all NaN. Falling back to densities per km^2.")
        gdf["conflict_rate_per_100k"] = (gdf["conflict_count"] / gdf["area_km2"])  # events per km2
        gdf["police_per_10k"] = gdf["police_count"] / gdf["area_km2"]

    # small constant to avoid log(0)
    eps = 1e-6
    gdf["log_conflict_rate"] = np.log(gdf["conflict_rate_per_100k"].fillna(0) + eps)
    gdf["log_police_density"] = np.log(gdf["police_per_10k"].fillna(0) + eps)

    data[country_key]["features"] = gdf
    print("  -> Rates & log-transforms created.")
    return data


def feature_engineering_pipeline(data: Dict, country_key: str = "ken", unit_id_col: str = "GID_2") -> Dict:
    """
    Master pipeline to create model-ready features for the specified country.
    Steps:
      1. Aggregate conflict & police counts to admin units
      2. Add population (zonal sum from raster)
      3. Create rates/densities and log transforms

    Returns the updated data dict with data[country_key]['features'].
    """
    print("Starting feature engineering pipeline for:", country_key)

    # 1. aggregate counts
    data = aggregate_conflict_and_police(data, country_key, unit_id_col=unit_id_col)

    # 2. add population (uses data[country_key]['population'] by default)
    data = add_population_from_raster(data, country_key, unit_id_col=unit_id_col)

    # 3. create rates / densities / logs
    data = create_rates_and_densities(data, country_key, unit_id_col=unit_id_col)

    print("Feature engineering pipeline complete.")
    return data


# # If run as script for debugging
# if __name__ == "__main__":
#     # Quick local test (will error if run outside your project)
#     from src.data_cleaning import clean_data_pipeline
#     d = clean_data_pipeline()
#     d = feature_engineering_pipeline(d, country_key="ken", unit_id_col="GID_2")
#     print(d["ken"]["features"].head())
