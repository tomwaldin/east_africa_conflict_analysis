"""
modeling.py

Modular spatial modeling pipeline using MGWR for Geographically Weighted Regression (GWR).
Designed to plug into your existing pipeline style:
  data = clean_data_pipeline()
  data = feature_engineering_pipeline(data)
  gdf, model = run_gwr_pipeline(data['ken']['features'], ...)
"""

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

# mgwr imports
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

# Logging / messages consistent with pipelines
def _msg(s: str):
    print(f"[modeling] {s}")


# -------------------------
# Utility / helper methods
# -------------------------
def _check_gdf(gdf: gpd.GeoDataFrame, y_var: str, x_vars: List[str]):
    """Validate inputs exist and are GeoDataFrame."""
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("gdf must be a GeoDataFrame.")
    missing = [c for c in [y_var] + x_vars if c not in gdf.columns]
    if missing:
        raise ValueError(f"Columns missing from gdf: {missing}")


def _warn_if_geographic(gdf: gpd.GeoDataFrame):
    """Warn if CRS is geographic (degrees)."""
    if gdf.crs is None:
        _msg("WARNING: GeoDataFrame has no CRS set.")
    elif gdf.crs.to_string().startswith("EPSG:4326") or "GEOGCS" in str(gdf.crs).upper():
        _msg("WARNING: GeoDataFrame is in a geographic CRS (degrees). "
             "GWR expects a projected CRS in metres. Consider reprojecting.")


def _prepare_gwr_data(
    gdf: gpd.GeoDataFrame,
    y_var: str,
    x_vars: List[str],
    standardize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare arrays for mgwr: coords, y, X.
    Optionally standardize predictors (recommended).

    Returns:
        coords: (n,2) numpy array of x,y coordinates (in CRS units, meters)
        y: (n,1) numpy array
        X: (n, k) numpy array (predictors standardized if requested)
        scaler: the StandardScaler instance used for X (or None)
    """
    # Extract coords from geometry centroids
    # Ensure geometries are not empty
    coords = np.vstack([ (pt.x, pt.y) for pt in gdf.geometry.centroid.to_numpy() ])
    y = gdf[y_var].to_numpy().reshape((-1, 1))
    X = gdf[x_vars].to_numpy().astype(float)

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return coords, y, X, scaler


def _select_bandwidth(coords, y, X, kernel="gaussian", bw_min=None, bw_max=None):
    """
    Select an optimal bandwidth using mgwr.sel_bw.Sel_BW.
    Now supports optional bw_min and bw_max to prevent crashes on small datasets.
    """
    _msg("Selecting bandwidth (this may take a moment)...")
    selector = Sel_BW(coords, y, X, kernel=kernel)

    # If user provided bounds, pass them through (prevents mgwr crash on small N)
    if bw_min is not None or bw_max is not None:
        bw = selector.search(
            bw_min=bw_min,
            bw_max=bw_max
        )
    else:
        # Default behavior (original code path)
        bw = selector.search()

    _msg(f"Selected bandwidth: {bw}")
    return bw



def _fit_gwr(coords: np.ndarray, y: np.ndarray, X: np.ndarray, bandwidth: float, kernel: str = "gaussian"):
    """Fit the GWR model and return the results object."""
    _msg("Fitting GWR model...")
    model = GWR(coords, y, X, bw=bandwidth, kernel=kernel)
    results = model.fit()
    _msg("GWR fit complete.")
    return results


def _attach_results_to_gdf(
    gdf: gpd.GeoDataFrame,
    gwr_results,
    x_vars: List[str],
    y_var: str,
    prefix: str = ""
) -> gpd.GeoDataFrame:
    """
    Attach GWR outputs to the GeoDataFrame.
    Adds:
      - local coefficients for each predictor (col names: <prefix><var>_coef)
      - local_intercept (prefix + "intercept")
      - local_R2 (prefix + "local_R2")
      - gwr_pred (prefix + "gwr_pred")
      - gwr_resid (prefix + "gwr_resid")
    """
    # params: shape (n, k+1) where first column is intercept
    try:
        params = gwr_results.params  # numpy array
        predy = gwr_results.predy
        resid = gwr_results.resid_response
        localR2 = getattr(gwr_results, "localR2", None)
    except Exception as e:
        raise RuntimeError(f"Unexpected mgwr results structure: {e}")

    n, p = params.shape
    # intercept
    gdf[f"{prefix}intercept"] = params[:, 0]

    # coefficients for predictors (params columns 1..)
    for i, var in enumerate(x_vars):
        colname = f"{prefix}{var}_coef"
        gdf[colname] = params[:, i + 1]

    # predictions and residuals
    gdf[f"{prefix}gwr_pred"] = predy.reshape(-1)
    gdf[f"{prefix}gwr_resid"] = resid.reshape(-1)

    # local R2 if available
    if localR2 is not None:
        gdf[f"{prefix}local_R2"] = localR2
    else:
        gdf[f"{prefix}local_R2"] = np.nan

    return gdf


# -------------------------
# Public pipeline functions
# -------------------------
def run_ols(gdf: gpd.GeoDataFrame, y_var: str, x_vars: List[str]):
    """
    Run a simple OLS for baseline comparison.
    Returns the fitted statsmodels results object.
    """
    _check_gdf(gdf, y_var, x_vars)
    # Build formula
    formula = y_var + " ~ " + " + ".join(x_vars)
    _msg(f"Running OLS: {formula}")
    model = smf.ols(formula=formula, data=gdf)
    results = model.fit()
    _msg("OLS complete.")
    return results


# def run_gwr_pipeline(
#     gdf: gpd.GeoDataFrame,
#     y_var: str = "log_conflict_rate_per_100k",
#     x_vars: Optional[List[str]] = None,
#     standardize: bool = True,
#     kernel: str = "gaussian",
#     auto_select_bw: bool = True,
#     bandwidth: Optional[float] = None
# ):
#     """
#     Run a modular GWR pipeline.

#     Parameters
#     ----------
#     gdf : GeoDataFrame
#         GeoDataFrame with geometry and variables already prepared.
#     y_var : str
#         Dependent variable column name (use log version).
#     x_vars : list[str]
#         List of predictor variable column names.
#     standardize : bool
#         Standardize predictors (recommended).
#     kernel : str
#         Kernel type for GWR (e.g., 'gaussian').
#     auto_select_bw : bool
#         If True, use Sel_BW to choose bandwidth. Otherwise use provided bandwidth.
#     bandwidth : float or None
#         If auto_select_bw is False, you must provide a bandwidth.

#     Returns
#     -------
#     tuple (gdf_with_results, gwr_results, bw)
#     - gdf_with_results : GeoDataFrame with local coefficients, local_R2, preds, resids appended
#     - gwr_results : mgwr results object
#     - bw : selected bandwidth
#     """
#     if x_vars is None:
#         x_vars = ["police_per_10k"]

#     _check_gdf(gdf, y_var, x_vars)
#     _warn_if_geographic(gdf)

#     # Prepare OLS baseline
#     _msg("Running baseline OLS for comparison...")
#     ols_res = run_ols(gdf, y_var, x_vars)
#     _msg(f"OLS R-squared: {ols_res.rsquared:.4f}")

#     # Prepare arrays for mgwr
#     coords, y, X, scaler = _prepare_gwr_data(gdf, y_var, x_vars, standardize=standardize)

#     # Bandwidth selection
#     if auto_select_bw:
#         bw = _select_bandwidth(coords, y, X, kernel=kernel)
#     else:
#         if bandwidth is None:
#             raise ValueError("If auto_select_bw is False, you must provide a bandwidth.")
#         bw = bandwidth
#         _msg(f"Using provided bandwidth: {bw}")

#     # Fit GWR
#     gwr_res = _fit_gwr(coords, y, X, bandwidth=bw, kernel=kernel)

#     # Attach results
#     gdf_out = gdf.copy()
#     gdf_out = _attach_results_to_gdf(gdf_out, gwr_res, x_vars, y_var, prefix="gwr_")

#     # Print some diagnostics
#     try:
#         aicc = getattr(gwr_res, "aicc", None)
#         if aicc is not None:
#             _msg(f"GWR AICc: {aicc:.3f}")
#     except Exception:
#         pass

#     # Return updated gdf and the results object for further inspection
#     return gdf_out, gwr_res, bw

def run_gwr_pipeline(
    gdf: gpd.GeoDataFrame,
    y_var: str = "log_conflict_rate_per_100k",
    x_vars: Optional[List[str]] = None,
    standardize: bool = True,
    kernel: str = "gaussian",
    auto_select_bw: bool = True,
    bandwidth: Optional[float] = None,
    bw_min: Optional[int] = None,   # <-- ADDED
    bw_max: Optional[int] = None    # <-- ADDED
):
    """
    Run a modular GWR pipeline.
    """
    if x_vars is None:
        x_vars = ["police_per_10k"]

    _check_gdf(gdf, y_var, x_vars)
    _warn_if_geographic(gdf)

    # Prepare OLS baseline
    _msg("Running baseline OLS for comparison...")
    ols_res = run_ols(gdf, y_var, x_vars)
    _msg(f"OLS R-squared: {ols_res.rsquared:.4f}")

    # Prepare arrays for mgwr
    coords, y, X, scaler = _prepare_gwr_data(gdf, y_var, x_vars, standardize=standardize)

    # Bandwidth selection
    if auto_select_bw:
        bw = _select_bandwidth(
            coords, y, X, 
            kernel=kernel,
            bw_min=bw_min,     # <-- PASSED THROUGH
            bw_max=bw_max      # <-- PASSED THROUGH
        )
    else:
        if bandwidth is None:
            raise ValueError("If auto_select_bw is False, you must provide a bandwidth.")
        bw = bandwidth
        _msg(f"Using provided bandwidth: {bw}")

    # Fit GWR
    gwr_res = _fit_gwr(coords, y, X, bandwidth=bw, kernel=kernel)

    # Attach results
    gdf_out = gdf.copy()
    gdf_out = _attach_results_to_gdf(gdf_out, gwr_res, x_vars, y_var, prefix="gwr_")

    # Print some diagnostics
    try:
        aicc = getattr(gwr_res, "aicc", None)
        if aicc is not None:
            _msg(f"GWR AICc: {aicc:.3f}")
    except Exception:
        pass

    return gdf_out, gwr_res, bw


# If run as script, provide a tiny smoke test (won't execute on import)
# if __name__ == "__main__":
#     _msg("This module is intended to be imported and used in your notebooks.")
#     _msg("Example usage:")
#     _msg("from src.modeling import run_gwr_pipeline")
#     _msg("gdf = data['ken']['features']")
#     _msg("gdf_out, results, bw = run_gwr_pipeline(gdf, y_var='log_conflict_rate_per_100k', x_vars=['police_per_10k'])")
