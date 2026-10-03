#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
import gc
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy import stats, ndimage

warnings.filterwarnings("ignore")

# =============================================================================
# PORTABLE PROJECT PATHS
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "outputs"

# =============================================================================
# USER SETTINGS
# =============================================================================

class CFG:
    # Daily precipitation.
    PRECIP_NC_FILE: Path = DATA_ROOT / "precipitation" / "CN05.1_Pre_1961_2023_daily_025x025.nc"

    # Final recomputed ENSO phase-event file.
    # This file should contain year/month information and the final phase column.
    ENSO_EXCEL: Path = (
        OUTPUT_ROOT / "01_enso_classification" /
        "ENSO_article_figS1_S2_NOAA_sensitivity_1961_2023_v2_panelwise.xlsx"
    )
    # None means automatically search all sheets and choose the monthly ENSO-event sheet.
    ENSO_SHEET_NAME: Optional[str] = None
    # The final ENSO phase column produced by your recomputation workflow.
    # If main-phase exists, it will be used first; otherwise the code falls back
    # to automatic phase-column detection.
    ENSO_PHASE_COLUMN: Optional[str] = "main-phase"

    # Analysis period and metric definitions.
    START_YEAR: int = 1961
    END_YEAR: int = 2023
    ANOMALY_BASELINE_START_YEAR: int = 1961
    ANOMALY_BASELINE_END_YEAR: int = 2023

    # MEP95 definition.
    WET_DAY_MM: float = 1.0
    MEP_DAILY_PERCENTILE: float = 95.0

    # Boundary files.
    SHP_GLOBAL: Path = DATA_ROOT / "boundaries" / "global_countries.shp"
    SHP_TEN_DASH: Path = DATA_ROOT / "boundaries" / "ten_dash_line.shp"
    SHP_PROVINCE: Path = DATA_ROOT / "boundaries" / "province_boundaries.shp"
    BASIN_GPKG: Path = DATA_ROOT / "rivers" / "all_river_levels_inside_six_basins.shp"

    PHASES: Tuple[str, str] = ("El Nino", "La Nina")

    # Figure 4 month/threshold design.
    # The reference-paper layout uses all 12 months in a 6 × 4 panel:
    # Dec/Jan, Nov/Feb, Oct/Mar, Sep/Apr, Aug/May, Jul/Jun;
    # each month has La Niña and El Niño panels.
    WARM_MONTHS: Tuple[int, ...] = tuple(range(1, 13))
    PLOT_LAYOUT: str = "paper_4cols"

    # Main and supporting thresholds.
    # Figure4 = MEP95 q90.
    # FigureS4 = MEP95 q95.
    # FigureS5 = Rx5day q90, used as a metric-sensitivity supporting figure.
    THRESHOLD_SPECS: Tuple[Dict[str, object], ...] = (
        {"figure_id": "Figure4", "metric": "MEP95", "threshold": 90},
        {"figure_id": "FigureS4", "metric": "MEP95", "threshold": 95},
        {"figure_id": "FigureS5", "metric": "Rx5day", "threshold": 90},
    )

    # Significance.
    # Wilson-score CDF logic follows the previous FAR/RR figure:
    # the Neutral event probability is significant when it falls outside the
    # 95% Wilson confidence interval of the ENSO-phase event probability.
    ALPHA: float = 0.05
    SIGNIFICANCE_METHOD: str = "wilson_cdf"
    CONFIDENCE_LEVEL: float = 0.95

    # Optional extra spatial control. For strict consistency with the previous
    # FAR/RR Wilson-CDF logic, keep this False. If reviewers request spatial
    # multiple-testing control, set this to True; then Wilson-significant cells
    # are additionally filtered by FDR from a two-proportion p-value.
    APPLY_FDR_AFTER_WILSON: bool = False
    USE_FDR: bool = False

    MIN_PHASE_N: int = 3
    MIN_NEUTRAL_N: int = 3

    # If True, after significance screening keep only the largest connected
    # significant region in each panel. Set False to outline all significant cells.
    KEEP_LARGEST_SIGNIFICANT_CLUSTER: bool = False
    CLUSTER_CONNECTIVITY: int = 2

    # Haldane-Anscombe probability correction for the monthly spatial footprint.
    # When enabled, both phase and Neutral probabilities use (c+0.5)/(n+1)
    # for all valid cells so the gridded monthly RR field stays finite.
    USE_HALDANE_CORRECTION: bool = True

    # Output folders.
    OUT_DIR: Path = OUTPUT_ROOT / "04_spatial_rr"
    NC_DIR_NAME: str = "00_netcdf"
    FIG_DIR_NAME: str = "01_figures"
    TABLE_DIR_NAME: str = "02_tables"

    # Build monthly grid metric only once. If the NetCDF already exists,
    # keep this False to avoid overwriting a file that may be open/locked.
    REBUILD_MONTHLY_METRIC_NC: bool = False
    GRID_METRIC_NC: Path = None

    # Two-stage Figure 4 workflow:
    #   Stage 1: calculate q90/q95 spatial RR/significance results and save them as NetCDF.
    #   Stage 2: reopen the saved RR/significance NetCDF and plot from it.
    #
    # Recommended:
    #   First run : RECALCULATE_RR_NC=True,  PLOT_FROM_RR_NC=True
    #   Replot   : RECALCULATE_RR_NC=False, PLOT_FROM_RR_NC=True
    RECALCULATE_RR_NC: bool = True
    PLOT_FROM_RR_NC: bool = True

    # When True and RECALCULATE_RR_NC=True, overwrite the RR/significance NC path
    # if it is not locked. If the file is locked, safe_write_netcdf writes a
    # timestamped alternative and the plotting stage uses that new file.
    OVERWRITE_RR_NC: bool = True

    # When using an existing grid metric NetCDF, overwrite/attach phase_code
    # from the finalized ENSO Excel file. This ensures that Fig. 4 always uses
    # the same final main-phase classification even if the old NetCDF does not
    # contain phase_code or was created before the ENSO file path was updated.
    OVERRIDE_PHASE_CODE_FROM_ENSO_FILE: bool = True

    # Map and export.
    MAP_EXTENT: Tuple[float, float, float, float] = (73.0, 136.0, 3.0, 55.0)
    FIG_SIZE: Tuple[float, float] = (16.0, 22.5)
    DPI: int = 500
    EXPORT_TIFF: bool = True
    EXPORT_PDF: bool = True
    FONT_FAMILY: str = "Times New Roman"

    # log2(RR) color design.
    LOG2RR_BOUNDS: Tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
    LOG2RR_COLORS: Tuple[str, ...] = (
        "#542788", "#8073ac", "#b2abd2", "#e6e6e6",
        "#fff7bc", "#fee391", "#fdb863", "#e66101"
    )

    # Plot display mode.
    # True  : show all valid log2(RR) values as background and outline significant footprints.
    # False : show only significant cells. If no cells pass FDR, maps will look blank.
    SHOW_ALL_RR_AS_BACKGROUND: bool = False

    DRAW_SIGNIFICANT_FOOTPRINT_OUTLINE: bool = True
    SIGNIFICANT_OUTLINE_COLOR: str = "black"
    SIGNIFICANT_OUTLINE_WIDTH: float = 0.95

    # Display "never" categories, following the reference paper's monthly RR maps.
    # These categories are shown only where the final Wilson-significant mask is True.
    SHOW_NEVER_CATEGORIES: bool = True
    NEVER_NEUTRAL_COLOR: str = "#d8d3e8"  # threshold never occurred in Neutral
    NEVER_EVENT_COLOR: str = "#f07b73"    # threshold never occurred in ENSO phase
    NEVER_EITHER_COLOR: str = "#b8c4b8"   # threshold never occurred in either group

    # Figure 4 follows the reference article's map legend: plot RR directly.
    PLOT_VALUE: str = "RR"
    RR_COLOR_BOUNDS: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)

    # China-only display and overlays.
    FILL_CHINA_WHITE: bool = True
    DRAW_PROVINCE_DASHED: bool = True
    DRAW_SIX_BASIN_LABELS: bool = True
    SIX_BASIN_LABELS: Dict[str, Tuple[float, float]] = None


MONTH_LABELS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
PHASE_LABELS = {"El Nino": "El Niño", "La Nina": "La Niña"}
CFG.SIX_BASIN_LABELS = {
    "Songhua": (126.0, 46.5),
    "Haihe": (116.5, 39.2),
    "Yellow": (108.0, 35.5),
    "Huaihe": (116.4, 33.4),
    "Yangtze": (111.0, 29.2),
    "Pearl": (111.4, 23.1),
}
CFG.GRID_METRIC_NC = CFG.OUT_DIR / CFG.NC_DIR_NAME / "Figure4_grid_monthly_MEP95_RX5day_anomaly_CN.nc"


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def output_dirs() -> Tuple[Path, Path, Path]:
    nc_dir = CFG.OUT_DIR / CFG.NC_DIR_NAME
    fig_dir = CFG.OUT_DIR / CFG.FIG_DIR_NAME
    tab_dir = CFG.OUT_DIR / CFG.TABLE_DIR_NAME
    for p in (nc_dir, fig_dir, tab_dir):
        p.mkdir(parents=True, exist_ok=True)
    return nc_dir, fig_dir, tab_dir


def set_figure_style() -> None:
    mpl.rcParams.update({
        "font.family": CFG.FONT_FAMILY,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def _sanitize_attr_value(value):
    """
    Convert attribute values to NetCDF-safe types.

    NetCDF4 and h5netcdf do not support boolean attributes directly. This helper
    converts bools to int8-like 0/1 values and converts unsupported objects to
    strings. It is applied to both dataset-level and variable-level attributes.
    """
    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (str, int, float, np.ndarray, list, tuple)):
        return value

    if value is None:
        return "None"

    return str(value)


def sanitize_dataset_for_netcdf(ds: xr.Dataset) -> xr.Dataset:
    """
    Return a shallow copy of ds with NetCDF-safe attributes.
    """
    ds2 = ds.copy()
    ds2.attrs = {str(k): _sanitize_attr_value(v) for k, v in ds2.attrs.items()}

    for name in list(ds2.data_vars) + list(ds2.coords):
        ds2[name].attrs = {
            str(k): _sanitize_attr_value(v)
            for k, v in ds2[name].attrs.items()
        }

    return ds2


def safe_write_netcdf(ds: xr.Dataset, out_path: Path) -> Path:
    """
    Write NetCDF safely on Windows.

    Why this is needed:
    - Windows often locks NetCDF files that are open in Python, Panoply, QGIS,
      ArcGIS, or another notebook kernel.
    - If the target is locked, writing to the same file or to a fixed *_new.nc
      file may still fail if that file also exists or is locked.
    - This function therefore tries the target first, then writes to unique
      timestamped alternatives.

    It does not delete or overwrite a locked file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert boolean/object attributes to NetCDF-safe values before writing.
    ds = sanitize_dataset_for_netcdf(ds)

    enc = {}
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.floating):
            enc[v] = {"zlib": True, "complevel": 4, "_FillValue": np.nan}
        else:
            enc[v] = {"zlib": True, "complevel": 4}

    def _write_once(path: Path) -> Path:
        # h5netcdf is often less sensitive to Windows file-manager locking.
        # If unavailable, xarray will raise and we retry with the default engine.
        try:
            ds.to_netcdf(path, mode="w", encoding=enc, engine="h5netcdf")
        except Exception:
            ds.to_netcdf(path, mode="w", encoding=enc)
        return path

    candidates = [out_path]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidates.append(out_path.with_name(f"{out_path.stem}_{timestamp}{out_path.suffix}"))

    # A second unique fallback in a subfolder reduces the chance of conflicts.
    fallback_dir = out_path.parent / "fallback_unlocked_outputs"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    candidates.append(fallback_dir / f"{out_path.stem}_{timestamp}{out_path.suffix}")

    errors = []
    for p in candidates:
        try:
            written = _write_once(p)
            if p != out_path:
                print(f"WARNING: target NetCDF was locked or unavailable.")
                print(f"         Wrote alternative file instead: {written}")
            return written
        except PermissionError as exc:
            errors.append(f"{p}: {exc}")
            continue
        except OSError as exc:
            # Some Windows/HDF5 lock errors arrive as OSError rather than PermissionError.
            errors.append(f"{p}: {exc}")
            continue

    raise PermissionError(
        "Failed to write NetCDF to the target and fallback paths. Close any open "
        "NetCDF viewers/notebooks using these files, or change CFG.OUT_DIR to a "
        "new empty folder.\nTried:\n" + "\n".join(errors)
    )

def read_shapefile_auto_encoding(shp_path: Path) -> gpd.GeoDataFrame:
    encodings = ["gb18030", "gbk", "cp936", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            gdf = gpd.read_file(shp_path, encoding=enc)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            else:
                gdf = gdf.to_crs("EPSG:4326")
            return gdf
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Could not read shapefile: {shp_path}; last error: {last_err}")


def load_china_boundary(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols = ["CN_NAME", "NAME", "SOVEREIGNT", "ADMIN", "CNTRY_NAME", "COUNTRY", "国家", "NAME_ZH"]
    for col in cols:
        if col in gdf.columns:
            s = gdf[col].astype(str)
            mask = s.str.contains("China|中国|CHN|中华人民共和国", case=False, na=False, regex=True)
            if mask.any():
                return gdf.loc[mask].to_crs("EPSG:4326").dissolve()
    raise ValueError("China boundary could not be identified from the global shapefile.")


def load_boundaries() -> Dict[str, Optional[gpd.GeoDataFrame]]:
    print("\nLoading boundaries...")
    out: Dict[str, Optional[gpd.GeoDataFrame]] = {"global": None, "china": None, "province": None, "ten_dash": None, "basin": None}

    gdf_global = read_shapefile_auto_encoding(CFG.SHP_GLOBAL)
    out["global"] = gdf_global
    out["china"] = load_china_boundary(gdf_global)

    if CFG.SHP_PROVINCE.exists():
        out["province"] = read_shapefile_auto_encoding(CFG.SHP_PROVINCE)
    else:
        print(f"WARNING: province boundary not found: {CFG.SHP_PROVINCE}")

    if CFG.SHP_TEN_DASH.exists():
        ten = read_shapefile_auto_encoding(CFG.SHP_TEN_DASH)
        out["ten_dash"] = ten.cx[72:137, 2:56]
    else:
        print(f"WARNING: ten-dash line not found: {CFG.SHP_TEN_DASH}")

    if CFG.BASIN_GPKG.exists():
        out["basin"] = gpd.read_file(CFG.BASIN_GPKG).to_crs("EPSG:4326")
    else:
        print(f"WARNING: basin boundary not found: {CFG.BASIN_GPKG}")

    return out


def normalize_enso_phase(x: object) -> str:
    """
    Normalize ENSO phase labels from the finalized ENSO-event file.

    Supported examples:
        El Nino, El Niño, nino, warm, 1
        La Nina, La Niña, nina, cold, -1
        Neutral, normal, 0, 中性
    """
    if pd.isna(x):
        return "Unknown"

    s = str(x).strip()
    sl = s.lower().replace("ñ", "n").replace("–", "-").replace("—", "-")

    if sl in ["1", "1.0", "+1", "+1.0"]:
        return "El Nino"
    if sl in ["-1", "-1.0"]:
        return "La Nina"
    if sl in ["0", "0.0"]:
        return "Neutral"

    if ("nino" in sl) or ("el n" in sl) or ("warm" in sl) or ("elnino" in sl):
        return "El Nino"
    if ("nina" in sl) or ("la n" in sl) or ("cold" in sl) or ("lanina" in sl):
        return "La Nina"
    if ("neutral" in sl) or ("normal" in sl) or ("none" in sl) or ("中性" in s):
        return "Neutral"

    return s


def phase_to_code(phase: str) -> int:
    if phase == "El Nino":
        return 1
    if phase == "La Nina":
        return -1
    if phase == "Neutral":
        return 0
    return 99


def _norm_colname(c: object) -> str:
    """
    Normalize column names for flexible matching.
    """
    s = str(c).strip().lower()
    for ch in [" ", "_", "-", ".", "/", "\\", "(", ")", "[", "]"]:
        s = s.replace(ch, "")
    return s


def _find_first_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    norm_map = {_norm_colname(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_colname(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _extract_year_month_from_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Extract year and month from either explicit year/month columns or a date/ym column.
    """
    year_col = _find_first_column(df, ("year", "yr", "年份", "年"))
    month_col = _find_first_column(df, ("month", "mon", "月份", "月"))

    if year_col is not None and month_col is not None:
        y = pd.to_numeric(df[year_col], errors="coerce")
        m = pd.to_numeric(df[month_col], errors="coerce")
        return pd.DataFrame({"year": y, "month": m}, index=df.index)

    date_col = _find_first_column(
        df,
        (
            "time", "date", "datetime", "ym", "yearmonth", "year_month",
            "年月", "日期", "月份时间", "monthdate"
        ),
    )
    if date_col is None:
        return None

    dt = pd.to_datetime(df[date_col], errors="coerce")
    if dt.isna().all():
        raw = df[date_col].astype(str).str.strip()
        raw2 = raw.str.replace("/", "-", regex=False).str.replace(".", "-", regex=False)
        dt = pd.to_datetime(raw2, errors="coerce")
        if dt.isna().all():
            raw_digits = raw.str.extract(r"(\d{4})\D*(\d{1,2})")
            if raw_digits.notna().any(axis=None):
                y = pd.to_numeric(raw_digits[0], errors="coerce")
                m = pd.to_numeric(raw_digits[1], errors="coerce")
                return pd.DataFrame({"year": y, "month": m}, index=df.index)

    return pd.DataFrame({"year": dt.dt.year, "month": dt.dt.month}, index=df.index)


def _get_user_specified_phase_column(df: pd.DataFrame) -> Optional[str]:
    """
    Return the user-specified ENSO phase column if it exists.
    Matching is normalized, so main-phase, main_phase and main phase are equivalent.
    """
    if not hasattr(CFG, "ENSO_PHASE_COLUMN") or CFG.ENSO_PHASE_COLUMN is None:
        return None
    wanted = _norm_colname(CFG.ENSO_PHASE_COLUMN)
    for c in df.columns:
        if _norm_colname(c) == wanted:
            return c
    return None


def _score_enso_sheet(df: pd.DataFrame, sheet_name: str) -> Tuple[int, Optional[str]]:
    """
    Score an Excel sheet by whether it appears to contain monthly ENSO events.
    """
    phase_candidates = (
        "main-phase", "mainphase", "main_phase", "main phase",
        "enso_phase", "ensophase", "phase", "phasefinal", "finalphase",
        "eventphase", "event_type", "eventtype", "enso", "ensoevent",
        "enso_event", "category", "classification", "class", "type",
        "位相", "事件类型", "事件", "类型", "分类"
    )
    phase_col = _get_user_specified_phase_column(df)
    if phase_col is None:
        phase_col = _find_first_column(df, phase_candidates)
    ym = _extract_year_month_from_df(df)

    score = 0
    if ym is not None:
        score += 10
    if phase_col is not None:
        score += 10
        vals = df[phase_col].map(normalize_enso_phase)
        score += int(vals.isin(["El Nino", "La Nina", "Neutral"]).sum() > 12) * 10

    name_low = str(sheet_name).lower()
    for key in ["monthly", "event", "phase", "enso", "classification", "result", "panel"]:
        if key in name_low:
            score += 1

    return score, phase_col


def load_enso_table() -> pd.DataFrame:
    """
    Load the finalized ENSO-event file from the previous workflow.

    The reader is flexible because the finalized event file may use different
    sheet or column names than the original ENSO_Calculation_Results.xlsx.
    """
    if not CFG.ENSO_EXCEL.exists():
        raise FileNotFoundError(
            f"ENSO event file not found: {CFG.ENSO_EXCEL}\n"
            "Please update CFG.ENSO_EXCEL to your finalized ENSO-event Excel file."
        )

    print(f"\nLoading finalized ENSO event file: {CFG.ENSO_EXCEL}")

    if CFG.ENSO_SHEET_NAME is not None:
        df = pd.read_excel(CFG.ENSO_EXCEL, sheet_name=CFG.ENSO_SHEET_NAME)
        chosen_sheet = str(CFG.ENSO_SHEET_NAME)
        phase_col = _score_enso_sheet(df, chosen_sheet)[1]
    else:
        xls = pd.ExcelFile(CFG.ENSO_EXCEL)
        best = None
        for sh in xls.sheet_names:
            try:
                tmp = pd.read_excel(CFG.ENSO_EXCEL, sheet_name=sh)
            except Exception:
                continue
            score, ph = _score_enso_sheet(tmp, sh)
            if best is None or score > best[0]:
                best = (score, sh, tmp, ph)

        if best is None or best[0] < 20:
            raise ValueError(
                "Could not automatically identify a monthly ENSO-event sheet. "
                "Please set CFG.ENSO_SHEET_NAME and ensure the sheet contains "
                "year/month and ENSO phase columns."
            )

        _, chosen_sheet, df, phase_col = best

    ym = _extract_year_month_from_df(df)
    if ym is None:
        raise ValueError(
            f"Chosen ENSO sheet '{chosen_sheet}' does not contain recognizable year/month information."
        )

    user_phase_col = _get_user_specified_phase_column(df)
    if user_phase_col is not None:
        phase_col = user_phase_col

    if phase_col is None:
        best_col = None
        best_count = -1
        for c in df.columns:
            vals = df[c].map(normalize_enso_phase)
            count = int(vals.isin(["El Nino", "La Nina", "Neutral"]).sum())
            if count > best_count:
                best_col = c
                best_count = count
        if best_count <= 0:
            raise ValueError(
                f"Chosen ENSO sheet '{chosen_sheet}' does not contain a recognizable ENSO phase column."
            )
        phase_col = best_col

    out = pd.DataFrame({
        "year": pd.to_numeric(ym["year"], errors="coerce"),
        "month": pd.to_numeric(ym["month"], errors="coerce"),
        "enso_phase": df[phase_col].map(normalize_enso_phase),
    })
    out = out.dropna(subset=["year", "month"]).copy()
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)

    out = out[
        (out["year"] >= CFG.START_YEAR) &
        (out["year"] <= CFG.END_YEAR) &
        (out["month"].between(1, 12)) &
        (out["enso_phase"].isin(["El Nino", "La Nina", "Neutral"]))
    ].copy()

    out = out.drop_duplicates(subset=["year", "month"], keep="first")
    out["ym"] = out["year"].astype(str) + "-" + out["month"].astype(str).str.zfill(2)
    out["phase_code"] = out["enso_phase"].map(phase_to_code).astype(np.int16)

    print(f"    Selected sheet : {chosen_sheet}")
    print(f"    Preferred col  : {getattr(CFG, 'ENSO_PHASE_COLUMN', None)}")
    print(f"    Phase column   : {phase_col}")
    print(f"    Valid records  : {len(out)}")
    print("    ENSO counts:")
    print(out["enso_phase"].value_counts().to_string())

    expected = (CFG.END_YEAR - CFG.START_YEAR + 1) * 12
    missing_months = expected - len(out)
    if missing_months > 0:
        print(f"WARNING: {missing_months} monthly records are missing after filtering.")

    return out

def standardize_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for c in ds.coords:
        low = c.lower()
        if low in ["longitude", "x"]:
            rename[c] = "lon"
        if low in ["latitude", "y"]:
            rename[c] = "lat"
    if rename:
        ds = ds.rename(rename)
    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("The precipitation dataset must include lon/lat coordinates.")
    return ds


def find_precip_var(ds: xr.Dataset) -> str:
    for v in ["pre", "precip", "precipitation", "pr", "rain"]:
        if v in ds.data_vars:
            return v
    for v in ds.data_vars:
        if "time" in ds[v].dims and len(ds[v].dims) >= 3:
            return v
    raise ValueError("No precipitation variable found.")


# =============================================================================
# BUILD GRID-CELL MONTHLY MEP95 ANOMALY
# =============================================================================

def read_daily_precip_for_china() -> xr.DataArray:
    ds = xr.open_dataset(CFG.PRECIP_NC_FILE)
    ds = standardize_lat_lon(ds)
    var = find_precip_var(ds)

    pr = ds[var].sel(time=slice(f"{CFG.START_YEAR}-01-01", f"{CFG.END_YEAR}-12-31"))
    pr = pr.where(np.isfinite(pr))
    pr = pr.where(pr >= 0)

    lon_min, lon_max, lat_min, lat_max = CFG.MAP_EXTENT
    pr = pr.sel(lon=slice(lon_min, lon_max))
    if pr.lat.values[0] < pr.lat.values[-1]:
        pr = pr.sel(lat=slice(lat_min, lat_max))
    else:
        pr = pr.sel(lat=slice(lat_max, lat_min))

    # Try polygon clipping.
    try:
        b = load_boundaries()
        import rioxarray  # noqa
        pr = pr.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
        pr = pr.rio.write_crs("EPSG:4326", inplace=False)
        pr = pr.rio.clip(b["china"].geometry, b["china"].crs, drop=True)
        print("Daily precipitation clipped to China boundary.")
    except Exception as exc:
        print(f"WARNING: China polygon clipping failed; rectangular extent is used. Reason: {exc}")

    return pr


def compute_wetday_p95_by_month(pr: xr.DataArray) -> xr.DataArray:
    arr = []
    for m in range(1, 13):
        print(f"  Wet-day P95 for calendar month {m:02d}")
        sub = pr.where(pr["time"].dt.month == m, drop=True)
        wet = sub.where(sub >= CFG.WET_DAY_MM)
        q = wet.quantile(CFG.MEP_DAILY_PERCENTILE / 100.0, dim="time", skipna=True)
        if "quantile" in q.coords:
            q = q.drop_vars("quantile")
        q = q.assign_coords(month=m).expand_dims(month=[m])
        arr.append(q)
    p95 = xr.concat(arr, dim="month")
    p95.name = "wetday_p95"
    return p95


def compute_monthly_mep95(pr: xr.DataArray, p95_by_month: xr.DataArray) -> xr.DataArray:
    month_indexer = xr.DataArray(pr["time"].dt.month.values, coords={"time": pr["time"]}, dims=("time",))
    daily_threshold = p95_by_month.sel(month=month_indexer)
    extreme_daily = pr.where((pr >= CFG.WET_DAY_MM) & (pr > daily_threshold), 0.0)
    mep95 = extreme_daily.resample(time="MS").sum(dim="time", skipna=True)
    mep95.name = "MEP95"
    mep95.attrs["units"] = "mm month-1"
    return mep95


def compute_monthly_rx5day(pr: xr.DataArray) -> xr.DataArray:
    """
    Monthly Rx5day: maximum 5-day accumulated precipitation in each month.
    """
    print("  Computing monthly Rx5day")
    rolling5 = pr.rolling(time=5, min_periods=5).sum()
    rx5day = rolling5.resample(time="MS").max(dim="time", skipna=True)
    rx5day.name = "Rx5day"
    rx5day.attrs["units"] = "mm"
    return rx5day


def compute_calendar_month_anomaly(da: xr.DataArray, out_name: str) -> xr.DataArray:
    """
    Convert a monthly metric into calendar-month anomalies.
    """
    base = da.where(
        (da["year"] >= CFG.ANOMALY_BASELINE_START_YEAR) &
        (da["year"] <= CFG.ANOMALY_BASELINE_END_YEAR),
        drop=True,
    )
    clim = base.groupby("time.month").mean(dim="time", skipna=True)
    anomaly = da.groupby("time.month") - clim
    anomaly.name = out_name
    anomaly.attrs["units"] = da.attrs.get("units", "mm")
    return anomaly


def add_enso_coords(da: xr.DataArray, enso_df: pd.DataFrame) -> xr.DataArray:
    t = pd.DatetimeIndex(da.time.values)
    ym = t.strftime("%Y-%m")
    phase_map = enso_df.set_index("ym")["phase_code"].to_dict()
    phase_code = np.array([phase_map.get(x, 99) for x in ym], dtype=np.int16)
    valid = phase_code != 99

    da = da.isel(time=valid)
    t = pd.DatetimeIndex(da.time.values)
    phase_code = phase_code[valid]

    da = da.assign_coords(
        year=("time", t.year.astype(np.int16)),
        # Use cal_month instead of month because wetday_p95 has a separate
        # dimension named month. This avoids an xarray coordinate-name conflict.
        cal_month=("time", t.month.astype(np.int16)),
        ym=("time", t.strftime("%Y-%m")),
        phase_code=("time", phase_code),
    )
    return da


def build_grid_metric_nc() -> Path:
    """
    Build one grid-cell monthly metric NetCDF containing both MEP95 and Rx5day.

    This file is the Stage-0 metric file. It is not the final RR map file.
    The final RR/significance NetCDFs are created later for each metric and threshold.
    """
    nc_dir, _, _ = output_dirs()

    enso_df = load_enso_table()
    pr = read_daily_precip_for_china()

    # --- MEP95 ---
    p95 = compute_wetday_p95_by_month(pr)
    mep95 = compute_monthly_mep95(pr, p95)
    mep95 = add_enso_coords(mep95, enso_df)
    mep95_anom = compute_calendar_month_anomaly(mep95, "MEP95_anomaly")

    # --- Rx5day ---
    rx5day = compute_monthly_rx5day(pr)
    rx5day = add_enso_coords(rx5day, enso_df)
    rx5day_anom = compute_calendar_month_anomaly(rx5day, "Rx5day_anomaly")

    # Align both metrics on the same time axis in case any month is missing.
    mep95, mep95_anom, rx5day, rx5day_anom = xr.align(
        mep95, mep95_anom, rx5day, rx5day_anom,
        join="inner",
    )

    ds_out = xr.Dataset({
        "MEP95": mep95.astype("float32"),
        "MEP95_anomaly": mep95_anom.astype("float32"),
        "Rx5day": rx5day.astype("float32"),
        "Rx5day_anomaly": rx5day_anom.astype("float32"),
        "wetday_p95": p95.astype("float32"),
    })

    for coord in ["year", "cal_month", "ym", "phase_code"]:
        if coord in mep95.coords:
            ds_out = ds_out.assign_coords({coord: mep95.coords[coord]})

    ds_out.attrs.update({
        "description": "Grid-cell monthly MEP95/Rx5day metrics and anomalies for Figure 4",
        "baseline": f"{CFG.ANOMALY_BASELINE_START_YEAR}-{CFG.ANOMALY_BASELINE_END_YEAR}",
        "period": f"{CFG.START_YEAR}-{CFG.END_YEAR}",
        "wet_day_mm": CFG.WET_DAY_MM,
        "mep_daily_percentile": CFG.MEP_DAILY_PERCENTILE,
        "rx5day_definition": "monthly maximum rolling 5-day precipitation total",
    })

    out = nc_dir / "Figure4_grid_monthly_MEP95_RX5day_anomaly_CN.nc"
    return safe_write_netcdf(ds_out, out)


# =============================================================================
# RR AND SIGNIFICANCE
# =============================================================================

def wilson_score_interval_counts(
    count: np.ndarray,
    n: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Wilson score confidence interval for binomial event probability.

    This is the spatial analogue of the CDF confidence interval used in the
    earlier FAR/RR figure: event probability is the exceedance probability at
    a selected threshold.
    """
    count = count.astype(float)
    n = n.astype(float)

    lo = np.full(count.shape, np.nan, dtype=float)
    hi = np.full(count.shape, np.nan, dtype=float)

    valid = n > 0
    if not np.any(valid):
        return lo, hi

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p = np.divide(count, n, out=np.full_like(count, np.nan), where=n > 0)

    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denom
    half_width = z * np.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n) / denom

    lo[valid] = centre[valid] - half_width[valid]
    hi[valid] = centre[valid] + half_width[valid]
    return lo, hi


def wilson_cdf_significance(
    c_phase: np.ndarray,
    n_phase: np.ndarray,
    c_neutral: np.ndarray,
    n_neutral: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wilson-CDF significance test used for Fig. 4.

    For a selected threshold, compute:
        P_phase   = P(event | ENSO phase)
        P_neutral = P(event | Neutral)

    Significance is defined when P_neutral lies outside the 95% Wilson
    confidence interval around P_phase. This matches the previous CDF/FAR/RR
    logic in which Neutral CDF outside the ENSO-event CDF confidence interval
    indicates a significant ENSO-related probability shift.
    """
    c_phase = c_phase.astype(float)
    n_phase = n_phase.astype(float)
    c_neutral = c_neutral.astype(float)
    n_neutral = n_neutral.astype(float)

    p_phase_raw = np.divide(
        c_phase, n_phase,
        out=np.full_like(c_phase, np.nan),
        where=n_phase > 0,
    )
    p_neutral_raw = np.divide(
        c_neutral, n_neutral,
        out=np.full_like(c_neutral, np.nan),
        where=n_neutral > 0,
    )

    ci_low, ci_high = wilson_score_interval_counts(
        c_phase,
        n_phase,
        alpha=CFG.ALPHA,
    )

    valid = (
        (n_phase >= CFG.MIN_PHASE_N) &
        (n_neutral >= CFG.MIN_NEUTRAL_N) &
        np.isfinite(p_phase_raw) &
        np.isfinite(p_neutral_raw) &
        np.isfinite(ci_low) &
        np.isfinite(ci_high)
    )

    sig = ((p_neutral_raw < ci_low) | (p_neutral_raw > ci_high)) & valid

    return sig, p_phase_raw, p_neutral_raw, ci_low, ci_high


def fdr_bh_mask(pvals: np.ndarray, alpha: float) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    valid = np.isfinite(p)
    sig = np.zeros(p.shape, dtype=bool)
    if valid.sum() == 0:
        return sig

    pv = p[valid]
    order = np.argsort(pv)
    pv_sorted = pv[order]
    m = pv_sorted.size
    limits = alpha * np.arange(1, m + 1) / m
    passed = pv_sorted <= limits
    if not np.any(passed):
        return sig
    cutoff = pv_sorted[np.where(passed)[0].max()]
    sig[valid] = p[valid] <= cutoff
    return sig


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(2, CFG.CLUSTER_CONNECTIVITY)
    labels, nlab = ndimage.label(mask, structure=structure)
    if nlab == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == counts.argmax()


def two_proportion_pvalue(c1: np.ndarray, n1: np.ndarray, c0: np.ndarray, n0: np.ndarray) -> np.ndarray:
    c1 = c1.astype(float)
    n1 = n1.astype(float)
    c0 = c0.astype(float)
    n0 = n0.astype(float)

    valid = (n1 >= CFG.MIN_PHASE_N) & (n0 >= CFG.MIN_NEUTRAL_N)
    pval = np.full(c1.shape, np.nan, dtype=float)
    if not valid.any():
        return pval

    p1 = np.divide(c1, n1, out=np.full_like(c1, np.nan), where=n1 > 0)
    p0 = np.divide(c0, n0, out=np.full_like(c0, np.nan), where=n0 > 0)
    pp = np.divide(c1 + c0, n1 + n0, out=np.full_like(c1, np.nan), where=(n1 + n0) > 0)
    se = np.sqrt(pp * (1.0 - pp) * (1.0 / n1 + 1.0 / n0))

    z = np.divide(p1 - p0, se, out=np.zeros_like(c1), where=(se > 0) & valid)
    pval[(se > 0) & valid] = 2.0 * stats.norm.sf(np.abs(z[(se > 0) & valid]))
    pval[(se <= 0) & valid] = 1.0
    return pval


def calculate_rr(c1: np.ndarray, n1: np.ndarray, c0: np.ndarray, n0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c1 = c1.astype(float)
    n1 = n1.astype(float)
    c0 = c0.astype(float)
    n0 = n0.astype(float)
    valid = (n1 >= CFG.MIN_PHASE_N) & (n0 >= CFG.MIN_NEUTRAL_N)

    if CFG.USE_HALDANE_CORRECTION:
        p1 = np.divide(c1 + 0.5, n1 + 1.0, out=np.full_like(c1, np.nan), where=n1 > 0)
        p0 = np.divide(c0 + 0.5, n0 + 1.0, out=np.full_like(c0, np.nan), where=n0 > 0)
    else:
        p1 = np.divide(c1, n1, out=np.full_like(c1, np.nan), where=n1 > 0)
        p0 = np.divide(c0, n0, out=np.full_like(c0, np.nan), where=n0 > 0)

    rr = np.divide(p1, p0, out=np.full_like(p1, np.nan), where=(p0 > 0) & valid)
    log2rr = np.log2(rr)
    log2rr[~valid] = np.nan
    return rr, log2rr, valid


def ensure_metric_time_phase_coords(ds_metric: xr.Dataset) -> xr.Dataset:
    """
    Ensure the grid metric dataset has time-based cal_month and phase_code
    coordinates. Works for both MEP95_anomaly and Rx5day_anomaly.
    """
    anomaly_candidates = [v for v in ["MEP95_anomaly", "Rx5day_anomaly"] if v in ds_metric]
    if not anomaly_candidates:
        raise KeyError(
            "The grid metric NetCDF must contain at least 'MEP95_anomaly' or 'Rx5day_anomaly'. "
            "The basin-level Fig. 2 NetCDF cannot be used for Fig. 4 spatial maps."
        )

    anom0 = ds_metric[anomaly_candidates[0]]
    if "time" not in anom0.dims or "time" not in anom0.coords:
        raise KeyError("The anomaly variable must contain a time dimension/coordinate.")

    time_index = pd.DatetimeIndex(pd.to_datetime(anom0["time"].values))
    cal_month = time_index.month.astype(np.int16)
    cal_year = time_index.year.astype(np.int16)
    ym = time_index.strftime("%Y-%m")

    phase_code = None
    if not CFG.OVERRIDE_PHASE_CODE_FROM_ENSO_FILE:
        if "phase_code" in anom0.coords and "time" in anom0["phase_code"].dims:
            phase_code = anom0["phase_code"].values.astype(np.int16)
        elif "phase_code" in ds_metric.coords and "time" in ds_metric["phase_code"].dims:
            phase_code = ds_metric["phase_code"].values.astype(np.int16)
        elif "phase_code" in ds_metric.data_vars and "time" in ds_metric["phase_code"].dims:
            phase_code = ds_metric["phase_code"].values.astype(np.int16)

    if phase_code is None:
        enso_df = load_enso_table()
        phase_map = enso_df.set_index("ym")["phase_code"].to_dict()
        phase_code = np.array([phase_map.get(x, 99) for x in ym], dtype=np.int16)

    valid = phase_code != 99
    if valid.sum() == 0:
        raise ValueError(
            "No valid ENSO phase_code values were matched to the grid metric NetCDF time axis. "
            "Please check CFG.ENSO_EXCEL, CFG.ENSO_SHEET_NAME and CFG.ENSO_PHASE_COLUMN."
        )

    if valid.sum() < len(valid):
        print(f"WARNING: {len(valid) - int(valid.sum())} metric months have no ENSO phase and will be dropped.")
        ds_metric = ds_metric.isel(time=valid)
        time_index = time_index[valid]
        cal_month = cal_month[valid]
        cal_year = cal_year[valid]
        ym = ym[valid]
        phase_code = phase_code[valid]

    drop_names = [v for v in ["cal_month", "cal_year", "ym", "phase_code"] if v in ds_metric.data_vars]
    if drop_names:
        ds_metric = ds_metric.drop_vars(drop_names)

    ds_metric = ds_metric.assign_coords(
        cal_month=("time", cal_month.astype(np.int16)),
        cal_year=("time", cal_year.astype(np.int16)),
        ym=("time", np.asarray(ym, dtype=object)),
        phase_code=("time", phase_code.astype(np.int16)),
    )

    return ds_metric


def calculate_spatial_rr(ds_metric: xr.Dataset, q: int, metric: str = "MEP95") -> xr.Dataset:
    """
    Calculate spatial RR/Wilson significance for one metric and one threshold.
    metric can be 'MEP95' or 'Rx5day'.
    """
    metric = str(metric)
    anom_name = f"{metric}_anomaly"
    if anom_name not in ds_metric:
        raise KeyError(
            f"Required variable {anom_name!r} is missing from the grid metric NetCDF. "
            "Set REBUILD_MONTHLY_METRIC_NC=True once to rebuild the metric file with both MEP95 and Rx5day."
        )

    anom = ds_metric[anom_name]
    lat = anom["lat"].values
    lon = anom["lon"].values
    vars_out = {}

    month_coord = anom["cal_month"]
    phase_coord = anom["phase_code"]

    for month in getattr(CFG, "WARM_MONTHS", tuple(range(1, 13))):
        month_mask = month_coord == month
        sub = anom.where(month_mask, drop=True)
        if sub.sizes.get("time", 0) == 0:
            continue

        vals = sub.values.astype(float)
        phase_codes = phase_coord.where(month_mask, drop=True).values.astype(int)
        threshold = np.nanpercentile(vals, q, axis=0)
        event = (vals >= threshold[None, :, :])
        finite = np.isfinite(vals)

        neutral_t = phase_codes == 0
        n_neutral = np.sum(finite & neutral_t[:, None, None], axis=0)
        c_neutral = np.sum(event & finite & neutral_t[:, None, None], axis=0)

        for phase in CFG.PHASES:
            phase_code = phase_to_code(phase)
            phase_t = phase_codes == phase_code

            n_phase = np.sum(finite & phase_t[:, None, None], axis=0)
            c_phase = np.sum(event & finite & phase_t[:, None, None], axis=0)

            rr, log2rr, valid = calculate_rr(c_phase, n_phase, c_neutral, n_neutral)

            wilson_sig, p_phase_raw, p_neutral_raw, ci_low, ci_high = wilson_cdf_significance(
                c_phase,
                n_phase,
                c_neutral,
                n_neutral,
            )

            pval = two_proportion_pvalue(c_phase, n_phase, c_neutral, n_neutral)
            if CFG.APPLY_FDR_AFTER_WILSON or CFG.USE_FDR:
                fdr_sig = fdr_bh_mask(pval, CFG.ALPHA)
                sig = wilson_sig & fdr_sig
            else:
                fdr_sig = np.zeros_like(wilson_sig, dtype=bool)
                sig = wilson_sig

            sig &= valid & np.isfinite(log2rr)

            display_sig = largest_connected_component(sig) if CFG.KEEP_LARGEST_SIGNIFICANT_CLUSTER else sig.copy()

            phase_key = "EN" if phase == "El Nino" else "LN"
            base = f"q{q}_{MONTH_LABELS[month]}_{phase_key}"

            never_neutral = (c_neutral == 0) & (n_neutral >= CFG.MIN_NEUTRAL_N)
            never_phase = (c_phase == 0) & (n_phase >= CFG.MIN_PHASE_N)
            never_either = never_neutral & never_phase

            vars_out[f"{base}_RR"] = (("lat", "lon"), rr.astype("float32"))
            vars_out[f"{base}_log2RR"] = (("lat", "lon"), log2rr.astype("float32"))
            vars_out[f"{base}_P_phase"] = (("lat", "lon"), p_phase_raw.astype("float32"))
            vars_out[f"{base}_P_neutral"] = (("lat", "lon"), p_neutral_raw.astype("float32"))
            vars_out[f"{base}_wilson_ci_low"] = (("lat", "lon"), ci_low.astype("float32"))
            vars_out[f"{base}_wilson_ci_high"] = (("lat", "lon"), ci_high.astype("float32"))
            vars_out[f"{base}_pvalue_twoprop"] = (("lat", "lon"), pval.astype("float32"))
            vars_out[f"{base}_sig_wilson"] = (("lat", "lon"), wilson_sig.astype(np.int8))
            vars_out[f"{base}_sig_fdr"] = (("lat", "lon"), fdr_sig.astype(np.int8))
            vars_out[f"{base}_sig_final"] = (("lat", "lon"), sig.astype(np.int8))
            vars_out[f"{base}_sig_display"] = (("lat", "lon"), display_sig.astype(np.int8))
            vars_out[f"{base}_never_neutral"] = (("lat", "lon"), never_neutral.astype(np.int8))
            vars_out[f"{base}_never_event"] = (("lat", "lon"), never_phase.astype(np.int8))
            vars_out[f"{base}_never_either"] = (("lat", "lon"), never_either.astype(np.int8))
            vars_out[f"{base}_threshold"] = (("lat", "lon"), threshold.astype("float32"))
            vars_out[f"{base}_n_phase"] = (("lat", "lon"), n_phase.astype(np.int16))
            vars_out[f"{base}_n_neutral"] = (("lat", "lon"), n_neutral.astype(np.int16))
            vars_out[f"{base}_c_phase"] = (("lat", "lon"), c_phase.astype(np.int16))
            vars_out[f"{base}_c_neutral"] = (("lat", "lon"), c_neutral.astype(np.int16))

            print(
                f"{metric} q{q} {MONTH_LABELS[month]} {phase:8s}: "
                f"Wilson sig={int(wilson_sig.sum())}, "
                f"final={int(sig.sum())}, display={int(display_sig.sum())}"
            )

    return xr.Dataset(
        vars_out,
        coords={"lat": lat, "lon": lon},
        attrs={
            "description": f"Spatial RR and Wilson-CDF significance for {metric} q{q} anomaly events",
            "metric": metric,
            "threshold_percentile": int(q),
            "event_definition": f"{metric}_anomaly >= grid-cell calendar-month q{q}",
            "RR": "P(event | ENSO phase) / P(event | Neutral)",
            "plot_variable": "RR",
            "significance": "Wilson-score CDF test: Neutral event probability outside ENSO-phase 95% confidence interval",
            "alpha": float(CFG.ALPHA),
            "apply_fdr_after_wilson": int(CFG.APPLY_FDR_AFTER_WILSON),
            "confidence_level": float(CFG.CONFIDENCE_LEVEL),
            "period": f"{CFG.START_YEAR}-{CFG.END_YEAR}",
            "baseline": f"{CFG.ANOMALY_BASELINE_START_YEAR}-{CFG.ANOMALY_BASELINE_END_YEAR}",
            "keep_largest_significant_cluster": int(CFG.KEEP_LARGEST_SIGNIFICANT_CLUSTER),
        },
    )


def rr_nc_path_for(figure_id: str, metric: str, q: int, nc_dir: Path) -> Path:
    """Canonical NetCDF output path for calculated spatial RR/significance results."""
    return nc_dir / f"{figure_id}_{metric}_q{q}_spatial_RR_Wilson_significance_ChinaOnly_12months.nc"


def open_rr_nc_for_plot(rr_nc: Path) -> xr.Dataset:
    """
    Open a calculated RR/significance NetCDF for plotting, load it into memory,
    and close the file handle immediately. This avoids Windows file-locking
    problems when the same file is later deleted or overwritten.
    """
    with xr.open_dataset(rr_nc) as ds:
        ds_loaded = ds.load()
    return ds_loaded


def write_summary_table(ds_rr: xr.Dataset, metric: str, q: int, table_dir: Path) -> Path:
    rows = []
    for month in getattr(CFG, "WARM_MONTHS", tuple(range(1, 13))):
        for phase in CFG.PHASES:
            phase_key = "EN" if phase == "El Nino" else "LN"
            base = f"q{q}_{MONTH_LABELS[month]}_{phase_key}"
            sig = ds_rr[f"{base}_sig_final"].values.astype(bool)
            wilson_sig = ds_rr[f"{base}_sig_wilson"].values.astype(bool)
            disp = ds_rr[f"{base}_sig_display"].values.astype(bool)
            z = ds_rr[f"{base}_log2RR"].values
            rows.append({
                "metric": metric,
                "threshold": q,
                "month": month,
                "month_label": MONTH_LABELS[month],
                "phase": phase,
                "n_wilson_significant_cells": int(wilson_sig.sum()),
                "n_final_significant_cells": int(sig.sum()),
                "n_displayed_cells": int(disp.sum()),
                "mean_log2RR_displayed": float(np.nanmean(np.where(disp, z, np.nan))) if disp.any() else np.nan,
                "max_log2RR_displayed": float(np.nanmax(np.where(disp, z, np.nan))) if disp.any() else np.nan,
                "min_log2RR_displayed": float(np.nanmin(np.where(disp, z, np.nan))) if disp.any() else np.nan,
            })
    out = table_dir / f"Figure4_{metric}_q{q}_spatial_rr_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


# =============================================================================
# PLOTTING
# =============================================================================

def cmap_norm() -> Tuple[mcolors.Colormap, mcolors.BoundaryNorm, np.ndarray]:
    """
    Article-style Risk Ratio color scale.

    Low RR (<1) is warm-colored, RR around 1 is near-neutral, and high RR (>1)
    is blue. The bounds follow the reference figure.
    """
    bounds = np.array(CFG.RR_COLOR_BOUNDS, dtype=float)
    colors = [
        "#e43d25",  # 0.2-0.3
        "#ef7469",  # 0.3-0.4
        "#f8b96e",  # 0.4-0.5
        "#fff2a8",  # 0.5-1.0
        "#d8f1f3",  # 1.0-1.5
        "#6fa0c9",  # 1.5-2.0
        "#3e74b7",  # 2.0-3.0
        "#092e9f",  # 3.0-4.0
    ]
    cmap = mcolors.ListedColormap(colors, name="RR_article_style")
    norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=False)
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    cmap.set_bad((1, 1, 1, 0))
    return cmap, norm, bounds


def _union_geometry(gdf: gpd.GeoDataFrame):
    """
    Return a single unioned geometry from a GeoDataFrame, compatible with
    different geopandas/shapely versions.
    """
    try:
        return gdf.geometry.union_all()
    except Exception:
        return gdf.geometry.unary_union


def make_region_mask(lon: np.ndarray, lat: np.ndarray, region_gdf: Optional[gpd.GeoDataFrame]) -> np.ndarray:
    """
    Build a lat × lon boolean mask for points inside the China boundary.

    The mask is used for display so only grid cells within China are retained.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    if region_gdf is None or len(region_gdf) == 0:
        return np.ones((len(lat), len(lon)), dtype=bool)

    geom = _union_geometry(region_gdf.to_crs("EPSG:4326"))

    # Prefer shapely vectorized contains when available.
    try:
        import shapely
        xx, yy = np.meshgrid(lon, lat)
        if hasattr(shapely, "contains_xy"):
            inside = shapely.contains_xy(geom, xx, yy)
        else:
            from shapely import vectorized
            inside = vectorized.contains(geom, xx, yy)
        return np.asarray(inside, dtype=bool)
    except Exception:
        from shapely.geometry import Point
        from shapely.prepared import prep
        pg = prep(geom)
        mask = np.zeros((len(lat), len(lon)), dtype=bool)
        for i, y in enumerate(lat):
            for j, x in enumerate(lon):
                p = Point(float(x), float(y))
                mask[i, j] = pg.contains(p) or pg.covers(p)
        return mask


def overlay_boolean_mask(ax, lon: np.ndarray, lat: np.ndarray, mask: np.ndarray, color: str, zorder: int) -> None:
    """
    Overlay a boolean mask as a single-color pcolormesh layer.
    """
    if mask is None or not np.any(mask):
        return
    cmap_mask = mcolors.ListedColormap([color])
    ax.pcolormesh(
        lon,
        lat,
        np.where(mask, 1.0, np.nan),
        cmap=cmap_mask,
        shading="nearest",
        transform=ccrs.PlateCarree(),
        zorder=zorder,
    )


def draw_boundaries(ax, b: Dict[str, Optional[gpd.GeoDataFrame]]) -> None:
    """
    Draw only China-related overlays: China boundary, dashed province lines,
    selected six-basin/river shapefile, and ten-dash line.
    Other countries' borders and ocean shading are not drawn.
    """
    trans = ccrs.PlateCarree()

    if b.get("china") is not None and CFG.FILL_CHINA_WHITE:
        b["china"].plot(ax=ax, facecolor="white", edgecolor="none", transform=trans, zorder=3)

    if b.get("province") is not None:
        b["province"].boundary.plot(
            ax=ax, edgecolor="0.62", linewidth=0.42,
            linestyle="--" if CFG.DRAW_PROVINCE_DASHED else "-",
            transform=trans, zorder=22,
        )

    if b.get("basin") is not None:
        basin = b["basin"].to_crs("EPSG:4326")
        geom_types = set(basin.geometry.geom_type.astype(str))
        is_polygon = any(("Polygon" in g) for g in geom_types)
        if is_polygon:
            basin.boundary.plot(ax=ax, edgecolor="0.18", linewidth=0.95, linestyle="-", transform=trans, zorder=24)
        else:
            basin.plot(ax=ax, facecolor="none", edgecolor="0.18", linewidth=0.65, linestyle="-", transform=trans, zorder=24)

    if b.get("china") is not None:
        b["china"].boundary.plot(ax=ax, edgecolor="0.20", linewidth=0.95, transform=trans, zorder=26)

    if b.get("ten_dash") is not None:
        b["ten_dash"].plot(ax=ax, facecolor="none", edgecolor="0.20", linewidth=0.65, transform=trans, zorder=27)


def draw_six_basin_labels(ax) -> None:
    """Label the six major river basins using approximate label positions."""
    if not CFG.DRAW_SIX_BASIN_LABELS or not CFG.SIX_BASIN_LABELS:
        return
    for name, (x, y) in CFG.SIX_BASIN_LABELS.items():
        ax.text(
            x, y, name,
            transform=ccrs.PlateCarree(),
            ha="center", va="center",
            fontsize=6.2, fontweight="bold", color="0.15", zorder=31,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.45, pad=0.5),
        )

def draw_outline(ax, lon: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> None:
    if not CFG.DRAW_SIGNIFICANT_FOOTPRINT_OUTLINE or not np.any(mask):
        return
    try:
        ax.contour(
            lon, lat, mask.astype(float), levels=[0.5],
            colors=CFG.SIGNIFICANT_OUTLINE_COLOR,
            linewidths=CFG.SIGNIFICANT_OUTLINE_WIDTH,
            transform=ccrs.PlateCarree(),
            zorder=18,
        )
    except Exception:
        pass


def reference_paper_panel_items() -> list:
    """
    Panel order following the uploaded reference figure:
    Dec LN | Dec EN | Jan LN | Jan EN; ... ; Jul LN | Jul EN | Jun LN | Jun EN.
    """
    month_pairs = [(12, 1), (11, 2), (10, 3), (9, 4), (8, 5), (7, 6)]
    items = []
    for left_m, right_m in month_pairs:
        items.extend([
            (left_m, "La Nina"), (left_m, "El Nino"),
            (right_m, "La Nina"), (right_m, "El Nino"),
        ])
    return items


def plot_spatial_footprints(ds_rr: xr.Dataset, metric: str, q: int, figure_id: str, boundaries: Dict[str, Optional[gpd.GeoDataFrame]]) -> Dict[str, str]:
    """
    Plot 12 calendar months × two ENSO phases using the reference-paper layout.
    Only Wilson-significant cells inside China are shown, with never categories.
    """
    set_figure_style()
    _, fig_dir, _ = output_dirs()

    lon = ds_rr["lon"].values
    lat = ds_rr["lat"].values
    cmap, norm, bounds = cmap_norm()
    china_mask = make_region_mask(lon, lat, boundaries.get("china"))

    if CFG.PLOT_LAYOUT == "paper_4cols":
        panel_items = reference_paper_panel_items()
        nrows, ncols = 6, 4
    elif CFG.PLOT_LAYOUT == "three_columns":
        panel_items = [(m, p) for m in getattr(CFG, "WARM_MONTHS", tuple(range(1, 13))) for p in CFG.PHASES]
        ncols = 3
        nrows = int(np.ceil(len(panel_items) / ncols))
    else:
        panel_items = [(m, p) for m in getattr(CFG, "WARM_MONTHS", tuple(range(1, 13))) for p in CFG.PHASES]
        nrows = len(getattr(CFG, "WARM_MONTHS", tuple(range(1, 13))))
        ncols = 2

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=CFG.FIG_SIZE, dpi=CFG.DPI,
        subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=False,
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    for panel_idx, (month, phase) in enumerate(panel_items):
        if CFG.PLOT_LAYOUT == "paper_4cols":
            r = panel_idx // 4
            c = panel_idx % 4
        elif CFG.PLOT_LAYOUT == "three_columns":
            r = panel_idx // ncols
            c = panel_idx % ncols
        else:
            r = list(getattr(CFG, "WARM_MONTHS", tuple(range(1, 13)))).index(month)
            c = 0 if phase == "El Nino" else 1

        ax = axes[r, c]
        phase_key = "EN" if phase == "El Nino" else "LN"
        base = f"q{q}_{MONTH_LABELS[month]}_{phase_key}"

        rr = ds_rr[f"{base}_RR"].values
        sig_mask = ds_rr[f"{base}_sig_display"].values.astype(bool) & china_mask
        never_neutral = ds_rr[f"{base}_never_neutral"].values.astype(bool) & sig_mask
        never_event = ds_rr[f"{base}_never_event"].values.astype(bool) & sig_mask
        never_either = ds_rr[f"{base}_never_either"].values.astype(bool) & sig_mask
        never_any = never_neutral | never_event | never_either

        plot_data = np.where(sig_mask & np.isfinite(rr) & (~never_any), rr, np.nan)

        ax.set_extent(CFG.MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.set_facecolor("white")
        draw_boundaries(ax, boundaries)

        ax.pcolormesh(
            lon, lat, plot_data, cmap=cmap, norm=norm, shading="nearest",
            transform=ccrs.PlateCarree(), zorder=10,
        )

        if CFG.SHOW_NEVER_CATEGORIES:
            overlay_boolean_mask(ax, lon, lat, never_either, CFG.NEVER_EITHER_COLOR, zorder=12)
            overlay_boolean_mask(ax, lon, lat, never_event & (~never_either), CFG.NEVER_EVENT_COLOR, zorder=13)
            overlay_boolean_mask(ax, lon, lat, never_neutral & (~never_either), CFG.NEVER_NEUTRAL_COLOR, zorder=14)

        draw_outline(ax, lon, lat, sig_mask)
        draw_boundaries(ax, boundaries)
        draw_six_basin_labels(ax)

        ax.set_xticks([])
        ax.set_yticks([])

        title_color = "blue" if phase == "La Nina" else "crimson"
        ax.set_title(f"{MONTH_LABELS[month]} {PHASE_LABELS.get(phase, phase)}", fontsize=15.0, fontweight="bold", color=title_color, pad=3.5)
        ax.text(
            0.018, 0.972, f"({chr(ord('a') + panel_idx)})",
            transform=ax.transAxes, ha="left", va="top", fontsize=10.8, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.1), zorder=32,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
            spine.set_edgecolor("0.45")

    if CFG.PLOT_LAYOUT == "paper_4cols":
        fig.subplots_adjust(left=0.035, right=0.985, bottom=0.105, top=0.965, hspace=0.170, wspace=0.035)
        cax = fig.add_axes([0.075, 0.055, 0.43, 0.022])
        never_anchor = (0.985, 0.056)
    elif CFG.PLOT_LAYOUT == "three_columns":
        fig.subplots_adjust(left=0.045, right=0.985, bottom=0.120, top=0.950, hspace=0.135, wspace=0.035)
        cax = fig.add_axes([0.22, 0.055, 0.50, 0.018])
        never_anchor = (0.985, 0.035)
    else:
        fig.subplots_adjust(left=0.085, right=0.985, bottom=0.075, top=0.970, hspace=0.055, wspace=0.030)
        cax = fig.add_axes([0.18, 0.035, 0.48, 0.014])
        never_anchor = (0.985, 0.030)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", boundaries=bounds, ticks=bounds, spacing="proportional", drawedges=True, extend="both")
    cbar.ax.tick_params(labelsize=10.2, length=2.8, width=0.75)
    for t in cbar.ax.get_xticklabels():
        t.set_fontweight("bold")
    cbar.outline.set_linewidth(0.75)
    try:
        cbar.dividers.set_linewidth(0.50)
    except Exception:
        pass

    fig.text(cax.get_position().x0 + cax.get_position().width / 2, cax.get_position().y0 - 0.018, "Risk ratio", ha="center", va="top", fontsize=10.8, fontweight="bold")

    if CFG.SHOW_NEVER_CATEGORIES:
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(facecolor=CFG.NEVER_NEUTRAL_COLOR, edgecolor="0.35", linewidth=0.3, label="Never in Neutral"),
            mpatches.Patch(facecolor=CFG.NEVER_EVENT_COLOR, edgecolor="0.35", linewidth=0.3, label="Never in event"),
            mpatches.Patch(facecolor=CFG.NEVER_EITHER_COLOR, edgecolor="0.35", linewidth=0.3, label="Never in either"),
        ]
        fig.legend(handles=handles, loc="lower right", bbox_to_anchor=never_anchor, ncol=3, frameon=False, fontsize=9.8, handlelength=3.8, handleheight=1.1, columnspacing=1.1)

    cluster_tag = "largest_cluster" if CFG.KEEP_LARGEST_SIGNIFICANT_CLUSTER else "all_wilson_cells"
    basename = f"{figure_id}_{metric}_anomaly_q{q}_Jan_Dec_spatial_RR_fromSavedNC_referenceLayout_ChinaOnly_Wilson_never_{cluster_tag}"
    png = fig_dir / f"{basename}.png"
    tif = fig_dir / f"{basename}.tif"
    pdf = fig_dir / f"{basename}.pdf"

    fig.savefig(png, dpi=CFG.DPI, bbox_inches="tight", facecolor="white")
    if CFG.EXPORT_TIFF:
        try:
            fig.savefig(tif, dpi=CFG.DPI, bbox_inches="tight", facecolor="white", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        except TypeError:
            fig.savefig(tif, dpi=CFG.DPI, bbox_inches="tight", facecolor="white", format="tiff")
    if CFG.EXPORT_PDF:
        fig.savefig(pdf, dpi=CFG.DPI, bbox_inches="tight", facecolor="white")

    plt.close(fig)
    return {"figure_id": figure_id, "metric": metric, "threshold": q, "png": str(png), "tif": str(tif), "pdf": str(pdf)}


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_workflow() -> None:
    nc_dir, fig_dir, table_dir = output_dirs()

    print("=" * 80)
    print("Figure 4 spatial footprint workflow")
    print("=" * 80)
    print(f"Daily precipitation: {CFG.PRECIP_NC_FILE}")
    print(f"Final ENSO events : {CFG.ENSO_EXCEL}")
    print(f"Output folder      : {CFG.OUT_DIR}")
    print(f"Threshold design   : MEP95 q90 main; MEP95 q95 and Rx5day q90 supporting")
    print(f"Plot months        : {getattr(CFG, 'WARM_MONTHS', tuple(range(1, 13)))}")
    print(f"Wilson alpha       : {CFG.ALPHA}")
    print(f"Largest cluster    : {CFG.KEEP_LARGEST_SIGNIFICANT_CLUSTER}")
    print(f"Display mode       : reference layout, China-only significant RR + never categories")
    print(f"Stage 1 calculate  : {CFG.RECALCULATE_RR_NC}")
    print(f"Stage 2 plot NC    : {CFG.PLOT_FROM_RR_NC}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Stage 0. Build or reuse the grid-cell monthly MEP95-anomaly NetCDF.
    # ------------------------------------------------------------------
    if CFG.REBUILD_MONTHLY_METRIC_NC or not CFG.GRID_METRIC_NC.exists():
        if CFG.REBUILD_MONTHLY_METRIC_NC and CFG.GRID_METRIC_NC.exists():
            print("REBUILD_MONTHLY_METRIC_NC=True, so the grid metric NetCDF will be rebuilt.")
            print("If the existing NetCDF is open/locked, close it or set REBUILD_MONTHLY_METRIC_NC=False.")
        metric_nc = build_grid_metric_nc()
    else:
        metric_nc = CFG.GRID_METRIC_NC
        print(f"Using existing grid metric NetCDF without rebuilding: {metric_nc}")

    # Load the metric data into memory and close the file handle immediately.
    # This makes the source NetCDF easier to delete/rename on Windows after the run.
    with xr.open_dataset(metric_nc) as _ds0:
        ds_metric = ensure_metric_time_phase_coords(_ds0)
        available_vars = [v for v in ["MEP95_anomaly", "Rx5day_anomaly"] if v in ds_metric]
        ds_metric = ds_metric[available_vars].load()

    boundaries = load_boundaries()

    report_rows = []

    for spec in CFG.THRESHOLD_SPECS:
        q = int(spec["threshold"])
        metric = str(spec.get("metric", "MEP95"))
        figure_id = str(spec["figure_id"])
        rr_nc_target = rr_nc_path_for(figure_id, metric, q, nc_dir)

        print("\n" + "-" * 80)
        print(f"{figure_id}: {metric} q{q}")
        print(f"RR/significance NC target: {rr_nc_target}")
        print("-" * 80)

        # --------------------------------------------------------------
        # Stage 1. Calculate spatial RR/significance and write NetCDF.
        # --------------------------------------------------------------
        if CFG.RECALCULATE_RR_NC or not rr_nc_target.exists():
            ds_rr_calc = calculate_spatial_rr(ds_metric, q=q, metric=metric)

            if CFG.OVERWRITE_RR_NC:
                rr_nc_written = safe_write_netcdf(ds_rr_calc, rr_nc_target)
            else:
                stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                rr_nc_written = safe_write_netcdf(
                    ds_rr_calc,
                    rr_nc_target.with_name(f"{rr_nc_target.stem}_{stamp}{rr_nc_target.suffix}")
                )

            print(f"    Written calculated RR/significance NC: {rr_nc_written}")

            # Summary is also based on the calculated result.
            summary_csv = write_summary_table(ds_rr_calc, metric, q, table_dir)
            print(f"    Written summary table: {summary_csv}")

            try:
                ds_rr_calc.close()
            except Exception:
                pass
            del ds_rr_calc
            gc.collect()

            rr_nc_for_plot = Path(rr_nc_written)
        else:
            rr_nc_for_plot = rr_nc_target
            print(f"    Using existing calculated RR/significance NC: {rr_nc_for_plot}")

            # If only plotting from an existing NC, recreate the summary table
            # from the NC so summary and figure remain synchronized.
            ds_rr_summary = open_rr_nc_for_plot(rr_nc_for_plot)
            summary_csv = write_summary_table(ds_rr_summary, metric, q, table_dir)
            try:
                ds_rr_summary.close()
            except Exception:
                pass
            del ds_rr_summary
            gc.collect()

        # --------------------------------------------------------------
        # Stage 2. Reopen saved RR/significance NetCDF and plot from it.
        # --------------------------------------------------------------
        if CFG.PLOT_FROM_RR_NC:
            print(f"    Plotting from saved NC: {rr_nc_for_plot}")
            ds_rr_plot = open_rr_nc_for_plot(rr_nc_for_plot)

            fig_info = plot_spatial_footprints(ds_rr_plot, metric, q, figure_id, boundaries)
            fig_info["rr_nc"] = str(rr_nc_for_plot)
            fig_info["summary_csv"] = str(summary_csv)
            report_rows.append(fig_info)

            try:
                ds_rr_plot.close()
            except Exception:
                pass
            del ds_rr_plot
            gc.collect()
        else:
            report_rows.append({
                "figure_id": figure_id,
                "metric": metric,
                "threshold": q,
                "png": "",
                "tif": "",
                "pdf": "",
                "rr_nc": str(rr_nc_for_plot),
                "summary_csv": str(summary_csv),
            })

    report = pd.DataFrame(report_rows)
    report_path = table_dir / "Figure4_spatial_footprint_output_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    method = pd.DataFrame([
        {"item": "Metric", "value": "Main metric: grid-cell monthly MEP95 from wet days above local calendar-month wet-day P95. Supporting metric: monthly Rx5day."},
        {"item": "Anomaly", "value": f"MEP95 and Rx5day anomalies relative to grid-cell calendar-month climatology for {CFG.ANOMALY_BASELINE_START_YEAR}-{CFG.ANOMALY_BASELINE_END_YEAR}."},
        {"item": "Event", "value": "At each grid cell and calendar month, event = MEP95 anomaly >= q90 or q95."},
        {"item": "ENSO classification", "value": "ENSO phases are read from the finalized ENSO-event file; the preferred final phase column is main-phase."},
        {"item": "RR", "value": "RR = P(event | ENSO phase) / P(event | Neutral); plotted directly as RR using the reference-article color scale."},
        {"item": "Significance", "value": "Wilson-score CDF significance: for each grid cell, month, threshold and ENSO phase, Neutral event probability is significant when it lies outside the 95% Wilson confidence interval of the ENSO-phase event probability. Optional FDR filtering can be enabled with APPLY_FDR_AFTER_WILSON=True."},
        {"item": "Two-stage workflow", "value": "Stage 1 calculates q90/q95 spatial RR/Wilson significance/never categories and writes a NetCDF file. Stage 2 reopens the saved NetCDF and plots from it."},
        {"item": "Spatial display", "value": "Only Wilson-significant cells within China are displayed. Ocean shading and other countries borders are removed; China boundary, dashed province boundaries, six-basin/rivers overlay, basin labels and the ten-dash line are retained."},
        {"item": "Spatial screening", "value": "Wilson-significant cells are used for display and outlines. Never-in-Neutral, never-in-ENSO and never-in-either categories are displayed as special mask classes. KEEP_LARGEST_SIGNIFICANT_CLUSTER=False shows all Wilson-significant cells; set it to True to retain only the largest connected significant cluster."},
        {"item": "Existing NC use", "value": "An existing grid-cell MEP95 anomaly NetCDF can be reused if it contains MEP95_anomaly and a time coordinate; cal_month and phase_code are attached or overwritten from time and the final ENSO file."},
        {"item": "Figures", "value": "Figure4 uses MEP95 q90; FigureS4 uses MEP95 q95; FigureS5 uses Rx5day q90. All figures are calculated for all 12 calendar months."},
    ])
    method_path = table_dir / "Figure4_method_readme.csv"
    method.to_csv(method_path, index=False, encoding="utf-8-sig")

    try:
        ds_metric.close()
    except Exception:
        pass
    del ds_metric
    gc.collect()

    print("\nFinished.")
    print(f"Report: {report_path}")
    print(f"Method readme: {method_path}")


if __name__ == "__main__":
    run_workflow()
