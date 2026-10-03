#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.cm import ScalarMappable
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy import stats

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
    # Existing Figure 4 result root.
    CALC_OUT_DIR = OUTPUT_ROOT / "04_spatial_rr"
    CALC_NC_DIR = CALC_OUT_DIR / "00_netcdf"

    # Existing monthly NetCDF files.
    NC_MEP95_Q90 = (
        CALC_NC_DIR
        / "Figure4_MEP95_q90_spatial_RR_Wilson_significance_ChinaOnly_12months.nc"
    )
    NC_MEP95_Q95 = (
        CALC_NC_DIR
        / "FigureS4_MEP95_q95_spatial_RR_Wilson_significance_ChinaOnly_12months.nc"
    )
    NC_RX5DAY_Q90 = (
        CALC_NC_DIR
        / "FigureS5_Rx5day_q90_spatial_RR_Wilson_significance_ChinaOnly_12months.nc"
    )

    # Figure 5 output root.
    OUT_DIR = CALC_OUT_DIR / "03_figure5_results"

    # Boundary files copied from the latest Figure 4 plotting code.
    SHP_GLOBAL = DATA_ROOT / "boundaries" / "global_countries.shp"
    SHP_PROVINCE = DATA_ROOT / "boundaries" / "province_boundaries.shp"
    BASIN_POLYGON_FILE = DATA_ROOT / "basins" / "six_selected_official_basins.shp"

    # Statistical settings.
    ALPHA = 0.05
    # With a 1961-2023 record, n=3 is too permissive for q90/q95 risk ratios.
    # These defaults require a minimally interpretable phase/Neutral sample.
    MIN_PHASE_N = 10
    MIN_NEUTRAL_N = 10

    # Strict reference-style percentage effects:
    # RR >= 1.50 is at least a 50% increase;
    # RR <= 0.50 is at least a 50% decrease.
    RR_EFFECT_HIGH = 1.50
    RR_EFFECT_LOW = 0.50
    # Flag, but do not post-hoc delete, cells whose realized upper-tail event
    # frequency exceeds the nominal tail by more than two percentage points.
    TIE_RATE_TOLERANCE = 0.02

    # Only the basin summary defines "recurrent" as at least two months.
    # Month-count maps themselves always show the full 0-12 range.
    BASIN_RECURRENT_MIN_MONTHS = 2
    # A basin cell enters the area denominator only when it has valid event and
    # Neutral sample/count information in at least this many calendar months.
    BASIN_MIN_VALID_MONTHS = 12
    ASYMMETRY_MIN_PAIRED_MONTHS = 10
    ASYMMETRY_MIN_TOTAL_ABS_LOG2RR = 1.0
    # Haldane-Anscombe correction is used only for the descriptive asymmetry
    # index so that zero-event months do not erase the entire paired-month map.
    ASYMMETRY_ZERO_CORRECTION = 0.5

    # FDR family: one calendar-month × ENSO-phase spatial map at a time.
    # This is the primary setting used by this script.
    FDR_FAMILY = "month_phase_map"

    # Figure settings.
    # Keep the current compact mainland-China map ratio.  Neighbouring-country
    # boundaries are drawn as background context; no South China Sea inset is
    # added in this version.
    MAP_EXTENT = (73.0, 136.8, 17.0, 54.5)
    MAP_LON_TICKS = (80, 90, 100, 110, 120, 130)
    MAP_LAT_TICKS = (20, 30, 40, 50)
    MAP_TICK_FONTSIZE = 11.0
    MAP_TICK_LENGTH = 3.2
    MAP_TICK_WIDTH = 0.75
    # The global polygon file is used only for a quiet land fill outside
    # China. Its outlines are deliberately not plotted because some versions
    # also store offshore claim/dash geometries.
    FILL_NEIGHBOURING_COUNTRIES = True
    NEIGHBOUR_LAND_COLOR = "#F2F2F2"
    NEIGHBOUR_BORDER_COLOR = "#8F8F8F"
    NEIGHBOUR_BORDER_WIDTH = 0.38
    COASTLINE_COLOR = "#555555"
    COASTLINE_WIDTH = 0.48
    CARTOPY_BOUNDARY_SCALE = "50m"
    # Match the near-square main composition while retaining readable map
    # panels in the vertical 4 x 2 sensitivity and 3 x 2 diagnostic layouts.
    MAIN_FIGSIZE = (10.2, 10.0)
    SENSITIVITY_FIGSIZE = (10.2, 18.2)
    DIAG_FIGSIZE = (10.2, 14.8)
    QUALITY_FIGSIZE = (12.0, 10.2)
    SCREENING_FIGSIZE = (11.5, 8.6)
    BASIN_FIGSIZE = (8.4, 6.0)
    DPI = 500
    FONT_FAMILY = "Times New Roman"
    # Titles, panel letters, row/metric explanations, and diagnostic notes use
    # one enlarged size and the same typeface throughout the four figures.
    PANEL_TITLE_FONTSIZE = 19.0
    MAIN_PANEL_TITLE_FONTSIZE = 16.5
    DIAG_PANEL_TITLE_FONTSIZE = 19.0
    PANEL_LABEL_FONTSIZE = 20.0
    ANNOTATION_FONTSIZE = 20.0
    COLORBAR_TICK_FONTSIZE = 17.0
    COLORBAR_LABEL_FONTSIZE = 20.0
    BASIN_TICK_FONTSIZE = 12.5
    BASIN_AXIS_FONTSIZE = 13.5
    BASIN_LEGEND_FONTSIZE = 10.5
    SHOW_SUPTITLE = False
    EXPORT_PNG = True
    EXPORT_TIF = True
    EXPORT_PDF = True

    CHINA_LINE_COLOR = "#3f3f3f"
    CHINA_LINE_WIDTH = 0.85
    PROVINCE_LINE_COLOR = "#8a8a8a"
    PROVINCE_LINE_WIDTH = 0.35
    PROVINCE_LINESTYLE = (0, (2.4, 2.0))
    # Six major river basins: blue-grey dashed boundaries in every map panel.
    # This colour is deliberately separated from the red/orange recurrence colours
    # while remaining clearly visible over both light and dark map cells.
    BASIN_LINE_COLOR = "#4C5A6A"
    BASIN_LINE_WIDTH = 0.85
    BASIN_LINESTYLE = (0, (4.0, 2.4))


MONTH_LABEL = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
PHASES = ("El Nino", "La Nina")
PHASE_KEY = {"El Nino": "EN", "La Nina": "LN"}
PHASE_LABEL = {"El Nino": "El Niño", "La Nina": "La Niña"}
BASIN_ORDER = ("Songhua", "Haihe", "Yellow", "Huaihe", "Yangtze", "Pearl")

BASIN_NAME_CANDIDATES = (
    "basin", "BASIN", "basin_en", "BASIN_EN", "basin_name", "BASIN_NAME",
    "name", "NAME", "Name", "NAME_EN", "name_en", "NAME_ZH", "name_zh",
    "流域", "流域名", "名称",
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def set_style() -> None:
    mpl.rcParams.update({
        "font.family": CFG.FONT_FAMILY,
        "font.serif": [CFG.FONT_FAMILY],
        "font.size": CFG.PANEL_TITLE_FONTSIZE,
        "axes.titlesize": CFG.PANEL_TITLE_FONTSIZE,
        "axes.labelsize": CFG.BASIN_AXIS_FONTSIZE,
        "xtick.labelsize": CFG.BASIN_TICK_FONTSIZE,
        "ytick.labelsize": CFG.BASIN_TICK_FONTSIZE,
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def output_dirs() -> Tuple[Path, Path, Path]:
    nc_dir = CFG.OUT_DIR / "00_netcdf"
    fig_dir = CFG.OUT_DIR / "01_figures"
    table_dir = CFG.OUT_DIR / "02_tables"
    for path in (nc_dir, fig_dir, table_dir):
        path.mkdir(parents=True, exist_ok=True)
    return nc_dir, fig_dir, table_dir


def read_vector(path: Path, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")
    last_error: Optional[Exception] = None
    for encoding in ("gb18030", "gbk", "cp936", "utf-8"):
        try:
            kwargs: Dict[str, object] = {"encoding": encoding}
            if layer is not None:
                kwargs["layer"] = layer
            gdf = gpd.read_file(path, **kwargs)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            else:
                gdf = gdf.to_crs("EPSG:4326")
            return gdf
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read {path}; last error: {last_error}")


def union_geometry(gdf: gpd.GeoDataFrame):
    try:
        return gdf.geometry.union_all()
    except Exception:
        return gdf.geometry.unary_union


def identify_china_rows(world: gpd.GeoDataFrame) -> np.ndarray:
    """Identify China records without depending on one fixed field name."""
    candidates = (
        "CN_NAME", "NAME", "SOVEREIGNT", "ADMIN", "COUNTRY",
        "CNTRY_NAME", "NAME_ZH", "国家", "国名",
    )
    for column in candidates:
        if column not in world.columns:
            continue
        mask = world[column].astype(str).str.contains(
            "China|中国|CHN|中华人民共和国",
            case=False,
            na=False,
            regex=True,
        )
        if mask.any():
            return np.asarray(mask, dtype=bool)
    raise ValueError("Could not identify China polygon from SHP_GLOBAL.")


def load_china_boundary() -> gpd.GeoDataFrame:
    world = read_vector(CFG.SHP_GLOBAL)
    china_rows = identify_china_rows(world)
    return world.loc[china_rows].copy().dissolve().to_crs("EPSG:4326")


def normalize_basin_name(value: object) -> str:
    text = str(value).strip()
    low = text.lower().replace("_", " ")
    rules = (
        (("songhua", "松花", "northeast", "东北"), "Songhua"),
        (("haihe", "hai river", "海河"), "Haihe"),
        (("yellow", "黄河"), "Yellow"),
        (("huaihe", "huai river", "淮河"), "Huaihe"),
        (("yangtze", "changjiang", "长江"), "Yangtze"),
        (("pearl", "zhujiang", "珠江"), "Pearl"),
    )
    for keys, label in rules:
        if any(key in low or key in text for key in keys):
            return label
    return text


def find_basin_name_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    for column in BASIN_NAME_CANDIDATES:
        if column in gdf.columns:
            return column
    for column in gdf.columns:
        if (
            column != "geometry"
            and gdf[column].dtype == object
            and gdf[column].nunique(dropna=True) >= 4
        ):
            return column
    return None


def load_six_basins(china: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    basins = gpd.clip(read_vector(CFG.BASIN_POLYGON_FILE), china)
    name_column = find_basin_name_column(basins)
    if name_column is None:
        raise ValueError(
            "Could not identify the basin-name field. Set a valid name field "
            "in BASIN_NAME_CANDIDATES."
        )
    basins["basin_name"] = basins[name_column].map(normalize_basin_name)
    basins = basins[basins["basin_name"].isin(BASIN_ORDER)].copy()
    basins = basins.dissolve(by="basin_name", as_index=False)
    missing = [name for name in BASIN_ORDER if name not in set(basins["basin_name"])]
    if missing:
        raise ValueError(f"Missing official basin polygons after name matching: {missing}")
    return basins[["basin_name", "geometry"]].to_crs("EPSG:4326")


def geometry_to_lines(
    gdf: Optional[gpd.GeoDataFrame],
    polygon_to_boundary: bool = True,
) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    data = gdf[gdf.geometry.notnull()].copy()
    try:
        data = data.explode(index_parts=False).reset_index(drop=True)
    except Exception:
        pass
    geometries = []
    for geom in data.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type in ("LineString", "MultiLineString"):
            geometries.append(geom)
        elif polygon_to_boundary and geom.geom_type in ("Polygon", "MultiPolygon"):
            geometries.append(geom.boundary)
    return gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")


def geometry_to_polygons(
    gdf: Optional[gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """Retain polygonal land only and discard every line-like feature."""
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    data = gdf[gdf.geometry.notnull()].copy()
    try:
        data = data.explode(index_parts=False).reset_index(drop=True)
    except Exception:
        pass
    polygons = [
        geom for geom in data.geometry
        if geom is not None
        and not geom.is_empty
        and geom.geom_type in ("Polygon", "MultiPolygon")
    ]
    return gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")


def load_boundaries() -> Dict[str, gpd.GeoDataFrame]:
    china_polygon = load_china_boundary()
    world = read_vector(CFG.SHP_GLOBAL)
    china_rows = identify_china_rows(world)
    neighbours = world.loc[~china_rows].copy().cx[65:145, 10:60]
    province = read_vector(CFG.SHP_PROVINCE) if CFG.SHP_PROVINCE.exists() else None
    if province is not None and not province.empty:
        # Some administrative line packages append maritime claim lines to the
        # provincial layer.  Clipping every administrative line to the actual
        # China land polygon removes those offshore records at source.
        province = gpd.clip(province, china_polygon)
        province = province.loc[
            province.geometry.geom_type.isin(
                ("LineString", "MultiLineString", "Polygon", "MultiPolygon")
            )
        ].copy()
    basins = load_six_basins(china_polygon)
    return {
        "china_polygon": china_polygon,
        # Polygon fills only: no global/China outline from SHP_GLOBAL is drawn.
        "neighbour_land": geometry_to_polygons(neighbours),
        "province": geometry_to_lines(province),
        "basin_polygons": basins,
        "basins": geometry_to_lines(basins),
    }


def geometry_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    geometry,
) -> np.ndarray:
    xx, yy = np.meshgrid(lon, lat)
    try:
        import shapely
        if hasattr(shapely, "contains_xy"):
            inside = shapely.contains_xy(geometry, xx, yy)
        else:
            from shapely import vectorized
            inside = vectorized.contains(geometry, xx, yy)
    except Exception:
        from shapely.geometry import Point
        from shapely.prepared import prep
        prepared = prep(geometry)
        inside = np.zeros(xx.shape, dtype=bool)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                point = Point(float(xx[i, j]), float(yy[i, j]))
                inside[i, j] = prepared.contains(point) or prepared.covers(point)
    return np.asarray(inside, dtype=bool)


def make_china_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    china: gpd.GeoDataFrame,
) -> np.ndarray:
    return geometry_mask(lon, lat, union_geometry(china.to_crs("EPSG:4326")))


def make_basin_masks(
    lon: np.ndarray,
    lat: np.ndarray,
    basins: gpd.GeoDataFrame,
) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    indexed = basins.set_index("basin_name")
    for basin in BASIN_ORDER:
        geom = indexed.loc[basin, "geometry"]
        if isinstance(geom, pd.Series):
            geom = geom.unary_union
        masks[basin] = geometry_mask(lon, lat, geom)
    return masks


def plot_line(
    gdf: gpd.GeoDataFrame,
    ax,
    color: str,
    linewidth: float,
    linestyle="-",
    zorder: int = 20,
) -> None:
    if gdf is None or gdf.empty:
        return
    gdf.plot(
        ax=ax,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        transform=ccrs.PlateCarree(),
        zorder=zorder,
    )


def fill_polygon(
    gdf: gpd.GeoDataFrame,
    ax,
    facecolor: str,
    zorder: int,
) -> None:
    """Fill polygonal context without adding any line geometry."""
    if gdf is None or gdf.empty:
        return
    gdf.plot(
        ax=ax,
        facecolor=facecolor,
        edgecolor="none",
        linewidth=0.0,
        transform=ccrs.PlateCarree(),
        zorder=zorder,
    )


def draw_boundaries(
    ax,
    boundaries: Dict[str, gpd.GeoDataFrame],
    include_provinces: bool = True,
    include_basins: bool = True,
) -> None:
    if include_provinces:
        plot_line(
            boundaries["province"], ax,
            CFG.PROVINCE_LINE_COLOR, CFG.PROVINCE_LINE_WIDTH,
            CFG.PROVINCE_LINESTYLE, 22,
        )
    if include_basins:
        plot_line(
            boundaries["basins"], ax,
            CFG.BASIN_LINE_COLOR, CFG.BASIN_LINE_WIDTH,
            CFG.BASIN_LINESTYLE, 24,
        )
    # National and coastal context is drawn by Cartopy in setup_map().  Do not
    # draw SHP_GLOBAL's China outline here; that dataset may contain residual
    # offshore claim/dash geometries in the same record.


def resolve_nc(preferred: Path, metric: str, q: int) -> Path:
    if preferred.exists():
        return preferred
    patterns = (
        f"*{metric}*q{q}*spatial_RR*Wilson*12months*.nc",
        f"*{metric}*q{q}*RR*Wilson*.nc",
        f"*{metric}*q{q}*.nc",
    )
    matches: List[Path] = []
    if CFG.CALC_NC_DIR.exists():
        for pattern in patterns:
            matches.extend(sorted(CFG.CALC_NC_DIR.glob(pattern)))
    matches = list(dict.fromkeys(path for path in matches if path.exists()))
    if matches:
        print(f"Preferred NetCDF not found; using: {matches[0]}")
        return matches[0]
    raise FileNotFoundError(
        f"Missing {metric} q{q} monthly RR NetCDF:\n{preferred}\n"
        f"Copy it into {CFG.CALC_NC_DIR} or update the path in CFG."
    )


# =============================================================================
# SIGNIFICANCE AND HOTSPOT CALCULATION
# =============================================================================

def fdr_bh_adjust(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return BH-adjusted q values and the rejection mask in input shape."""
    p = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(p)
    qvalues = np.full(p.shape, np.nan, dtype=float)
    rejected = np.zeros(p.shape, dtype=bool)
    if not np.any(valid):
        return qvalues, rejected
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    ranks = np.arange(1, ranked.size + 1, dtype=float)

    # BH adjusted p values must be monotone with rank.
    adjusted_ranked = ranked * ranked.size / ranks
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted_unsorted = np.empty_like(adjusted_ranked)
    adjusted_unsorted[order] = adjusted_ranked
    qvalues[valid] = adjusted_unsorted
    rejected[valid] = adjusted_unsorted <= alpha
    return qvalues, rejected


def fdr_bh_mask(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Backward-compatible BH rejection-mask helper."""
    return fdr_bh_adjust(pvalues, alpha)[1]


def wilson_interval(
    count: np.ndarray,
    sample_size: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Wilson score interval for a binomial event probability."""
    count = np.asarray(count, dtype=float)
    sample_size = np.asarray(sample_size, dtype=float)
    low = np.full(count.shape, np.nan, dtype=float)
    high = np.full(count.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(count) & np.isfinite(sample_size)
        & (sample_size > 0) & (count >= 0) & (count <= sample_size)
    )
    if not np.any(valid):
        return low, high
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    n = sample_size[valid]
    phat = count[valid] / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(
        phat * (1.0 - phat) / n + z * z / (4.0 * n * n)
    ) / denom
    low[valid] = np.clip(centre - half, 0.0, 1.0)
    high[valid] = np.clip(centre + half, 0.0, 1.0)
    return low, high


def reference_wilson_test(
    c_phase: np.ndarray,
    n_phase: np.ndarray,
    c_neutral: np.ndarray,
    n_neutral: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproduce the reference-paper significance rule and its matching p value.

    The Neutral probability is compared with the Wilson interval around the
    ENSO-phase probability. The returned two-sided score p value is the
    inversion of that same Wilson interval; therefore raw p < alpha and the
    Wilson decision are the same test up to floating-point boundary cases.

    This intentionally reproduces the reference method. It does not pretend
    that the estimated Neutral probability is known without uncertainty.
    """
    c_phase = np.asarray(c_phase, dtype=float)
    n_phase = np.asarray(n_phase, dtype=float)
    c_neutral = np.asarray(c_neutral, dtype=float)
    n_neutral = np.asarray(n_neutral, dtype=float)
    valid = (
        np.isfinite(c_phase) & np.isfinite(n_phase)
        & np.isfinite(c_neutral) & np.isfinite(n_neutral)
        & (n_phase >= CFG.MIN_PHASE_N) & (n_neutral >= CFG.MIN_NEUTRAL_N)
        & (c_phase >= 0) & (c_neutral >= 0)
        & (c_phase <= n_phase) & (c_neutral <= n_neutral)
    )
    p_phase = np.divide(
        c_phase, n_phase,
        out=np.full(c_phase.shape, np.nan, dtype=float),
        where=n_phase > 0,
    )
    p_neutral = np.divide(
        c_neutral, n_neutral,
        out=np.full(c_neutral.shape, np.nan, dtype=float),
        where=n_neutral > 0,
    )
    low, high = wilson_interval(c_phase, n_phase, alpha)
    significant = (
        valid & np.isfinite(low) & np.isfinite(high)
        & ((p_neutral < low) | (p_neutral > high))
    )

    pvalue = np.full(c_phase.shape, np.nan, dtype=float)
    interior = valid & (p_neutral > 0.0) & (p_neutral < 1.0)
    se0 = np.sqrt(
        p_neutral * (1.0 - p_neutral)
        / np.where(n_phase > 0, n_phase, np.nan)
    )
    zscore = np.divide(
        p_phase - p_neutral,
        se0,
        out=np.zeros(c_phase.shape, dtype=float),
        where=interior & np.isfinite(se0) & (se0 > 0),
    )
    pvalue[interior] = 2.0 * stats.norm.sf(np.abs(zscore[interior]))

    # Boundary null probabilities have zero score variance. Their limiting
    # score p is 1 for equality and 0 for any departure.
    boundary = valid & ((p_neutral == 0.0) | (p_neutral == 1.0))
    pvalue[boundary] = np.where(
        np.isclose(p_phase[boundary], p_neutral[boundary]), 1.0, 0.0
    )
    return significant, pvalue, p_phase, p_neutral, low, high


def get_required(
    ds: xr.Dataset,
    name: str,
    alternatives: Iterable[str] = (),
) -> np.ndarray:
    for candidate in (name, *alternatives):
        if candidate in ds:
            return np.asarray(ds[candidate].values)
    tried = ", ".join((name, *alternatives))
    raise KeyError(f"None of the required NetCDF variables exists: {tried}")


def calculate_hotspots(
    ds: xr.Dataset,
    q: int,
    china_mask: np.ndarray,
) -> Tuple[xr.Dataset, pd.DataFrame]:
    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    ny, nx = len(lat), len(lon)
    shape_phase = (len(PHASES), ny, nx)

    # Descriptive effect magnitude and inferential support are kept separate.
    # This distinction is central to the revised Figure 5 design.
    inc_effect_count = np.zeros(shape_phase, dtype=np.int16)
    dec_effect_count = np.zeros(shape_phase, dtype=np.int16)
    inc_supported_count = np.zeros(shape_phase, dtype=np.int16)
    dec_supported_count = np.zeros(shape_phase, dtype=np.int16)
    valid_month_count = np.zeros(shape_phase, dtype=np.int16)
    wilson_count = np.zeros(shape_phase, dtype=np.int16)
    raw_score_count = np.zeros(shape_phase, dtype=np.int16)
    fdr_count = np.zeros(shape_phase, dtype=np.int16)
    zero_count = np.zeros(shape_phase, dtype=np.int16)
    effect_month_count = np.zeros(shape_phase, dtype=np.int16)
    tail_frequency_excess_count = np.zeros((ny, nx), dtype=np.int16)
    asymmetry_log2rr = np.full(
        (len(PHASES), 12, ny, nx), np.nan, dtype=np.float32
    )
    diagnostics: List[Dict[str, object]] = []
    nominal_event_rate = 1.0 - q / 100.0

    for month in range(1, 13):
        monthly: Dict[str, Dict[str, np.ndarray]] = {}
        for phase in PHASES:
            phase_key = PHASE_KEY[phase]
            base = f"q{q}_{MONTH_LABEL[month]}_{phase_key}"
            monthly[phase] = {
                "c_phase": get_required(ds, f"{base}_c_phase").astype(float),
                "n_phase": get_required(ds, f"{base}_n_phase").astype(float),
                "c_neutral": get_required(ds, f"{base}_c_neutral").astype(float),
                "n_neutral": get_required(ds, f"{base}_n_neutral").astype(float),
                "stored_wilson": get_required(
                    ds,
                    f"{base}_sig_wilson",
                    alternatives=(f"{base}_sig_final", f"{base}_sig_display"),
                ).astype(bool),
            }

        # Tied-threshold audit: realized all-phase event frequency can exceed
        # the nominal q-tail frequency when many values equal the percentile.
        en_data = monthly["El Nino"]
        ln_data = monthly["La Nina"]
        same_neutral = (
            np.isclose(en_data["c_neutral"], ln_data["c_neutral"], equal_nan=True)
            & np.isclose(en_data["n_neutral"], ln_data["n_neutral"], equal_nan=True)
        )
        c_total = (
            en_data["c_phase"] + ln_data["c_phase"] + en_data["c_neutral"]
        )
        n_total = (
            en_data["n_phase"] + ln_data["n_phase"] + en_data["n_neutral"]
        )
        realized_rate = np.divide(
            c_total, n_total,
            out=np.full((ny, nx), np.nan, dtype=float),
            where=(n_total > 0) & same_neutral,
        )
        tail_frequency_excess = (
            china_mask & np.isfinite(realized_rate)
            & (realized_rate > nominal_event_rate + CFG.TIE_RATE_TOLERANCE)
        )
        tail_frequency_excess_count += tail_frequency_excess.astype(np.int16)

        for phase_index, phase in enumerate(PHASES):
            data = monthly[phase]
            c_phase = data["c_phase"]
            n_phase = data["n_phase"]
            c_neutral = data["c_neutral"]
            n_neutral = data["n_neutral"]
            stored_wilson = data["stored_wilson"]

            wilson, score_p, p_phase, p_neutral, ci_low, ci_high = (
                reference_wilson_test(
                    c_phase, n_phase, c_neutral, n_neutral, CFG.ALPHA
                )
            )
            sample_valid = (
                china_mask
                & np.isfinite(p_phase) & np.isfinite(p_neutral)
                & (n_phase >= CFG.MIN_PHASE_N)
                & (n_neutral >= CFG.MIN_NEUTRAL_N)
            )
            zero_any = sample_valid & ((c_phase == 0) | (c_neutral == 0))
            finite_nonzero = sample_valid & (~zero_any)
            # Raw RR is mathematically defined when the numerator is zero
            # (RR=0), but not when the Neutral denominator is zero.
            rr_defined = sample_valid & (p_neutral > 0)
            rr = np.divide(
                p_phase, p_neutral,
                out=np.full((ny, nx), np.nan, dtype=float),
                where=rr_defined,
            )
            # A small, explicitly documented Haldane-Anscombe correction is
            # used only for the descriptive paired-phase asymmetry index. It
            # is not used to decide the 50% effect maps or Wilson support.
            correction = float(CFG.ASYMMETRY_ZERO_CORRECTION)
            p_phase_corrected = np.divide(
                c_phase + correction,
                n_phase + 2.0 * correction,
                out=np.full((ny, nx), np.nan, dtype=float),
                where=sample_valid,
            )
            p_neutral_corrected = np.divide(
                c_neutral + correction,
                n_neutral + 2.0 * correction,
                out=np.full((ny, nx), np.nan, dtype=float),
                where=sample_valid,
            )
            corrected_rr = np.divide(
                p_phase_corrected,
                p_neutral_corrected,
                out=np.full((ny, nx), np.nan, dtype=float),
                where=(
                    sample_valid
                    & np.isfinite(p_phase_corrected)
                    & np.isfinite(p_neutral_corrected)
                    & (p_neutral_corrected > 0)
                ),
            )
            corrected_log2rr = np.log2(corrected_rr)
            asymmetry_log2rr[phase_index, month - 1] = np.where(
                sample_valid & np.isfinite(corrected_log2rr),
                corrected_log2rr,
                np.nan,
            ).astype(np.float32)

            # The BH family is predeclared by geography and sample coverage,
            # not selected after observing whether counts are zero. Boundary
            # score p values are therefore included, while zero-driven
            # rejections are reported separately below.
            fdr_valid = sample_valid & np.isfinite(score_p)
            qvalue, fdr = fdr_bh_adjust(
                np.where(fdr_valid, score_p, np.nan), CFG.ALPHA
            )
            raw_score_sig = fdr_valid & (score_p < CFG.ALPHA)
            fdr &= fdr_valid
            wilson_valid = wilson & sample_valid
            wilson_rr_defined = wilson_valid & rr_defined

            # Main-map colours describe estimated effect magnitude. Wilson
            # support is retained separately for diagnostics/NetCDF output.
            effect_eligible = rr_defined & np.isfinite(rr) & (rr >= 0)
            enhanced_effect = effect_eligible & (rr >= CFG.RR_EFFECT_HIGH)
            reduced_effect = effect_eligible & (rr <= CFG.RR_EFFECT_LOW)
            effect = enhanced_effect | reduced_effect
            enhanced_supported = enhanced_effect & wilson_valid
            reduced_supported = reduced_effect & wilson_valid
            wilson_increase = (
                wilson_rr_defined & np.isfinite(rr) & (rr > 1.0)
            )
            wilson_decrease = (
                wilson_rr_defined & np.isfinite(rr) & (rr < 1.0)
            )

            valid_month_count[phase_index] += sample_valid.astype(np.int16)
            wilson_count[phase_index] += wilson_valid.astype(np.int16)
            raw_score_count[phase_index] += raw_score_sig.astype(np.int16)
            fdr_count[phase_index] += fdr.astype(np.int16)
            zero_count[phase_index] += zero_any.astype(np.int16)
            inc_effect_count[phase_index] += enhanced_effect.astype(np.int16)
            dec_effect_count[phase_index] += reduced_effect.astype(np.int16)
            inc_supported_count[phase_index] += enhanced_supported.astype(np.int16)
            dec_supported_count[phase_index] += reduced_supported.astype(np.int16)
            effect_month_count[phase_index] += effect.astype(np.int16)

            audit_domain = sample_valid & np.isfinite(ci_low) & np.isfinite(ci_high)
            mismatch = audit_domain & (stored_wilson != wilson)
            valid_p = score_p[fdr_valid]
            valid_q = qvalue[fdr_valid]
            diagnostics.append({
                "phase": PHASE_LABEL[phase],
                "month": month,
                "month_label": MONTH_LABEL[month],
                "significance_method": (
                    "Neutral probability outside ENSO-phase 95% Wilson interval"
                ),
                "bh_pvalue_method": "score-test inversion of the same Wilson interval",
                "valid_china_cells": int(sample_valid.sum()),
                "finite_nonzero_tested_cells": int(finite_nonzero.sum()),
                "bh_family_valid_cells": int(fdr_valid.sum()),
                "finite_nonzero_effect_domain_cells": int(finite_nonzero.sum()),
                "rr_defined_effect_domain_cells": int(rr_defined.sum()),
                "stored_wilson_mismatch_cells": int(mismatch.sum()),
                "upper_tail_frequency_excess_cells": int(
                    tail_frequency_excess.sum()
                ),
                "minimum_raw_score_pvalue": (
                    float(np.nanmin(valid_p)) if valid_p.size else np.nan
                ),
                "minimum_BH_qvalue": (
                    float(np.nanmin(valid_q)) if valid_q.size else np.nan
                ),
                "BH_first_rank_threshold": (
                    float(CFG.ALPHA / valid_p.size) if valid_p.size else np.nan
                ),
                "wilson_significant_valid_cells": int(wilson_valid.sum()),
                "wilson_significant_nonzero_cells": int(
                    (wilson_valid & finite_nonzero).sum()
                ),
                "raw_score_p_lt_alpha_cells": int(raw_score_sig.sum()),
                "bh_fdr_significant_valid_cells": int(fdr.sum()),
                "bh_fdr_significant_nonzero_cells": int(
                    (fdr & finite_nonzero).sum()
                ),
                "bh_fdr_significant_zero_event_cells": int(
                    (fdr & zero_any).sum()
                ),
                "zero_event_cells": int(zero_any.sum()),
                "estimated_enhanced_effect_cells": int(enhanced_effect.sum()),
                "estimated_reduced_effect_cells": int(reduced_effect.sum()),
                "wilson_increase_direction_cells": int(wilson_increase.sum()),
                "wilson_decrease_direction_cells": int(wilson_decrease.sum()),
                "wilson_supported_enhanced_cells": int(enhanced_supported.sum()),
                "wilson_supported_reduced_cells": int(reduced_supported.sum()),
                # Backward-compatible aliases: these are strict intersections.
                "enhanced_hotspot_cells": int(enhanced_supported.sum()),
                "reduced_hotspot_cells": int(reduced_supported.sum()),
            })

    en = asymmetry_log2rr[0]
    ln = asymmetry_log2rr[1]
    paired = np.isfinite(en) & np.isfinite(ln)
    en_pair = np.where(paired, en, 0.0)
    ln_pair = np.where(paired, ln, 0.0)
    paired_months = np.sum(paired, axis=0)
    asym_numerator = np.sum(np.abs(en_pair + ln_pair), axis=0)
    asym_denominator = np.sum(np.abs(en_pair) + np.abs(ln_pair), axis=0)
    asymmetry = np.divide(
        asym_numerator,
        asym_denominator,
        out=np.full((ny, nx), np.nan, dtype=float),
        where=asym_denominator > 0,
    )
    asymmetry[
        (paired_months < CFG.ASYMMETRY_MIN_PAIRED_MONTHS)
        | (asym_denominator < CFG.ASYMMETRY_MIN_TOTAL_ABS_LOG2RR)
        | (~china_mask)
    ] = np.nan

    retention = np.divide(
        fdr_count,
        wilson_count,
        out=np.full(shape_phase, np.nan, dtype=float),
        where=wilson_count > 0,
    )
    derived = xr.Dataset(
        {
            "china_mask": (("lat", "lon"), china_mask.astype(np.int8)),
            "enhanced_month_count": (
                ("phase", "lat", "lon"), inc_effect_count.astype(np.int8)
            ),
            "reduced_month_count": (
                ("phase", "lat", "lon"), dec_effect_count.astype(np.int8)
            ),
            "wilson_supported_enhanced_month_count": (
                ("phase", "lat", "lon"), inc_supported_count.astype(np.int8)
            ),
            "wilson_supported_reduced_month_count": (
                ("phase", "lat", "lon"), dec_supported_count.astype(np.int8)
            ),
            "analysis_valid_month_count": (
                ("phase", "lat", "lon"), valid_month_count.astype(np.int8)
            ),
            "wilson_significant_month_count": (
                ("phase", "lat", "lon"), wilson_count.astype(np.int8)
            ),
            "raw_wilson_score_p_lt_alpha_month_count": (
                ("phase", "lat", "lon"), raw_score_count.astype(np.int8)
            ),
            "bh_fdr_wilson_score_month_count": (
                ("phase", "lat", "lon"), fdr_count.astype(np.int8)
            ),
            "zero_event_month_count": (
                ("phase", "lat", "lon"), zero_count.astype(np.int8)
            ),
            "effect_month_count": (
                ("phase", "lat", "lon"), effect_month_count.astype(np.int8)
            ),
            "fdr_retention_fraction": (
                ("phase", "lat", "lon"), retention.astype(np.float32)
            ),
            "upper_tail_frequency_excess_month_count": (
                ("lat", "lon"), tail_frequency_excess_count.astype(np.int8)
            ),
            "enso_asymmetry_index": (
                ("lat", "lon"), asymmetry.astype(np.float32)
            ),
            "paired_valid_month_count": (
                ("lat", "lon"), paired_months.astype(np.int8)
            ),
            "paired_total_abs_log2RR": (
                ("lat", "lon"), asym_denominator.astype(np.float32)
            ),
        },
        coords={
            "phase": [PHASE_LABEL[p] for p in PHASES],
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "threshold_percentile": int(q),
            "main_significance": (
                "Neutral probability outside ENSO-phase 95% Wilson interval"
            ),
            "main_effect_thresholds": (
                f"RR >= {CFG.RR_EFFECT_HIGH} or RR <= {CFG.RR_EFFECT_LOW}"
            ),
            "rr_calculation": (
                "empirical count ratio without continuity correction; "
                "phase-zero/Neutral-positive gives RR=0; Neutral-zero is "
                "undefined and excluded from effect colours"
            ),
            "main_colour_role": (
                "descriptive effect magnitude; no pointwise significance claim"
            ),
            "wilson_support_role": (
                "supplementary diagnostic only; not overlaid on Figure 5"
            ),
            "fdr_method": "Benjamini-Hochberg",
            "fdr_raw_pvalue": "score-test inversion of reference Wilson interval",
            "fdr_family": CFG.FDR_FAMILY,
            "fdr_alpha": float(CFG.ALPHA),
            "fdr_role": "supplementary robustness diagnostic, not main hotspot mask",
            "zero_event_policy": (
                "included in the predeclared BH family and retained as a "
                "separate diagnostic; phase-zero/Neutral-positive may enter "
                "the RR=0 reduction map, Neutral-zero cannot enter RR colours"
            ),
            "asymmetry_index": (
                "sum_m(abs(log2RR_EN + log2RR_LN)) / "
                "sum_m(abs(log2RR_EN) + abs(log2RR_LN)); "
                "0=equal-and-opposite mirror, 1=fully non-mirror; "
                f"RR uses Haldane-Anscombe correction {CFG.ASYMMETRY_ZERO_CORRECTION} "
                "for this descriptive index only"
            ),
        },
    )
    derived["enhanced_month_count"].attrs.update({
        "long_name": "months with estimated empirical RR at or above 1.50",
        "inferential_role": "effect-size recurrence; not significance",
    })
    derived["reduced_month_count"].attrs.update({
        "long_name": "months with estimated empirical RR at or below 0.50",
        "inferential_role": "effect-size recurrence; not significance",
    })
    derived["wilson_supported_enhanced_month_count"].attrs.update({
        "long_name": "months meeting RR >= 1.50 and reference-Wilson significance",
    })
    derived["wilson_supported_reduced_month_count"].attrs.update({
        "long_name": "months meeting RR <= 0.50 and reference-Wilson significance",
    })
    return derived, pd.DataFrame(diagnostics)


def coordinate_edges(values: np.ndarray, clip: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Cell edges for a monotonic one-dimensional centre coordinate."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Latitude/longitude coordinates must be 1-D with >=2 cells.")
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    if clip is not None:
        edges = np.clip(edges, clip[0], clip[1])
    return edges


def gridcell_area_km2(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Exact spherical area of each regular/rectilinear latitude-longitude cell."""
    radius_km = 6371.0088
    lat_edges = np.deg2rad(coordinate_edges(lat, (-90.0, 90.0)))
    lon_edges = np.deg2rad(coordinate_edges(lon))
    d_sin_lat = np.abs(np.diff(np.sin(lat_edges)))
    d_lon = np.abs(np.diff(lon_edges))
    return (radius_km ** 2) * d_sin_lat[:, None] * d_lon[None, :]


def basin_summary(
    derived: xr.Dataset,
    basin_masks: Dict[str, np.ndarray],
) -> pd.DataFrame:
    lat = np.asarray(derived["lat"].values)
    lon = np.asarray(derived["lon"].values)
    cell_area = gridcell_area_km2(lat, lon)
    valid_months = np.asarray(derived["analysis_valid_month_count"].values)
    rows: List[Dict[str, object]] = []
    for phase_index, phase in enumerate(PHASES):
        # Basin translation uses the strict intersection, not the descriptive
        # effect-only colours of Figure 5. This keeps basin recurrence an
        # inferentially supported result.
        inc = np.asarray(
            derived["wilson_supported_enhanced_month_count"].values[phase_index]
        )
        dec = np.asarray(
            derived["wilson_supported_reduced_month_count"].values[phase_index]
        )
        for basin in BASIN_ORDER:
            nominal_mask = basin_masks[basin] & np.isfinite(cell_area)
            mask = nominal_mask & (
                valid_months[phase_index] >= CFG.BASIN_MIN_VALID_MONTHS
            )
            nominal_area = float(np.sum(cell_area[nominal_mask]))
            denom = float(np.sum(cell_area[mask]))
            if denom <= 0:
                inc_fraction = np.nan
                dec_fraction = np.nan
                inc_mean = np.nan
                dec_mean = np.nan
                inc_area = np.nan
                dec_area = np.nan
            else:
                inc_area = float(np.sum(
                    cell_area[mask & (inc >= CFG.BASIN_RECURRENT_MIN_MONTHS)]
                ))
                dec_area = float(np.sum(
                    cell_area[mask & (dec >= CFG.BASIN_RECURRENT_MIN_MONTHS)]
                ))
                inc_fraction = 100.0 * inc_area / denom
                dec_fraction = 100.0 * dec_area / denom
                inc_mean = float(np.average(inc[mask], weights=cell_area[mask]))
                dec_mean = float(np.average(dec[mask], weights=cell_area[mask]))
            rows.append({
                "basin": basin,
                "phase": PHASE_LABEL[phase],
                "recurrent_definition_months": CFG.BASIN_RECURRENT_MIN_MONTHS,
                "minimum_valid_months_for_denominator": CFG.BASIN_MIN_VALID_MONTHS,
                "nominal_basin_grid_area_km2": nominal_area,
                "valid_analysis_area_km2": denom,
                "valid_area_coverage_percent": (
                    100.0 * denom / nominal_area if nominal_area > 0 else np.nan
                ),
                "enhanced_recurrent_area_km2": inc_area,
                "reduced_recurrent_area_km2": dec_area,
                "enhanced_recurrent_area_percent": inc_fraction,
                "reduced_recurrent_area_percent": dec_fraction,
                "area_weighted_mean_enhanced_month_count": inc_mean,
                "area_weighted_mean_reduced_month_count": dec_mean,
            })
    return pd.DataFrame(rows)


# =============================================================================
# PLOTTING
# =============================================================================

def count_cmap() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    """Exact 13-bin 0-12 month palette sampled from the reference figure."""
    colors = [
        "#ffffff",  # 0
        "#88c5ea",  # 1
        "#3c82c5",  # 2
        "#14499d",  # 3
        "#e9e18a",  # 4
        "#f3a017",  # 5
        "#e71b1a",  # 6
        "#c486b8",  # 7
        "#ac539c",  # 8
        "#771d77",  # 9
        "#5ec0ba",  # 10
        "#117b7d",  # 11
        "#2d4e4d",  # 12
    ]
    cmap = mcolors.ListedColormap(colors)
    cmap.set_bad("white", alpha=0.0)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, 13.5, 1.0), cmap.N)
    return cmap, norm


def effect_count_cmap() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    """Effect maps use the same reference 0-12 discrete month palette."""
    return count_cmap()


def diagnostic_cmap() -> Tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    return count_cmap()


def setup_map(ax, boundaries: Dict[str, gpd.GeoDataFrame]) -> None:
    ax.set_extent(CFG.MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    # Fill only neighbouring terrestrial polygons. China and the ocean remain
    # white, so the China-only raster keeps the highest visual priority.
    if CFG.FILL_NEIGHBOURING_COUNTRIES:
        fill_polygon(
            boundaries["neighbour_land"],
            ax,
            CFG.NEIGHBOUR_LAND_COLOR,
            1,
        )
    fill_polygon(boundaries["china_polygon"], ax, "white", 2)

    # Cartopy BORDERS uses Natural Earth's land-only admin-0 boundary lines.
    # It therefore supplies country context without the ten-dash maritime line.
    ax.add_feature(
        cfeature.BORDERS.with_scale(CFG.CARTOPY_BOUNDARY_SCALE),
        facecolor="none",
        edgecolor=CFG.NEIGHBOUR_BORDER_COLOR,
        linewidth=CFG.NEIGHBOUR_BORDER_WIDTH,
        zorder=25,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale(CFG.CARTOPY_BOUNDARY_SCALE),
        facecolor="none",
        edgecolor=CFG.COASTLINE_COLOR,
        linewidth=CFG.COASTLINE_WIDTH,
        zorder=26,
    )
    ax.set_xticks(CFG.MAP_LON_TICKS, crs=ccrs.PlateCarree())
    ax.set_yticks(CFG.MAP_LAT_TICKS, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(
        LongitudeFormatter(
            zero_direction_label=False,
            dateline_direction_label=False,
            degree_symbol="°",
            number_format=".0f",
        )
    )
    ax.yaxis.set_major_formatter(
        LatitudeFormatter(
            degree_symbol="°",
            number_format=".0f",
        )
    )
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=CFG.MAP_TICK_FONTSIZE,
        length=CFG.MAP_TICK_LENGTH,
        width=CFG.MAP_TICK_WIDTH,
        pad=1.8,
        top=False,
        right=False,
        labeltop=False,
        labelright=False,
    )
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontfamily(CFG.FONT_FAMILY)
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_edgecolor("#777777")


def plot_grid(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    data: np.ndarray,
    cmap,
    norm,
    boundaries: Dict[str, gpd.GeoDataFrame],
    title: str,
    panel_label: str,
    title_color: str = "#222222",
    annotate_all_zero: bool = False,
    title_fontsize: Optional[float] = None,
    panel_label_inside: bool = False,
    title_x: Optional[float] = None,
    title_fontweight: str = "bold",
) -> None:
    setup_map(ax, boundaries)
    ax.pcolormesh(
        lon,
        lat,
        data,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        transform=ccrs.PlateCarree(),
        zorder=10,
        antialiased=False,
        rasterized=True,
    )
    # Keep the national, provincial and six-basin lines above the raster.
    draw_boundaries(ax, boundaries)
    if title:
        ax.set_title(
            title,
            loc="left",
            x=(0.075 if title_x is None else title_x),
            pad=9.0,
            fontsize=(
                CFG.PANEL_TITLE_FONTSIZE
                if title_fontsize is None else title_fontsize
            ),
            fontweight=title_fontweight,
            color=title_color,
        )
    if panel_label:
        if panel_label_inside:
            label_x, label_y = 0.018, 0.982
            label_va = "top"
            label_bbox = {
                "facecolor": "white", "edgecolor": "none",
                "alpha": 0.76, "pad": 0.25,
            }
            label_clip = True
        else:
            # Journal-style panel number immediately outside the upper-left
            # map frame.  The small vertical offset avoids touching the frame
            # without creating an excessive gap between subplot rows.
            label_x, label_y = 0.000, 1.015
            label_va = "bottom"
            label_bbox = None
            label_clip = False
        ax.text(
            label_x, label_y, panel_label,
            transform=ax.transAxes,
            ha="left", va=label_va,
            fontsize=CFG.PANEL_LABEL_FONTSIZE,
            fontweight="bold",
            bbox=label_bbox,
            clip_on=label_clip,
            zorder=40,
        )
    finite = np.asarray(data)[np.isfinite(data)]
    if annotate_all_zero and finite.size and np.nanmax(finite) == 0:
        ax.text(
            0.035, 0.91,
            "No valid grid cell passed\nBH-FDR (q < 0.05)",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=CFG.ANNOTATION_FONTSIZE,
            fontweight="bold", color="#555555",
            bbox={
                "facecolor": "white", "edgecolor": "#b0b0b0",
                "alpha": 0.90, "boxstyle": "round,pad=0.35",
            },
            zorder=50,
        )


def add_wilson_stippling(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    supported_month_count: np.ndarray,
    minimum_months: int = 1,
    maximum_points: int = 3500,
) -> int:
    """Overlay deterministic, spatially thinned Wilson-support stippling."""
    supported = (
        np.isfinite(supported_month_count)
        & (supported_month_count >= minimum_months)
    )
    if not np.any(supported):
        return 0

    yy, xx = np.where(supported)
    if yy.size > maximum_points:
        # Thin in both dimensions rather than taking the first N cells, which
        # would create a geographic sampling bias in the displayed stippling.
        stride = int(np.ceil(np.sqrt(yy.size / maximum_points)))
        keep = (yy % stride == 0) & (xx % stride == 0)
        yy, xx = yy[keep], xx[keep]
    ax.scatter(
        lon[xx], lat[yy],
        s=2.0,
        marker=".",
        c="#111111",
        edgecolors="none",
        alpha=0.68,
        transform=ccrs.PlateCarree(),
        zorder=18,
        rasterized=True,
    )
    return int(yy.size)


def add_count_colorbar(fig, cmap, norm, label: str, ax_pos) -> None:
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes(ax_pos)
    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        ticks=range(0, 13),
        spacing="uniform",
        drawedges=True,
    )
    cbar.set_label(
        label,
        fontsize=CFG.COLORBAR_LABEL_FONTSIZE,
        fontweight="bold",
        labelpad=3.0,
    )
    cbar.ax.tick_params(
        labelsize=CFG.COLORBAR_TICK_FONTSIZE,
        length=3.0,
        width=0.8,
        pad=1.8,
    )
    cbar.outline.set_linewidth(0.8)
    try:
        cbar.dividers.set_linewidth(0.45)
    except Exception:
        pass


def add_effect_count_colorbar(fig, cmap, norm, ax_pos) -> None:
    """Reference-style 13-bin horizontal legend for 0-12 month counts."""
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes(ax_pos)
    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        ticks=range(0, 13),
        spacing="uniform",
        drawedges=True,
    )
    cbar.set_label(
        "Number of months",
        fontsize=CFG.COLORBAR_LABEL_FONTSIZE,
        fontweight="bold",
        labelpad=4.0,
    )
    cbar.ax.tick_params(
        labelsize=CFG.COLORBAR_TICK_FONTSIZE,
        length=3.2,
        width=0.8,
        pad=2.0,
    )
    cbar.outline.set_linewidth(0.8)
    try:
        cbar.dividers.set_linewidth(0.40)
    except Exception:
        pass


def plot_basin_panel(
    ax,
    summary: pd.DataFrame,
    panel_label: Optional[str] = None,
) -> None:
    y = np.arange(len(BASIN_ORDER), dtype=float)
    offsets = {
        ("El Niño", "enhanced"): 0.21,
        ("La Niña", "enhanced"): 0.07,
        ("El Niño", "reduced"): -0.07,
        ("La Niña", "reduced"): -0.21,
    }
    styles = {
        ("El Niño", "enhanced"): ("#c62828", "o", True, "El Niño enhanced"),
        ("La Niña", "enhanced"): ("#1565c0", "s", True, "La Niña enhanced"),
        ("El Niño", "reduced"): ("#c62828", "o", False, "El Niño reduced"),
        ("La Niña", "reduced"): ("#1565c0", "s", False, "La Niña reduced"),
    }

    for phase in ("El Niño", "La Niña"):
        phase_data = summary[summary["phase"] == phase].set_index("basin")
        for direction, column in (
            ("enhanced", "enhanced_recurrent_area_percent"),
            ("reduced", "reduced_recurrent_area_percent"),
        ):
            color, marker, filled, label = styles[(phase, direction)]
            values = phase_data.loc[list(BASIN_ORDER), column].values.astype(float)
            yy = y + offsets[(phase, direction)]
            ax.scatter(
                values,
                yy,
                s=42,
                marker=marker,
                facecolor=color if filled else "white",
                edgecolor=color,
                linewidth=1.2,
                label=label,
                zorder=10,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(BASIN_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Fraction of valid basin area with Wilson-supported effects in "
        f"≥{CFG.BASIN_RECURRENT_MIN_MONTHS} months (%)",
        fontsize=CFG.BASIN_AXIS_FONTSIZE,
        fontweight="bold",
    )
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=CFG.BASIN_TICK_FONTSIZE)
    ax.set_title(
        "Area-normalized Wilson-supported recurrence",
        pad=3.0,
        fontsize=CFG.PANEL_TITLE_FONTSIZE,
        fontweight="bold",
    )
    if panel_label:
        ax.text(
            0.012, 0.985, panel_label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=CFG.PANEL_LABEL_FONTSIZE, fontweight="bold",
        )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.19),
        ncol=2,
        frameon=False,
        fontsize=CFG.BASIN_LEGEND_FONTSIZE,
        handletextpad=0.4,
        labelspacing=0.35,
    )


def plot_basin_summary_figure(
    summary: pd.DataFrame,
    out_prefix: Path,
) -> Dict[str, str]:
    """Export the normalized basin comparison as a Figure 7/SI candidate."""
    fig, ax = plt.subplots(figsize=CFG.BASIN_FIGSIZE, dpi=CFG.DPI)
    plot_basin_panel(ax, summary)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.90, bottom=0.29)
    return save_figure(fig, out_prefix)


def save_figure(fig, prefix: Path) -> Dict[str, str]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    if CFG.EXPORT_PNG:
        path = prefix.with_suffix(".png")
        fig.savefig(path, dpi=CFG.DPI, bbox_inches="tight", facecolor="white")
        outputs["png"] = str(path)
    if CFG.EXPORT_TIF:
        path = prefix.with_suffix(".tif")
        try:
            fig.savefig(
                path,
                dpi=CFG.DPI,
                bbox_inches="tight",
                facecolor="white",
                format="tiff",
                pil_kwargs={"compression": "tiff_lzw"},
            )
        except TypeError:
            fig.savefig(
                path,
                dpi=CFG.DPI,
                bbox_inches="tight",
                facecolor="white",
                format="tiff",
            )
        outputs["tif"] = str(path)
    if CFG.EXPORT_PDF:
        path = prefix.with_suffix(".pdf")
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs["pdf"] = str(path)
    plt.close(fig)
    return outputs


def extract_effect_layers(
    derived: xr.Dataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return comparable 12-month effect counts; white denotes missing coverage."""
    lon = np.asarray(derived["lon"].values)
    lat = np.asarray(derived["lat"].values)
    inc = np.asarray(derived["enhanced_month_count"].values)
    dec = np.asarray(derived["reduced_month_count"].values)
    valid_months = np.asarray(derived["analysis_valid_month_count"].values)
    china_mask = np.asarray(derived["china_mask"].values, dtype=bool)
    # A 0 means "tested in all 12 calendar months but never met the rule";
    # white means the 12-month comparison was not available. This avoids
    # confusing genuine zeros with missing/insufficient coverage.
    comparable = china_mask[None, :, :] & (valid_months == 12)
    inc = np.where(comparable, inc, np.nan)
    dec = np.where(comparable, dec, np.nan)
    return lon, lat, inc, dec


def plot_main_effect_figure(
    derived: xr.Dataset,
    boundaries: Dict[str, gpd.GeoDataFrame],
    out_prefix: Path,
) -> Dict[str, str]:
    """Figure 5: MEP95 q90 effect magnitude only; no significance overlay."""
    lon, lat, inc, dec = extract_effect_layers(derived)

    count_map, count_norm = effect_count_cmap()
    fig, axes = plt.subplots(
        2, 2,
        figsize=CFG.MAIN_FIGSIZE,
        dpi=CFG.DPI,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=False,
    )
    # Keep the complete panel labels as fixed strings so their wording,
    # sequence, baseline and left edge exactly follow the reference figure.
    panels = (
        (inc[1], "(a) La Niña 50% increase the probability"),
        (dec[1], "(b) La Niña 50% decrease the probability"),
        (inc[0], "(c) El Niño 50% increase the probability"),
        (dec[0], "(d) El Niño 50% decrease the probability"),
    )
    for ax, (data, title) in zip(np.asarray(axes).ravel(), panels):
        plot_grid(
            ax, lon, lat, data,
            count_map, count_norm, boundaries,
            title,
            "",
            "#222222",
            title_fontsize=CFG.MAIN_PANEL_TITLE_FONTSIZE,
            title_x=0.0,
            title_fontweight="normal",
        )

    fig.subplots_adjust(
        left=0.040, right=0.985, top=0.945, bottom=0.160,
        hspace=0.175, wspace=0.200,
    )
    add_effect_count_colorbar(
        fig, count_map, count_norm,
        ax_pos=(0.045, 0.055, 0.910, 0.028),
    )
    return save_figure(fig, out_prefix)


def plot_effect_sensitivity_figure(
    mep95_q95: xr.Dataset,
    rx5day_q90: xr.Dataset,
    boundaries: Dict[str, gpd.GeoDataFrame],
    out_prefix: Path,
) -> Dict[str, str]:
    """Figure S12: vertical 4 x 2 effect-magnitude sensitivity figure."""
    lon_q95, lat_q95, inc_q95, dec_q95 = extract_effect_layers(mep95_q95)
    lon_rx5, lat_rx5, inc_rx5, dec_rx5 = extract_effect_layers(rx5day_q90)
    cmap, norm = count_cmap()

    fig, axes = plt.subplots(
        4, 2,
        figsize=CFG.SENSITIVITY_FIGSIZE,
        dpi=CFG.DPI,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=False,
    )
    # Columns are metrics; rows are phase-direction combinations.  This keeps
    # matched MEP95 and Rx5day maps adjacent while producing a portrait figure.
    rows = (
        ((lon_q95, lat_q95, inc_q95[0]), (lon_rx5, lat_rx5, inc_rx5[0])),
        ((lon_q95, lat_q95, inc_q95[1]), (lon_rx5, lat_rx5, inc_rx5[1])),
        ((lon_q95, lat_q95, dec_q95[0]), (lon_rx5, lat_rx5, dec_rx5[0])),
        ((lon_q95, lat_q95, dec_q95[1]), (lon_rx5, lat_rx5, dec_rx5[1])),
    )
    panel_index = 0
    for row_index, data_row in enumerate(rows):
        for column_index, (lon, lat, data) in enumerate(data_row):
            plot_grid(
                axes[row_index, column_index],
                lon, lat, data,
                cmap, norm, boundaries,
                "",
                f"({chr(ord('a') + panel_index)})",
                "#222222",
                panel_label_inside=False,
            )
            panel_index += 1

    fig.subplots_adjust(
        left=0.040, right=0.985, top=0.992, bottom=0.105,
        hspace=0.085, wspace=0.180,
    )
    add_effect_count_colorbar(
        fig, cmap, norm,
        ax_pos=(0.055, 0.030, 0.890, 0.016),
    )
    return save_figure(fig, out_prefix)


def plot_fdr_diagnostic(
    derived: xr.Dataset,
    boundaries: Dict[str, gpd.GeoDataFrame],
    out_prefix: Path,
) -> Dict[str, str]:
    lon = np.asarray(derived["lon"].values)
    lat = np.asarray(derived["lat"].values)
    wilson = np.asarray(derived["wilson_significant_month_count"].values)
    fdr = np.asarray(derived["bh_fdr_wilson_score_month_count"].values)
    zero = np.asarray(derived["zero_event_month_count"].values)
    valid_months = np.asarray(derived["analysis_valid_month_count"].values)
    china_mask = np.asarray(derived["china_mask"].values, dtype=bool)
    comparable = china_mask[None, :, :] & (valid_months == 12)
    wilson = np.where(comparable, wilson, np.nan)
    fdr = np.where(comparable, fdr, np.nan)
    zero = np.where(comparable, zero, np.nan)
    cmap, norm = count_cmap()

    fig, axes = plt.subplots(
        3, 2,
        figsize=CFG.DIAG_FIGSIZE,
        dpi=CFG.DPI,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=False,
    )
    axes = np.asarray(axes)
    rows = (wilson, fdr, zero)
    panel_index = 0
    for row_index, data in enumerate(rows):
        for phase_index, phase in enumerate(PHASES):
            plot_grid(
                axes[row_index, phase_index],
                lon, lat, data[phase_index],
                cmap, norm, boundaries,
                "",
                f"({chr(ord('a') + panel_index)})",
                "#222222",
                annotate_all_zero=False,
                panel_label_inside=False,
            )
            panel_index += 1

    fig.subplots_adjust(
        left=0.040, right=0.985, top=0.988, bottom=0.155,
        hspace=0.110, wspace=0.200,
    )
    add_effect_count_colorbar(
        fig, cmap, norm,
        ax_pos=(0.045, 0.050, 0.910, 0.022),
    )
    return save_figure(fig, out_prefix)


def plot_quality_diagnostic(
    derived: xr.Dataset,
    boundaries: Dict[str, gpd.GeoDataFrame],
    out_prefix: Path,
) -> Dict[str, str]:
    """Sample coverage and percentile-threshold audit for Figure S14."""
    lon = np.asarray(derived["lon"].values)
    lat = np.asarray(derived["lat"].values)
    valid = np.asarray(derived["analysis_valid_month_count"].values)
    tail_excess = np.asarray(
        derived["upper_tail_frequency_excess_month_count"].values
    )
    paired = np.asarray(derived["paired_valid_month_count"].values)
    china_mask = np.asarray(derived["china_mask"].values, dtype=bool)

    valid = np.where(china_mask[None, :, :], valid, np.nan)
    tail_excess = np.where(china_mask, tail_excess, np.nan)
    paired = np.where(china_mask, paired, np.nan)
    cmap, norm = diagnostic_cmap()

    fig, axes = plt.subplots(
        2, 2,
        figsize=CFG.QUALITY_FIGSIZE,
        dpi=CFG.DPI,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=False,
    )
    panels = (
        (valid[0], "El Niño: valid-analysis months"),
        (valid[1], "La Niña: valid-analysis months"),
        (tail_excess, "Upper-tail frequency-excess months"),
        (paired, "Paired months for asymmetry"),
    )
    for panel_index, (ax, (data, title)) in enumerate(
        zip(np.asarray(axes).ravel(), panels)
    ):
        phase_color = (
            "#d7191c" if panel_index == 0
            else "#184db5" if panel_index == 1
            else "#222222"
        )
        plot_grid(
            ax, lon, lat, data, cmap, norm, boundaries,
            title,
            f"({chr(ord('a') + panel_index)})",
            phase_color,
            title_fontsize=CFG.DIAG_PANEL_TITLE_FONTSIZE,
        )

    fig.subplots_adjust(
        left=0.055, right=0.965, top=0.965, bottom=0.125,
        hspace=0.145, wspace=0.055,
    )
    add_count_colorbar(
        fig, cmap, norm,
        "Number of calendar months",
        ax_pos=(0.205, 0.055, 0.590, 0.018),
    )
    return save_figure(fig, out_prefix)


def plot_screening_cascade(
    diagnostics: pd.DataFrame,
    out_prefix: Path,
) -> Dict[str, str]:
    """Show where candidate grid-months are lost in the strict intersection."""
    panels = (
        (
            "El Niño", "increase",
            "estimated_enhanced_effect_cells",
            "wilson_increase_direction_cells",
            "wilson_supported_enhanced_cells",
        ),
        (
            "La Niña", "increase",
            "estimated_enhanced_effect_cells",
            "wilson_increase_direction_cells",
            "wilson_supported_enhanced_cells",
        ),
        (
            "El Niño", "decrease",
            "estimated_reduced_effect_cells",
            "wilson_decrease_direction_cells",
            "wilson_supported_reduced_cells",
        ),
        (
            "La Niña", "decrease",
            "estimated_reduced_effect_cells",
            "wilson_decrease_direction_cells",
            "wilson_supported_reduced_cells",
        ),
    )
    categories = (
        "50% effect\nestimate",
        "Wilson, same\ndirection",
        "Both rules",
    )
    colors = ("#9e9e9e", "#d8a32d", "#6a51a3")

    panel_values: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
    for phase, direction, effect_col, wilson_col, both_col in panels:
        subset = diagnostics.loc[diagnostics["phase"] == phase]
        denominator = float(subset["finite_nonzero_tested_cells"].sum())
        counts = np.array([
            subset[effect_col].sum(),
            subset[wilson_col].sum(),
            subset[both_col].sum(),
        ], dtype=float)
        percentages = np.divide(
            100.0 * counts,
            denominator,
            out=np.zeros_like(counts),
            where=denominator > 0,
        )
        panel_values.append((phase, direction, counts, percentages))

    global_max = max(
        float(np.max(values)) for _, _, _, values in panel_values
    )
    y_max = max(1.0, global_max * 1.22)

    fig, axes = plt.subplots(
        2, 2,
        figsize=CFG.SCREENING_FIGSIZE,
        dpi=CFG.DPI,
        sharey=True,
    )
    for panel_index, (ax, panel) in enumerate(
        zip(np.asarray(axes).ravel(), panel_values)
    ):
        phase, direction, counts, percentages = panel
        x = np.arange(3)
        bars = ax.bar(
            x, percentages,
            width=0.68,
            color=colors,
            edgecolor="#333333",
            linewidth=0.55,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", color="#d7d7d7", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=11.0)
        phase_color = "#d7191c" if phase == "El Niño" else "#184db5"
        ax.set_title(
            f"{phase}: {direction}",
            fontsize=CFG.DIAG_PANEL_TITLE_FONTSIZE,
            fontweight="bold",
            color=phase_color,
            pad=4.0,
        )
        ax.text(
            0.015, 0.985,
            f"({chr(ord('a') + panel_index)})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=CFG.PANEL_LABEL_FONTSIZE,
            fontweight="bold",
        )
        for bar, count, percentage in zip(bars, counts, percentages):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.025 * y_max,
                f"{percentage:.2f}%\n(n={int(count):,})",
                ha="center", va="bottom",
                fontsize=9.2,
            )

    axes[0, 0].set_ylabel(
        "Share of finite non-zero grid-month estimates (%)",
        fontsize=CFG.BASIN_AXIS_FONTSIZE,
        fontweight="bold",
    )
    axes[1, 0].set_ylabel(
        "Share of finite non-zero grid-month estimates (%)",
        fontsize=CFG.BASIN_AXIS_FONTSIZE,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.105, right=0.985, top=0.950, bottom=0.105,
        hspace=0.30, wspace=0.12,
    )
    return save_figure(fig, out_prefix)


# =============================================================================
# ONE-DATASET AND MAIN WORKFLOWS
# =============================================================================

def write_dataset(ds: xr.Dataset, path: Path) -> Path:
    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name in ds.data_vars
    }
    try:
        ds.to_netcdf(path, mode="w", engine="h5netcdf", encoding=encoding)
    except Exception:
        ds.to_netcdf(path, mode="w", encoding=encoding)
    return path


def analyze_one(
    nc_path: Path,
    metric: str,
    q: int,
    role: str,
    boundaries: Dict[str, gpd.GeoDataFrame],
) -> Tuple[Dict[str, object], xr.Dataset, pd.DataFrame]:
    """Calculate one dataset and save reproducibility tables without plotting."""
    resolved = resolve_nc(nc_path, metric, q)
    print("\n" + "=" * 96)
    print(f"{metric} q{q} ({role})")
    print(f"Input: {resolved}")
    print("=" * 96)

    with xr.open_dataset(resolved) as source:
        ds = source.load()

    lon = np.asarray(ds["lon"].values)
    lat = np.asarray(ds["lat"].values)
    china_mask = make_china_mask(lon, lat, boundaries["china_polygon"])

    derived, diagnostics = calculate_hotspots(ds, q, china_mask)

    print(
        "Screening summary: "
        f"reference-Wilson(valid family)="
        f"{int(diagnostics['wilson_significant_valid_cells'].sum())}, "
        f"raw Wilson-score p<0.05="
        f"{int(diagnostics['raw_score_p_lt_alpha_cells'].sum())}, "
        f"BH-FDR="
        f"{int(diagnostics['bh_fdr_significant_valid_cells'].sum())}, "
        f"BH-FDR zero-event="
        f"{int(diagnostics['bh_fdr_significant_zero_event_cells'].sum())}, "
        f"estimated increase effects="
        f"{int(diagnostics['estimated_enhanced_effect_cells'].sum())}, "
        f"estimated decrease effects="
        f"{int(diagnostics['estimated_reduced_effect_cells'].sum())}, "
        f"Wilson-supported increase effects="
        f"{int(diagnostics['wilson_supported_enhanced_cells'].sum())}, "
        f"Wilson-supported decrease effects="
        f"{int(diagnostics['wilson_supported_reduced_cells'].sum())}, "
        f"zero-event={int(diagnostics['zero_event_cells'].sum())}, "
        f"stored-Wilson mismatches="
        f"{int(diagnostics['stored_wilson_mismatch_cells'].sum())}"
    )
    empty_fdr_maps = diagnostics.loc[
        diagnostics["bh_fdr_significant_valid_cells"] == 0,
        ["phase", "month_label", "bh_family_valid_cells",
         "minimum_raw_score_pvalue", "minimum_BH_qvalue"],
    ]
    if not empty_fdr_maps.empty:
        print(
            "BH-FDR-adjusted Wilson score had zero valid-cell rejections in "
            f"{len(empty_fdr_maps)}/24 month-phase maps. "
            "See the diagnostics CSV for minimum score p and BH q values."
        )

    nc_dir, _, table_dir = output_dirs()
    stem = f"{metric}_q{q}"
    mode_tag = "EffectMagnitude_and_BHFDRdiagnostics"
    derived_path = write_dataset(
        derived,
        nc_dir / f"{stem}_{mode_tag}.nc",
    )
    diagnostics_path = table_dir / f"{stem}_monthly_FDR_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")

    record = {
        "role": role,
        "metric": metric,
        "threshold": q,
        "source_nc": str(resolved),
        "derived_nc": str(derived_path),
        "diagnostics_csv": str(diagnostics_path),
    }
    return record, derived, diagnostics


def main() -> None:
    set_style()
    _, fig_dir, table_dir = output_dirs()
    figure_set_dir = fig_dir / "four_figure_design_basin_4C5A6A"
    figure_set_dir.mkdir(parents=True, exist_ok=True)
    boundaries = load_boundaries()

    specifications = (
        ("main", CFG.NC_MEP95_Q90, "MEP95", 90, "main effect magnitude"),
        ("q95", CFG.NC_MEP95_Q95, "MEP95", 95, "threshold sensitivity"),
        ("rx5", CFG.NC_RX5DAY_Q90, "Rx5day", 90, "metric sensitivity"),
    )

    analysis_records: List[Dict[str, object]] = []
    derived_by_key: Dict[str, xr.Dataset] = {}
    for key, path, metric, q, role in specifications:
        record, derived, _diagnostics = analyze_one(
            path, metric, q, role, boundaries
        )
        analysis_records.append(record)
        derived_by_key[key] = derived

    # Exactly four scientific figure compositions are produced below.
    main_outputs = plot_main_effect_figure(
        derived_by_key["main"],
        boundaries,
        figure_set_dir / "Figure5_MEP95_q90_effect_magnitude_only",
    )
    sensitivity_outputs = plot_effect_sensitivity_figure(
        derived_by_key["q95"],
        derived_by_key["rx5"],
        boundaries,
        figure_set_dir / "FigureS12_effect_sensitivity_MEP95_q95_and_Rx5day_q90",
    )
    mep95_fdr_outputs = plot_fdr_diagnostic(
        derived_by_key["main"],
        boundaries,
        figure_set_dir / "FigureS11_MEP95_q90_Wilson_BHFDR_zero_event_diagnostic",
    )
    rx5day_fdr_outputs = plot_fdr_diagnostic(
        derived_by_key["rx5"],
        boundaries,
        figure_set_dir / "FigureS13_Rx5day_q90_Wilson_BHFDR_zero_event_diagnostic",
    )

    analysis_report_path = table_dir / "Figure5_three_dataset_analysis_report.csv"
    pd.DataFrame(analysis_records).to_csv(
        analysis_report_path, index=False, encoding="utf-8-sig"
    )
    figure_records = (
        {
            "figure": "Figure5",
            "scientific_role": "main MEP95 q90 effect magnitude",
            **main_outputs,
        },
        {
            "figure": "FigureS12",
            "scientific_role": "MEP95 q95 and Rx5day q90 effect sensitivity",
            **sensitivity_outputs,
        },
        {
            "figure": "FigureS11",
            "scientific_role": "MEP95 q90 Wilson, BH-FDR, and zero-event diagnostics",
            **mep95_fdr_outputs,
        },
        {
            "figure": "FigureS13",
            "scientific_role": "Rx5day q90 Wilson, BH-FDR, and zero-event diagnostics",
            **rx5day_fdr_outputs,
        },
    )
    figure_report_path = table_dir / "Figure5_exactly_four_figures_report.csv"
    pd.DataFrame(figure_records).to_csv(
        figure_report_path, index=False, encoding="utf-8-sig"
    )

    print("\nFinished exactly four Figure 5-series compositions:")
    print("  1. Figure5: MEP95 q90 effect magnitude")
    print("  2. FigureS12: MEP95 q95 + Rx5day q90 effect sensitivity")
    print("  3. FigureS11: Wilson + BH-FDR + zero-event diagnostic")
    print("  4. FigureS13: Rx5day Wilson + BH-FDR + zero-event diagnostic")
    print(f"Figure report: {figure_report_path}")


if __name__ == "__main__":
    main()