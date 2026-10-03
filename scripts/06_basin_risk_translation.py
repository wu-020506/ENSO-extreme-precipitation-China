#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from scipy import stats
from shapely.geometry import box

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
    # -------------------------------------------------------------------------
    # 1. Existing Figure 4 calculation folder and monthly NetCDF files
    # -------------------------------------------------------------------------
    CALC_OUT_DIR = OUTPUT_ROOT / "04_spatial_rr"
    CALC_NC_DIR = CALC_OUT_DIR / "00_netcdf"

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
    GRID_METRIC_NC = (
        CALC_NC_DIR / "Figure4_grid_monthly_MEP95_RX5day_anomaly_CN.nc"
    )
    AUTO_FIND_NC_IF_PATH_MISSING = True

    # -------------------------------------------------------------------------
    # 2. Vector files used in the existing Figure 4/Figure 5 workflow
    # -------------------------------------------------------------------------
    SHP_GLOBAL = DATA_ROOT / "boundaries" / "global_countries.shp"
    SHP_PROVINCE = DATA_ROOT / "boundaries" / "province_boundaries.shp"
    SHP_TEN_DASH = DATA_ROOT / "boundaries" / "ten_dash_line.shp"
    BASIN_POLYGON_FILE = DATA_ROOT / "basins" / "six_selected_official_basins.shp"

    # Unified publication-output collection.  Calculation inputs keep their
    # original paths; only newly generated Figure 6 products are redirected.
    COLLECTION_ROOT = CALC_OUT_DIR / "05_CEE_Fig4_Fig5_Fig6_reformatted"
    OUT_DIR = COLLECTION_ROOT / "03_Figure6"

    # -------------------------------------------------------------------------
    # 3. Statistical definitions locked to the final Figure 5 workflow
    # -------------------------------------------------------------------------
    ALPHA = 0.05
    MIN_PHASE_N = 10
    MIN_NEUTRAL_N = 10
    RR_EFFECT_HIGH = 1.50
    RR_EFFECT_LOW = 0.50

    # A recurrent basin hotspot means a boundary-weighted grid cell meets the
    # corresponding rule in at least this many of the 12 calendar months.
    RECURRENT_MIN_MONTHS = 2
    REQUIRE_ALL_MONTHS_FOR_RECURRENCE_DENOMINATOR = True

    # Basin-scale Figure S15 inference. One observation is one calendar year.
    # The phase coefficient is adjusted for a linear year trend, and its
    # uncertainty uses Newey-West HAC standard errors.
    BASIN_FDR_ALPHA = 0.05
    BASIN_CONFIDENCE_LEVEL = 0.95
    # None uses floor[4(n/100)^(2/9)], a conventional data-dependent
    # Newey-West maximum lag. Set an integer >= 0 only for a pre-specified
    # alternative analysis.
    BASIN_HAC_MAX_LAG = None
    # Absolute ENSO-minus-Neutral difference in annual basin event area,
    # expressed in percentage points. Figure S15 uses a deliberately focused
    # 0--20 percentage-point display range. Values above 20 are NOT truncated
    # in the calculations, annotations, or CSV files; they use the colourbar's
    # explicit over-range colour instead.
    BASIN_AREA_DIFFERENCE_BOUNDS = (
        0.0, 0.5, 1.0, 2.0, 3.0, 5.0,
        7.5, 10.0, 12.5, 15.0, 17.5, 20.0,
    )

    # Equal-area projection used for fractional cell-basin intersections.
    EQUAL_AREA_CRS = "EPSG:6933"

    # Figure/export settings.
    DPI = 500
    FONT_FAMILY = "Times New Roman"
    EXPORT_PNG = True
    EXPORT_TIF = True
    EXPORT_PDF = True
    MAIN_FIGSIZE = (14.5, 15.2)
    SUPPORT_FIGSIZE = (12.0, 9.7)
    SENSITIVITY_FIGSIZE = (11.0, 12.0)
    # Match the Figure 5 typography: Times New Roman, left-aligned regular
    # panel titles, and enlarged common explanatory text.
    PANEL_TITLE_SIZE = 17.5
    # Shared with Figures 4 and 5: outside-frame, upper-left, bold panel label.
    PANEL_LABEL_SIZE = 16.0
    AXIS_LABEL_SIZE = 16.0
    # Enlarged from the locked v16 design. This controls every heatmap x/y
    # tick label while preserving the v16 titles, panel labels and layout.
    TICK_SIZE = 18.0
    LEGEND_SIZE = 15.5
    # Enlarged colourbar title and numerical tick labels.
    COLORBAR_LABEL_SIZE = 20.0
    COLORBAR_TICK_SIZE = 18.0
    # Numeric values printed inside the S14/S15 heatmap cells.
    HEATMAP_VALUE_SIZE = 10.5

    # Bar-panel typography and geometry. These settings apply consistently to
    # Figure 6e and the Figure S16 sensitivity bars.
    BAR_TITLE_SIZE = 18.0
    BAR_AXIS_LABEL_SIZE = 17.0
    # Enlarged by 2 pt from v16 for all 0--100% and basin tick labels.
    BAR_TICK_SIZE = 18.0
    BAR_LEGEND_SIZE = 15.5
    BAR_HEIGHT = 0.19
    BAR_XLIM = (0.0, 100.0)
    BAR_XTICKS = tuple(np.arange(0.0, 101.0, 20.0))
    BAR_FRAME_WIDTH = 0.95

    # Compact CEE multi-panel spacing. Figure 6 contains basin-month matrices
    # rather than geographic maps, so its axes retain basin/month coordinates
    # and a full rectangular frame instead of longitude/latitude labels.
    LAYOUT_W_PAD = 0.020
    LAYOUT_H_PAD = 0.035
    LAYOUT_WSPACE = 0.018
    LAYOUT_HSPACE = 0.045

    # Four solid colours for all recurrent-area bars. The requested first
    # colour '#82D8C' is not a valid 6-digit hexadecimal code and is therefore
    # completed as '#82D8CC'. No hatching or transparency is used.
    LA_NINA_INCREASE_COLOR = "#82D8CC"
    LA_NINA_DECREASE_COLOR = "#D6F0ED"
    EL_NINO_DECREASE_COLOR = "#FDC7C7"
    EL_NINO_INCREASE_COLOR = "#FD9C99"

    # Heatmaps use the same fixed range to permit direct panel comparison.
    AREA_BOUNDS = tuple(np.arange(0.0, 110.0, 10.0))
    # Wilson-retention heatmaps need finer resolution near zero. This scale is
    # fixed a priori for Figure S14 and does not alter the screening rule.
    SUPPORT_RETENTION_BOUNDS = (
        0.0, 0.1, 0.5, 1.0, 2.0, 5.0,
        10.0, 20.0, 40.0, 60.0, 80.0, 100.0,
    )
    TOP_CASE_LABELS = 8


PHASES = ("El Nino", "La Nina")
PHASE_LABEL = {"El Nino": "El Niño", "La Nina": "La Niña"}
PHASE_KEY = {"El Nino": "EN", "La Nina": "LN"}
MONTH_LABEL = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
BASIN_ORDER = ("Songhua", "Haihe", "Yellow", "Huaihe", "Yangtze", "Pearl")
BASIN_NAME_CANDIDATES = (
    "basin", "BASIN", "basin_en", "BASIN_EN", "basin_name", "BASIN_NAME",
    "name", "NAME", "Name", "NAME_EN", "name_en", "NAME_ZH", "name_zh",
    "流域", "流域名", "名称",
)
MONTHS = tuple(range(1, 13))
MONTH_NAMES = tuple(MONTH_LABEL[m] for m in MONTHS)
MONTH_TICK_LABELS = ("J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D")
SCRIPT_VERSION = "Figure6_CEE_v16_large_all_ticks_colorbar_labels"


# =============================================================================
# PATH, VECTOR, MASK, AND STATISTICAL HELPERS
# =============================================================================

def resolve_nc(preferred: Path, metric: str, q: int) -> Path:
    """Resolve one monthly NetCDF without silently selecting a different metric."""
    if preferred.exists():
        return preferred
    if not CFG.AUTO_FIND_NC_IF_PATH_MISSING:
        raise FileNotFoundError(f"Required NetCDF does not exist: {preferred}")
    patterns = (
        f"*{metric}*q{q}*spatial_RR*Wilson*12months*.nc",
        f"*{metric}*q{q}*RR*Wilson*.nc",
        f"*{metric}*q{q}*.nc",
    )
    matches: List[Path] = []
    if CFG.CALC_NC_DIR.exists():
        for pattern in patterns:
            matches.extend(sorted(CFG.CALC_NC_DIR.glob(pattern)))
    matches = list(dict.fromkeys(path for path in matches if path.is_file()))
    if not matches:
        raise FileNotFoundError(
            f"Cannot find the {metric} q{q} monthly RR/count NetCDF.\n"
            f"Configured path: {preferred}\n"
            f"Search folder: {CFG.CALC_NC_DIR}\n"
            "Update CFG.CALC_OUT_DIR or the corresponding CFG.NC_* path."
        )
    if len(matches) > 1:
        print(f"[Path audit] Multiple {metric} q{q} candidates were found:")
        for candidate in matches:
            print(f"  - {candidate}")
        print(f"[Path audit] Using the first ranked match: {matches[0]}")
    else:
        print(f"[Path audit] Configured NetCDF missing; found: {matches[0]}")
    return matches[0]


def resolve_grid_metric_nc() -> Path:
    """Resolve the year-resolved Stage-0 metric file required by Figure S15."""
    if CFG.GRID_METRIC_NC.exists():
        return CFG.GRID_METRIC_NC
    if not CFG.AUTO_FIND_NC_IF_PATH_MISSING:
        raise FileNotFoundError(
            f"Year-resolved metric NetCDF does not exist: {CFG.GRID_METRIC_NC}"
        )
    patterns = (
        "*grid_monthly*MEP95*RX5day*anomaly*.nc",
        "*grid_monthly*MEP95*Rx5day*anomaly*.nc",
        "*monthly*MEP95*anomaly*.nc",
    )
    matches: List[Path] = []
    if CFG.CALC_NC_DIR.exists():
        for pattern in patterns:
            matches.extend(sorted(CFG.CALC_NC_DIR.glob(pattern)))
    matches = list(dict.fromkeys(path for path in matches if path.is_file()))
    if not matches:
        raise FileNotFoundError(
            "Figure S15 requires the year-resolved Figure 4 Stage-0 NetCDF.\n"
            f"Configured path: {CFG.GRID_METRIC_NC}\n"
            "Expected file: Figure4_grid_monthly_MEP95_RX5day_anomaly_CN.nc\n"
            "Run the Figure 4 calculation script once with its monthly metric "
            "NetCDF output enabled. Counts-only RR files cannot be used for "
            "trend-adjusted basin-year HAC inference."
        )
    if len(matches) > 1:
        print("[Path audit] Multiple year-resolved metric files were found:")
        for candidate in matches:
            print(f"  - {candidate}")
        print(f"[Path audit] Using the first ranked match: {matches[0]}")
    return matches[0]


def audit_input_paths() -> Tuple[Dict[Tuple[str, int], Path], Path]:
    """Check every required input before any expensive calculation starts."""
    print("=" * 88)
    print("Figure 6 input-path audit")
    print("=" * 88)
    resolved = {
        ("MEP95", 90): resolve_nc(CFG.NC_MEP95_Q90, "MEP95", 90),
        ("MEP95", 95): resolve_nc(CFG.NC_MEP95_Q95, "MEP95", 95),
        ("Rx5day", 90): resolve_nc(CFG.NC_RX5DAY_Q90, "Rx5day", 90),
    }
    grid_metric_nc = resolve_grid_metric_nc()
    critical_vectors = {
        "Global country polygon": CFG.SHP_GLOBAL,
        "Six-basin polygon": CFG.BASIN_POLYGON_FILE,
    }
    optional_vectors = {
        "Province boundary (optional for Figure 6)": CFG.SHP_PROVINCE,
        "Ten-dash line (optional for Figure 6)": CFG.SHP_TEN_DASH,
    }
    missing_critical = []
    for label, path in critical_vectors.items():
        ok = path.exists()
        print(f"[{'OK' if ok else 'MISSING'}] {label}: {path}")
        if not ok:
            missing_critical.append((label, path))
    for label, path in optional_vectors.items():
        print(f"[{'OK' if path.exists() else 'OPTIONAL MISSING'}] {label}: {path}")
    for (metric, q), path in resolved.items():
        print(f"[OK] {metric} q{q} NetCDF: {path}")
    print(f"[OK] Year-resolved metric NetCDF: {grid_metric_nc}")
    print(f"[OUTPUT] Figure 6 folder: {CFG.OUT_DIR}")
    if missing_critical:
        detail = "\n".join(f"  {label}: {path}" for label, path in missing_critical)
        raise FileNotFoundError(
            "The following critical vector inputs are missing:\n"
            f"{detail}\nUpdate the corresponding path(s) in CFG before rerunning."
        )
    return resolved, grid_metric_nc


def read_vector(path: Path, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Vector file does not exist: {path}")
    last_error: Optional[Exception] = None
    for encoding in ("gb18030", "gbk", "cp936", "utf-8"):
        try:
            kwargs: Dict[str, object] = {"encoding": encoding}
            if layer is not None:
                kwargs["layer"] = layer
            gdf = gpd.read_file(path, **kwargs)
            if gdf.empty:
                raise ValueError(f"Vector file contains no features: {path}")
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            else:
                gdf = gdf.to_crs("EPSG:4326")
            return gdf
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read vector file {path}; last error: {last_error}")


def union_geometry(gdf: gpd.GeoDataFrame):
    try:
        return gdf.geometry.union_all()
    except Exception:
        return gdf.geometry.unary_union


def load_china_boundary() -> gpd.GeoDataFrame:
    world = read_vector(CFG.SHP_GLOBAL)
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
            selected = world.loc[mask, ["geometry"]].copy()
            return gpd.GeoDataFrame(
                {"geometry": [union_geometry(selected)]},
                crs="EPSG:4326",
            )
    available = ", ".join(map(str, world.columns))
    raise ValueError(
        "China could not be identified in CFG.SHP_GLOBAL. Available fields: "
        f"{available}"
    )


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
    basins = read_vector(CFG.BASIN_POLYGON_FILE)
    name_column = find_basin_name_column(basins)
    if name_column is None:
        raise ValueError(
            "No basin-name field was identified. Add the correct field name "
            "to BASIN_NAME_CANDIDATES."
        )
    basins["basin_name"] = basins[name_column].map(normalize_basin_name)
    basins = basins[basins["basin_name"].isin(BASIN_ORDER)].copy()
    china_geom = union_geometry(china)
    basins["geometry"] = basins.geometry.intersection(china_geom)
    basins = basins[basins.geometry.notnull() & (~basins.geometry.is_empty)].copy()
    basins = basins.dissolve(by="basin_name", as_index=False)
    missing = [name for name in BASIN_ORDER if name not in set(basins["basin_name"])]
    if missing:
        raise ValueError(
            "The following basins were not matched from the basin vector: "
            f"{missing}. Check the basin-name field and normalization rules."
        )
    order = {name: index for index, name in enumerate(BASIN_ORDER)}
    basins["_order"] = basins["basin_name"].map(order)
    basins = basins.sort_values("_order").drop(columns="_order")
    return basins[["basin_name", "geometry"]].to_crs("EPSG:4326")


def load_boundaries() -> Dict[str, gpd.GeoDataFrame]:
    china = load_china_boundary()
    basins = load_six_basins(china)
    return {"china_polygon": china, "basin_polygons": basins}


def geometry_mask(lon: np.ndarray, lat: np.ndarray, geometry) -> np.ndarray:
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


def get_required(
    ds: xr.Dataset,
    name: str,
    alternatives: Iterable[str] = (),
) -> np.ndarray:
    for candidate in (name, *alternatives):
        if candidate in ds:
            return np.asarray(ds[candidate].values)
    tried = ", ".join((name, *alternatives))
    raise KeyError(f"Required NetCDF variable not found. Tried: {tried}")


def audit_netcdf_structure(
    path: Path,
    metric: str,
    q: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate grid coordinates and every monthly count variable before plotting."""
    with xr.open_dataset(path) as ds:
        if "lon" not in ds.coords and "lon" not in ds.variables:
            raise KeyError(f"{metric} q{q} NetCDF has no 'lon' coordinate: {path}")
        if "lat" not in ds.coords and "lat" not in ds.variables:
            raise KeyError(f"{metric} q{q} NetCDF has no 'lat' coordinate: {path}")
        lon = np.asarray(ds["lon"].values)
        lat = np.asarray(ds["lat"].values)
        if lon.ndim != 1 or lat.ndim != 1:
            raise ValueError(
                f"{metric} q{q} requires one-dimensional lon/lat coordinates; "
                f"received lon{lon.shape}, lat{lat.shape}."
            )
        expected_shape = (lat.size, lon.size)
        missing: List[str] = []
        wrong_shape: List[str] = []
        for month in MONTHS:
            for phase in PHASES:
                base = f"q{q}_{MONTH_LABEL[month]}_{PHASE_KEY[phase]}"
                for suffix in ("c_phase", "n_phase", "c_neutral", "n_neutral"):
                    name = f"{base}_{suffix}"
                    if name not in ds:
                        missing.append(name)
                    elif tuple(ds[name].shape) != expected_shape:
                        wrong_shape.append(f"{name}: {tuple(ds[name].shape)}")
        if missing:
            examples = "\n  ".join(missing[:12])
            raise KeyError(
                f"{metric} q{q} NetCDF is missing {len(missing)} required count "
                f"variables. First entries:\n  {examples}"
            )
        if wrong_shape:
            examples = "\n  ".join(wrong_shape[:12])
            raise ValueError(
                f"{metric} q{q} contains variables not shaped as {expected_shape}:\n"
                f"  {examples}"
            )
    print(
        f"[Schema audit] {metric} q{q}: 96 monthly count arrays verified; "
        f"grid={lat.size} x {lon.size}."
    )
    return lon, lat


def load_year_resolved_anomaly(
    path: Path,
    metric: str,
    expected_lon: np.ndarray,
    expected_lat: np.ndarray,
) -> xr.DataArray:
    """Load and validate one time x lat x lon anomaly cube for resampling."""
    variable = f"{metric}_anomaly"
    with xr.open_dataset(path) as source:
        if variable not in source:
            raise KeyError(
                f"Year-resolved metric file has no {variable!r}: {path}"
            )
        da = source[variable]
        required_dims = {"time", "lat", "lon"}
        if not required_dims.issubset(set(da.dims)):
            raise ValueError(
                f"{variable} must contain time/lat/lon dimensions; got {da.dims}."
            )
        da = da.transpose("time", "lat", "lon")
        lon = np.asarray(da["lon"].values)
        lat = np.asarray(da["lat"].values)
        same_grid = (
            lon.shape == expected_lon.shape
            and lat.shape == expected_lat.shape
            and np.allclose(lon, expected_lon, rtol=0.0, atol=1.0e-8)
            and np.allclose(lat, expected_lat, rtol=0.0, atol=1.0e-8)
        )
        if not same_grid:
            raise ValueError(
                "The year-resolved metric NetCDF grid differs from the Figure "
                "4 RR/count grid. Automatic regridding is disabled because it "
                "would alter basin-area weights."
            )

        time_index = pd.DatetimeIndex(pd.to_datetime(da["time"].values))
        cal_month = time_index.month.astype(np.int16)
        cal_year = time_index.year.astype(np.int16)

        phase_code = None
        for container in (da.coords, source.coords, source.data_vars):
            if "phase_code" in container:
                candidate = container["phase_code"]
                if "time" in candidate.dims:
                    phase_code = np.asarray(candidate.values, dtype=np.int16)
                    break
        if phase_code is None:
            raise KeyError(
                "The Stage-0 metric NetCDF has no time-resolved phase_code. "
                "Re-run the latest Figure 4 calculation script so ENSO phases "
                "are attached to Figure4_grid_monthly_MEP95_RX5day_anomaly_CN.nc."
            )
        if phase_code.shape[0] != da.sizes["time"]:
            raise ValueError(
                "phase_code length does not match the metric time dimension."
            )
        da = da.assign_coords(
            cal_month=("time", cal_month),
            cal_year=("time", cal_year),
            phase_code=("time", phase_code),
        ).load()

    valid_phase = np.isin(da["phase_code"].values, (-1, 0, 1))
    if not np.any(valid_phase):
        raise ValueError("No El Niño, La Niña, or Neutral months were found.")
    if not np.all(valid_phase):
        print(
            f"[Stage-0 audit] Dropping {int((~valid_phase).sum())} months with "
            "unknown phase codes."
        )
        da = da.isel(time=valid_phase)
    print(
        f"[Stage-0 audit] {variable}: {da.sizes['time']} monthly fields, "
        f"{da.sizes['lat']} x {da.sizes['lon']} grid."
    )
    return da


def wilson_interval(
    count: np.ndarray,
    sample_size: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    count = np.asarray(count, dtype=float)
    sample_size = np.asarray(sample_size, dtype=float)
    low = np.full(count.shape, np.nan, dtype=float)
    high = np.full(count.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(count)
        & np.isfinite(sample_size)
        & (sample_size > 0)
        & (count >= 0)
        & (count <= sample_size)
    )
    if not np.any(valid):
        return low, high
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    n = sample_size[valid]
    phat = count[valid] / n
    denominator = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(
        phat * (1.0 - phat) / n + z * z / (4.0 * n * n)
    ) / denominator
    low[valid] = np.clip(centre - half, 0.0, 1.0)
    high[valid] = np.clip(centre + half, 0.0, 1.0)
    return low, high


def reference_wilson_screen(
    c_phase: np.ndarray,
    n_phase: np.ndarray,
    c_neutral: np.ndarray,
    n_neutral: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the reference-paper Wilson screening rule only."""
    c_phase = np.asarray(c_phase, dtype=float)
    n_phase = np.asarray(n_phase, dtype=float)
    c_neutral = np.asarray(c_neutral, dtype=float)
    n_neutral = np.asarray(n_neutral, dtype=float)
    valid = (
        np.isfinite(c_phase)
        & np.isfinite(n_phase)
        & np.isfinite(c_neutral)
        & np.isfinite(n_neutral)
        & (n_phase >= CFG.MIN_PHASE_N)
        & (n_neutral >= CFG.MIN_NEUTRAL_N)
        & (c_phase >= 0)
        & (c_neutral >= 0)
        & (c_phase <= n_phase)
        & (c_neutral <= n_neutral)
    )
    p_phase = np.divide(
        c_phase,
        n_phase,
        out=np.full(c_phase.shape, np.nan, dtype=float),
        where=n_phase > 0,
    )
    p_neutral = np.divide(
        c_neutral,
        n_neutral,
        out=np.full(c_neutral.shape, np.nan, dtype=float),
        where=n_neutral > 0,
    )
    low, high = wilson_interval(c_phase, n_phase, alpha)
    significant = (
        valid
        & np.isfinite(low)
        & np.isfinite(high)
        & ((p_neutral < low) | (p_neutral > high))
    )
    return significant, p_phase, p_neutral, low, high


def fdr_bh_adjust(
    pvalues: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
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
    adjusted = ranked * ranked.size / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    unsorted = np.empty_like(adjusted)
    unsorted[order] = adjusted
    qvalues[valid] = unsorted
    rejected[valid] = unsorted <= alpha
    return qvalues, rejected


# =============================================================================
# STYLE AND OUTPUT HELPERS
# =============================================================================

def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [CFG.FONT_FAMILY, "DejaVu Serif"],
        "font.size": CFG.TICK_SIZE,
        "font.weight": "bold",
        "axes.titlesize": CFG.PANEL_TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.labelsize": CFG.AXIS_LABEL_SIZE,
        "axes.labelweight": "bold",
        "xtick.labelsize": CFG.TICK_SIZE,
        "ytick.labelsize": CFG.TICK_SIZE,
        "legend.fontsize": CFG.LEGEND_SIZE,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def apply_bold_axis_typography(ax: plt.Axes) -> None:
    """Bold titles, panel labels, axis labels, and all major tick labels."""
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick_label.set_fontweight("bold")


def compact_constrained_layout(fig: plt.Figure) -> None:
    """Reduce panel gaps while keeping labels and colour bars collision-free."""
    try:
        engine = fig.get_layout_engine()
        if engine is not None:
            engine.set(
                w_pad=CFG.LAYOUT_W_PAD,
                h_pad=CFG.LAYOUT_H_PAD,
                wspace=CFG.LAYOUT_WSPACE,
                hspace=CFG.LAYOUT_HSPACE,
            )
            return
    except Exception:
        pass
    try:
        fig.set_constrained_layout_pads(
            w_pad=CFG.LAYOUT_W_PAD,
            h_pad=CFG.LAYOUT_H_PAD,
            wspace=CFG.LAYOUT_WSPACE,
            hspace=CFG.LAYOUT_HSPACE,
        )
    except Exception:
        pass


def output_dirs() -> Tuple[Path, Path]:
    fig_dir = CFG.OUT_DIR / "01_figures"
    table_dir = CFG.OUT_DIR / "02_tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir


def save_figure(fig: plt.Figure, prefix: Path) -> Dict[str, str]:
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


# =============================================================================
# FRACTIONAL BASIN-AREA WEIGHTS
# =============================================================================

def coordinate_edges(values: np.ndarray, latitude: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Latitude and longitude must be one-dimensional arrays.")
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    if latitude:
        edges = np.clip(edges, -90.0, 90.0)
    return edges


def build_fractional_basin_area_weights(
    lon: np.ndarray,
    lat: np.ndarray,
    basins: gpd.GeoDataFrame,
) -> Dict[str, np.ndarray]:
    """Return cell-basin intersection areas (km2), including boundary fractions."""
    lon_edges = coordinate_edges(lon)
    lat_edges = coordinate_edges(lat, latitude=True)
    polygons = [
        box(
            min(lon_edges[j], lon_edges[j + 1]),
            min(lat_edges[i], lat_edges[i + 1]),
            max(lon_edges[j], lon_edges[j + 1]),
            max(lat_edges[i], lat_edges[i + 1]),
        )
        for i in range(len(lat))
        for j in range(len(lon))
    ]
    cells = gpd.GeoSeries(polygons, crs="EPSG:4326").to_crs(CFG.EQUAL_AREA_CRS)
    basin_ea = basins.to_crs(CFG.EQUAL_AREA_CRS).set_index("basin_name")

    weights: Dict[str, np.ndarray] = {}
    for basin in BASIN_ORDER:
        geom = basin_ea.loc[basin, "geometry"]
        if isinstance(geom, pd.Series):
            geom = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
        candidate = cells.intersects(geom)
        area = np.zeros(len(cells), dtype=float)
        if np.any(candidate):
            intersections = cells.loc[candidate].intersection(geom)
            area[np.asarray(candidate)] = np.asarray(intersections.area) / 1.0e6
        weights[basin] = area.reshape(len(lat), len(lon))
        if not np.any(weights[basin] > 0):
            raise ValueError(f"No grid-cell intersection was found for basin: {basin}")
    return weights


def weighted_fraction(
    condition: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float, float]:
    domain = valid & np.isfinite(weights) & (weights > 0)
    denominator = float(np.sum(weights[domain]))
    if denominator <= 0:
        return np.nan, np.nan, np.nan
    affected = float(np.sum(weights[domain & condition]))
    return 100.0 * affected / denominator, affected, denominator


def weighted_mean(values: np.ndarray, valid: np.ndarray, weights: np.ndarray) -> float:
    domain = valid & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    denominator = float(np.sum(weights[domain]))
    if denominator <= 0:
        return np.nan
    return float(np.sum(weights[domain] * values[domain]) / denominator)


# =============================================================================
# YEAR-RESOLVED BASIN INFERENCE FOR FIGURE S15
# =============================================================================

def build_yearly_basin_event_area(
    anomaly: xr.DataArray,
    metric: str,
    q: int,
    basin_weights: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Calculate the q-event-affected fraction of each basin for every year.

    Thresholds are calculated once from all available years for each grid cell
    and calendar month, exactly as in the Figure 4 calculation. The yearly
    basin statistic is an area-weighted percentage, not a count of grid cells.
    """
    rows: List[Dict[str, object]] = []
    for month in MONTHS:
        month_mask = np.asarray(anomaly["cal_month"].values) == month
        sub = anomaly.isel(time=month_mask)
        if sub.sizes.get("time", 0) == 0:
            continue
        values = np.asarray(sub.values, dtype=float)
        threshold = np.nanpercentile(values, q, axis=0)
        finite = np.isfinite(values) & np.isfinite(threshold)[None, :, :]
        event = finite & (values >= threshold[None, :, :])
        years = np.asarray(sub["cal_year"].values, dtype=int)
        phase_codes = np.asarray(sub["phase_code"].values, dtype=int)

        for basin in BASIN_ORDER:
            weights = np.asarray(basin_weights[basin], dtype=float)
            usable_weight = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
            denominator = np.sum(
                finite * usable_weight[None, :, :], axis=(1, 2)
            )
            numerator = np.sum(
                event * usable_weight[None, :, :], axis=(1, 2)
            )
            area_percent = np.divide(
                100.0 * numerator,
                denominator,
                out=np.full(numerator.shape, np.nan, dtype=float),
                where=denominator > 0,
            )
            for year, phase_code, area, valid_area in zip(
                years, phase_codes, area_percent, denominator
            ):
                rows.append({
                    "metric": metric,
                    "threshold_percentile": q,
                    "cal_year": int(year),
                    "month": int(month),
                    "month_label": MONTH_LABEL[month],
                    "phase_code": int(phase_code),
                    "basin": basin,
                    "event_area_percent": float(area),
                    "valid_area_km2": float(valid_area),
                })
    table = pd.DataFrame(rows)
    expected = len(BASIN_ORDER) * len(MONTHS)
    groups = table.groupby(["basin", "month"], observed=True).size()
    if groups.size != expected:
        print(
            f"[Yearly-area audit] Expected {expected} basin-month groups; "
            f"found {groups.size}. Missing groups will remain undefined."
        )
    return table


# =============================================================================
# TREND-ADJUSTED NEWEY-WEST HAC INFERENCE FOR FIGURE S15
# =============================================================================

def newey_west_hac_regression(
    response: np.ndarray,
    phase_indicator: np.ndarray,
    years: np.ndarray,
) -> Dict[str, float]:
    """Fit area ~ phase + linear year trend with Newey-West uncertainty.

    The phase coefficient is the trend-adjusted ENSO-minus-Neutral difference
    in basin event area (percentage points). Newey-West covariance permits
    heteroskedasticity and short-lag serial correlation in annual residuals.
    """
    y = np.asarray(response, dtype=float)
    phase = np.asarray(phase_indicator, dtype=float)
    year = np.asarray(years, dtype=float)
    valid = np.isfinite(y) & np.isfinite(phase) & np.isfinite(year)
    y, phase, year = y[valid], phase[valid], year[valid]
    order = np.argsort(year)
    y, phase, year = y[order], phase[order], year[order]
    n = y.size
    year_decades = (year - np.mean(year)) / 10.0
    design = np.column_stack((np.ones(n), phase, year_decades))
    k = design.shape[1]
    if n <= k + 2 or np.linalg.matrix_rank(design) < k:
        return {
            "coefficient": np.nan,
            "standard_error": np.nan,
            "t_statistic": np.nan,
            "p_two_sided": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "year_trend_per_decade": np.nan,
            "hac_max_lag": np.nan,
            "n_model": float(n),
        }

    xtx_inverse = np.linalg.pinv(design.T @ design)
    beta = xtx_inverse @ design.T @ y
    residual = y - design @ beta
    if CFG.BASIN_HAC_MAX_LAG is None:
        max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    else:
        max_lag = int(CFG.BASIN_HAC_MAX_LAG)
    max_lag = max(0, min(max_lag, n - 1))

    score = design * residual[:, None]
    meat = score.T @ score
    for lag in range(1, max_lag + 1):
        bartlett_weight = 1.0 - lag / (max_lag + 1.0)
        lag_covariance = score[lag:].T @ score[:-lag]
        meat += bartlett_weight * (lag_covariance + lag_covariance.T)
    # Finite-sample correction; asymptotically it converges to one.
    covariance = (n / (n - k)) * xtx_inverse @ meat @ xtx_inverse
    variance = float(covariance[1, 1])
    standard_error = np.sqrt(variance) if variance > 0 else np.nan
    degrees_freedom = n - k
    if np.isfinite(standard_error) and standard_error > 0:
        t_statistic = float(beta[1] / standard_error)
        pvalue = float(2.0 * stats.t.sf(abs(t_statistic), degrees_freedom))
        interval_alpha = 1.0 - CFG.BASIN_CONFIDENCE_LEVEL
        critical = float(stats.t.ppf(
            1.0 - interval_alpha / 2.0,
            degrees_freedom,
        ))
        ci_low = float(beta[1] - critical * standard_error)
        ci_high = float(beta[1] + critical * standard_error)
    else:
        t_statistic = pvalue = ci_low = ci_high = np.nan
    return {
        "coefficient": float(beta[1]),
        "standard_error": float(standard_error),
        "t_statistic": t_statistic,
        "p_two_sided": pvalue,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "year_trend_per_decade": float(beta[2]),
        "hac_max_lag": float(max_lag),
        "n_model": float(n),
    }


def trend_adjusted_hac_inference(yearly_area: pd.DataFrame) -> pd.DataFrame:
    """Trend-adjusted OLS inference with Newey-West HAC uncertainty."""
    phase_codes = {"El Niño": 1, "La Niña": -1}
    rows: List[Dict[str, object]] = []
    for phase, phase_code in phase_codes.items():
        for month in MONTHS:
            for basin in BASIN_ORDER:
                subset = yearly_area[
                    (yearly_area["month"] == month)
                    & (yearly_area["basin"] == basin)
                    & (yearly_area["phase_code"].isin((0, phase_code)))
                ].copy()
                subset = subset[
                    np.isfinite(subset["event_area_percent"])
                    & np.isfinite(subset["cal_year"])
                ].sort_values("cal_year")
                n_phase = int((subset["phase_code"] == phase_code).sum())
                n_neutral = int((subset["phase_code"] == 0).sum())
                phase_mean = subset.loc[
                    subset["phase_code"] == phase_code,
                    "event_area_percent",
                ].mean()
                neutral_mean = subset.loc[
                    subset["phase_code"] == 0,
                    "event_area_percent",
                ].mean()
                raw_difference = phase_mean - neutral_mean
                if (
                    n_phase < CFG.MIN_PHASE_N
                    or n_neutral < CFG.MIN_NEUTRAL_N
                ):
                    fit = {
                        "coefficient": np.nan,
                        "standard_error": np.nan,
                        "t_statistic": np.nan,
                        "p_two_sided": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "year_trend_per_decade": np.nan,
                        "hac_max_lag": np.nan,
                        "n_model": float(len(subset)),
                    }
                else:
                    fit = newey_west_hac_regression(
                        subset["event_area_percent"].to_numpy(dtype=float),
                        (subset["phase_code"] == phase_code).to_numpy(dtype=float),
                        subset["cal_year"].to_numpy(dtype=float),
                    )
                effect = float(fit["coefficient"])
                rows.append({
                    "metric": "MEP95",
                    "threshold_percentile": 90,
                    "phase": phase,
                    "phase_code": phase_code,
                    "month": month,
                    "month_label": MONTH_LABEL[month],
                    "basin": basin,
                    "n_phase_years": n_phase,
                    "n_neutral_years": n_neutral,
                    "phase_mean_event_area_percent": float(phase_mean),
                    "neutral_mean_event_area_percent": float(neutral_mean),
                    "raw_difference_percentage_points": float(raw_difference),
                    "difference_percentage_points": effect,
                    "absolute_difference_percentage_points": (
                        abs(effect) if np.isfinite(effect) else np.nan
                    ),
                    "hac_standard_error": fit["standard_error"],
                    "hac_t_statistic": fit["t_statistic"],
                    "hac_p_two_sided": fit["p_two_sided"],
                    "hac_ci_low": fit["ci_low"],
                    "hac_ci_high": fit["ci_high"],
                    "linear_year_trend_percentage_points_per_decade": (
                        fit["year_trend_per_decade"]
                    ),
                    "hac_max_lag": fit["hac_max_lag"],
                    "n_model": fit["n_model"],
                })

    inference = pd.DataFrame(rows)
    inference["bh_fdr_q"] = np.nan
    inference["bh_fdr_reject"] = False
    for phase in ("El Niño", "La Niña"):
        mask = (
            (inference["phase"] == phase)
            & np.isfinite(inference["hac_p_two_sided"])
        )
        qvalues, rejected = fdr_bh_adjust(
            inference.loc[mask, "hac_p_two_sided"].to_numpy(dtype=float),
            CFG.BASIN_FDR_ALPHA,
        )
        inference.loc[mask, "bh_fdr_q"] = qvalues
        inference.loc[mask, "bh_fdr_reject"] = rejected
    inference["hac_ci_excludes_zero"] = (
        (inference["hac_ci_low"] > 0)
        | (inference["hac_ci_high"] < 0)
    )
    inference["direction"] = np.where(
        inference["difference_percentage_points"] >= 0,
        "increase",
        "decrease",
    )
    inference["n_tests_in_bh_family"] = len(BASIN_ORDER) * len(MONTHS)
    inference["inference_method"] = (
        "OLS adjusted for linear year trend; Newey-West HAC standard errors; "
        "two-sided t test; BH-FDR within 72 basin-month tests per ENSO phase"
    )
    return inference


def build_hac_publication_table(inference: pd.DataFrame) -> pd.DataFrame:
    """Create the complete external table accompanying Figure S15."""
    table = inference.copy()
    phase_rank = {"El Niño": 0, "La Niña": 1}
    basin_rank = {name: index for index, name in enumerate(BASIN_ORDER)}
    table["_phase_rank"] = table["phase"].map(phase_rank)
    table["_basin_rank"] = table["basin"].map(basin_rank)
    table = table.sort_values(
        ["_phase_rank", "_basin_rank", "month"],
        kind="stable",
    )
    table["effect_95_ci_percentage_points"] = table.apply(
        lambda row: (
            f"{row['difference_percentage_points']:.2f} "
            f"[{row['hac_ci_low']:.2f}, {row['hac_ci_high']:.2f}]"
            if np.isfinite(row["difference_percentage_points"])
            and np.isfinite(row["hac_ci_low"])
            and np.isfinite(row["hac_ci_high"])
            else "NA"
        ),
        axis=1,
    )
    table["BH_FDR_supported_q_le_0.05"] = np.where(
        table["bh_fdr_reject"].fillna(False).astype(bool),
        "Yes",
        "No",
    )
    selected = table[[
        "phase",
        "basin",
        "month",
        "month_label",
        "direction",
        "n_phase_years",
        "n_neutral_years",
        "phase_mean_event_area_percent",
        "neutral_mean_event_area_percent",
        "raw_difference_percentage_points",
        "difference_percentage_points",
        "hac_standard_error",
        "hac_ci_low",
        "hac_ci_high",
        "effect_95_ci_percentage_points",
        "linear_year_trend_percentage_points_per_decade",
        "hac_t_statistic",
        "hac_p_two_sided",
        "bh_fdr_q",
        "BH_FDR_supported_q_le_0.05",
        "hac_max_lag",
        "n_model",
    ]].copy()
    # Preserve full precision for p and q values so very small probabilities
    # are written in scientific notation rather than rounded to 0.0000.
    probability_columns = {"hac_p_two_sided", "bh_fdr_q"}
    round_columns = [
        column
        for column in selected.select_dtypes(include=[np.number]).columns
        if column not in probability_columns
    ]
    selected[round_columns] = selected[round_columns].round(4)
    return selected


def save_hac_publication_table(
    inference: pd.DataFrame,
    table_dir: Path,
) -> Dict[str, str]:
    """Save the complete machine-readable Figure S15 inference table."""
    table = build_hac_publication_table(inference)
    csv_path = table_dir / "TableS1_FigureS15_trend_adjusted_HAC_inference.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    outputs = {"csv": str(csv_path)}

    notes = pd.DataFrame({
        "Item": [
            "Response",
            "Effect",
            "Model",
            "Uncertainty",
            "Confidence interval",
            "Multiple testing",
            "Interpretation",
            "Permutation testing",
        ],
        "Definition": [
            "Annual fraction of valid basin area affected by MEP95 q90 events",
            "ENSO-minus-Neutral phase coefficient, in percentage points",
            "Event area = intercept + ENSO phase + linear year trend",
            "Newey-West HAC standard errors with a data-dependent maximum lag",
            f"{100.0 * CFG.BASIN_CONFIDENCE_LEVEL:.0f}% two-sided HAC interval",
            "BH-FDR across 6 basins x 12 months (72 tests) separately by phase",
            "q <= 0.05 is treated as FDR-supported",
            "Not used in this Figure 6 workflow",
        ],
    })
    notes_path = table_dir / "TableS1_FigureS15_method_notes.csv"
    notes.to_csv(notes_path, index=False, encoding="utf-8-sig")
    outputs["notes_csv"] = str(notes_path)
    return outputs


def build_manuscript_recurrent_table(
    recurrent: pd.DataFrame,
) -> pd.DataFrame:
    """Six-row main-text table corresponding exactly to Figure 6e."""
    subset = recurrent[
        (recurrent["metric"] == "MEP95")
        & (recurrent["threshold_percentile"] == 90)
        & (recurrent["support"] == "effect")
    ].copy()
    rows: List[Dict[str, object]] = []
    for basin in BASIN_ORDER:
        row: Dict[str, object] = {"River basin": basin}
        for phase in ("El Niño", "La Niña"):
            record = subset[
                (subset["basin"] == basin) & (subset["phase"] == phase)
            ]
            if record.empty:
                increase = decrease = np.nan
            else:
                increase = float(record.iloc[0]["increase_recurrent_area_percent"])
                decrease = float(record.iloc[0]["decrease_recurrent_area_percent"])
            row[f"{phase}: increase (%)"] = increase
            row[f"{phase}: decrease (%)"] = decrease
        rows.append(row)
    table = pd.DataFrame(rows)
    numeric = table.select_dtypes(include=[np.number]).columns
    table[numeric] = table[numeric].round(1)
    return table


def _formatted_peak_effect(
    subset: pd.DataFrame,
    direction: str,
) -> str:
    """Format the largest positive/negative effect and its calendar month."""
    valid = subset[np.isfinite(subset["difference_percentage_points"])].copy()
    if direction == "increase":
        valid = valid[valid["difference_percentage_points"] > 0]
        if valid.empty:
            return "—"
        record = valid.loc[valid["difference_percentage_points"].idxmax()]
        magnitude = float(record["difference_percentage_points"])
    elif direction == "decrease":
        valid = valid[valid["difference_percentage_points"] < 0]
        if valid.empty:
            return "—"
        record = valid.loc[valid["difference_percentage_points"].idxmin()]
        magnitude = abs(float(record["difference_percentage_points"]))
    else:
        raise ValueError("direction must be 'increase' or 'decrease'")
    return f"{magnitude:.1f} ({record['month_label']})"


def build_manuscript_hac_summary_table(
    inference: pd.DataFrame,
) -> pd.DataFrame:
    """Twelve-row summary of all monthly HAC estimates without cherry-picking."""
    rows: List[Dict[str, object]] = []
    for phase in ("El Niño", "La Niña"):
        for basin in BASIN_ORDER:
            subset = inference[
                (inference["phase"] == phase)
                & (inference["basin"] == basin)
            ].copy()
            effect = subset["difference_percentage_points"].to_numpy(dtype=float)
            finite = np.isfinite(effect)
            rows.append({
                "ENSO phase": phase,
                "River basin": basin,
                "Valid months (n)": int(np.sum(finite)),
                "Increase months (n)": int(np.sum(finite & (effect > 0))),
                "Decrease months (n)": int(np.sum(finite & (effect < 0))),
                "Median |effect| (pp)": (
                    float(np.nanmedian(np.abs(effect))) if np.any(finite) else np.nan
                ),
                "Largest increase (pp; month)": _formatted_peak_effect(
                    subset, "increase"
                ),
                "Largest decrease (pp; month)": _formatted_peak_effect(
                    subset, "decrease"
                ),
                "95% CI excludes zero (n)": int(
                    subset["hac_ci_excludes_zero"].fillna(False).astype(bool).sum()
                ),
                "BH-FDR q≤0.05 (n)": int(
                    subset["bh_fdr_reject"].fillna(False).astype(bool).sum()
                ),
            })
    table = pd.DataFrame(rows)
    table["Median |effect| (pp)"] = table["Median |effect| (pp)"].round(1)
    return table


def write_word_ready_html_table(
    table: pd.DataFrame,
    title: str,
    notes: Sequence[str],
    output_path: Path,
) -> None:
    """Write a clean three-line-style table that can be pasted into Word."""
    table_html = table.to_html(
        index=False,
        border=0,
        na_rep="—",
        justify="center",
        escape=True,
    )
    note_html = "".join(
        f"<p class='note'><sup>{index + 1}</sup> {note}</p>"
        for index, note in enumerate(notes)
    )
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: 'Times New Roman', serif; margin: 28px; color: #111; }}
h2 {{ font-size: 12pt; line-height: 1.25; margin: 0 0 10px 0; font-weight: bold; }}
table {{ border-collapse: collapse; width: 100%; font-size: 10pt; }}
thead {{ border-top: 1.5pt solid #000; border-bottom: 0.8pt solid #000; }}
tbody {{ border-bottom: 1.5pt solid #000; }}
th {{ font-weight: bold; padding: 6px 7px; text-align: center; vertical-align: bottom; }}
td {{ padding: 5px 7px; text-align: center; vertical-align: middle; }}
td:first-child, th:first-child {{ text-align: left; }}
tbody tr:nth-child(even) {{ background: #f7f7f7; }}
.note {{ font-size: 9pt; line-height: 1.25; margin: 5px 0 0 0; }}
</style>
</head>
<body>
<h2>{title}</h2>
{table_html}
{note_html}
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def save_manuscript_tables(
    recurrent: pd.DataFrame,
    inference: pd.DataFrame,
    table_dir: Path,
) -> Dict[str, str]:
    """Save one compact main-text table and one compact supplementary table."""
    recurrent_table = build_manuscript_recurrent_table(recurrent)
    hac_summary = build_manuscript_hac_summary_table(inference)

    recurrent_csv = table_dir / "Table1_Figure6_recurrent_affected_area.csv"
    recurrent_html = table_dir / "Table1_Figure6_recurrent_affected_area.html"
    recurrent_table.to_csv(recurrent_csv, index=False, encoding="utf-8-sig")
    write_word_ready_html_table(
        recurrent_table,
        "Table 1 | Recurrent basin area affected by ENSO-associated changes "
        "in MEP95 q90 event probability.",
        (
            "Values are percentages of valid basin area meeting RR ≥ 1.5 "
            "(increase) or RR ≤ 0.5 (decrease) in at least two calendar months.",
            "Only area with valid RR estimates in all 12 calendar months is "
            "included in the denominator; fractional grid–basin intersection "
            "weights are calculated in an equal-area projection.",
            "This table summarizes effect-size recurrence and is not conditioned "
            "on statistical significance.",
        ),
        recurrent_html,
    )

    hac_csv = table_dir / "TableS2_FigureS15_HAC_summary.csv"
    hac_html = table_dir / "TableS2_FigureS15_HAC_summary.html"
    hac_summary.to_csv(hac_csv, index=False, encoding="utf-8-sig")
    write_word_ready_html_table(
        hac_summary,
        "Table S2 | Basin-level summary of monthly trend-adjusted MEP95 q90 "
        "event-area effects.",
        (
            "Effect is the ENSO-minus-Neutral coefficient from event area = "
            "intercept + ENSO phase + linear year trend and is reported in "
            "percentage points (pp).",
            "Uncertainty uses Newey–West HAC standard errors; BH-FDR is applied "
            "to 72 basin–month tests separately for El Niño and La Niña.",
            "Largest increase and decrease are descriptive maxima across the 12 "
            "months; their individual selection should not be interpreted as a "
            "separate significance test.",
        ),
        hac_html,
    )
    return {
        "table1_csv": str(recurrent_csv),
        "table1_html": str(recurrent_html),
        "tableS2_csv": str(hac_csv),
        "tableS2_html": str(hac_html),
    }


# =============================================================================
# MONTHLY CALCULATION
# =============================================================================

def read_counts(
    ds: xr.Dataset,
    q: int,
    month: int,
    phase: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = f"q{q}_{MONTH_LABEL[month]}_{PHASE_KEY[phase]}"
    return (
        get_required(ds, f"{base}_c_phase").astype(float),
        get_required(ds, f"{base}_n_phase").astype(float),
        get_required(ds, f"{base}_c_neutral").astype(float),
        get_required(ds, f"{base}_n_neutral").astype(float),
    )


def calculate_dataset(
    source_path: Path,
    metric: str,
    q: int,
    basin_weights: Mapping[str, np.ndarray],
    china_mask: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate monthly and recurrent basin summaries for one dataset."""
    resolved = resolve_nc(source_path, metric, q)
    print(f"Reading {metric} q{q}: {resolved}")
    with xr.open_dataset(resolved) as source:
        ds = source.load()

    ny, nx = china_mask.shape
    shape = (len(PHASES), 12, ny, nx)
    valid_all = np.zeros(shape, dtype=bool)
    effect_inc_all = np.zeros(shape, dtype=bool)
    effect_dec_all = np.zeros(shape, dtype=bool)
    wilson_inc_all = np.zeros(shape, dtype=bool)
    wilson_dec_all = np.zeros(shape, dtype=bool)
    rows: List[Dict[str, object]] = []

    for month_index, month in enumerate(MONTHS):
        for phase_index, phase in enumerate(PHASES):
            c_phase, n_phase, c_neutral, n_neutral = read_counts(
                ds, q, month, phase
            )
            wilson, p_phase, p_neutral, _low, _high = (
                reference_wilson_screen(
                    c_phase, n_phase, c_neutral, n_neutral, CFG.ALPHA
                )
            )
            valid = (
                china_mask
                & np.isfinite(p_phase)
                & np.isfinite(p_neutral)
                & (n_phase >= CFG.MIN_PHASE_N)
                & (n_neutral >= CFG.MIN_NEUTRAL_N)
            )
            rr_defined = valid & (p_neutral > 0)
            rr = np.divide(
                p_phase,
                p_neutral,
                out=np.full((ny, nx), np.nan, dtype=float),
                where=rr_defined,
            )
            effect_inc = rr_defined & np.isfinite(rr) & (rr >= CFG.RR_EFFECT_HIGH)
            effect_dec = rr_defined & np.isfinite(rr) & (rr <= CFG.RR_EFFECT_LOW)
            wilson_valid = wilson & valid

            valid_all[phase_index, month_index] = rr_defined
            effect_inc_all[phase_index, month_index] = effect_inc
            effect_dec_all[phase_index, month_index] = effect_dec
            wilson_inc_all[phase_index, month_index] = effect_inc & wilson_valid
            wilson_dec_all[phase_index, month_index] = effect_dec & wilson_valid

            for basin in BASIN_ORDER:
                weights = basin_weights[basin]
                values: Dict[str, float] = {}
                for support, inc_mask, dec_mask in (
                    ("effect", effect_inc, effect_dec),
                    ("wilson", effect_inc & wilson_valid, effect_dec & wilson_valid),
                ):
                    inc_pct, inc_km2, valid_km2 = weighted_fraction(
                        inc_mask, rr_defined, weights
                    )
                    dec_pct, dec_km2, _ = weighted_fraction(
                        dec_mask, rr_defined, weights
                    )
                    values[f"{support}_increase_area_percent"] = inc_pct
                    values[f"{support}_decrease_area_percent"] = dec_pct
                    values[f"{support}_increase_area_km2"] = inc_km2
                    values[f"{support}_decrease_area_km2"] = dec_km2
                    values["valid_area_km2"] = valid_km2

                # S14 diagnoses how much of the Figure 6 strong-effect area is
                # retained when the Neutral probability lies outside the ENSO-
                # phase 95% Wilson score interval. This retention rate is
                # distinct from absolute basin-area percent.
                for direction in ("increase", "decrease"):
                    effect_area = values[f"effect_{direction}_area_km2"]
                    supported_area = values[f"wilson_{direction}_area_km2"]
                    values[f"wilson_{direction}_retention_percent"] = (
                        100.0 * supported_area / effect_area
                        if np.isfinite(effect_area) and effect_area > 0
                        else np.nan
                    )

                basin_p_phase = weighted_mean(p_phase, valid, weights)
                basin_p_neutral = weighted_mean(p_neutral, valid, weights)
                if (
                    np.isfinite(basin_p_phase)
                    and np.isfinite(basin_p_neutral)
                    and basin_p_neutral > 0
                ):
                    basin_rr = basin_p_phase / basin_p_neutral
                    basin_log2rr = np.log2(basin_rr) if basin_rr > 0 else np.nan
                else:
                    basin_rr = np.nan
                    basin_log2rr = np.nan

                rows.append({
                    "metric": metric,
                    "threshold_percentile": q,
                    "basin": basin,
                    "phase": PHASE_LABEL[phase],
                    "phase_key": phase,
                    "month": month,
                    "month_label": MONTH_LABEL[month],
                    "basin_phase_probability": basin_p_phase,
                    "basin_neutral_probability": basin_p_neutral,
                    "basin_RR_ratio_of_area_weighted_probabilities": basin_rr,
                    "basin_log2RR": basin_log2rr,
                    **values,
                })

    monthly = pd.DataFrame(rows)
    recurrent_rows: List[Dict[str, object]] = []
    for phase_index, phase in enumerate(PHASES):
        valid_count = valid_all[phase_index].sum(axis=0)
        if CFG.REQUIRE_ALL_MONTHS_FOR_RECURRENCE_DENOMINATOR:
            recurrence_valid = valid_count == 12
        else:
            recurrence_valid = valid_count > 0

        count_sets = {
            "effect": (
                effect_inc_all[phase_index].sum(axis=0),
                effect_dec_all[phase_index].sum(axis=0),
            ),
            "wilson": (
                wilson_inc_all[phase_index].sum(axis=0),
                wilson_dec_all[phase_index].sum(axis=0),
            ),
        }
        for basin in BASIN_ORDER:
            weights = basin_weights[basin]
            for support, (inc_count, dec_count) in count_sets.items():
                inc_pct, inc_km2, denom_km2 = weighted_fraction(
                    inc_count >= CFG.RECURRENT_MIN_MONTHS,
                    recurrence_valid,
                    weights,
                )
                dec_pct, dec_km2, _ = weighted_fraction(
                    dec_count >= CFG.RECURRENT_MIN_MONTHS,
                    recurrence_valid,
                    weights,
                )
                recurrent_rows.append({
                    "metric": metric,
                    "threshold_percentile": q,
                    "basin": basin,
                    "phase": PHASE_LABEL[phase],
                    "phase_key": phase,
                    "support": support,
                    "recurrent_minimum_months": CFG.RECURRENT_MIN_MONTHS,
                    "valid_month_requirement": (
                        12 if CFG.REQUIRE_ALL_MONTHS_FOR_RECURRENCE_DENOMINATOR
                        else 1
                    ),
                    "valid_recurrence_area_km2": denom_km2,
                    "increase_recurrent_area_km2": inc_km2,
                    "decrease_recurrent_area_km2": dec_km2,
                    "increase_recurrent_area_percent": inc_pct,
                    "decrease_recurrent_area_percent": dec_pct,
                })

    return monthly, pd.DataFrame(recurrent_rows)


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def area_cmap_norm() -> Tuple[mcolors.Colormap, mcolors.BoundaryNorm]:
    """Discrete light-to-dark purple scale shared by the Figure 6 series."""
    base = mpl.colormaps["Purples"]
    colors = base(np.linspace(0.08, 0.96, len(CFG.AREA_BOUNDS) - 1))
    cmap = mcolors.ListedColormap(colors, name="basin_area_percent_purple")
    cmap.set_bad("#eeeeee")
    norm = mcolors.BoundaryNorm(CFG.AREA_BOUNDS, cmap.N, clip=True)
    return cmap, norm


def support_retention_cmap_norm() -> Tuple[mcolors.Colormap, mcolors.BoundaryNorm]:
    """Purple scale with extra resolution for small Wilson retention."""
    base = mpl.colormaps["Purples"]
    bounds = np.asarray(CFG.SUPPORT_RETENTION_BOUNDS, dtype=float)
    colors = base(np.linspace(0.06, 0.98, bounds.size - 1))
    cmap = mcolors.ListedColormap(colors, name="support_retention_purple")
    cmap.set_bad("#d9d9d9")
    norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap, norm


def basin_difference_cmap_norm() -> Tuple[mcolors.Colormap, mcolors.BoundaryNorm]:
    """Focused S15 scale; retain an explicit colour for values above 20 pp."""
    base = mpl.colormaps["Purples"]
    bounds = np.asarray(CFG.BASIN_AREA_DIFFERENCE_BOUNDS, dtype=float)
    colors = base(np.linspace(0.05, 0.98, bounds.size - 1))
    cmap = mcolors.ListedColormap(colors, name="basin_difference_purple")
    cmap.set_bad("#d9d9d9")
    cmap.set_over(base(0.995))
    norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=False)
    return cmap, norm


def matrix_for(
    monthly: pd.DataFrame,
    phase: str,
    direction: str,
    support: str,
) -> np.ndarray:
    column = f"{support}_{direction}_area_percent"
    subset = monthly[monthly["phase"] == phase]
    table = subset.pivot(index="basin", columns="month", values=column)
    return table.reindex(index=BASIN_ORDER, columns=MONTHS).to_numpy(dtype=float)


def retention_matrix_for(
    monthly: pd.DataFrame,
    phase: str,
    direction: str,
    support: str,
) -> np.ndarray:
    column = f"{support}_{direction}_retention_percent"
    subset = monthly[monthly["phase"] == phase]
    table = subset.pivot(index="basin", columns="month", values=column)
    return table.reindex(index=BASIN_ORDER, columns=MONTHS).to_numpy(dtype=float)


def basin_effect_matrix(
    inference: pd.DataFrame,
    phase: str,
    direction: str,
) -> np.ndarray:
    """Return directional magnitudes of trend-adjusted basin-area effects."""
    if direction not in {"increase", "decrease"}:
        raise ValueError("direction must be 'increase' or 'decrease'")
    subset = inference[inference["phase"] == phase]
    signed_table = subset.pivot(
        index="basin", columns="month", values="difference_percentage_points"
    ).reindex(index=BASIN_ORDER, columns=MONTHS)
    signed = signed_table.to_numpy(dtype=float)
    if direction == "increase":
        magnitude = np.where(np.isfinite(signed), np.maximum(signed, 0.0), np.nan)
    else:
        magnitude = np.where(np.isfinite(signed), np.maximum(-signed, 0.0), np.nan)
    return magnitude


def style_heatmap_axis(
    ax: plt.Axes,
    matrix: np.ndarray,
    cmap,
    norm,
    title: str,
    panel_label: str = "",
    annotate_values: bool = False,
) -> None:
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(MONTH_TICK_LABELS)
    ax.set_yticks(np.arange(len(BASIN_ORDER)))
    ax.set_yticklabels(BASIN_ORDER)
    if title:
        ax.set_title(title, loc="left", x=0.0, pad=8, fontweight="bold")
    if panel_label:
        ax.text(
            -0.018,
            1.018,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=CFG.PANEL_LABEL_SIZE,
            fontweight="bold",
            fontfamily=CFG.FONT_FAMILY,
            clip_on=False,
            zorder=10,
        )
    ax.set_xlabel("")
    ax.set_ylabel("River basin")
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BASIN_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    if annotate_values:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if not np.isfinite(value):
                    label = "–"
                    color = "#555555"
                elif value == 0:
                    label = "0"
                    color = "#333333"
                elif value < 0.1:
                    label = "<0.1"
                    color = "#333333"
                elif value < 10:
                    label = f"{value:.1f}"
                    color = "#333333" if value < 5 else "white"
                else:
                    label = f"{value:.0f}"
                    color = "white" if value >= 10 else "#333333"
                ax.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=CFG.HEATMAP_VALUE_SIZE,
                    fontweight="normal",
                    color=color,
                    zorder=5,
                )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")
    apply_bold_axis_typography(ax)


def add_area_colorbar(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    cmap,
    norm,
    label: str = "",
    ticks: Optional[Sequence[float]] = None,
    extend: str = "neither",
) -> None:
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=list(axes),
        orientation="horizontal",
        ticks=CFG.AREA_BOUNDS if ticks is None else ticks,
        spacing="uniform",
        fraction=0.045,
        pad=0.075,
        aspect=45,
        extend=extend,
    )
    if label:
        cbar.set_label(
            label,
            fontsize=CFG.COLORBAR_LABEL_SIZE,
            fontweight="bold",
            labelpad=5.0,
        )
    cbar.ax.tick_params(
        labelsize=CFG.COLORBAR_TICK_SIZE,
        length=3.0,
        width=0.7,
        pad=2.5,
    )
    cbar.ax.xaxis.label.set_fontweight("bold")
    for tick_label in cbar.ax.get_xticklabels():
        tick_label.set_fontweight("bold")
    cbar.outline.set_linewidth(0.65)
    try:
        cbar.dividers.set_linewidth(0.35)
        cbar.dividers.set_color("#777777")
    except Exception:
        pass


def recurrent_styles() -> Dict[Tuple[str, str], Tuple[str, str]]:
    """Assign one opaque solid colour to each phase-direction combination."""
    return {
        ("El Niño", "increase"): (
            CFG.EL_NINO_INCREASE_COLOR, "El Niño event: increase"
        ),
        ("El Niño", "decrease"): (
            CFG.EL_NINO_DECREASE_COLOR, "El Niño event: decrease"
        ),
        ("La Niña", "increase"): (
            CFG.LA_NINA_INCREASE_COLOR, "La Niña event: increase"
        ),
        ("La Niña", "decrease"): (
            CFG.LA_NINA_DECREASE_COLOR, "La Niña event: decrease"
        ),
    }


def effect_legend_handles_and_labels() -> Tuple[List[Patch], List[str]]:
    """Explicit proxies prevent Matplotlib's private '_nolegend_' labels."""
    labels = [
        "El Niño event: increase",
        "El Niño event: decrease",
        "La Niña event: increase",
        "La Niña event: decrease",
    ]
    handles = [
        Patch(
            facecolor=CFG.EL_NINO_INCREASE_COLOR,
            edgecolor="#555555",
            linewidth=1.1,
        ),
        Patch(
            facecolor=CFG.EL_NINO_DECREASE_COLOR,
            edgecolor="#555555",
            linewidth=1.1,
        ),
        Patch(
            facecolor=CFG.LA_NINA_INCREASE_COLOR,
            edgecolor="#555555",
            linewidth=1.1,
        ),
        Patch(
            facecolor=CFG.LA_NINA_DECREASE_COLOR,
            edgecolor="#555555",
            linewidth=1.1,
        ),
    ]
    return handles, labels


def add_effect_legend(
    ax: plt.Axes,
    y_anchor: float = -0.205,
    columns: int = 4,
) -> None:
    handles, labels = effect_legend_handles_and_labels()
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.50, y_anchor),
        ncol=columns,
        frameon=False,
        prop={
            "family": CFG.FONT_FAMILY,
            "size": CFG.BAR_LEGEND_SIZE,
            "weight": "bold",
        },
        handlelength=2.15,
        handleheight=1.15,
        handletextpad=0.65,
        columnspacing=1.45,
        borderaxespad=0.0,
    )
    for legend_text in legend.get_texts():
        legend_text.set_fontweight("bold")


def style_bar_axis(
    ax: plt.Axes,
    title: str,
    panel_label: str = "",
) -> None:
    """Apply one publication-style frame and typography to every bar panel."""
    ax.set_yticks(np.arange(len(BASIN_ORDER), dtype=float))
    ax.set_yticklabels(BASIN_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(*CFG.BAR_XLIM)
    ax.set_xticks(CFG.BAR_XTICKS)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax.grid(axis="x", color="#d5d5d5", linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=CFG.BAR_TICK_SIZE,
        width=0.85,
        length=4.0,
        direction="out",
    )
    ax.set_xlabel("")
    ax.set_ylabel(
        "River basin",
        fontsize=CFG.BAR_AXIS_LABEL_SIZE,
        fontweight="bold",
    )
    if title:
        ax.set_title(
            title,
            loc="left",
            x=0.0,
            pad=9,
            fontsize=CFG.BAR_TITLE_SIZE,
            fontweight="bold",
        )
    if panel_label:
        ax.text(
            -0.018,
            1.018,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=CFG.PANEL_LABEL_SIZE,
            fontweight="bold",
            fontfamily=CFG.FONT_FAMILY,
            clip_on=False,
            zorder=10,
        )
    # A complete rectangular frame is clearer in the final journal layout.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(CFG.BAR_FRAME_WIDTH)
        spine.set_color("#3f3f3f")
    apply_bold_axis_typography(ax)


def plot_recurrent_bar_panel(
    ax: plt.Axes,
    recurrent: pd.DataFrame,
    support: str,
    title: str,
    panel_label: str = "",
    show_legend: bool = True,
    legend_columns: int = 4,
    legend_y_anchor: float = -0.205,
) -> None:
    subset = recurrent[recurrent["support"] == support].copy()
    y = np.arange(len(BASIN_ORDER), dtype=float)
    # With an inverted y axis these offsets give a top-to-bottom order that
    # exactly matches the legend: EN increase, EN decrease, LN increase,
    # LN decrease.
    offsets = {
        ("El Niño", "increase"): -0.30,
        ("El Niño", "decrease"): -0.10,
        ("La Niña", "increase"): 0.10,
        ("La Niña", "decrease"): 0.30,
    }
    for (phase, direction), (color, _label) in recurrent_styles().items():
        values = (
            subset[subset["phase"] == phase]
            .set_index("basin")
            .reindex(BASIN_ORDER)[f"{direction}_recurrent_area_percent"]
            .to_numpy(dtype=float)
        )
        ax.barh(
            y + offsets[(phase, direction)],
            values,
            height=CFG.BAR_HEIGHT,
            color=color,
            edgecolor="#555555",
            linewidth=0.85,
            zorder=4,
        )
    style_bar_axis(ax, title, panel_label=panel_label)
    if show_legend:
        add_effect_legend(
            ax,
            y_anchor=legend_y_anchor,
            columns=legend_columns,
        )


def panel_e_recurrent_area_table(
    recurrent: pd.DataFrame,
    support: str = "effect",
) -> pd.DataFrame:
    """Return the four recurrent-area series displayed in Figure 6e."""
    subset = recurrent[recurrent["support"] == support].copy()
    rows: List[Dict[str, object]] = []
    for basin in BASIN_ORDER:
        for phase in ("El Niño", "La Niña"):
            part = subset[
                (subset["basin"] == basin) & (subset["phase"] == phase)
            ]
            if part.empty:
                continue
            record = part.iloc[0]
            for direction in ("increase", "decrease"):
                rows.append({
                    "basin": basin,
                    "phase": phase,
                    "direction": direction,
                    "recurrent_area_percent": float(
                        record[f"{direction}_recurrent_area_percent"]
                    ),
                    "recurrent_minimum_months": CFG.RECURRENT_MIN_MONTHS,
                })
    return pd.DataFrame(rows)


def plot_panel_e_recurrent_area(
    ax: plt.Axes,
    recurrent: pd.DataFrame,
) -> pd.DataFrame:
    """Plot the simplified, full-width recurrent-area summary."""
    plot_recurrent_bar_panel(
        ax,
        recurrent,
        support="effect",
        title="Recurrent affected basin area",
        panel_label="(e)",
        show_legend=True,
    )
    return panel_e_recurrent_area_table(recurrent, support="effect")


# =============================================================================
# FIGURE COMPOSITIONS
# =============================================================================

def plot_figure6(
    monthly: pd.DataFrame,
    recurrent: pd.DataFrame,
    out_prefix: Path,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    cmap, norm = area_cmap_norm()
    fig = plt.figure(
        figsize=CFG.MAIN_FIGSIZE,
        constrained_layout=True,
    )
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 1.05))
    compact_constrained_layout(fig)
    heat_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    synthesis_ax = fig.add_subplot(grid[2, :])
    heatmap_specs = (
        ("El Niño", "increase", "El Niño 50% increase the probability", "(a)"),
        ("La Niña", "increase", "La Niña 50% increase the probability", "(b)"),
        ("El Niño", "decrease", "El Niño 50% decrease the probability", "(c)"),
        ("La Niña", "decrease", "La Niña 50% decrease the probability", "(d)"),
    )
    for ax, (phase, direction, title, panel_label) in zip(heat_axes, heatmap_specs):
        style_heatmap_axis(
            ax,
            matrix_for(monthly, phase, direction, "effect"),
            cmap,
            norm,
            title,
            panel_label=panel_label,
        )
    add_area_colorbar(
        fig,
        heat_axes,
        cmap,
        norm,
        "Affected fraction of valid basin area (%)",
    )
    panel_e_table = plot_panel_e_recurrent_area(synthesis_ax, recurrent)
    return save_figure(fig, out_prefix), panel_e_table


def plot_support_heatmaps(
    monthly: pd.DataFrame,
    support: str,
    out_prefix: Path,
) -> Dict[str, str]:
    if support != "wilson":
        raise ValueError("Figure S14 supports only 'wilson' retention.")
    cmap, norm = support_retention_cmap_norm()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=CFG.SUPPORT_FIGSIZE,
        constrained_layout=True,
    )
    compact_constrained_layout(fig)
    specs = (
        ("El Niño", "increase", "(a)"),
        ("La Niña", "increase", "(b)"),
        ("El Niño", "decrease", "(c)"),
        ("La Niña", "decrease", "(d)"),
    )
    for ax, (phase, direction, panel) in zip(axes.ravel(), specs):
        style_heatmap_axis(
            ax,
            retention_matrix_for(monthly, phase, direction, support),
            cmap,
            norm,
            "",
            panel_label=panel,
            annotate_values=True,
        )
    add_area_colorbar(
        fig,
        list(axes.ravel()),
        cmap,
        norm,
        "Strong-effect area retained by Wilson-based screening (%)",
        ticks=CFG.SUPPORT_RETENTION_BOUNDS,
    )
    return save_figure(fig, out_prefix)


def plot_basin_effect_heatmaps(
    inference: pd.DataFrame,
    out_prefix: Path,
) -> Dict[str, str]:
    """Plot effect magnitudes only; formal HAC inference is external Table S1."""
    cmap, norm = basin_difference_cmap_norm()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=CFG.SUPPORT_FIGSIZE,
        constrained_layout=True,
    )
    compact_constrained_layout(fig)
    specs = (
        ("El Niño", "increase", "(a)"),
        ("La Niña", "increase", "(b)"),
        ("El Niño", "decrease", "(c)"),
        ("La Niña", "decrease", "(d)"),
    )
    maximum_magnitude = 0.0
    for ax, (phase, direction, panel) in zip(axes.ravel(), specs):
        magnitude = basin_effect_matrix(inference, phase, direction)
        finite_magnitude = magnitude[np.isfinite(magnitude)]
        if finite_magnitude.size:
            maximum_magnitude = max(
                maximum_magnitude,
                float(np.max(finite_magnitude)),
            )
        style_heatmap_axis(
            ax,
            magnitude,
            cmap,
            norm,
            "",
            panel_label=panel,
            annotate_values=True,
        )
    add_area_colorbar(
        fig,
        list(axes.ravel()),
        cmap,
        norm,
        "Trend-adjusted effect magnitude (percentage points)",
        ticks=CFG.BASIN_AREA_DIFFERENCE_BOUNDS,
        extend=(
            "max"
            if maximum_magnitude > CFG.BASIN_AREA_DIFFERENCE_BOUNDS[-1]
            else "neither"
        ),
    )
    return save_figure(fig, out_prefix)


def plot_sensitivity_recurrence(
    q95_recurrent: pd.DataFrame,
    rx5_recurrent: pd.DataFrame,
    out_prefix: Path,
) -> Dict[str, str]:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=CFG.SENSITIVITY_FIGSIZE,
        constrained_layout=True,
    )
    compact_constrained_layout(fig)
    plot_recurrent_bar_panel(
        axes[0],
        q95_recurrent,
        "effect",
        "",
        panel_label="(a)",
        show_legend=False,
    )
    plot_recurrent_bar_panel(
        axes[1],
        rx5_recurrent,
        "effect",
        "",
        panel_label="(b)",
        show_legend=True,
        legend_columns=2,
        legend_y_anchor=-0.19,
    )
    return save_figure(fig, out_prefix)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main() -> None:
    plt.close("all")
    set_style()
    print(f"Running: {SCRIPT_VERSION}")
    resolved_inputs, grid_metric_nc = audit_input_paths()

    # Validate all NetCDF schemas and coordinate grids before reading vector
    # data or calculating fractional intersections.
    grids: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
    for key, resolved in resolved_inputs.items():
        grids[key] = audit_netcdf_structure(resolved, key[0], key[1])
    lon, lat = grids[("MEP95", 90)]
    for (metric, q), (source_lon, source_lat) in grids.items():
        same_grid = (
            source_lon.shape == lon.shape
            and source_lat.shape == lat.shape
            and np.allclose(source_lon, lon, rtol=0.0, atol=1.0e-8)
            and np.allclose(source_lat, lat, rtol=0.0, atol=1.0e-8)
        )
        if not same_grid:
            raise ValueError(
                f"{metric} q{q} uses a different lon/lat grid from MEP95 q90. "
                "Automatic regridding is intentionally disabled because it "
                "would alter the fractional basin-area weights."
            )

    boundaries = load_boundaries()
    china_mask = make_china_mask(lon, lat, boundaries["china_polygon"])
    if not np.any(china_mask):
        raise ValueError(
            "The China polygon does not overlap the NetCDF grid. Check the "
            "coordinate reference system and longitude convention."
        )
    basin_weights = build_fractional_basin_area_weights(
        lon, lat, boundaries["basin_polygons"]
    )
    print(f"[Spatial audit] China-mask grid cells: {int(china_mask.sum())}")
    for basin in BASIN_ORDER:
        print(
            f"[Spatial audit] {basin} fractional grid area: "
            f"{float(np.sum(basin_weights[basin])):,.1f} km²"
        )

    fig_dir, table_dir = output_dirs()

    # Figure S15 is calculated from year-resolved fields rather than from
    # already aggregated event counts. One calendar year is one observation
    # for a given basin and calendar month.
    mep95_anomaly = load_year_resolved_anomaly(
        grid_metric_nc,
        "MEP95",
        lon,
        lat,
    )
    yearly_basin_area = build_yearly_basin_event_area(
        mep95_anomaly,
        "MEP95",
        90,
        basin_weights,
    )
    del mep95_anomaly
    basin_inference = trend_adjusted_hac_inference(yearly_basin_area)
    yearly_basin_area.to_csv(
        table_dir / "FigureS15_yearly_basin_q90_event_area.csv",
        index=False,
        encoding="utf-8-sig",
    )
    basin_inference.to_csv(
        table_dir / "FigureS15_trend_adjusted_NeweyWest_HAC_BHFDR_full.csv",
        index=False,
        encoding="utf-8-sig",
    )
    table_s1_outputs = save_hac_publication_table(basin_inference, table_dir)
    print(
        "[Figure S15/Table S1 audit] BH-FDR-supported HAC tests: "
        f"{int(basin_inference['bh_fdr_reject'].sum())} / "
        f"{int(np.isfinite(basin_inference['hac_p_two_sided']).sum())}."
    )

    all_monthly: List[pd.DataFrame] = []
    all_recurrent: List[pd.DataFrame] = []
    datasets = (
        (CFG.NC_MEP95_Q90, "MEP95", 90),
        (CFG.NC_MEP95_Q95, "MEP95", 95),
        (CFG.NC_RX5DAY_Q90, "Rx5day", 90),
    )
    results: Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for _configured_path, metric, q in datasets:
        resolved = resolved_inputs[(metric, q)]
        monthly, recurrent = calculate_dataset(
            resolved, metric, q, basin_weights, china_mask
        )
        results[(metric, q)] = (monthly, recurrent)
        all_monthly.append(monthly)
        all_recurrent.append(recurrent)

    monthly_all = pd.concat(all_monthly, ignore_index=True)
    recurrent_all = pd.concat(all_recurrent, ignore_index=True)
    monthly_all.to_csv(
        table_dir / "Figure6_monthly_basin_area_and_RR.csv",
        index=False,
        encoding="utf-8-sig",
    )
    recurrent_all.to_csv(
        table_dir / "Figure6_recurrent_basin_area.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mep90_monthly, mep90_recurrent = results[("MEP95", 90)]
    _mep95_monthly, mep95_recurrent = results[("MEP95", 95)]
    _rx5_monthly, rx5_recurrent = results[("Rx5day", 90)]

    main_outputs, panel_e_table = plot_figure6(
        mep90_monthly,
        mep90_recurrent,
        fig_dir / "Figure6_MEP95_q90_basin_risk_translation",
    )
    panel_e_table.to_csv(
        table_dir / "Figure6_panel_e_recurrent_affected_area.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manuscript_table_outputs = save_manuscript_tables(
        mep90_recurrent,
        basin_inference,
        table_dir,
    )

    output_records = [
        {
            "figure": "Figure6",
            "role": (
                "MEP95 q90 basin-scale affected-area and recurrence translation"
            ),
            "manuscript_table_csv": manuscript_table_outputs.get(
                "table1_csv", ""
            ),
            "manuscript_table_html": manuscript_table_outputs.get(
                "table1_html", ""
            ),
            **main_outputs,
        },
        {
            "figure": "FigureS14",
            "role": "MEP95 q90 Wilson retention within strong-effect area",
            **plot_support_heatmaps(
                mep90_monthly,
                "wilson",
                fig_dir / "FigureS14_MEP95_q90_Wilson_retention_of_effect_area",
            ),
        },
        {
            "figure": "FigureS15",
            "role": (
                "MEP95 q90 trend-adjusted basin-year event-area effect sizes; "
                "formal Newey-West HAC inference is reported in Table S1"
            ),
            "table_s1_csv": table_s1_outputs.get("csv", ""),
            "table_s1_notes_csv": table_s1_outputs.get("notes_csv", ""),
            "summary_table_csv": manuscript_table_outputs.get(
                "tableS2_csv", ""
            ),
            "summary_table_html": manuscript_table_outputs.get(
                "tableS2_html", ""
            ),
            **plot_basin_effect_heatmaps(
                basin_inference,
                fig_dir
                / "FigureS15_MEP95_q90_trend_adjusted_effect_size",
            ),
        },
        {
            "figure": "FigureS16",
            "role": "MEP95 q95 and Rx5day q90 basin recurrence sensitivity",
            **plot_sensitivity_recurrence(
                mep95_recurrent,
                rx5_recurrent,
                fig_dir / "FigureS16_basin_recurrence_sensitivity",
            ),
        },
    ]
    pd.DataFrame(output_records).to_csv(
        table_dir / "Figure6_figure_output_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFinished the Figure 6 series:")
    for record in output_records:
        print(f"  {record['figure']}: {record['role']}")
    print(f"Tables: {table_dir}")
    print(f"External Figure S15 inference table: {table_s1_outputs}")
    print(f"Manuscript-ready compact tables: {manuscript_table_outputs}")


if __name__ == "__main__":
    main()
