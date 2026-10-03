# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from statsmodels.stats.proportion import proportion_confint

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
    # --- input ---
    PRECIP_NC_FILE = DATA_ROOT / "precipitation" / "CN05.1_Pre_1961_2023_daily_025x025.nc"

    ENSO_EXCEL_FILE = (
        OUTPUT_ROOT / "01_enso_classification" /
        "ENSO_article_figS1_S2_NOAA_sensitivity_1961_2023_v2_panelwise.xlsx"
    )
    ENSO_SHEET = "Main_monthly_data"

    # This should be the exact six-basin polygon file used in Figure 1.
    BASIN_POLYGON_FILE = DATA_ROOT / "basins" / "six_selected_official_basins.shp"
    BASIN_LAYER_NAME = None

    # Fallback folder if BASIN_POLYGON_FILE does not exist.
    BASIN_SEARCH_FOLDER = DATA_ROOT / "basins"

    OUT_DIR = OUTPUT_ROOT / "02_basin_distribution_rr_far"

    # --- period ---
    START_YEAR = 1961
    END_YEAR = 2023

    # --- ENSO columns ---
    PHASE_COL = "main_phase"
    EVENT_COL = "main_event"
    INDEX_COL = "nino34_detrended"

    # Your reported sample counts. The script stops if they do not match.
    STRICT_ENSO_COUNTS = True
    EXPECTED_COUNTS = {"El Nino": 189, "La Nina": 209, "Neutral": 358}

    # --- methods ---
    MAIN_METHOD = "MEP95"
    SUPPORT_METHOD = "RX5day"
    METHODS = ["MEP95", "RX5day"]

    # MEP95 wet-day threshold
    WET_DAY_MM = 1.0

    # Extreme thresholds. Both are calculated and marked in CDF panels.
    THRESHOLDS = [90, 95]

    # Figure 2 design:
    # Use monthly anomalies instead of raw MEP95/RX5day for A-F CDFs and
    # for the q90/q95 event definition used in RR/FAR. The raw metrics are
    # still stored in NetCDF and exported to checking tables.
    USE_CALENDAR_MONTH_ANOMALY_FOR_FIG2 = True
    ANOMALY_BASELINE_START_YEAR = START_YEAR
    ANOMALY_BASELINE_END_YEAR = END_YEAR


    # Panel (g) risk summary threshold.
    PANEL_G_THRESHOLD = 90

    # Show q90/q95 reference lines, but hide q90/q95 labels in legend and text.
    SHOW_Q_LINES = True
    SHOW_Q_TEXT = False
    Q_LINES_IN_LEGEND = False

    # --- basin settings ---
    BASIN_ORDER = [
        "Songhua River Basin",
        "Haihe River Basin",
        "Yellow River Basin",
        "Huaihe River Basin",
        "Yangtze River Basin",
        "Pearl River Basin",
    ]

    BASIN_SHORT_EN = {
        "Songhua River Basin": "Songhua",
        "Haihe River Basin": "Haihe",
        "Yellow River Basin": "Yellow",
        "Huaihe River Basin": "Huaihe",
        "Yangtze River Basin": "Yangtze",
        "Pearl River Basin": "Pearl",
    }

    BASIN_SHORT_CN = {
        "Songhua River Basin": "松花江",
        "Haihe River Basin": "海河",
        "Yellow River Basin": "黄河",
        "Huaihe River Basin": "淮河",
        "Yangtze River Basin": "长江",
        "Pearl River Basin": "珠江",
    }

    BASIN_KEYWORDS = {
        "Songhua River Basin": ["Songhua", "Northeast", "Northeast Rivers", "松花", "东北诸河", "东北"],
        "Haihe River Basin": ["Haihe", "海河"],
        "Yellow River Basin": ["Yellow", "Huang", "黄河"],
        "Huaihe River Basin": ["Huaihe", "淮河"],
        "Yangtze River Basin": ["Yangtze", "Changjiang", "长江"],
        "Pearl River Basin": ["Pearl", "Zhujiang", "珠江"],
    }

    # Set to a known field name if your basin file has one, e.g. "NAME" or "basin".
    BASIN_NAME_FIELD = None

    CHINA_ALBERS = (
        "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 "
        "+lon_0=105 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )

    # --- probability / uncertainty ---
    # Figure 2 uses empirical event probabilities throughout:
    #     P = event_count / valid_sample_count
    # No Haldane/continuity correction is applied to Figure 2, Figure S1, or Figure S2.
    # Zero-event handling is explicit in rr_far():
    #     phase count = 0 and Neutral probability > 0 -> RR = 0
    #     Neutral probability = 0 -> RR = NaN (undefined)
    CONFIDENCE_LEVEL = 0.95
    RUN_BOOTSTRAP = True
    N_BOOTSTRAP = 1000
    RANDOM_SEED = 20260723

    # --- figure style ---
    FIG_DPI = 600
    FIG_SIZE = (13.8, 9.4)
    EXPORT_TIFF = True
    EXPORT_PDF = True
    SHARE_XLIM_FOR_CDF = False

    COLORS = {"El Nino": "#d7191c", "La Nina": "#2166ac", "Neutral": "#444444"}
    LINEWIDTHS = {"El Nino": 1.9, "La Nina": 1.9, "Neutral": 1.65}
    PHASES = ["El Nino", "La Nina", "Neutral"]
    PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

    Q_LINE_STYLE = {
        90: dict(color="black", linestyle="--", linewidth=1.0, alpha=0.78),
        95: dict(color="0.35", linestyle=":", linewidth=1.15, alpha=0.78),
    }

    RR_YLIM = None
    FAR_YLIM = (-0.05, 1.0)
    GRID = True


TABLE_DIR = CFG.OUT_DIR / "01_tables"
FIG_DIR = CFG.OUT_DIR / "02_figures"
NC_DIR = CFG.OUT_DIR / "03_netcdf"
for folder in [CFG.OUT_DIR, TABLE_DIR, FIG_DIR, NC_DIR]:
    Path(folder).mkdir(parents=True, exist_ok=True)


# =============================================================================
# BILINGUAL LABELS
# =============================================================================

TXT = {
    "EN": {
        "title_MEP95": "Figure 2. Basin-wise distributional shifts and extreme-risk metrics based on MEP95 anomaly",
        "title_RX5day": "Figure S. Basin-wise distributional shifts and extreme-risk metrics based on Rx5day anomaly",
        "metric_MEP95": "Monthly MEP95 (mm)",
        "metric_RX5day": "Monthly Rx5day (mm)",
        "cdf": "Empirical CDF",
        "cdf_percent": "Cumulative probability (%)",
        "phase_legend": "ENSO phase",
        "metric_legend": "Risk metric",
        "rr": "Relative risk (RR)",
        "far": "FAR",
        "rr_left": "RR, left axis",
        "far_right": "FAR, right axis",
        "panel_g": "Basin-level RR and FAR",
        "note": ("CDF panels show basin-mean monthly metrics. The left y-axis gives empirical CDF values, "
                 "whereas the right y-axis gives cumulative probability in percent. Vertical reference lines "
                 "mark q90 and q95 but are not labeled in the legend. Panel (g) uses q{q} events."),
    },
    "CN": {
        "title_MEP95": "图2. 基于 MEP95 的流域尺度极端降水分布变化与风险指标",
        "title_RX5day": "图S. 基于 Rx5day 的流域尺度极端降水分布变化与风险指标",
        "metric_MEP95": "月尺度 MEP95 (mm)",
        "metric_RX5day": "月尺度 Rx5day (mm)",
        "cdf": "经验 CDF",
        "cdf_percent": "累积概率 (%)",
        "phase_legend": "ENSO 位相",
        "metric_legend": "风险指标",
        "rr": "相对风险 RR",
        "far": "归因风险比例 FAR",
        "rr_left": "RR，左轴",
        "far_right": "FAR，右轴",
        "panel_g": "流域尺度 RR 与 FAR",
        "note": ("CDF 子图为流域平均月尺度指标。左侧纵轴为 0–1 的经验 CDF，右侧纵轴为对应的累积百分比。"
                 "竖向参考线表示 q90 和 q95，但不在图例中显示文字。图 (g) 基于 q{q} 事件。"),
    },
}

PHASE_LABELS = {
    "EN": {"El Nino": "El Niño", "La Nina": "La Niña", "Neutral": "Neutral"},
    "CN": {"El Nino": "厄尔尼诺", "La Nina": "拉尼娜", "Neutral": "中性"},
}

BASIN_SHORT = {"EN": CFG.BASIN_SHORT_EN, "CN": CFG.BASIN_SHORT_CN}


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def set_font(lang: str) -> None:
    """
    Use Times New Roman as the primary journal font.

    For Chinese-language figures, Matplotlib still needs a CJK fallback font for
    Chinese glyphs. Therefore the font list is ordered as:
        Times New Roman -> SimSun/SimHei/Microsoft YaHei -> DejaVu
    English letters, numbers, axes, and math-like labels will use Times New Roman
    whenever it is installed on your computer.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "SimSun",
            "SimHei",
            "Microsoft YaHei",
            "Noto Sans CJK SC",
            "DejaVu Serif",
        ],
        "font.sans-serif": [
            "Times New Roman",
            "SimSun",
            "SimHei",
            "Microsoft YaHei",
            "Noto Sans CJK SC",
            "DejaVu Sans",
        ],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })



def coord_name(da: xr.DataArray, names: Iterable[str]) -> str:
    lower = {n.lower() for n in names}
    for c in da.coords:
        if c.lower() in lower:
            return c
    for d in da.dims:
        if d.lower() in lower:
            return d
    raise ValueError(f"Cannot find coordinate among {names}. dims={da.dims}, coords={list(da.coords)}")


def choose_precip_var(ds: xr.Dataset) -> str:
    for v in ["pre", "precip", "precipitation", "pr", "rain", "tp"]:
        if v in ds.data_vars:
            return v
    for v in ds.data_vars:
        dims = [d.lower() for d in ds[v].dims]
        if "time" in dims:
            return v
    raise ValueError(f"No precipitation variable found. Variables: {list(ds.data_vars)}")


def normalize_phase(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower().replace("ñ", "n").replace("í", "i")
    s = s.replace("_", " ").replace("-", " ")
    if "el" in s and "nino" in s:
        return "El Nino"
    if "la" in s and "nina" in s:
        return "La Nina"
    if "neutral" in s:
        return "Neutral"
    return None


def safe_text(row: pd.Series) -> str:
    return " ".join(str(v) for v in row.values if pd.notna(v))


def save_fig(fig: plt.Figure, basename: str) -> Dict[str, str]:
    png = FIG_DIR / f"{basename}.png"
    tif = FIG_DIR / f"{basename}.tif"
    pdf = FIG_DIR / f"{basename}.pdf"

    fig.savefig(png, dpi=CFG.FIG_DPI, bbox_inches="tight", facecolor="white")

    if CFG.EXPORT_TIFF:
        try:
            fig.savefig(tif, dpi=CFG.FIG_DPI, bbox_inches="tight", facecolor="white",
                        format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        except TypeError:
            fig.savefig(tif, dpi=CFG.FIG_DPI, bbox_inches="tight", facecolor="white", format="tiff")

    if CFG.EXPORT_PDF:
        fig.savefig(pdf, dpi=CFG.FIG_DPI, bbox_inches="tight", facecolor="white")

    plt.close(fig)
    return {"figure": basename, "png": str(png), "tif": str(tif), "pdf": str(pdf)}


# =============================================================================
# LOAD INPUTS
# =============================================================================

def load_daily_precip() -> xr.DataArray:
    """
    Open daily precipitation lazily.

    Important memory fix:
    Do NOT apply `pr = pr.where(pr >= 0)` here. With the netCDF4 backend this
    may read the entire 1961-2023 daily array into memory and trigger a
    MemoryError before spatial cropping. Negative or invalid values are handled
    later only on the small monthly / calendar-month subsets.
    """
    if not CFG.PRECIP_NC_FILE.exists():
        raise FileNotFoundError(f"Precipitation NetCDF not found: {CFG.PRECIP_NC_FILE}")

    ds = xr.open_dataset(CFG.PRECIP_NC_FILE)
    var = choose_precip_var(ds)
    pr = ds[var]

    t = coord_name(pr, ["time", "t"])
    y = coord_name(pr, ["lat", "latitude", "y"])
    x = coord_name(pr, ["lon", "longitude", "x"])

    ren = {}
    if t != "time":
        ren[t] = "time"
    if y != "lat":
        ren[y] = "lat"
    if x != "lon":
        ren[x] = "lon"
    if ren:
        pr = pr.rename(ren)

    pr["time"] = pd.to_datetime(pr["time"].values)
    pr = pr.sel(time=slice(f"{CFG.START_YEAR}-01-01", f"{CFG.END_YEAR}-12-31"))

    # Sorting by coordinate is safe because it only changes indexing metadata;
    # no precipitation values are read here.
    if pr["lat"].values[0] > pr["lat"].values[-1]:
        pr = pr.sortby("lat")

    pr.name = "daily_precip"
    print(f"Precip var: {var}; lazy shape={pr.shape}")
    print("Daily precipitation opened lazily; invalid-value masking is deferred to monthly subsets.")
    return pr



def load_enso() -> Tuple[pd.DataFrame, Tuple[str, str, str]]:
    if not CFG.ENSO_EXCEL_FILE.exists():
        raise FileNotFoundError(f"ENSO Excel not found: {CFG.ENSO_EXCEL_FILE}")

    xls = pd.ExcelFile(CFG.ENSO_EXCEL_FILE)
    sheet = CFG.ENSO_SHEET if CFG.ENSO_SHEET in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(CFG.ENSO_EXCEL_FILE, sheet_name=sheet)

    phase_col = CFG.PHASE_COL if CFG.PHASE_COL in df.columns else None
    event_col = CFG.EVENT_COL if CFG.EVENT_COL in df.columns else None
    index_col = CFG.INDEX_COL if CFG.INDEX_COL in df.columns else None

    if phase_col is None:
        for c in ["main_phase", "phase", "enso_phase", "noaa_phase"]:
            if c in df.columns:
                phase_col = c
                break

    if event_col is None:
        for c in ["main_event", "event", "enso_event", "noaa_event"]:
            if c in df.columns:
                event_col = c
                break

    if index_col is None:
        for c in ["nino34_detrended", "nino34_anom", "noaa_oni_detrended", "oni"]:
            if c in df.columns:
                index_col = c
                break

    if phase_col is None or index_col is None:
        raise ValueError(f"Cannot identify ENSO columns. Existing columns: {list(df.columns)}")

    if "year" not in df.columns or "month" not in df.columns:
        raise ValueError("ENSO table must contain year and month columns.")

    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["month"] = pd.to_numeric(out["month"], errors="coerce")
    out = out.dropna(subset=["year", "month"]).copy()
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out = out[(out["year"] >= CFG.START_YEAR) & (out["year"] <= CFG.END_YEAR)].copy()

    out["ym"] = out["year"].astype(str) + "-" + out["month"].astype(str).str.zfill(2)
    out["date"] = pd.to_datetime(out["ym"] + "-01")
    out["phase"] = out[phase_col].apply(normalize_phase)
    out["enso_index"] = pd.to_numeric(out[index_col], errors="coerce")
    out["event_label"] = out[event_col] if event_col else ""

    out = out.dropna(subset=["phase", "enso_index"]).sort_values("date").reset_index(drop=True)

    counts = out["phase"].value_counts().to_dict()
    print("ENSO phase counts:")
    for p in CFG.PHASES:
        print(f"  {p}: {counts.get(p, 0)}")

    if CFG.STRICT_ENSO_COUNTS:
        errors = []
        for p, exp in CFG.EXPECTED_COUNTS.items():
            got = int(counts.get(p, 0))
            if got != exp:
                errors.append(f"{p}: expected {exp}, got {got}")
        if errors:
            raise ValueError("ENSO sample counts do not match.\n" + "\n".join(errors))

    return out, (phase_col, event_col or "", index_col)


def align_time(metric: xr.DataArray, enso: pd.DataFrame) -> Tuple[xr.DataArray, pd.DataFrame]:
    metric_ym = pd.to_datetime(metric["time"].values).strftime("%Y-%m")
    enso_ym = enso["ym"].astype(str).values

    common = sorted(set(metric_ym) & set(enso_ym))
    if not common:
        raise RuntimeError("No common year-month between metric and ENSO table.")

    mp = {ym: i for i, ym in enumerate(metric_ym)}
    ep = {ym: i for i, ym in enumerate(enso_ym)}

    metric2 = metric.isel(time=[mp[ym] for ym in common])
    metric2 = metric2.assign_coords(time=pd.to_datetime([ym + "-01" for ym in common]))

    enso2 = enso.iloc[[ep[ym] for ym in common]].copy().reset_index(drop=True)
    print(f"Aligned months: {common[0]} to {common[-1]}, n={len(common)}")
    return metric2, enso2


# =============================================================================
# METRICS
# =============================================================================

def _time_index(pr: xr.DataArray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pr["time"].values))


def _select_one_month(pr: xr.DataArray, year: int, month: int, time_index: Optional[pd.DatetimeIndex] = None) -> xr.DataArray:
    if time_index is None:
        time_index = _time_index(pr)
    pos = np.where((time_index.year == year) & (time_index.month == month))[0]
    return pr.isel(time=pos)


def _empty_month_like(pr: xr.DataArray, time_value: pd.Timestamp, name: str) -> xr.DataArray:
    base = pr.isel(time=0, drop=True)
    arr = xr.full_like(base, np.nan, dtype=np.float32)
    arr = arr.expand_dims(time=[pd.Timestamp(time_value)])
    arr.name = name
    return arr


def clean_precip_subset(sub: xr.DataArray) -> xr.DataArray:
    """
    Clean a small precipitation subset after spatial/time selection.

    This function is intentionally applied only to monthly or calendar-month
    slices, not to the full daily dataset, to avoid full-array memory allocation.
    """
    sub = sub.astype(np.float32)
    sub = sub.where(np.isfinite(sub))
    sub = sub.where(sub >= 0)
    return sub


def compute_monthly_rx5day_lowmem(pr: xr.DataArray) -> xr.DataArray:
    """
    Memory-safe monthly Rx5day.

    Do NOT use:
        pr.rolling(time=5).sum().resample(time='MS').max()

    because that creates a full daily rolling array and may require several GB
    of memory. This function processes one calendar month at a time, loads only
    one monthly output grid, and then concatenates the 756 monthly grids.
    """
    print("Compute Rx5day with low-memory month-by-month loop.")

    time_index = _time_index(pr)
    month_starts = pd.date_range(
        f"{CFG.START_YEAR}-01-01",
        f"{CFG.END_YEAR}-12-01",
        freq="MS",
    )

    monthly = []
    for k, start_time in enumerate(month_starts, start=1):
        year = int(start_time.year)
        month = int(start_time.month)
        sub = _select_one_month(pr, year, month, time_index)
        sub = clean_precip_subset(sub)

        if sub.sizes.get("time", 0) < 5:
            out_m = _empty_month_like(pr, start_time, "RX5day")
        else:
            # Rolling only inside this single month, so memory use is small.
            out_m = (
                sub.rolling(time=5, min_periods=5)
                .sum()
                .max(dim="time", skipna=True)
                .astype(np.float32)
                .load()
                .expand_dims(time=[pd.Timestamp(start_time)])
            )
            out_m.name = "RX5day"

        monthly.append(out_m)

        if k % 60 == 0 or k == len(month_starts):
            print(f"  Rx5day processed {k}/{len(month_starts)} months")

    out = xr.concat(monthly, dim="time")
    out.name = "RX5day"
    out["ym"] = out.time.dt.strftime("%Y-%m")
    return out


def compute_monthly_mep95_lowmem(pr: xr.DataArray) -> xr.DataArray:
    """
    Memory-safe monthly MEP95.

    MEP95 is calculated as the monthly accumulated precipitation from wet days
    exceeding the local wet-day P95 threshold for the same calendar month.
    The threshold is calculated separately for Jan, Feb, ..., Dec.
    """
    print("Compute MEP95 with low-memory calendar-month/year loop.")

    time_index = _time_index(pr)
    month_outputs = []

    for calendar_month in range(1, 13):
        print(f"  MEP95 calendar-month threshold: month={calendar_month:02d}")

        cm_pos = np.where(time_index.month == calendar_month)[0]
        pr_cm = clean_precip_subset(pr.isel(time=cm_pos))
        wet_cm = pr_cm.where(pr_cm >= CFG.WET_DAY_MM)

        p95 = wet_cm.quantile(0.95, dim="time", skipna=True)

        if "quantile" in p95.dims:
            p95 = p95.isel(quantile=0, drop=True)
        elif "quantile" in p95.coords:
            p95 = p95.squeeze(drop=True)
            try:
                p95 = p95.drop_vars("quantile")
            except Exception:
                pass

        p95 = p95.astype(np.float32).load()

        for year in range(CFG.START_YEAR, CFG.END_YEAR + 1):
            start_time = pd.Timestamp(year=year, month=calendar_month, day=1)
            sub = _select_one_month(pr, year, calendar_month, time_index)
            sub = clean_precip_subset(sub)

            if sub.sizes.get("time", 0) == 0:
                out_m = _empty_month_like(pr, start_time, "MEP95")
            else:
                wet_sub = sub.where(sub >= CFG.WET_DAY_MM)
                extreme = wet_sub.where(wet_sub > p95)
                out_m = (
                    extreme.sum(dim="time", skipna=True)
                    .astype(np.float32)
                    .load()
                    .expand_dims(time=[start_time])
                )
                out_m.name = "MEP95"

            month_outputs.append(out_m)

    out = xr.concat(month_outputs, dim="time").sortby("time")
    out.name = "MEP95"
    out["ym"] = out.time.dt.strftime("%Y-%m")
    return out


def compute_monthly_metric(pr: xr.DataArray, method: str) -> xr.DataArray:
    """
    Compute monthly MEP95 or Rx5day using a low-memory algorithm.

    This function replaces the previous whole-array rolling calculation that
    triggered:
        MemoryError: Unable to allocate 2.39 GiB ...
    """
    m = method.upper()

    if m == "RX5DAY":
        out = compute_monthly_rx5day_lowmem(pr)
    elif m == "MEP95":
        out = compute_monthly_mep95_lowmem(pr)
    else:
        raise ValueError("method must be MEP95 or RX5day.")

    out = out.sel(time=slice(f"{CFG.START_YEAR}-01-01", f"{CFG.END_YEAR}-12-01"))
    if out["lat"].values[0] > out["lat"].values[-1]:
        out = out.sortby("lat")
    out["ym"] = out.time.dt.strftime("%Y-%m")
    return out


# =============================================================================
# BASIN POLYGONS AND WEIGHTS
# =============================================================================

def read_gdf(path: Path, layer=None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Empty vector file: {path}")
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    return gdf.to_crs(epsg=4326)


def polygon_layer(gdf: gpd.GeoDataFrame) -> bool:
    return bool(set(gdf.geometry.geom_type.unique()) & {"Polygon", "MultiPolygon"})


def standardize_basins(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()

    if CFG.BASIN_NAME_FIELD and CFG.BASIN_NAME_FIELD in out.columns:
        texts = out[CFG.BASIN_NAME_FIELD].astype(str)
    elif "basin_name" in out.columns:
        texts = out["basin_name"].astype(str)
    else:
        texts = out.drop(columns="geometry").apply(safe_text, axis=1)

    assigned = []
    for txt in texts:
        low = txt.lower()
        found = None
        for basin, keys in CFG.BASIN_KEYWORDS.items():
            if any(k.lower() in low for k in keys):
                found = basin
                break
        assigned.append(found)

    out["basin_name"] = assigned
    return out.dropna(subset=["basin_name"]).copy()


def search_basins(folder: Path) -> gpd.GeoDataFrame:
    if not folder.exists():
        raise FileNotFoundError(f"Fallback basin folder not found: {folder}")

    parts = []
    for ext in ["*.shp", "*.gpkg", "*.geojson"]:
        for path in folder.rglob(ext):
            try:
                gdf = read_gdf(path)
                if not polygon_layer(gdf):
                    continue
                std = standardize_basins(gdf)
                if not std.empty:
                    parts.append(std)
            except Exception:
                continue

    if not parts:
        raise RuntimeError("No usable basin polygon found. Please set BASIN_POLYGON_FILE manually.")

    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=parts[0].crs)


def load_basins() -> gpd.GeoDataFrame:
    if CFG.BASIN_POLYGON_FILE.exists():
        print(f"Read basin polygons: {CFG.BASIN_POLYGON_FILE}")
        gdf = read_gdf(CFG.BASIN_POLYGON_FILE, CFG.BASIN_LAYER_NAME)
    else:
        print("BASIN_POLYGON_FILE not found; searching fallback folder.")
        gdf = search_basins(CFG.BASIN_SEARCH_FOLDER)

    if not polygon_layer(gdf):
        raise ValueError("Basin file is not a polygon layer.")

    gdf = standardize_basins(gdf)
    missing = [b for b in CFG.BASIN_ORDER if b not in set(gdf["basin_name"])]
    if missing:
        raise ValueError("Missing basin polygons:\n" + "\n".join(missing))

    gdf = gdf[gdf["basin_name"].isin(CFG.BASIN_ORDER)].copy()
    gdf = gdf.dissolve(by="basin_name", as_index=False)
    gdf["basin_name"] = pd.Categorical(gdf["basin_name"], categories=CFG.BASIN_ORDER, ordered=True)
    return gdf.sort_values("basin_name").reset_index(drop=True)


def crop_precip_to_six_basin_extent(pr: xr.DataArray, basins: gpd.GeoDataFrame, buffer_deg: float = 1.0) -> xr.DataArray:
    """
    First spatially screen the daily precipitation data to the six-basin envelope,
    then calculate MEP95/Rx5day only inside this reduced domain.

    Workflow:
        six basin polygons -> crop daily precipitation -> calculate monthly metrics
        -> area-weighted basin means -> q90/q95 events -> NetCDF -> figures.

    The exact basin attribution is still controlled by basin-grid intersection
    weights, so boundary grid cells are area-weighted rather than simply counted.
    """
    ensure_output_dirs()

    minx, miny, maxx, maxy = basins.total_bounds
    lon_min = float(minx) - buffer_deg
    lon_max = float(maxx) + buffer_deg
    lat_min = float(miny) - buffer_deg
    lat_max = float(maxy) + buffer_deg

    if pr["lat"].values[0] > pr["lat"].values[-1]:
        pr = pr.sortby("lat")

    lon_values = pr["lon"].values

    # Support both -180..180 and 0..360 longitude conventions.
    if np.nanmax(lon_values) > 180 and lon_min < 0:
        lon_min = lon_min % 360
        lon_max = lon_max % 360

    if lon_min <= lon_max:
        pr_crop = pr.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    else:
        pr1 = pr.sel(lon=slice(lon_min, np.nanmax(lon_values)), lat=slice(lat_min, lat_max))
        pr2 = pr.sel(lon=slice(np.nanmin(lon_values), lon_max), lat=slice(lat_min, lat_max))
        pr_crop = xr.concat([pr1, pr2], dim="lon").sortby("lon")

    if pr_crop.sizes.get("lat", 0) == 0 or pr_crop.sizes.get("lon", 0) == 0:
        raise RuntimeError(
            "After cropping to the six-basin extent, no precipitation grid cells remain. "
            "Please check basin CRS, precipitation lon/lat coordinates, or buffer_deg."
        )

    print("Six-basin spatial screening completed.")
    print(f"  Basin bounds       : lon {minx:.2f}–{maxx:.2f}, lat {miny:.2f}–{maxy:.2f}")
    print(f"  Buffered crop box  : lon {lon_min:.2f}–{lon_max:.2f}, lat {lat_min:.2f}–{lat_max:.2f}")
    print(f"  Original grid size : lat={pr.sizes.get('lat')}, lon={pr.sizes.get('lon')}")
    print(f"  Cropped grid size  : lat={pr_crop.sizes.get('lat')}, lon={pr_crop.sizes.get('lon')}")

    pd.DataFrame([{
        "basin_lon_min": minx,
        "basin_lon_max": maxx,
        "basin_lat_min": miny,
        "basin_lat_max": maxy,
        "buffer_deg": buffer_deg,
        "crop_lon_min": lon_min,
        "crop_lon_max": lon_max,
        "crop_lat_min": lat_min,
        "crop_lat_max": lat_max,
        "original_nlat": pr.sizes.get("lat"),
        "original_nlon": pr.sizes.get("lon"),
        "cropped_nlat": pr_crop.sizes.get("lat"),
        "cropped_nlon": pr_crop.sizes.get("lon"),
    }]).to_csv(TABLE_DIR / "six_basin_precip_crop_extent.csv", index=False, encoding="utf-8-sig")

    return pr_crop


def edges(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 1:
        return np.array([vals[0] - 0.125, vals[0] + 0.125])
    mids = (vals[:-1] + vals[1:]) / 2
    return np.concatenate([[vals[0] - (mids[0] - vals[0])], mids, [vals[-1] + (vals[-1] - mids[-1])]])


def build_grid(metric: xr.DataArray) -> gpd.GeoDataFrame:
    lon = metric["lon"].values
    lat = metric["lat"].values
    xe = edges(lon)
    ye = edges(lat)

    recs, geoms = [], []
    for iy in range(len(lat)):
        for ix in range(len(lon)):
            recs.append({"lat_index": iy, "lon_index": ix, "flat_index": iy * len(lon) + ix,
                         "lon": float(lon[ix]), "lat": float(lat[iy])})
            geoms.append(box(float(xe[ix]), float(ye[iy]), float(xe[ix+1]), float(ye[iy+1])))

    return gpd.GeoDataFrame(recs, geometry=geoms, crs="EPSG:4326")


def basin_grid_weights(metric: xr.DataArray, basins: gpd.GeoDataFrame) -> pd.DataFrame:
    print("Calculate basin-grid intersection weights.")
    grid = build_grid(metric).to_crs(CFG.CHINA_ALBERS)
    bas = basins.to_crs(CFG.CHINA_ALBERS)
    sidx = grid.sindex
    rows = []

    for _, brow in bas.iterrows():
        basin = str(brow["basin_name"])
        geom = brow.geometry
        idx = list(sidx.query(geom, predicate="intersects"))
        cand = grid.iloc[idx].copy()
        if cand.empty:
            raise RuntimeError(f"No grid cells intersect {basin}")

        cand["intersect_area_m2"] = cand.geometry.intersection(geom).area
        cand = cand[cand["intersect_area_m2"] > 0].copy()
        cand["weight"] = cand["intersect_area_m2"] / cand["intersect_area_m2"].sum()
        cand["basin"] = basin

        rows.append(cand[["basin", "lat_index", "lon_index", "flat_index", "intersect_area_m2", "weight"]])
        print(f"  {basin}: {len(cand)} cells")

    w = pd.concat(rows, ignore_index=True)
    ensure_output_dirs()
    w.to_csv(TABLE_DIR / "six_basin_grid_intersection_weights.csv", index=False, encoding="utf-8-sig")
    return w


# =============================================================================
# BASIN-MEAN AND RR/FAR
# =============================================================================

def basin_mean(metric: xr.DataArray, weights: pd.DataFrame, enso: pd.DataFrame, method: str) -> pd.DataFrame:
    arr = metric.values
    flat = arr.reshape(arr.shape[0], -1)
    rows = []

    for basin in CFG.BASIN_ORDER:
        wdf = weights[weights["basin"] == basin]
        idx = wdf["flat_index"].to_numpy(dtype=int)
        w = wdf["weight"].to_numpy(dtype=float)

        vals = flat[:, idx]
        valid = np.isfinite(vals)
        num = np.nansum(vals * w[None, :], axis=1)
        den = np.sum(valid * w[None, :], axis=1)
        bm = np.where(den > 0, num / den, np.nan)

        one = pd.DataFrame({
            "method": method,
            "basin": basin,
            "time": pd.to_datetime(metric["time"].values),
            "year": enso["year"].values,
            "month": enso["month"].values,
            "ym": enso["ym"].values,
            "phase": enso["phase"].values,
            "event_label": enso["event_label"].values,
            "enso_index": enso["enso_index"].values,
            "metric": bm,
        })
        rows.append(one)

    return pd.concat(rows, ignore_index=True).dropna(subset=["metric", "phase"])


def add_threshold_events(df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    thr_rows = []

    for q in CFG.THRESHOLDS:
        out[f"threshold_q{q}"] = np.nan
        out[f"event_q{q}"] = 0

        for (basin, month), sub in out.groupby(["basin", "month"]):
            thr = float(np.nanpercentile(sub["metric"].astype(float), q))
            m = (out["basin"] == basin) & (out["month"] == month)
            out.loc[m, f"threshold_q{q}"] = thr
            out.loc[m, f"event_q{q}"] = (out.loc[m, "metric"].astype(float) >= thr).astype(int)

            thr_rows.append({"method": method, "basin": basin, "month": month,
                             "threshold": q, "threshold_value": thr})

    return out, pd.DataFrame(thr_rows)


def empirical_probability(count: int, n: int) -> float:
    """
    Empirical binomial event probability used by the entire Figure 2 series.

    P = count / n

    Figure 2 intentionally does not apply a Haldane/continuity correction.
    """
    if n <= 0:
        return np.nan
    return float(count) / float(n)


def empirical_rr(p_phase: float, p_neutral: float) -> float:
    """
    Relative risk based on empirical probabilities.

    Rules
    -----
    - finite P_phase and P_neutral > 0: RR = P_phase / P_neutral
    - P_phase = 0 and P_neutral > 0: RR = 0
    - P_neutral = 0 or invalid: RR is undefined (NaN)
    """
    if not np.isfinite(p_phase) or not np.isfinite(p_neutral):
        return np.nan
    if p_neutral <= 0:
        return np.nan
    return float(p_phase / p_neutral)


def bootstrap_rr(bdf: pd.DataFrame, q: int, phase: str, rng: np.random.Generator) -> Tuple[float, float]:
    """
    Month-preserving bootstrap 95% CI for the empirical RR.

    The resampling design is unchanged from the original Figure 2 workflow.
    The only change is the RR estimator: every bootstrap replicate now uses
    empirical probabilities P=count/n, with no continuity correction.

    Replicates in which the resampled Neutral event probability is zero are
    skipped because RR is undefined for those replicates.
    """
    if not CFG.RUN_BOOTSTRAP:
        return np.nan, np.nan

    ev = f"event_q{q}"
    ph = bdf[bdf["phase"] == phase]
    ne = bdf[bdf["phase"] == "Neutral"]
    vals = []

    if len(ph) < 3 or len(ne) < 3:
        return np.nan, np.nan

    for _ in range(CFG.N_BOOTSTRAP):
        ph_s, ne_s = [], []
        for m in sorted(bdf["month"].unique()):
            p_m = ph[ph["month"] == m]
            n_m = ne[ne["month"] == m]
            if len(p_m):
                ph_s.append(
                    p_m.sample(
                        len(p_m),
                        replace=True,
                        random_state=int(rng.integers(0, 2**31 - 1)),
                    )
                )
            if len(n_m):
                ne_s.append(
                    n_m.sample(
                        len(n_m),
                        replace=True,
                        random_state=int(rng.integers(0, 2**31 - 1)),
                    )
                )

        if not ph_s or not ne_s:
            continue

        pp = pd.concat(ph_s)
        nn = pd.concat(ne_s)

        p1 = empirical_probability(int(pp[ev].sum()), len(pp))
        p0 = empirical_probability(int(nn[ev].sum()), len(nn))
        rr_boot = empirical_rr(p1, p0)

        if np.isfinite(rr_boot):
            vals.append(rr_boot)

    if len(vals) < 20:
        return np.nan, np.nan

    a = (1 - CFG.CONFIDENCE_LEVEL) / 2
    lo, hi = np.percentile(vals, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


def rr_far(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Calculate basin-level q90/q95 empirical probabilities, RR and FAR.

    Figure-2 estimator
    ------------------
    P_phase   = count_phase / n_phase
    P_neutral = count_neutral / n_neutral
    RR        = P_phase / P_neutral
    FAR       = 1 - 1/RR, interpreted only when RR > 1

    Zero-event rules
    ----------------
    - phase count = 0, Neutral count > 0 -> RR = 0
    - Neutral count = 0 -> RR = NaN (undefined)

    Wilson intervals are still calculated directly from the observed raw
    binomial counts. They are not continuity-corrected.
    """
    rows = []
    rng = np.random.default_rng(CFG.RANDOM_SEED)

    for basin in CFG.BASIN_ORDER:
        bdf = df[df["basin"] == basin].copy()

        for q in CFG.THRESHOLDS:
            ev = f"event_q{q}"
            neutral = bdf[bdf["phase"] == "Neutral"]
            n0 = len(neutral)
            c0 = int(neutral[ev].sum())
            p0 = empirical_probability(c0, n0)

            for phase in ["El Nino", "La Nina"]:
                ph = bdf[bdf["phase"] == phase]
                n1 = len(ph)
                c1 = int(ph[ev].sum())
                p1 = empirical_probability(c1, n1)

                if n1:
                    p1lo, p1hi = proportion_confint(
                        c1,
                        n1,
                        alpha=1-CFG.CONFIDENCE_LEVEL,
                        method="wilson",
                    )
                else:
                    p1lo, p1hi = np.nan, np.nan

                if n0:
                    p0lo, p0hi = proportion_confint(
                        c0,
                        n0,
                        alpha=1-CFG.CONFIDENCE_LEVEL,
                        method="wilson",
                    )
                else:
                    p0lo, p0hi = np.nan, np.nan

                rr = empirical_rr(p1, p0)
                far = 1.0 - 1.0 / rr if np.isfinite(rr) and rr > 1 else np.nan
                rrlo, rrhi = bootstrap_rr(bdf, q, phase, rng)

                if not np.isfinite(rr):
                    risk_direction = "undefined"
                elif rr > 1:
                    risk_direction = "increase"
                elif rr < 1:
                    risk_direction = "decrease"
                else:
                    risk_direction = "no_change"

                if c0 == 0:
                    zero_event_status = "neutral_zero_rr_undefined"
                elif c1 == 0:
                    zero_event_status = "phase_zero_rr_zero"
                else:
                    zero_event_status = "none"

                rows.append({
                    "method": method,
                    "basin": basin,
                    "threshold": q,
                    "phase": phase,
                    "n_phase": n1,
                    "count_phase": c1,
                    "P_phase": p1,
                    "P_phase_wilson_low": p1lo,
                    "P_phase_wilson_high": p1hi,
                    "n_neutral": n0,
                    "count_neutral": c0,
                    "P_neutral": p0,
                    "P_neutral_wilson_low": p0lo,
                    "P_neutral_wilson_high": p0hi,
                    "RR": rr,
                    "RR_bootstrap_low": rrlo,
                    "RR_bootstrap_high": rrhi,
                    "log2RR": np.log2(rr) if np.isfinite(rr) and rr > 0 else np.nan,
                    "FAR": far,
                    "risk_direction": risk_direction,
                    "zero_event_status": zero_event_status,
                    "probability_estimator": "empirical_count_over_n",
                    "continuity_correction": "none",
                })

    return pd.DataFrame(rows)



# =============================================================================
# CEE-STYLE PLOT FROM NETCDF + METRIC-FIRST WORKFLOW
# =============================================================================

# New output structure requested by the user:
# calculate MEP95 and Rx5day -> save NC -> make figures from NC.
CFG.OUT_DIR = OUTPUT_ROOT / "02_basin_distribution_rr_far"
TABLE_DIR = CFG.OUT_DIR / "02_tables"
FIG_DIR = CFG.OUT_DIR / "03_figures"
NC_DIR = CFG.OUT_DIR / "01_netcdf"
for folder in [CFG.OUT_DIR, TABLE_DIR, FIG_DIR, NC_DIR]:
    Path(folder).mkdir(parents=True, exist_ok=True)

NC_FILE = NC_DIR / "six_basin_MEP95_RX5day_metric_events.nc"

def ensure_output_dirs():
    """Create all output folders at runtime. This avoids OSError in notebooks."""
    for folder in [CFG.OUT_DIR, TABLE_DIR, FIG_DIR, NC_DIR]:
        Path(folder).mkdir(parents=True, exist_ok=True)

# Controls
# If a NetCDF file is already open elsewhere, direct overwriting can fail on Windows.
# This version writes to a temporary file first and, if needed, creates a timestamped new file.
REBUILD_NC = True
PLOT_ONLY_FROM_EXISTING_NC = False

# CDF x-axis:
# All basin panels use the same physical length and the same 0-100% percentile range.
# Tick labels are selected adaptively from percentile positions and show real metric
# values in parentheses. q90/q95 vertical reference lines are always drawn.
X_PERCENT_LIMIT = (0, 100)
X_PERCENT_TICK_CANDIDATES = [5, 25, 50, 75, 90, 95, 99]
X_PERCENT_MAX_TICKS = 7

# Better CEE-style visual settings
CFG.FIG_SIZE = (14.2, 9.6)
CFG.COLORS = {"El Nino": "#d73027", "La Nina": "#2166ac", "Neutral": "#3d3d3d"}
CFG.LINEWIDTHS = {"El Nino": 2.05, "La Nina": 2.05, "Neutral": 1.75}
CFG.Q_LINES_IN_LEGEND = False
CFG.SHOW_Q_TEXT = False
CFG.SHOW_Q_LINES = True
# q90/q95 visual reference lines in CDF panels.
# User-requested color scheme:
#   q90 = red dashed line
#   q95 = blue dashed line
CFG.Q_LINE_STYLE = {
    90: dict(color="#d73027", linestyle="--", linewidth=1.15, alpha=0.92),
    95: dict(color="#2166ac", linestyle="--", linewidth=1.15, alpha=0.92),
}

TXT["EN"].update({
    "title_MEP95": "Figure 2. ENSO-related distributional shifts and basin-scale extreme-precipitation risk based on MEP95 anomaly",
    "title_RX5day": "Figure S. ENSO-related distributional shifts and basin-scale extreme-precipitation risk based on Rx5day anomaly",
    "bottom_percent_axis": "Reference percentile (%) with true value in parentheses",
    "top_true_axis": "Basin-specific metric value",
    "note": (
        "CDF panels use basin-mean monthly metrics. The x-axis is standardized as a basin-specific "
        "reference percentile; true metric values are shown in parentheses and on the top axis. "
        "Vertical reference lines indicate q90 and q95 but are not shown in the legend. "
        "Panel (g) is calculated from calendar-month q{q} exceedance events."
    ),
})
TXT["CN"].update({
    "title_MEP95": "图2. 基于 MEP95 的 ENSO 相关分布变化与流域尺度极端降水风险",
    "title_RX5day": "图S. 基于 Rx5day 的 ENSO 相关分布变化与流域尺度极端降水风险",
    "bottom_percent_axis": "参考百分位 (%)，括号内为真实指标值",
    "top_true_axis": "流域对应真实指标值",
    "note": (
        "CDF 子图使用流域平均月尺度指标。x 轴以流域自身参考百分位为主，"
        "括号内为对应真实指标值，顶部 x 轴为真实指标值。竖向参考线表示 q90 和 q95，"
        "但不进入图例。图 (g) 基于逐日历月 q{q} 超阈值事件计算。"
    ),
})


def phase_code(phase: str) -> int:
    return {"El Nino": 1, "La Nina": -1, "Neutral": 0}.get(str(phase), -9)


def phase_from_code(code: int) -> str:
    return {1: "El Nino", -1: "La Nina", 0: "Neutral"}.get(int(code), "Unknown")


def fmt_value(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def _grid_template_from_precip(pr: xr.DataArray) -> xr.DataArray:
    """
    Create a tiny one-time-slice template only for building the basin-grid
    intersection weights. It carries lon/lat grid geometry without holding any
    monthly metric arrays in memory.
    """
    tmp = pr.isel(time=[0]).copy(deep=False)
    tmp.name = "grid_template"
    return tmp


def _fill_method_arrays_from_events(
    method_index: int,
    method: str,
    events_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    time_index_map: Dict[pd.Timestamp, int],
    metric_all: np.ndarray,
    event_all: np.ndarray,
    threshold_all: np.ndarray,
) -> None:
    for bi, basin in enumerate(CFG.BASIN_ORDER):
        sub = events_df[events_df["basin"] == basin]
        for _, row in sub.iterrows():
            ti = time_index_map.get(pd.to_datetime(row["time"]))
            if ti is None:
                continue
            metric_all[method_index, bi, ti] = float(row["metric"])
            for qi, q in enumerate(CFG.THRESHOLDS):
                event_all[method_index, bi, ti, qi] = int(row[f"event_q{q}"])

        for calendar_month in range(1, 13):
            for qi, q in enumerate(CFG.THRESHOLDS):
                one = threshold_df[
                    (threshold_df["basin"] == basin)
                    & (threshold_df["month"] == calendar_month)
                    & (threshold_df["threshold"] == q)
                ]
                if not one.empty:
                    threshold_all[method_index, bi, calendar_month - 1, qi] = float(one["threshold_value"].iloc[0])



def safe_write_netcdf(ds: xr.Dataset, target: Path) -> Path:
    """
    Safely write NetCDF on Windows.

    Why this is needed
    ------------------
    Windows does not allow overwriting a NetCDF file that is still open by
    Python, Panoply, ArcGIS, QGIS, another notebook cell, or a previous
    xr.open_dataset() handle. Directly calling

        ds.to_netcdf(target)

    may therefore raise:
        PermissionError: [Errno 13] Permission denied

    Strategy
    --------
    1. Always write to a temporary NetCDF file first.
    2. Try to replace the standard target file.
    3. If the standard target is locked, keep the locked file untouched and
       save a new timestamped NetCDF file instead.
    4. Return the actual file path written, so plotting uses the new file.
    """
    ensure_output_dirs()
    target = Path(target)

    tmp = target.with_name(target.stem + "__tmp_write.nc")
    if tmp.exists():
        try:
            tmp.unlink()
        except PermissionError:
            tmp = target.with_name(
                target.stem + "__tmp_write_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".nc"
            )

    # Write the temporary file first. This avoids opening the locked target in
    # write mode.
    ds.to_netcdf(tmp)

    # If target does not exist, simply rename tmp to target.
    if not target.exists():
        tmp.replace(target)
        return target

    # If target exists, try to replace it.
    try:
        target.unlink()
        tmp.replace(target)
        return target
    except PermissionError:
        # The old target is locked. Use a new timestamped file instead.
        final = target.with_name(
            target.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + target.suffix
        )
        tmp.replace(final)
        print("=" * 88)
        print("WARNING: the standard NetCDF file is locked and could not be overwritten.")
        print(f"Locked file : {target}")
        print(f"New file    : {final}")
        print("The figures and tables will be generated from the new NetCDF file.")
        print("=" * 88)
        return final

def build_nc_first() -> Path:
    ensure_output_dirs()
    print("=" * 88)
    print("Step 1: select six basins, crop precipitation, calculate MEP95/Rx5day, then save NetCDF")
    print("=" * 88)

    # 1) Strictly select the six basin polygons first.
    basins = load_basins()
    print("Selected six basins:")
    for b in CFG.BASIN_ORDER:
        print(f"  - {b}")

    # 2) Load daily precipitation and crop it to the six-basin envelope.
    # This avoids calculating metrics over unrelated grids.
    pr = load_daily_precip()
    pr = crop_precip_to_six_basin_extent(pr, basins, buffer_deg=1.0)

    # Keep the cropped daily data lazy. Do not call .load() here.
    # Subsets are converted to float32 inside clean_precip_subset().

    # 3) Load ENSO labels.
    enso, enso_cols = load_enso()

    # 4) Build basin-grid intersection weights from the cropped grid only.
    grid_template = _grid_template_from_precip(pr)
    weights = basin_grid_weights(grid_template, basins)

    # 5) Create the target time axis from the ENSO table. Each method will be
    # aligned to this same axis and then written into preallocated arrays.
    target_times = list(pd.to_datetime(enso["date"].values))
    time_index_map = {t: i for i, t in enumerate(target_times)}
    n_method = len(CFG.METHODS)
    n_basin = len(CFG.BASIN_ORDER)
    n_time = len(target_times)
    n_threshold = len(CFG.THRESHOLDS)

    metric_all = np.full((n_method, n_basin, n_time), np.nan, dtype=np.float32)
    event_all = np.full((n_method, n_basin, n_time, n_threshold), 0, dtype=np.int16)
    threshold_all = np.full((n_method, n_basin, 12, n_threshold), np.nan, dtype=np.float32)

    all_df_for_check = {}

    # 6) Process one method at a time. This is the key memory fix:
    # no dictionary of full monthly gridded metrics is kept in memory.
    for mi, method in enumerate(CFG.METHODS):
        print("-" * 88)
        print(f"Low-memory method processing: {method}")
        print("-" * 88)

        metric = compute_monthly_metric(pr, method)
        metric, ens = align_time(metric, enso)

        # Check that the aligned time axis matches the ENSO target.
        if list(pd.to_datetime(ens["date"].values)) != target_times:
            raise RuntimeError(f"Aligned time axis for {method} does not match the ENSO target time axis.")

        bmean = basin_mean(metric, weights, ens, method)
        events, thr = add_threshold_events(bmean, method)

        _fill_method_arrays_from_events(
            method_index=mi,
            method=method,
            events_df=events,
            threshold_df=thr,
            time_index_map=time_index_map,
            metric_all=metric_all,
            event_all=event_all,
            threshold_all=threshold_all,
        )

        # Save small readable tables immediately.
        method_dir = TABLE_DIR / method
        method_dir.mkdir(parents=True, exist_ok=True)
        events.to_csv(method_dir / f"{method}_basin_monthly_metric_events.csv", index=False, encoding="utf-8-sig")
        thr.to_csv(method_dir / f"{method}_basin_calendar_month_thresholds.csv", index=False, encoding="utf-8-sig")
        all_df_for_check[method] = events

        # Release the gridded monthly metric before the next method.
        del metric, bmean, events, thr

    phase_codes = np.array([phase_code(p) for p in enso["phase"].values], dtype=np.int16)
    years = enso["year"].to_numpy(dtype=np.int16)
    months = enso["month"].to_numpy(dtype=np.int16)
    enso_index = enso["enso_index"].to_numpy(dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "basin_mean_metric": (("method", "basin", "time"), metric_all),
            "event": (("method", "basin", "time", "threshold"), event_all),
            "threshold_value": (("method", "basin", "calendar_month", "threshold"), threshold_all),
            "enso_phase_code": (("time",), phase_codes),
            "enso_index": (("time",), enso_index),
            "year": (("time",), years),
            "month": (("time",), months),
        },
        coords={
            "method": CFG.METHODS,
            "basin": CFG.BASIN_ORDER,
            "time": pd.to_datetime(target_times),
            "calendar_month": np.arange(1, 13, dtype=np.int16),
            "threshold": np.array(CFG.THRESHOLDS, dtype=np.int16),
        },
        attrs={
            "description": "Six-basin MEP95 and Rx5day metrics, q90/q95 events, thresholds, and ENSO labels.",
            "workflow": "six-basin selection -> cropped precipitation -> low-memory metric calculation -> NetCDF -> plotting from NetCDF",
            "main_method": CFG.MAIN_METHOD,
            "supporting_method": CFG.SUPPORT_METHOD,
            "phase_code": "El Nino=1, La Nina=-1, Neutral=0",
            "event_definition": "event=1 if basin_mean_metric exceeds the basin-calendar-month percentile threshold.",
            "MEP95_definition": "Monthly accumulated precipitation from wet days exceeding the local wet-day 95th percentile.",
            "RX5day_definition": "Monthly maximum consecutive 5-day precipitation amount.",
            "ENSO_phase_column": enso_cols[0],
            "ENSO_event_column": enso_cols[1],
            "ENSO_index_column": enso_cols[2],
        },
    )

    ds["basin_mean_metric"].attrs["units"] = "mm"
    ds["threshold_value"].attrs["units"] = "mm"

    nc_written = safe_write_netcdf(ds, NC_FILE)

    # Export a combined monthly checking table from the in-memory Dataset.
    ensure_output_dirs()
    for method in CFG.METHODS:
        print(f"Export monthly checking table from NetCDF: {method}")
        nc_to_method_dataframe(ds, method).to_csv(
            TABLE_DIR / f"{method}_monthly_metrics_from_NC.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(f"Saved NetCDF: {nc_written}")
    return nc_written


def _find_coord_index(ds: xr.Dataset, coord_name: str, value) -> int:
    """
    Find the exact integer position of a coordinate value.

    This avoids xarray .sel() ambiguity and fixes cases where a selected value
    still keeps an unexpected extra dimension, which previously caused:
        TypeError: basin_mean_metric is not scalar after squeeze; shape=(2,), size=2
    """
    values = [str(v) for v in ds[coord_name].values]
    value_str = str(value)

    matches = [i for i, v in enumerate(values) if v == value_str]
    if len(matches) != 1:
        raise ValueError(
            f"Coordinate selection for {coord_name}={value_str!r} is not unique. "
            f"matches={matches}, available={values}"
        )
    return matches[0]


def _to_scalar(value, name: str = "value"):
    """
    Convert a numpy/xarray value to a Python scalar.

    This is now only used after explicit integer indexing. If this fails, it
    means the dimension order or coordinate indexing has changed unexpectedly.
    """
    arr = np.asarray(value)
    arr = np.squeeze(arr)

    if arr.size != 1:
        raise TypeError(f"{name} is not scalar after squeeze; shape={arr.shape}, size={arr.size}")

    return arr.item()


def nc_to_method_dataframe(ds: xr.Dataset, method: str) -> pd.DataFrame:
    """
    Convert one method in the NetCDF Dataset to a long DataFrame.

    Strict indexing version:
    - first find the exact integer position of the requested method;
    - then index method, basin, time, and threshold using isel();
    - never rely on xarray .sel(method=...) for metric extraction.

    This fixes the previous error:
        basin_mean_metric is not scalar after squeeze; shape=(2,), size=2
    """
    rows = []

    method_i = _find_coord_index(ds, "method", method)

    times = pd.to_datetime(ds["time"].values)
    basins = [str(v) for v in ds["basin"].values]
    thresholds = [int(v) for v in ds["threshold"].values]

    phase_codes = ds["enso_phase_code"].values.astype(int)
    years = ds["year"].values.astype(int)
    months = ds["month"].values.astype(int)
    enso_index = ds["enso_index"].values.astype(float)

    metric_da = ds["basin_mean_metric"]
    event_da = ds["event"]

    # Safety check for expected dimensions.
    required_metric_dims = {"method", "basin", "time"}
    required_event_dims = {"method", "basin", "time", "threshold"}

    if not required_metric_dims.issubset(set(metric_da.dims)):
        raise ValueError(f"basin_mean_metric dims are unexpected: {metric_da.dims}")
    if not required_event_dims.issubset(set(event_da.dims)):
        raise ValueError(f"event dims are unexpected: {event_da.dims}")

    for bi, basin_name in enumerate(basins):
        for ti, t in enumerate(times):
            metric_value = metric_da.isel(method=method_i, basin=bi, time=ti).values

            row = {
                "method": method,
                "basin": basin_name,
                "time": t,
                "year": int(years[ti]),
                "month": int(months[ti]),
                "phase_code": int(phase_codes[ti]),
                "phase": phase_from_code(int(phase_codes[ti])),
                "enso_index": float(enso_index[ti]),
                "metric": float(_to_scalar(metric_value, "basin_mean_metric")),
            }

            for qi, q in enumerate(thresholds):
                ev_value = event_da.isel(method=method_i, basin=bi, time=ti, threshold=qi).values
                row[f"event_q{q}"] = int(_to_scalar(ev_value, f"event_q{q}"))

            rows.append(row)

    return pd.DataFrame(rows)



def method_display_name(method: str) -> str:
    """
    Journal-style display name for the metric used in the figure.
    """
    if getattr(CFG, "USE_CALENDAR_MONTH_ANOMALY_FOR_FIG2", False):
        return f"{method} anomaly"
    return method



def get_figure_specs() -> List[Dict[str, object]]:
    """
    Final manuscript plotting plan.

    Main text
    ---------
    Figure 2 : MEP95 anomaly + q90

    Supplementary information
    -------------------------
    Figure S1: Rx5day anomaly + q90
    Figure S2: MEP95 anomaly + q95 (sensitivity)

    x-axis styles
    -------------
    - main figure: percentile + basin-specific anomaly values in parentheses
    - supplementary figures: percentile only
    """
    return [
        {
            "figure_id": "Figure2",
            "figure_label": "Figure 2",
            "method": "MEP95",
            "threshold": 90,
            "xaxis_mode": "dual_value",
            "role": "main",
            "title_tag": "MEP95 anomaly",
        },
        {
            "figure_id": "FigureS1",
            "figure_label": "Figure S1",
            "method": "RX5day",
            "threshold": 90,
            "xaxis_mode": "percentile_only",
            "role": "supporting",
            "title_tag": "Rx5day anomaly",
        },
        {
            "figure_id": "FigureS2",
            "figure_label": "Figure S2",
            "method": "MEP95",
            "threshold": 95,
            "xaxis_mode": "percentile_only",
            "role": "supporting",
            "title_tag": "MEP95 anomaly sensitivity",
        },
    ]


def x_axis_unit_label_for_spec(fig_spec: Dict[str, object]) -> str:
    """
    Bottom x-axis unit label following the user's reference figure style:

        Percentile of monthly rain
        (Anomaly value: mm/month)

    For this manuscript:
    - MEP95 is a monthly accumulated extreme-precipitation metric, so the
      anomaly unit is written as mm/month.
    - Rx5day is a monthly maximum 5-day amount, so the anomaly unit is written
      as mm.
    """
    method = str(fig_spec["method"])
    if method == "MEP95":
        return "Percentile of monthly MEP95\n(Anomaly value: mm/month)"
    if method == "RX5day":
        return "Percentile of monthly Rx5day\n(Anomaly value: mm)"
    return "Percentile of monthly metric\n(Anomaly value: mm)"



def displayed_threshold_for_spec(fig_spec: Dict[str, object]) -> int:
    return int(fig_spec["threshold"])


def displayed_threshold_label(fig_spec: Dict[str, object]) -> str:
    q = displayed_threshold_for_spec(fig_spec)
    return f"{q}th event threshold"


def xaxis_mode_for_spec(fig_spec: Dict[str, object]) -> str:
    return str(fig_spec.get("xaxis_mode", "dual_value"))



def add_calendar_month_anomaly_events(df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert raw basin-mean monthly metrics to calendar-month anomalies and then
    recompute q90/q95 events from the anomalies.

    Definition
    ----------
    For each basin and calendar month:

        metric_anomaly = metric_raw - climatological_mean(metric_raw)

    where the climatology is calculated over CFG.ANOMALY_BASELINE_START_YEAR to
    CFG.ANOMALY_BASELINE_END_YEAR.

    Why this is used for Fig. 2
    ---------------------------
    Raw MEP95 is a non-negative accumulated precipitation amount. Therefore its
    CDF cannot naturally show negative departures or a meaningful zero-anomaly
    reference line. The anomaly version centers each basin-calendar-month
    distribution around its climatology, making x = 0 interpretable as the
    climatological baseline.

    Event definition
    ----------------
    q90/q95 events are recomputed after anomaly calculation using the same strict
    basin × calendar-month threshold structure:

        event_q90 = metric_anomaly >= q90(metric_anomaly | basin, calendar month)

    This keeps the RR/FAR calculation seasonally controlled.
    """
    out = df.copy()
    out["metric_raw"] = out["metric"].astype(float)

    baseline = out[
        (out["year"].astype(int) >= int(CFG.ANOMALY_BASELINE_START_YEAR))
        & (out["year"].astype(int) <= int(CFG.ANOMALY_BASELINE_END_YEAR))
    ].copy()

    if baseline.empty:
        raise ValueError(
            "No baseline data available for anomaly calculation. "
            "Check CFG.ANOMALY_BASELINE_START_YEAR and CFG.ANOMALY_BASELINE_END_YEAR."
        )

    clim = (
        baseline
        .groupby(["basin", "month"], as_index=False)["metric_raw"]
        .mean()
        .rename(columns={"metric_raw": "metric_climatology"})
    )

    out = out.merge(clim, on=["basin", "month"], how="left")
    out["metric_anomaly"] = out["metric_raw"] - out["metric_climatology"]

    if out["metric_anomaly"].isna().any():
        missing = out[out["metric_anomaly"].isna()][["basin", "month"]].drop_duplicates()
        raise ValueError(f"Missing climatology for some basin-month pairs:\n{missing}")

    # Keep the plotting/risk pipeline unchanged by using the generic 'metric'
    # column for the anomaly after preserving the raw values.
    out["metric"] = out["metric_anomaly"]
    out["metric_type"] = "calendar-month anomaly"
    out["metric_units"] = "mm"
    out["anomaly_baseline"] = f"{CFG.ANOMALY_BASELINE_START_YEAR}-{CFG.ANOMALY_BASELINE_END_YEAR}"

    # Remove original raw-event columns from the NetCDF so that anomaly events
    # are recomputed cleanly and unambiguously.
    for q in CFG.THRESHOLDS:
        for col in [f"threshold_q{q}", f"event_q{q}"]:
            if col in out.columns:
                out = out.drop(columns=[col])

    out, threshold_df = add_threshold_events(out, method)
    threshold_df["metric_type"] = "calendar-month anomaly"
    threshold_df["anomaly_baseline"] = out["anomaly_baseline"].iloc[0]
    threshold_df["threshold_units"] = "mm"

    return out, threshold_df





def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical cumulative distribution function.

    Returns
    -------
    x : sorted finite values
    y : cumulative probability from 1/n to 1

    This function is used by panels (a-f). It was missing in v12 and caused:
        NameError: name 'ecdf' is not defined
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    v = np.sort(v)

    if len(v) == 0:
        return np.array([]), np.array([])

    y = np.arange(1, len(v) + 1, dtype=float) / float(len(v))
    return v, y


def transform_to_reference_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Kept only for backward compatibility. The current CDF design for panels (a-f)
    uses the real metric value on the x-axis instead of a 0-100 percentile axis.
    """
    ref = np.asarray(reference, dtype=float)
    ref = np.sort(ref[np.isfinite(ref)])
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)

    if len(ref) == 0:
        return out

    good = np.isfinite(values)
    out[good] = np.searchsorted(ref, values[good], side="right") / len(ref) * 100.0
    return out


def _round_step_for_range(xmin: float, xmax: float) -> float:
    """
    Choose a visually reasonable tick step for the actual-value x-axis.
    """
    span = max(float(xmax - xmin), 1e-6)
    raw = span / 5.0
    power = 10 ** np.floor(np.log10(raw))
    scaled = raw / power

    if scaled <= 1.2:
        nice = 1
    elif scaled <= 2.5:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10

    return nice * power


def choose_real_axis_limits(reference: np.ndarray) -> tuple:
    """
    Compute basin-specific real-value x-limits for CDF panels.

    Design logic requested by the user:
    1. Each basin is plotted according to its own real metric range.
    2. All subplot widths remain identical because the panel geometry is fixed.
    3. A dashed x=0 anomaly-reference line is included.
       Therefore the left bound is extended slightly below zero when all values
       are positive, so the zero line is visible inside the panel frame.
    """
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]

    if len(ref) == 0:
        return (-1.0, 1.0)

    xmin = float(np.nanmin(ref))
    xmax = float(np.nanmax(ref))

    if np.isclose(xmin, xmax):
        pad = 0.1 * max(abs(xmax), 1.0)
        xmin -= pad
        xmax += pad
    else:
        pad = 0.05 * (xmax - xmin)
        xmin -= pad
        xmax += pad

    # Make x=0 visible as a reference line.
    if xmin >= 0:
        xmin = -0.03 * max(xmax - xmin, xmax, 1.0)

    return (xmin, xmax)


def choose_real_ticks(xmin: float, xmax: float) -> List[float]:
    """
    Build a sparse, CEE-style real-value x-axis.

    Design target:
    - about 4-5 main ticks;
    - visually clean spacing;
    - keep 0 when it falls in the visible range.
    """
    span = max(float(xmax - xmin), 1e-6)
    target_intervals = 4  # gives about 5 major ticks
    raw_step = span / target_intervals
    step = _round_step_for_range(0.0, raw_step * 5.0)

    # If the initial step still produces too many ticks, increase it.
    while True:
        start = np.floor(xmin / step) * step
        ticks = np.arange(start, xmax + 0.5 * step, step)
        ticks = [float(t) for t in ticks if xmin - 1e-9 <= t <= xmax + 1e-9]
        if len(ticks) <= 5:
            break
        step *= 2.0

    # If there are too few ticks, reduce the step once.
    if len(ticks) < 4:
        step = step / 2.0
        start = np.floor(xmin / step) * step
        ticks = np.arange(start, xmax + 0.5 * step, step)
        ticks = [float(t) for t in ticks if xmin - 1e-9 <= t <= xmax + 1e-9]

    # Keep 0 as a key reference if visible.
    if xmin < 0 < xmax and not np.any(np.isclose(ticks, 0.0)):
        ticks.append(0.0)

    ticks = sorted(set([round(float(t), 10) for t in ticks]))
    return ticks


def basin_overall_thresholds(df: pd.DataFrame, basin: str) -> Dict[int, float]:
    """
    For CDF display only, derive one basin-level q90 and q95 value from the full
    monthly metric distribution of that basin.

    Important note:
    - Fig. 2g still uses the month-specific event definition from the workflow.
    - Panels (a-f) use these pooled basin-level q-values only as visual markers
      on the CDF curves.
    """
    vals = df.loc[df["basin"] == basin, "metric"].astype(float).values
    vals = vals[np.isfinite(vals)]

    out = {}
    for q in [90, 95]:
        out[q] = float(np.nanpercentile(vals, q)) if len(vals) else np.nan
    return out


def style_ax_cee(ax):
    if CFG.GRID:
        ax.grid(True, axis="both", lw=0.35, color="0.86", alpha=0.78)
    ax.tick_params(labelsize=8.5, length=3, width=0.8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.9)



def choose_q_label_positions(qmap: Dict[int, float], xmin: float, xmax: float) -> Dict[int, tuple]:
    """
    Choose cleaner q90/q95 label positions.

    To reduce overlap with the CDF curves:
    - place the labels in the upper part of the panel rather than near the bottom;
    - slightly separate q90 and q95 horizontally when their x values are close.
    """
    span = max(xmax - xmin, 1e-6)
    q90 = qmap.get(90, np.nan)
    q95 = qmap.get(95, np.nan)

    xpos = {}
    ypos = {90: 0.84, 95: 0.93}

    if np.isfinite(q90):
        xpos[90] = q90
    if np.isfinite(q95):
        xpos[95] = q95

    if np.isfinite(q90) and np.isfinite(q95):
        if abs(q95 - q90) < 0.07 * span:
            xpos[90] = q90 - 0.025 * span
            xpos[95] = q95 + 0.025 * span

    out = {}
    for q in [90, 95]:
        qx = xpos.get(q, qmap.get(q, np.nan))
        qx = min(max(qx, xmin + 0.03 * span), xmax - 0.03 * span)
        out[q] = (qx, ypos[q])
    return out





def choose_basin_percent_ticks(reference: np.ndarray) -> List[int]:
    """
    Use the same clean percentile ticks for all A-F panels.

    Corrected user-requested style:
        0 / 20 / 40 / 60 / 80 / 100

    The CDF itself is still plotted in basin-specific percentile space, and the
    actual anomaly values shown in parentheses are still calculated separately
    for each basin.
    """
    return [0, 20, 40, 60, 80, 100]


def bottom_dual_tick_labels(reference: np.ndarray, percent_ticks: Optional[List[int]] = None) -> List[str]:
    """
    Bottom x-axis labels in the requested two-line format:
        percentile
        (basin-specific actual value)

    Example:
        20
        (-37.5)

    The values in parentheses are always calculated from the individual basin's
    own distribution, not from a shared national distribution.
    """
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]

    if percent_ticks is None:
        percent_ticks = choose_basin_percent_ticks(ref)

    if len(ref) == 0:
        return [f"{p}\n(nan)" for p in percent_ticks]

    vals = np.nanpercentile(ref, percent_ticks)
    labels = [f"{int(p)}\n({fmt_value(v)})" for p, v in zip(percent_ticks, vals)]
    return labels


def zero_value_percentile(reference: np.ndarray) -> float:
    """
    Convert the actual value 0 to its percentile-axis location.
    If 0 lies outside the observed range, no line is drawn.
    """
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return np.nan
    if np.nanmin(ref) <= 0 <= np.nanmax(ref):
        return float(transform_to_reference_percentile(np.array([0.0]), ref)[0])
    return np.nan





def plot_cdf_percent_x(ax, df: pd.DataFrame, rrdf: pd.DataFrame, basin: str, method: str,
                       lang: str, panel: str, fig_spec: Dict[str, object], legend=False, top_label=False):
    """
    CDF panels (a-f).

    This version supports the final manuscript plan:
    - Figure 2 (MEP95 anomaly + q90) uses percentile labels with basin-specific
      anomaly values in parentheses.
    - Figure S1 (Rx5day anomaly + q90) and Figure S2 (MEP95 anomaly + q95)
      use percentile labels only.
    """
    bdf = df[df["basin"] == basin].copy()
    ref = bdf["metric"].astype(float).values
    ref = ref[np.isfinite(ref)]

    q_show = displayed_threshold_for_spec(fig_spec)
    xaxis_mode = xaxis_mode_for_spec(fig_spec)

    # subtle tail shading to indicate the extreme tail
    ax.add_patch(
        plt.Rectangle((90, 0), 10, 1, facecolor="#f1f1f1", edgecolor="none", alpha=0.48, zorder=0)
    )

    # CDFs of each ENSO phase in percentile space
    for ph in CFG.PHASES:
        real_x, y = ecdf(bdf.loc[bdf["phase"] == ph, "metric"].values)
        pct_x = transform_to_reference_percentile(real_x, ref)
        if len(pct_x):
            ax.plot(
                pct_x,
                y,
                color=CFG.COLORS[ph],
                lw=CFG.LINEWIDTHS[ph],
                label=PHASE_LABELS["EN"][ph],
                zorder=3,
            )

    # zero-anomaly reference line
    zero_pct = zero_value_percentile(ref)
    if np.isfinite(zero_pct):
        ax.axvline(
            zero_pct,
            label="_nolegend_",
            zorder=1,
            color="0.55",
            linestyle="--",
            linewidth=1.0,
            alpha=0.95,
        )

    # only one threshold line per figure
    q_actual = np.nan
    if len(ref):
        q_actual = float(np.nanpercentile(ref, q_show))

    st = CFG.Q_LINE_STYLE[q_show].copy()
    ax.axvline(q_show, label=displayed_threshold_label(fig_spec), zorder=2, **st)

    # annotate the actual anomaly value above the top frame
    if np.isfinite(q_actual):
        ax.text(
            q_show,
            1.012,
            f"{fmt_value(q_actual)}",
            transform=ax.get_xaxis_transform(),
            color=st.get("color", "0.25"),
            fontsize=8.8,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
            zorder=6,
            clip_on=False,
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    percent_ticks = choose_basin_percent_ticks(ref)
    ax.set_xticks(percent_ticks)

    if xaxis_mode == "percentile_only":
        ax.set_xticklabels([str(int(p)) for p in percent_ticks], fontsize=11.0, fontweight="bold")
    else:
        ax.set_xticklabels(bottom_dual_tick_labels(ref, percent_ticks), fontsize=10.8, fontweight="bold")

    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    ax.set_title(
        f"{panel} {BASIN_SHORT['EN'][basin]}",
        loc="left",
        fontsize=13.8,
        fontweight="bold",
        pad=5,
    )

    style_ax_cee(ax)
    ax.tick_params(axis="both", labelsize=11.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    if legend:
        phase_handles = [
            Line2D([0], [0], color=CFG.COLORS["El Nino"], lw=CFG.LINEWIDTHS["El Nino"], label=PHASE_LABELS["EN"]["El Nino"]),
            Line2D([0], [0], color=CFG.COLORS["La Nina"], lw=CFG.LINEWIDTHS["La Nina"], label=PHASE_LABELS["EN"]["La Nina"]),
            Line2D([0], [0], color=CFG.COLORS["Neutral"], lw=CFG.LINEWIDTHS["Neutral"], label=PHASE_LABELS["EN"]["Neutral"]),
        ]
        # Per the latest layout request, remove the q90 explanation row from
        # the legend while keeping the rest unchanged. For figures using q95,
        # retain the threshold legend entry.
        if q_show != 90:
            phase_handles.append(
                Line2D(
                    [0], [0],
                    color=st.get("color", "0.25"),
                    lw=st.get("linewidth", 1.2),
                    linestyle=st.get("linestyle", "--"),
                    label=displayed_threshold_label(fig_spec),
                )
            )
        ax.legend(
            handles=phase_handles,
            frameon=False,
            fontsize=10.6,
            loc="upper left",
            handlelength=2.5,
            ncol=1,
            title=None,
        )

def safe_yerr(center: np.ndarray, low: np.ndarray, high: np.ndarray) -> Optional[np.ndarray]:
    """
    Build a safe asymmetric y-error array for matplotlib.errorbar.

    This fixes:
        NameError: name 'safe_yerr' is not defined

    It also prevents negative or non-finite CI distances from breaking the plot.
    """
    center = np.asarray(center, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)

    if center.size == 0:
        return None

    if np.all(~np.isfinite(low)) or np.all(~np.isfinite(high)):
        return None

    lower = center - low
    upper = high - center

    yerr = np.vstack([lower, upper])
    yerr = np.where(np.isfinite(yerr) & (yerr >= 0), yerr, 0.0)
    return yerr





def plot_g_cee(ax_rr, rrdf: pd.DataFrame, method: str, lang: str, fig_spec: Dict[str, object]):
    """
    Panel (g): basin-level RR and FAR.

    The risk threshold is linked to the current figure specification:
    - Figure 2   : MEP95 anomaly + q90
    - Figure S1  : Rx5day anomaly + q90
    - Figure S2  : MEP95 anomaly + q95
    """
    q = displayed_threshold_for_spec(fig_spec)
    sub = rrdf[rrdf["threshold"] == q].copy()
    ax_far = ax_rr.twinx()

    x = np.arange(len(CFG.BASIN_ORDER))
    offset = {"El Nino": -0.16, "La Nina": 0.16}

    for ph in ["El Nino", "La Nina"]:
        p = sub[sub["phase"] == ph].set_index("basin").reindex(CFG.BASIN_ORDER).reset_index()
        xp = x + offset[ph]

        rr = p["RR"].astype(float).values
        lo = p["RR_bootstrap_low"].astype(float).values if "RR_bootstrap_low" in p else np.full_like(rr, np.nan)
        hi = p["RR_bootstrap_high"].astype(float).values if "RR_bootstrap_high" in p else np.full_like(rr, np.nan)

        ax_rr.errorbar(
            xp,
            rr,
            yerr=safe_yerr(rr, lo, hi),
            fmt="o-",
            lw=1.55,
            markersize=5.3,
            capsize=2.6,
            color=CFG.COLORS[ph],
            ecolor=CFG.COLORS[ph],
            zorder=4,
        )

        far = p["FAR"].astype(float).values
        ax_far.plot(
            xp,
            far,
            "^--",
            lw=1.35,
            markersize=5.1,
            color=CFG.COLORS[ph],
            alpha=0.88,
            zorder=3,
        )

        invalid = ~np.isfinite(far)
        if np.any(invalid):
            ax_far.plot(
                xp[invalid],
                np.zeros(invalid.sum()),
                "x",
                color=CFG.COLORS[ph],
                markersize=4.6,
                alpha=0.76,
            )

    ax_rr.axhline(1, color="0.35", lw=0.95, ls="--")
    ax_far.axhline(0, color="0.45", lw=0.85, ls=":")

    # Keep the panel identifier even though the panel title is removed.
    ax_rr.text(
        0.0,
        1.02,
        "(g)",
        transform=ax_rr.transAxes,
        ha="left",
        va="bottom",
        fontsize=14.0,
        fontweight="bold",
        clip_on=False,
    )

    # Remove the panel-(g) title to avoid any overlap with the lower
    # x-axis labels/units of panels (d-f). The panel identity is conveyed by
    # the manuscript caption and the panel ordering, so an embedded title is
    # not required here.
    ax_rr.set_ylabel("RR", fontsize=12.8, fontweight="bold", labelpad=6)
    ax_far.set_ylabel("FAR", fontsize=12.8, fontweight="bold", labelpad=6)

    ax_rr.set_xticks(x)
    ax_rr.set_xticklabels(
        [BASIN_SHORT["EN"][b] for b in CFG.BASIN_ORDER],
        rotation=0,
        ha="center",
        fontstyle="normal",
        fontfamily="Times New Roman",
        fontweight="bold",
    )

    if CFG.RR_YLIM:
        ax_rr.set_ylim(CFG.RR_YLIM)
    if CFG.FAR_YLIM:
        ax_far.set_ylim(CFG.FAR_YLIM)

    if CFG.GRID:
        ax_rr.grid(True, axis="y", lw=0.35, color="0.86", alpha=0.78)

    ax_rr.tick_params(labelsize=11.4, length=3, width=0.8)
    ax_far.tick_params(axis="y", labelsize=11.4, length=3, width=0.8)
    for lbl in ax_rr.get_xticklabels() + ax_rr.get_yticklabels() + ax_far.get_yticklabels():
        lbl.set_fontweight("bold")

    for sp in ax_rr.spines.values():
        sp.set_linewidth(0.9)
    for sp in ax_far.spines.values():
        sp.set_linewidth(0.9)

    legend_handles = [
        Line2D([0], [0], color=CFG.COLORS["El Nino"], marker="o", lw=1.55, linestyle="-", label="El Niño RR"),
        Line2D([0], [0], color=CFG.COLORS["El Nino"], marker="^", lw=1.35, linestyle="--", label="El Niño FAR"),
        Line2D([0], [0], color=CFG.COLORS["La Nina"], marker="o", lw=1.55, linestyle="-", label="La Niña RR"),
        Line2D([0], [0], color=CFG.COLORS["La Nina"], marker="^", lw=1.35, linestyle="--", label="La Niña FAR"),
    ]
    ax_rr.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        frameon=False,
        fontsize=10.8,
        ncol=4,
        handlelength=2.2,
        columnspacing=1.2,
    )


def make_figure_from_nc_df(df: pd.DataFrame, rrdf: pd.DataFrame, fig_spec: Dict[str, object], lang: str) -> Dict[str, str]:
    set_font(lang)

    method = str(fig_spec["method"])
    xaxis_mode = xaxis_mode_for_spec(fig_spec)
    figure_id = str(fig_spec["figure_id"])

    fig = plt.figure(figsize=CFG.FIG_SIZE, dpi=CFG.FIG_DPI)
    gs = fig.add_gridspec(
        3,
        6,
        # Tighten the spacing among panels (a-f) slightly, while preserving
        # enough space before panel (g).
        height_ratios=[1, 1, 0.95],
        hspace=0.38,
        wspace=0.26,
    )

    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
    ]
    axg = fig.add_subplot(gs[2, 0:6])

    for i, basin in enumerate(CFG.BASIN_ORDER):
        plot_cdf_percent_x(
            axes[i],
            df,
            rrdf,
            basin,
            method,
            lang,
            CFG.PANEL_LABELS[i],
            fig_spec=fig_spec,
            legend=(i == 0),
            top_label=(i in [0, 1, 2]),
        )
        if i in [0, 3]:
            axes[i].set_ylabel("CDF", fontsize=12.4, fontweight="bold")
        if i in [3, 4, 5]:
            if xaxis_mode == "percentile_only":
                axes[i].set_xlabel(
                    x_axis_unit_label_for_spec(fig_spec),
                    fontsize=11.4,
                    fontweight="bold",
                )
            else:
                axes[i].set_xlabel(
                    x_axis_unit_label_for_spec(fig_spec),
                    fontsize=11.2,
                    fontweight="bold",
                )

    plot_g_cee(axg, rrdf, method, lang, fig_spec=fig_spec)

    # Move panel (g) a bit further downward to ensure a clean separation
    # from the x-axis labels/units of panels (d-f).
    pos = axg.get_position()
    axg.set_position([pos.x0, pos.y0 - 0.012, pos.width, pos.height])

    # No overall figure title embedded. Use manuscript caption instead.
    # Keep a slightly roomier layout so panel (g) stays visually separated
    # from the lower x-axis labels of panels (d-f).
    fig.subplots_adjust(
        left=0.055, right=0.985, top=0.965, bottom=0.060,
        hspace=0.38, wspace=0.26,
    )

    basename = f"{figure_id}_{method}_anomaly_q{displayed_threshold_for_spec(fig_spec)}_{xaxis_mode}_EN_reference_unit_label"
    return save_fig(fig, basename)


def plot_all_from_nc(nc_path: Path):
    ensure_output_dirs()
    print("=" * 88)
    print("Step 2: read NetCDF, convert to anomalies, and make manuscript figures")
    print("=" * 88)

    ds = xr.open_dataset(nc_path)

    try:
        all_df, all_rr, all_thresholds, all_raw = {}, {}, {}, {}
        fig_rows = []

        # First compute and export all method-level datasets.
        for method in CFG.METHODS:
            raw_df = nc_to_method_dataframe(ds, method)

            if getattr(CFG, "USE_CALENDAR_MONTH_ANOMALY_FOR_FIG2", False):
                df, threshold_df = add_calendar_month_anomaly_events(raw_df, method)
            else:
                df = raw_df.copy()
                threshold_df = pd.DataFrame()

            risk = rr_far(df, method)

            all_raw[method] = raw_df
            all_df[method] = df
            all_rr[method] = risk
            all_thresholds[method] = threshold_df

            raw_df.to_csv(
                TABLE_DIR / f"{method}_raw_monthly_metrics_from_NC.csv",
                index=False,
                encoding="utf-8-sig",
            )
            df.to_csv(
                TABLE_DIR / f"{method}_monthly_anomaly_events_for_plotting.csv",
                index=False,
                encoding="utf-8-sig",
            )
            risk.to_csv(
                TABLE_DIR / f"{method}_anomaly_RR_FAR_all_thresholds.csv",
                index=False,
                encoding="utf-8-sig",
            )
            if not threshold_df.empty:
                threshold_df.to_csv(
                    TABLE_DIR / f"{method}_anomaly_q90_q95_thresholds.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

        # Then generate manuscript figures following the final plan.
        for fig_spec in get_figure_specs():
            method = str(fig_spec["method"])
            for lang in ["EN"]:
                info = make_figure_from_nc_df(all_df[method], all_rr[method], fig_spec, lang)
                info["figure_id"] = str(fig_spec["figure_id"])
                info["figure_label"] = str(fig_spec["figure_label"])
                info["method"] = method
                info["display_method"] = method_display_name(method)
                info["threshold"] = int(fig_spec["threshold"])
                info["xaxis_mode"] = str(fig_spec["xaxis_mode"])
                info["language"] = lang
                info["metric_type"] = "calendar-month anomaly" if getattr(CFG, "USE_CALENDAR_MONTH_ANOMALY_FOR_FIG2", False) else "raw metric"
                info["role"] = str(fig_spec["role"])
                info["netcdf_used"] = str(nc_path)
                fig_rows.append(info)

        fig_report = pd.DataFrame(fig_rows)
        fig_report.to_csv(FIG_DIR / "manuscript_figure_file_report.csv", index=False, encoding="utf-8-sig")

        xlsx = TABLE_DIR / "manuscript_figure2_supporting_tables.xlsx"
        readme = pd.DataFrame([
            {"item": "workflow", "value": "Raw MEP95/Rx5day are saved in NetCDF; figures are drawn from calendar-month anomalies."},
            {"item": "netcdf_used", "value": str(nc_path)},
            {"item": "anomaly_definition", "value": "metric_anomaly = raw basin-mean monthly metric - basin-calendar-month climatology."},
            {"item": "baseline", "value": f"{CFG.ANOMALY_BASELINE_START_YEAR}-{CFG.ANOMALY_BASELINE_END_YEAR}"},
            {"item": "probability", "value": "Empirical event probability P = event_count / valid_sample_count for Figure 2, Figure S1 and Figure S2; no Haldane/continuity correction."},
            {"item": "RR", "value": "RR = P_phase / P_neutral. If phase event count=0 and Neutral probability>0, RR=0; if Neutral probability=0, RR is undefined (NaN)."},
            {"item": "FAR", "value": "FAR = 1 - 1/RR and is interpreted/exported only when finite RR > 1."},
            {"item": "RR_bootstrap", "value": "Month-preserving bootstrap CI uses the same empirical RR estimator; bootstrap replicates with Neutral event probability=0 are skipped because RR is undefined."},
            {"item": "Figure2", "value": "MEP95 anomaly + q90 (main text); x-axis follows the reference style: percentile ticks with anomaly values in parentheses and a unit line of Anomaly value: mm/month."},
            {"item": "FigureS1", "value": "Rx5day anomaly + q90 (supporting); x-axis uses basin-specific percentile ticks with an anomaly-unit line."},
            {"item": "FigureS2", "value": "MEP95 anomaly + q95 sensitivity (supporting); x-axis uses basin-specific percentile ticks with an anomaly-unit line."},
            {"item": "zero_line", "value": "The grey dashed line marks anomaly = 0."},
            {"item": "q_lines", "value": "q90 is a red dashed line and q95 is a blue dashed line; only the threshold relevant to the current figure is displayed."},
            {"item": "panel_g", "value": "Panel (g) summarizes RR and FAR using the same threshold shown in panels (a-f)."},
        ])

        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            readme.to_excel(writer, sheet_name="README", index=False)
            fig_report.to_excel(writer, sheet_name="Figure_files", index=False)
            for method in CFG.METHODS:
                all_raw[method].to_excel(writer, sheet_name=f"{method}_raw", index=False)
                all_df[method].to_excel(writer, sheet_name=f"{method}_anomaly", index=False)
                all_rr[method].to_excel(writer, sheet_name=f"{method}_RR_FAR", index=False)
                if not all_thresholds[method].empty:
                    all_thresholds[method].to_excel(writer, sheet_name=f"{method}_q_thresholds", index=False)

        print(fig_report.to_string(index=False))

    finally:
        try:
            ds.close()
        except Exception:
            pass


def main():
    ensure_output_dirs()
    print("=" * 88)
    print("CEE manuscript figures: Figure 2 (MEP95+q90), Figure S1 (Rx5day+q90), Figure S2 (MEP95+q95 sensitivity)")
    print("RR estimator: empirical P=count/n; no Haldane correction in the Figure 2 series")
    print("=" * 88)
    print(f"Output folder: {CFG.OUT_DIR}")
    print(f"NetCDF file  : {NC_FILE}")
    print("=" * 88)

    if PLOT_ONLY_FROM_EXISTING_NC:
        if not NC_FILE.exists():
            raise FileNotFoundError(f"Existing NetCDF not found: {NC_FILE}")
        nc_path = NC_FILE
    else:
        if NC_FILE.exists() and not REBUILD_NC:
            print(f"Using existing NetCDF: {NC_FILE}")
            nc_path = NC_FILE
        else:
            nc_path = build_nc_first()

    plot_all_from_nc(nc_path)

    print("=" * 88)
    print("Finished.")
    print(f"NetCDF : {NC_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIG_DIR}")
    print("=" * 88)


if __name__ == "__main__":
    main()
