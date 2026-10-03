# -*- coding: utf-8 -*-

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# PORTABLE PROJECT PATHS
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "outputs"

# ============================================================
# 1. User settings
# ============================================================

ERSST_DIR = DATA_ROOT / "ersst_v6"
OUT_DIR = OUTPUT_ROOT / "01_enso_classification"

START_YEAR = 1961
END_YEAR = 2023

ARTICLE_CLIM_START_YEAR = 1961
ARTICLE_CLIM_END_YEAR = 2023

NOAA_CLIM_START_YEAR = 1991
NOAA_CLIM_END_YEAR = 2020

MAIN_THRESHOLD = 0.5
SENS_THRESHOLD = 1.0
NOAA_MIN_CONSECUTIVE_MONTHS = 5

INCLUDE_ONI_BOUNDARY_MONTHS = True

AUTO_DOWNLOAD_MISSING_ERSST = True
NOAA_ERSST_V6_BASE_URL = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-extended-reconstructed/"
    "v6/access"
)

USE_AREA_WEIGHTED_NINO34 = False

DPI = 600
EXPORT_TIFF = True


# ============================================================
# 2. Output folders and plotting style
# ============================================================

TABLE_DIR = OUT_DIR / "01_tables"
PNG_DIR = OUT_DIR / "02_figures_png"
TIFF_DIR = OUT_DIR / "03_figures_tiff"
LOG_DIR = OUT_DIR / "04_logs"

for folder in [OUT_DIR, TABLE_DIR, PNG_DIR, TIFF_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    try:
        import matplotlib.font_manager as fm
        fonts = {f.name for f in fm.fontManager.ttflist}
        font_name = "Times New Roman" if "Times New Roman" in fonts else "DejaVu Sans"
    except Exception:
        font_name = "DejaVu Sans"

    plt.rcParams.update({
        "font.family": font_name,
        "font.size": 10,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


configure_matplotlib()

MONTH_LABELS = list("JFMAMJJASOND")
PHASE_LABELS = {"La Nina": "La Nina", "Neutral": "Neutral", "El Nino": "El Nino"}
PHASE_COLORS = {"La Nina": "blue", "Neutral": "black", "El Nino": "red"}

LEGEND_HANDLELENGTH = 3.2  # longer legend line samples
FIG_WIDTH = 7.2
FIG_HEIGHT = 4.8


# ============================================================
# 3. Time and file utilities
# ============================================================

def ym_to_timestamp(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1)


def add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return ts + pd.DateOffset(months=n)


def required_ersst_months() -> List[pd.Timestamp]:
    start = ym_to_timestamp(START_YEAR, 1)
    end = ym_to_timestamp(END_YEAR, 12)

    if INCLUDE_ONI_BOUNDARY_MONTHS:
        start = add_months(start, -1)
        end = add_months(end, 1)

    return list(pd.date_range(start=start, end=end, freq="MS"))


def ersst_filename(ts: pd.Timestamp) -> str:
    return f"ersst.v6.{ts.year}{ts.month:02d}.nc"


def build_ersst_inventory() -> pd.DataFrame:
    rows = []
    for ts in required_ersst_months():
        filename = ersst_filename(ts)
        path = ERSST_DIR / filename
        exists = path.exists()
        rows.append({
            "year": ts.year,
            "month": ts.month,
            "time": ts,
            "filename": filename,
            "path": str(path),
            "exists": bool(exists),
            "size_bytes": path.stat().st_size if exists else np.nan,
            "used_for_main_article_method": START_YEAR <= ts.year <= END_YEAR,
            "used_for_noaa_sensitivity": True,
        })
    return pd.DataFrame(rows)


# ============================================================
# 4. Optional ERSST downloader
# ============================================================

def create_requests_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=5, pool_maxsize=5)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def ensure_ersst_files(inventory: pd.DataFrame) -> pd.DataFrame:
    missing = inventory[~inventory["exists"]].copy()

    if missing.empty:
        print("All required ERSST v6 files are available.")
        return inventory

    missing.to_csv(LOG_DIR / "missing_ersst_before_download.csv", index=False, encoding="utf-8-sig")
    print("Missing ERSST v6 files detected:")
    print(missing[["year", "month", "filename"]].to_string(index=False))

    if not AUTO_DOWNLOAD_MISSING_ERSST:
        raise FileNotFoundError(
            f"Missing ERSST files. See: {LOG_DIR / 'missing_ersst_before_download.csv'}"
        )

    print("\nAUTO_DOWNLOAD_MISSING_ERSST=True. Downloading missing files...")
    ERSST_DIR.mkdir(parents=True, exist_ok=True)
    session = create_requests_session()

    report_rows = []
    for _, rec in missing.iterrows():
        filename = rec["filename"]
        url = f"{NOAA_ERSST_V6_BASE_URL}/{filename}"
        out_path = ERSST_DIR / filename
        tmp_path = out_path.with_suffix(".nc.part")

        success = False
        message = ""

        for attempt in range(1, 6):
            try:
                print(f"  Downloading {filename} ({attempt}/5) ...", end=" ", flush=True)
                response = session.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=(30, 150),
                    stream=True,
                )
                response.raise_for_status()

                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)

                tmp_path.replace(out_path)
                size = out_path.stat().st_size
                if size < 100000:
                    raise ValueError(f"downloaded file too small: {size} bytes")

                success = True
                message = f"downloaded {size} bytes"
                print("OK")
                break

            except Exception as exc:
                message = str(exc)
                print(f"failed: {message[:90]}")
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                if attempt < 5:
                    time.sleep(5 * attempt)

        report_rows.append({"filename": filename, "url": url, "success": success, "message": message})

    pd.DataFrame(report_rows).to_csv(LOG_DIR / "ersst_download_report.csv", index=False, encoding="utf-8-sig")

    inventory_after = build_ersst_inventory()
    if not inventory_after["exists"].all():
        still_missing = inventory_after[~inventory_after["exists"]]
        still_missing.to_csv(LOG_DIR / "still_missing_ersst.csv", index=False, encoding="utf-8-sig")
        raise FileNotFoundError(
            f"Some ERSST files are still missing. See: {LOG_DIR / 'still_missing_ersst.csv'}"
        )

    return inventory_after


# ============================================================
# 5. NetCDF reading and Niño3.4 extraction
# ============================================================

def detect_nc_engine() -> Optional[str]:
    try:
        import netCDF4  # noqa: F401
        return "netcdf4"
    except Exception:
        try:
            import h5netcdf  # noqa: F401
            return "h5netcdf"
        except Exception:
            return None


def open_dataset_safe(path: Path, engine: Optional[str]) -> xr.Dataset:
    if engine:
        return xr.open_dataset(path, decode_times=True, engine=engine)
    return xr.open_dataset(path, decode_times=True)


def get_coord_name(da: xr.DataArray, candidates: Iterable[str]) -> str:
    candidates_lower = {c.lower() for c in candidates}

    for c in da.coords:
        if c.lower() in candidates_lower:
            return c

    for d in da.dims:
        if d.lower() in candidates_lower:
            return d

    raise ValueError(
        f"Cannot identify coordinate among {candidates}. "
        f"coords={list(da.coords)}, dims={list(da.dims)}"
    )


def select_lat(da: xr.DataArray, lat_name: str, lat_min: float, lat_max: float) -> xr.DataArray:
    vals = da[lat_name].values
    if vals[0] <= vals[-1]:
        return da.sel({lat_name: slice(lat_min, lat_max)})
    return da.sel({lat_name: slice(lat_max, lat_min)})


def select_lon(da: xr.DataArray, lon_name: str, lon_min: float, lon_max: float) -> xr.DataArray:
    vals = da[lon_name].values
    if vals[0] <= vals[-1]:
        return da.sel({lon_name: slice(lon_min, lon_max)})
    return da.sel({lon_name: slice(lon_max, lon_min)})


def spatial_mean_nino34(da: xr.DataArray, lat_name: str, lon_name: str) -> float:
    if USE_AREA_WEIGHTED_NINO34:
        try:
            weights = np.cos(np.deg2rad(da[lat_name]))
            out = da.weighted(weights).mean(dim=[lat_name, lon_name], skipna=True)
            return float(out.values)
        except Exception:
            pass
    return float(da.mean(dim=[lat_name, lon_name], skipna=True).values)


def extract_nino34_sst(path: Path, year: int, month: int, engine: Optional[str]) -> Dict:
    ds = open_dataset_safe(path, engine)

    try:
        if "sst" not in ds.variables:
            raise KeyError(f"sst not found in {path.name}; available variables: {list(ds.data_vars)}")

        sst = ds["sst"]
        lat_name = get_coord_name(sst, ["lat", "latitude", "y"])
        lon_name = get_coord_name(sst, ["lon", "longitude", "x"])

        if "lev" in sst.dims:
            try:
                lev_vals = sst["lev"].values
                surface_lev = lev_vals[np.argmin(np.abs(lev_vals))]
                sst = sst.sel(lev=surface_lev)
            except Exception:
                sst = sst.isel(lev=0)

        if "time" in sst.dims:
            sst = sst.isel(time=0)

        # Niño3.4: 5S-5N, 170W-120W.
        if np.nanmax(sst[lon_name].values) > 180:
            lon_min, lon_max = 190, 240
        else:
            lon_min, lon_max = -170, -120

        region = select_lat(sst, lat_name, -5, 5)
        region = select_lon(region, lon_name, lon_min, lon_max)

        if region.size == 0:
            raise ValueError(f"Niño3.4 region is empty for {path.name}")

        return {
            "year": year,
            "month": month,
            "time": pd.Timestamp(year=year, month=month, day=1),
            "sst": spatial_mean_nino34(region, lat_name, lon_name),
            "source_file": path.name,
        }

    finally:
        ds.close()


def load_nino34_ersst() -> Tuple[pd.DataFrame, pd.DataFrame]:
    inventory = ensure_ersst_files(build_ersst_inventory())
    engine = detect_nc_engine()
    print(f"NetCDF engine: {engine or 'xarray default'}")

    rows = []
    failures = []

    for i, rec in inventory.iterrows():
        try:
            rows.append(
                extract_nino34_sst(
                    path=Path(rec["path"]),
                    year=int(rec["year"]),
                    month=int(rec["month"]),
                    engine=engine,
                )
            )
        except Exception as exc:
            failures.append({"filename": rec["filename"], "error": str(exc)})

        if (i + 1) % 100 == 0 or i == len(inventory) - 1:
            print(f"  Processed ERSST {i + 1}/{len(inventory)}")

    if failures:
        pd.DataFrame(failures).to_csv(LOG_DIR / "ersst_read_failures.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError(f"ERSST read failures detected. See: {LOG_DIR / 'ersst_read_failures.csv'}")

    nino = pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)
    return nino, inventory


# ============================================================
# 6. ENSO calculations
# ============================================================

def subset_main_period(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()
    return out.sort_values(["year", "month"]).reset_index(drop=True)


def polynomial_fit_1d(y: pd.Series, degree: int) -> pd.Series:
    s = pd.to_numeric(y, errors="coerce").astype(float)
    valid = s.notna()
    fit = pd.Series(np.nan, index=s.index, dtype=float)

    if valid.sum() < degree + 1:
        return fit

    x = np.arange(len(s), dtype=float)
    coeff = np.polyfit(x[valid], s[valid], deg=degree)
    fit.loc[:] = np.polyval(coeff, x)
    return fit


def classify_monthly_threshold(df: pd.DataFrame, index_col: str, threshold: float, phase_col: str) -> pd.DataFrame:
    out = df.copy()
    out[phase_col] = "Neutral"
    out.loc[out[index_col] >= threshold, phase_col] = "El Nino"
    out.loc[out[index_col] <= -threshold, phase_col] = "La Nina"
    return out


def label_contiguous_events(df: pd.DataFrame, phase_col: str, event_col: str, label_prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[event_col] = "No Event"

    event_id = 0
    current_phase = None
    start_i = None
    idxs = list(out.index)

    for pos, idx in enumerate(idxs + [None]):
        phase = "END" if idx is None else out.loc[idx, phase_col]

        if phase in ["El Nino", "La Nina"]:
            if current_phase is None:
                current_phase = phase
                start_i = idx
            elif phase != current_phase:
                event_id += 1
                end_i = idxs[pos - 1]
                sy = int(out.loc[start_i, "year"])
                sm = int(out.loc[start_i, "month"])
                ey = int(out.loc[end_i, "year"])
                em = int(out.loc[end_i, "month"])
                label = f"{PHASE_LABELS[current_phase]} {label_prefix} {event_id} ({sy}-{sm:02d} to {ey}-{em:02d})"
                out.loc[start_i:end_i, event_col] = label
                current_phase = phase
                start_i = idx
        else:
            if current_phase is not None:
                event_id += 1
                end_i = idxs[pos - 1]
                sy = int(out.loc[start_i, "year"])
                sm = int(out.loc[start_i, "month"])
                ey = int(out.loc[end_i, "year"])
                em = int(out.loc[end_i, "month"])
                label = f"{PHASE_LABELS[current_phase]} {label_prefix} {event_id} ({sy}-{sm:02d} to {ey}-{em:02d})"
                out.loc[start_i:end_i, event_col] = label
                current_phase = None
                start_i = None

    return out


def classify_noaa5_run_length(
    df: pd.DataFrame,
    index_col: str,
    threshold: float,
    phase_col: str,
    event_col: str,
    min_run: int = NOAA_MIN_CONSECUTIVE_MONTHS,
) -> pd.DataFrame:
    out = df.copy()
    out[phase_col] = "Neutral"
    out[event_col] = "No Event"

    valid_idx = out[out[index_col].notna()].index.to_list()
    values = out.loc[valid_idx, index_col].to_numpy(dtype=float)

    event_id = 0
    i = 0
    n = len(values)

    while i < n:
        if values[i] >= threshold:
            sign = "El Nino"
            start = i
            while i < n and values[i] >= threshold:
                i += 1
            end = i - 1
        elif values[i] <= -threshold:
            sign = "La Nina"
            start = i
            while i < n and values[i] <= -threshold:
                i += 1
            end = i - 1
        else:
            i += 1
            continue

        length = end - start + 1
        original_start = valid_idx[start]
        original_end = valid_idx[end]

        if length >= min_run:
            event_id += 1
            sy = int(out.loc[original_start, "year"])
            sm = int(out.loc[original_start, "month"])
            ey = int(out.loc[original_end, "year"])
            em = int(out.loc[original_end, "month"])
            label = f"{PHASE_LABELS[sign]} NOAA-style event {event_id} ({sy}-{sm:02d} to {ey}-{em:02d})"
            out.loc[original_start:original_end, phase_col] = sign
            out.loc[original_start:original_end, event_col] = label
        else:
            out.loc[original_start:original_end, event_col] = f"Below duration criterion ({length} months)"

    return out


def add_article_main_and_sensitivities(nino_ext: pd.DataFrame) -> pd.DataFrame:
    df = subset_main_period(nino_ext)

    clim_src = df[(df["year"] >= ARTICLE_CLIM_START_YEAR) & (df["year"] <= ARTICLE_CLIM_END_YEAR)]
    monthly_clim = clim_src.groupby("month")["sst"].mean()

    df["nino34_clim_1961_2023"] = df["month"].map(monthly_clim)
    df["nino34_anom"] = df["sst"] - df["nino34_clim_1961_2023"]

    df["nino34_linear_fit"] = polynomial_fit_1d(df["nino34_anom"], degree=1)
    df["nino34_quadratic_fit"] = polynomial_fit_1d(df["nino34_anom"], degree=2)
    df["nino34_detrended"] = df["nino34_anom"] - df["nino34_quadratic_fit"]

    df["nino34_30yr_mean_constant_beginning"] = df["nino34_anom"].rolling(window=360, min_periods=300).mean()
    first30_mean = df[(df["year"] >= START_YEAR) & (df["year"] <= START_YEAR + 29)]["nino34_anom"].mean()
    df.loc[df["nino34_30yr_mean_constant_beginning"].isna(), "nino34_30yr_mean_constant_beginning"] = first30_mean
    df["nino34_anom_30yr"] = df["nino34_anom"] - df["nino34_30yr_mean_constant_beginning"]

    df = classify_monthly_threshold(df, "nino34_detrended", MAIN_THRESHOLD, "main_phase")
    df = label_contiguous_events(df, "main_phase", "main_event", "monthly event")

    df = classify_monthly_threshold(df, "nino34_anom", MAIN_THRESHOLD, "phase_no_detrend")
    df = label_contiguous_events(df, "phase_no_detrend", "event_no_detrend", "no-detrend event")

    df = classify_monthly_threshold(df, "nino34_detrended", SENS_THRESHOLD, "phase_1c")
    df = label_contiguous_events(df, "phase_1c", "event_1c", "1C threshold event")

    df = classify_monthly_threshold(df, "nino34_anom_30yr", MAIN_THRESHOLD, "phase_30yr")
    df = label_contiguous_events(df, "phase_30yr", "event_30yr", "30-year mean removal event")

    return df


def add_noaa_sensitivity(nino_ext: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    ext = nino_ext.copy().sort_values(["year", "month"]).reset_index(drop=True)
    clim_src = ext[(ext["year"] >= NOAA_CLIM_START_YEAR) & (ext["year"] <= NOAA_CLIM_END_YEAR)]
    monthly_clim = clim_src.groupby("month")["sst"].mean()

    ext["noaa_clim_1991_2020"] = ext["month"].map(monthly_clim)
    ext["noaa_nino34_anom"] = ext["sst"] - ext["noaa_clim_1991_2020"]
    ext["noaa_oni"] = ext["noaa_nino34_anom"].rolling(window=3, center=True, min_periods=3).mean()

    noaa_main = subset_main_period(ext)[["year", "month", "noaa_clim_1991_2020", "noaa_nino34_anom", "noaa_oni"]].copy()
    noaa_main["noaa_oni_quadratic_fit"] = polynomial_fit_1d(noaa_main["noaa_oni"], degree=2)
    noaa_main["noaa_oni_detrended"] = noaa_main["noaa_oni"] - noaa_main["noaa_oni_quadratic_fit"]

    noaa_main = classify_noaa5_run_length(
        noaa_main,
        index_col="noaa_oni_detrended",
        threshold=MAIN_THRESHOLD,
        phase_col="noaa_phase",
        event_col="noaa_event",
    )

    return main.merge(noaa_main, on=["year", "month"], how="left")


def build_all_enso() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nino_ext, inventory = load_nino34_ersst()
    main = add_article_main_and_sensitivities(nino_ext)
    main = add_noaa_sensitivity(nino_ext, main)
    return main, inventory


# ============================================================
# 7. Tables
# ============================================================

def monthly_counts(df: pd.DataFrame, phase_col: str, definition: str, index_col: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for m in range(1, 13):
        sub = df[df["month"] == m].copy()
        if index_col is not None:
            sub = sub[sub[index_col].notna()]

        rows.append({
            "definition": definition,
            "phase_col": phase_col,
            "month": m,
            "month_label": MONTH_LABELS[m - 1],
            "La Nina": int((sub[phase_col] == "La Nina").sum()),
            "Neutral": int((sub[phase_col] == "Neutral").sum()),
            "El Nino": int((sub[phase_col] == "El Nino").sum()),
            "Total_valid": int(len(sub)),
        })

    return pd.DataFrame(rows)


def event_episodes(df: pd.DataFrame, phase_col: str, event_col: str, index_col: str, definition: str) -> pd.DataFrame:
    rows = []
    ev = df[df[phase_col].isin(["El Nino", "La Nina"])].copy()

    if ev.empty:
        return pd.DataFrame(columns=[
            "definition", "event_label", "phase", "start_year", "start_month",
            "end_year", "end_month", "duration_months", "mean_index", "max_abs_index"
        ])

    for label, sub in ev.groupby(event_col):
        sub = sub.sort_values(["year", "month"])
        rows.append({
            "definition": definition,
            "event_label": label,
            "phase": PHASE_LABELS[sub[phase_col].iloc[0]],
            "start_year": int(sub["year"].iloc[0]),
            "start_month": int(sub["month"].iloc[0]),
            "end_year": int(sub["year"].iloc[-1]),
            "end_month": int(sub["month"].iloc[-1]),
            "duration_months": int(len(sub)),
            "mean_index": float(sub[index_col].mean()),
            "max_abs_index": float(sub[index_col].abs().max()),
        })

    return pd.DataFrame(rows)


def build_summary_tables(enso: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    definitions = [
        ("manuscript_quadratic_detrend_0.5C", "main_phase", "main_event", "nino34_detrended"),
        ("no_detrend_0.5C", "phase_no_detrend", "event_no_detrend", "nino34_anom"),
        ("threshold_1.0C", "phase_1c", "event_1c", "nino34_detrended"),
        ("30yr_running_mean_removal", "phase_30yr", "event_30yr", "nino34_anom_30yr"),
        ("NOAA_style_ONI_event", "noaa_phase", "noaa_event", "noaa_oni_detrended"),
    ]

    monthly_list = []
    episode_list = []
    summary_rows = []

    for definition, phase_col, event_col, index_col in definitions:
        monthly_list.append(monthly_counts(enso, phase_col, definition, index_col))
        episode_list.append(event_episodes(enso, phase_col, event_col, index_col, definition))

        valid = enso[enso[index_col].notna()]
        vc = valid[phase_col].value_counts().to_dict()

        summary_rows.append({
            "definition": definition,
            "index_col": index_col,
            "phase_col": phase_col,
            "event_col": event_col,
            "valid_months": int(len(valid)),
            "La_Nina_months": int(vc.get("La Nina", 0)),
            "Neutral_months": int(vc.get("Neutral", 0)),
            "El_Nino_months": int(vc.get("El Nino", 0)),
            "min_index": float(valid[index_col].min()) if len(valid) else np.nan,
            "max_index": float(valid[index_col].max()) if len(valid) else np.nan,
        })

    return {
        "Summary_statistics": pd.DataFrame(summary_rows),
        "Monthly_counts_for_figures": pd.concat(monthly_list, ignore_index=True),
        "Event_episodes": pd.concat(episode_list, ignore_index=True),
    }


# ============================================================
# 8. Figures
# ============================================================

def save_figure(fig: plt.Figure, basename: str) -> Dict:
    png = PNG_DIR / f"{basename}.png"
    tif = TIFF_DIR / f"{basename}.tif"

    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    if EXPORT_TIFF:
        try:
            fig.savefig(
                tif,
                dpi=DPI,
                bbox_inches="tight",
                facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"},
            )
        except TypeError:
            fig.savefig(tif, dpi=DPI, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    return {
        "figure": basename,
        "png_path": str(png),
        "tiff_path": str(tif) if EXPORT_TIFF else "",
        "png_size_mb": png.stat().st_size / 1024 / 1024 if png.exists() else np.nan,
        "tiff_size_mb": tif.stat().st_size / 1024 / 1024 if EXPORT_TIFF and tif.exists() else np.nan,
    }


def get_counts(tables: Dict[str, pd.DataFrame], definition: str) -> pd.DataFrame:
    counts = tables["Monthly_counts_for_figures"]
    return counts[counts["definition"] == definition].sort_values("month").copy()


def style_axes_box(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def apply_left_title(ax, title: str) -> None:
    ax.set_title(title, loc="left", pad=8)


def plot_phase_lines(ax, counts: pd.DataFrame, linestyle: str, label_suffix: str = "") -> None:
    x = np.arange(12)
    for phase in ["La Nina", "Neutral", "El Nino"]:
        ax.plot(
            x,
            counts[phase].values,
            color=PHASE_COLORS[phase],
            linestyle=linestyle,
            linewidth=1.8,
            label=f"{PHASE_LABELS[phase]}{label_suffix}",
        )


def style_count_axis(ax) -> None:
    ax.set_ylabel("Count")
    ax.set_xlabel("Month")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(MONTH_LABELS)
    ax.grid(True, axis="both", alpha=0.30, linewidth=0.7)
    ax.set_xlim(-0.5, 11.5)
    style_axes_box(ax)


def add_standard_legend(ax, loc: str = "best") -> None:
    ax.legend(
        loc=loc,
        fontsize=8.5,
        frameon=True,
        handlelength=LEGEND_HANDLELENGTH,
        borderaxespad=0.8,
        fancybox=False,
    )


def make_figures(enso: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    reports = []

    # -----------------------------
    # Figure S1a
    # -----------------------------
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plot_phase_lines(ax, get_counts(tables, "no_detrend_0.5C"), "--", " no detrend")
    plot_phase_lines(ax, get_counts(tables, "manuscript_quadratic_detrend_0.5C"), "-", " quad. detrend")
    apply_left_title(ax, "(a) ENSO counts using quadratically detrend and no detrend")
    style_count_axis(ax)
    add_standard_legend(ax, loc="upper right")
    fig.tight_layout()
    reports.append(save_figure(fig, "Figure_S1a_ENSO_counts_detrending"))

    # -----------------------------
    # Figure S1b
    # -----------------------------
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plot_phase_lines(ax, get_counts(tables, "manuscript_quadratic_detrend_0.5C"), "-", "")
    plot_phase_lines(ax, get_counts(tables, "threshold_1.0C"), "--", " 1°C threshold")
    apply_left_title(ax, "(b) ENSO counts using 0.5 °C and 1.0 °C threshold")
    style_count_axis(ax)
    add_standard_legend(ax, loc="upper right")
    fig.tight_layout()
    reports.append(save_figure(fig, "Figure_S1b_ENSO_counts_threshold"))

    # -----------------------------
    # Figure S2a
    # -----------------------------
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    x = pd.to_datetime(enso["time"])
    # Colors matched to the uploaded article-style figure:
    ax.plot(x, enso["nino34_anom"], color="tab:blue", linewidth=1.0, label="Monthly Niño3.4")
    ax.plot(x, enso["nino34_linear_fit"], color="tab:orange", linewidth=1.4, label="Linear fit")
    ax.plot(x, enso["nino34_quadratic_fit"], color="black", linewidth=1.4, label="Quadratic fit")
    ax.plot(
        x,
        enso["nino34_30yr_mean_constant_beginning"],
        color="tab:green",
        linewidth=1.4,
        label="30-year mean with constant beginning",
    )
    apply_left_title(ax, "(a)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Year")
    ax.grid(True, axis="both", alpha=0.30, linewidth=0.7)
    style_axes_box(ax)
    add_standard_legend(ax, loc="upper left")
    fig.tight_layout()
    reports.append(save_figure(fig, "Figure_S2a_Nino34_trend_components"))

    # -----------------------------
    # Figure S2b
    # -----------------------------
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plot_phase_lines(ax, get_counts(tables, "manuscript_quadratic_detrend_0.5C"), "-", "")
    plot_phase_lines(ax, get_counts(tables, "30yr_running_mean_removal"), "--", " 30-year running mean")
    apply_left_title(ax, "(b) Number of ENSO events")
    style_count_axis(ax)
    add_standard_legend(ax, loc="upper right")
    fig.tight_layout()
    reports.append(save_figure(fig, "Figure_S2b_ENSO_counts_30yr_sensitivity"))

    # -----------------------------
    # Additional method comparison
    # -----------------------------
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    plot_phase_lines(ax, get_counts(tables, "manuscript_quadratic_detrend_0.5C"), "-", " manuscript method")
    plot_phase_lines(ax, get_counts(tables, "NOAA_style_ONI_event"), "--", " NOAA-style ONI")
    apply_left_title(ax, "(c) Number of ENSO events")
    style_count_axis(ax)
    add_standard_legend(ax, loc="upper right")
    fig.tight_layout()
    reports.append(save_figure(fig, "Figure_S3_ENSO_counts_manuscript_vs_NOAA"))

    return pd.DataFrame(reports)


# ============================================================
# 9. Export
# ============================================================

def build_readme() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "item": "purpose",
            "value": "Create article-style Figure S1/S2 ENSO diagnostics and one NOAA-style sensitivity comparison.",
        },
        {
            "item": "main_period",
            "value": f"{START_YEAR}-01 to {END_YEAR}-12",
        },
        {
            "item": "main_method",
            "value": "Monthly Niño3.4 anomaly, 1961-2023 monthly climatology, quadratic detrending, monthly ±0.5 °C threshold.",
        },
        {
            "item": "noaa_sensitivity",
            "value": "1991-2020 climatology, centered 3-month ONI-like index, quadratic detrending, ±0.5 °C threshold, at least five consecutive months.",
        },
        {
            "item": "main_columns_for_downstream_analysis",
            "value": "year, month, time, main_phase, main_event, nino34_detrended",
        },
        {
            "item": "sensitivity_columns",
            "value": "phase_no_detrend, event_no_detrend, phase_1c, event_1c, phase_30yr, event_30yr, noaa_phase, noaa_event.",
        },
    ])


def build_method_key() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "definition": "manuscript_quadratic_detrend_0.5C",
            "role": "main",
            "index_col": "nino34_detrended",
            "phase_col": "main_phase",
            "event_col": "main_event",
            "description": "Monthly Niño3.4 anomaly, quadratic detrending, ±0.5 °C threshold.",
        },
        {
            "definition": "no_detrend_0.5C",
            "role": "sensitivity",
            "index_col": "nino34_anom",
            "phase_col": "phase_no_detrend",
            "event_col": "event_no_detrend",
            "description": "Monthly Niño3.4 anomaly without quadratic detrending, ±0.5 °C threshold.",
        },
        {
            "definition": "threshold_1.0C",
            "role": "sensitivity",
            "index_col": "nino34_detrended",
            "phase_col": "phase_1c",
            "event_col": "event_1c",
            "description": "Monthly detrended Niño3.4 anomaly, ±1.0 °C threshold.",
        },
        {
            "definition": "30yr_running_mean_removal",
            "role": "sensitivity",
            "index_col": "nino34_anom_30yr",
            "phase_col": "phase_30yr",
            "event_col": "event_30yr",
            "description": "Monthly Niño3.4 anomaly after removing the 30-year running mean.",
        },
        {
            "definition": "NOAA_style_ONI_event",
            "role": "sensitivity",
            "index_col": "noaa_oni_detrended",
            "phase_col": "noaa_phase",
            "event_col": "noaa_event",
            "description": "NOAA-style ONI event sensitivity using centered 3-month ONI-like index and five-consecutive-month duration rule.",
        },
    ])


def build_missing_report(enso: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    expected = (END_YEAR - START_YEAR + 1) * 12
    return pd.DataFrame([
        {"item": "main_months_expected", "value": expected, "status": "info"},
        {"item": "main_months_actual", "value": len(enso), "status": "ok" if len(enso) == expected else "check"},
        {"item": "required_ersst_first", "value": required_ersst_months()[0].strftime("%Y-%m"), "status": "info"},
        {"item": "required_ersst_last", "value": required_ersst_months()[-1].strftime("%Y-%m"), "status": "info"},
        {"item": "missing_ersst_after_check", "value": int((~inventory["exists"]).sum()), "status": "ok" if inventory["exists"].all() else "check"},
        {"item": "main_index_defined_months", "value": int(enso["nino34_detrended"].notna().sum()), "status": "ok" if enso["nino34_detrended"].notna().sum() == expected else "check"},
        {"item": "noaa_index_defined_months", "value": int(enso["noaa_oni_detrended"].notna().sum()), "status": "ok" if enso["noaa_oni_detrended"].notna().sum() == expected else "check"},
    ])


def core_monthly_columns(enso: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "year", "month", "time", "sst", "source_file",
        "nino34_clim_1961_2023", "nino34_anom", "nino34_linear_fit", "nino34_quadratic_fit",
        "nino34_detrended", "main_phase", "main_event",
        "phase_no_detrend", "event_no_detrend",
        "phase_1c", "event_1c",
        "nino34_30yr_mean_constant_beginning", "nino34_anom_30yr", "phase_30yr", "event_30yr",
        "noaa_clim_1991_2020", "noaa_nino34_anom", "noaa_oni", "noaa_oni_quadratic_fit",
        "noaa_oni_detrended", "noaa_phase", "noaa_event",
    ]
    return enso[[c for c in cols if c in enso.columns]].copy()


def export_outputs(
    enso: pd.DataFrame,
    inventory: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    figure_report: pd.DataFrame,
) -> Path:
    excel_path = OUT_DIR / "ENSO_article_figS1_S2_NOAA_sensitivity_1961_2023_v2_panelwise.xlsx"

    readme = build_readme()
    method_key = build_method_key()
    missing_report = build_missing_report(enso, inventory)
    main_data = core_monthly_columns(enso)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        method_key.to_excel(writer, sheet_name="Method_key", index=False)
        missing_report.to_excel(writer, sheet_name="Missing_report", index=False)
        main_data.to_excel(writer, sheet_name="Main_monthly_data", index=False)
        tables["Summary_statistics"].to_excel(writer, sheet_name="Summary_statistics", index=False)
        tables["Monthly_counts_for_figures"].to_excel(writer, sheet_name="Monthly_counts_for_figures", index=False)
        tables["Event_episodes"].to_excel(writer, sheet_name="Event_episodes", index=False)
        figure_report.to_excel(writer, sheet_name="Figure_files", index=False)

        workbook = writer.book
        for ws in workbook.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                max_len = 0
                letter = col[0].column_letter
                for cell in col[:200]:
                    try:
                        max_len = max(max_len, len(str(cell.value)))
                    except Exception:
                        pass
                ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 60)

    # CSV copies
    readme.to_csv(TABLE_DIR / "README.csv", index=False, encoding="utf-8-sig")
    method_key.to_csv(TABLE_DIR / "Method_key.csv", index=False, encoding="utf-8-sig")
    missing_report.to_csv(TABLE_DIR / "Missing_report.csv", index=False, encoding="utf-8-sig")
    main_data.to_csv(TABLE_DIR / "Main_monthly_data.csv", index=False, encoding="utf-8-sig")
    tables["Summary_statistics"].to_csv(TABLE_DIR / "Summary_statistics.csv", index=False, encoding="utf-8-sig")
    tables["Monthly_counts_for_figures"].to_csv(TABLE_DIR / "Monthly_counts_for_figures.csv", index=False, encoding="utf-8-sig")
    tables["Event_episodes"].to_csv(TABLE_DIR / "Event_episodes.csv", index=False, encoding="utf-8-sig")
    figure_report.to_csv(TABLE_DIR / "Figure_files.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(TABLE_DIR / "ERSST_file_inventory.csv", index=False, encoding="utf-8-sig")

    return excel_path


# ============================================================
# 10. Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Article-style Figure S1/S2 ENSO diagnostics + NOAA-style sensitivity")
    print("Panel-wise figure export version")
    print("=" * 80)
    print(f"ERSST folder : {ERSST_DIR}")
    print(f"Output folder: {OUT_DIR}")
    print("Figures      : S1a, S1b, S2a, S2b, and one manuscript-vs-NOAA comparison figure")
    print("=" * 80)

    enso, inventory = build_all_enso()
    tables = build_summary_tables(enso)
    figure_report = make_figures(enso, tables)
    excel_path = export_outputs(enso, inventory, tables, figure_report)

    print("\nFinished.")
    print(f"Excel file : {excel_path}")
    print(f"CSV tables : {TABLE_DIR}")
    print(f"PNG figures: {PNG_DIR}")
    print(f"TIFF figs  : {TIFF_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()