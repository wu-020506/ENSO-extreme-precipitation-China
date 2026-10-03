#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import re
import time
import gc
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from scipy import stats, ndimage

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.geometry import box


warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    # New ERA5 folders produced by your latest download script
    # -------------------------------------------------------------------------
    # The expanded-domain ERA5 archive.  It must contain all grid cells needed
    # by CALC_EXTENT below.  This is the global archive used in the latest
    # workflow, not the earlier 55E-180E / 5S-70N subset.
    ERA5_ROOT = DATA_ROOT / "era5"

    UV_DIR = ERA5_ROOT / "01_UV_500_700hPa"
    Q_DIR = ERA5_ROOT / "02_SpecificHumidity_700hPa"
    RH_DIR = ERA5_ROOT / "03_RelativeHumidity_700hPa"
    OMEGA_DIR = ERA5_ROOT / "04_VerticalVelocity_700hPa"
    SP_DIR = ERA5_ROOT / "05_SurfacePressure"
    EVAP_DIR = ERA5_ROOT / "07_Evaporation"

    # Same ENSO monthly classification used by Figures 3–6
    PHASE_SOURCE = (
        OUTPUT_ROOT / "04_spatial_rr" / "00_netcdf" /
        "Figure4_grid_monthly_MEP95_RX5day_anomaly_CN.nc"
    )

    # Exact official six-basin polygon file.
    # NO fuzzy basin searching is used, avoiding the previous Songhua problem.
    BASIN_FILE = DATA_ROOT / "basins" / "six_selected_official_basins.shp"

    # Keep the expanded-domain results separate from the earlier China-centred
    # Figure 7 outputs so that no accepted figure is overwritten.
    OUT_ROOT = OUTPUT_ROOT / "07_era5_mechanisms"
    OUT_DIR = OUT_ROOT / "01_figures"
    TABLE_DIR = OUT_ROOT / "02_tables"
    CACHE_DIR = OUT_ROOT / "03_cache"

    # Layout-only version tag.  Every redraw goes to a NEW folder so Windows,
    # Jupyter, or an image viewer cannot show an older same-name PNG by mistake.
    # Every code version writes outputs into its own independent folder.
    # To create a new version folder, only update FIGURE_LAYOUT_TAG.
    FIGURE_LAYOUT_TAG = "v112_q75_connected_anomaly_boxes"
    # Preserve every v111 plotting choice, but replace the fixed 0.8 anomaly-box
    # threshold with the panel-specific 75th percentile of finite absolute
    # gridpoint anomalies.  Positive and negative components remain separate and
    # only the largest qualifying connected component is boxed.

    VERSION_OUTPUT_DIR = OUT_DIR / FIGURE_LAYOUT_TAG
    COMBINED_BOARD_DIR = VERSION_OUTPUT_DIR / "combined_boards"
    SINGLE_PANEL_DIR = VERSION_OUTPUT_DIR / "single_panels"
    HEATMAP_FIG_DIR = VERSION_OUTPUT_DIR / "supplementary_original_heatmap"

    # -------------------------------------------------------------------------
    # SPEED / CACHE CONTROLS
    # -------------------------------------------------------------------------
    # Seasonal ENSO-neutral composite effects are tiny compared with raw ERA5.
    # Once calculated, save one NetCDF per season and reuse it for all later
    # layout / legend / spacing edits.
    USE_SEASON_EFFECT_CACHE = True
    FORCE_RECOMPUTE_SEASON_EFFECT_CACHE = False
    SEASON_EFFECT_CACHE_DIR = CACHE_DIR / "season_effects"
    SEASON_EFFECT_CACHE_VERSION = "v9_qkey_spacing_refined"

    # FAST_PREVIEW=True is recommended while adjusting the figure.  It renders
    # PNG only at 300 dpi.  Set False only for the final journal export, which
    # restores 500-dpi PNG/TIF plus PDF.
    FAST_PREVIEW = True

    # Formal analysis period.  Monthly products and the JJA/MAM/SON seasonal
    # composites cover 1961--2023.  DJF is labelled by its Jan--Feb year; if
    # the phase source starts in January 1961, DJF 1961 is excluded rather
    # than assigning an unsupported phase to December 1960.
    START_YEAR = 1961
    END_YEAR = 2023

    # Raw archive coverage.  The archive contains 1960, but a raw ERA5 month
    # is used only when the same month has an ENSO phase classification.
    INPUT_START_YEAR = 1960
    INPUT_END_YEAR = 2023

    # Full Figure 7 seasonal series.
    # Each tuple is (year offset relative to season-year, calendar month).
    # DJF is assigned to the Jan-Feb year: e.g. DJF 1962 = Dec 1961 +
    # Jan-Feb 1962.  An incomplete DJF at either end is never constructed.
    SEASON_DEFS = {
        "JJA": ((0, 6), (0, 7), (0, 8)),
        "DJF": ((-1, 12), (0, 1), (0, 2)),
        "MAM": ((0, 3), (0, 4), (0, 5)),
        "SON": ((0, 9), (0, 10), (0, 11)),
    }
    SEASON_ORDER = ("JJA", "DJF", "MAM", "SON")

    # Keep one colour scale per diagnostic across all seasons and both ENSO phases.
    # This makes JJA/DJF/MAM/SON directly comparable.
    USE_COMMON_COLOR_SCALE_ACROSS_SEASONS = True

    # Also regenerate the original basin-month mechanism heatmap.
    # The heatmap contains MFC700, ascent700, RH700 and surface evaporation,
    # each shown for El Niño and La Niña using bias-corrected Hedges' g.
    # This supplementary basin-month heatmap is a SECOND full 1961-2023 ERA5
    # pass in the original script.  Keep it OFF while refining Figure 7.
    EXPORT_BASIN_MONTH_HEATMAP = True
    # If you later turn EXPORT_BASIN_MONTH_HEATMAP=True, reuse an existing
    # Hedges-g table instead of rereading 63 years of ERA5 whenever possible.
    USE_EXISTING_HEATMAP_TABLE = True
    REBUILD_BASIN_MONTH_HEATMAP_DATA = False

    EQUAL_AREA_CRS = "EPSG:6933"
    HEATMAP_WEIGHT_CACHE = CACHE_DIR / "ERA5_grid_fractional_six_basin_area_km2.nc"
    HEATMAP_TABLE = TABLE_DIR / "ERA5_basin_month_mechanism_Hedges_g.csv"

    # ERA5 CDS monthly-mean evaporation is normally a monthly mean of daily
    # accumulation. Convert m -> mm and multiply by days in month.
    EVAP_ACCUMULATION_MODE = "monthly_mean_daily_total"
    ERA5_EVAPORATION_UPWARD_IS_NEGATIVE = True

    HEATMAP_LEVELS = (
        -1.50, -1.00, -0.75, -0.50, -0.25, 0.00,
         0.25,  0.50,  0.75,  1.00,  1.50,
    )
    HEATMAP_FIGSIZE = (13.8, 13.2)
    HEATMAP_TITLE_SIZE = 13.0
    HEATMAP_TICK_SIZE = 11.0
    HEATMAP_CBAR_TICK_SIZE = 12.5
    HEATMAP_CBAR_LABEL_SIZE = 13.5
    HEATMAP_WSPACE = 0.16
    HEATMAP_HSPACE = 0.18
    HEATMAP_CBAR_WIDTH = 0.72

    # Expanded physical-mechanism domain requested for the new calculation.
    # A 5-degree buffer is retained around the plotted domain so spatial
    # derivatives are not evaluated exactly at the displayed boundary.
    # Order: west, east, south, north.
    CALC_EXTENT = (35.0, 185.0, -20.0, 65.0)
    PLOT_EXTENT = (40.0, 180.0, -15.0, 60.0)
    MAP_CENTRAL_LONGITUDE = 110.0

    # Reference-like geographic ticks:
    # longitude roughly every 20 degrees, latitude every 15 degrees.
    LON_TICKS = (60, 80, 100, 120, 140, 160)
    LAT_TICKS = (-15, 0, 15, 30, 45, 60)

    # Reference-style discrete arrow field.  All finite, non-zero sampled
    # vectors are retained: values below the physical threshold are cyan and
    # direction-only; values at/above it are dark blue and magnitude-scaled.
    QUIVER_SPACING_DEG = 4.5

    # A robust reference magnitude is retained as a fallback for any future
    # vector row without an explicit physical threshold.  The current wind and
    # moisture-flux rows are scaled directly from their 1 and 5 thresholds.
    QUIVER_REF_PERCENTILE = 80.0
    QUIVER_MIN_REF_FRACTION = 0.0
    QUIVER_CAP_PERCENTILE = 95.0
    # Display-only length cap relative to the first visible (threshold) arrow.
    # This keeps strong vectors readable without allowing neighbouring arrows
    # to overlap excessively in the slightly denser field.
    QUIVER_MAX_THRESHOLD_MULTIPLIER = 1.55
    # Minimum physical clearance between adjacent candidate arrows.  The final
    # x/y strides are increased automatically when the panel is too small for
    # the requested arrow length, preventing long arrows from overlapping.
    QUIVER_OVERLAP_CLEARANCE_FACTOR = 1.18

    # The threshold arrow has this physical length in both the map and key.
    # Quiver units are inches, so size is stable across single/combined figures
    # and across 300/500-dpi export.
    WIND_ARROW_LENGTH_INCH = 0.160
    QFLUX_ARROW_LENGTH_INCH = 0.160
    QUIVER_WIDTH = 0.0110
    QUIVER_OUTLINE_LINEWIDTH_PT = 0.90
    QUIVER_SMOOTH_SIGMA = 1.15
    QUIVER_HEADWIDTH = 4.0
    QUIVER_HEADLENGTH = 5.0
    QUIVER_HEADAXISLENGTH = 4.5
    QUIVER_LOW_ALPHA = 0.96
    QUIVER_HIGH_ALPHA = 1.00
    QUIVER_LOW_COLOR = "#20A4C3"
    QUIVER_HIGH_COLOR = "#171568"
    # Backward-compatible alias used by progress messages and the key frame.
    QUIVER_COLOR = QUIVER_HIGH_COLOR
    # Below-threshold vectors show direction only: all receive this fraction
    # of the physical threshold-arrow length, irrespective of their magnitude.
    QUIVER_LOW_DIRECTION_LENGTH_FACTOR = 0.60
    WIND_VECTOR_MIN_MAGNITUDE = 1.0
    QFLUX_VECTOR_MIN_MAGNITUDE = 5.0

    # Single-panel export settings.
    SINGLE_FIGSIZE = (8.6, 4.95)
    SINGLE_LEFT = 0.085
    SINGLE_RIGHT = 0.985
    SINGLE_TOP = 0.905
    SINGLE_BOTTOM = 0.185

    BASIN_COLOR = "#7B1FA2"
    BASIN_LINEWIDTH = 1.15
    BASIN_LINESTYLE = (0, (4.0, 2.0))

    # Geographic context: diagnostic shading remains visible over China and
    # the ocean, while other countries are overlaid with a restrained grey.
    OTHER_COUNTRY_FACE = "#F5F5F5"
    CHINA_FACE = "#F5F5F5"
    OCEAN_FACE = "#F5F5F5"
    # False = keep the atmospheric shading visible over all countries;
    # True  = overlay the neutral-grey mask on non-study countries after shading.
    OVERLAY_OTHER_COUNTRIES_AFTER_SHADING = False

    QKEY_COLOR = QUIVER_COLOR
    QKEY_TEXT_SIZE = 10.2
    TITLE_PAD = 3.0
    # Unit-free threshold key, tucked just inside the upper-right map frame.
    # The integer marks the transition from short cyan direction-only arrows
    # to dark-blue magnitude-proportional arrows.
    QKEY_BOX_X = 0.995
    QKEY_BOX_Y = 0.995
    QKEY_BOX_WIDTH = 0.078
    QKEY_BOX_HEIGHT = 0.100
    QKEY_BOX_LINEWIDTH = 1.20
    QKEY_LABEL_SEPARATION_INCH = 0.025
    # False keeps the threshold key in both panels of each vector row; set True
    # only if a future compact layout should retain the right-hand key alone.
    BOARD_SHOW_QKEY_ONLY_ON_RIGHT_PANEL = False
    SHOW_QUIVER_NUMERIC_KEY = True
    # Slightly mute saturated extremes so the significant centres remain
    # readable without competing with coastlines and vector arrows.
    CMAP_LIGHTEN_FACTOR = 0.0
    # Width of the low-value pale band in the anomaly shading.
    WHITE_CENTER_BIN_COUNT = 2
    # Exact physical-value limits of the pale low-value centre. These are
    # colour-bin limits, not statistical filters, so the calculations are unchanged.
    WHITE_CENTER_THRESHOLD_BY_SHADE = {
        "div500": 0.20,
        "mfc700": 0.20,
        "ascent700": 0.005,
    }
    SYMLOG_LINTHRESH_BY_SHADE = {
        "div500": 0.20,
        "mfc700": 0.20,
        "ascent700": 0.005,
    }
    SHADE_N_INTERVALS = 10

    # Final main-figure board layout.  v109 relaxes the previous -0.150 value
    # just enough to separate the two active map frames and their inner labels,
    # while keeping the paired panels visually compact.  Single panels do not
    # use this setting.
    WSPACE = -0.100
    HSPACE = 0.035

    FIGSIZE = (16.2, 14.5)
    BOARD_LEFT = 0.056
    BOARD_RIGHT = 0.942
    BOARD_TOP = 0.968
    BOARD_BOTTOM = 0.036

    DPI = 300 if FAST_PREVIEW else 500

    FONT_FAMILY = "Times New Roman"
    TITLE_SIZE = 15.5
    # Combined-board panel letters/titles only: 15.5 + 6.0 = 21.5 pt.
    # v110 used +4.0 (19.5 pt), so v111 is exactly two points larger.
    BOARD_TITLE_SIZE = TITLE_SIZE + 6.0
    TICK_SIZE = 15.5
    ROW_LABEL_SIZE = 12.5
    CBAR_TICK_SIZE = 13.0
    CBAR_LABEL_SIZE = 14.0
    BASIN_LABEL_SIZE = 6.8

    # Board-only controls.  The colourbar is positioned from the active frame
    # of the right-hand map, so the physical gap is short and deterministic.
    BOARD_ROW_HSPACE = 0.200
    BOARD_CBAR_PAD = 0.006
    BOARD_CBAR_WIDTH = 0.014
    BOARD_QKEY_BOX_X = 1.000
    # Board-only enlargement; the single-panel arrow key retains its original
    # 0.078 x 0.100 frame and 10.2-pt number.
    BOARD_QKEY_SCALE = 1.25
    BOARD_QKEY_TEXT_SIZE = 12.5
    REMOVE_COMBINED_BOARD_COLORBARS = False

    # Gridpoint significance stippling (black dots) for the shaded field.
    # The statistical test itself is unchanged: grid-cell-wise Welch two-sample
    # t tests on detrended ENSO-phase versus Neutral season-year anomalies.
    SIGNIFICANCE_ALPHA = 0.05
    SHOW_SIGNIFICANCE_STIPPLING = True
    # Dots are restricted to original p < 0.05 grid cells that also remain
    # inside the final displayed coloured regions. This avoids putting dots in
    # background areas introduced only by display morphology/dilation.  This
    # Smaller dots with a slightly denser sampling.
    STIPPLE_ONLY_WITHIN_DISPLAYED_FILL = True
    SIGNIFICANCE_STIPPLE_SIZE = 4.2
    SIGNIFICANCE_STIPPLE_ALPHA = 0.88
    SIGNIFICANCE_STIPPLE_STRIDE = 7
    SIGNIFICANCE_STIPPLE_COLOR = "#000000"
    SIGNIFICANCE_STIPPLE_ZORDER = 12.0

    # Science-style anomaly core boxes
    THRESHOLD_BOX_BUFFER_LON = 1.5
    THRESHOLD_BOX_BUFFER_LAT = 1.0
    THRESHOLD_BOX_FRAME_INSET_DEG = 0.25
    THRESHOLD_BOX_MIN_CELLS = 20
    # Apply boxes only to the first two rows.  For each panel independently,
    # calculate Q75 of |anomaly| across every finite displayed grid cell.  Using
    # absolute values provides one symmetric magnitude threshold for positive
    # and negative components, while allowing the two diagnostics (with
    # different physical units) to retain their own data-adaptive thresholds.
    THRESHOLD_BOX_PERCENTILE_BY_SHADE = {
        "div500": 75.0,
        "mfc700": 75.0,
    }
    POSITIVE_BOX_COLOR = "#D73027"
    NEGATIVE_BOX_COLOR = "#4575B4"
    THRESHOLD_BOX_LINEWIDTH = 2.0
    # Shading is zorder 2, threshold boxes are 8.8, and the cyan/dark vector
    # layers are 9.8/10.0.  The arrows therefore remain fully visible wherever
    # they cross a box edge.
    THRESHOLD_BOX_ZORDER = 8.8
    RH_CONTOUR_COLOR = "#8F33C8"
    RH_CONTOUR_LINEWIDTH = 0.82
    RH_CONTOUR_LINESTYLE = "solid"
    # Additional display-only amplitude filter for stippling: only grid cells
    # that are both statistically significant and exceed the following anomaly
    # magnitudes are dotted.  This suppresses weak-but-significant background
    # speckles and keeps the emphasis on robust anomaly cores.
    SIGNIFICANCE_STIPPLE_VALUE_THRESHOLD_BY_SHADE = {
        "div500": 0.4,
        "mfc700": 0.35,
        "ascent700": 0.01,
    }
    # Xie-style dotted significance regions: still based on the original
    # grid-cell Welch p < 0.05 test, but only coherent contiguous clusters are
    # dotted so the significance marking appears as broader dotted regions
    # rather than as isolated black speckles.
    SIGNIFICANCE_STIPPLE_MIN_CLUSTER_CELLS = 16
    SIGNIFICANCE_STIPPLE_CLOSE_ITERS = 1
    SIGNIFICANCE_STIPPLE_OPEN_ITERS = 0
    SIGNIFICANCE_STIPPLE_DILATE_ITERS = 0
    SIGNIFICANCE_OUTLINE_COLOR = "#222222"
    SIGNIFICANCE_OUTLINE_WIDTH = 0.65
    SIGNIFICANCE_OUTLINE_STYLE = "solid"
    # Colour-fill mode. False = show the broader detrended ENSO-minus-Neutral
    # anomaly field; statistical significance is indicated separately by black
    # stippling. True restores the v56 significant-region-only shading.
    FILL_ONLY_SIGNIFICANT_LARGE_REGIONS = False
    MIN_SIGNIFICANT_REGION_CELLS = 42
    SIGNIFICANCE_MASK_CLOSE_ITERS = 5
    SIGNIFICANCE_MASK_DILATE_ITERS = 3
    SIGNIFICANCE_MASK_OPEN_ITERS = 0
    # In addition to p < 0.05, require a visible anomaly magnitude before
    # retaining colour. This removes many tiny weak-colour fragments while
    # preserving coherent, physically meaningful anomaly centres.
    SIGNIFICANT_FILL_EFFECT_STEP_FACTOR = 1.00
    # Publication-style display-only smoothing: slightly stronger than the
    # previous version so the colour patches become more contiguous and less
    # speckled, while keeping the underlying composite statistics unchanged.
    DISPLAY_SMOOTH_SIGMA = 3.10

    # Minimum seasonal samples after phase classification.
    MIN_PHASE_N = 8
    MIN_NEUTRAL_N = 12

    # Publication-safe option.
    # False = keep all finite ERA5 pressure-level values, matching your recent
    #         "available-data-only" Figure 7 approach.
    # True  = additionally remove pressure levels that are underground.
    USE_SURFACE_PRESSURE_MASK = False

    # If the terrain mask is enabled, require surface pressure to exceed the
    # pressure level by this amount.
    PRESSURE_CLEARANCE_HPA = 0.0

    # Shared row colour maps.  Each row is linked to one of the three supplied
    # RGB reference palettes; see REFERENCE_RGB_ANCHORS below.
    CMAPS = {
        # Restore the original three distinct publication colour maps.
        "div500": "science_div500",
        "mfc700": "science_mfc700",
        "ascent700": "science_ascent700",
    }

    # Use robust symmetric limits so both ENSO phases in one row use
    # exactly the same colour scale.
    ROBUST_PERCENTILE = 97.0

    # Force all values inside the physical neutral threshold to the central
    # white colour bins.  This does not turn them into missing data: genuine
    # Keep signed low-magnitude values instead of snapping them to zero.  The
    # two central bins use very pale sign-specific colours, so weak anomalies
    # remain visible and are not confused with missing-data/background areas.
    MASK_NEAR_ZERO_VALUES = False
    LOW_VALUE_BLEND_TO_WHITE = 0.72
    # Threshold expressed as a multiple of the first positive colour-step.
    ZERO_MASK_STEP_FACTOR = 0.18

    # Edge-fill control:
    # Masked display-only areas are rendered with the same colour as the map
    # background. The plotted anomaly field is also padded to the plotting
    # boundary so contourf does not leave a white halo near the frame.
    FILL_MASKED_WITH_BACKGROUND = True
    MAP_BACKGROUND_COLOR = "#F5F5F5"
    EDGE_PADDING_CELLS = 2

    # -----------------------------------------------------------------
    # Variable-specific display-only refinements
    # These do NOT change the underlying composite statistics.
    # They only make the published shading more journal-like: cleaner, more
    # contiguous, and easier to read.  This version is tuned to emphasise the
    # main anomaly centres rather than fragmented local patches: stronger
    # smoothing, stricter small-patch removal, and simpler colourbar labels.
    # -----------------------------------------------------------------
    DISPLAY_SMOOTH_SIGMA_BY_SHADE = {
        # Xie et al. (2025) Fig. 3 inspired style: keep the anomaly field
        # visually continuous, but with slightly variable-specific smoothing
        # so the large-scale structures dominate over grid noise.
        "div500": 3.80,
        "mfc700": 3.20,
        "ascent700": 2.80,
    }
    ZERO_MASK_STEP_FACTOR_BY_SHADE = {
        # Retained for any optional display filtering, but this version no
        # longer masks the near-zero band; the aim is a more continuous fill
        # pattern like Fig. 3 in Xie et al. (2025).
        "div500": 0.18,
        "mfc700": 0.16,
        "ascent700": 0.14,
    }
    MIN_SIGNIFICANT_REGION_CELLS_BY_SHADE = {
        "div500": 120,
        "mfc700": 70,
        "ascent700": 45,
    }
    SIGNIFICANT_FILL_EFFECT_STEP_FACTOR_BY_SHADE = {
        "div500": 1.40,
        "mfc700": 1.25,
        "ascent700": 1.05,
    }
    SIGNIFICANCE_MASK_CLOSE_ITERS_BY_SHADE = {
        "div500": 3,
        "mfc700": 3,
        "ascent700": 2,
    }
    SIGNIFICANCE_MASK_DILATE_ITERS_BY_SHADE = {
        "div500": 3,
        "mfc700": 1,
        "ascent700": 1,
    }
    # Remove very tiny isolated coloured patches after display-only smoothing.
    MIN_DISPLAY_PATCH_CELLS_BY_SHADE = {
        # Keep broad-scale structures but suppress only the tiniest isolated
        # fragments; the colour field is intended to remain more continuous.
        "div500": 42,
        "mfc700": 26,
        "ascent700": 18,
    }
    # Extra display-only morphology to suppress ring-like / circular tiny spots.
    DISPLAY_PATCH_OPEN_ITERS_BY_SHADE = {
        "div500": 1,
        "mfc700": 1,
        "ascent700": 0,
    }
    DISPLAY_PATCH_CLOSE_ITERS_BY_SHADE = {
        "div500": 3,
        "mfc700": 3,
        "ascent700": 1,
    }

    # Slightly reduced publication colour ranges for cleaner A/B-row colourbars.
    FIXED_DISPLAY_MAX_BY_SHADE = {
        "div500": 1.4,
        "mfc700": 1.1,
        "ascent700": 0.04,
    }
    # Use many evenly spaced contour levels for a smoother, more continuous
    # anomaly field, while the colorbar still shows the fixed publication tick
    # values. This better matches the appearance of Fig. 3 in Xie et al. (2025).
    DISPLAY_LEVEL_COUNT_BY_SHADE = {
        "div500": 12,
        "mfc700": 10,
        "ascent700": 8,
    }
    # Major colorbar labels are intentionally kept sparse for a cleaner publication-style colour bar.
    # Positive values follow a quasi-log spacing, and the negative side mirrors
    # those same magnitudes. Intermediate colour-box boundaries are still shown
    # as minor ticks, but only a small set of major labels is printed.
    COLORBAR_TICK_DECIMALS_BY_SHADE = {
        "div500": 1,
        "mfc700": 1,
        "ascent700": 3,
    }

    FIXED_COLORBAR_TICKS_BY_SHADE = {
        # Readable publication-style major labels. The spacing is simple and
        # easy to read, while still keeping more detail near the neutral band.
        "div500": (-1.4, -0.7, -0.2, 0.0, 0.2, 0.7, 1.4),
        "mfc700": (-1.1, -0.5, -0.2, 0.0, 0.2, 0.5, 1.1),
        "ascent700": (-0.04, -0.01, -0.005, 0.0, 0.005, 0.01, 0.04),
    }
    # Explicit publication labels so the ±0.20 and ±0.005 colour-bar numbers
    # are printed exactly instead of being changed by automatic rounding. This keeps the displayed
    # numeric labels fully aligned with the actual tick locations and the
    # chosen white-centre thresholds.
    FIXED_COLORBAR_TICK_LABELS_BY_SHADE = {
        "div500": ("-1.4", "-0.7", "-0.2", "0", "0.2", "0.7", "1.4"),
        "mfc700": ("-1.1", "-0.5", "-0.2", "0", "0.2", "0.5", "1.1"),
        "ascent700": ("-0.040", "-0.010", "-0.005", "0", "0.005", "0.010", "0.040"),
    }
    # Publication display boundaries.  Compared with v98, additional colour
    # intervals are inserted outside the pale centre to create a smoother
    # gradient while retaining exact boundaries at ±0.20 and ±0.005.
    CUSTOM_DISPLAY_LEVELS_BY_SHADE = {
        "div500": (
            -1.40, -1.20, -1.00, -0.80, -0.60, -0.45, -0.32,
            -0.20, 0.00, 0.20,
             0.32,  0.45,  0.60,  0.80,  1.00,  1.20,  1.40,
        ),
        "mfc700": (
            -1.10, -0.90, -0.75, -0.60, -0.45, -0.35,
            -0.20, 0.00, 0.20,
             0.35,  0.45,  0.60,  0.75,  0.90,  1.10,
        ),
        "ascent700": (
            -0.0400, -0.0300, -0.0225, -0.0150, -0.0100, -0.0075,
            -0.0050,  0.0000,  0.0050,
             0.0075,  0.0100,  0.0150,  0.0225,  0.0300,  0.0400,
        ),
    }
    COLORBAR_SUBDIVISIONS_BY_SHADE = {
        "div500": 3,
        "mfc700": 3,
        "ascent700": 3,
    }
    EXPORT_PNG = True
    EXPORT_TIF = not FAST_PREVIEW
    EXPORT_PDF = not FAST_PREVIEW

    # Export both the publication-ready 3 x 2 boards and the corresponding
    # individual panels.  Set either switch to False only to shorten a later
    # redraw; it does not change the calculations.
    EXPORT_COMBINED_BOARDS = True
    EXPORT_SINGLE_PANELS = True


PHASE_CODE = {"El Nino": 1, "La Nina": -1}
PHASE_LABEL = {"El Nino": "El Niño", "La Nina": "La Niña"}
PHASES = ("El Nino", "La Nina")
MONTHS = tuple(range(1, 13))
MONTH_LABELS = ("J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D")

BASIN_ORDER = ("Songhua", "Haihe", "Yellow", "Huaihe", "Yangtze", "Pearl")

BASIN_LABEL = {
    "Songhua": "Songhua",
    "Haihe": "Haihe",
    "Yellow": "Yellow",
    "Huaihe": "Huaihe",
    "Yangtze": "Yangtze",
    "Pearl": "Pearl",
}

BASIN_ALIASES = {
    "Songhua": ("songhua", "song hua", "松花江", "松花"),
    "Haihe": ("haihe", "hai he", "海河"),
    "Yellow": ("yellow", "huanghe", "huang he", "黄河"),
    "Huaihe": ("huaihe", "huai he", "淮河"),
    "Yangtze": ("yangtze", "changjiang", "chang jiang", "长江"),
    "Pearl": ("pearl", "zhujiang", "zhu jiang", "珠江"),
}


# =============================================================================
# STYLE
# =============================================================================

def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [CFG.FONT_FAMILY, "DejaVu Serif"],
        "font.size": CFG.TICK_SIZE,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": CFG.TICK_SIZE,
        "ytick.labelsize": CFG.TICK_SIZE,
        "axes.unicode_minus": False,
        "figure.facecolor": CFG.MAP_BACKGROUND_COLOR,
        "savefig.facecolor": CFG.MAP_BACKGROUND_COLOR,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# =============================================================================
# FILE / VARIABLE HELPERS
# =============================================================================

@lru_cache(maxsize=None)
def find_year_file(folder: Path, year: int) -> Path:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    matches = sorted(folder.glob(f"*{year}*.nc"))
    if not matches:
        raise FileNotFoundError(
            f"No NetCDF file containing year {year} was found in:\n{folder}"
        )
    if len(matches) > 1:
        print(f"[WARN] Multiple files found for {year} in {folder}. Using {matches[0].name}")
    return matches[0]


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename = {}

    for old, new in (
        ("longitude", "lon"),
        ("latitude", "lat"),
        ("valid_time", "time"),
        ("date", "time"),
        ("pressure_level", "level"),
        ("isobaricInhPa", "level"),
        ("plev", "level"),
    ):
        if old in ds.dims or old in ds.coords:
            if new not in ds.dims and new not in ds.coords:
                rename[old] = new

    if rename:
        ds = ds.rename(rename)

    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise KeyError(
            f"Could not identify lon/lat coordinates. Dataset coords: {list(ds.coords)}"
        )

    # Work internally in 0..360 degrees.  This is essential because the new
    # calculation domain extends to 185 degrees E; converting to -180..180
    # would turn 185E into 175W and break a simple 35..185 slice.
    lon = np.asarray(ds["lon"].values, dtype=float)
    if np.nanmin(lon) < -1.0e-8 or np.nanmax(lon) >= 360.0:
        new_lon = np.mod(lon, 360.0)
        ds = ds.assign_coords(lon=new_lon).sortby("lon")

    if float(ds["lat"][0]) > float(ds["lat"][-1]):
        ds = ds.sortby("lat")

    if float(ds["lon"][0]) > float(ds["lon"][-1]):
        ds = ds.sortby("lon")

    return ds


def resolve_variable(ds: xr.Dataset, candidates: Sequence[str]) -> xr.DataArray:
    lower = {name.lower(): name for name in ds.data_vars}

    for candidate in candidates:
        if candidate in ds.data_vars:
            return ds[candidate]
        if candidate.lower() in lower:
            return ds[lower[candidate.lower()]]

    # Conservative normalized-name fallback.
    def norm(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    normalized = {norm(name): name for name in ds.data_vars}
    for candidate in candidates:
        key = norm(candidate)
        if key in normalized:
            return ds[normalized[key]]

    raise KeyError(
        f"Could not find any of {tuple(candidates)}.\n"
        f"Available data variables: {list(ds.data_vars)}"
    )


def select_pressure_level(da: xr.DataArray, level_hpa: float) -> xr.DataArray:
    level_dim = next(
        (name for name in ("level", "pressure_level", "isobaricInhPa", "plev")
         if name in da.dims or name in da.coords),
        None,
    )

    if level_dim is None:
        # The latest download filenames explicitly encode 500/700 hPa.
        # A field without a level dimension is accepted only if it was already
        # downloaded as one single level.
        return da.squeeze(drop=True)

    values = np.asarray(da[level_dim].values, dtype=float)

    # Some datasets may express pressure in Pa rather than hPa.
    if np.nanmedian(values) > 2000.0:
        target = level_hpa * 100.0
    else:
        target = level_hpa

    index = int(np.nanargmin(np.abs(values - target)))
    selected = float(values[index])

    tolerance = 2.0 if target < 2000 else 200.0
    if abs(selected - target) > tolerance:
        raise ValueError(
            f"Requested {level_hpa} hPa but nearest available level is {selected} "
            f"in coordinate {level_dim!r}."
        )

    return da.isel({level_dim: index}).squeeze(drop=True)


def select_one_month(
    da: xr.DataArray,
    year: int,
    month: int,
) -> xr.DataArray:
    """Select exactly one calendar month from an annual ERA5 file."""
    if "time" not in da.dims:
        raise KeyError(f"No time dimension found in {da.name!r}; dims={da.dims}")

    mask = (da["time"].dt.year == int(year)) & (da["time"].dt.month == int(month))
    out = da.sel(time=mask)

    if out.sizes.get("time", 0) != 1:
        times = pd.DatetimeIndex(pd.to_datetime(out["time"].values))
        raise ValueError(
            f"Expected exactly one field for {year:04d}-{month:02d}, "
            f"but found {out.sizes.get('time', 0)}: "
            f"{times.strftime('%Y-%m').tolist()}"
        )

    return out


def season_month_pairs(
    season_name: str,
    season_year: int,
) -> Tuple[Tuple[int, int], ...]:
    """Return exact (calendar year, calendar month) pairs for one season-year."""
    if season_name not in CFG.SEASON_DEFS:
        raise KeyError(f"Unknown season: {season_name}")

    return tuple(
        (season_year + year_offset, month)
        for year_offset, month in CFG.SEASON_DEFS[season_name]
    )


def season_year_range(season_name: str) -> range:
    """
    Return candidate analysis season-years, 1961--2023.

    Whether all three calendar months have phase classifications is checked
    separately by supported_season_years().  This separation prevents raw
    ERA5 coverage from being mistaken for ENSO-phase support.
    """
    if season_name not in CFG.SEASON_DEFS:
        raise KeyError(f"Unknown season: {season_name}")

    first_pairs = season_month_pairs(season_name, CFG.START_YEAR)
    last_pairs = season_month_pairs(season_name, CFG.END_YEAR)
    required_years = [year for year, _ in (*first_pairs, *last_pairs)]
    if min(required_years) < CFG.INPUT_START_YEAR or max(required_years) > CFG.INPUT_END_YEAR:
        raise ValueError(
            f"{season_name}: analysis years {CFG.START_YEAR}-{CFG.END_YEAR} "
            f"require raw inputs outside {CFG.INPUT_START_YEAR}-{CFG.INPUT_END_YEAR}."
        )

    return range(CFG.START_YEAR, CFG.END_YEAR + 1)


def supported_season_years(
    season_name: str,
    phase_table: pd.DataFrame,
) -> Tuple[int, ...]:
    """Return only season-years whose three months have phase records.

    This is especially important for DJF.  When the phase source starts at
    January 1961, DJF 1961 would require an unknown December 1960 phase and is
    therefore excluded; DJF then begins in 1962.  No month is imputed and no
    two-month DJF average is allowed.
    """
    available_pairs = set(
        zip(
            phase_table["cal_year"].astype(int),
            phase_table["cal_month"].astype(int),
        )
    )

    supported: list[int] = []
    excluded: list[Tuple[int, Tuple[Tuple[int, int], ...]]] = []

    for season_year in season_year_range(season_name):
        missing = tuple(
            pair
            for pair in season_month_pairs(season_name, season_year)
            if pair not in available_pairs
        )
        if missing:
            excluded.append((season_year, missing))
        else:
            supported.append(season_year)

    if not supported:
        raise ValueError(
            f"{season_name}: no season-year has all three ENSO phase records."
        )

    if excluded:
        detail = "; ".join(
            f"{year}: " + ", ".join(f"{y:04d}-{m:02d}" for y, m in missing)
            for year, missing in excluded
        )
        print(
            f"[PHASE SUPPORT | {season_name}] excluded incomplete "
            f"season-year(s): {detail}"
        )

    print(
        f"[PHASE SUPPORT | {season_name}] complete three-month candidates: "
        f"{supported[0]}-{supported[-1]} (n={len(supported)})"
    )
    return tuple(supported)


def crop_to_extent(
    da: xr.DataArray,
    extent: Sequence[float],
    label: str,
) -> xr.DataArray:
    """Crop a normalized 0..360 field and verify that the crop is non-empty."""
    west, east, south, north = map(float, extent)

    if not (0.0 <= west < east <= 360.0):
        raise ValueError(
            f"{label}: longitude extent must be an increasing 0..360 interval; "
            f"received {extent}."
        )
    if not (-90.0 <= south < north <= 90.0):
        raise ValueError(f"{label}: invalid latitude extent {extent}.")

    cropped = da.sel(lon=slice(west, east), lat=slice(south, north))
    if cropped.sizes.get("lon", 0) < 3 or cropped.sizes.get("lat", 0) < 3:
        raise ValueError(
            f"{label}: crop is empty or too small. Requested {extent}; "
            f"available lon={float(da.lon.min()):g}..{float(da.lon.max()):g}, "
            f"lat={float(da.lat.min()):g}..{float(da.lat.max()):g}."
        )
    return cropped


def crop_to_calc(da: xr.DataArray) -> xr.DataArray:
    return crop_to_extent(da, CFG.CALC_EXTENT, "CALC_EXTENT")


def crop_to_plot(da: xr.DataArray) -> xr.DataArray:
    return crop_to_extent(da, CFG.PLOT_EXTENT, "PLOT_EXTENT")



def group_month_pairs_by_year(
    month_pairs: Sequence[Tuple[int, int]],
) -> Dict[int, Tuple[int, ...]]:
    """Group requested season months by annual file so each file is opened once."""
    grouped: Dict[int, list[int]] = {}
    for year, month in month_pairs:
        grouped.setdefault(int(year), []).append(int(month))
    return {year: tuple(months) for year, months in grouped.items()}


def load_uv_season(
    month_pairs: Sequence[Tuple[int, int]],
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Load U/V at 500 and 700 hPa for one season.

    SPEED FIX:
    The previous version opened the same annual UV NetCDF once for every month.
    JJA/MAM/SON therefore opened one annual file three times.  This version
    groups months by year and opens each annual UV file only once.
    """
    requested = tuple((int(y), int(m)) for y, m in month_pairs)
    grouped = group_month_pairs_by_year(requested)
    store: Dict[str, Dict[Tuple[int, int], xr.DataArray]] = {
        "u500": {}, "v500": {}, "u700": {}, "v700": {}
    }

    for year, months in grouped.items():
        path = find_year_file(CFG.UV_DIR, year)
        with xr.open_dataset(path) as raw:
            ds = normalize_dataset(raw)
            u = resolve_variable(ds, ("u", "u_component_of_wind", "ucomponentofwind"))
            v = resolve_variable(ds, ("v", "v_component_of_wind", "vcomponentofwind"))

            u500_all = select_pressure_level(u, 500.0)
            v500_all = select_pressure_level(v, 500.0)
            u700_all = select_pressure_level(u, 700.0)
            v700_all = select_pressure_level(v, 700.0)

            for month in months:
                pair = (year, month)
                store["u500"][pair] = crop_to_calc(
                    select_one_month(u500_all, year, month)
                ).load().astype("float32")
                store["v500"][pair] = crop_to_calc(
                    select_one_month(v500_all, year, month)
                ).load().astype("float32")
                store["u700"][pair] = crop_to_calc(
                    select_one_month(u700_all, year, month)
                ).load().astype("float32")
                store["v700"][pair] = crop_to_calc(
                    select_one_month(v700_all, year, month)
                ).load().astype("float32")

    return (
        xr.concat([store["u500"][pair] for pair in requested], dim="time"),
        xr.concat([store["v500"][pair] for pair in requested], dim="time"),
        xr.concat([store["u700"][pair] for pair in requested], dim="time"),
        xr.concat([store["v700"][pair] for pair in requested], dim="time"),
    )


def load_single_pressure_season(
    folder: Path,
    month_pairs: Sequence[Tuple[int, int]],
    candidates: Sequence[str],
    level_hpa: float = 700.0,
) -> xr.DataArray:
    """
    Load one pressure-level variable for a season.

    SPEED FIX:
    Months that belong to the same calendar year are read from one open annual
    NetCDF handle instead of reopening that file for every month.
    """
    requested = tuple((int(y), int(m)) for y, m in month_pairs)
    grouped = group_month_pairs_by_year(requested)
    fields: Dict[Tuple[int, int], xr.DataArray] = {}

    for year, months in grouped.items():
        path = find_year_file(folder, year)
        with xr.open_dataset(path) as raw:
            ds = normalize_dataset(raw)
            da_all = resolve_variable(ds, candidates)
            da_all = select_pressure_level(da_all, level_hpa)

            for month in months:
                pair = (year, month)
                da = select_one_month(da_all, year, month)
                da = crop_to_calc(da)
                fields[pair] = da.load().astype("float32")

    return xr.concat([fields[pair] for pair in requested], dim="time")


def load_surface_pressure_season(
    month_pairs: Sequence[Tuple[int, int]],
) -> xr.DataArray:
    """Load surface pressure for one season, opening each annual file once."""
    requested = tuple((int(y), int(m)) for y, m in month_pairs)
    grouped = group_month_pairs_by_year(requested)
    fields: Dict[Tuple[int, int], xr.DataArray] = {}

    for year, months in grouped.items():
        path = find_year_file(CFG.SP_DIR, year)
        with xr.open_dataset(path) as raw:
            ds = normalize_dataset(raw)
            sp_all = resolve_variable(ds, ("sp", "surface_pressure", "surfacepressure"))

            for month in months:
                pair = (year, month)
                sp = crop_to_calc(
                    select_one_month(sp_all, year, month)
                ).load().astype("float32")
                fields[pair] = sp

    sp = xr.concat([fields[pair] for pair in requested], dim="time")

    median = float(sp.median(skipna=True))
    if median > 2000.0:
        sp = sp / 100.0

    sp.attrs["units"] = "hPa"
    return sp


# =============================================================================
# ENSO PHASE TABLE
# =============================================================================

def load_phase_table() -> pd.DataFrame:
    path = CFG.PHASE_SOURCE

    if not path.exists():
        raise FileNotFoundError(
            f"ENSO phase source not found:\n{path}"
        )

    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    else:
        with xr.open_dataset(path) as ds:
            time_name = next(
                (
                    name for name in ("time", "valid_time")
                    if name in ds.coords or name in ds.variables
                ),
                None,
            )
            if time_name is None:
                raise KeyError("Phase NetCDF has no time/valid_time coordinate.")

            time_index = pd.DatetimeIndex(pd.to_datetime(ds[time_name].values))
            ntime = len(time_index)

            def get_time_vector(names: Sequence[str], default: np.ndarray) -> np.ndarray:
                for name in names:
                    if name in ds.coords or name in ds.data_vars:
                        candidate = ds[name]
                        if time_name in candidate.dims:
                            values = np.asarray(candidate.values).ravel()
                            if values.size == ntime:
                                return values
                return np.asarray(default)

            cal_year = get_time_vector(("cal_year", "year"), time_index.year.to_numpy())
            cal_month = get_time_vector(("cal_month", "month"), time_index.month.to_numpy())

            phase = None
            for name in ("phase_code", "enso_phase_code", "ENSO_phase_code"):
                if name in ds.coords or name in ds.data_vars:
                    candidate = ds[name]
                    if time_name in candidate.dims:
                        values = np.asarray(candidate.values).ravel()
                        if values.size == ntime:
                            phase = values
                            break

            if phase is None:
                raise KeyError(
                    "No time-resolved phase_code was found in the phase source."
                )

            table = pd.DataFrame({
                "cal_year": cal_year,
                "cal_month": cal_month,
                "phase_code": phase,
            })

    required = {"cal_year", "cal_month", "phase_code"}
    if not required.issubset(table.columns):
        raise KeyError(f"Phase table must contain {sorted(required)}")

    table = table.loc[:, ["cal_year", "cal_month", "phase_code"]].copy()
    table = table.apply(pd.to_numeric, errors="coerce").dropna().astype(int)

    table = table[
        # Keep all phase records provided within the raw archive span, while
        # formally validating January 1961--December 2023 below.
        table["cal_year"].between(CFG.INPUT_START_YEAR, CFG.INPUT_END_YEAR)
        & table["cal_month"].between(1, 12)
        & table["phase_code"].isin((-1, 0, 1))
    ]

    table = (
        table
        .drop_duplicates(["cal_year", "cal_month"])
        .sort_values(["cal_year", "cal_month"])
        .reset_index(drop=True)
    )

    # The formal calendar-month record itself must be complete.  December
    # 1960 is deliberately not required: if absent, supported_season_years()
    # excludes DJF 1961 without inventing an ENSO category.
    available_pairs = set(
        zip(table["cal_year"].astype(int), table["cal_month"].astype(int))
    )
    expected_pairs = {
        (year, month)
        for year in range(CFG.START_YEAR, CFG.END_YEAR + 1)
        for month in range(1, 13)
    }
    missing_pairs = sorted(expected_pairs - available_pairs)
    if missing_pairs:
        preview = ", ".join(
            f"{year:04d}-{month:02d}" for year, month in missing_pairs[:12]
        )
        suffix = " ..." if len(missing_pairs) > 12 else ""
        raise ValueError(
            "The ENSO phase table is incomplete within the formal "
            f"1961-2023 monthly analysis period. Missing: {preview}{suffix}"
        )

    if (CFG.START_YEAR - 1, 12) not in available_pairs:
        print(
            "[PHASE SUPPORT | DJF] December 1960 is absent; DJF 1961 will "
            "be excluded and DJF composites will begin with DJF 1962."
        )

    return table


def classify_season_pairs(
    phase_table: pd.DataFrame,
    month_pairs: Sequence[Tuple[int, int]],
) -> Optional[int]:
    """
    Classify one three-month season.

    Neutral: all three months are Neutral.
    ENSO: at least two months have the same ENSO phase and no month has the
    opposite ENSO phase. This preserves the previous Figure 7 rule while
    correctly handling DJF across two calendar years.
    """

    codes = []

    for year, month in month_pairs:
        subset = phase_table[
            (phase_table["cal_year"] == int(year))
            & (phase_table["cal_month"] == int(month))
        ]

        if len(subset) != 1:
            return None

        codes.append(int(subset.iloc[0]["phase_code"]))

    if all(code == 0 for code in codes):
        return 0

    majority = len(codes) // 2 + 1

    if sum(code == 1 for code in codes) >= majority and -1 not in codes:
        return 1

    if sum(code == -1 for code in codes) >= majority and 1 not in codes:
        return -1

    return None

# =============================================================================
# PHYSICAL DIAGNOSTICS
# =============================================================================

EARTH_RADIUS_M = 6_371_000.0


def spherical_horizontal_divergence(
    flux_u: xr.DataArray,
    flux_v: xr.DataArray,
) -> xr.DataArray:
    """
    Horizontal divergence on a regular latitude-longitude grid:

        div(F) = 1/(R cos(phi)) dF_u/dlambda
               + 1/(R cos(phi)) d(F_v cos(phi))/dphi

    The derivative is calculated over the full downloaded domain first.
    """

    flux_u, flux_v = xr.align(flux_u, flux_v, join="exact")

    coslat = xr.DataArray(
        np.cos(np.deg2rad(flux_u["lat"].values)),
        dims=("lat",),
        coords={"lat": flux_u["lat"]},
    )

    # xarray differentiate() returns derivative per degree.
    # Convert to derivative per radian by multiplying by 180/pi.
    deg_to_rad = 180.0 / np.pi

    du_dlambda = flux_u.differentiate("lon") * deg_to_rad
    dvcos_dphi = (flux_v * coslat).differentiate("lat") * deg_to_rad

    safe_coslat = xr.where(np.abs(coslat) > 1.0e-6, coslat, np.nan)

    divergence = (
        du_dlambda + dvcos_dphi
    ) / (EARTH_RADIUS_M * safe_coslat)

    return divergence


def complete_season_mean(da: xr.DataArray) -> xr.DataArray:
    """
    Use a season-year grid cell only if all three constituent months are finite.
    """
    complete = da.notnull().all("time")
    return da.mean("time", skipna=True).where(complete)


def apply_pressure_masks(
    u500: xr.DataArray,
    v500: xr.DataArray,
    u700: xr.DataArray,
    v700: xr.DataArray,
    q700: xr.DataArray,
    rh700: xr.DataArray,
    omega700: xr.DataArray,
    sp_hpa: xr.DataArray,
) -> Tuple[xr.DataArray, ...]:

    clearance = CFG.PRESSURE_CLEARANCE_HPA

    mask500 = sp_hpa >= (500.0 + clearance)
    mask700 = sp_hpa >= (700.0 + clearance)

    return (
        u500.where(mask500),
        v500.where(mask500),
        u700.where(mask700),
        v700.where(mask700),
        q700.where(mask700),
        rh700.where(mask700),
        omega700.where(mask700),
    )


# =============================================================================
# BUILD SEASON-YEAR DATA
# =============================================================================

def build_seasonal_dataset(
    season_name: str,
    phase_table: pd.DataFrame,
) -> Tuple[xr.Dataset, np.ndarray, np.ndarray]:

    storage: Dict[str, list[np.ndarray]] = {
        "div500": [],
        "u700": [],
        "v700": [],
        "mfc700": [],
        "qfu700": [],
        "qfv700": [],
        "ascent700": [],
        "rh700": [],
    }

    sample_years: list[int] = []
    sample_codes: list[int] = []

    out_lon = None
    out_lat = None

    for season_year in supported_season_years(season_name, phase_table):
        month_pairs = season_month_pairs(season_name, season_year)
        season_code = classify_season_pairs(phase_table, month_pairs)

        if season_code is None:
            continue

        month_text = ", ".join(f"{y:04d}-{m:02d}" for y, m in month_pairs)
        phase_text = (
            "El Niño" if season_code == 1
            else "La Niña" if season_code == -1
            else "Neutral"
        )
        print(f"[{season_name}] {season_year}: {phase_text} | {month_text}")

        u500, v500, u700, v700 = load_uv_season(month_pairs)

        q700 = load_single_pressure_season(
            CFG.Q_DIR,
            month_pairs,
            ("q", "specific_humidity", "specifichumidity"),
            700.0,
        )

        rh700 = load_single_pressure_season(
            CFG.RH_DIR,
            month_pairs,
            ("r", "rh", "relative_humidity", "relativehumidity"),
            700.0,
        )

        omega700 = load_single_pressure_season(
            CFG.OMEGA_DIR,
            month_pairs,
            ("w", "omega", "vertical_velocity", "verticalvelocity"),
            700.0,
        )

        (
            u500, v500, u700, v700,
            q700, rh700, omega700
        ) = xr.align(
            u500, v500, u700, v700,
            q700, rh700, omega700,
            join="inner",
        )

        if u500.sizes.get("time", 0) != 3:
            raise ValueError(
                f"{season_name} {season_year}: aligned data do not contain all three months: "
                f"{month_pairs}"
            )

        if CFG.USE_SURFACE_PRESSURE_MASK:
            sp = load_surface_pressure_season(month_pairs)
            (
                u500, v500, u700, v700,
                q700, rh700, omega700, sp
            ) = xr.align(
                u500, v500, u700, v700,
                q700, rh700, omega700, sp,
                join="inner",
            )

            (
                u500, v500, u700, v700,
                q700, rh700, omega700
            ) = apply_pressure_masks(
                u500, v500, u700, v700,
                q700, rh700, omega700, sp
            )

        # Calculate derivatives on the large downloaded domain FIRST.
        div500 = spherical_horizontal_divergence(u500, v500)

        qfu = q700 * u700
        qfv = q700 * v700
        mfc700 = -spherical_horizontal_divergence(qfu, qfv)
        ascent700 = -omega700

        # Units used in plotting.
        div500 = div500 * 1.0e6
        div500.attrs["units"] = "10^-6 s^-1"

        mfc700 = mfc700 * 1.0e8
        mfc700.attrs["units"] = "10^-8 s^-1"

        qfu_plot = qfu * 1000.0
        qfv_plot = qfv * 1000.0
        qfu_plot.attrs["units"] = "g kg^-1 m s^-1"
        qfv_plot.attrs["units"] = "g kg^-1 m s^-1"

        ascent700.attrs["units"] = "Pa s^-1"
        rh700.attrs["units"] = "%"

        # Crop only after divergence/MFC are calculated.
        fields = {
            "div500": crop_to_plot(div500),
            "u700": crop_to_plot(u700),
            "v700": crop_to_plot(v700),
            "mfc700": crop_to_plot(mfc700),
            "qfu700": crop_to_plot(qfu_plot),
            "qfv700": crop_to_plot(qfv_plot),
            "ascent700": crop_to_plot(ascent700),
            "rh700": crop_to_plot(rh700),
        }

        seasonal_fields = {
            name: complete_season_mean(da)
            for name, da in fields.items()
        }

        if out_lon is None:
            out_lon = np.asarray(seasonal_fields["div500"]["lon"].values, dtype=float)
            out_lat = np.asarray(seasonal_fields["div500"]["lat"].values, dtype=float)

        for name, da in seasonal_fields.items():
            da = da.transpose("lat", "lon")
            storage[name].append(np.asarray(da.values, dtype="float32"))

        sample_years.append(season_year)
        sample_codes.append(season_code)

    if not sample_years:
        raise RuntimeError(f"No classified {season_name} season-year samples were constructed.")

    assert out_lon is not None and out_lat is not None

    dataset = xr.Dataset(
        {
            name: (("sample_year", "lat", "lon"), np.stack(values, axis=0))
            for name, values in storage.items()
        },
        coords={
            "sample_year": np.asarray(sample_years, dtype=int),
            "lat": out_lat,
            "lon": out_lon,
        },
    )

    codes = np.asarray(sample_codes, dtype=int)
    years = np.asarray(sample_years, dtype=int)

    print()
    print(f"[SAMPLES | {season_name}]")
    print(f"  El Niño : {np.sum(codes == 1)}")
    print(f"  La Niña : {np.sum(codes == -1)}")
    print(f"  Neutral : {np.sum(codes == 0)}")
    print(f"  Retained season-years: {years.min()}-{years.max()} (n={len(years)})")

    return dataset, years, codes

# =============================================================================
# DETREND + COMPOSITES
# =============================================================================

def linear_detrend_cube(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    """
    Remove intercept and linear year trend independently at each grid cell.
    """

    y = np.asarray(values, dtype=float)
    x = np.asarray(years, dtype=float)

    x = x - np.nanmean(x)
    reshape = (len(x),) + (1,) * (y.ndim - 1)
    xx = x.reshape(reshape)

    valid = np.isfinite(y)

    n = valid.sum(axis=0).astype(float)
    sx = np.sum(np.where(valid, xx, 0.0), axis=0)
    sy = np.nansum(y, axis=0)
    sxx = np.sum(np.where(valid, xx * xx, 0.0), axis=0)
    sxy = np.nansum(np.where(valid, y * xx, np.nan), axis=0)

    denominator = n * sxx - sx * sx

    slope = np.divide(
        n * sxy - sx * sy,
        denominator,
        out=np.full_like(sy, np.nan, dtype=float),
        where=(n >= 3) & (denominator > 0),
    )

    intercept = np.divide(
        sy - slope * sx,
        n,
        out=np.full_like(sy, np.nan, dtype=float),
        where=n > 0,
    )

    fitted = intercept[None, ...] + slope[None, ...] * xx

    return np.where(valid, y - fitted, np.nan)


def composite_difference(
    da: xr.DataArray,
    years: np.ndarray,
    codes: np.ndarray,
    phase_code: int,
) -> np.ndarray:

    residual = linear_detrend_cube(
        np.asarray(da.values, dtype=float),
        years,
    )

    phase = residual[codes == phase_code]
    neutral = residual[codes == 0]

    n_phase = np.sum(np.isfinite(phase), axis=0)
    n_neutral = np.sum(np.isfinite(neutral), axis=0)

    with np.errstate(invalid="ignore"):
        effect = (
            np.nanmean(phase, axis=0)
            -
            np.nanmean(neutral, axis=0)
        )

    valid = (
        (n_phase >= CFG.MIN_PHASE_N)
        & (n_neutral >= CFG.MIN_NEUTRAL_N)
    )

    return np.where(valid, effect, np.nan)


def compute_composites(
    seasonal: xr.Dataset,
    years: np.ndarray,
    codes: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:

    variables = tuple(seasonal.data_vars)

    result: Dict[str, Dict[str, np.ndarray]] = {}

    for phase_name in PHASES:
        phase_code = PHASE_CODE[phase_name]
        result[phase_name] = {}

        for variable in variables:
            print(f"[COMPOSITE] {phase_name} | {variable}")

            result[phase_name][variable] = composite_difference(
                seasonal[variable],
                years,
                codes,
                phase_code,
            )

    return result


def phase_neutral_significance_mask(
    da: xr.DataArray,
    years: np.ndarray,
    codes: np.ndarray,
    phase_code: int,
) -> np.ndarray:
    """Welch two-sample t-test on detrended season-year anomalies."""

    residual = linear_detrend_cube(
        np.asarray(da.values, dtype=float),
        years,
    )

    phase = residual[codes == phase_code]
    neutral = residual[codes == 0]

    n_phase = np.sum(np.isfinite(phase), axis=0)
    n_neutral = np.sum(np.isfinite(neutral), axis=0)
    valid = (n_phase >= CFG.MIN_PHASE_N) & (n_neutral >= CFG.MIN_NEUTRAL_N)

    with np.errstate(invalid="ignore"):
        _stat, pvals = stats.ttest_ind(
            phase,
            neutral,
            axis=0,
            equal_var=False,
            nan_policy="omit",
        )

    mask = valid & np.isfinite(pvals) & (pvals < CFG.SIGNIFICANCE_ALPHA)
    return np.asarray(mask, dtype=bool)


def compute_significance_masks(
    seasonal: xr.Dataset,
    years: np.ndarray,
    codes: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Gridpoint significance masks for the shaded physical diagnostics."""

    shaded_variables = ("div500", "mfc700", "ascent700")
    result: Dict[str, Dict[str, np.ndarray]] = {}

    for phase_name in PHASES:
        phase_code = PHASE_CODE[phase_name]
        result[phase_name] = {}
        for variable in shaded_variables:
            print(f"[SIGNIFICANCE] {phase_name} | {variable}")
            result[phase_name][variable] = phase_neutral_significance_mask(
                seasonal[variable],
                years,
                codes,
                phase_code,
            )

    return result


# =============================================================================
# SEASON EFFECT CACHE
# =============================================================================

def season_effect_cache_signature() -> str:
    """Configuration signature used to reject stale seasonal effect caches."""
    return "|".join([
        str(CFG.SEASON_EFFECT_CACHE_VERSION),
        f"years={CFG.START_YEAR}-{CFG.END_YEAR}",
        f"calc={tuple(CFG.CALC_EXTENT)}",
        f"plot={tuple(CFG.PLOT_EXTENT)}",
        f"mask={int(CFG.USE_SURFACE_PRESSURE_MASK)}",
        f"clearance={CFG.PRESSURE_CLEARANCE_HPA:g}",
        f"minphase={CFG.MIN_PHASE_N}",
        f"minneutral={CFG.MIN_NEUTRAL_N}",
    ])


def season_effect_cache_path(season_name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", season_name)
    return CFG.SEASON_EFFECT_CACHE_DIR / f"Figure7_{safe}_composite_effects.nc"


def save_season_effect_cache(
    season_name: str,
    lon: np.ndarray,
    lat: np.ndarray,
    effects: Dict[str, Dict[str, np.ndarray]],
    significance: Dict[str, Dict[str, np.ndarray]],
) -> Path:
    """
    Save only the final detrended ENSO-minus-Neutral effects needed for plotting.

    These files are small and make later figure-only edits very fast because the
    1961-2023 ERA5 archive no longer has to be reread.
    """
    path = season_effect_cache_path(season_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    phase_prefix = {"El Nino": "EN", "La Nina": "LN"}
    data_vars = {}
    for phase_name in PHASES:
        prefix = phase_prefix[phase_name]
        for variable, values in effects[phase_name].items():
            data_vars[f"{prefix}_{variable}"] = (
                ("lat", "lon"),
                np.asarray(values, dtype="float32"),
            )
        for variable, values in significance[phase_name].items():
            data_vars[f"{prefix}_{variable}_sig"] = (
                ("lat", "lon"),
                np.asarray(values, dtype="int8"),
            )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "lat": np.asarray(lat, dtype=float),
            "lon": np.asarray(lon, dtype=float),
        },
        attrs={
            "season": season_name,
            "cache_signature": season_effect_cache_signature(),
            "description": (
                "Detrended ENSO-minus-Neutral Figure 7 composite effects. "
                "Safe for layout-only redraws."
            ),
        },
    )

    tmp = path.with_name(path.stem + ".tmp.nc")
    if tmp.exists():
        tmp.unlink()
    ds.to_netcdf(tmp)
    ds.close()
    tmp.replace(path)
    print(f"[CACHE WRITE] {season_name}: {path}")
    return path


def load_season_effect_cache(
    season_name: str,
) -> Optional[Dict[str, object]]:
    """Load a valid seasonal effect cache; return None when absent/stale."""
    if not CFG.USE_SEASON_EFFECT_CACHE:
        return None
    if CFG.FORCE_RECOMPUTE_SEASON_EFFECT_CACHE:
        return None

    path = season_effect_cache_path(season_name)
    if not path.exists():
        return None

    try:
        with xr.open_dataset(path) as ds:
            signature = str(ds.attrs.get("cache_signature", ""))
            if signature != season_effect_cache_signature():
                print(f"[CACHE STALE] {season_name}: configuration changed -> recompute")
                return None

            lon = np.asarray(ds["lon"].values, dtype=float)
            lat = np.asarray(ds["lat"].values, dtype=float)

            effects: Dict[str, Dict[str, np.ndarray]] = {
                "El Nino": {},
                "La Nina": {},
            }
            significance: Dict[str, Dict[str, np.ndarray]] = {
                "El Nino": {},
                "La Nina": {},
            }
            for phase_name, prefix in (("El Nino", "EN"), ("La Nina", "LN")):
                prefix_token = prefix + "_"
                for name in ds.data_vars:
                    if not name.startswith(prefix_token):
                        continue
                    variable = name[len(prefix_token):]
                    if variable.endswith("_sig"):
                        variable = variable[:-4]
                        significance[phase_name][variable] = np.asarray(
                            ds[name].values,
                            dtype=bool,
                        )
                    else:
                        effects[phase_name][variable] = np.asarray(
                            ds[name].values,
                            dtype=float,
                        )
    except Exception as exc:
        print(f"[CACHE READ FAILED] {season_name}: {exc}")
        return None

    required = {
        "div500", "u700", "v700", "mfc700",
        "qfu700", "qfv700", "ascent700", "rh700",
    }
    if any(not required.issubset(effects[phase]) for phase in PHASES):
        print(f"[CACHE INVALID] {season_name}: missing effect variables -> recompute")
        return None
    if any(not {"div500", "mfc700", "ascent700"}.issubset(significance[phase]) for phase in PHASES):
        print(f"[CACHE INVALID] {season_name}: missing significance masks -> recompute")
        return None

    print(f"[CACHE HIT] {season_name}: raw ERA5 calculation skipped")
    return {
        "lon": lon,
        "lat": lat,
        "effects": effects,
        "significance": significance,
    }


# =============================================================================
# SIX BASINS
# =============================================================================

def normalize_text(value: object) -> str:
    return re.sub(
        r"[\s_\-–—/\\()\[\]{}]+",
        "",
        str(value).strip().lower(),
    )


def canonical_basin_name_from_row(row: pd.Series) -> Optional[str]:
    text_parts = []

    for column, value in row.items():
        if column == "geometry" or pd.isna(value):
            continue
        text_parts.append(str(value))

    text = normalize_text(" ".join(text_parts))

    for canonical, aliases in BASIN_ALIASES.items():
        for alias in aliases:
            if normalize_text(alias) in text:
                return canonical

    return None


def load_six_basins() -> gpd.GeoDataFrame:
    path = CFG.BASIN_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Official six-basin shapefile not found:\n{path}"
        )

    basins = gpd.read_file(path)

    if basins.crs is None:
        raise ValueError("The six-basin shapefile has no CRS.")

    basins = basins.to_crs("EPSG:4326").copy()

    basins["basin_name"] = basins.apply(
        canonical_basin_name_from_row,
        axis=1,
    )

    basins = basins[
        basins["basin_name"].isin(BASIN_ORDER)
    ].copy()

    basins = basins.dissolve(
        by="basin_name",
        as_index=False,
    )

    found = set(basins["basin_name"])
    missing = [
        basin for basin in BASIN_ORDER
        if basin not in found
    ]

    if missing:
        print("\n[BASIN DEBUG] Available columns:")
        print(list(gpd.read_file(path).columns))

        raise ValueError(
            "Could not identify all six official basins. Missing: "
            + ", ".join(missing)
            + "\nUpdate BASIN_ALIASES only if your official shapefile "
              "uses different basin names."
        )

    basins["basin_name"] = pd.Categorical(
        basins["basin_name"],
        categories=BASIN_ORDER,
        ordered=True,
    )

    return (
        basins
        .sort_values("basin_name")
        .reset_index(drop=True)
    )


# =============================================================================
# MAP BACKGROUND
# =============================================================================

_COUNTRY_GEOMETRY_CACHE = None


def get_country_geometry_groups():
    """Return cached Natural Earth study-country and other-country geometry."""
    global _COUNTRY_GEOMETRY_CACHE
    if _COUNTRY_GEOMETRY_CACHE is not None:
        return _COUNTRY_GEOMETRY_CACHE

    shp = shpreader.natural_earth(
        resolution="50m",
        category="cultural",
        name="admin_0_countries",
    )

    study_geometries = []
    other_geometries = []
    study_names = {"china", "taiwan"}
    west, east, south, north = CFG.PLOT_EXTENT
    visible_box = box(west, south, east, north)

    for record in shpreader.Reader(shp).records():
        attrs = record.attributes

        name = (
            attrs.get("ADMIN")
            or attrs.get("NAME")
            or attrs.get("NAME_LONG")
            or ""
        )

        geometry = record.geometry
        if geometry is None or geometry.is_empty:
            continue
        if not geometry.intersects(visible_box):
            continue

        if str(name).strip().lower() in study_names:
            study_geometries.append(geometry)
        else:
            other_geometries.append(geometry)

    _COUNTRY_GEOMETRY_CACHE = (study_geometries, other_geometries)
    return _COUNTRY_GEOMETRY_CACHE


def get_china_geometry():
    """Compatibility wrapper returning the study-country geometry."""
    study_geometries, _ = get_country_geometry_groups()
    return study_geometries


def shade_other_countries(ax) -> None:
    """Optionally overlay a neutral-grey mask on countries outside China/Taiwan."""
    if not CFG.OVERLAY_OTHER_COUNTRIES_AFTER_SHADING:
        return

    _, other_geometries = get_country_geometry_groups()
    if other_geometries:
        ax.add_geometries(
            other_geometries,
            crs=ccrs.PlateCarree(),
            facecolor=CFG.OTHER_COUNTRY_FACE,
            edgecolor="none",
            linewidth=0.0,
            zorder=4,
        )


def draw_map_base(
    ax,
    basins: gpd.GeoDataFrame,
    show_basin_labels: bool = True,
) -> None:

    west, east, south, north = CFG.PLOT_EXTENT

    # Keep the drawing canvas consistent with the map background so masked shading regions
    # are covered by the same light-grey colour as land and ocean.
    ax.set_facecolor(CFG.OTHER_COUNTRY_FACE)

    ax.set_extent(
        (west, east, south, north),
        crs=ccrs.PlateCarree(),
    )

    # Reference-paper style: light-grey background over both land and ocean.
    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor=CFG.OTHER_COUNTRY_FACE,
        edgecolor="none",
        zorder=0,
    )

    ax.add_feature(
        cfeature.OCEAN.with_scale("50m"),
        facecolor=CFG.OCEAN_FACE,
        edgecolor="none",
        zorder=0,
    )

    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.60,
        edgecolor="#333333",
        zorder=6,
    )

    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.45,
        edgecolor="#707070",
        zorder=5,
    )

    # Keep the study-country land white where a diagnostic field is missing.
    china_geoms = get_china_geometry()

    if china_geoms:
        ax.add_geometries(
            china_geoms,
            crs=ccrs.PlateCarree(),
            facecolor=CFG.CHINA_FACE,
            edgecolor="none",
            linewidth=0.0,
            zorder=1,
        )
        ax.add_geometries(
            china_geoms,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="#202020",
            linewidth=0.85,
            zorder=7,
        )

    # Six basin boundaries.
    basins.boundary.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        color=CFG.BASIN_COLOR,
        linewidth=CFG.BASIN_LINEWIDTH,
        linestyle=CFG.BASIN_LINESTYLE,
        zorder=8,
    )

    if show_basin_labels:
        for row in basins.itertuples():
            point = row.geometry.representative_point()

            ax.text(
                point.x,
                point.y,
                BASIN_LABEL[str(row.basin_name)],
                transform=ccrs.PlateCarree(),
                fontsize=CFG.BASIN_LABEL_SIZE,
                fontweight="bold",
                ha="center",
                va="center",
                color=CFG.BASIN_COLOR,
                zorder=9,
                path_effects=[
                    pe.withStroke(
                        linewidth=2.0,
                        foreground="white",
                    )
                ],
            )

    # Reference-like longitude/latitude ticks.
    ax.set_xticks(
        CFG.LON_TICKS,
        crs=ccrs.PlateCarree(),
    )

    ax.set_yticks(
        CFG.LAT_TICKS,
        crs=ccrs.PlateCarree(),
    )

    ax.xaxis.set_major_formatter(
        LongitudeFormatter(
            zero_direction_label=False,
            dateline_direction_label=True,
            degree_symbol="°",
        )
    )

    ax.yaxis.set_major_formatter(
        LatitudeFormatter(
            degree_symbol="°",
        )
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.0,
        width=0.8,
        pad=2.0,
        labelsize=CFG.TICK_SIZE,
    )

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_fontfamily(CFG.FONT_FAMILY)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#303030")


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def _symlog_boundaries(vmax: float, linthresh: float, n_log_each_side: int = 8) -> np.ndarray:
    """Return symmetric-log-style contour boundaries in data space."""

    vmax = float(abs(vmax))
    linthresh = float(abs(linthresh))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    if not np.isfinite(linthresh) or linthresh <= 0:
        linthresh = max(vmax * 0.1, 1.0e-6)
    if vmax <= linthresh:
        return np.linspace(-vmax, vmax, max(5, 2 * n_log_each_side + 1))

    positive = np.geomspace(linthresh, vmax, int(max(3, n_log_each_side)))
    positive = np.unique(np.asarray(positive, dtype=float))
    negative = -positive[::-1]
    return np.concatenate([negative, [0.0], positive])


def symmetric_levels(
    arrays: Iterable[np.ndarray],
    n_intervals: int = 12,
    shade_name: Optional[str] = None,
) -> np.ndarray:

    values = []

    for array in arrays:
        a = np.asarray(array, dtype=float)
        finite = np.abs(a[np.isfinite(a)])
        if finite.size:
            values.append(finite)

    if not values:
        return np.linspace(-1.0, 1.0, n_intervals + 1)

    all_values = np.concatenate(values)

    vmax = float(
        np.nanpercentile(
            all_values,
            CFG.ROBUST_PERCENTILE,
        )
    )

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(all_values))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fixed_max_map = getattr(CFG, "FIXED_DISPLAY_MAX_BY_SHADE", {})
    if shade_name is not None and shade_name in fixed_max_map:
        fixed_max = float(fixed_max_map[shade_name])
        if np.isfinite(fixed_max) and fixed_max > 0:
            vmax = fixed_max

    linthresh_map = getattr(CFG, "SYMLOG_LINTHRESH_BY_SHADE", {})
    linthresh = float(linthresh_map.get(shade_name, max(vmax * 0.1, 1.0e-6)))

    custom_map = getattr(CFG, "CUSTOM_DISPLAY_LEVELS_BY_SHADE", {})
    if shade_name is not None and shade_name in custom_map:
        return _deduplicate_sorted_ticks(np.asarray(custom_map[shade_name], dtype=float), atol=1.0e-12)

    # Use a true symmetric-log-style set of boundaries in data space.  The
    # plotting norm is also transformed accordingly so both the field rendering
    # and the colour bar use the same logic.
    level_count_map = getattr(CFG, "DISPLAY_LEVEL_COUNT_BY_SHADE", {})
    n_each_side = max(5, int(level_count_map.get(shade_name, n_intervals + 1)) // 2)
    return _deduplicate_sorted_ticks(
        _symlog_boundaries(vmax=vmax, linthresh=linthresh, n_log_each_side=n_each_side),
        atol=1.0e-12,
    )


def near_zero_mask_threshold(levels: np.ndarray) -> float:
    """Return the absolute value threshold below which shading is left white."""

    values = np.asarray(levels, dtype=float)
    positive = np.sort(np.unique(values[values > 0]))
    if positive.size == 0:
        return 0.0
    return float(positive[0]) * float(CFG.ZERO_MASK_STEP_FACTOR)



def fill_field_edge_to_background(field: np.ndarray, pad_cells: int = 2) -> np.ndarray:
    """
    Remove artificial blank margins caused by contourf interpolation/masking.
    Missing display cells are filled with NaN internally and rendered through
    the colormap bad colour equal to the map background.
    """
    arr = np.asarray(field, dtype=float).copy()

    if pad_cells <= 0:
        return arr

    valid = np.isfinite(arr)
    if not valid.any():
        return arr

    # Nearest-neighbour padding of the outer valid anomaly field.
    # This only affects display edges and does not change the composite data.
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True
    )
    padded = arr[tuple(idx)]

    # Keep the original valid field and only replace the artificial border.
    border = np.zeros_like(valid, dtype=bool)
    border[:pad_cells, :] = True
    border[-pad_cells:, :] = True
    border[:, :pad_cells] = True
    border[:, -pad_cells:] = True

    arr[border & (~valid)] = padded[border & (~valid)]
    return arr


def set_background_bad_color(cmap):
    """Make masked areas exactly match the map background."""
    cmap = cmap.copy()
    cmap.set_bad(CFG.MAP_BACKGROUND_COLOR)
    return cmap


def extend_field_to_plot_frame(
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    extent: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extend the plotted field by one outer ring so contourf shading reaches the
    map frame without leaving a pale halo near the panel boundary.

    This is a display-only operation: the outer ring simply repeats the nearest
    edge values and is placed half a grid step beyond the plotting extent, so
    the colour fill is clipped exactly at the map frame.
    """
    x = np.asarray(lon, dtype=float)
    y = np.asarray(lat, dtype=float)
    z = np.asarray(field, dtype=float)

    if z.ndim != 2 or z.shape != (y.size, x.size):
        return x, y, z

    if x.size < 2 or y.size < 2:
        return x, y, z

    west, east, south, north = map(float, extent)

    dx_left = abs(x[1] - x[0])
    dx_right = abs(x[-1] - x[-2])
    dy_bottom = abs(y[1] - y[0])
    dy_top = abs(y[-1] - y[-2])

    x_ext = np.empty(x.size + 2, dtype=float)
    y_ext = np.empty(y.size + 2, dtype=float)
    x_ext[1:-1] = x
    y_ext[1:-1] = y

    # Put the synthetic ring slightly outside the plotting extent, then let the
    # axes clipping trim it exactly to the frame edge. This removes the visible
    # unfilled buffer around the boundary.
    x_ext[0] = west - 0.51 * dx_left
    x_ext[-1] = east + 0.51 * dx_right
    y_ext[0] = south - 0.51 * dy_bottom
    y_ext[-1] = north + 0.51 * dy_top

    z_ext = np.pad(z, ((1, 1), (1, 1)), mode='edge')
    return x_ext, y_ext, z_ext


def apply_low_value_display_treatment(
    field: np.ndarray,
    levels: np.ndarray,
    shade_name: Optional[str] = None,
) -> np.ndarray:
    """Optionally snap the low-value interval to zero.

    v102 leaves this option disabled so the original signed weak anomalies use
    the two pale central colour bins.  The legacy switch remains available for
    reproducibility of older layouts.
    """

    source = np.ma.array(field, copy=False)
    arr = np.asarray(source.filled(np.nan), dtype=float)
    if not CFG.MASK_NEAR_ZERO_VALUES:
        return np.ma.array(arr, mask=np.ma.getmaskarray(source))

    threshold_map = getattr(CFG, "WHITE_CENTER_THRESHOLD_BY_SHADE", {})
    if shade_name is not None and shade_name in threshold_map:
        threshold = float(threshold_map[shade_name])
    else:
        threshold = near_zero_mask_threshold(levels)
        threshold *= (_get_zero_mask_factor(shade_name) / float(CFG.ZERO_MASK_STEP_FACTOR))
    if not np.isfinite(threshold) or threshold <= 0:
        return np.ma.array(arr, mask=np.ma.getmaskarray(source))

    result = arr.copy()
    neutral = np.isfinite(result) & (np.abs(result) <= threshold)
    result[neutral] = 0.0
    return np.ma.array(result, mask=np.ma.getmaskarray(source))


def _nearest_nice_step(raw_step: float) -> float:
    """Return a visually clean step close to raw_step."""

    raw_step = float(abs(raw_step))
    if not np.isfinite(raw_step) or raw_step <= 0:
        return 1.0

    exponent = math.floor(math.log10(raw_step))
    base = 10.0 ** exponent
    fraction = raw_step / base

    candidates = np.array([1.0, 2.0, 2.5, 4.0, 5.0, 10.0])
    nice_fraction = float(candidates[np.argmin(np.abs(candidates - fraction))])
    return nice_fraction * base


def _get_display_smooth_sigma(shade_name: Optional[str] = None) -> float:
    if shade_name is not None:
        mapping = getattr(CFG, "DISPLAY_SMOOTH_SIGMA_BY_SHADE", {})
        if shade_name in mapping:
            return float(mapping[shade_name])
    return float(getattr(CFG, "DISPLAY_SMOOTH_SIGMA", 0.0))


def _get_zero_mask_factor(shade_name: Optional[str] = None) -> float:
    if shade_name is not None:
        mapping = getattr(CFG, "ZERO_MASK_STEP_FACTOR_BY_SHADE", {})
        if shade_name in mapping:
            return float(mapping[shade_name])
    return float(getattr(CFG, "ZERO_MASK_STEP_FACTOR", 0.65))


def _get_min_region_cells(shade_name: Optional[str] = None) -> int:
    if shade_name is not None:
        mapping = getattr(CFG, "MIN_SIGNIFICANT_REGION_CELLS_BY_SHADE", {})
        if shade_name in mapping:
            return int(mapping[shade_name])
    return int(getattr(CFG, "MIN_SIGNIFICANT_REGION_CELLS", 30))


def _get_min_display_patch_cells(shade_name: Optional[str] = None) -> int:
    """Return the minimum connected patch size retained for display.

    This is a display-only cleanup threshold and does not alter the underlying
    statistical significance calculation.
    """
    if shade_name is not None:
        mapping = getattr(CFG, "MIN_DISPLAY_PATCH_CELLS_BY_SHADE", {})
        if shade_name in mapping:
            return int(mapping[shade_name])
    return int(getattr(CFG, "MIN_DISPLAY_PATCH_CELLS", 0))


def _get_display_patch_open_iters(shade_name: Optional[str] = None) -> int:
    if shade_name is not None:
        mapping = getattr(CFG, "DISPLAY_PATCH_OPEN_ITERS_BY_SHADE", {})
        if shade_name in mapping:
            return int(mapping[shade_name])
    return 0


def _get_display_patch_close_iters(shade_name: Optional[str] = None) -> int:
    if shade_name is not None:
        mapping = getattr(CFG, "DISPLAY_PATCH_CLOSE_ITERS_BY_SHADE", {})
        if shade_name in mapping:
            return int(mapping[shade_name])
    return 0


def _get_effect_factor(shade_name: Optional[str] = None) -> float:
    if shade_name is not None:
        mapping = getattr(CFG, "SIGNIFICANT_FILL_EFFECT_STEP_FACTOR_BY_SHADE", {})
        if shade_name in mapping:
            return float(mapping[shade_name])
    return float(getattr(CFG, "SIGNIFICANT_FILL_EFFECT_STEP_FACTOR", 0.92))


def _get_close_iters(shade_name: Optional[str] = None) -> int:
    if shade_name is not None:
        mapping = getattr(CFG, "SIGNIFICANCE_MASK_CLOSE_ITERS_BY_SHADE", {})
        if shade_name in mapping:
            return int(mapping[shade_name])
    return int(getattr(CFG, "SIGNIFICANCE_MASK_CLOSE_ITERS", 4))


def _get_dilate_iters(shade_name: Optional[str] = None) -> int:
    if shade_name is not None:
        mapping = getattr(CFG, "SIGNIFICANCE_MASK_DILATE_ITERS_BY_SHADE", {})
        if shade_name in mapping:
            return int(mapping[shade_name])
    return int(getattr(CFG, "SIGNIFICANCE_MASK_DILATE_ITERS", 2))


def _get_stipple_value_threshold(shade_name: Optional[str] = None) -> float:
    if shade_name is not None:
        mapping = getattr(CFG, "SIGNIFICANCE_STIPPLE_VALUE_THRESHOLD_BY_SHADE", {})
        if shade_name in mapping:
            return float(mapping[shade_name])
    return 0.0


def _deduplicate_sorted_ticks(values: np.ndarray, atol: float = 1.0e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return values
    keep = [values[0]]
    for val in values[1:]:
        if not np.isclose(val, keep[-1], atol=atol, rtol=0.0):
            keep.append(val)
    return np.asarray(keep, dtype=float)


def mechanism_colorbar_ticks(
    levels: np.ndarray,
    shade_name: Optional[str] = None,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Return physical-value tick labels for a SymLog colour bar.

    For v102, use explicit labels where requested so values such as ±0.20 and
    ±0.005 are displayed exactly and remain perfectly consistent with the
    physical tick positions.
    """

    fixed_map = getattr(CFG, "FIXED_COLORBAR_TICKS_BY_SHADE", {})
    if shade_name is not None and shade_name in fixed_map:
        ticks = np.asarray(fixed_map[shade_name], dtype=float)
    else:
        ticks = _deduplicate_sorted_ticks(np.asarray(levels, dtype=float), atol=1.0e-12)

    if ticks.size == 0:
        return np.array([0.0]), ("0",)

    explicit_label_map = getattr(CFG, "FIXED_COLORBAR_TICK_LABELS_BY_SHADE", {})
    if shade_name is not None and shade_name in explicit_label_map:
        labels = tuple(explicit_label_map[shade_name])
        if len(labels) != int(ticks.size):
            raise ValueError(
                f"FIXED_COLORBAR_TICK_LABELS_BY_SHADE[{shade_name!r}] must have the same length as the corresponding ticks."
            )
        return ticks, labels

    decimals_map = getattr(CFG, "COLORBAR_TICK_DECIMALS_BY_SHADE", {})
    if shade_name is not None and shade_name in decimals_map:
        decimals = int(decimals_map[shade_name])
    else:
        max_abs = float(np.nanmax(np.abs(ticks)))
        decimals = 2 if max_abs < 0.1 else 1

    labels = []
    fmt = "{:0." + str(decimals) + "f}"
    for v in ticks:
        if np.isclose(v, 0.0, atol=1.0e-12):
            labels.append("0")
        else:
            labels.append(fmt.format(v))
    return ticks, tuple(labels)


REFERENCE_RGB_ANCHORS_BY_CMAP = {
    # Unified reference palette: negative anomalies are blue and positive
    # anomalies are red, matching the supplied six-panel circulation figure.
    "science_div500": {
        "negative": (
            "#26007A", "#2F29B8", "#355CD6", "#5689E1", "#8EB7EA", "#C5DCF4",
        ),
        "positive": (
            "#F7D1BC", "#F5A27A", "#F36B4E", "#E83B31", "#C51B29", "#8B001D",
        ),
    },
    "science_mfc700": {
        "negative": ("#26007A", "#2F29B8", "#355CD6", "#5689E1", "#8EB7EA", "#C5DCF4"),
        "positive": ("#F7D1BC", "#F5A27A", "#F36B4E", "#E83B31", "#C51B29", "#8B001D"),
    },
    "science_ascent700": {
        "negative": ("#26007A", "#2F29B8", "#355CD6", "#5689E1", "#8EB7EA", "#C5DCF4"),
        "positive": ("#F7D1BC", "#F5A27A", "#F36B4E", "#E83B31", "#C51B29", "#8B001D"),
    },
}

CMAP_SHADE_NAME = {
    "science_div500": "div500",
    "science_mfc700": "mfc700",
    "science_ascent700": "ascent700",
}


def _sample_anchor_gradient(anchors: Sequence[str], count: int) -> list:
    """Return ``count`` evenly sampled colours through the supplied RGB anchors."""

    count = int(count)
    if count <= 0:
        return []
    if count == 1:
        return [mcolors.to_rgba(anchors[len(anchors) // 2])]

    gradient = mcolors.LinearSegmentedColormap.from_list(
        "reference_rgb_segment",
        list(anchors),
        N=max(256, count),
    )
    return [gradient(x) for x in np.linspace(0.0, 1.0, count)]


def _blend_colour_with_white(colour: str, white_fraction: float) -> Tuple[float, float, float, float]:
    """Return a pale, still chromatic version of ``colour``."""

    fraction = float(np.clip(white_fraction, 0.0, 1.0))
    rgba = np.asarray(mcolors.to_rgba(colour), dtype=float)
    rgba[:3] = (1.0 - fraction) * rgba[:3] + fraction
    return tuple(rgba)


def publication_cmap(
    name: str,
    levels: Optional[Sequence[float]] = None,
    n_bins: Optional[int] = None,
):
    """Build one exact colour per physical interval from the reference RGB codes.

    This implementation intentionally returns a ``ListedColormap`` whose
    length is identical to ``len(levels) - 1``.  It fixes the previous v98
    mismatch in which a 256-colour LUT was combined with a much smaller
    ``BoundaryNorm`` and could therefore select only a small part of the LUT.
    """

    if name not in REFERENCE_RGB_ANCHORS_BY_CMAP:
        return plt.get_cmap(name)

    if levels is None:
        count = max(3, int(n_bins) if n_bins is not None else 11)
        anchors = REFERENCE_RGB_ANCHORS_BY_CMAP[name]
        joined = (
            list(anchors["negative"])
            + [_blend_colour_with_white(anchors["negative"][-1], CFG.LOW_VALUE_BLEND_TO_WHITE)]
            + [_blend_colour_with_white(anchors["positive"][0], CFG.LOW_VALUE_BLEND_TO_WHITE)]
            + list(anchors["positive"])
        )
        cmap = mcolors.LinearSegmentedColormap.from_list(name, joined, N=count)
        cmap.set_bad("white")
        return cmap

    boundaries = np.asarray(levels, dtype=float)
    if boundaries.ndim != 1 or boundaries.size < 2 or np.any(np.diff(boundaries) <= 0):
        raise ValueError("publication_cmap requires strictly increasing one-dimensional levels.")

    shade_name = CMAP_SHADE_NAME[name]
    threshold = float(CFG.WHITE_CENTER_THRESHOLD_BY_SHADE[shade_name])
    midpoints = 0.5 * (boundaries[:-1] + boundaries[1:])

    negative_index = np.where(midpoints < -threshold)[0]
    neutral_index = np.where(np.abs(midpoints) <= threshold)[0]
    positive_index = np.where(midpoints > threshold)[0]

    anchors = REFERENCE_RGB_ANCHORS_BY_CMAP[name]
    interval_colors = [mcolors.to_rgba("#FFFFFF")] * int(midpoints.size)

    for idx, color in zip(
        negative_index,
        _sample_anchor_gradient(anchors["negative"], negative_index.size),
    ):
        interval_colors[int(idx)] = color

    for idx, color in zip(
        positive_index,
        _sample_anchor_gradient(anchors["positive"], positive_index.size),
    ):
        interval_colors[int(idx)] = color

    # Replace the former white centre with two very pale signed colours.  The
    # negative half uses the near-zero negative anchor and the positive half
    # uses the near-zero positive anchor, so weak anomalies retain their sign.
    pale_negative = _blend_colour_with_white(
        anchors["negative"][-1], CFG.LOW_VALUE_BLEND_TO_WHITE
    )
    pale_positive = _blend_colour_with_white(
        anchors["positive"][0], CFG.LOW_VALUE_BLEND_TO_WHITE
    )
    for idx in neutral_index:
        interval_colors[int(idx)] = (
            pale_negative if midpoints[int(idx)] < 0 else pale_positive
        )

    cmap = mcolors.ListedColormap(interval_colors, name=name, N=len(interval_colors))
    cmap.set_under(interval_colors[0])
    cmap.set_over(interval_colors[-1])
    cmap.set_bad("white")
    return cmap

def get_shade_norm(levels: np.ndarray, shade_name: Optional[str] = None):
    """Return a discrete publication norm with equal-width colour boxes.

    The level values themselves follow a quasi-log / symmetric-log spacing in
    data space, but the colourbar boxes are shown with uniform visual width,
    which matches common journal styling and keeps the labels aligned neatly.
    """
    vals = np.asarray(levels, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        vals = np.array([-1.0, 0.0, 1.0], dtype=float)
    vals = np.unique(vals)
    # publication_cmap() returns one colour per interval, so the normalisation
    # must use the identical number of colours.  This guarantees strict
    # alignment between map fill, colour-bar boxes, boundaries and tick marks.
    return BoundaryNorm(vals, ncolors=int(vals.size - 1), clip=False)


def validate_reference_color_scales() -> None:
    """Fail early if any map/colour-bar palette is internally inconsistent."""

    for shade_name, cmap_name in CFG.CMAPS.items():
        levels = np.asarray(CFG.CUSTOM_DISPLAY_LEVELS_BY_SHADE[shade_name], dtype=float)
        threshold = float(CFG.WHITE_CENTER_THRESHOLD_BY_SHADE[shade_name])

        if not np.any(np.isclose(levels, -threshold, atol=1.0e-12, rtol=0.0)):
            raise ValueError(f"{shade_name}: -threshold is missing from display levels.")
        if not np.any(np.isclose(levels, threshold, atol=1.0e-12, rtol=0.0)):
            raise ValueError(f"{shade_name}: +threshold is missing from display levels.")

        cmap = publication_cmap(cmap_name, levels=levels)
        norm = get_shade_norm(levels, shade_name=shade_name)
        expected_bins = int(levels.size - 1)
        if cmap.N != expected_bins:
            raise ValueError(
                f"{shade_name}: colormap has {cmap.N} colours but {expected_bins} intervals."
            )

        midpoints = 0.5 * (levels[:-1] + levels[1:])
        mapped_bins = np.asarray(norm(midpoints), dtype=int)
        if not np.array_equal(mapped_bins, np.arange(expected_bins, dtype=int)):
            raise ValueError(f"{shade_name}: BoundaryNorm bins are not aligned with display intervals.")

        rgba = np.asarray(cmap(np.arange(cmap.N)), dtype=float)
        neutral = np.abs(midpoints) <= threshold
        if not neutral.any():
            raise ValueError(f"{shade_name}: no central pale interval was constructed.")
        if np.any(np.all(np.isclose(rgba[neutral, :3], 1.0, atol=1.0e-12), axis=1)):
            raise ValueError(f"{shade_name}: an unintended pure-white central bin remains.")

        negative_centre = rgba[neutral & (midpoints < 0), :3]
        positive_centre = rgba[neutral & (midpoints >= 0), :3]
        if negative_centre.size == 0 or positive_centre.size == 0:
            raise ValueError(f"{shade_name}: both signed pale central bins are required.")
        if np.allclose(negative_centre[-1], positive_centre[0], atol=1.0e-4, rtol=0.0):
            raise ValueError(f"{shade_name}: negative and positive low-value colours are indistinguishable.")

        if shade_name == "mfc700":
            # Row 2 must contain no pure-grey bins, including the pale centre.
            chroma = np.ptp(rgba[:, :3], axis=1)
            if np.any(chroma <= 0.005):
                raise ValueError("mfc700: a grey or near-grey colour bin was found.")

        print(
            f"[COLOUR SCALE OK] {shade_name} | bins={expected_bins} | "
            f"pale signed fill for |x| <= {threshold:g}"
        )


def smooth_field_for_display(field: np.ndarray, shade_name: Optional[str] = None) -> np.ndarray:
    """Apply mild NaN-aware Gaussian smoothing for cleaner contiguous shading."""

    sigma = _get_display_smooth_sigma(shade_name)
    arr = np.asarray(field, dtype=float)
    if sigma <= 0:
        return arr

    finite = np.isfinite(arr)
    if not finite.any():
        return arr

    values = np.where(finite, arr, 0.0)
    weights = finite.astype(float)

    smooth_values = ndimage.gaussian_filter(values, sigma=sigma, mode="nearest")
    smooth_weights = ndimage.gaussian_filter(weights, sigma=sigma, mode="nearest")

    out = np.full_like(arr, np.nan, dtype=float)
    valid = smooth_weights > 1.0e-12
    out[valid] = smooth_values[valid] / smooth_weights[valid]
    return out


def remove_small_display_patches(
    field: np.ndarray,
    levels: np.ndarray,
    shade_name: Optional[str] = None,
) -> np.ndarray:
    """Suppress tiny isolated coloured patches in the displayed field.

    This function keeps only the retained connected components and masks
    everything else to the same background colour as the map, which avoids
    ring-like remnants or coloured outlines around discarded patches.
    """

    arr = np.asarray(field, dtype=float)
    min_cells = _get_min_display_patch_cells(shade_name)
    if min_cells <= 1:
        return np.ma.array(arr, mask=~np.isfinite(arr))

    threshold_map = getattr(CFG, "WHITE_CENTER_THRESHOLD_BY_SHADE", {})
    if shade_name is not None and shade_name in threshold_map:
        step_thr = float(threshold_map[shade_name])
    else:
        step_thr = near_zero_mask_threshold(levels)
        step_thr *= (_get_zero_mask_factor(shade_name) / float(CFG.ZERO_MASK_STEP_FACTOR))
    step_thr = max(step_thr, 0.0)

    structure = np.ones((3, 3), dtype=bool)
    open_iters = _get_display_patch_open_iters(shade_name)
    close_iters = _get_display_patch_close_iters(shade_name)

    # The complete low-value interval is valid data and must retain its pale
    # signed fill; it must not be converted into missing-data background while
    # small coloured anomaly patches are cleaned.
    keep_mask = np.isfinite(arr) & (np.abs(arr) <= step_thr)

    for sign in (1.0, -1.0):
        mask = np.isfinite(arr) & ((arr * sign) > step_thr)
        if not mask.any():
            continue

        if open_iters > 0:
            mask = ndimage.binary_opening(mask, structure=structure, iterations=open_iters)
        if close_iters > 0:
            mask = ndimage.binary_closing(mask, structure=structure, iterations=close_iters)

        if not mask.any():
            continue

        labels, nlab = ndimage.label(mask)
        if nlab == 0:
            continue

        counts = np.bincount(labels.ravel())
        for lab in range(1, nlab + 1):
            if lab < counts.size and counts[lab] >= min_cells:
                keep_mask |= (labels == lab)

    out = arr.copy()
    out[~keep_mask] = np.nan
    return np.ma.array(out, mask=~np.isfinite(out))


def smooth_vector_for_quiver(field: np.ndarray) -> np.ndarray:
    """Smooth vector components for cleaner small-arrow display only."""

    sigma = float(getattr(CFG, "QUIVER_SMOOTH_SIGMA", 0.0))
    arr = np.asarray(field, dtype=float)
    if sigma <= 0:
        return arr

    finite = np.isfinite(arr)
    if not finite.any():
        return arr

    values = np.where(finite, arr, 0.0)
    weights = finite.astype(float)
    smooth_values = ndimage.gaussian_filter(values, sigma=sigma, mode="nearest")
    smooth_weights = ndimage.gaussian_filter(weights, sigma=sigma, mode="nearest")

    out = np.full_like(arr, np.nan, dtype=float)
    valid = smooth_weights > 1.0e-12
    out[valid] = smooth_values[valid] / smooth_weights[valid]
    return out


def prepare_display_significance_mask(
    mask: np.ndarray,
    field: Optional[np.ndarray] = None,
    levels: Optional[np.ndarray] = None,
    shade_name: Optional[str] = None,
) -> np.ndarray:
    """Build a cleaner, spatially concentrated significance mask for display.

    Statistical significance itself is unchanged. The extra morphology and
    effect-magnitude threshold are display-only filters used to suppress tiny
    fragmented patches in the final figure.
    """

    sig = np.asarray(mask, dtype=bool)
    if sig.ndim != 2 or not sig.any():
        return np.zeros_like(sig, dtype=bool)

    # Require a visible anomaly magnitude in addition to significance.
    if field is not None and levels is not None:
        arr = np.asarray(field, dtype=float)
        threshold = near_zero_mask_threshold(np.asarray(levels, dtype=float))
        threshold *= _get_effect_factor(shade_name)

        strong = np.isfinite(arr) & (np.abs(arr) >= threshold)
        sig = sig & strong

    if not sig.any():
        return np.zeros_like(sig, dtype=bool)

    structure = np.ones((3, 3), dtype=bool)

    # Opening removes isolated pixels; closing joins short gaps and smooths edges.
    if int(CFG.SIGNIFICANCE_MASK_OPEN_ITERS) > 0:
        sig = ndimage.binary_opening(
            sig,
            structure=structure,
            iterations=int(CFG.SIGNIFICANCE_MASK_OPEN_ITERS),
        )

    sig = ndimage.binary_closing(
        sig,
        structure=structure,
        iterations=int(CFG.SIGNIFICANCE_MASK_CLOSE_ITERS),
    )
    sig = ndimage.binary_fill_holes(sig)

    if _get_dilate_iters(shade_name) > 0:
        sig = ndimage.binary_dilation(
            sig,
            structure=structure,
            iterations=_get_dilate_iters(shade_name),
        )

    labels, num = ndimage.label(
        sig,
        structure=np.ones((3, 3), dtype=int),
    )
    if num == 0:
        return np.zeros_like(sig, dtype=bool)

    counts = np.bincount(labels.ravel())
    keep_labels = np.where(
        counts >= _get_min_region_cells(shade_name)
    )[0]
    keep_labels = keep_labels[keep_labels != 0]

    if keep_labels.size == 0:
        return np.zeros_like(sig, dtype=bool)

    return np.isin(labels, keep_labels)


def build_significance_stipple_mask(
    raw_significance_mask: np.ndarray,
    displayed_field: Optional[np.ndarray] = None,
    shade_name: Optional[str] = None,
) -> np.ndarray:
    """Return a cleaned dotted-region mask for p<alpha significance.

    The statistical test is unchanged: grid-cell Welch p < 0.05.  For the
    display only, isolated significant pixels are removed so the stippling
    resembles the coherent dotted significant regions shown in Fig. 3 of Xie
    et al. (2025).
    """

    sig = np.asarray(raw_significance_mask, dtype=bool)
    if sig.ndim != 2 or not sig.any():
        return np.zeros_like(sig, dtype=bool)

    shown_data = None
    if displayed_field is not None:
        shown = np.ma.array(displayed_field, copy=False)
        shown_data = np.asarray(shown.data, dtype=float)
        shown_mask = np.ma.getmaskarray(shown)
        visible = (~shown_mask) & np.isfinite(shown_data)
        if visible.shape != sig.shape:
            raise ValueError(
                "Significance stipple mask and displayed shading have different "
                f"shapes: significance={sig.shape}, displayed={visible.shape}"
            )
        if bool(getattr(CFG, "STIPPLE_ONLY_WITHIN_DISPLAYED_FILL", True)):
            sig = sig & visible

        amp_thr = float(_get_stipple_value_threshold(shade_name))
        if amp_thr > 0.0:
            sig = sig & (np.abs(shown_data) >= amp_thr)

    if not sig.any():
        return np.zeros_like(sig, dtype=bool)

    structure = np.ones((3, 3), dtype=bool)
    open_iters = int(getattr(CFG, "SIGNIFICANCE_STIPPLE_OPEN_ITERS", 0))
    close_iters = int(getattr(CFG, "SIGNIFICANCE_STIPPLE_CLOSE_ITERS", 0))
    dilate_iters = int(getattr(CFG, "SIGNIFICANCE_STIPPLE_DILATE_ITERS", 0))

    if open_iters > 0:
        sig = ndimage.binary_opening(sig, structure=structure, iterations=open_iters)
    if close_iters > 0:
        sig = ndimage.binary_closing(sig, structure=structure, iterations=close_iters)
    if dilate_iters > 0:
        sig = ndimage.binary_dilation(sig, structure=structure, iterations=dilate_iters)

    sig = ndimage.binary_fill_holes(sig)
    labels, num = ndimage.label(sig, structure=np.ones((3, 3), dtype=int))
    if num == 0:
        return np.zeros_like(sig, dtype=bool)

    counts = np.bincount(labels.ravel())
    min_cells = int(getattr(CFG, "SIGNIFICANCE_STIPPLE_MIN_CLUSTER_CELLS", 1))
    keep_labels = np.where(counts >= min_cells)[0]
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(sig, dtype=bool)

    return np.isin(labels, keep_labels)


def plot_significance_stippling(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Plot sparse black dots at grid cells with nominal p < alpha.

    The stride is display-only and does not alter the significance calculation.
    """

    if not bool(getattr(CFG, "SHOW_SIGNIFICANCE_STIPPLING", True)):
        return

    sig = np.asarray(mask, dtype=bool)
    x = np.asarray(lon, dtype=float)
    y = np.asarray(lat, dtype=float)

    if sig.ndim != 2 or sig.shape != (y.size, x.size) or not sig.any():
        return

    stride = max(1, int(getattr(CFG, "SIGNIFICANCE_STIPPLE_STRIDE", 1)))
    sig_sub = sig[::stride, ::stride]
    if not sig_sub.any():
        return

    lon_sub = x[::stride]
    lat_sub = y[::stride]
    xx, yy = np.meshgrid(lon_sub, lat_sub)

    ax.scatter(
        xx[sig_sub],
        yy[sig_sub],
        s=float(getattr(CFG, "SIGNIFICANCE_STIPPLE_SIZE", 2.6)),
        c=str(getattr(CFG, "SIGNIFICANCE_STIPPLE_COLOR", "#000000")),
        marker="o",
        linewidths=0.0,
        edgecolors="none",
        alpha=float(getattr(CFG, "SIGNIFICANCE_STIPPLE_ALPHA", 0.78)),
        transform=ccrs.PlateCarree(),
        zorder=float(getattr(CFG, "SIGNIFICANCE_STIPPLE_ZORDER", 7.5)),
        rasterized=True,
    )


def filter_significant_regions_by_area(
    mask: np.ndarray,
    field: Optional[np.ndarray] = None,
    levels: Optional[np.ndarray] = None,
    shade_name: Optional[str] = None,
) -> np.ndarray:
    """Return the cleaned display significance mask."""

    sig = np.asarray(mask, dtype=bool)
    if sig.ndim != 2 or not sig.any():
        return sig

    if not CFG.FILL_ONLY_SIGNIFICANT_LARGE_REGIONS:
        return sig

    return prepare_display_significance_mask(sig, field=field, levels=levels, shade_name=shade_name)


def apply_significance_area_fill_mask(
    field: np.ndarray,
    mask: np.ndarray,
    levels: np.ndarray,
    shade_name: Optional[str] = None,
) -> np.ndarray:
    """Optionally restrict colour shading to statistically significant regions.

    v57 behaviour
    -------------
    When ``CFG.FILL_ONLY_SIGNIFICANT_LARGE_REGIONS`` is False, statistical
    significance is NOT used to mask the colour field.  The map therefore
    displays the broader detrended ENSO-minus-Neutral anomaly pattern after
    display-only smoothing, the reduced near-zero threshold, and light
    connected-patch cleanup.  Raw Welch-test p < 0.05 grid cells are shown
    separately by black stippling.

    When the option is True, the previous v56 significant-region masking
    behaviour is retained.
    """

    arr = np.ma.array(field, copy=False)

    if not bool(CFG.FILL_ONLY_SIGNIFICANT_LARGE_REGIONS):
        return np.ma.array(
            np.asarray(arr, dtype=float),
            mask=np.ma.getmaskarray(arr),
        )

    sig_keep = filter_significant_regions_by_area(
        mask,
        field=np.asarray(arr, dtype=float),
        levels=levels,
        shade_name=shade_name,
    )

    combined_mask = np.ma.getmaskarray(arr) | (~sig_keep)

    return np.ma.array(
        np.asarray(arr, dtype=float),
        mask=combined_mask,
    )


def export_near_zero_thresholds(
    season_name: str,
    row_specs: Sequence[Mapping[str, object]],
    levels_by_shade: Mapping[str, np.ndarray],
) -> None:
    """Write and print the pale low-value thresholds for each figure."""

    threshold_dir = CFG.VERSION_OUTPUT_DIR / "threshold_summaries"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    path = threshold_dir / f"{season_name}_near_zero_thresholds.txt"

    panel_letters = (("a", "b"), ("c", "d"), ("e", "f"))
    lines = [
        f"Season: {season_name}",
        "Pale signed central colour intervals (threshold endpoints included):",
        "Shading shows the detrended ENSO-minus-Neutral anomaly field; low-magnitude negative and positive values retain separate pale hues, and black dots mark original grid-cell Welch p < 0.05 results.",
        "",
    ]

    print(
        f"[SIGNIFICANCE DISPLAY] {season_name} | "
        f"min_region={CFG.MIN_SIGNIFICANT_REGION_CELLS} cells | "
        f"effect_factor={CFG.SIGNIFICANT_FILL_EFFECT_STEP_FACTOR} | "
        f"open={CFG.SIGNIFICANCE_MASK_OPEN_ITERS} | "
        f"close={CFG.SIGNIFICANCE_MASK_CLOSE_ITERS} | "
        f"dilate={CFG.SIGNIFICANCE_MASK_DILATE_ITERS}"
    )

    for row, spec in enumerate(row_specs):
        shade_name = str(spec["shade"])
        levels = np.asarray(levels_by_shade[shade_name], dtype=float)
        threshold = float(CFG.WHITE_CENTER_THRESHOLD_BY_SHADE[shade_name])
        label = str(spec["cbar_label"])
        stipple_thr = _get_stipple_value_threshold(shade_name)
        lines.append(
            f"Panels ({panel_letters[row][0]}) and ({panel_letters[row][1]}) | {label} | pale signed fill if |x| <= {threshold:g} | stipple only if p < {CFG.SIGNIFICANCE_ALPHA:g} and |x| >= {stipple_thr:g}"
        )
        print(
            f"[PALE-CENTRE THRESHOLD] {season_name} | panels ({panel_letters[row][0]})/({panel_letters[row][1]}) | {shade_name} | |x| <= {threshold:g} | stipple if |x| >= {stipple_thr:g}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_significance_outline(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Backward-compatible wrapper for raw significance stippling."""
    plot_significance_stippling(ax, lon, lat, np.asarray(mask, dtype=bool))


def vector_stride(
    lon: np.ndarray,
    lat: np.ndarray,
    spacing_deg: float,
) -> Tuple[int, int]:

    dlon = float(np.nanmedian(np.abs(np.diff(lon))))
    dlat = float(np.nanmedian(np.abs(np.diff(lat))))

    sx = max(
        1,
        int(round(spacing_deg / dlon)),
    )

    sy = max(
        1,
        int(round(spacing_deg / dlat)),
    )

    return sy, sx


def overlap_safe_vector_stride(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    base_spacing_deg: float,
    max_arrow_length_inches: float,
) -> Tuple[int, int]:
    """Return centred x/y strides large enough to avoid arrow overlap.

    The requested arrow length is expressed in physical inches.  This helper
    converts that length to longitude/latitude spacing using the current panel
    dimensions, then combines it with the user-set minimum angular interval.
    Consequently, single-panel and compact 3 x 2 figures retain comparable
    arrow separation even though their axes have different physical sizes.
    """

    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    dlon = float(np.nanmedian(np.abs(np.diff(lon_arr))))
    dlat = float(np.nanmedian(np.abs(np.diff(lat_arr))))

    if not np.isfinite(dlon) or dlon <= 0 or not np.isfinite(dlat) or dlat <= 0:
        return vector_stride(lon_arr, lat_arr, base_spacing_deg)

    position = ax.get_position()
    axes_width_in = float(ax.figure.get_figwidth()) * float(position.width)
    axes_height_in = float(ax.figure.get_figheight()) * float(position.height)
    lon_span = float(np.nanmax(lon_arr) - np.nanmin(lon_arr))
    lat_span = float(np.nanmax(lat_arr) - np.nanmin(lat_arr))

    clearance = max(1.0, float(CFG.QUIVER_OVERLAP_CLEARANCE_FACTOR))
    arrow_span_in = max(0.0, float(max_arrow_length_inches)) * clearance

    required_lon_deg = float(base_spacing_deg)
    required_lat_deg = float(base_spacing_deg)
    if np.isfinite(axes_width_in) and axes_width_in > 0 and lon_span > 0:
        required_lon_deg = max(
            required_lon_deg,
            arrow_span_in * lon_span / axes_width_in,
        )
    if np.isfinite(axes_height_in) and axes_height_in > 0 and lat_span > 0:
        required_lat_deg = max(
            required_lat_deg,
            arrow_span_in * lat_span / axes_height_in,
        )

    sx = max(1, int(math.ceil(required_lon_deg / dlon)))
    sy = max(1, int(math.ceil(required_lat_deg / dlat)))
    return sy, sx


def nice_reference_magnitude(
    u: np.ndarray,
    v: np.ndarray,
    percentile: Optional[float] = None,
) -> float:
    """Return a robust, publication-friendly 1/2/5 reference magnitude.

    Using a high-but-not-extreme percentile prevents a few very strong vectors
    from forcing the whole field to become visually tiny.
    """

    magnitude = np.hypot(u, v)
    finite = magnitude[
        np.isfinite(magnitude)
        & (magnitude > 0)
    ]

    if finite.size == 0:
        return 1.0

    pct = CFG.QUIVER_REF_PERCENTILE if percentile is None else float(percentile)
    pct = float(np.clip(pct, 1.0, 99.9))
    target = float(np.nanpercentile(finite, pct))

    exponent = math.floor(
        math.log10(target)
    ) if target > 0 else 0

    scaled = target / (10.0 ** exponent)

    if scaled < 1.5:
        leading = 1.0
    elif scaled < 3.5:
        leading = 2.0
    else:
        leading = 5.0

    return leading * (10.0 ** exponent)


def combined_vector_reference(
    field_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    percentile: Optional[float] = None,
) -> float:
    """Return one shared reference magnitude from multiple panels.

    This allows El Niño and La Niña (or any paired panels) to share exactly
    the same vector scale and legend reference.
    """

    mags: list[np.ndarray] = []
    for u, v in field_pairs:
        mag = np.hypot(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        finite = mag[np.isfinite(mag) & (mag > 0)]
        if finite.size:
            mags.append(finite.ravel())

    if not mags:
        return 1.0

    merged = np.concatenate(mags)
    zeros = np.zeros_like(merged)
    return nice_reference_magnitude(merged, zeros, percentile=percentile)


def draw_threshold_boxes(ax, lon, lat, field, shade_name):
    """Box the largest connected region beyond the panel's |x| Q75 threshold.

    The threshold is calculated independently for each displayed panel from
    all finite gridpoint magnitudes, ``percentile(abs(field), 75)``. Positive
    and negative components are labelled separately so opposite signs cannot
    be merged across a narrow zero band. Their areas are then compared and only
    the largest component overall is retained. The compact buffer is clipped
    slightly inside ``PLOT_EXTENT`` so the complete rectangle remains visible.
    """
    arr = np.asarray(field, dtype=float)

    percentile_map = getattr(CFG, "THRESHOLD_BOX_PERCENTILE_BY_SHADE", {})
    if shade_name not in percentile_map:
        return

    percentile = float(percentile_map[shade_name])
    if not np.isfinite(percentile) or not (0.0 < percentile < 100.0):
        return

    finite_values = arr[np.isfinite(arr)]
    if finite_values.size == 0:
        return

    # One symmetric, panel-specific magnitude threshold ensures that positive
    # and negative anomaly cores compete on exactly the same basis.  A signed
    # percentile would systematically favour positive values and is therefore
    # inappropriate for the existing two-sign connected-component logic.
    threshold = float(
        np.nanpercentile(np.abs(finite_values), percentile)
    )
    if not np.isfinite(threshold) or threshold <= 0:
        return

    candidates = []
    structure = np.ones((3, 3), dtype=int)

    for sign_mask, color, sign in (
        (np.isfinite(arr) & (arr >= threshold), CFG.POSITIVE_BOX_COLOR, "positive"),
        (np.isfinite(arr) & (arr <= -threshold), CFG.NEGATIVE_BOX_COLOR, "negative"),
    ):
        if not np.any(sign_mask):
            continue

        labels, n = ndimage.label(sign_mask, structure=structure)
        if n == 0:
            continue

        counts = np.bincount(labels.ravel())
        counts[0] = 0
        largest = int(np.argmax(counts))

        cell_count = int(counts[largest])
        if cell_count < int(CFG.THRESHOLD_BOX_MIN_CELLS):
            continue

        yy, xx = np.where(labels == largest)
        component_values = np.abs(arr[labels == largest])

        candidates.append({
            "xmin": float(np.asarray(lon)[xx].min()),
            "xmax": float(np.asarray(lon)[xx].max()),
            "ymin": float(np.asarray(lat)[yy].min()),
            "ymax": float(np.asarray(lat)[yy].max()),
            "color": color,
            "sign": sign,
            "cell_count": cell_count,
            "peak": float(np.nanmax(component_values)),
        })

    if not candidates:
        return

    # Largest area first; peak magnitude resolves a possible area tie.
    chosen = max(candidates, key=lambda item: (item["cell_count"], item["peak"]))

    west, east, south, north = map(float, CFG.PLOT_EXTENT)
    inset = max(0.0, float(CFG.THRESHOLD_BOX_FRAME_INSET_DEG))

    xmin = max(west + inset, chosen["xmin"] - CFG.THRESHOLD_BOX_BUFFER_LON)
    xmax = min(east - inset, chosen["xmax"] + CFG.THRESHOLD_BOX_BUFFER_LON)
    ymin = max(south + inset, chosen["ymin"] - CFG.THRESHOLD_BOX_BUFFER_LAT)
    ymax = min(north - inset, chosen["ymax"] + CFG.THRESHOLD_BOX_BUFFER_LAT)

    if not (xmax > xmin and ymax > ymin):
        return

    patch = Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        edgecolor=chosen["color"],
        linewidth=CFG.THRESHOLD_BOX_LINEWIDTH,
        transform=ccrs.PlateCarree(),
        zorder=CFG.THRESHOLD_BOX_ZORDER,
        clip_on=True,
    )
    patch.set_clip_path(ax.patch)
    ax.add_patch(patch)


def add_vector_key_box(
    ax,
    quiver,
    threshold: float,
    units: str,
    box_right: Optional[float] = None,
    box_scale: float = 1.0,
    text_size: Optional[float] = None,
) -> None:
    """Add a boxed unit-free QuiverKey that exactly matches map arrows."""

    # Retained in the call signature for compatibility with the row settings;
    # deliberately omitted from the visible key at the user's request.
    _ = units

    right = float(CFG.QKEY_BOX_X if box_right is None else box_right)
    top = float(CFG.QKEY_BOX_Y)
    key_scale = max(0.1, float(box_scale))
    width = float(CFG.QKEY_BOX_WIDTH) * key_scale
    height = float(CFG.QKEY_BOX_HEIGHT) * key_scale
    key_text_size = float(
        CFG.QKEY_TEXT_SIZE if text_size is None else text_size
    )
    left = right - width
    bottom = top - height

    # Draw the white frame first, then place a real QuiverKey on top.  Because
    # QuiverKey reuses ``quiver``, the map and key arrows cannot drift apart in
    # shaft width, head proportions, alpha, colour or scale.
    frame = Rectangle(
        (left, bottom),
        width,
        height,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor=CFG.QKEY_COLOR,
        linewidth=float(CFG.QKEY_BOX_LINEWIDTH) * key_scale,
        alpha=0.92,
        zorder=24,
        clip_on=True,
    )
    ax.add_patch(frame)

    key = ax.quiverkey(
        quiver,
        X=left + 0.5 * width,
        Y=bottom + 0.34 * height,
        U=float(threshold),
        label=f"{int(np.rint(threshold))}",
        labelpos="N",
        labelsep=float(CFG.QKEY_LABEL_SEPARATION_INCH) * key_scale,
        coordinates="axes",
        fontproperties={
            "family": CFG.FONT_FAMILY,
            "size": key_text_size,
            "weight": "bold",
        },
    )
    key.text.set_color(CFG.QKEY_COLOR)
    key.set_zorder(25)


def plot_thresholded_small_arrows(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    units: str,
    desired_length_inches: float,
    shared_ref: Optional[float] = None,
    show_key: bool = True,
    min_display_magnitude: Optional[float] = None,
    key_box_right: Optional[float] = None,
    key_box_scale: float = 1.0,
    key_text_size: Optional[float] = None,
):
    """Draw cyan direction arrows below, and proportional dark arrows above, the key."""

    u_sm = smooth_vector_for_quiver(u)
    v_sm = smooth_vector_for_quiver(v)

    if shared_ref is None:
        ref = nice_reference_magnitude(
            u_sm,
            v_sm,
            percentile=CFG.QUIVER_REF_PERCENTILE,
        )
    else:
        ref = float(shared_ref)

    mag = np.hypot(u_sm, v_sm)
    fixed_threshold = (
        0.0 if min_display_magnitude is None else float(min_display_magnitude)
    )
    min_magnitude = max(
        fixed_threshold,
        float(CFG.QUIVER_MIN_REF_FRACTION) * ref,
    )
    finite_nonzero = (
        np.isfinite(u_sm)
        & np.isfinite(v_sm)
        & np.isfinite(mag)
        & (mag > np.finfo(float).eps)
    )
    high_valid = finite_nonzero & (mag >= min_magnitude)
    low_valid = finite_nonzero & (mag < min_magnitude) & (min_magnitude > 0)

    # Below-threshold arrows carry direction only.  Normalising them to one
    # short fixed magnitude prevents weak numerical vectors from becoming dots
    # and follows the cyan-arrow convention in the supplied reference figure.
    low_display_magnitude = (
        min_magnitude * float(CFG.QUIVER_LOW_DIRECTION_LENGTH_FACTOR)
    )
    direction_u = np.divide(
        u_sm,
        mag,
        out=np.zeros_like(u_sm, dtype=float),
        where=finite_nonzero,
    )
    direction_v = np.divide(
        v_sm,
        mag,
        out=np.zeros_like(v_sm, dtype=float),
        where=finite_nonzero,
    )
    u_low_display = direction_u * low_display_magnitude
    v_low_display = direction_v * low_display_magnitude

    # High-value arrow length remains proportional to the original magnitude.
    # Only the robust upper tail is capped to protect the overlap-safe layout.
    valid_magnitudes = mag[high_valid]
    if valid_magnitudes.size:
        cap = float(np.nanpercentile(valid_magnitudes, CFG.QUIVER_CAP_PERCENTILE))
        if min_magnitude > 0:
            cap = min(
                cap,
                min_magnitude * float(CFG.QUIVER_MAX_THRESHOLD_MULTIPLIER),
            )
        if np.isfinite(cap) and cap > 0:
            shorten = np.minimum(1.0, cap / np.maximum(mag, np.finfo(float).eps))
            u_sm = u_sm * shorten
            v_sm = v_sm * shorten

    max_arrow_length_inches = (
        float(desired_length_inches)
        * float(CFG.QUIVER_MAX_THRESHOLD_MULTIPLIER)
    )
    sy, sx = overlap_safe_vector_stride(
        ax,
        lon,
        lat,
        base_spacing_deg=CFG.QUIVER_SPACING_DEG,
        max_arrow_length_inches=max_arrow_length_inches,
    )
    # Start from half a stride so arrows are centred within the map instead of
    # being clipped against the four panel boundaries.
    y0 = sy // 2
    x0 = sx // 2
    lon_q = np.asarray(lon, dtype=float)[x0::sx]
    lat_q = np.asarray(lat, dtype=float)[y0::sy]
    u_low_q = np.where(low_valid, u_low_display, np.nan)[y0::sy, x0::sx]
    v_low_q = np.where(low_valid, v_low_display, np.nan)[y0::sy, x0::sx]
    u_high_q = np.where(high_valid, u_sm, np.nan)[y0::sy, x0::sx]
    v_high_q = np.where(high_valid, v_sm, np.nan)[y0::sy, x0::sx]

    # Map the first visible magnitude (1 for wind; 5 for moisture flux) to the
    # requested physical length.  The QuiverKey uses the same object and U, so
    # its sample arrow is exactly identical to a threshold-strength map arrow.
    scale_magnitude = min_magnitude if min_magnitude > 0 else ref
    scale = max(scale_magnitude, np.finfo(float).eps) / max(
        float(desired_length_inches), np.finfo(float).eps
    )
    # Open outlined arrows reproduce the line-arrow appearance of the supplied
    # reference.  Both cyan and dark-blue layers share identical geometry and
    # outline thickness; only colour, length rule and z-order differ.
    common_quiver_kwargs = {
        "transform": ccrs.PlateCarree(),
        "angles": "uv",
        "pivot": "middle",
        "units": "inches",
        "scale_units": "inches",
        "scale": scale,
        "width": CFG.QUIVER_WIDTH,
        "headwidth": CFG.QUIVER_HEADWIDTH,
        "headlength": CFG.QUIVER_HEADLENGTH,
        "headaxislength": CFG.QUIVER_HEADAXISLENGTH,
        "minlength": 0.0,
        "facecolors": "none",
        "linewidths": CFG.QUIVER_OUTLINE_LINEWIDTH_PT,
    }

    low_quiver = ax.quiver(
        lon_q,
        lat_q,
        u_low_q,
        v_low_q,
        edgecolors=CFG.QUIVER_LOW_COLOR,
        alpha=CFG.QUIVER_LOW_ALPHA,
        zorder=9.8,
        **common_quiver_kwargs,
    )
    high_quiver = ax.quiver(
        lon_q,
        lat_q,
        u_high_q,
        v_high_q,
        edgecolors=CFG.QUIVER_HIGH_COLOR,
        alpha=CFG.QUIVER_HIGH_ALPHA,
        zorder=10,
        **common_quiver_kwargs,
    )

    if show_key and CFG.SHOW_QUIVER_NUMERIC_KEY and fixed_threshold > 0:
        add_vector_key_box(
            ax,
            high_quiver,
            fixed_threshold,
            units,
            box_right=key_box_right,
            box_scale=key_box_scale,
            text_size=key_text_size,
        )

    # Preserve the historical single-Quiver return interface while exposing
    # the cyan direction-only layer for validation or optional restyling.
    high_quiver.low_magnitude_quiver = low_quiver
    return high_quiver


def _save_one_format(fig: plt.Figure, path: Path, **kwargs) -> None:
    """Save with an explicit progress message so long raster writes are visible."""
    start = time.perf_counter()
    print(f"[SAVE START] {path.name}")
    fig.savefig(path, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"[SAVE DONE ] {path.name} | {elapsed:.1f} s")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)

    if CFG.EXPORT_PNG:
        _save_one_format(
            fig,
            stem.with_suffix(".png"),
            dpi=CFG.DPI,
            bbox_inches="tight",
            facecolor="white",
        )

    if CFG.EXPORT_TIF:
        path = stem.with_suffix(".tif")
        try:
            _save_one_format(
                fig,
                path,
                dpi=CFG.DPI,
                bbox_inches="tight",
                facecolor="white",
                format="tiff",
                pil_kwargs={"compression": "tiff_lzw"},
            )
        except TypeError:
            _save_one_format(
                fig,
                path,
                dpi=CFG.DPI,
                bbox_inches="tight",
                facecolor="white",
                format="tiff",
            )

    if CFG.EXPORT_PDF:
        _save_one_format(
            fig,
            stem.with_suffix(".pdf"),
            bbox_inches="tight",
            facecolor="white",
        )


# =============================================================================
# MAIN FIGURE
# =============================================================================

def apply_fixed_colorbar_ticks(cb, levels: np.ndarray, shade_name: Optional[str] = None) -> None:
    """Use sparse publication-style physical-value labels with equal-width colour boxes.

    Major ticks show a sparse set of physical values, while minor ticks mark
    the intermediate colour-box boundaries so the colourbar remains neat and
    CEE-like without labelling every subdivision.

    Important for the v102 reference-palette version:
    - the tick *positions* are kept exactly on the real physical values;
    - the tick *labels* are printed explicitly where needed (e.g. ±0.20 and ±0.005);
    - no manual xlim reset is applied, so Matplotlib keeps the boundary-colourbar
      mapping unchanged and the labels stay aligned with the coloured boxes.
    """

    from matplotlib.ticker import FixedLocator

    major_ticks, major_labels = mechanism_colorbar_ticks(levels, shade_name=shade_name)
    major_ticks = np.asarray(major_ticks, dtype=float)
    minor_ticks = _deduplicate_sorted_ticks(np.asarray(levels, dtype=float), atol=1.0e-12)

    cb.ax.minorticks_on()
    cb.set_ticks(major_ticks)
    cb.set_ticklabels(list(major_labels))

    # Add unlabeled minor ticks at the intermediate colour-box boundaries.
    if minor_ticks.size >= 2:
        cb.ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))

    cb.ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        top=False,
        labelbottom=True,
        length=4.0,
        width=0.9,
        pad=1.2,
    )
    cb.ax.tick_params(
        axis="x",
        which="minor",
        bottom=True,
        top=False,
        labelbottom=False,
        length=2.2,
        width=0.7,
    )
    cb.ax.xaxis.get_offset_text().set_visible(False)
    for label in cb.ax.get_xticklabels():
        label.set_horizontalalignment("center")
        label.set_verticalalignment("top")
        label.set_fontweight("bold")
        label.set_fontfamily(CFG.FONT_FAMILY)


def apply_fixed_vertical_colorbar_ticks(
    cb,
    levels: np.ndarray,
    shade_name: Optional[str] = None,
) -> None:
    """Use denser, boundary-aligned numeric ticks on board colourbars."""

    from matplotlib.ticker import FixedLocator

    boundaries = _deduplicate_sorted_ticks(
        np.asarray(levels, dtype=float),
        atol=1.0e-12,
    )
    if boundaries.size == 0:
        boundaries = np.array([0.0], dtype=float)

    # Label every second colour boundary, plus the two pale-centre thresholds
    # and zero.  This increases the board from roughly seven to about nine
    # labels per bar without crowding, and every label sits on a true bin edge.
    major_indices = np.arange(0, boundaries.size, 2, dtype=int)
    if major_indices[-1] != boundaries.size - 1:
        major_indices = np.append(major_indices, boundaries.size - 1)
    major_ticks = boundaries[major_indices]

    centre_threshold = float(
        getattr(CFG, "WHITE_CENTER_THRESHOLD_BY_SHADE", {}).get(
            shade_name,
            0.0,
        )
    )
    required_values = (0.0, -centre_threshold, centre_threshold)
    required_ticks = []
    for required in required_values:
        matches = np.flatnonzero(
            np.isclose(boundaries, required, atol=1.0e-12, rtol=0.0)
        )
        if matches.size:
            required_ticks.append(float(boundaries[matches[0]]))
    if required_ticks:
        major_ticks = _deduplicate_sorted_ticks(
            np.sort(np.concatenate([major_ticks, required_ticks])),
            atol=1.0e-12,
        )

    decimals = int(
        getattr(CFG, "COLORBAR_TICK_DECIMALS_BY_SHADE", {}).get(
            shade_name,
            2,
        )
    )
    fmt = "{:0." + str(decimals) + "f}"
    major_labels = tuple(
        "0" if np.isclose(value, 0.0, atol=1.0e-12) else fmt.format(value)
        for value in major_ticks
    )
    minor_ticks = boundaries

    cb.ax.minorticks_on()
    cb.set_ticks(major_ticks)
    cb.set_ticklabels(list(major_labels))
    if minor_ticks.size >= 2:
        cb.ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))

    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.yaxis.set_label_position("right")
    cb.ax.tick_params(
        axis="y",
        which="major",
        left=False,
        right=True,
        labelleft=False,
        labelright=True,
        length=4.0,
        width=0.9,
        pad=2.0,
    )
    cb.ax.tick_params(
        axis="y",
        which="minor",
        left=False,
        right=True,
        labelleft=False,
        labelright=False,
        length=2.2,
        width=0.7,
    )
    cb.ax.yaxis.get_offset_text().set_visible(False)
    for label in cb.ax.get_yticklabels():
        label.set_horizontalalignment("left")
        label.set_verticalalignment("center")
        label.set_fontweight("bold")
        label.set_fontfamily(CFG.FONT_FAMILY)


def save_single_panel_figure(fig: plt.Figure, stem: Path) -> None:
    """Single panels use the same fast/publication export policy as boards."""
    save_figure(fig, stem)


def compute_common_color_levels(
    season_results: Dict[str, Dict[str, object]],
) -> Dict[str, np.ndarray]:
    """One symmetric colour scale per diagnostic across every season and phase."""
    levels = {}

    for shade_name in ("div500", "mfc700", "ascent700"):
        arrays = []
        for season_name in CFG.SEASON_ORDER:
            effects = season_results[season_name]["effects"]
            arrays.extend([
                effects["El Nino"][shade_name],
                effects["La Nina"][shade_name],
            ])

        levels[shade_name] = symmetric_levels(arrays, n_intervals=CFG.SHADE_N_INTERVALS, shade_name=shade_name)

    return levels


def make_single_panel_figures(
    season_name: str,
    lon: np.ndarray,
    lat: np.ndarray,
    effects: Dict[str, Dict[str, np.ndarray]],
    significance: Dict[str, Dict[str, np.ndarray]],
    basins: gpd.GeoDataFrame,
    common_color_levels: Optional[Dict[str, np.ndarray]] = None,
) -> None:

    set_style()

    row_specs = (
        {
            "shade": "div500",
            "u": "u700",
            "v": "v700",
            "cmap": CFG.CMAPS["div500"],
            "cbar_label": r"500-hPa divergence anomaly (10$^{-6}$ s$^{-1}$)",
            "vector_units": r"m s$^{-1}$",
            "vector_min": CFG.WIND_VECTOR_MIN_MAGNITUDE,
            "name": "div500_wind700",
        },
        {
            "shade": "mfc700",
            "u": "qfu700",
            "v": "qfv700",
            "cmap": CFG.CMAPS["mfc700"],
            "cbar_label": r"700-hPa moisture-flux convergence proxy (10$^{-8}$ s$^{-1}$)",
            "vector_units": r"g kg$^{-1}$ m s$^{-1}$",
            "vector_min": CFG.QFLUX_VECTOR_MIN_MAGNITUDE,
            "name": "mfc700_qv700",
        },
        {
            "shade": "ascent700",
            "u": None,
            "v": None,
            "cmap": CFG.CMAPS["ascent700"],
            "cbar_label": r"700-hPa ascent anomaly, $-\omega$ (Pa s$^{-1}$)",
            "vector_units": None,
            "vector_min": None,
            "name": "ascent700_rh700",
        },
    )

    local_levels = {}
    for spec in row_specs:
        shade = spec["shade"]
        local_levels[shade] = symmetric_levels(
            [effects["El Nino"][shade], effects["La Nina"][shade]],
            n_intervals=CFG.SHADE_N_INTERVALS,
            shade_name=shade,
        )

    # Shared RH contour levels within this season.
    rh_abs_list = []
    for phase_name in PHASES:
        rh = effects[phase_name]["rh700"]
        arr = np.abs(rh[np.isfinite(rh)])
        if arr.size:
            rh_abs_list.append(arr)

    if rh_abs_list:
        rh_abs = np.concatenate(rh_abs_list)
        rhmax = max(2.0, float(np.nanpercentile(rh_abs, 95)))
    else:
        rhmax = 4.0

    rhstep = max(2.0, math.ceil(rhmax / 3.0 / 2.0) * 2.0)
    rh_levels = np.array(
        [-3 * rhstep, -2 * rhstep, -rhstep, rhstep, 2 * rhstep, 3 * rhstep],
        dtype=float,
    )

    panel_letters = (("a", "b"), ("c", "d"), ("e", "f"))
    export_near_zero_thresholds(season_name, row_specs, local_levels)
    projection = ccrs.PlateCarree(
        central_longitude=CFG.MAP_CENTRAL_LONGITUDE
    )

    season_out_dir = CFG.SINGLE_PANEL_DIR / season_name
    season_out_dir.mkdir(parents=True, exist_ok=True)

    for row, spec in enumerate(row_specs):
        shade_name = spec["shade"]
        levels = (
            common_color_levels[shade_name]
            if common_color_levels is not None
            else local_levels[shade_name]
        )
        cmap = set_background_bad_color(
            publication_cmap(spec["cmap"], levels=levels)
        )
        norm = get_shade_norm(levels, shade_name=shade_name)

        shared_vector_ref = None
        if spec["u"] is not None and spec["v"] is not None:
            shared_vector_ref = combined_vector_reference(
                [
                    (effects[phase_name][spec["u"]], effects[phase_name][spec["v"]])
                    for phase_name in PHASES
                ],
                percentile=CFG.QUIVER_REF_PERCENTILE,
            )

        for col, phase_name in enumerate(PHASES):
            fig = plt.figure(figsize=CFG.SINGLE_FIGSIZE, facecolor="white")
            ax = fig.add_axes([
                CFG.SINGLE_LEFT,
                CFG.SINGLE_BOTTOM,
                CFG.SINGLE_RIGHT - CFG.SINGLE_LEFT,
                CFG.SINGLE_TOP - CFG.SINGLE_BOTTOM,
            ], projection=projection)

            draw_map_base(ax, basins, show_basin_labels=False)

            shade = effects[phase_name][shade_name]
            shade_plot = smooth_field_for_display(shade, shade_name=shade_name)
            shade_plot = apply_low_value_display_treatment(shade_plot, levels, shade_name=shade_name)
            shade_plot = apply_significance_area_fill_mask(
                shade_plot,
                significance[phase_name][shade_name],
                levels,
                shade_name=shade_name,
            )
            shade_plot = remove_small_display_patches(
                shade_plot,
                levels,
                shade_name=shade_name,
            )
            shade_plot = fill_field_edge_to_background(
                shade_plot,
                pad_cells=CFG.EDGE_PADDING_CELLS,
            )
            lon_fill, lat_fill, shade_fill = extend_field_to_plot_frame(
                lon, lat, shade_plot, CFG.PLOT_EXTENT
            )
            cf = ax.contourf(
                lon_fill, lat_fill, shade_fill,
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend="both",
                transform=ccrs.PlateCarree(),
                zorder=2,
                antialiased=False,
            )

            # Optional neutral-grey mask for non-study countries.
            # By default it is disabled so the atmospheric shading remains
            # visible over the full plotted domain.

            shade_other_countries(ax)

            # Black stippling: original grid-cell Welch p < 0.05, restricted to
            # cells retained by the final coloured display mask.
            stipple_mask = build_significance_stipple_mask(
                significance[phase_name][shade_name],
                displayed_field=shade_plot,
                shade_name=shade_name,
            )
            plot_significance_stippling(ax, lon, lat, stipple_mask)
            draw_threshold_boxes(ax, lon, lat, np.asarray(shade_plot), shade_name)

            if spec["u"] is not None and spec["v"] is not None:
                u = effects[phase_name][spec["u"]]
                v = effects[phase_name][spec["v"]]

                plot_thresholded_small_arrows(
                    ax, lon, lat, u, v,
                    units=spec["vector_units"],
                    desired_length_inches=(
                        CFG.WIND_ARROW_LENGTH_INCH
                        if row == 0
                        else CFG.QFLUX_ARROW_LENGTH_INCH
                    ),
                    shared_ref=shared_vector_ref,
                    show_key=True,
                    min_display_magnitude=spec["vector_min"],
                )

            if row == 2:
                rh = effects[phase_name]["rh700"]
                cs = ax.contour(
                    lon, lat, rh,
                    levels=rh_levels,
                    colors=CFG.RH_CONTOUR_COLOR,
                    linewidths=CFG.RH_CONTOUR_LINEWIDTH,
                    linestyles=CFG.RH_CONTOUR_LINESTYLE,
                    transform=ccrs.PlateCarree(),
                    zorder=10,
                )
                ax.clabel(
                    cs,
                    fmt=lambda x: f"{int(np.rint(x))}",
                    inline=True,
                    fontsize=8.0,
                    inline_spacing=3,
                    colors=CFG.RH_CONTOUR_COLOR,
                )

            letter = panel_letters[row][col]
            ax.set_title(
                f"({letter}) {PHASE_LABEL[phase_name]}",
                loc="left",
                fontsize=CFG.TITLE_SIZE,
                fontweight="bold",
                color="black",
                pad=CFG.TITLE_PAD,
            )

            # Short, centered single-panel colorbar.
            cax = fig.add_axes([0.145, 0.074, 0.71, 0.040])
            cb = fig.colorbar(
                cf,
                cax=cax,
                orientation="horizontal",
                extend="both",
                spacing="uniform",
                drawedges=True,
            )
            apply_fixed_colorbar_ticks(cb, levels, shade_name=shade_name)
            cb.set_label(
                spec["cbar_label"],
                fontsize=CFG.CBAR_LABEL_SIZE,
                fontweight="bold",
                labelpad=1.3,
            )
            cb.ax.tick_params(
                labelsize=CFG.CBAR_TICK_SIZE,
                width=0.8,
                length=2.2,
                pad=1.0,
            )
            if hasattr(cb, "dividers") and cb.dividers is not None:
                cb.dividers.set_color("0.35")
                cb.dividers.set_linewidth(0.6)
            cb.outline.set_linewidth(0.9)

            phase_file = "ElNino" if phase_name == "El Nino" else "LaNina"
            output_stem = (
                season_out_dir
                / f"Figure7Series_{CFG.FIGURE_LAYOUT_TAG}_{season_name}_{letter}_{phase_file}_{spec['name']}_single"
            )

            save_single_panel_figure(fig, output_stem)
            plt.close(fig)

    print(f"[PLOT DONE] {season_name}: {season_out_dir}")

def make_combined_six_panel_figure(
    season_name: str,
    lon: np.ndarray,
    lat: np.ndarray,
    effects: Dict[str, Dict[str, np.ndarray]],
    significance: Dict[str, Dict[str, np.ndarray]],
    basins: gpd.GeoDataFrame,
    common_color_levels: Optional[Dict[str, np.ndarray]] = None,
) -> None:

    set_style()

    row_specs = (
        {
            "shade": "div500",
            "u": "u700",
            "v": "v700",
            "cmap": CFG.CMAPS["div500"],
            "cbar_label": r"500-hPa divergence anomaly (10$^{-6}$ s$^{-1}$)",
            "vector_units": r"m s$^{-1}$",
            "vector_min": CFG.WIND_VECTOR_MIN_MAGNITUDE,
            "name": "div500_wind700",
        },
        {
            "shade": "mfc700",
            "u": "qfu700",
            "v": "qfv700",
            "cmap": CFG.CMAPS["mfc700"],
            "cbar_label": r"700-hPa moisture-flux convergence proxy (10$^{-8}$ s$^{-1}$)",
            "vector_units": r"g kg$^{-1}$ m s$^{-1}$",
            "vector_min": CFG.QFLUX_VECTOR_MIN_MAGNITUDE,
            "name": "mfc700_qv700",
        },
        {
            "shade": "ascent700",
            "u": None,
            "v": None,
            "cmap": CFG.CMAPS["ascent700"],
            "cbar_label": r"700-hPa ascent anomaly, $-\omega$ (Pa s$^{-1}$)",
            "vector_units": None,
            "vector_min": None,
            "name": "ascent700_rh700",
        },
    )

    local_levels = {}
    for spec in row_specs:
        shade = spec["shade"]
        local_levels[shade] = symmetric_levels(
            [effects["El Nino"][shade], effects["La Nina"][shade]],
            n_intervals=CFG.SHADE_N_INTERVALS,
            shade_name=shade,
        )

    rh_abs_list = []
    for phase_name in PHASES:
        rh = effects[phase_name]["rh700"]
        arr = np.abs(rh[np.isfinite(rh)])
        if arr.size:
            rh_abs_list.append(arr)

    if rh_abs_list:
        rh_abs = np.concatenate(rh_abs_list)
        rhmax = max(2.0, float(np.nanpercentile(rh_abs, 95)))
    else:
        rhmax = 4.0

    rhstep = max(2.0, math.ceil(rhmax / 3.0 / 2.0) * 2.0)
    rh_levels = np.array([-3 * rhstep, -2 * rhstep, -rhstep, rhstep, 2 * rhstep, 3 * rhstep], dtype=float)

    panel_letters = (("a", "b"), ("c", "d"), ("e", "f"))
    export_near_zero_thresholds(season_name, row_specs, local_levels)
    projection = ccrs.PlateCarree(
        central_longitude=CFG.MAP_CENTRAL_LONGITUDE
    )

    fig = plt.figure(figsize=CFG.FIGSIZE, facecolor="white")
    outer = fig.add_gridspec(
        nrows=3,
        ncols=1,
        left=CFG.BOARD_LEFT,
        right=CFG.BOARD_RIGHT,
        bottom=CFG.BOARD_BOTTOM,
        top=CFG.BOARD_TOP,
        hspace=CFG.BOARD_ROW_HSPACE,
    )

    for row, spec in enumerate(row_specs):
        map_grid = outer[row].subgridspec(
            1, 2,
            wspace=CFG.WSPACE,
        )
        shade_name = spec["shade"]
        levels = common_color_levels[shade_name] if common_color_levels is not None else local_levels[shade_name]
        cmap = set_background_bad_color(
            publication_cmap(spec["cmap"], levels=levels)
        )
        norm = get_shade_norm(levels, shade_name=shade_name)
        contour_handle = None
        row_axes = []

        shared_vector_ref = None
        if spec["u"] is not None and spec["v"] is not None:
            shared_vector_ref = combined_vector_reference(
                [
                    (effects[phase_name][spec["u"]], effects[phase_name][spec["v"]])
                    for phase_name in PHASES
                ],
                percentile=CFG.QUIVER_REF_PERCENTILE,
            )

        for col, phase_name in enumerate(PHASES):
            ax = fig.add_subplot(map_grid[0, col], projection=projection)
            row_axes.append(ax)
            draw_map_base(ax, basins, show_basin_labels=False)

            shade = effects[phase_name][shade_name]
            shade_plot = smooth_field_for_display(shade, shade_name=shade_name)
            shade_plot = apply_low_value_display_treatment(shade_plot, levels, shade_name=shade_name)
            shade_plot = apply_significance_area_fill_mask(
                shade_plot,
                significance[phase_name][shade_name],
                levels,
                shade_name=shade_name,
            )
            shade_plot = remove_small_display_patches(
                shade_plot,
                levels,
                shade_name=shade_name,
            )
            shade_plot = fill_field_edge_to_background(
                shade_plot,
                pad_cells=CFG.EDGE_PADDING_CELLS,
            )
            lon_fill, lat_fill, shade_fill = extend_field_to_plot_frame(
                lon, lat, shade_plot, CFG.PLOT_EXTENT
            )
            contour_handle = ax.contourf(
                lon_fill, lat_fill, shade_fill, levels=levels, cmap=cmap, norm=norm,
                extend="both", transform=ccrs.PlateCarree(), zorder=2,
                antialiased=False,
            )

            shade_other_countries(ax)

            # Black stippling: original grid-cell Welch p < 0.05, restricted to
            # cells retained by the final coloured display mask.
            stipple_mask = build_significance_stipple_mask(
                significance[phase_name][shade_name],
                displayed_field=shade_plot,
                shade_name=shade_name,
            )
            plot_significance_stippling(ax, lon, lat, stipple_mask)
            draw_threshold_boxes(ax, lon, lat, np.asarray(shade_plot), shade_name)

            if spec["u"] is not None and spec["v"] is not None:
                u = effects[phase_name][spec["u"]]
                v = effects[phase_name][spec["v"]]
                plot_thresholded_small_arrows(
                    ax, lon, lat, u, v,
                    units=spec["vector_units"],
                    desired_length_inches=(CFG.WIND_ARROW_LENGTH_INCH if row == 0 else CFG.QFLUX_ARROW_LENGTH_INCH),
                    shared_ref=shared_vector_ref,
                    show_key=(
                        (not CFG.BOARD_SHOW_QKEY_ONLY_ON_RIGHT_PANEL)
                        or col == 1
                    ),
                    min_display_magnitude=spec["vector_min"],
                    key_box_right=CFG.BOARD_QKEY_BOX_X,
                    key_box_scale=CFG.BOARD_QKEY_SCALE,
                    key_text_size=CFG.BOARD_QKEY_TEXT_SIZE,
                )

            if row == 2:
                rh = effects[phase_name]["rh700"]
                cs = ax.contour(
                    lon, lat, rh, levels=rh_levels,
                    colors=CFG.RH_CONTOUR_COLOR,
                    linewidths=CFG.RH_CONTOUR_LINEWIDTH,
                    linestyles=CFG.RH_CONTOUR_LINESTYLE,
                    transform=ccrs.PlateCarree(), zorder=10,
                )
                ax.clabel(
                    cs,
                    fmt=lambda x: f"{int(np.rint(x))}",
                    inline=True,
                    fontsize=8.0,
                    inline_spacing=3,
                    colors=CFG.RH_CONTOUR_COLOR,
                )

            letter = panel_letters[row][col]
            ax.set_title(
                f"({letter}) {PHASE_LABEL[phase_name]}",
                loc="left",
                fontsize=CFG.BOARD_TITLE_SIZE,
                fontweight="bold",
                color="black",
                pad=CFG.TITLE_PAD,
            )

        if not CFG.REMOVE_COMBINED_BOARD_COLORBARS:
            # Anchor the bar to the active frame of the right-hand map.  This
            # removes the large aspect-ratio-dependent blank gap produced by a
            # separate GridSpec column and makes the bar height match the map.
            row_axes[-1].apply_aspect()
            right_box = row_axes[-1].get_position()
            cax = fig.add_axes([
                right_box.x1 + CFG.BOARD_CBAR_PAD,
                right_box.y0,
                CFG.BOARD_CBAR_WIDTH,
                right_box.height,
            ])

            cb = fig.colorbar(
                contour_handle,
                cax=cax,
                orientation="vertical",
                extend="both",
                spacing="uniform",
                drawedges=True,
            )
            apply_fixed_vertical_colorbar_ticks(
                cb,
                levels,
                shade_name=shade_name,
            )
            cb.ax.tick_params(
                axis="y",
                labelsize=CFG.CBAR_TICK_SIZE,
                width=0.8,
                length=2.2,
                pad=1.0,
            )
            if hasattr(cb, "dividers") and cb.dividers is not None:
                cb.dividers.set_color("0.35")
                cb.dividers.set_linewidth(0.6)
            cb.outline.set_linewidth(0.9)

    season_out_dir = CFG.COMBINED_BOARD_DIR / season_name
    season_out_dir.mkdir(parents=True, exist_ok=True)
    stem = season_out_dir / f"Figure7Series_{CFG.FIGURE_LAYOUT_TAG}_{season_name}_sixpanel_board"
    save_figure(fig, stem)
    plt.close(fig)
    print(f"[BOARD DONE] {season_name}: {stem}")



# =============================================================================
# ORIGINAL BASIN-MONTH MECHANISM HEATMAP (UPDATED CEE STYLE)
# =============================================================================

def coordinate_edges(values: np.ndarray, latitude: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Latitude/longitude coordinates must be one-dimensional with >=2 cells.")
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    if latitude:
        edges = np.clip(edges, -90.0, 90.0)
    return edges


def build_fractional_basin_weights(
    lon: np.ndarray,
    lat: np.ndarray,
    basins: gpd.GeoDataFrame,
    cache_path: Path,
) -> Dict[str, np.ndarray]:
    """Fractional cell-basin intersection area (km2) in equal-area projection."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        try:
            with xr.open_dataset(cache_path) as ds:
                same_lon = np.array_equal(np.asarray(ds["lon"].values), np.asarray(lon))
                same_lat = np.array_equal(np.asarray(ds["lat"].values), np.asarray(lat))
                if same_lon and same_lat and all(name in ds for name in BASIN_ORDER):
                    print(f"[CACHE] basin intersection weights: {cache_path}")
                    return {
                        name: np.asarray(ds[name].values, dtype=float)
                        for name in BASIN_ORDER
                    }
                cached_shape = (
                    int(ds.sizes.get("lat", -1)),
                    int(ds.sizes.get("lon", -1)),
                )
                requested_shape = (int(len(lat)), int(len(lon)))
                print(
                    "[CACHE INVALID] basin-weight grid does not match the "
                    f"current heatmap grid: cached={cached_shape}, "
                    f"requested={requested_shape}. Rebuilding the cache."
                )
        except Exception as exc:
            print(f"[WARN] Could not reuse weight cache: {exc}")

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

    print(f"[HEATMAP] Building {len(polygons):,} grid-cell polygons for basin weights.")
    cells = gpd.GeoSeries(polygons, crs="EPSG:4326").to_crs(CFG.EQUAL_AREA_CRS)
    basin_ea = basins.to_crs(CFG.EQUAL_AREA_CRS).set_index("basin_name")

    weights: Dict[str, np.ndarray] = {}
    for basin in BASIN_ORDER:
        geom = basin_ea.loc[basin, "geometry"]
        if isinstance(geom, pd.Series):
            geom = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union

        candidate = np.asarray(cells.intersects(geom), dtype=bool)
        area = np.zeros(len(cells), dtype=float)
        if np.any(candidate):
            area[candidate] = (
                np.asarray(cells.loc[candidate].intersection(geom).area, dtype=float)
                / 1.0e6
            )
        arr = area.reshape(len(lat), len(lon))
        if not np.any(arr > 0):
            raise ValueError(f"No grid-cell intersection found for basin: {basin}")
        weights[basin] = arr
        print(f"[HEATMAP WEIGHTS] {basin}: {np.nansum(arr):,.0f} km2")

    xr.Dataset(
        {
            basin: (("lat", "lon"), values.astype("float32"))
            for basin, values in weights.items()
        },
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(cache_path)

    return weights


def weighted_basin_series(da: xr.DataArray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(da.values, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Expected time x lat x lon array, got shape={values.shape}")
    weight_values = np.asarray(weights, dtype=float)
    if weight_values.ndim != 2:
        raise ValueError(
            f"Expected a 2-D basin-weight array, got shape={weight_values.shape}"
        )
    if tuple(values.shape[-2:]) != tuple(weight_values.shape):
        raise ValueError(
            "Basin-weight grid mismatch after heatmap-grid harmonisation: "
            f"field={tuple(values.shape[-2:])}, weights={weight_values.shape}. "
            "Delete no data manually; rerun the revised script so the field "
            "is placed on the declared common grid and the weight cache is "
            "validated automatically."
        )
    w3 = weight_values[None, :, :]
    valid = np.isfinite(values) & np.isfinite(w3) & (w3 > 0)
    numerator = np.sum(np.where(valid, values * w3, 0.0), axis=(1, 2))
    denominator = np.sum(np.where(valid, w3, 0.0), axis=(1, 2))
    return np.divide(
        numerator,
        denominator,
        out=np.full(values.shape[0], np.nan, dtype=float),
        where=denominator > 0,
    )


def load_uv_year_for_heatmap(year: int) -> Tuple[xr.DataArray, xr.DataArray]:
    path = find_year_file(CFG.UV_DIR, year)
    with xr.open_dataset(path) as raw:
        ds = normalize_dataset(raw)
        u = resolve_variable(ds, ("u", "u_component_of_wind", "uwind", "u_wind"))
        v = resolve_variable(ds, ("v", "v_component_of_wind", "vwind", "v_wind"))
        u = select_pressure_level(u, 700.0)
        v = select_pressure_level(v, 700.0)
        u, v = xr.align(u, v, join="inner")
        u = crop_to_calc(u).load()
        v = crop_to_calc(v).load()
    return u, v


def load_pressure_year_for_heatmap(
    folder: Path,
    year: int,
    candidates: Sequence[str],
    level_hpa: float,
) -> xr.DataArray:
    path = find_year_file(folder, year)
    with xr.open_dataset(path) as raw:
        ds = normalize_dataset(raw)
        da = resolve_variable(ds, candidates)
        da = select_pressure_level(da, level_hpa)
        da = crop_to_calc(da).load()
    return da


def monthly_time_normalize(da: xr.DataArray) -> xr.DataArray:
    """
    Canonicalize monthly ERA5 timestamps to the first day of each month.

    Different CDS monthly products can encode the same calendar month using
    slightly different timestamps.  An exact xr.align(..., join="inner")
    would then remove every month even though the datasets are temporally
    compatible.  This helper converts every timestamp to YYYY-MM-01 first and
    averages duplicate records within a month when necessary.
    """
    if "time" not in da.coords:
        raise KeyError("Monthly ERA5 field has no time coordinate.")

    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(da["time"].values)))
    if dates.size == 0:
        return da

    canonical = pd.to_datetime({
        "year": dates.year,
        "month": dates.month,
        "day": np.ones(dates.size, dtype=int),
    })

    da = da.assign_coords(time=("time", canonical.to_numpy(dtype="datetime64[ns]")))

    # Average any duplicate records that now refer to the same calendar month.
    if pd.Index(canonical).duplicated().any():
        da = da.groupby("time").mean(dim="time", skipna=True)

    return da.sortby("time")


def _drop_or_collapse_extra_dims(da: xr.DataArray, label: str) -> xr.DataArray:
    """Collapse harmless singleton/expver dimensions before heatmap alignment."""
    for dim in list(da.dims):
        if dim in ("time", "lat", "lon"):
            continue

        if da.sizes.get(dim, 0) == 1:
            da = da.isel({dim: 0}, drop=True)
            continue

        # ERA5 can occasionally retain expver=1/5.  Merge available values
        # rather than allowing this auxiliary dimension to break transpose.
        if dim.lower() in ("expver", "number"):
            da = da.mean(dim=dim, skipna=True)
            continue

        raise ValueError(
            f"Unexpected non-spatial dimension {dim!r} in {label}: "
            f"shape={dict(da.sizes)}"
        )

    return da


def align_monthly_heatmap_fields(
    u700: xr.DataArray,
    v700: xr.DataArray,
    q700: xr.DataArray,
    rh700: xr.DataArray,
    omega700: xr.DataArray,
    evap: xr.DataArray,
    year: int,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Robustly align the six monthly fields used by the basin heatmap.

    Time is aligned by canonical calendar month.  Pressure-level fields are
    taken as the reference grid.  If evaporation uses a numerically different
    but overlapping grid, it is interpolated to that reference grid before the
    final exact temporal alignment.
    """
    fields = {
        "u700": _drop_or_collapse_extra_dims(monthly_time_normalize(u700), "u700"),
        "v700": _drop_or_collapse_extra_dims(monthly_time_normalize(v700), "v700"),
        "q700": _drop_or_collapse_extra_dims(monthly_time_normalize(q700), "q700"),
        "rh700": _drop_or_collapse_extra_dims(monthly_time_normalize(rh700), "rh700"),
        "omega700": _drop_or_collapse_extra_dims(monthly_time_normalize(omega700), "omega700"),
        "evap": _drop_or_collapse_extra_dims(monthly_time_normalize(evap), "evap"),
    }

    # First align the five pressure-level fields.
    u700, v700, q700, rh700, omega700 = xr.align(
        fields["u700"], fields["v700"], fields["q700"],
        fields["rh700"], fields["omega700"],
        join="inner",
    )

    if u700.sizes.get("time", 0) == 0:
        details = {
            k: [str(pd.Timestamp(t).date()) for t in np.asarray(v["time"].values)[:3]]
            for k, v in fields.items() if "time" in v.coords
        }
        raise RuntimeError(
            f"{year}: no common months among pressure-level ERA5 fields after "
            f"timestamp normalization. First timestamps: {details}"
        )

    evap = fields["evap"]

    # Put evaporation on the pressure-level reference grid when required.
    same_lon = (
        evap.sizes.get("lon", -1) == u700.sizes.get("lon", -2)
        and np.allclose(
            np.asarray(evap["lon"].values, dtype=float),
            np.asarray(u700["lon"].values, dtype=float),
            atol=1e-8, rtol=0.0, equal_nan=True,
        )
    )
    same_lat = (
        evap.sizes.get("lat", -1) == u700.sizes.get("lat", -2)
        and np.allclose(
            np.asarray(evap["lat"].values, dtype=float),
            np.asarray(u700["lat"].values, dtype=float),
            atol=1e-8, rtol=0.0, equal_nan=True,
        )
    )

    if not (same_lon and same_lat):
        print(f"[HEATMAP GRID] {year}: interpolating evaporation to pressure-level grid.")
        evap = evap.interp(
            lon=u700["lon"],
            lat=u700["lat"],
            method="linear",
        )

    # Final alignment is now principally a calendar-month intersection.
    u700, v700, q700, rh700, omega700, evap = xr.align(
        u700, v700, q700, rh700, omega700, evap,
        join="inner",
    )

    ntime = u700.sizes.get("time", 0)
    if ntime == 0:
        raise RuntimeError(
            f"{year}: zero common monthly records after ERA5 field alignment. "
            "Check the timestamps and spatial grids of 07_Evaporation versus "
            "the pressure-level folders."
        )

    if ntime != 12:
        months = [pd.Timestamp(t).strftime("%Y-%m") for t in np.asarray(u700["time"].values)]
        print(f"[WARN] {year}: {ntime} aligned monthly records -> {months}")

    return u700, v700, q700, rh700, omega700, evap


def load_evaporation_year_for_heatmap(year: int) -> xr.DataArray:
    path = find_year_file(CFG.EVAP_DIR, year)
    with xr.open_dataset(path) as raw:
        ds = normalize_dataset(raw)
        da = resolve_variable(ds, ("e", "evaporation", "pev", "evap"))
        da = monthly_time_normalize(da)
        da = crop_to_calc(da).astype("float32").load()

    units = str(da.attrs.get("units", "")).lower().replace("**", "^").strip()
    metre_units = (
        units in ("m", "metre", "meter", "m/day", "m day-1", "m d-1")
        or units.startswith("m of water")
        or units.startswith("m ")
    )
    mm_units = any(token in units for token in ("kg m-2", "kg m^-2", "kg/m2", "mm"))

    if metre_units:
        da = da * 1000.0
    elif mm_units:
        pass
    elif not units:
        # CDS ERA5 evaporation is normally in metres of water equivalent.
        # Do not silently convert missing-unit data.
        raise ValueError(
            f"Evaporation file {path.name} has no units attribute. Verify the CDS product before plotting."
        )
    else:
        raise ValueError(f"Unsupported evaporation units '{units}' in {path.name}")

    if CFG.EVAP_ACCUMULATION_MODE == "monthly_mean_daily_total":
        days = xr.DataArray(
            pd.DatetimeIndex(np.asarray(da["time"].values)).days_in_month.astype(float),
            dims=("time",),
            coords={"time": da["time"]},
        )
        da = da * days
    elif CFG.EVAP_ACCUMULATION_MODE == "monthly_total":
        pass
    else:
        raise ValueError(
            "EVAP_ACCUMULATION_MODE must be 'monthly_mean_daily_total' or 'monthly_total'."
        )

    if CFG.ERA5_EVAPORATION_UPWARD_IS_NEGATIVE:
        da = -da

    da.attrs["units"] = "mm month^-1"
    return da


def build_basin_month_raw_table(
    phase_table: pd.DataFrame,
    basins: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Build 1961--2023 monthly basin means without retaining full cubes."""
    phase_lookup = {
        (int(row.cal_year), int(row.cal_month)): int(row.phase_code)
        for row in phase_table.itertuples(index=False)
    }

    weights: Optional[Dict[str, np.ndarray]] = None
    grid_lon: Optional[np.ndarray] = None
    grid_lat: Optional[np.ndarray] = None
    rows = []

    for year in range(CFG.START_YEAR, CFG.END_YEAR + 1):
        print(f"[HEATMAP READ] {year}")
        u700, v700 = load_uv_year_for_heatmap(year)
        q700 = load_pressure_year_for_heatmap(
            CFG.Q_DIR, year,
            ("q", "specific_humidity", "specifichumidity"),
            700.0,
        )
        rh700 = load_pressure_year_for_heatmap(
            CFG.RH_DIR, year,
            ("r", "rh", "relative_humidity", "relativehumidity"),
            700.0,
        )
        omega700 = load_pressure_year_for_heatmap(
            CFG.OMEGA_DIR, year,
            ("w", "omega", "vertical_velocity", "verticalvelocity"),
            700.0,
        )
        evap = load_evaporation_year_for_heatmap(year)

        u700, v700, q700, rh700, omega700, evap = align_monthly_heatmap_fields(
            u700=u700,
            v700=v700,
            q700=q700,
            rh700=rh700,
            omega700=omega700,
            evap=evap,
            year=year,
        )

        qfu = q700 * u700
        qfv = q700 * v700
        mfc700 = -spherical_horizontal_divergence(qfu, qfv) * 1.0e8
        ascent700 = -omega700

        # RH in some products may be stored in 0-1 rather than percent.
        rh_values = np.asarray(rh700.values, dtype=float)
        finite_rh = np.abs(rh_values[np.isfinite(rh_values)])
        if finite_rh.size and np.nanpercentile(finite_rh, 95) <= 1.5:
            rh700 = rh700 * 100.0

        fields = {
            "mfc700": crop_to_plot(mfc700),
            "ascent700": crop_to_plot(ascent700),
            "rh700": crop_to_plot(rh700),
            "evap": crop_to_plot(evap),
        }
        fields = {
            name: da.transpose("time", "lat", "lon")
            for name, da in fields.items()
        }

        # ERA5 folders can legitimately use different native resolutions
        # (the failing run mixed a 301 x 561 field with 151 x 281 basin
        # weights).  Harmonise all intensive diagnostics to the mfc700 grid
        # before one common set of fractional basin areas is applied.
        # Lock every analysis year to the first successfully processed
        # mfc700 grid.  Some annual ERA5 files in the archive use 0.25 degree
        # (151 x 281 here), whereas others use 0.125 degree (301 x 561).
        # Reusing first-year basin weights is valid only after this second,
        # across-year grid lock.
        fields = harmonize_basin_heatmap_grids(
            fields,
            year,
            target_lon=grid_lon,
            target_lat=grid_lat,
        )

        if weights is None:
            grid_lon = np.asarray(fields["mfc700"]["lon"].values, dtype=float)
            grid_lat = np.asarray(fields["mfc700"]["lat"].values, dtype=float)
            weights = build_fractional_basin_weights(
                grid_lon, grid_lat, basins, CFG.HEATMAP_WEIGHT_CACHE
            )

        times = pd.DatetimeIndex(np.asarray(fields["mfc700"]["time"].values))
        basin_series = {
            variable: {
                basin: weighted_basin_series(da, weights[basin])
                for basin in BASIN_ORDER
            }
            for variable, da in fields.items()
        }

        matched_months = 0
        for it, stamp in enumerate(times):
            code = phase_lookup.get((int(stamp.year), int(stamp.month)), 99)
            if code not in (-1, 0, 1):
                continue
            matched_months += 1
            for basin in BASIN_ORDER:
                rows.append({
                    "cal_year": int(stamp.year),
                    "cal_month": int(stamp.month),
                    "phase_code": int(code),
                    "basin": basin,
                    "mfc700": float(basin_series["mfc700"][basin][it]),
                    "ascent700": float(basin_series["ascent700"][basin][it]),
                    "rh700": float(basin_series["rh700"][basin][it]),
                    "evap": float(basin_series["evap"][basin][it]),
                })

        if matched_months == 0:
            print(
                f"[WARN] {year}: ERA5 months were present but none matched the ENSO phase table. "
                f"ERA5 months={[int(m) for m in times.month.tolist()]}"
            )
        else:
            print(f"[HEATMAP MATCH] {year}: {matched_months} ENSO-classified months x 6 basins")

    table = pd.DataFrame(rows)
    if table.empty:
        phase_min = None if phase_table.empty else (
            int(phase_table["cal_year"].min()),
            int(phase_table["cal_month"].iloc[0]),
        )
        phase_max = None if phase_table.empty else (
            int(phase_table["cal_year"].max()),
            int(phase_table["cal_month"].iloc[-1]),
        )
        raise RuntimeError(
            "No monthly basin mechanism records were constructed even after "
            "canonical month alignment. This means the remaining mismatch is "
            "between the ERA5 calendar months and the ENSO phase table, or the "
            "input fields contain no usable months. "
            f"Phase-table rows={len(phase_table)}, range={phase_min}..{phase_max}."
        )
    return table


def detrended_scalar_effect(
    values: np.ndarray,
    years: np.ndarray,
    codes: np.ndarray,
    phase_code: int,
) -> Tuple[float, int, int]:
    values = np.asarray(values, dtype=float)
    years = np.asarray(years, dtype=float)
    codes = np.asarray(codes, dtype=int)
    usable = np.isfinite(values) & np.isfinite(years) & np.isin(codes, (-1, 0, 1))
    values = values[usable]
    years = years[usable]
    codes = codes[usable]

    if values.size < 3:
        return np.nan, 0, 0

    residual = linear_detrend_cube(values[:, None], years)[:, 0]
    phase = residual[codes == phase_code]
    neutral = residual[codes == 0]
    phase = phase[np.isfinite(phase)]
    neutral = neutral[np.isfinite(neutral)]

    n_phase = len(phase)
    n_neutral = len(neutral)
    if n_phase < CFG.MIN_PHASE_N or n_neutral < CFG.MIN_NEUTRAL_N:
        return np.nan, n_phase, n_neutral

    pooled_df = n_phase + n_neutral - 2
    pooled_variance = (
        (n_phase - 1) * np.var(phase, ddof=1)
        + (n_neutral - 1) * np.var(neutral, ddof=1)
    ) / pooled_df

    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return np.nan, n_phase, n_neutral

    cohen_d = (np.mean(phase) - np.mean(neutral)) / np.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * (n_phase + n_neutral) - 9.0)
    return float(cohen_d * correction), n_phase, n_neutral


def compute_basin_month_hedges_table(raw: pd.DataFrame) -> pd.DataFrame:
    variables = ("mfc700", "ascent700", "rh700", "evap")
    rows = []

    for variable in variables:
        for basin in BASIN_ORDER:
            basin_sub = raw[raw["basin"] == basin]
            for month in MONTHS:
                sub = basin_sub[basin_sub["cal_month"] == month]
                years = sub["cal_year"].to_numpy(dtype=int)
                codes = sub["phase_code"].to_numpy(dtype=int)
                values = sub[variable].to_numpy(dtype=float)

                for phase_name in PHASES:
                    effect, n_phase, n_neutral = detrended_scalar_effect(
                        values, years, codes, PHASE_CODE[phase_name]
                    )
                    rows.append({
                        "variable": variable,
                        "phase": PHASE_LABEL[phase_name],
                        "basin": basin,
                        "calendar_month": month,
                        "hedges_g_phase_minus_neutral": effect,
                        "phase_years": n_phase,
                        "neutral_years": n_neutral,
                    })

    return pd.DataFrame(rows)


def plot_basin_month_heatmaps(table: pd.DataFrame, output_prefix: Path) -> None:
    """Updated compact CEE-style version of the original mechanism heatmap."""
    variables = (
        ("mfc700", "700-hPa moisture-flux convergence"),
        ("ascent700", r"700-hPa ascent ($-\omega$)"),
        ("rh700", "700-hPa relative humidity"),
        ("evap", "Surface evaporation"),
    )
    phase_labels = ("El Niño", "La Niña")
    levels = np.asarray(CFG.HEATMAP_LEVELS, dtype=float)

    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad(CFG.OTHER_COUNTRY_FACE)
    norm = get_shade_norm(levels, shade_name=shade_name if "shade_name" in locals() else None)

    fig, axes = plt.subplots(
        4, 2,
        figsize=CFG.HEATMAP_FIGSIZE,
        sharex=True,
        sharey=True,
        facecolor="white",
    )

    letters = iter("abcdefgh")
    image = None

    for row, (variable, variable_label) in enumerate(variables):
        for col, phase in enumerate(phase_labels):
            ax = axes[row, col]
            sub = table[
                (table["variable"] == variable)
                & (table["phase"] == phase)
            ]
            matrix = sub.pivot(
                index="basin",
                columns="calendar_month",
                values="hedges_g_phase_minus_neutral",
            )
            matrix = matrix.reindex(index=BASIN_ORDER, columns=MONTHS)

            image = ax.imshow(
                matrix.values,
                cmap=cmap,
                norm=norm,
                aspect="auto",
                interpolation="nearest",
                zorder=1,
            )

            ax.set_title(
                f"({next(letters)}) {phase}: {variable_label}",
                loc="left",
                fontsize=CFG.HEATMAP_TITLE_SIZE,
                fontweight="bold",
                pad=5,
            )

            ax.set_xticks(np.arange(12), MONTH_LABELS)
            ax.set_yticks(np.arange(6), BASIN_ORDER)
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=CFG.HEATMAP_TICK_SIZE,
                length=3.0,
                width=0.8,
                pad=2.0,
            )
            for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
                tick.set_fontweight("bold")
                tick.set_fontfamily(CFG.FONT_FAMILY)

            # Thin white cell boundaries for clean CEE-like matrix separation.
            ax.set_xticks(np.arange(-0.5, 12, 1.0), minor=True)
            ax.set_yticks(np.arange(-0.5, 6, 1.0), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.75, zorder=3)
            ax.tick_params(which="minor", bottom=False, left=False)

            for spine in ax.spines.values():
                spine.set_linewidth(0.85)
                spine.set_color("#333333")

            if col == 0:
                ax.set_ylabel(
                    "River basin",
                    fontsize=CFG.HEATMAP_TICK_SIZE,
                    fontweight="bold",
                )

    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.982,
        bottom=0.135,
        wspace=CFG.HEATMAP_WSPACE,
        hspace=CFG.HEATMAP_HSPACE,
    )

    if image is None:
        raise RuntimeError("Heatmap image was not created.")

    cbar_width = CFG.HEATMAP_CBAR_WIDTH
    cax = fig.add_axes([
        0.5 - cbar_width / 2.0,
        0.050,
        cbar_width,
        0.030,
    ])
    cbar = fig.colorbar(
        image,
        cax=cax,
        orientation="horizontal",
        ticks=levels,
        spacing="uniform",
        drawedges=True,
        extend="both",
        extendfrac=0.035,
    )
    cbar.set_label(
        r"Detrended ENSO$-$Neutral effect (Hedges' $g$)",
        fontsize=CFG.HEATMAP_CBAR_LABEL_SIZE,
        fontweight="bold",
        labelpad=2.5,
    )
    cbar.ax.tick_params(
        labelsize=CFG.HEATMAP_CBAR_TICK_SIZE,
        length=2.5,
        width=0.8,
        pad=1.2,
    )
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontweight("bold")
        tick.set_fontfamily(CFG.FONT_FAMILY)
    cbar.outline.set_linewidth(0.85)
    if hasattr(cbar, "dividers"):
        cbar.dividers.set_color("white")
        cbar.dividers.set_linewidth(0.65)

    save_figure(fig, output_prefix)
    print(f"[HEATMAP DONE] {output_prefix}")


def make_original_mechanism_heatmap(
    phase_table: pd.DataFrame,
    basins: gpd.GeoDataFrame,
) -> None:
    """
    Plot the supplementary basin-month mechanism heatmap.

    SPEED FIX:
    The original script always reread the complete 1961-2023 ERA5 archive.
    Now an existing Hedges-g CSV is reused by default.  Full rebuilding occurs
    only when REBUILD_BASIN_MONTH_HEATMAP_DATA=True.
    """
    if not CFG.EXPORT_BASIN_MONTH_HEATMAP:
        print("[HEATMAP SKIP] EXPORT_BASIN_MONTH_HEATMAP=False")
        return

    CFG.HEATMAP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = CFG.HEATMAP_FIG_DIR / f"FigureS22_{CFG.FIGURE_LAYOUT_TAG}_ERA5_basin_month_mechanism_heatmaps_CEE_refined"

    if (
        CFG.USE_EXISTING_HEATMAP_TABLE
        and CFG.HEATMAP_TABLE.exists()
        and not CFG.REBUILD_BASIN_MONTH_HEATMAP_DATA
    ):
        print(f"[HEATMAP CACHE HIT] {CFG.HEATMAP_TABLE}")
        heatmap_table = pd.read_csv(CFG.HEATMAP_TABLE)
        plot_basin_month_heatmaps(heatmap_table, output_prefix)
        return

    if not CFG.REBUILD_BASIN_MONTH_HEATMAP_DATA:
        print(
            "[HEATMAP SKIP] No existing heatmap table was found and "
            "REBUILD_BASIN_MONTH_HEATMAP_DATA=False. "
            "This avoids an unnecessary second 63-year ERA5 pass."
        )
        return

    print()
    print("=" * 80)
    print("REBUILDING BASIN-MONTH MECHANISM HEATMAP FROM RAW ERA5")
    print("=" * 80)

    raw_table = build_basin_month_raw_table(phase_table, basins)
    heatmap_table = compute_basin_month_hedges_table(raw_table)

    CFG.HEATMAP_TABLE.parent.mkdir(parents=True, exist_ok=True)
    heatmap_table.to_csv(CFG.HEATMAP_TABLE, index=False, encoding="utf-8-sig")
    plot_basin_month_heatmaps(heatmap_table, output_prefix)

    print(f"[HEATMAP TABLE] {CFG.HEATMAP_TABLE}")


# =============================================================================
# MAIN
# =============================================================================

def audit_inputs() -> None:
    """Fail early with a clear message if the expanded-domain archive is incomplete."""
    # Each tuple is (folder, first required raw year).  Because DJF 1961 is
    # excluded when December 1960 has no phase label, every calculated field
    # now requires ERA5 beginning in 1961, not an unsupported 1960 phase.
    required = {
        "UV 500/700 hPa": (CFG.UV_DIR, CFG.START_YEAR),
        "specific humidity 700 hPa": (CFG.Q_DIR, CFG.START_YEAR),
        "relative humidity 700 hPa": (CFG.RH_DIR, CFG.START_YEAR),
        "vertical velocity 700 hPa": (CFG.OMEGA_DIR, CFG.START_YEAR),
    }
    if CFG.USE_SURFACE_PRESSURE_MASK:
        required["surface pressure"] = (CFG.SP_DIR, CFG.START_YEAR)
    if CFG.EXPORT_BASIN_MONTH_HEATMAP and CFG.REBUILD_BASIN_MONTH_HEATMAP_DATA:
        required["evaporation"] = (CFG.EVAP_DIR, CFG.START_YEAR)

    print("[INPUT AUDIT]")
    for label, (folder, first_required_year) in required.items():
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing ERA5 folder for {label}:\n  {folder}\n"
                "The expanded calculation requires the GLOBAL archive; do not "
                "point ERA5_ROOT to the older 55E-180E subset."
            )
        first = find_year_file(folder, first_required_year)
        last = find_year_file(folder, CFG.INPUT_END_YEAR)
        print(f"  [OK] {label}: {folder}")
        print(
            f"       required raw coverage {first_required_year}-"
            f"{CFG.INPUT_END_YEAR}: {first.name} ... {last.name}"
        )

    for label, path in (
        ("ENSO phase source", CFG.PHASE_SOURCE),
        ("six-basin polygons", CFG.BASIN_FILE),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}:\n  {path}")
        print(f"  [OK] {label}: {path}")

    # Check actual coordinate coverage using one UV file because U/V provide
    # the derivative grid for both circulation and moisture-flux convergence.
    sample = find_year_file(CFG.UV_DIR, CFG.START_YEAR)
    with xr.open_dataset(sample) as raw:
        sample_ds = normalize_dataset(raw)
        lon_min = float(sample_ds.lon.min())
        lon_max = float(sample_ds.lon.max())
        lat_min = float(sample_ds.lat.min())
        lat_max = float(sample_ds.lat.max())

    west, east, south, north = CFG.CALC_EXTENT
    tolerance = 0.26  # one typical ERA5 0.25-degree cell
    if (
        lon_min > west + tolerance
        or lon_max < east - tolerance
        or lat_min > south + tolerance
        or lat_max < north - tolerance
    ):
        raise ValueError(
            "The selected ERA5 archive does not cover CALC_EXTENT.\n"
            f"  requested : {CFG.CALC_EXTENT}\n"
            f"  available : lon {lon_min:g}..{lon_max:g}, "
            f"lat {lat_min:g}..{lat_max:g}\n"
            "Use a sufficiently large ERA5 archive under data/era5 or another archive that "
            "fully covers 35E-185E and 20S-65N."
        )

    print(
        f"  [OK] coordinate coverage: lon {lon_min:g}..{lon_max:g}, "
        f"lat {lat_min:g}..{lat_max:g}"
    )


def _same_horizontal_grid(
    left: xr.DataArray,
    right: xr.DataArray,
    atol: float = 1.0e-8,
) -> bool:
    """Return True only when two fields have the same ordered lat/lon grid."""
    for coordinate in ("lat", "lon"):
        if coordinate not in left.coords or coordinate not in right.coords:
            return False
        if left.sizes.get(coordinate, -1) != right.sizes.get(coordinate, -2):
            return False
        if not np.allclose(
            np.asarray(left[coordinate].values, dtype=float),
            np.asarray(right[coordinate].values, dtype=float),
            atol=atol,
            rtol=0.0,
            equal_nan=False,
        ):
            return False
    return True


def harmonize_basin_heatmap_grids(
    fields: Mapping[str, xr.DataArray],
    year: int,
    target_lon: Optional[np.ndarray] = None,
    target_lat: Optional[np.ndarray] = None,
) -> Dict[str, xr.DataArray]:
    """
    Put every basin-month diagnostic on one explicitly declared grid.

    The moisture-flux-convergence field is the reference because it is the
    core diagnostic of this heatmap and is commonly the coarsest native grid.
    Regridding the other *intensive* fields to this grid avoids inventing a
    finer effective resolution and ensures that one set of fractional
    cell-basin intersection areas is valid for every variable.
    """
    if "mfc700" not in fields:
        raise KeyError("mfc700 is required as the basin-heatmap reference grid")

    reference = fields["mfc700"].transpose("time", "lat", "lon")

    # Within one year the reference is mfc700.  Across years, however, the
    # archive can change native resolution.  Once target_lon/target_lat have
    # been set from the first year, mfc700 itself must also be interpolated to
    # that fixed grid before any other variable or cached weight is used.
    if (target_lon is None) != (target_lat is None):
        raise ValueError("target_lon and target_lat must be supplied together")
    if target_lon is not None and target_lat is not None:
        target_lon = np.asarray(target_lon, dtype=float)
        target_lat = np.asarray(target_lat, dtype=float)
        same_target = (
            reference.sizes.get("lon", -1) == target_lon.size
            and reference.sizes.get("lat", -1) == target_lat.size
            and np.allclose(
                np.asarray(reference["lon"].values, dtype=float),
                target_lon,
                atol=1.0e-8,
                rtol=0.0,
            )
            and np.allclose(
                np.asarray(reference["lat"].values, dtype=float),
                target_lat,
                atol=1.0e-8,
                rtol=0.0,
            )
        )
        if not same_target:
            old_shape = (
                int(reference.sizes["lat"]),
                int(reference.sizes["lon"]),
            )
            target_shape = (int(target_lat.size), int(target_lon.size))
            print(
                f"[HEATMAP GRID LOCK] {year} mfc700: {old_shape} -> "
                f"{target_shape} (fixed first-year grid)."
            )
            attrs = dict(reference.attrs)
            reference = reference.interp(
                lon=xr.DataArray(target_lon, dims=("lon",)),
                lat=xr.DataArray(target_lat, dims=("lat",)),
                method="linear",
            )
            reference.attrs.update(attrs)
    output: Dict[str, xr.DataArray] = {}
    reference_shape = (
        int(reference.sizes["lat"]),
        int(reference.sizes["lon"]),
    )

    for name, source in fields.items():
        # Use the already grid-locked mfc700 rather than the original source,
        # otherwise mfc700 could silently revert to its current-year grid.
        da = (
            reference.copy(deep=False)
            if name == "mfc700"
            else source.transpose("time", "lat", "lon")
        )
        source_shape = (int(da.sizes["lat"]), int(da.sizes["lon"]))
        if not _same_horizontal_grid(da, reference):
            print(
                f"[HEATMAP GRID] {year} {name}: {source_shape} -> "
                f"{reference_shape} (linear interpolation to mfc700 grid)."
            )
            attrs = dict(da.attrs)
            da = da.interp(
                lon=reference["lon"],
                lat=reference["lat"],
                method="linear",
            )
            da.attrs.update(attrs)

        da, reference_time = xr.align(da, reference, join="inner", exclude=("lat", "lon"))
        if da.sizes.get("time", 0) != reference.sizes.get("time", 0):
            raise RuntimeError(
                f"{year} {name}: time axis changed during heatmap-grid "
                "harmonisation. Check missing calendar months."
            )
        if tuple(da.shape[-2:]) != reference_shape:
            raise RuntimeError(
                f"{year} {name}: failed to reach the common heatmap grid; "
                f"received {tuple(da.shape[-2:])}, expected {reference_shape}."
            )
        if not np.isfinite(np.asarray(da.values, dtype=float)).any():
            raise RuntimeError(
                f"{year} {name}: interpolation produced no finite values. "
                "Check coordinate order and spatial overlap."
            )
        output[name] = da.astype("float32")

    return output


def main() -> None:
    total_start = time.perf_counter()

    print("=" * 80)
    print("Figure 7 full seasonal series | FAST CACHE-AWARE VERSION")
    print("=" * 80)
    print(f"ERA5 root      : {CFG.ERA5_ROOT}")
    print(f"Analysis years : {CFG.START_YEAR}-{CFG.END_YEAR}")
    print(f"Seasons        : {CFG.SEASON_ORDER}")
    print(f"Calc extent    : {CFG.CALC_EXTENT}")
    print(f"Plot extent    : {CFG.PLOT_EXTENT}")
    print(
        "Vector params  : "
        f"style=dual-blue-open-quiver, spacing_deg={CFG.QUIVER_SPACING_DEG}, "
        f"width={CFG.QUIVER_WIDTH}, arrow_length_in={CFG.WIND_ARROW_LENGTH_INCH}, "
        f"outline_width_pt={CFG.QUIVER_OUTLINE_LINEWIDTH_PT}, "
        f"low_direction_factor={CFG.QUIVER_LOW_DIRECTION_LENGTH_FACTOR}, "
        f"max_length_factor={CFG.QUIVER_MAX_THRESHOLD_MULTIPLIER}, "
        f"overlap_clearance={CFG.QUIVER_OVERLAP_CLEARANCE_FACTOR}, "
        f"wind_min={CFG.WIND_VECTOR_MIN_MAGNITUDE}, "
        f"qflux_min={CFG.QFLUX_VECTOR_MIN_MAGNITUDE}, "
        f"smooth_sigma={CFG.QUIVER_SMOOTH_SIGMA}, "
        f"low_color={CFG.QUIVER_LOW_COLOR}, high_color={CFG.QUIVER_HIGH_COLOR}, "
        f"numeric_key={CFG.SHOW_QUIVER_NUMERIC_KEY}"
    )
    print(
        "Fill params    : "
        f"near_zero_factor_default={CFG.ZERO_MASK_STEP_FACTOR}, "
        f"display_smooth_sigma_default={CFG.DISPLAY_SMOOTH_SIGMA}, "
        f"div500_sigma={CFG.DISPLAY_SMOOTH_SIGMA_BY_SHADE.get('div500')}, "
        f"div500_zero_factor={CFG.ZERO_MASK_STEP_FACTOR_BY_SHADE.get('div500')}, "
        f"div500_min_region={CFG.MIN_SIGNIFICANT_REGION_CELLS_BY_SHADE.get('div500')}, "
        f"n_intervals={CFG.SHADE_N_INTERVALS}"
    )
    print(f"Column wspace  : {CFG.WSPACE}")
    print(f"Fast preview   : {CFG.FAST_PREVIEW}")
    print(
        "Export formats : "
        f"PNG={CFG.EXPORT_PNG}, TIF={CFG.EXPORT_TIF}, PDF={CFG.EXPORT_PDF}, "
        f"DPI={CFG.DPI}"
    )
    print(f"Effect cache   : {CFG.USE_SEASON_EFFECT_CACHE}")
    print("=" * 80)

    # Verify the complete map/colour-bar fill mapping before any ERA5 files
    # are opened.  A configuration error therefore stops immediately instead
    # of appearing only after a long calculation.
    validate_reference_color_scales()

    CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
    CFG.VERSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CFG.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CFG.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CFG.SEASON_EFFECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Basin polygons are always needed for plotting.
    basins = load_six_basins()

    # First try the compact seasonal-effect caches.  If all four are available,
    # no raw ERA5 file needs to be read merely to change figure styling.
    season_results: Dict[str, Dict[str, object]] = {}
    missing_seasons: list[str] = []

    for season_name in CFG.SEASON_ORDER:
        cached = load_season_effect_cache(season_name)
        if cached is None:
            missing_seasons.append(season_name)
        else:
            season_results[season_name] = cached

    need_heatmap_raw = (
        CFG.EXPORT_BASIN_MONTH_HEATMAP
        and CFG.REBUILD_BASIN_MONTH_HEATMAP_DATA
    )
    need_raw_era5 = bool(missing_seasons) or need_heatmap_raw

    phase_table: Optional[pd.DataFrame] = None

    if need_raw_era5:
        print()
        print("[RAW ERA5 REQUIRED]")
        if missing_seasons:
            print("  Missing/stale seasonal caches:", ", ".join(missing_seasons))
        if need_heatmap_raw:
            print("  Basin-month heatmap raw rebuild is enabled.")

        audit_inputs()
        phase_table = load_phase_table()

        for season_name in missing_seasons:
            season_start = time.perf_counter()
            print()
            print("#" * 80)
            print(f"CALCULATING {season_name}")
            print("#" * 80)

            seasonal, years, codes = build_seasonal_dataset(
                season_name,
                phase_table,
            )

            effects = compute_composites(
                seasonal,
                years,
                codes,
            )
            significance = compute_significance_masks(
                seasonal,
                years,
                codes,
            )

            lon = np.asarray(seasonal["lon"].values, dtype=float)
            lat = np.asarray(seasonal["lat"].values, dtype=float)

            season_results[season_name] = {
                "lon": lon,
                "lat": lat,
                "effects": effects,
                "significance": significance,
            }

            if CFG.USE_SEASON_EFFECT_CACHE:
                save_season_effect_cache(
                    season_name=season_name,
                    lon=lon,
                    lat=lat,
                    effects=effects,
                    significance=significance,
                )

            del seasonal
            gc.collect()

            elapsed = time.perf_counter() - season_start
            print(f"[SEASON DONE] {season_name} | {elapsed / 60.0:.1f} min")
    else:
        print()
        print("[FAST REDRAW] All seasonal effect caches found.")
        print("[FAST REDRAW] Raw 1961-2023 ERA5 calculation is completely skipped.")

    # Phase table is only needed for a raw heatmap rebuild.  Existing heatmap
    # tables and disabled heatmaps do not require it.
    if need_heatmap_raw and phase_table is None:
        phase_table = load_phase_table()

    if CFG.USE_COMMON_COLOR_SCALE_ACROSS_SEASONS:
        common_levels = compute_common_color_levels(season_results)
        print("[COLOR SCALE] Common diagnostic scales across all seasons.")
        print(
            "[STYLE] v102: reference blue-negative/red-positive shading, "
            "purple thresholded streamlines with vertical boxed keys, "
            "column-wise a/d-b/e-c/f lettering, integer RH labels, and "
            "purple solid RH contours."
        )
    else:
        common_levels = None
        print("[COLOR SCALE] Season-specific diagnostic scales.")

    # Plot from cached/computed effects.
    for season_name in CFG.SEASON_ORDER:
        plot_start = time.perf_counter()
        result = season_results[season_name]

        if CFG.EXPORT_COMBINED_BOARDS:
            make_combined_six_panel_figure(
                season_name=season_name,
                lon=result["lon"],
                lat=result["lat"],
                effects=result["effects"],
                significance=result["significance"],
                basins=basins,
                common_color_levels=common_levels,
            )

        if CFG.EXPORT_SINGLE_PANELS:
            make_single_panel_figures(
                season_name=season_name,
                lon=result["lon"],
                lat=result["lat"],
                effects=result["effects"],
                significance=result["significance"],
                basins=basins,
                common_color_levels=common_levels,
            )

        print(
            f"[PLOT SEASON DONE] {season_name} | "
            f"{time.perf_counter() - plot_start:.1f} s"
        )
        gc.collect()

    make_original_mechanism_heatmap(
        phase_table if phase_table is not None else pd.DataFrame(),
        basins,
    )

    total_elapsed = time.perf_counter() - total_start
    print()
    print("=" * 80)
    print("ALL REQUESTED FIGURE 7 OUTPUTS COMPLETED")
    print(f"Output root : {CFG.OUT_ROOT}")
    print(f"Total time  : {total_elapsed / 60.0:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
