# ENSO and extreme-precipitation risk across China — core analysis code

This folder contains a cleaned public-release version of the core Python workflows used for ENSO classification, six-basin extreme-precipitation risk, spatial RR, recurrence, basin-scale risk translation, and ERA5 mechanism diagnostics.

## Included scripts

| Script | Role |
|---|---|
| `01_enso_classification.py` | ERSST v6 Niño3.4 extraction, quadratic detrending, monthly ENSO classification, sensitivity definitions |
| `02_basin_distribution_rr_far.py` | MEP95/Rx5day basin aggregation, q90/q95 events, CDFs, empirical RR/FAR |
| `03_month_threshold_log2rr.py` | Basin × month × threshold log2(RR) and permutation support |
| `04a_monthly_spatial_rr.py` | Monthly grid-cell spatial RR/Wilson products |
| `04b_seasonal_spatial_rr.py` | Seasonal pooling and seasonal Figure 4 / monthly supporting plots |
| `05_recurrence_hotspots.py` | Recurrent large-effect regions and Wilson/FDR diagnostics |
| `06_basin_risk_translation.py` | Fractional basin-area translation and recurrence summaries |
| `07_era5_mechanisms.py` | ERA5 circulation, moisture and ascent diagnostics |

## Core RR/FAR convention

For Figure 2 and the descriptive recurrence / basin-translation products, event probability is empirical: `P = c/n`, and `RR = P_phase/P_neutral`. If the ENSO-phase event count is zero and Neutral probability is positive, `RR=0`; if Neutral probability is zero, empirical RR is undefined (`NaN`). FAR is interpreted only for `RR>1` as `FAR = 1 - 1/RR`.

Sparse monthly analyses use the specific stabilization rules documented in `docs/METHODS_AND_SCRIPT_MAP.md`.

## Folder layout

```text
.
├── scripts/
├── data/                 # local inputs; not tracked
├── outputs/              # generated products; not tracked
├── docs/
├── requirements.txt
├── environment.yml
└── .gitignore
```

## Run order

```bash
python scripts/01_enso_classification.py
python scripts/02_basin_distribution_rr_far.py
python scripts/03_month_threshold_log2rr.py
python scripts/04a_monthly_spatial_rr.py
python scripts/04b_seasonal_spatial_rr.py
python scripts/05_recurrence_hotspots.py
python scripts/06_basin_risk_translation.py
python scripts/07_era5_mechanisms.py
```

## Data

Large climate datasets and vector data are intentionally excluded. See `data/README.md`.

## Public-release checklist

Before publishing, verify redistribution permissions for any auxiliary vector data, choose a software licence, and add the final manuscript citation/DOI after publication.
