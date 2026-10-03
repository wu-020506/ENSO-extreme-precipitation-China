# Methods and script map

Only the central calculation workflows are included; old plotting experiments, layout-only variants, notebooks, and debugging scripts were excluded.

## RR handling by workflow

- **02 Figure 2 series:** empirical `P=c/n`; no continuity correction. Neutral zero-event probability → RR undefined. FAR only for RR>1.
- **03 month × threshold:** empirical RR normally; Haldane–Anscombe stabilization only when one compared group has zero events.
- **04a monthly spatial footprint:** when `USE_HALDANE_CORRECTION=True`, both phase and Neutral probabilities use `(c+0.5)/(n+1)` for all valid cells. This keeps monthly gridded RR finite. The public variable name was clarified; the numerical calculation is unchanged.
- **04b seasonal footprint:** event counts and valid sample sizes are pooled across the three constituent months and seasonal RR is recalculated. Monthly RR is not averaged. Zero-count cases use the script's zero-case correction.
- **05 recurrence:** empirical RR is recalculated from stored counts. Phase-zero/Neutral-positive gives RR=0; Neutral-zero is undefined and excluded from effect-colour recurrence.
- **06 basin translation:** empirical RR is used for descriptive effect magnitude and fractional basin-area summaries.

## Public-release edits

The scientific formulas were preserved. Edits are limited to portable repository-relative paths, clearer naming of the monthly spatial Haldane flag, the latest compact Figure 2 spacing, documentation, dependency files, and git-ignore rules.
