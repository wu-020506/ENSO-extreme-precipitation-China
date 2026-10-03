# Local data layout

```text
data/
├── ersst_v6/
├── precipitation/
│   └── CN05.1_Pre_1961_2023_daily_025x025.nc
├── basins/
│   └── six_selected_official_basins.shp (+ sidecar files)
├── boundaries/
│   ├── global_countries.shp
│   ├── province_boundaries.shp
│   └── ten_dash_line.shp
├── rivers/
│   └── all_river_levels_inside_six_basins.shp
└── era5/
    ├── 01_UV_500_700hPa/
    ├── 02_SpecificHumidity_700hPa/
    ├── 03_RelativeHumidity_700hPa/
    ├── 04_VerticalVelocity_700hPa/
    ├── 05_SurfacePressure/
    └── 07_Evaporation/
```

The repository does not redistribute CN05.1, ERA5, administrative boundaries, basin polygons, or large intermediate NetCDF files. Obtain them from their legitimate data providers and place them locally.
