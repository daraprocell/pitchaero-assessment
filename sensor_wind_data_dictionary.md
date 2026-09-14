# `sensor_wind.csv` - Data Dictionary

One CSV, two kinds of rows. The `record_type` column says which kind each row is:

| `record_type` | What the row is | Populated columns |
|---|---|---|
| `sensor_obs` | One report from the sensor | `timestamp_utc` + `obs_*` |
| `hrrr_forecast` | One HRRR forecast value-set for one valid time, from one model run | `timestamp_utc` + `hrrr_*` |

Columns not applicable to a row's record type are empty. Missing values within an applicable column are also empty.

All timestamps are UTC, ISO 8601.

## Shared column

| Column | Description |
|---|---|
| `timestamp_utc` | For `sensor_obs`: the time the sensor reported. For `hrrr_forecast`: the forecast **valid time** (the future moment the forecast is about). |

## Sensor observation columns (`sensor_obs` rows)

| Column | Unit | Description |
|---|---|---|
| `obs_wind_speed_ms` | m/s | Wind speed measured by the sensor's anemometer. |
| `obs_wind_direction_deg` | degrees | Direction the wind is blowing **from**, clockwise from true north (meteorological convention). |
| `obs_air_temp_c` | °C | Air temperature as reported by the device (uncalibrated). |
| `obs_rh_pct` | % | Relative humidity as reported by the device (uncalibrated). |
| `obs_pressure_pa` | Pa | Station pressure as reported by the device (uncalibrated, not sea-level reduced). |

## HRRR forecast columns (`hrrr_forecast` rows)

HRRR is NOAA's High-Resolution Rapid Refresh model (~3 km grid). Values are extracted at the model grid point nearest the sensor. A new model run is issued every hour; each run produces forecasts for a range of lead times. Runs at 00, 06, 12 and 18 UTC forecast further ahead than the others. This means a given valid time is usually covered by **multiple rows** - one per (run, lead) combination that lands on it.

| Column | Unit | Description |
|---|---|---|
| `hrrr_run_time_utc` | - | When the model run was initialized (issued). |
| `hrrr_forecast_hour` | hours | Lead time: `timestamp_utc` − `hrrr_run_time_utc`. |
| `hrrr_u10_ms` | m/s | U (eastward) wind component at 10 m above ground. |
| `hrrr_v10_ms` | m/s | V (northward) wind component at 10 m above ground. |
| `hrrr_wind10_ms` | m/s | Wind speed at 10 m above ground. |
| `hrrr_gust_ms` | m/s | Surface wind gust. |
| `hrrr_u80_ms` | m/s | U wind component at 80 m above ground. |
| `hrrr_v80_ms` | m/s | V wind component at 80 m above ground. |
| `hrrr_t2m_k` | K | Temperature at 2 m above ground. |
| `hrrr_d2m_k` | K | Dewpoint temperature at 2 m above ground. |
| `hrrr_pressure_sfc_pa` | Pa | Surface pressure at the model grid cell (not sea-level reduced). |
| `hrrr_pbl_height_m` | m | Planetary boundary layer height. |
| `hrrr_friction_velocity_ms` | m/s | Friction velocity (u*). |
| `hrrr_roughness_length_m` | m | Surface roughness length. |
| `hrrr_dswrf_wm2` | W/m² | Downward shortwave radiation flux at the surface. |
| `hrrr_prate_kgm2s` | kg/m²/s | Precipitation rate. |
| `hrrr_total_cloud_pct` | % | Total cloud cover. |

## Provenance

- Sensor: WireWarrior unit located at 42.8783, −115.0089 (southern Idaho), mounted 22.1 m above ground. Site ground elevation ≈ 1009 m MSL.
- Period of record: 2025-04-01 00:00 UTC → 2026-04-01 00:00 UTC.
- The file is real production data. Beyond assembling the two sources into one file, only minor cleaning was done.