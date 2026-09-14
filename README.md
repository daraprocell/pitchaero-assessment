# HRRR wind bias correction — WireWarrior sensor, southern Idaho

A machine-learning correction for HRRR's 10 m wind forecasts at a single sensor site
(42.8783, −115.0089; 22.1 m AGL), trained on one year of paired forecasts and observations.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # macOS + LightGBM also needs: brew install libomp
mkdir -p data && cp /path/to/your/sensor_wind.csv.gz data/

python src/01_test_data.py --input data/sensor_wind.csv.gz
python src/02_build_table.py
python src/03_train.py 
python src/04_evaluate.py
```

Each step writes a JSON report and refuses to run if the previous step's validation checks 
failed. If LightGBM cannot load, step 3 falls back to scikit-learn's 
`HistGradientBoostingRegressor` (12.9% instead of 13.4%).

---

## How I framed the problem

**Target.** Mean observed wind speed over the 60 minutes centred on each HRRR valid time,
requiring at least 45 valid minutes. HRRR's wind is an hourly snapshot representing a 3 km
grid cell; scoring it against a single 1-minute sensor reading would mostly measure
turbulence that no forecast can predict. `figures/02_baseline.png` panel a shows that raw 
HRRR RMSE falls monotonically from 1.77 m/s (1-minute target) to 1.41 m/s (120-minute target)
without any change to the forecast. 

**Rows.** One row per (run, lead) forecast, not per valid hour. Each valid hour is covered by
up to 24 forecasts. Scope is **leads 0–18 h** (165221 usable rows): every hourly run produces
these, so each lead covers every hour of the day evenly. Leads 19–48 h exist only from the
00/06/12/18Z runs and would weight some hours unevenly.

**Features (22).** HRRR only, all known at forecast issue time: wind at 10/80 m and their
ratio and difference, gust and gust excess, direction as sin/cos plus veer with height, PBL
height, friction velocity, radiation, cloud, temperature, dewpoint depression, surface
pressure, precipitation rate, hour-of-day as sin/cos, and lead time. No observation derived
features, so this is a forecast correction rather than nowcasting. Day-of-year was
deliberately excluded: with one year of data the model would see each date once and could
memorise specific weather events. Roughness length was dropped (only two distinct values).

**Split**
Split by UTC valid date: days 8–14 of every month are test, days 20–26 are validation for early stopping, and the remaining days are training, excluding a one-day buffer on either side of each test/validation block. All forecasts sharing a valid hour stay in the same split.
This evaluates held-out periods within one year at one site, with training dates both before and after test periods. Performance on future years or other sites remains untested. The day-block bootstrap accounts for within-day dependence but may understate uncertainty from multiday weather systems

---

## Results — held-out test set, 38,228 forecasts, leads 0–18 h

**Headline: 13.4% RMSE improvement over raw HRRR on a held-out test set (95% CI: 9.9–16.8%).**

| Model | RMSE (m/s) | MAE (m/s) | Bias (m/s) | RMSE improvement |
|---|---|---|---|---|
| **Raw HRRR 10 m (baseline)** | 1.504 | 1.147 | +0.205 | — |
| Constant shift + scale (2 params) | 1.451 | — | — | 3.5% |
| Linear regression (22 features) | 1.381 | 1.032 | +0.076 | 8.2% |
| **LightGBM (22 features)** | **1.303** | **0.975** | **+0.040** | **13.4%** |

95% CI on the LightGBM improvement, from a block bootstrap resampling whole test days:
**9.9% to 16.8%**. MAE improves 15.0%.

**By lead bucket** — the gain is stable, so this is not a short-range trick:

| Lead | Raw HRRR RMSE | Corrected RMSE | Improvement |
|---|---|---|---|
| 0–6 h | 1.374 | 1.193 | 13.1% |
| 7–12 h | 1.512 | 1.308 | 13.5% |
| 13–18 h | 1.636 | 1.415 | 13.5% |

The ladder is the interesting part: a constant shift-and-scale
(`corrected = 0.493 + 0.848 × HRRR₁₀`) buys 3.5%, letting that correction vary linearly with
conditions buys 8.2%, and letting it vary non-linearly buys 13.4%. Most of the gain comes
from removing conditional bias, reducing errors using information already present in HRRR.

---

## Visual evidence

**`outputs/figures/03_where_it_helps.png`**
Improvement reaches about 20% in the early morning and falls to 5–7% around
midday. The bias panel suggests one explanation: HRRR runs +0.4 to +0.5 m/s
too windy in the early morning and is nearly unbiased around midday.
This is consistent with the diurnal pattern in `02_baseline.png`.
The single highest improvement is at 21:00 MST (22%), but these hourly
results pool correlated forecasts, so I would not overinterpret one hour.
All diurnal plots use fixed MST (UTC−7), without daylight-saving adjustments.

<img width="1820" height="1170" alt="03_where_it_helps" src="https://github.com/user-attachments/assets/580ed31b-5ce1-4a90-9fd1-4e5a496d2edf" />


**`outputs/figures/04_error_structure.png`**
The error distribution recenters near zero and narrows (σ 1.49 to 1.30 m/s).
However, the overall improvement is not uniform: RMSE worsens by 6.4% for
observed winds of 6–8 m/s and 20.0% for 8–10 m/s. The bottom-right panel
shows bias by observed speed; both forecasts underpredict the strongest
winds. These observation-conditioned bins describe retrospective errors,
not conditions identifiable in advance.

<img width="1820" height="1170" alt="04_error_structure" src="https://github.com/user-attachments/assets/2e4ddcd3-a301-4a03-993a-58f27764d721" />


The corrected forecasts also have less variability: predicted SD is
2.13 m/s, versus 2.48 observed and 2.35 for raw HRRR. This is consistent
with squared-error training favoring predictions toward the conditional
mean, and motivates checking threshold performance before operational use.
Overall, **58.7%** of individual forecasts have smaller absolute errors
after correction.

**`outputs/figures/05_case_studies.png`**
The selection rule uses observed daily mean wind: take the windiest
(14 May 2025, 9.68 m/s) and calmest (13 Feb 2026, 1.81 m/s) test days.
This selects contrasting conditions independently of which days show the
largest model improvement.

At each valid hour, the plotted forecasts average overlapping runs with
leads of 6–12 h. The case-study RMSE values therefore describe those
averaged forecasts, rather than individual forecasts as in the headline
results. The windy day shows almost no improvement (1.09 to 1.08 m/s).
The calm day improves modestly (1.03 to 0.98 m/s), but both forecasts miss
a 3.2 m/s spike at 12 UTC. These examples illustrate that correction
does not reliably recover every local fluctuation or missed event.

<img width="1690" height="1040" alt="05_case_studies" src="https://github.com/user-attachments/assets/31f18018-9607-4c92-a355-59841391dc9b" />


**`outputs/figures/01_qc.png`** documents the data-quality decisions below;

<img width="1820" height="1170" alt="01_qc" src="https://github.com/user-attachments/assets/f963d9ee-a741-4392-9417-ac8900355672" />

**`outputs/figures/02_baseline.png`** documents the target-window comparison
and the baseline's structure.

<img width="1820" height="1170" alt="02_baseline" src="https://github.com/user-attachments/assets/91d8e336-047e-422a-996c-3fd8e8d641cc" />


---

## Data issues found and how they were handled

| Issue | Finding | Handling |
|---|---|---|
| **Sensor fault signature** | 4,877 rows read exactly 0 m/s at exactly 225°, usually with pressure = 0. HRRR's median wind at those times is 3.4 m/s vs 1.8 m/s during other 0 m/s reports. These are likely not calms. | Wind set to missing. The 11,666 other zero-wind reports were retained as calms: calm hours matter most for conductor cooling. |
| **Uncalibrated thermo channels** | Air temp reaching 48.5 °C (likely solar heating); 45 rows with temp and RH both exactly 0; 6,561 rows with pressure 0 or < 80 kPa. | Flagged and excluded from QC use. Not used as features (they are observations, unavailable at forecast time). A bad pressure reading alone doesn't invalidate the wind. |
| **Clock alignment** | Lagged cross-correlation peaks at −5 min (wind) and −15 min (temp), essentially zero; temp r = 0.98 at zero lag. | No timezone or offset correction needed. |
| **Sensor gaps** | 524,770 reports over 525,600 possible minutes; 498 minute buckets contain two reports. | Averaged onto a regular minute grid; hours with < 45 valid minutes get no target (7 hours). |
| **HRRR missing fields** | 2,343 `wind10` values missing in March but recoverable exactly from u/v; 80 m winds sparse in late March and absent on the final day; 5 runs missing on 21 Feb. | `wind10` back-filled from components (verified identical to 0.01 m/s where both exist). 80 m left missing. LightGBM handles NaN natively. Other runs cover the missing hours. |
| **Roughness length** | Only 2 distinct values all year. | Dropped as a feature. |
| **Height mismatch** | Sensor at 22.1 m, HRRR at 10/80 m. A log-profile interpolation to 22.1 m made things worse (RMSE 1.56 → 1.89) because HRRR's 10 m wind already *exceeds* the observed 22 m wind. Damage is worst overnight (RMSE 2.11), when HRRR's 80/10 ratio is 1.59 and the surface has decoupled, so the smooth-profile assumption fails. | Raw 10 m HRRR kept as the baseline; the site-specific offset is left for the model to learn. The 80/10 ratio is retained as a stability feature rather than as an assumed physical relationship. |
| **Grid representativeness** | HRRR surface pressure is ~870 Pa higher than the (height-adjusted) sensor pressure, implying the model grid cell sits 80–100 m below the site. | Noted as a caveat, not corrected. The sensor's barometer is uncalibrated, so a constant instrument offset would look identical. Confirming this needs HRRR's terrain-height field. HRRR's 10/80 m levels are relative to its own terrain, so this does not shift the comparison. |

**Validation gates.** Each step runs explicit checks and exits non-zero on failure (9 in step 1,
6 in step 2, 9 in step 3) — e.g. rows partition cleanly, no duplicate keys, lead = valid − run,
`wind10` equals `hypot(u,v)`, target mean matches the raw minute mean, errors grow with lead
time, train/val/test days are disjoint, no observation-derived features, predictions physical.

---

## Looking ahead

**Priority 1 — probabilistic output, not a better point forecast.** The variance shrinkage
above is the most actionable finding. Quantile regression (or conformal intervals) would give
calibrated uncertainty, and for dynamic line rating the operationally useful quantity is a
conservative low wind quantile, not the conditional mean. I would expect little RMSE change
but a substantially more useful product. I would pair it with threshold-based skill scores
(hit rate / false alarm ratio for hours below ~2 m/s) since RMSE is dominated by the 80% of
hours below 6 m/s.

**Priority 2 — more HRRR, from the AWS Open Data archive (via Herbie).** Two things I would
pull: (a) neighbouring grid points, so the model can learn which cell actually represents the
site and can see local gradients; (b) HRRR's terrain-height field, to settle the elevation
question above.

**Priority 3 — features with the clearest meteorological rationale.** In rough order of
expected value: a proper stability parameter (bulk Richardson number or Obukhov length, from
the 2 m / 10 m / 80 m fields) to replace my shear-ratio proxy; low-level lapse rate and
inversion strength, which govern nocturnal decoupling; 700 hPa wind and mean sea-level 
pressure gradient as synoptic-regime indicators; and seasonal surface roughness (this is 
irrigated Snake River Plain cropland, so roughness changes materially through the growing 
season, which a static 2-value field cannot represent). Nearby ASOS/mesonet observations 
would also let me separate "HRRR got the weather wrong" from "HRRR got the site wrong."

**Priority 4 — models and analyses.** More training data is the real constraint: one year means
each season is seen once, which is why I excluded day-of-year. With multi-year data I would
revisit seasonal features and use proper rolling-origin CV. Beyond that: (a) an analog ensemble,
which is a strong and interpretable baseline for this exact problem and would tell me how much
of the 13% is really "look up similar past forecasts"; (b) a short-lead nowcasting variant using
recent observations — excluded here deliberately, but it should dominate at 0–3 h and is worth
quantifying separately; (c) a sequence model (temporal CNN or LSTM) over the forecast trajectory
rather than independent hours, to capture ramp timing errors — the 14 May case shows the ramp
shape is right but the timing and amplitude are off, which per-hour correction cannot fix;
(d) multi-site training if other WireWarrior units exist, with site embeddings, which would
show how much of this correction transfers and how much is genuinely local.

**Also worth doing quickly:** permutation importance instead of LightGBM gain (gain splits
credit between correlated features like `wind10`/`wind80`), and a gust-specific target, since
the current model predicts hourly-mean wind only.

---

## Repository layout

```
src/01_test_data.py      load data, QC, fault detection, clock alignment  → interim/*.parquet, qc_report.json
src/02_build_table.py    target construction, baselines, join             → interim/model_table.parquet
src/03_train.py          features, blocked split, linear + LightGBM       → interim/predictions.parquet, models/
src/04_evaluate.py       evaluation figures and metrics                   → figures/, step4_report.json
outputs/                 all generated artefacts (reports, figures, models)
```

Random seed fixed at 7.
Python 3.8/3.12 and LightGBM/sklearn backends.
