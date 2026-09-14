"""
Step 3 — Train correction models and score them honestly against raw HRRR.

Scope: lead times 0-18 h (every hourly HRRR run produces these, so each lead covers every hour of day evenly).

Split:
  test  = days 8-14 of every month   (23% of rows)
  val   = days 20-26 of every month  (23%, used for LightGBM early stopping)
  train = everything else, minus a 1-day embargo on each side of every test/val block
Splitting on the *valid* date keeps all ~19 (run, lead) rows that share a valid hour on the same side.

Models:
  raw HRRR 10 m       the assignment's baseline, no fitting
  linear regression   on the same feature set, standardized, median-imputed
  LightGBM            gradient boosting, native NaN handling, early stopping on val

Features are HRRR-only (all known at forecast issue time) plus hour-of-day and lead time.
No observation-derived feature is used, so this is a pure forecast correction, not a nowcast.

Outputs (under --out):
  interim/test_predictions.parquet
  models/*.pkl
  step3_report.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError) as exc:  # macOS wheels need libomp; fall back rather than fail
    print(f"LightGBM unavailable ({type(exc).__name__}); falling back to sklearn HistGradientBoostingRegressor.\n"
          f"For the LightGBM results, install OpenMP:  brew install libomp")
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    HAS_LGB = False

LEAD_MIN, LEAD_MAX = 0, 18
TEST_DAYS = range(8, 15)
VAL_DAYS = range(20, 27)
EMBARGO_DAYS = {7, 15, 19, 27}
LEAD_BUCKETS = [(0, 6), (7, 12), (13, 18)]
SEED = 7


def add_features(d):
    """All features are derivable from the HRRR forecast itself plus the valid time."""
    f = pd.DataFrame(index=d.index)

    # Wind magnitude and shear
    f["wind10"] = d.hrrr_wind10_ms
    f["wind80"] = d.hrrr_wind80_ms
    f["shear_ratio"] = d.hrrr_wind80_ms / d.hrrr_wind10_ms.clip(lower=0.1)
    f["shear_diff"] = d.hrrr_wind80_ms - d.hrrr_wind10_ms
    f["gust"] = d.hrrr_gust_ms
    f["gust_excess"] = d.hrrr_gust_ms - d.hrrr_wind10_ms

    # Direction as sin/cos so that 359 deg and 1 deg are adjacent, plus veering with height
    dir10 = np.degrees(np.arctan2(-d.hrrr_u10_ms, -d.hrrr_v10_ms)) % 360
    dir80 = np.degrees(np.arctan2(-d.hrrr_u80_ms, -d.hrrr_v80_ms)) % 360
    f["dir10_sin"], f["dir10_cos"] = np.sin(np.radians(dir10)), np.cos(np.radians(dir10))
    f["veer"] = ((dir80 - dir10 + 180) % 360) - 180

    # Stability / boundary-layer state
    f["pbl_height"] = d.hrrr_pbl_height_m
    f["log_pbl"] = np.log(d.hrrr_pbl_height_m.clip(lower=1))
    f["ustar"] = d.hrrr_friction_velocity_ms
    f["ustar_over_wind10"] = d.hrrr_friction_velocity_ms / d.hrrr_wind10_ms.clip(lower=0.1)
    f["dswrf"] = d.hrrr_dswrf_wm2
    f["cloud"] = d.hrrr_total_cloud_pct

    # Thermodynamics
    f["t2m"] = d.hrrr_t2m_k
    f["dewpoint_depression"] = d.hrrr_t2m_k - d.hrrr_d2m_k
    f["pressure_sfc"] = d.hrrr_pressure_sfc_pa
    f["prate"] = d.hrrr_prate_kgm2s

    # Time: diurnal cycle is physical and repeats ~365 times, so it generalizes.
    # Day-of-year is deliberately excluded: with one year of data the model sees each
    # season once and could memorize specific weather regimes instead of learning physics.
    hour = d.hour_local
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["lead_h"] = d.hrrr_forecast_hour
    return f


def assign_split(valid):
    day = valid.dt.day
    s = pd.Series("train", index=valid.index)
    s[day.isin(EMBARGO_DAYS)] = "embargo"
    s[day.isin(VAL_DAYS)] = "val"
    s[day.isin(TEST_DAYS)] = "test"
    return s


def scores(pred, obs):
    e = np.asarray(pred) - np.asarray(obs)
    return {"n": int(len(e)), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)

    step2 = json.loads((out / "step2_report.json").read_text())
    if not all(step2["checks"].values()):
        sys.exit(f"Step 2 checks did not all pass: {step2['checks']}")

    t = pd.read_parquet(out / "interim" / "model_table.parquet")
    d = t[t.usable & t.hrrr_forecast_hour.between(LEAD_MIN, LEAD_MAX)].copy().reset_index(drop=True)

    X = add_features(d)
    y = d.obs_ws_mean
    d["split"] = assign_split(d.valid)

    tr, va, te = (d.split == "train"), (d.split == "val"), (d.split == "test")
    feat = list(X.columns)

    # Linear regression: median impute + standardize (LightGBM needs neither)
    lin = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", LinearRegression())])
    lin.fit(X[tr], y[tr])

    # Gradient boosting. LightGBM when available; sklearn's equivalent otherwise.
    # Both handle NaN natively (no imputation) and use a validation set for early stopping.
    if HAS_LGB:
        params = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=63,
                      min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=4)
        dtrain = lgb.Dataset(X[tr], y[tr], feature_name=feat)
        dval = lgb.Dataset(X[va], y[va], feature_name=feat, reference=dtrain)
        gbm = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        best_iter = int(gbm.best_iteration)
        gbm_predict = lambda M: gbm.predict(M, num_iteration=gbm.best_iteration)
        importance = pd.Series(gbm.feature_importance("gain"), index=feat)
        importance_kind = "lightgbm_gain"
        backend = "lightgbm"
    else:
        # sklearn's built-in early stopping carves a RANDOM validation slice, which would leak
        # across our blocked split. Instead, grow the model incrementally with warm_start and
        # score each increment on the real (day-blocked) validation set.
        gbm = HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=False, warm_start=True, max_iter=0, random_state=SEED)
        step, patience, best_rmse, best_iter, stale = 25, 4, np.inf, 0, 0
        for n in range(step, 1501, step):
            gbm.set_params(max_iter=n)
            gbm.fit(X[tr], y[tr])
            rmse = float(np.sqrt(np.mean((gbm.predict(X[va]) - y[va]) ** 2)))
            if rmse < best_rmse - 1e-5:
                best_rmse, best_iter, stale = rmse, n, 0
            else:
                stale += 1
                if stale >= patience:
                    break
        gbm = HistGradientBoostingRegressor(  # refit from scratch at the best iteration count
            learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=False, max_iter=best_iter, random_state=SEED)
        gbm.fit(X[tr], y[tr])
        best_iter = int(best_iter)
        gbm_predict = lambda M: gbm.predict(M)
        # Permutation importance is expensive; a 15k-row sample is plenty for ranking
        vs = X[va].sample(n=min(15000, int(va.sum())), random_state=SEED)
        pi = permutation_importance(gbm, vs, y[va].loc[vs.index], n_repeats=3,
                                    random_state=SEED, scoring="neg_root_mean_squared_error", n_jobs=1)
        importance = pd.Series(pi.importances_mean, index=feat)
        importance_kind = "permutation_rmse_drop"
        backend = "sklearn_histgradientboosting"

    d["pred_linear"] = lin.predict(X).clip(min=0)
    d["pred_lgbm"] = pd.Series(gbm_predict(X), index=X.index).clip(lower=0)

    # Scoring, test set only, identical rows for every model
    T = d[te]
    results = {name: scores(T[col], T.obs_ws_mean)
               for name, col in [("raw_hrrr", "base_raw10"), ("linear", "pred_linear"), ("lgbm", "pred_lgbm")]}
    for k in ("linear", "lgbm"):
        results[k]["rmse_improvement_pct"] = 100 * (1 - results[k]["rmse"] / results["raw_hrrr"]["rmse"])
        results[k]["mae_improvement_pct"] = 100 * (1 - results[k]["mae"] / results["raw_hrrr"]["mae"])

    by_bucket = {}
    for lo, hi in LEAD_BUCKETS:
        B = T[T.hrrr_forecast_hour.between(lo, hi)]
        r = {name: scores(B[col], B.obs_ws_mean)
             for name, col in [("raw_hrrr", "base_raw10"), ("linear", "pred_linear"), ("lgbm", "pred_lgbm")]}
        r["lgbm_rmse_improvement_pct"] = 100 * (1 - r["lgbm"]["rmse"] / r["raw_hrrr"]["rmse"])
        by_bucket[f"f{lo:02d}-f{hi:02d}"] = r

    # Block bootstrap over test days: is the improvement stable, or one lucky week?
    # Resample whole days (not rows) because rows within a day are highly correlated.
    rng = np.random.default_rng(SEED)
    day_code, _ = pd.factorize(T.valid.dt.floor("D"))  # integer codes avoid tz-aware comparison pitfalls
    groups = [np.flatnonzero(day_code == c) for c in range(day_code.max() + 1)]
    err_raw = (T.base_raw10 - T.obs_ws_mean).to_numpy()
    err_gbm = (T.pred_lgbm - T.obs_ws_mean).to_numpy()
    boot = []
    for _ in range(400):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[p] for p in pick])
        r_raw = np.sqrt(np.mean(err_raw[idx] ** 2))
        r_gbm = np.sqrt(np.mean(err_gbm[idx] ** 2))
        boot.append(100 * (1 - r_gbm / r_raw))
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    imp = importance.sort_values(ascending=False)

    train_days = set(d.loc[tr, "valid"].dt.floor("D"))
    test_days = set(T.valid.dt.floor("D"))
    val_days = set(d.loc[va, "valid"].dt.floor("D"))
    min_gap = min(abs((a - b).days) for a in test_days for b in [max(x for x in train_days if x < a)] ) if train_days else 0

    checks = {
        "E1_splits_disjoint_by_day": len(train_days & test_days) == 0 and len(train_days & val_days) == 0 and len(test_days & val_days) == 0,
        "E2_embargo_gap_at_least_1_day": min_gap >= 2,
        "E3_no_observation_features": not any(c.startswith("obs") for c in feat),
        "E4_all_leads_in_scope": bool(d.hrrr_forecast_hour.between(LEAD_MIN, LEAD_MAX).all()),
        "E5_test_share_reasonable": 0.15 <= te.mean() <= 0.30,
        "E6_predictions_physical": bool((d.pred_lgbm >= 0).all() and d.pred_lgbm.max() < 40),
        "E7_gbm_stopped_before_cap": best_iter < (3000 if HAS_LGB else 1500),
        "E8_lgbm_beats_baseline_on_test": results["lgbm"]["rmse"] < results["raw_hrrr"]["rmse"],
        "E9_bootstrap_ci_finite": bool(np.isfinite(ci).all()),
    }

    report = {
        "scope": {"lead_hours": [LEAD_MIN, LEAD_MAX], "rows": len(d),
                  "train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
                  "embargo_dropped": int((d.split == "embargo").sum())},
        "split_rule": {"test_days_of_month": list(TEST_DAYS), "val_days_of_month": list(VAL_DAYS),
                       "embargo_days": sorted(EMBARGO_DAYS)},
        "n_features": len(feat), "features": feat,
        "gbm_backend": backend,
        "lgbm_best_iteration": best_iter,
        "test_results": results,
        "test_results_by_lead_bucket": by_bucket,
        "lgbm_rmse_improvement_95pct_ci": ci,
        "feature_importance_kind": importance_kind,
        "feature_importance_top15": imp.head(15).round(4).to_dict(),
        "checks": {k: bool(v) for k, v in checks.items()},
    }

    keep = ["valid", "run", "hrrr_forecast_hour", "hour_local", "obs_ws_mean", "obs_ws_std", "obs_calm_frac",
            "base_raw10", "base_log22", "pred_linear", "pred_lgbm", "hrrr_wind80_ms", "hrrr_pbl_height_m",
            "hrrr_u10_ms", "hrrr_v10_ms", "split"]
    d[keep].to_parquet(out / "interim" / "predictions.parquet", index=False)
    with open(out / "models" / "lgbm.pkl", "wb") as fh:
        pickle.dump(gbm, fh)
    with open(out / "models" / "linear.pkl", "wb") as fh:
        pickle.dump(lin, fh)
    (out / "step3_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report["checks"], indent=2))
    print(f"\nTest set ({results['raw_hrrr']['n']:,} rows), leads {LEAD_MIN}-{LEAD_MAX} h")
    print(f"  raw HRRR   RMSE {results['raw_hrrr']['rmse']:.3f}  MAE {results['raw_hrrr']['mae']:.3f}  bias {results['raw_hrrr']['bias']:+.3f}")
    print(f"  linear     RMSE {results['linear']['rmse']:.3f}  MAE {results['linear']['mae']:.3f}  bias {results['linear']['bias']:+.3f}   ({results['linear']['rmse_improvement_pct']:.1f}% RMSE)")
    print(f"  LightGBM   RMSE {results['lgbm']['rmse']:.3f}  MAE {results['lgbm']['mae']:.3f}  bias {results['lgbm']['bias']:+.3f}   ({results['lgbm']['rmse_improvement_pct']:.1f}% RMSE)")
    print(f"  95% CI on LightGBM improvement: {ci[0]:.1f}% to {ci[1]:.1f}%")

    failed = [k for k, v in checks.items() if not v]
    if failed:
        print("VALIDATION FAILED:", failed)
        sys.exit(1)
    print("All step 3 checks passed.")


if __name__ == "__main__":
    main()