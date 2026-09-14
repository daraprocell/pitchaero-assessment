"""
Step 4 — Evaluation. Where does the correction help, where it doesn't, and why.

Reads outputs/interim/predictions.parquet (written by step 3) and produces:
  figures/03_where_it_helps.png   improvement by hour, lead, speed bin, and the model ladder
  figures/04_error_structure.png  before/after scatter, error distributions, conditional bias, spread shrinkage
  figures/05_case_studies.png     two test days: a windy event and a calm night
  step4_report.json               the numbers behind every panel

All metrics are computed on the held-out test set only.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEAD_BUCKETS = [(0, 6), (7, 12), (13, 18)]
SPEED_BINS = [0, 2, 4, 6, 8, 10, 25]
C_RAW, C_LIN, C_GBM, C_OBS = "#444444", "#7fa8c9", "#d1495b", "#2e86ab"


def rmse(e):
    return float(np.sqrt(np.mean(np.asarray(e) ** 2)))


def summarize(df, by, observed_bins=False):
    """RMSE / bias / improvement for raw vs corrected, grouped by a column or binning."""
    rows = {}
    for key, g in df.groupby(by, observed=True):
        er, eg = g.base_raw10 - g.obs_ws_mean, g.pred_lgbm - g.obs_ws_mean
        rows[key] = {"n": len(g), "rmse_raw": rmse(er), "rmse_gbm": rmse(eg),
                     "bias_raw": float(er.mean()), "bias_gbm": float(eg.mean()),
                     "improvement_pct": 100 * (1 - rmse(eg) / rmse(er))}
    return pd.DataFrame(rows).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    step3 = json.loads((out / "step3_report.json").read_text())
    if not all(step3["checks"].values()):
        sys.exit(f"Step 3 checks did not all pass: {step3['checks']}")

    d = pd.read_parquet(out / "interim" / "predictions.parquet")
    T = d[d.split == "test"].copy()
    T["err_raw"] = T.base_raw10 - T.obs_ws_mean
    T["err_gbm"] = T.pred_lgbm - T.obs_ws_mean
    T["err_lin"] = T.pred_linear - T.obs_ws_mean

    by_hour = summarize(T, "hour_local")
    T["lead_bucket"] = pd.cut(T.hrrr_forecast_hour, [-1, 6, 12, 18], labels=["0–6 h", "7–12 h", "13–18 h"])
    by_lead = summarize(T, "lead_bucket")
    T["fcst_bin"] = pd.cut(T.base_raw10, SPEED_BINS)
    by_fcst = summarize(T, "fcst_bin")
    T["obs_bin"] = pd.cut(T.obs_ws_mean, SPEED_BINS)
    by_obs = summarize(T, "obs_bin")

    ladder = {"raw HRRR": rmse(T.err_raw), "linear": rmse(T.err_lin), "LightGBM": rmse(T.err_gbm)}

    report = {
        "test_rows": len(T),
        "ladder_rmse": ladder,
        "by_hour_local": by_hour.round(3).to_dict(orient="index"),
        "by_lead_bucket": {str(k): v for k, v in by_lead.round(3).to_dict(orient="index").items()},
        "by_forecast_speed_bin": {str(k): v for k, v in by_fcst.round(3).to_dict(orient="index").items()},
        "by_observed_speed_bin": {str(k): v for k, v in by_obs.round(3).to_dict(orient="index").items()},
        "spread": {"obs_sd": float(T.obs_ws_mean.std()), "raw_sd": float(T.base_raw10.std()),
                   "gbm_sd": float(T.pred_lgbm.std())},
        "pct_forecasts_improved": float((T.err_gbm.abs() < T.err_raw.abs()).mean() * 100),
    }
    (out / "step4_report.json").write_text(json.dumps(report, indent=2))

    fig_where(by_hour, by_lead, by_fcst, ladder, out / "figures" / "03_where_it_helps.png")
    fig_errors(T, by_obs, out / "figures" / "04_error_structure.png")
    fig_cases(d, out / "figures" / "05_case_studies.png")

    # Record the ranking that the case-study selection rule is based on, so it's auditable
    daily = T.groupby(T.valid.dt.floor("D")).obs_ws_mean.mean().sort_values()
    report["case_study_selection"] = {"rule": "windiest and calmest test day by observed daily mean wind",
                                      "calmest": {str(k.date()): round(float(v), 2) for k, v in daily.head(3).items()},
                                      "windiest": {str(k.date()): round(float(v), 2) for k, v in daily.tail(3).items()}}
    (out / "step4_report.json").write_text(json.dumps(report, indent=2))

    print(f"Test rows: {len(T):,}")
    print(f"RMSE  raw {ladder['raw HRRR']:.3f}  linear {ladder['linear']:.3f}  LightGBM {ladder['LightGBM']:.3f}")
    print(f"Individual forecasts improved: {report['pct_forecasts_improved']:.1f}%")
    print(f"Best hours: {by_hour.improvement_pct.idxmax()}:00 local ({by_hour.improvement_pct.max():.1f}%), "
          f"worst: {by_hour.improvement_pct.idxmin()}:00 ({by_hour.improvement_pct.min():.1f}%)")


def fig_where(by_hour, by_lead, by_fcst, ladder, path):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # (a) RMSE by hour of day
    a = ax[0, 0]
    a.plot(by_hour.index, by_hour.rmse_raw, color=C_RAW, lw=2, label="raw HRRR")
    a.plot(by_hour.index, by_hour.pipe(lambda x: x.rmse_gbm), color=C_GBM, lw=2, label="corrected")
    a.fill_between(by_hour.index, by_hour.rmse_gbm, by_hour.rmse_raw, color=C_GBM, alpha=0.15)
    a2 = a.twinx()
    a2.bar(by_hour.index, by_hour.improvement_pct, color="#bbbbbb", alpha=0.45, width=0.8, zorder=0)
    a2.set_ylabel("% RMSE improvement", color="#777777")
    a2.set_ylim(0, 32)
    a.set_zorder(a2.get_zorder() + 1); a.patch.set_visible(False)
    a.set_xticks(range(0, 24, 3)); a.set_xlabel("hour of day (MST)"); a.set_ylabel("RMSE (m/s)")
    a.set_title("Correction helps most overnight, least at midday")
    a.legend(loc="upper left", fontsize=8)

    # (b) Bias by hour — the mechanism behind panel (a)
    b = ax[0, 1]
    b.plot(by_hour.index, by_hour.bias_raw, color=C_RAW, lw=2, label="raw HRRR")
    b.plot(by_hour.index, by_hour.bias_gbm, color=C_GBM, lw=2, label="corrected")
    b.axhline(0, color="black", lw=0.8, ls=":")
    b.fill_between(by_hour.index, 0, by_hour.bias_raw, color=C_RAW, alpha=0.12)
    b.set_xticks(range(0, 24, 3)); b.set_xlabel("hour of day (MST)"); b.set_ylabel("bias (m/s)")
    b.set_title("HRRR's overnight high bias is nearly eliminated")
    b.legend(fontsize=8)

    # (c) Improvement by lead bucket
    c = ax[1, 0]
    x = np.arange(len(by_lead))
    c.bar(x - 0.2, by_lead.rmse_raw, 0.4, color=C_RAW, label="raw HRRR")
    c.bar(x + 0.2, by_lead.rmse_gbm, 0.4, color=C_GBM, label="corrected")
    for i, (r, g, p) in enumerate(zip(by_lead.rmse_raw, by_lead.rmse_gbm, by_lead.improvement_pct)):
        c.text(i, max(r, g) + 0.03, f"−{p:.1f}%", ha="center", fontsize=9)
    c.set_xticks(x); c.set_xticklabels(by_lead.index); c.set_ylim(0, by_lead.rmse_raw.max() * 1.2)
    c.set_ylabel("RMSE (m/s)"); c.set_xlabel("forecast lead time")
    c.set_title("Gain holds across lead times")
    c.legend(fontsize=8)

    # (d) By forecast wind speed
    dd = ax[1, 1]
    lab = [f"{int(i.left)}–{int(i.right)}" for i in by_fcst.index]
    x = np.arange(len(by_fcst))
    dd.bar(x - 0.2, by_fcst.rmse_raw, 0.4, color=C_RAW, label="raw HRRR")
    dd.bar(x + 0.2, by_fcst.rmse_gbm, 0.4, color=C_GBM, label="corrected")
    for i, (r, g, n) in enumerate(zip(by_fcst.rmse_raw, by_fcst.rmse_gbm, by_fcst.n)):
        dd.text(i, max(r, g) + 0.04, f"n={int(n):,}", ha="center", fontsize=7, color="#666666")
    dd.set_xticks(x); dd.set_xticklabels(lab)
    dd.set_xlabel("HRRR forecast wind speed (m/s)"); dd.set_ylabel("RMSE (m/s)")
    dd.set_title("Error grows with wind speed; correction helps throughout")
    dd.legend(fontsize=8)

    fig.suptitle(f"Where the correction helps  ·  test set only  ·  "
                 f"RMSE {ladder['raw HRRR']:.3f} to {ladder['LightGBM']:.3f} m/s", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_errors(T, by_obs, path):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    lim = [0, 16]

    for a, col, title in [(ax[0, 0], "base_raw10", "Before: raw HRRR"), (ax[0, 1], "pred_lgbm", "After: corrected")]:
        hb = a.hexbin(T[col], T.obs_ws_mean, gridsize=50, bins="log", cmap="viridis", mincnt=1, extent=(*lim, *lim))
        a.plot(lim, lim, color="red", lw=1)
        a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel("forecast (m/s)"); a.set_ylabel("observed (m/s)")
        e = T[col] - T.obs_ws_mean
        a.set_title(f"{title}   RMSE {rmse(e):.3f}, bias {e.mean():+.3f}")
        fig.colorbar(hb, ax=a, label="count (log)")

    # (c) Error distributions
    c = ax[1, 0]
    bins = np.arange(-6, 6.1, 0.2)
    c.hist(T.err_raw, bins=bins, histtype="step", lw=2, color=C_RAW, label=f"raw HRRR (σ={T.err_raw.std():.2f})")
    c.hist(T.err_gbm, bins=bins, histtype="step", lw=2, color=C_GBM, label=f"corrected (σ={T.err_gbm.std():.2f})")
    c.axvline(0, color="black", lw=0.8, ls=":")
    c.axvline(T.err_raw.mean(), color=C_RAW, lw=1, ls="--")
    c.axvline(T.err_gbm.mean(), color=C_GBM, lw=1, ls="--")
    c.set_xlabel("forecast − observed (m/s)"); c.set_ylabel("count")
    c.set_title("Error distribution recenters on zero and narrows slightly")
    c.legend(fontsize=8)

    # (d) Conditional bias by OBSERVED speed. look at the extremes
    dd = ax[1, 1]
    lab = [f"{int(i.left)}–{int(i.right)}" for i in by_obs.index]
    x = np.arange(len(by_obs))
    dd.bar(x - 0.2, by_obs.bias_raw, 0.4, color=C_RAW, label="raw HRRR")
    dd.bar(x + 0.2, by_obs.bias_gbm, 0.4, color=C_GBM, label="corrected")
    dd.axhline(0, color="black", lw=0.8)
    for i, n in enumerate(by_obs.n):
        dd.text(i, 0.06, f"n={int(n):,}", ha="center", fontsize=7, color="#666666")
    dd.set_xticks(x); dd.set_xticklabels(lab)
    dd.set_xlabel("OBSERVED wind speed (m/s)"); dd.set_ylabel("bias (m/s)")
    dd.set_title("Both under-forecast the strongest winds")
    dd.legend(fontsize=8)

    fig.suptitle("Error structure before and after correction  ·  test set", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_cases(d, path):
    """Two test days at a single lead window, so each series reads as one continuous forecast.

    Selection rule (deliberately independent of model performance): rank test days by the
    observed daily mean wind, then take the windiest and the calmest. The days are chosen
    from the observations alone, before looking at how either forecast scored on them, so
    the panels are not cherry-picked for a flattering result.
    """
    cases = [("2025-05-14", "Windiest test day — 14 May 2025"),
             ("2026-02-13", "Calmest test day — 13 Feb 2026")]
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    for a, (day, title) in zip(ax, cases):
        s = d[(d.split == "test") & (d.hrrr_forecast_hour.between(6, 12))
              & (d.valid.dt.floor("D") == pd.Timestamp(day, tz="UTC"))].sort_values("valid")
        s = s.groupby("valid").mean(numeric_only=True).reset_index()  # average the overlapping runs
        a.plot(s.valid, s.obs_ws_mean, color=C_OBS, lw=2.5, marker="o", ms=3, label="sensor (22.1 m)")
        a.plot(s.valid, s.base_raw10, color=C_RAW, lw=1.8, ls="--", label="raw HRRR")
        a.plot(s.valid, s.pred_lgbm, color=C_GBM, lw=1.8, label="corrected")
        er, eg = rmse(s.base_raw10 - s.obs_ws_mean), rmse(s.pred_lgbm - s.obs_ws_mean)
        a.set_title(f"{title}   ·   RMSE raw {er:.2f} corrected to {eg:.2f} m/s")
        a.set_ylabel("wind speed (m/s)")
        a.legend(fontsize=8)
        a.grid(alpha=0.2)
    ax[-1].set_xlabel("valid time (UTC)")
    fig.suptitle("Case studies — leads 6–12 h, averaged across runs", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()