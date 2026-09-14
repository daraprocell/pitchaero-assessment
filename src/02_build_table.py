"""
Step 2 — Build the modeling table: one row per HRRR (run, lead) forecast, paired with what the sensor measured.

Target: mean of QC'd 1-minute sensor wind speed over the 60 minutes centered on the HRRR valid time
        [valid - 30 min, valid + 30 min). Requires >= 45 valid minutes, otherwise the target is NaN.

Baselines (no learning involved):
  base_raw10  HRRR 10 m wind speed as-is (the assignment's "raw HRRR" baseline)
  base_log22  HRRR wind interpolated to the sensor height (22.1 m) assuming a log wind profile
              between HRRR's 10 m and 80 m winds

Outputs (under --out):
  interim/model_table.parquet
  step2_report.json
  figures/02_baseline.png
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

PERIOD_START = pd.Timestamp("2025-04-01 00:00", tz="UTC")
PERIOD_END = pd.Timestamp("2026-04-01 00:00", tz="UTC")
SENSOR_HEIGHT_M = 22.1
WINDOW = (-30, 30)          # minutes relative to valid time, [start, end)
MIN_VALID_MINUTES = 45
LOCAL_OFFSET_H = -7         # Mountain Standard Time, used only for plots of "hour of day"
# Fraction of the way from 10 m to 80 m, in log-height space: ln(22.1/10) / ln(80/10) ~= 0.38. Assumes stable conditions.
LOG_WEIGHT_22 = np.log(SENSOR_HEIGHT_M / 10) / np.log(80 / 10)


def minute_grid(obs, col):
    """Put a 1-minute sensor column on a complete, regular minute grid (NaN where missing)."""
    grid = pd.date_range(PERIOD_START, PERIOD_END - pd.Timedelta(minutes=1), freq="min")
    s = obs.set_index(obs.t.dt.floor("min"))[col].groupby(level=0).mean()  # 498 minutes have 2 reports
    return s.reindex(grid)


def window_stats(values, grid_start, times, start_min, end_min):
    """Mean and count of non-NaN minutes in [time + start_min, time + end_min) for each time. Uses cumulative sums."""
    ok = ~np.isnan(values)
    csum = np.concatenate([[0.0], np.cumsum(np.where(ok, values, 0.0))])
    ccnt = np.concatenate([[0], np.cumsum(ok)])
    idx = ((times - grid_start).total_seconds() // 60).astype(int).to_numpy()
    lo = np.clip(idx + start_min, 0, len(values))
    hi = np.clip(idx + end_min, 0, len(values))
    n = ccnt[hi] - ccnt[lo]
    mean = (csum[hi] - csum[lo]) / np.where(n > 0, n, np.nan)
    return mean, n


def build_targets(obs):
    """One row per valid hour: target wind plus diagnostic stats of the minutes inside the window."""
    hours = pd.date_range(PERIOD_START, PERIOD_END - pd.Timedelta(hours=1), freq="h")
    ws = minute_grid(obs, "obs_ws_qc")
    wd = minute_grid(obs, "obs_wd_qc")
    g0 = ws.index[0]

    mean, n = window_stats(ws.to_numpy(), g0, hours, *WINDOW)
    sq, _ = window_stats(ws.to_numpy() ** 2, g0, hours, *WINDOW)
    calm, _ = window_stats(np.where(np.isnan(ws), np.nan, (ws == 0).astype(float)), g0, hours, *WINDOW)

    # Vector-mean direction (speed-weighted), for diagnostics only
    rad = np.deg2rad(wd.to_numpy())
    u = -ws.to_numpy() * np.sin(rad)
    v = -ws.to_numpy() * np.cos(rad)
    um, _ = window_stats(u, g0, hours, *WINDOW)
    vm, _ = window_stats(v, g0, hours, *WINDOW)

    t = pd.DataFrame({
        "valid": hours,
        "obs_n_minutes": n,
        "obs_ws_mean": np.where(n >= MIN_VALID_MINUTES, mean, np.nan),   # <-- the target
        "obs_ws_std": np.sqrt(np.clip(sq - mean ** 2, 0, None)),
        "obs_calm_frac": calm,
        "obs_wd_vecmean": (np.rad2deg(np.arctan2(-um, -vm)) + 360) % 360,
    })
    return t


def compare_windows(obs, hrrr):
    """How the raw-HRRR baseline scores against targets built with different averaging windows (f01-f18 only)."""
    ws = minute_grid(obs, "obs_ws_qc")
    h = hrrr[hrrr.hrrr_forecast_hour.between(1, 18) & hrrr.hrrr_wind10_ms.notna()]
    hours = pd.DatetimeIndex(h.valid)
    rows = []
    for name, (a, b) in {"1 min": (0, 1), "10 min": (-5, 5), "30 min": (-15, 15),
                         "60 min": (-30, 30), "60 min before": (-60, 0), "120 min": (-60, 60)}.items():
        mean, n = window_stats(ws.to_numpy(), ws.index[0], hours, a, b)
        k = ~np.isnan(mean) & (n >= 0.75 * (b - a))
        err = h.hrrr_wind10_ms.to_numpy()[k] - mean[k]
        rows.append({"window": name, "n": int(k.sum()), "rmse": float(np.sqrt(np.mean(err ** 2))),
                     "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err)),
                     "r": float(np.corrcoef(h.hrrr_wind10_ms.to_numpy()[k], mean[k])[0, 1])})
    return pd.DataFrame(rows)


def scores(pred, obs):
    e = pred - obs
    return {"n": int(e.notna().sum()), "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(e.abs().mean()), "bias": float(e.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)

    # Gate
    step1 = json.loads((out / "qc_report.json").read_text())
    if not all(step1["checks"].values()):
        sys.exit(f"Step 1 checks did not all pass: {step1['checks']}")

    obs = pd.read_parquet(out / "interim" / "obs_clean.parquet")
    hrrr = pd.read_parquet(out / "interim" / "hrrr_clean.parquet")

    targets = build_targets(obs)
    table = hrrr.merge(targets, on="valid", how="left", validate="many_to_one")

    table["base_raw10"] = table.hrrr_wind10_ms
    table["base_log22"] = table.hrrr_wind10_ms + LOG_WEIGHT_22 * (table.hrrr_wind80_ms - table.hrrr_wind10_ms)
    table["usable"] = table.obs_ws_mean.notna() & table.hrrr_wind10_ms.notna()
    table["hour_local"] = (table.valid + pd.Timedelta(hours=LOCAL_OFFSET_H)).dt.hour
    table = table.sort_values(["valid", "run"]).reset_index(drop=True)

    u = table[table.usable]
    both = u[u.base_log22.notna()]        # score both baselines on identical rows
    base_scores = {"raw10_all_usable": scores(u.base_raw10, u.obs_ws_mean),
                   "raw10_same_rows_as_log22": scores(both.base_raw10, both.obs_ws_mean),
                   "log22_same_rows": scores(both.base_log22, both.obs_ws_mean)}
    # Per-lead scores. Plain column math + groupby().mean() works on every pandas version
    err = u.assign(err_raw10=u.base_raw10 - u.obs_ws_mean, err_log22=u.base_log22 - u.obs_ws_mean)
    err = err.assign(se_raw10=err.err_raw10 ** 2, se_log22=err.err_log22 ** 2)
    g = err.groupby("hrrr_forecast_hour")
    by_lead = pd.DataFrame({"rmse_raw10": np.sqrt(g.se_raw10.mean()),
                            "bias_raw10": g.err_raw10.mean(),
                            "rmse_log22": np.sqrt(g.se_log22.mean()),
                            "n": g.size()})
    windows = compare_windows(obs, hrrr)

    minute_mean = float(obs.obs_ws_qc.mean())
    hourly_target_mean = float(targets.obs_ws_mean.mean())
    short = by_lead.loc[1:6, "rmse_raw10"].mean()
    long_ = by_lead.loc[37:48, "rmse_raw10"].mean()
    r_raw = float(np.corrcoef(u.base_raw10, u.obs_ws_mean)[0, 1])

    checks = {
        "D1_one_target_per_forecast_row": len(table) == len(hrrr),
        "D2_target_hours_ge_95pct": targets.obs_ws_mean.notna().mean() >= 0.95,
        "D3_hourly_mean_matches_minute_mean": abs(hourly_target_mean - minute_mean) < 0.1,
        "D4_target_physical_range": bool(targets.obs_ws_mean.dropna().between(0, 60).all()),
        "D5_baseline_in_plausible_range": 0.5 < base_scores["raw10_all_usable"]["rmse"] < 4.0 and r_raw > 0.6,
        "D6_errors_grow_with_lead": bool(long_ > short),
    }

    report = {
        "target_definition": {"window_min": list(WINDOW), "min_valid_minutes": MIN_VALID_MINUTES},
        "rows": {"table": len(table), "usable": int(table.usable.sum()),
                 "no_target": int(table.obs_ws_mean.isna().sum()), "no_hrrr_wind": int(table.hrrr_wind10_ms.isna().sum()),
                 "usable_missing_80m": int((table.usable & table.hrrr_wind80_ms.isna()).sum())},
        "target_hours": {"total": len(targets), "with_target": int(targets.obs_ws_mean.notna().sum())},
        "means_ms": {"obs_minute": minute_mean, "obs_hourly_target": hourly_target_mean,
                     "hrrr_wind10": float(u.base_raw10.mean()), "hrrr_log22": float(both.base_log22.mean())},
        "baselines_all_leads": base_scores,
        "raw10_r": r_raw,
        "window_comparison_f01_f18": windows.round(4).to_dict(orient="records"),
        "rmse_raw10_f01_f06_vs_f37_f48": [float(short), float(long_)],
        "checks": {k: bool(v) for k, v in checks.items()},
    }

    table.to_parquet(out / "interim" / "model_table.parquet", index=False)
    (out / "step2_report.json").write_text(json.dumps(report, indent=2))
    make_figure(u, by_lead, windows, out / "figures" / "02_baseline.png")

    print(json.dumps(report["checks"], indent=2))
    failed = [k for k, v in checks.items() if not v]
    if failed:
        print("VALIDATION FAILED:", failed)
        sys.exit(1)
    print("All step 2 checks passed.")


def make_figure(u, by_lead, windows, path):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # (a) Target window choice
    w = windows.set_index("window")
    ax[0, 0].bar(w.index, w.rmse, color="slategray")
    for i, (r, c) in enumerate(zip(w.rmse, w.r)):
        ax[0, 0].text(i, r + 0.02, f"RMSE {r:.2f}\nr {c:.3f}", ha="center", fontsize=8)
    ax[0, 0].set_ylim(0, w.rmse.max() * 1.25)
    ax[0, 0].set_ylabel("raw HRRR RMSE (m/s)")
    ax[0, 0].set_title("Same forecasts, different target averaging windows (f01–f18)")
    ax[0, 0].tick_params(axis="x", labelsize=8, rotation=15)

    # (b) Error vs lead time
    ax[0, 1].plot(by_lead.index, by_lead.rmse_raw10, label="RMSE, raw HRRR 10 m", color="black")
    ax[0, 1].plot(by_lead.index, by_lead.rmse_log22, label="RMSE, HRRR adjusted to 22.1 m", color="darkorange")
    ax[0, 1].plot(by_lead.index, by_lead.bias_raw10, label="bias, raw HRRR 10 m", color="black", ls="--")
    ax[0, 1].axhline(0, color="gray", lw=0.5)
    ax[0, 1].set_xlabel("lead time (h)"); ax[0, 1].set_ylabel("m/s")
    ax[0, 1].set_title("Baseline error by lead time")
    ax[0, 1].legend(fontsize=8)

    # (c) Forecast vs observed
    hb = ax[1, 0].hexbin(u.base_raw10, u.obs_ws_mean, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    lim = [0, 16]
    ax[1, 0].plot(lim, lim, color="red", lw=1, label="perfect forecast")
    ax[1, 0].set_xlim(lim); ax[1, 0].set_ylim(lim)
    ax[1, 0].set_xlabel("HRRR 10 m wind (m/s)"); ax[1, 0].set_ylabel("sensor hourly mean (m/s)")
    ax[1, 0].set_title("Raw HRRR vs sensor, all leads")
    ax[1, 0].legend(loc="upper left", fontsize=8)
    fig.colorbar(hb, ax=ax[1, 0], label="count (log)")

    # (d) Diurnal cycle
    d = u.groupby("hour_local")[["obs_ws_mean", "base_raw10", "base_log22"]].mean()
    ax[1, 1].plot(d.index, d.obs_ws_mean, color="steelblue", lw=2.5, label="sensor (22.1 m)")
    ax[1, 1].plot(d.index, d.base_raw10, color="black", label="raw HRRR 10 m")
    ax[1, 1].plot(d.index, d.base_log22, color="darkorange", label="HRRR adjusted to 22.1 m")
    ax[1, 1].set_xticks(range(0, 24, 3))
    ax[1, 1].set_xlabel("hour of day (MST, UTC−7)"); ax[1, 1].set_ylabel("mean wind (m/s)")
    ax[1, 1].set_title("Average daily cycle")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()