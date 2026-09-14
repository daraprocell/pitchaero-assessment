"""
Step 1 — Load and inspect the raw sensor/HRRR csv file.

Outputs:
  interim/obs_clean.parquet   sensor rows + QC flags + obs_ws_qc (wind with faults set to NaN)
  interim/hrrr_clean.parquet  HRRR rows with parsed times, wind10 back-filled from u/v, wind80 derived
  qc_report.json              counts, missingness, and the pass/fail validation gate
  figures/01_qc.png           visual evidence

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
OBS_COLS = ["obs_wind_speed_ms", "obs_wind_direction_deg", "obs_air_temp_c", "obs_rh_pct", "obs_pressure_pa"]


def load(path):
    df = pd.read_csv(path, low_memory=False)
    obs = df[df.record_type == "sensor_obs"].dropna(axis=1, how="all").copy()
    hrrr = df[df.record_type == "hrrr_forecast"].dropna(axis=1, how="all").copy()
    obs["t"] = pd.to_datetime(obs.timestamp_utc, utc=True)
    hrrr["valid"] = pd.to_datetime(hrrr.timestamp_utc, utc=True)
    hrrr["run"] = pd.to_datetime(hrrr.hrrr_run_time_utc, utc=True)
    return df, obs.sort_values("t").reset_index(drop=True), hrrr.sort_values(["valid", "run"]).reset_index(drop=True)


def qc_obs(obs):
    ws, wd = obs.obs_wind_speed_ms, obs.obs_wind_direction_deg
    # Fault signature: speed exactly 0 with direction exactly 225. Heavily co-occurs with pressure == 0.
    # Genuine calms report varied directions. This pattern is validated against HRRR below.
    obs["flag_wind_fault"] = (ws == 0) & (wd == 225)
    obs["flag_thermo_fault"] = (obs.obs_air_temp_c == 0) & (obs.obs_rh_pct == 0)
    obs["flag_pressure_bad"] = (obs.obs_pressure_pa == 0) | (obs.obs_pressure_pa < 80000)
    obs["flag_real_calm"] = (ws == 0) & ~obs.flag_wind_fault

    obs["obs_ws_qc"] = ws.where(~obs.flag_wind_fault)
    obs["obs_wd_qc"] = wd.where(~obs.flag_wind_fault & (ws > 0))  # direction undefined in calm
    obs["obs_temp_qc"] = obs.obs_air_temp_c.where(~obs.flag_thermo_fault)
    obs["obs_rh_qc"] = obs.obs_rh_pct.where(~obs.flag_thermo_fault)
    obs["obs_pressure_qc"] = obs.obs_pressure_pa.where(~obs.flag_pressure_bad)
    return obs


def qc_hrrr(hrrr):
    calc10 = np.hypot(hrrr.hrrr_u10_ms, hrrr.hrrr_v10_ms)
    both = hrrr.hrrr_wind10_ms.notna() & calc10.notna()
    wind10_maxdiff = float((hrrr.hrrr_wind10_ms[both] - calc10[both]).abs().max())
    hrrr["wind10_backfilled"] = hrrr.hrrr_wind10_ms.isna() & calc10.notna()
    hrrr["hrrr_wind10_ms"] = hrrr.hrrr_wind10_ms.fillna(calc10)
    hrrr["hrrr_wind80_ms"] = np.hypot(hrrr.hrrr_u80_ms, hrrr.hrrr_v80_ms)
    return hrrr, wind10_maxdiff


def minute_series(obs, col):
    s = obs.set_index(obs.t.dt.floor("min"))[col].groupby(level=0).mean()
    return s.reindex(pd.date_range(PERIOD_START, PERIOD_END - pd.Timedelta(minutes=1), freq="min"))


def lag_correlation(obs, hrrr):
    """Correlate centered 10-min obs means at (valid + lag) with HRRR f01. Detects clock/timezone offsets."""
    f01 = hrrr[hrrr.hrrr_forecast_hour == 1].set_index("valid")
    ws = minute_series(obs, "obs_ws_qc").rolling(10, center=True, min_periods=6).mean()
    tt = minute_series(obs, "obs_temp_qc").rolling(10, center=True, min_periods=6).mean()
    rows = []
    for lag in range(-120, 121, 5):
        idx = f01.index + pd.Timedelta(minutes=lag)
        rec = {"lag_min": lag}
        for name, s, h in [("r_wind", ws, f01.hrrr_wind10_ms), ("r_temp", tt, f01.hrrr_t2m_k)]:
            o = s.reindex(idx).values
            m = ~np.isnan(o) & h.notna().values
            rec[name] = float(np.corrcoef(o[m], h.values[m])[0, 1])
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to sensor_wind.csv(.gz)")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "interim").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    raw, obs, hrrr = load(args.input)
    obs = qc_obs(obs)
    hrrr, wind10_maxdiff = qc_hrrr(hrrr)

    # HRRR 10 m wind at the nearest hour for each obs minute — used to validate the fault flag
    hr_ref = hrrr[hrrr.hrrr_forecast_hour.between(1, 6)].groupby("valid").hrrr_wind10_ms.mean()
    obs["hrrr10_ref"] = obs.t.dt.round("h").map(hr_ref)
    med_fault = float(obs.loc[obs.flag_wind_fault, "hrrr10_ref"].median())
    med_calm = float(obs.loc[obs.flag_real_calm, "hrrr10_ref"].median())
    med_all = float(obs.loc[obs.obs_ws_qc > 0, "hrrr10_ref"].median())

    lags = lag_correlation(obs, hrrr)
    peak_wind = int(lags.loc[lags.r_wind.idxmax(), "lag_min"])
    peak_temp = int(lags.loc[lags.r_temp.idxmax(), "lag_min"])

    expected_minutes = int((PERIOD_END - PERIOD_START).total_seconds() // 60)
    expected_runs = pd.date_range(PERIOD_START, PERIOD_END - pd.Timedelta(hours=1), freq="h")
    missing_runs = expected_runs.difference(hrrr.run.unique())
    valid_ws_minutes = int(obs.obs_ws_qc.notna().sum())

    checks = {
        "C1_rows_partition": len(obs) + len(hrrr) == len(raw),
        "C2_times_in_period": bool(obs.t.between(PERIOD_START, PERIOD_END).all()
                                   and hrrr.valid.between(PERIOD_START, PERIOD_END).all()),
        "C3_no_duplicate_keys": bool(not obs.t.duplicated().any() and not hrrr.duplicated(["run", "valid"]).any()),
        "C4_lead_matches_times": bool(np.allclose((hrrr.valid - hrrr.run).dt.total_seconds() / 3600, hrrr.hrrr_forecast_hour)),
        "C5_wind10_equals_hypot_uv": wind10_maxdiff < 0.01,
        "C6_fault_flag_distinct_from_calms": (med_fault - med_calm) > 1.0,
        "C7_obs_wind_coverage_ge_95pct": valid_ws_minutes / expected_minutes >= 0.95,
        "C8_no_clock_offset_gt_30min": abs(peak_wind) <= 30 and abs(peak_temp) <= 30,
        "C9_physical_ranges": bool(obs.obs_ws_qc.dropna().between(0, 60).all()
                                   and obs.obs_wd_qc.dropna().between(0, 360).all()
                                   and (hrrr.hrrr_wind10_ms.dropna() >= 0).all()),
    }

    report = {
        "rows": {"total": len(raw), "sensor_obs": len(obs), "hrrr_forecast": len(hrrr)},
        "obs": {
            "expected_minutes": expected_minutes,
            "rows": len(obs),
            "minute_buckets_with_2_reports": int(obs.t.dt.floor("min").duplicated().sum()),
            "wind_fault_rows(0 m/s @ 225deg)": int(obs.flag_wind_fault.sum()),
            "real_calm_rows": int(obs.flag_real_calm.sum()),
            "thermo_fault_rows": int(obs.flag_thermo_fault.sum()),
            "pressure_bad_rows": int(obs.flag_pressure_bad.sum()),
            "valid_wind_minutes": valid_ws_minutes,
            "max_gap_min": float(obs.t.diff().dt.total_seconds().max() / 60),
        },
        "fault_flag_validation_hrrr10_median_ms": {"fault_signature": med_fault, "real_calms": med_calm, "all_nonzero_obs": med_all},
        "hrrr": {
            "runs_present": int(hrrr.run.nunique()),
            "missing_runs": [str(r) for r in missing_runs],
            "valid_times": int(hrrr.valid.nunique()),
            "rows_per_valid_time_median": float(hrrr.groupby("valid").size().median()),
            "max_lead_h": float(hrrr.hrrr_forecast_hour.max()),
            "wind10_backfilled_from_uv": int(hrrr.wind10_backfilled.sum()),
            "wind10_vs_hypot_maxdiff": wind10_maxdiff,
            "missing_frac": hrrr.filter(like="hrrr_").isna().mean().round(4).to_dict(),
            "roughness_unique_values": int(hrrr.hrrr_roughness_length_m.nunique()),
        },
        "time_alignment": {"peak_lag_wind_min": peak_wind, "peak_lag_temp_min": peak_temp,
                           "r_wind_at_0": float(lags.loc[lags.lag_min == 0, "r_wind"].iloc[0]),
                           "r_wind_at_peak": float(lags.r_wind.max())},
        "checks": {k: bool(v) for k, v in checks.items()},
    }

    obs.drop(columns=["record_type", "timestamp_utc"]).to_parquet(out / "interim" / "obs_clean.parquet", index=False)
    hrrr.drop(columns=["record_type", "timestamp_utc", "hrrr_run_time_utc"]).to_parquet(out / "interim" / "hrrr_clean.parquet", index=False)
    (out / "qc_report.json").write_text(json.dumps(report, indent=2))

    make_figure(obs, hrrr, lags, out / "figures" / "01_qc.png")

    print(json.dumps(report["checks"], indent=2))
    failed = [k for k, v in checks.items() if not v]
    if failed:
        print("VALIDATION FAILED:", failed)
        sys.exit(1)
    print("All step 1 checks passed.")


def make_figure(obs, hrrr, lags, path):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    day = obs.groupby(obs.t.dt.floor("D")).agg(n=("t", "size"), fault=("flag_wind_fault", "sum"), calm=("flag_real_calm", "sum"))
    ax[0, 0].plot(day.index, 100 * day.n / 1440, lw=1, label="reports / 1440 min")
    ax[0, 0].bar(day.index, 100 * day.fault / 1440, color="crimson", width=1, label="wind-fault rows (0 m/s @ 225°)")
    ax[0, 0].set_ylabel("% of day"); ax[0, 0].set_title("Sensor completeness and fault rows per day")
    ax[0, 0].legend(fontsize=8)

    hd = hrrr.groupby(hrrr.valid.dt.floor("D"))
    ax[0, 1].plot(hd.hrrr_u10_ms.apply(lambda s: 100 * s.notna().mean()), lw=1, label="u10/v10 present")
    ax[0, 1].plot(hd.hrrr_u80_ms.apply(lambda s: 100 * s.notna().mean()), lw=1, label="u80/v80 present")
    ax[0, 1].set_ylabel("% of HRRR rows"); ax[0, 1].set_title("HRRR field availability by valid day")
    ax[0, 1].legend(fontsize=8)

    bins = np.arange(0, 14, 0.5)
    for mask, label, c in [(obs.flag_wind_fault, "potential fault signature", "crimson"),
                           (obs.flag_real_calm, "0 m/s reports not at 225 degrees)", "steelblue"),
                           (obs.obs_ws_qc > 0, "all other (non-zero) obs", "gray")]:
        ax[1, 0].hist(obs.loc[mask, "hrrr10_ref"].dropna(), bins=bins, density=True, histtype="step", lw=2, color=c, label=label)
    ax[1, 0].set_xlabel("HRRR 10 m wind at that hour (m/s)"); ax[1, 0].set_ylabel("density")
    ax[1, 0].set_title("Potential Fault Signature: 0m/s at 225 degrees")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(lags.lag_min, lags.r_wind, label="wind: obs 10-min mean vs HRRR f01")
    ax2 = ax[1, 1].twinx()
    ax2.plot(lags.lag_min, lags.r_temp, color="darkorange", label="temp")
    ax[1, 1].axvline(0, color="k", lw=0.5)
    ax[1, 1].set_xlabel("obs time offset relative to HRRR valid time (min)")
    ax[1, 1].set_ylabel("r (wind)"); ax2.set_ylabel("r (temp)")
    ax[1, 1].set_title("Time alignment check (no hour-scale offset expected)")
    ax[1, 1].legend(loc="lower left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()