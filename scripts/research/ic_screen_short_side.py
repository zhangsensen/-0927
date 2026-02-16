#!/usr/bin/env python3
"""
IC Screening for Short-Side Margin Factors

Computes 4 short-side margin factors and runs cross-sectional Spearman rank IC analysis.

Factors:
  1. SHORT_RATIO        = rqye / rzrqye       (做空占比, bounded [0,1])
  2. SHORT_INTENSITY     = rqmcl / volume      (融券卖出占成交量)
  3. SHORT_COVER_SPEED   = rqchl / (rqmcl+eps) (融券偿还/卖出比)
  4. NET_MARGIN_FLOW     = (rzmre - rzche)/rzye (净融资流入)

Output: results/ic_screen_short_side/
"""

import glob
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "raw" / "ETF"
OUT_DIR = BASE_DIR / "results" / "ic_screen_short_side"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Period split
TRAIN_END = "20250430"
HO_START = "20250501"


# ──────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────

def load_daily() -> dict[str, pd.DataFrame]:
    """Load daily OHLCV data for all ETFs. Returns {code: df} with string YYYYMMDD trade_date index."""
    daily_files = sorted(glob.glob(str(RAW_DIR / "daily" / "*.parquet")))
    result = {}
    for f in daily_files:
        fname = os.path.basename(f)
        code = fname.split(".")[0]
        df = pd.read_parquet(f)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df.set_index("trade_date").sort_index()
        result[code] = df
    return result


def build_panel(daily_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """Build aligned close and volume panels. Index=trade_date (str), columns=ETF codes."""
    codes = sorted(daily_data.keys())
    all_dates = sorted(set().union(*(df.index for df in daily_data.values())))
    idx = pd.Index(all_dates, name="trade_date")

    close_panel = pd.DataFrame(index=idx, columns=codes, dtype=float)
    vol_panel = pd.DataFrame(index=idx, columns=codes, dtype=float)

    for code, df in daily_data.items():
        close_panel.loc[df.index, code] = df["adj_close"].values
        vol_panel.loc[df.index, code] = df["vol"].values

    close_panel = close_panel.ffill().fillna(1.0)
    vol_panel = vol_panel.ffill().fillna(0.0)
    return close_panel, vol_panel, idx


def load_margin_panels(trade_dates: pd.Index) -> dict[str, pd.DataFrame]:
    """Load margin data for all required fields.
    Returns dict of panels: {col_name: DataFrame(index=trade_date, columns=short_code)}."""
    f = RAW_DIR / "margin" / "margin_pool43_2020_now.parquet"
    df = pd.read_parquet(f)
    df["code"] = df["ts_code"].str.split(".").str[0]
    df = df.set_index("trade_date").sort_index()

    needed_cols = ["rzye", "rqye", "rzmre", "rzche", "rqmcl", "rqchl", "rzrqye"]
    panels = {}
    for col in needed_cols:
        pivot = df.pivot_table(index="trade_date", columns="code", values=col, aggfunc="last")
        pivot = pivot.reindex(trade_dates).ffill()
        panels[col] = pivot
    return panels


# ──────────────────────────────────────────────
# Factor computation
# ──────────────────────────────────────────────

def compute_short_side_factors(
    margin_panels: dict[str, pd.DataFrame],
    vol_panel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute 4 short-side margin factors."""
    factors = {}

    rqye = margin_panels["rqye"]
    rzrqye = margin_panels["rzrqye"]
    rqmcl = margin_panels["rqmcl"]
    rqchl = margin_panels["rqchl"]
    rzmre = margin_panels["rzmre"]
    rzche = margin_panels["rzche"]
    rzye = margin_panels["rzye"]

    margin_codes = rqye.columns
    eps = 1e-8

    # 1. SHORT_RATIO = rqye / rzrqye  (bounded [0,1])
    rzrqye_safe = rzrqye.replace(0, np.nan)
    factors["SHORT_RATIO"] = rqye / rzrqye_safe

    # 2. SHORT_INTENSITY = rqmcl / volume  (融券卖出占成交量)
    vol_aligned = vol_panel[vol_panel.columns.intersection(margin_codes)]
    vol_safe = vol_aligned.replace(0, np.nan)
    factors["SHORT_INTENSITY"] = rqmcl[vol_aligned.columns] / vol_safe

    # 3. SHORT_COVER_SPEED = rqchl / (rqmcl + eps)  (融券偿还/卖出比)
    factors["SHORT_COVER_SPEED"] = rqchl / (rqmcl + eps)

    # 4. NET_MARGIN_FLOW = (rzmre - rzche) / rzye  (净融资流入)
    rzye_safe = rzye.replace(0, np.nan)
    factors["NET_MARGIN_FLOW"] = (rzmre - rzche) / rzye_safe

    return factors


# ──────────────────────────────────────────────
# IC computation
# ──────────────────────────────────────────────

def compute_ic_series(factor_panel: pd.DataFrame, fwd_ret_panel: pd.DataFrame) -> pd.Series:
    """Compute daily cross-sectional Spearman rank IC."""
    common_dates = factor_panel.index.intersection(fwd_ret_panel.index)
    common_codes = factor_panel.columns.intersection(fwd_ret_panel.columns)

    ic_values = {}
    for dt in common_dates:
        f_vals = factor_panel.loc[dt, common_codes]
        r_vals = fwd_ret_panel.loc[dt, common_codes]
        mask = f_vals.notna() & r_vals.notna()
        n_valid = mask.sum()
        if n_valid < 5:
            continue
        f_clean = f_vals[mask]
        r_clean = r_vals[mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic, _ = stats.spearmanr(f_clean.values, r_clean.values)
        if not np.isnan(ic):
            ic_values[dt] = ic

    return pd.Series(ic_values)


def compute_rank_autocorr(factor_panel: pd.DataFrame) -> float:
    """Compute average lag-1 rank autocorrelation across ETFs.
    Measures how stable cross-sectional ranks are day-to-day."""
    ranks = factor_panel.rank(axis=1, pct=True)
    # For each ETF, compute autocorr(1) of its rank percentile time series
    autocorrs = []
    for col in ranks.columns:
        s = ranks[col].dropna()
        if len(s) < 20:
            continue
        ac = s.autocorr(lag=1)
        if not np.isnan(ac):
            autocorrs.append(ac)
    return np.mean(autocorrs) if autocorrs else np.nan


def ic_summary(ic_series: pd.Series, train_end: str, ho_start: str) -> dict:
    """Compute IC summary statistics split by Train/HO periods."""
    if len(ic_series) == 0:
        empty = {"mean_IC": np.nan, "IC_IR": np.nan, "IC_pos_rate": np.nan, "n_days": 0}
        return {
            f"{p}_{k}": v
            for p in ["full", "train", "ho"]
            for k, v in empty.items()
        }

    ic_series.index = ic_series.index.astype(str)
    train_ic = ic_series[ic_series.index <= train_end]
    ho_ic = ic_series[ic_series.index >= ho_start]

    def _stats(s):
        if len(s) == 0:
            return {"mean_IC": np.nan, "IC_IR": np.nan, "IC_pos_rate": np.nan, "n_days": 0}
        mean_ic = s.mean()
        std_ic = s.std()
        ic_ir = mean_ic / std_ic if std_ic > 1e-10 else 0.0
        pos_rate = (s > 0).mean()
        return {"mean_IC": mean_ic, "IC_IR": ic_ir, "IC_pos_rate": pos_rate, "n_days": len(s)}

    full = _stats(ic_series)
    train = _stats(train_ic)
    ho = _stats(ho_ic)

    return {
        "full_mean_IC": full["mean_IC"],
        "full_IC_IR": full["IC_IR"],
        "full_IC_pos_rate": full["IC_pos_rate"],
        "full_n_days": full["n_days"],
        "train_mean_IC": train["mean_IC"],
        "train_IC_IR": train["IC_IR"],
        "train_IC_pos_rate": train["IC_pos_rate"],
        "train_n_days": train["n_days"],
        "ho_mean_IC": ho["mean_IC"],
        "ho_IC_IR": ho["IC_IR"],
        "ho_IC_pos_rate": ho["IC_pos_rate"],
        "ho_n_days": ho["n_days"],
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SHORT-SIDE MARGIN FACTOR IC SCREENING")
    print("=" * 70)

    # 1. Load daily data
    print("\n[1/4] Loading daily OHLCV data...")
    daily_data = load_daily()
    close_panel, vol_panel, trade_dates = build_panel(daily_data)
    etf_codes = list(close_panel.columns)
    print(f"  {len(etf_codes)} ETFs, {len(trade_dates)} trading days")
    print(f"  Date range: {trade_dates[0]} ~ {trade_dates[-1]}")

    # 2. Compute forward returns (5-day, matching FREQ=5)
    print("\n[2/4] Computing 5-day forward returns...")
    fwd_ret = close_panel.shift(-5) / close_panel - 1
    n_valid_fwd = fwd_ret.notna().sum().sum()
    print(f"  Valid forward return observations: {n_valid_fwd:,}")

    # 3. Load margin data and compute factors
    print("\n[3/4] Loading margin data and computing short-side factors...")
    margin_panels = load_margin_panels(trade_dates)
    margin_codes = margin_panels["rqye"].columns
    print(f"  Margin data: {len(margin_codes)} ETFs")
    print(f"  Margin codes: {sorted(margin_codes)[:5]}... (showing first 5)")

    factors = compute_short_side_factors(margin_panels, vol_panel)
    print(f"  Computed {len(factors)} factors: {sorted(factors.keys())}")

    # Data quality check
    print("\n  Data quality check:")
    for name, panel in factors.items():
        n_total = panel.size
        n_nan = panel.isna().sum().sum()
        n_inf = np.isinf(panel.values).sum() if not panel.empty else 0
        pct_valid = (1 - n_nan / n_total) * 100 if n_total > 0 else 0
        print(f"    {name:25s}: {pct_valid:.1f}% valid, {n_inf} inf values")
        # Replace inf with NaN
        factors[name] = panel.replace([np.inf, -np.inf], np.nan)

    # 4. Run IC screening
    print("\n[4/4] Running cross-sectional Spearman rank IC...")
    print("-" * 70)

    bounded_flags = {
        "SHORT_RATIO": True,    # [0, 1]
        "SHORT_INTENSITY": False,
        "SHORT_COVER_SPEED": False,
        "NET_MARGIN_FLOW": False,
    }

    results = []
    ic_ts_dict = {}
    for name in sorted(factors.keys()):
        factor = factors[name]
        ic_series = compute_ic_series(factor, fwd_ret)
        ic_ts_dict[name] = ic_series
        summary = ic_summary(ic_series, TRAIN_END, HO_START)
        summary["factor"] = name
        summary["rank_autocorr"] = compute_rank_autocorr(factor)
        summary["bounded"] = bounded_flags.get(name, False)
        results.append(summary)

        sign = "+" if summary["full_mean_IC"] > 0 else ""
        print(
            f"  {name:25s}  IC={sign}{summary['full_mean_IC']:.4f}  "
            f"IR={summary['full_IC_IR']:.3f}  "
            f"pos={summary['full_IC_pos_rate']:.1%}  "
            f"rank_ac={summary['rank_autocorr']:.3f}  "
            f"n={summary['full_n_days']}"
        )

    # Build results DataFrame
    results_df = pd.DataFrame(results).set_index("factor")
    results_df = results_df.sort_values("full_IC_IR", ascending=False, key=abs)

    # Compute HO/Train ratio
    results_df["ho_train_ratio"] = (
        results_df["ho_mean_IC"].abs() / results_df["train_mean_IC"].abs().replace(0, np.nan)
    )

    # Apply filter: |IC| > 0.03 AND HO |IC| >= 0.7 * Train |IC|
    results_df["pass_ic_threshold"] = results_df["full_mean_IC"].abs() > 0.03
    results_df["pass_ho_stability"] = results_df["ho_train_ratio"] >= 0.7
    results_df["WINNER"] = results_df["pass_ic_threshold"] & results_df["pass_ho_stability"]

    # Print summary table
    print("\n" + "=" * 70)
    print("SHORT-SIDE MARGIN FACTOR IC RESULTS")
    print("=" * 70)

    # Formatted output table
    print(f"\n{'Factor':<25s} {'Full_IC':>8s} {'Full_IR':>8s} {'Train_IC':>9s} "
          f"{'HO_IC':>8s} {'HO/Train':>9s} {'Rank_AC':>8s} {'Bounded':>8s} {'PASS':>5s}")
    print("-" * 100)
    for name in results_df.index:
        row = results_df.loc[name]
        sign_full = "+" if row["full_mean_IC"] > 0 else ""
        sign_train = "+" if row["train_mean_IC"] > 0 else ""
        sign_ho = "+" if row["ho_mean_IC"] > 0 else ""
        bounded_str = "Yes" if row["bounded"] else "No"
        pass_str = "YES" if row["WINNER"] else "no"
        ho_ratio_str = f"{row['ho_train_ratio']:.2f}" if not np.isnan(row["ho_train_ratio"]) else "N/A"
        print(
            f"{name:<25s} {sign_full}{row['full_mean_IC']:.4f}  {row['full_IC_IR']:+.3f}  "
            f"{sign_train}{row['train_mean_IC']:.4f}   "
            f"{sign_ho}{row['ho_mean_IC']:.4f}  {ho_ratio_str:>8s}  "
            f"{row['rank_autocorr']:.3f}     {bounded_str:>3s}   {pass_str:>3s}"
        )

    # Winners
    winners = results_df[results_df["WINNER"]]
    print(f"\n{'=' * 70}")
    print(f"WINNERS ({len(winners)} / {len(results_df)} factors pass filter)")
    print(f"Criteria: |IC| > 0.03 AND HO |IC| >= 0.7 * Train |IC|")
    print(f"{'=' * 70}")
    if len(winners) > 0:
        for name in winners.index:
            row = winners.loc[name]
            sign = "+" if row["full_mean_IC"] > 0 else ""
            print(
                f"  {name:25s}  IC={sign}{row['full_mean_IC']:.4f}  "
                f"IR={row['full_IC_IR']:.3f}  "
                f"Train_IC={row['train_mean_IC']:.4f}  "
                f"HO_IC={row['ho_mean_IC']:.4f}  "
                f"HO/Train={row['ho_train_ratio']:.2f}  "
                f"Rank_AC={row['rank_autocorr']:.3f}  "
                f"Bounded={'Yes' if row['bounded'] else 'No'}"
            )
    else:
        print("  No factors passed the filter criteria.")

    # Near-misses
    near_miss = results_df[
        (results_df["pass_ic_threshold"] | results_df["pass_ho_stability"])
        & ~results_df["WINNER"]
    ]
    if len(near_miss) > 0:
        print(f"\nNEAR MISSES ({len(near_miss)} factors):")
        for name in near_miss.index:
            row = near_miss.loc[name]
            print(
                f"  {name:25s}  IC={row['full_mean_IC']:.4f}  "
                f"IR={row['full_IC_IR']:.3f}  "
                f"ic_ok={'Y' if row['pass_ic_threshold'] else 'N'}  "
                f"ho_ok={'Y' if row['pass_ho_stability'] else 'N'}  "
                f"HO/Train={row['ho_train_ratio']:.2f}"
            )

    # Save outputs
    results_df.to_csv(OUT_DIR / "ic_results_short_side.csv")
    if len(winners) > 0:
        winners.to_csv(OUT_DIR / "ic_winners_short_side.csv")

    # Save IC time series
    ic_ts_df = pd.DataFrame(ic_ts_dict)
    ic_ts_df.to_csv(OUT_DIR / "ic_timeseries_short_side.csv")

    print(f"\nResults saved to {OUT_DIR}/")

    return results_df


if __name__ == "__main__":
    results = main()
