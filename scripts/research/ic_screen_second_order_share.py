#!/usr/bin/env python3
"""
IC Screening for Second-Order Fund Share Factors

Computes SHARE_STABILITY and SHARE_SKEW from fund_share data with
multi-window analysis (10, 20, 40 days), plus rank EMA preprocessing test.

Factors:
  SHARE_STABILITY_wD = 1.0 / rolling_std(share_chg_5d, window=w)
    - High = stable institutional behavior = conviction
  SHARE_SKEW_wD = rolling_skew(share_chg_5d, window=w)
    - Positive skew = occasional large inflows, steady small outflows

Output: results/ic_screen_second_order_share/
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
OUT_DIR = BASE_DIR / "results" / "ic_screen_second_order_share"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Period split
TRAIN_END = "20250430"
HO_START = "20250501"

# Multi-window grid
WINDOWS = [10, 20, 40]

# Epsilon for division safety
EPS = 1e-10


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


def build_close_panel(daily_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Index]:
    """Build aligned close panel. Index=trade_date (str), columns=ETF codes."""
    codes = sorted(daily_data.keys())
    all_dates = sorted(set().union(*(df.index for df in daily_data.values())))
    idx = pd.Index(all_dates, name="trade_date")

    close_panel = pd.DataFrame(index=idx, columns=codes, dtype=float)
    for code, df in daily_data.items():
        close_panel.loc[df.index, code] = df["adj_close"].values

    close_panel = close_panel.ffill().fillna(1.0)
    return close_panel, idx


def load_fund_share(trade_dates: pd.Index) -> pd.DataFrame:
    """Load fund_share for all ETFs, forward-filled to daily calendar.
    Returns DataFrame with index=trade_date (str), columns=ETF codes, values=fd_share."""
    files = sorted(glob.glob(str(RAW_DIR / "fund_share" / "fund_share_*.parquet")))
    series_list = []
    for f in files:
        fname = os.path.basename(f)
        code = fname.replace("fund_share_", "").replace(".parquet", "")
        df = pd.read_parquet(f)
        df["trade_date"] = df["trade_date"].dt.strftime("%Y%m%d")
        df = df.drop_duplicates("trade_date", keep="last").set_index("trade_date").sort_index()
        series_list.append(df["fd_share"].rename(code))

    panel = pd.DataFrame(index=trade_dates)
    for s in series_list:
        panel = panel.join(s, how="left")

    panel = panel.ffill()
    return panel


# ──────────────────────────────────────────────
# Factor computation
# ──────────────────────────────────────────────

def compute_second_order_factors(
    share_panel: pd.DataFrame,
    windows: list[int],
) -> dict[str, pd.DataFrame]:
    """Compute SHARE_STABILITY and SHARE_SKEW for multiple windows.

    share_chg_5d = (fd - fd.shift(5)) / (fd.shift(5) + eps)
    SHARE_STABILITY_wD = 1.0 / rolling_std(share_chg_5d, w)  [NaN where std < eps]
    SHARE_SKEW_wD = rolling_skew(share_chg_5d, w)
    """
    # Base signal: 5-day share change rate
    fd_shifted = share_panel.shift(5)
    share_chg_5d = (share_panel - fd_shifted) / (fd_shifted + EPS)

    factors = {}
    for w in windows:
        min_periods = max(w // 2, 5)

        # SHARE_STABILITY
        rolling_std = share_chg_5d.rolling(w, min_periods=min_periods).std()
        stability = 1.0 / rolling_std
        # Where std < eps, the value is astronomically large -> set to NaN
        stability[rolling_std < EPS] = np.nan
        factors[f"SHARE_STABILITY_{w}D"] = stability

        # SHARE_SKEW
        skew = share_chg_5d.rolling(w, min_periods=min_periods).skew()
        factors[f"SHARE_SKEW_{w}D"] = skew

    return factors


def apply_rank_ema(factor_panel: pd.DataFrame, halflife: int = 5) -> pd.DataFrame:
    """Apply rank EMA preprocessing: cross-sectional rank -> EWM smoothing.
    This tests whether smoothing daily ranks improves IC stability."""
    # Cross-sectional rank (pct)
    ranks = factor_panel.rank(axis=1, pct=True)
    # EWM smoothing along time axis for each ETF
    smoothed = ranks.ewm(halflife=halflife, min_periods=halflife).mean()
    return smoothed


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


def compute_rank_autocorr(factor_panel: pd.DataFrame, lag: int) -> float:
    """Compute average cross-sectional rank autocorrelation at given lag.
    Measures how stable the cross-sectional ranking is over time."""
    ranks = factor_panel.rank(axis=1, pct=True)
    ranks_lagged = ranks.shift(lag)

    autocorrs = []
    common_dates = ranks.index[lag:]  # Skip initial NaN rows
    for dt in common_dates:
        r_now = ranks.loc[dt]
        r_lag = ranks_lagged.loc[dt]
        mask = r_now.notna() & r_lag.notna()
        if mask.sum() < 5:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = stats.spearmanr(r_now[mask].values, r_lag[mask].values)
        if not np.isnan(corr):
            autocorrs.append(corr)

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
        ic_ir = mean_ic / std_ic if std_ic > EPS else 0.0
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
    print("SECOND-ORDER FUND SHARE FACTOR IC SCREENING")
    print("=" * 70)

    # 1. Load daily data
    print("\n[1/5] Loading daily OHLCV data...")
    daily_data = load_daily()
    close_panel, trade_dates = build_close_panel(daily_data)
    etf_codes = list(close_panel.columns)
    print(f"  {len(etf_codes)} ETFs, {len(trade_dates)} trading days")
    print(f"  Date range: {trade_dates[0]} ~ {trade_dates[-1]}")

    # 2. Compute forward returns (5-day, matching FREQ=5)
    print("\n[2/5] Computing 5-day forward returns...")
    fwd_ret = close_panel.shift(-5) / close_panel - 1
    n_valid_fwd = fwd_ret.notna().sum().sum()
    print(f"  Valid forward return observations: {n_valid_fwd:,}")

    # 3. Load fund_share and compute second-order factors
    print("\n[3/5] Loading fund_share data...")
    share_panel = load_fund_share(trade_dates)
    n_share_etfs = share_panel.notna().any().sum()
    print(f"  fund_share loaded for {n_share_etfs} ETFs")
    print(f"  NaN rate: {share_panel.isna().mean().mean():.1%}")

    print("\n[4/5] Computing second-order factors...")
    factors = compute_second_order_factors(share_panel, WINDOWS)
    print(f"  Computed {len(factors)} factors: {sorted(factors.keys())}")

    # Print factor coverage
    for name, panel in sorted(factors.items()):
        valid_rate = panel.notna().mean().mean()
        print(f"    {name}: valid rate {valid_rate:.1%}")

    # 5. Run IC screening
    print("\n[5/5] Running cross-sectional Spearman rank IC + rank autocorrelation...")
    print("-" * 70)

    results = []
    ic_series_dict = {}

    for name in sorted(factors.keys()):
        factor = factors[name]
        ic_ser = compute_ic_series(factor, fwd_ret)
        ic_series_dict[name] = ic_ser
        summary = ic_summary(ic_ser, TRAIN_END, HO_START)

        # Rank autocorrelation
        rank_autocorr_1 = compute_rank_autocorr(factor, lag=1)
        rank_autocorr_5 = compute_rank_autocorr(factor, lag=5)

        summary["factor"] = name
        summary["rank_autocorr_1"] = rank_autocorr_1
        summary["rank_autocorr_5"] = rank_autocorr_5
        results.append(summary)

        sign = "+" if summary["full_mean_IC"] > 0 else ""
        print(
            f"  {name:25s}  IC={sign}{summary['full_mean_IC']:.4f}  "
            f"IR={summary['full_IC_IR']:.3f}  "
            f"pos={summary['full_IC_pos_rate']:.1%}  "
            f"RankAC1={rank_autocorr_1:.3f}  "
            f"RankAC5={rank_autocorr_5:.3f}"
        )

    # Build results DataFrame
    results_df = pd.DataFrame(results).set_index("factor")
    results_df = results_df.sort_values("full_IC_IR", ascending=False, key=abs)

    # Apply filter: |IC| > 0.03 AND HO |IC| >= 0.7 * Train |IC|
    results_df["pass_ic_threshold"] = results_df["full_mean_IC"].abs() > 0.03
    results_df["ho_train_ratio"] = (
        results_df["ho_mean_IC"].abs() / results_df["train_mean_IC"].abs().replace(0, np.nan)
    )
    results_df["pass_ho_stability"] = results_df["ho_train_ratio"] >= 0.7
    results_df["WINNER"] = results_df["pass_ic_threshold"] & results_df["pass_ho_stability"]

    # Print main results table
    print("\n" + "=" * 70)
    print("MAIN RESULTS: SECOND-ORDER SHARE FACTORS")
    print("=" * 70)

    cols_display = [
        "full_mean_IC", "full_IC_IR", "full_IC_pos_rate", "full_n_days",
        "train_mean_IC", "train_IC_IR",
        "ho_mean_IC", "ho_IC_IR", "ho_n_days",
        "rank_autocorr_1", "rank_autocorr_5",
        "ho_train_ratio", "WINNER",
    ]

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results_df[cols_display].to_string())

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
                f"RankAC1={row['rank_autocorr_1']:.3f}  "
                f"RankAC5={row['rank_autocorr_5']:.3f}"
            )
    else:
        print("  No factors passed the filter criteria.")

    # Near misses
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

    # ── Best window analysis ──
    print(f"\n{'=' * 70}")
    print("BEST WINDOW ANALYSIS")
    print(f"{'=' * 70}")

    for factor_type in ["SHARE_STABILITY", "SHARE_SKEW"]:
        type_rows = results_df[results_df.index.str.startswith(factor_type)]
        if len(type_rows) == 0:
            continue
        best_idx = type_rows["full_IC_IR"].abs().idxmax()
        best_row = type_rows.loc[best_idx]
        print(
            f"  {factor_type}: best window = {best_idx}  "
            f"|IC|={abs(best_row['full_mean_IC']):.4f}  "
            f"|IR|={abs(best_row['full_IC_IR']):.3f}  "
            f"RankAC1={best_row['rank_autocorr_1']:.3f}"
        )

    # ── Rank EMA preprocessing test ──
    print(f"\n{'=' * 70}")
    print("RANK EMA PREPROCESSING TEST (halflife=5)")
    print(f"{'=' * 70}")

    ema_results = []
    for name in sorted(factors.keys()):
        factor_raw = factors[name]
        factor_ema = apply_rank_ema(factor_raw, halflife=5)

        # IC for raw
        ic_raw = ic_series_dict[name]
        raw_summary = ic_summary(ic_raw, TRAIN_END, HO_START)

        # IC for EMA-smoothed
        ic_ema = compute_ic_series(factor_ema, fwd_ret)
        ema_summary = ic_summary(ic_ema, TRAIN_END, HO_START)

        # Rank autocorrelation for EMA
        ema_rank_ac1 = compute_rank_autocorr(factor_ema, lag=1)
        ema_rank_ac5 = compute_rank_autocorr(factor_ema, lag=5)

        ema_results.append({
            "factor": name,
            "raw_IC": raw_summary["full_mean_IC"],
            "raw_IR": raw_summary["full_IC_IR"],
            "raw_HO_IC": raw_summary["ho_mean_IC"],
            "ema_IC": ema_summary["full_mean_IC"],
            "ema_IR": ema_summary["full_IC_IR"],
            "ema_HO_IC": ema_summary["ho_mean_IC"],
            "IC_delta": ema_summary["full_mean_IC"] - raw_summary["full_mean_IC"],
            "IR_delta": ema_summary["full_IC_IR"] - raw_summary["full_IC_IR"],
            "ema_RankAC1": ema_rank_ac1,
            "ema_RankAC5": ema_rank_ac5,
        })

        print(
            f"  {name:25s}  "
            f"raw_IC={raw_summary['full_mean_IC']:+.4f} -> ema_IC={ema_summary['full_mean_IC']:+.4f}  "
            f"raw_IR={raw_summary['full_IC_IR']:+.3f} -> ema_IR={ema_summary['full_IC_IR']:+.3f}  "
            f"IC_delta={ema_summary['full_mean_IC'] - raw_summary['full_mean_IC']:+.4f}  "
            f"ema_RankAC1={ema_rank_ac1:.3f}"
        )

    ema_df = pd.DataFrame(ema_results).set_index("factor")

    # Summary of EMA effect
    avg_ir_delta = ema_df["IR_delta"].mean()
    n_improved = (ema_df["IR_delta"].abs() > ema_df["raw_IR"].abs() * 0.01).sum()  # >1% improvement
    print(f"\n  Average IR delta: {avg_ir_delta:+.4f}")
    print(f"  Factors with >1% |IR| improvement: {n_improved}/{len(ema_df)}")

    # ── Save outputs ──
    results_df.to_csv(OUT_DIR / "ic_results_all.csv")
    ema_df.to_csv(OUT_DIR / "ic_rank_ema_comparison.csv")
    if len(winners) > 0:
        winners.to_csv(OUT_DIR / "ic_winners.csv")
        # Save IC time series for winners
        ic_ts_dict = {name: ic_series_dict[name] for name in winners.index}
        ic_ts_df = pd.DataFrame(ic_ts_dict)
        ic_ts_df.to_csv(OUT_DIR / "ic_timeseries_winners.csv")
        print(f"\nIC time series for {len(winners)} winners saved.")

    print(f"\nAll results saved to {OUT_DIR}/")
    return results_df, ema_df


if __name__ == "__main__":
    results, ema = main()
