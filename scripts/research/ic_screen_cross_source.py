#!/usr/bin/env python3
"""
IC Screening for Cross-Source Interaction Factors

Computes 3 cross-source interaction factors that combine fund_share, margin,
OHLCV price, and volume data, then runs cross-sectional Spearman rank IC analysis
with orthogonality checks against all active OHLCV + non-OHLCV factors.

Factors:
  1. SHARE_PRICE_DIV  = rolling_corr(share_chg_5d, ret_5d, window=20)
  2. FLOW_QUALITY     = share_chg_10d / (vol_ratio_20d + eps)
  3. LEVERAGE_FLOW    = margin_chg_10d - share_chg_10d

Output: results/ic_screen_cross_source/
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
NON_OHLCV_DIR = BASE_DIR / "results" / "non_ohlcv_factors"
OUT_DIR = BASE_DIR / "results" / "ic_screen_cross_source"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Period split
TRAIN_END = "20250430"
HO_START = "20250501"

# 18 active OHLCV factors for orthogonality check
ACTIVE_OHLCV_FACTORS = [
    "ADX_14D", "AMIHUD_ILLIQUIDITY", "BREAKOUT_20D", "CALMAR_RATIO_60D",
    "CORRELATION_TO_MARKET_20D", "GK_VOL_RATIO_20D", "MAX_DD_60D", "MOM_20D",
    "OBV_SLOPE_10D", "PRICE_POSITION_20D", "PRICE_POSITION_120D", "PV_CORR_20D",
    "SHARPE_RATIO_20D", "SLOPE_20D", "UP_DOWN_VOL_RATIO_20D", "VOL_RATIO_20D",
    "VORTEX_14D",
]

# 6 active non-OHLCV factors
ACTIVE_NON_OHLCV_FACTORS = [
    "SHARE_CHG_5D", "SHARE_CHG_10D", "SHARE_CHG_20D", "SHARE_ACCEL",
    "MARGIN_CHG_10D", "MARGIN_BUY_RATIO",
]


# ──────────────────────────────────────────────
# Data loading helpers (reuse pattern from ic_screen_non_ohlcv.py)
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


def load_fund_share(trade_dates: pd.Index) -> pd.DataFrame:
    """Load fund_share for all ETFs, forward-filled to daily calendar."""
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


def load_precomputed_factor(name: str) -> pd.DataFrame:
    """Load a precomputed factor from results/non_ohlcv_factors/.
    Converts Timestamp index to YYYYMMDD string to match daily panels."""
    path = NON_OHLCV_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Precomputed factor not found: {path}")
    df = pd.read_parquet(path)
    # Convert Timestamp index to YYYYMMDD string
    if hasattr(df.index, 'strftime'):
        df.index = df.index.strftime("%Y%m%d")
    else:
        df.index = df.index.astype(str)
    return df


# ──────────────────────────────────────────────
# Factor computation
# ──────────────────────────────────────────────

def compute_cross_source_factors(
    close_panel: pd.DataFrame,
    vol_panel: pd.DataFrame,
    share_panel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute 3 cross-source interaction factors."""
    factors = {}
    eps = 1e-10

    # Load precomputed non-OHLCV factors
    share_chg_10d = load_precomputed_factor("SHARE_CHG_10D")
    margin_chg_10d = load_precomputed_factor("MARGIN_CHG_10D")

    # Align columns to common ETFs
    common_codes = sorted(
        set(close_panel.columns)
        .intersection(share_panel.columns)
    )
    common_dates = close_panel.index

    # ── Factor 1: SHARE_PRICE_DIV ──
    # Rolling correlation between share_chg_5d and price_return_5d, window=20
    # Per-ETF rolling correlation: negative = institutions moving against price
    print("  Computing SHARE_PRICE_DIV...")
    share_chg_5d = (share_panel[common_codes] - share_panel[common_codes].shift(5)) / (
        share_panel[common_codes].shift(5) + eps
    )
    ret_5d = close_panel[common_codes].pct_change(5)

    # Rolling correlation per ETF (element-wise)
    share_price_div = share_chg_5d.rolling(20, min_periods=10).corr(ret_5d)
    factors["SHARE_PRICE_DIV"] = share_price_div
    n_valid = share_price_div.notna().sum().sum()
    print(f"    Valid obs: {n_valid:,}, coverage: {len(common_codes)} ETFs")

    # ── Factor 2: FLOW_QUALITY ──
    # share_chg_10d / (vol_ratio_20d + eps)
    # Quiet accumulation: high share inflow + low volume activity = smart money
    print("  Computing FLOW_QUALITY...")
    # Align share_chg_10d to close_panel columns
    flow_codes = sorted(
        set(share_chg_10d.columns).intersection(close_panel.columns)
    )
    vol_ma20 = vol_panel[flow_codes].rolling(20, min_periods=10).mean()
    vol_ratio_20d = vol_panel[flow_codes] / (vol_ma20 + eps)

    # Align indices
    share_chg_aligned = share_chg_10d.reindex(index=common_dates, columns=flow_codes)
    vol_ratio_aligned = vol_ratio_20d.reindex(index=common_dates, columns=flow_codes)

    flow_quality = share_chg_aligned / (vol_ratio_aligned + eps)
    factors["FLOW_QUALITY"] = flow_quality
    n_valid = flow_quality.notna().sum().sum()
    print(f"    Valid obs: {n_valid:,}, coverage: {len(flow_codes)} ETFs")

    # ── Factor 3: LEVERAGE_FLOW ──
    # margin_chg_10d - share_chg_10d
    # Divergence: margin up + share down = leveraged conviction
    print("  Computing LEVERAGE_FLOW...")
    leverage_codes = sorted(
        set(margin_chg_10d.columns)
        .intersection(share_chg_10d.columns)
    )
    margin_aligned = margin_chg_10d.reindex(index=common_dates, columns=leverage_codes)
    share_aligned = share_chg_10d.reindex(index=common_dates, columns=leverage_codes)

    leverage_flow = margin_aligned - share_aligned
    factors["LEVERAGE_FLOW"] = leverage_flow
    n_valid = leverage_flow.notna().sum().sum()
    print(f"    Valid obs: {n_valid:,}, coverage: {len(leverage_codes)} ETFs")

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
# Rank autocorrelation
# ──────────────────────────────────────────────

def compute_rank_autocorr(factor_panel: pd.DataFrame, lag: int = 1) -> float:
    """Compute average cross-sectional rank autocorrelation at given lag.
    High autocorr = stable factor ranks = compatible with hysteresis."""
    dates = factor_panel.dropna(how="all").index
    if len(dates) < lag + 10:
        return np.nan

    autocorrs = []
    for i in range(lag, len(dates)):
        dt_now = dates[i]
        dt_prev = dates[i - lag]
        rank_now = factor_panel.loc[dt_now].rank(pct=True)
        rank_prev = factor_panel.loc[dt_prev].rank(pct=True)
        mask = rank_now.notna() & rank_prev.notna()
        if mask.sum() < 5:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = stats.spearmanr(rank_now[mask].values, rank_prev[mask].values)
        if not np.isnan(corr):
            autocorrs.append(corr)

    return np.mean(autocorrs) if autocorrs else np.nan


# ──────────────────────────────────────────────
# Orthogonality check
# ──────────────────────────────────────────────

def compute_ohlcv_factor_panels(close_panel, vol_panel, daily_data):
    """Compute OHLCV factors using the factor library for orthogonality checks."""
    import sys
    sys.path.insert(0, str(BASE_DIR / "src"))
    from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary

    # Build price dict
    codes = sorted(close_panel.columns)
    all_dates = close_panel.index

    high_panel = pd.DataFrame(index=all_dates, columns=codes, dtype=float)
    low_panel = pd.DataFrame(index=all_dates, columns=codes, dtype=float)

    for code, df in daily_data.items():
        if code in codes:
            high_panel.loc[df.index, code] = df["adj_high"].values if "adj_high" in df.columns else df["high"].values
            low_panel.loc[df.index, code] = df["adj_low"].values if "adj_low" in df.columns else df["low"].values

    high_panel = high_panel.ffill().fillna(1.0)
    low_panel = low_panel.ffill().fillna(1.0)

    prices = {
        "close": close_panel,
        "high": high_panel,
        "low": low_panel,
        "volume": vol_panel,
    }

    lib = PreciseFactorLibrary()
    all_factors_multi = lib.compute_all_factors(prices)

    # Extract per-factor panels
    ohlcv_panels = {}
    for factor_name in ACTIVE_OHLCV_FACTORS:
        if factor_name in all_factors_multi.columns.get_level_values(0):
            ohlcv_panels[factor_name] = all_factors_multi[factor_name]

    return ohlcv_panels


def compute_orthogonality(
    new_factor: pd.DataFrame,
    reference_panels: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Compute cross-sectional rank correlation between new factor and each reference factor.
    Returns {ref_name: avg_abs_rank_corr}."""
    results = {}
    common_dates = new_factor.dropna(how="all").index

    for ref_name, ref_panel in sorted(reference_panels.items()):
        shared_dates = common_dates.intersection(ref_panel.dropna(how="all").index)
        shared_codes = new_factor.columns.intersection(ref_panel.columns)

        if len(shared_dates) < 20 or len(shared_codes) < 5:
            results[ref_name] = np.nan
            continue

        corrs = []
        # Sample every 5th date for speed
        sample_dates = shared_dates[::5]
        for dt in sample_dates:
            f_new = new_factor.loc[dt, shared_codes]
            f_ref = ref_panel.loc[dt, shared_codes]
            mask = f_new.notna() & f_ref.notna()
            if mask.sum() < 5:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr, _ = stats.spearmanr(f_new[mask].values, f_ref[mask].values)
            if not np.isnan(corr):
                corrs.append(abs(corr))

        results[ref_name] = np.mean(corrs) if corrs else np.nan

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CROSS-SOURCE INTERACTION FACTOR IC SCREENING")
    print("=" * 70)

    # 1. Load daily data
    print("\n[1/7] Loading daily OHLCV data...")
    daily_data = load_daily()
    close_panel, vol_panel, trade_dates = build_panel(daily_data)
    etf_codes = list(close_panel.columns)
    print(f"  {len(etf_codes)} ETFs, {len(trade_dates)} trading days")
    print(f"  Date range: {trade_dates[0]} ~ {trade_dates[-1]}")

    # 2. Compute forward returns (5-day, matching FREQ=5)
    print("\n[2/7] Computing 5-day forward returns...")
    fwd_ret = close_panel.shift(-5) / close_panel - 1
    n_valid_fwd = fwd_ret.notna().sum().sum()
    print(f"  Valid forward return observations: {n_valid_fwd:,}")

    # 3. Load fund_share
    print("\n[3/7] Loading fund_share data...")
    share_panel = load_fund_share(trade_dates)
    print(f"  Fund share panel: {share_panel.shape[1]} ETFs, {share_panel.shape[0]} dates")

    # 4. Compute cross-source interaction factors
    print("\n[4/7] Computing 3 cross-source interaction factors...")
    cross_factors = compute_cross_source_factors(close_panel, vol_panel, share_panel)
    print(f"  Total: {len(cross_factors)} factors computed")

    # 5. IC screening
    print("\n[5/7] Running cross-sectional Spearman rank IC...")
    print("-" * 70)

    results = []
    ic_series_dict = {}
    for name in sorted(cross_factors.keys()):
        factor = cross_factors[name]
        ic_s = compute_ic_series(factor, fwd_ret)
        ic_series_dict[name] = ic_s
        summary = ic_summary(ic_s, TRAIN_END, HO_START)
        summary["factor"] = name
        results.append(summary)

        sign = "+" if summary["full_mean_IC"] > 0 else ""
        print(
            f"  {name:25s}  IC={sign}{summary['full_mean_IC']:.4f}  "
            f"IR={summary['full_IC_IR']:.3f}  "
            f"pos={summary['full_IC_pos_rate']:.1%}  "
            f"n={summary['full_n_days']}"
        )

    # 6. Rank autocorrelation
    print("\n[6/7] Computing rank autocorrelation (lag-1)...")
    for r in results:
        name = r["factor"]
        rank_ac = compute_rank_autocorr(cross_factors[name], lag=1)
        r["rank_autocorr"] = rank_ac
        print(f"  {name:25s}  rank_autocorr={rank_ac:.3f}")

    # 7. Orthogonality check
    print("\n[7/7] Computing orthogonality vs active factors...")
    print("  Loading OHLCV factor library...")
    ohlcv_panels = compute_ohlcv_factor_panels(close_panel, vol_panel, daily_data)
    print(f"  Computed {len(ohlcv_panels)} OHLCV factor panels")

    # Load non-OHLCV panels
    non_ohlcv_panels = {}
    for fname in ACTIVE_NON_OHLCV_FACTORS:
        try:
            non_ohlcv_panels[fname] = load_precomputed_factor(fname)
        except FileNotFoundError:
            print(f"  WARNING: {fname} not found, skipping")

    all_ref_panels = {}
    all_ref_panels.update(ohlcv_panels)
    all_ref_panels.update(non_ohlcv_panels)
    print(f"  Total reference factors: {len(all_ref_panels)}")

    for r in results:
        name = r["factor"]
        print(f"\n  {name}:")
        ortho = compute_orthogonality(cross_factors[name], all_ref_panels)
        # Find max correlation
        max_corr_name = max(ortho, key=lambda k: ortho[k] if not np.isnan(ortho.get(k, np.nan)) else -1)
        max_corr_val = ortho[max_corr_name]
        r["max_corr_vs_active"] = max_corr_val
        r["max_corr_factor"] = max_corr_name

        # Print top-5 correlated
        sorted_ortho = sorted(ortho.items(), key=lambda x: -x[1] if not np.isnan(x[1]) else -999)
        for ref_name, corr_val in sorted_ortho[:5]:
            print(f"    vs {ref_name:30s}  |rho|={corr_val:.3f}")

    # Build results DataFrame
    results_df = pd.DataFrame(results).set_index("factor")

    # Apply filter: |IC| > 0.03 AND HO |IC| >= 0.7 * Train |IC|
    results_df["ho_train_ratio"] = (
        results_df["ho_mean_IC"].abs() / results_df["train_mean_IC"].abs().replace(0, np.nan)
    )
    results_df["pass_ic_threshold"] = results_df["full_mean_IC"].abs() > 0.03
    results_df["pass_ho_stability"] = results_df["ho_train_ratio"] >= 0.7
    results_df["WINNER"] = results_df["pass_ic_threshold"] & results_df["pass_ho_stability"]

    # Print summary table
    print("\n" + "=" * 70)
    print("CROSS-SOURCE INTERACTION FACTOR IC RESULTS")
    print("=" * 70)

    cols_display = [
        "full_mean_IC", "full_IC_IR",
        "train_mean_IC", "ho_mean_IC",
        "ho_train_ratio",
        "rank_autocorr", "max_corr_vs_active", "max_corr_factor",
        "WINNER",
    ]

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results_df[cols_display].to_string())

    # Compact output table
    print("\n" + "=" * 70)
    print("COMPACT SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Factor':<25s} {'Full_IC':>8s} {'Full_IR':>8s} {'Train_IC':>9s} {'HO_IC':>8s} "
          f"{'HO/Tr':>7s} {'RankAC':>7s} {'MaxCorr':>8s} {'vs_Factor':<25s} {'WIN':>4s}")
    print("-" * 130)
    for _, row in results_df.iterrows():
        name = row.name
        sign = "+" if row["full_mean_IC"] > 0 else ""
        win_str = "YES" if row["WINNER"] else "no"
        print(
            f"{name:<25s} {sign}{row['full_mean_IC']:>7.4f} {row['full_IC_IR']:>8.3f} "
            f"{row['train_mean_IC']:>9.4f} {row['ho_mean_IC']:>8.4f} "
            f"{row['ho_train_ratio']:>7.2f} {row['rank_autocorr']:>7.3f} "
            f"{row['max_corr_vs_active']:>8.3f} {row['max_corr_factor']:<25s} "
            f"{win_str:>4s}"
        )

    # Winners
    winners = results_df[results_df["WINNER"]]
    print(f"\n{'=' * 70}")
    print(f"WINNERS ({len(winners)} / {len(results_df)} factors)")
    print(f"Criteria: |IC| > 0.03 AND HO |IC| >= 0.7 * Train |IC|")
    print(f"{'=' * 70}")
    if len(winners) > 0:
        for name in winners.index:
            row = winners.loc[name]
            sign = "+" if row["full_mean_IC"] > 0 else ""
            print(
                f"  {name:25s}  IC={sign}{row['full_mean_IC']:.4f}  "
                f"IR={row['full_IC_IR']:.3f}  "
                f"Train={row['train_mean_IC']:.4f}  "
                f"HO={row['ho_mean_IC']:.4f}  "
                f"HO/Tr={row['ho_train_ratio']:.2f}  "
                f"RankAC={row['rank_autocorr']:.3f}  "
                f"MaxCorr={row['max_corr_vs_active']:.3f} vs {row['max_corr_factor']}"
            )
    else:
        print("  No factors passed the filter criteria.")

    # Save outputs
    results_df.to_csv(OUT_DIR / "ic_results_cross_source.csv")
    print(f"\nResults saved to {OUT_DIR}/")

    # Save factor parquets for winners
    for name in cross_factors:
        factor_path = OUT_DIR / f"{name}.parquet"
        cross_factors[name].to_parquet(factor_path)
        print(f"  Saved {factor_path}")

    # Save IC time series
    ic_ts_df = pd.DataFrame(ic_series_dict)
    ic_ts_df.to_csv(OUT_DIR / "ic_timeseries_cross_source.csv")
    print(f"  IC time series saved.")

    return results_df


if __name__ == "__main__":
    results = main()
