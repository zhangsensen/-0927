#!/usr/bin/env python3
"""
IC Screening for Non-OHLCV Candidate Factors

Computes ~15 candidate factors from fund_share, margin, fund_nav (premium),
and FX data, then runs cross-sectional Spearman rank IC analysis.

Output: results/ic_screen_non_ohlcv/
"""

import glob
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "raw" / "ETF"
OUT_DIR = BASE_DIR / "results" / "ic_screen_non_ohlcv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# QDII ETFs (monitored but not traded in A_SHARE_ONLY mode)
QDII_CODES = {"159920", "513050", "513100", "513130", "513180", "513400", "513500", "513520"}

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
        # e.g. "159801.SZ_daily_20200218_20260212.parquet"
        code = fname.split(".")[0]  # "159801"
        df = pd.read_parquet(f)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df.set_index("trade_date").sort_index()
        result[code] = df
    return result


def build_panel(daily_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """Build aligned close and volume panels. Index=trade_date (str), columns=ETF codes."""
    codes = sorted(daily_data.keys())
    # Union of all trade dates
    all_dates = sorted(set().union(*(df.index for df in daily_data.values())))
    idx = pd.Index(all_dates, name="trade_date")

    close_panel = pd.DataFrame(index=idx, columns=codes, dtype=float)
    vol_panel = pd.DataFrame(index=idx, columns=codes, dtype=float)

    for code, df in daily_data.items():
        close_panel.loc[df.index, code] = df["adj_close"].values
        vol_panel.loc[df.index, code] = df["vol"].values

    # ffill for late-IPO NaN, then fillna(1.0) for close (factor scores will be NaN anyway)
    close_panel = close_panel.ffill().fillna(1.0)
    vol_panel = vol_panel.ffill().fillna(0.0)
    return close_panel, vol_panel, idx


def load_fund_share(trade_dates: pd.Index) -> pd.DataFrame:
    """Load fund_share for all ETFs, forward-filled to daily calendar.
    Returns DataFrame with index=trade_date (str), columns=ETF codes, values=fd_share."""
    files = sorted(glob.glob(str(RAW_DIR / "fund_share" / "fund_share_*.parquet")))
    codes = sorted(set(trade_dates))  # not codes, just for ref
    all_codes = []
    series_list = []
    for f in files:
        fname = os.path.basename(f)
        # "fund_share_159801.parquet" -> code "159801"
        code = fname.replace("fund_share_", "").replace(".parquet", "")
        all_codes.append(code)
        df = pd.read_parquet(f)
        # trade_date is datetime64 -> convert to string YYYYMMDD
        df["trade_date"] = df["trade_date"].dt.strftime("%Y%m%d")
        df = df.drop_duplicates("trade_date", keep="last").set_index("trade_date").sort_index()
        series_list.append(df["fd_share"].rename(code))

    # Merge all into panel aligned to trade_dates
    panel = pd.DataFrame(index=trade_dates)
    for s in series_list:
        panel = panel.join(s, how="left")

    # Forward fill gaps (fund_share has gaps)
    panel = panel.ffill()
    return panel


def load_margin(trade_dates: pd.Index) -> dict[str, pd.DataFrame]:
    """Load margin data. Returns dict of column panels: {col_name: DataFrame(index=trade_date, columns=ts_code_short)}."""
    f = RAW_DIR / "margin" / "margin_pool43_2020_now.parquet"
    df = pd.read_parquet(f)
    # trade_date is string YYYYMMDD, ts_code like "510050.SH"
    # Extract short code: "510050"
    df["code"] = df["ts_code"].str.split(".").str[0]
    df = df.set_index("trade_date").sort_index()

    margin_cols = ["rzye", "rqye", "rzmre", "rzche"]
    panels = {}
    for col in margin_cols:
        pivot = df.pivot_table(index="trade_date", columns="code", values=col, aggfunc="last")
        # Reindex to trade_dates, ffill
        pivot = pivot.reindex(trade_dates).ffill()
        panels[col] = pivot
    return panels


def load_fund_nav(trade_dates: pd.Index) -> pd.DataFrame:
    """Load fund_nav for all ETFs, aligned by ann_date (T+1) to prevent lookahead.
    Returns DataFrame with index=trade_date (str), columns=ETF codes, values=unit_nav."""
    files = sorted(glob.glob(str(RAW_DIR / "fund_nav" / "fund_nav_*.parquet")))
    series_list = []
    codes = []
    for f in files:
        fname = os.path.basename(f)
        code = fname.replace("fund_nav_", "").replace(".parquet", "")
        codes.append(code)
        df = pd.read_parquet(f)

        # Use ann_date for alignment (publication date = trade_date + 1 business day)
        # This means the NAV known on ann_date was computed for trade_date
        # To prevent lookahead, we index by ann_date: the day the NAV becomes available
        if "ann_date" not in df.columns or df["ann_date"].isna().all():
            # Fallback: shift trade_date by 1 day
            df["trade_date"] = df["trade_date"].dt.strftime("%Y%m%d")
            df = df.drop_duplicates("trade_date", keep="last").set_index("trade_date").sort_index()
            s = df["unit_nav"].rename(code)
            # Shift by 1 day to approximate T+1
            s.index = s.index.to_series().shift(-1).ffill().values
        else:
            df["ann_date"] = df["ann_date"].astype(str)
            df = df.drop_duplicates("ann_date", keep="last").set_index("ann_date").sort_index()
            s = df["unit_nav"].rename(code)
        series_list.append(s)

    panel = pd.DataFrame(index=trade_dates)
    for s in series_list:
        panel = panel.join(s, how="left")
    panel = panel.ffill()
    return panel


def load_fx(trade_dates: pd.Index) -> pd.DataFrame:
    """Load USDCNH daily data. Returns Series indexed by trade_date (str)."""
    f = RAW_DIR / "fx" / "usdcnh_daily.parquet"
    df = pd.read_parquet(f)
    df = df.drop_duplicates("trade_date", keep="last").set_index("trade_date").sort_index()
    fx = df["bid_close"].reindex(trade_dates).ffill()
    return fx


# ──────────────────────────────────────────────
# Factor computation
# ──────────────────────────────────────────────

def compute_share_factors(share_panel: pd.DataFrame, vol_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute 6 fund_share factors."""
    factors = {}

    # SHARE_CHG_xD: pct change over x days
    for days in [5, 10, 20]:
        shifted = share_panel.shift(days)
        chg = (share_panel - shifted) / shifted.replace(0, np.nan)
        factors[f"SHARE_CHG_{days}D"] = chg

    # SHARE_ACCEL: SHARE_CHG_5D - SHARE_CHG_20D
    factors["SHARE_ACCEL"] = factors["SHARE_CHG_5D"] - factors["SHARE_CHG_20D"]

    # SHARE_TURNOVER: daily fd_share change / daily volume
    daily_chg = share_panel.diff(1).abs()
    vol_safe = vol_panel.replace(0, np.nan)
    factors["SHARE_TURNOVER"] = daily_chg / vol_safe

    # SHARE_RANK_CHG_10D: cross-sectional rank of SHARE_CHG_10D, then diff(10)
    rank_10d = factors["SHARE_CHG_10D"].rank(axis=1, pct=True)
    factors["SHARE_RANK_CHG_10D"] = rank_10d.diff(10)

    return factors


def compute_margin_factors(
    margin_panels: dict[str, pd.DataFrame],
    close_panel: pd.DataFrame,
    vol_panel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute 5 margin factors."""
    factors = {}
    rzye = margin_panels["rzye"]
    rqye = margin_panels["rqye"]
    rzmre = margin_panels["rzmre"]
    rzche = margin_panels["rzche"]

    # Only for codes present in margin data
    margin_codes = rzye.columns

    # MARGIN_BUY_RATIO: rzmre / (adj_close * vol) - proxy for margin buy intensity
    daily_amount = close_panel[close_panel.columns.intersection(margin_codes)] * vol_panel[vol_panel.columns.intersection(margin_codes)]
    daily_amount_safe = daily_amount.replace(0, np.nan)
    factors["MARGIN_BUY_RATIO"] = rzmre / daily_amount_safe

    # MARGIN_CHG_10D: pct change of rzye over 10 days
    rzye_shift = rzye.shift(10)
    factors["MARGIN_CHG_10D"] = (rzye - rzye_shift) / rzye_shift.replace(0, np.nan)

    # MARGIN_LONG_SHORT: rzye / rqye
    rqye_safe = rqye.replace(0, np.nan)
    factors["MARGIN_LONG_SHORT"] = rzye / rqye_safe

    # MARGIN_NET_FLOW: rzmre - rzche (daily net margin inflow)
    factors["MARGIN_NET_FLOW"] = rzmre - rzche

    # MARGIN_DENSITY: rzye / (adj_close * vol_ma20)
    vol_ma20 = vol_panel[vol_panel.columns.intersection(margin_codes)].rolling(20, min_periods=10).mean()
    denominator = close_panel[close_panel.columns.intersection(margin_codes)] * vol_ma20
    denominator_safe = denominator.replace(0, np.nan)
    factors["MARGIN_DENSITY"] = rzye / denominator_safe

    return factors


def compute_premium_factors(
    nav_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute 2 premium factors (QDII only where NAV != close)."""
    factors = {}

    # Only for QDII ETFs
    qdii_codes = sorted(QDII_CODES.intersection(set(close_panel.columns)).intersection(set(nav_panel.columns)))
    if not qdii_codes:
        print("WARNING: No QDII codes found for premium factors")
        return factors

    close_qdii = close_panel[qdii_codes]
    nav_qdii = nav_panel[qdii_codes]

    # Premium rate: (close - NAV) / NAV
    nav_safe = nav_qdii.replace(0, np.nan)
    premium = (close_qdii - nav_qdii) / nav_safe

    # PREMIUM_ZSCORE: z-score over 60-day rolling window
    roll_mean = premium.rolling(60, min_periods=20).mean()
    roll_std = premium.rolling(60, min_periods=20).std()
    roll_std_safe = roll_std.replace(0, np.nan)
    factors["PREMIUM_ZSCORE"] = (premium - roll_mean) / roll_std_safe

    # PREMIUM_MOMENTUM: 10-day slope of premium rate (linear regression slope)
    def rolling_slope(series, window=10):
        """Compute rolling OLS slope."""
        result = pd.Series(np.nan, index=series.index)
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        for i in range(window - 1, len(series)):
            y = series.iloc[i - window + 1 : i + 1].values
            if np.isnan(y).sum() > window // 2:
                continue
            y_clean = np.where(np.isnan(y), 0, y)
            y_mean = np.nanmean(y)
            slope = ((x - x_mean) * (y_clean - y_mean)).sum() / x_var
            result.iloc[i] = slope
        return result

    premium_mom = pd.DataFrame(index=premium.index, columns=qdii_codes, dtype=float)
    for code in qdii_codes:
        premium_mom[code] = rolling_slope(premium[code], window=10)
    factors["PREMIUM_MOMENTUM"] = premium_mom

    return factors


def compute_fx_factors(fx_series: pd.Series, trade_dates: pd.Index, etf_codes: list[str]) -> dict[str, pd.DataFrame]:
    """Compute 2 FX factors (broadcast to all ETFs)."""
    factors = {}

    # FX_MOMENTUM_10D: pct change of USDCNH over 10 days
    fx_shift = fx_series.shift(10)
    fx_mom = (fx_series - fx_shift) / fx_shift.replace(0, np.nan)

    # FX_VOL_20D: 20-day rolling std of daily returns
    fx_ret = fx_series.pct_change()
    fx_vol = fx_ret.rolling(20, min_periods=10).std()

    # Broadcast to all ETFs (same value across all ETFs for each day)
    factors["FX_MOMENTUM_10D"] = pd.DataFrame(
        {code: fx_mom for code in etf_codes}, index=trade_dates
    )
    factors["FX_VOL_20D"] = pd.DataFrame(
        {code: fx_vol for code in etf_codes}, index=trade_dates
    )

    return factors


# ──────────────────────────────────────────────
# IC computation
# ──────────────────────────────────────────────

def compute_ic_series(factor_panel: pd.DataFrame, fwd_ret_panel: pd.DataFrame) -> pd.Series:
    """Compute daily cross-sectional Spearman rank IC.
    Returns Series of IC values indexed by trade_date."""
    common_dates = factor_panel.index.intersection(fwd_ret_panel.index)
    common_codes = factor_panel.columns.intersection(fwd_ret_panel.columns)

    ic_values = {}
    for dt in common_dates:
        f_vals = factor_panel.loc[dt, common_codes]
        r_vals = fwd_ret_panel.loc[dt, common_codes]
        # Drop NaN
        mask = f_vals.notna() & r_vals.notna()
        n_valid = mask.sum()
        if n_valid < 5:  # Need at least 5 valid obs for meaningful correlation
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
    # Ensure index is string for comparison
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
    print("NON-OHLCV FACTOR IC SCREENING")
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

    # 3. Load and compute fund_share factors
    print("\n[3/7] Loading fund_share data and computing factors...")
    share_panel = load_fund_share(trade_dates)
    share_factors = compute_share_factors(share_panel, vol_panel)
    print(f"  Computed {len(share_factors)} fund_share factors: {sorted(share_factors.keys())}")

    # 4. Load and compute margin factors
    print("\n[4/7] Loading margin data and computing factors...")
    margin_panels = load_margin(trade_dates)
    margin_factors = compute_margin_factors(margin_panels, close_panel, vol_panel)
    print(f"  Computed {len(margin_factors)} margin factors: {sorted(margin_factors.keys())}")

    # 5. Load and compute premium factors (QDII only)
    print("\n[5/7] Loading fund_nav data and computing premium factors...")
    nav_panel = load_fund_nav(trade_dates)
    premium_factors = compute_premium_factors(nav_panel, close_panel)
    print(f"  Computed {len(premium_factors)} premium factors: {sorted(premium_factors.keys())}")
    print(f"  QDII codes used: {sorted(QDII_CODES.intersection(set(close_panel.columns)))}")

    # 6. Load and compute FX factors
    print("\n[6/7] Loading FX data and computing factors...")
    fx_series = load_fx(trade_dates)
    fx_factors = compute_fx_factors(fx_series, trade_dates, etf_codes)
    print(f"  Computed {len(fx_factors)} FX factors: {sorted(fx_factors.keys())}")

    # Combine all factors
    all_factors = {}
    all_factors.update(share_factors)
    all_factors.update(margin_factors)
    all_factors.update(premium_factors)
    all_factors.update(fx_factors)
    print(f"\n  Total: {len(all_factors)} factors")

    # 7. Run IC screening
    print("\n[7/7] Running cross-sectional Spearman rank IC...")
    print("-" * 70)

    results = []
    for name in sorted(all_factors.keys()):
        factor = all_factors[name]
        ic_series = compute_ic_series(factor, fwd_ret)
        summary = ic_summary(ic_series, TRAIN_END, HO_START)
        summary["factor"] = name
        results.append(summary)

        # Print progress
        sign = "+" if summary["full_mean_IC"] > 0 else ""
        print(
            f"  {name:30s}  IC={sign}{summary['full_mean_IC']:.4f}  "
            f"IR={summary['full_IC_IR']:.3f}  "
            f"pos={summary['full_IC_pos_rate']:.1%}  "
            f"n={summary['full_n_days']}"
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

    # Print summary table
    print("\n" + "=" * 70)
    print("IC SCREENING RESULTS SUMMARY")
    print("=" * 70)

    cols_display = [
        "full_mean_IC", "full_IC_IR", "full_IC_pos_rate", "full_n_days",
        "train_mean_IC", "train_IC_IR",
        "ho_mean_IC", "ho_IC_IR", "ho_n_days",
        "ho_train_ratio", "WINNER",
    ]

    pd.set_option("display.width", 200)
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
                f"  {name:30s}  IC={sign}{row['full_mean_IC']:.4f}  "
                f"IR={row['full_IC_IR']:.3f}  "
                f"Train_IC={row['train_mean_IC']:.4f}  "
                f"HO_IC={row['ho_mean_IC']:.4f}  "
                f"HO/Train={row['ho_train_ratio']:.2f}"
            )
    else:
        print("  No factors passed the filter criteria.")

    # Also show near-misses (pass one but not both)
    near_miss = results_df[
        (results_df["pass_ic_threshold"] | results_df["pass_ho_stability"])
        & ~results_df["WINNER"]
    ]
    if len(near_miss) > 0:
        print(f"\nNEAR MISSES ({len(near_miss)} factors):")
        for name in near_miss.index:
            row = near_miss.loc[name]
            print(
                f"  {name:30s}  IC={row['full_mean_IC']:.4f}  "
                f"IR={row['full_IC_IR']:.3f}  "
                f"ic_ok={'Y' if row['pass_ic_threshold'] else 'N'}  "
                f"ho_ok={'Y' if row['pass_ho_stability'] else 'N'}  "
                f"HO/Train={row['ho_train_ratio']:.2f}"
            )

    # Save outputs
    results_df.to_csv(OUT_DIR / "ic_results_all.csv")
    if len(winners) > 0:
        winners.to_csv(OUT_DIR / "ic_winners.csv")
    print(f"\nResults saved to {OUT_DIR}/")

    # Save IC time series for winners (useful for downstream analysis)
    if len(winners) > 0:
        ic_ts_dict = {}
        for name in winners.index:
            ic_ts_dict[name] = compute_ic_series(all_factors[name], fwd_ret)
        ic_ts_df = pd.DataFrame(ic_ts_dict)
        ic_ts_df.to_csv(OUT_DIR / "ic_timeseries_winners.csv")
        print(f"IC time series for {len(winners)} winners saved.")

    return results_df


if __name__ == "__main__":
    results = main()
