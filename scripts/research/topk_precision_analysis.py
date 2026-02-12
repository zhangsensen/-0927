#!/usr/bin/env python3
"""
Top-K Precision Metrics for Factor Evaluation

With POS_SIZE=2, we only select the top-2 ETFs. Full-ranking Spearman IC
penalizes errors in the bottom 40 ETFs which are irrelevant to portfolio
construction. This script computes decision-relevant metrics:

  1. TOP_2_PRECISION: |{factor_top2} ∩ {actual_top2}| / 2
  2. TOP_2_RECALL:    same as precision for K=2
  3. TOP_5_HIT_RATE:  |{factor_top2} ∩ {actual_top5}| > 0 (binary)
  4. BOUNDARY_IC:     Spearman IC on factor's top-10 only
  5. SPEARMAN_IC:     Standard full cross-section Spearman IC

Output: results/topk_precision/
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "results" / "topk_precision"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2025-04-30")
HO_START = pd.Timestamp("2025-05-01")

# 24 active factors from config
OHLCV_FACTORS = [
    "ADX_14D",
    "AMIHUD_ILLIQUIDITY",
    "BREAKOUT_20D",
    "CALMAR_RATIO_60D",
    "CORRELATION_TO_MARKET_20D",
    "GK_VOL_RATIO_20D",
    "MAX_DD_60D",
    "MOM_20D",
    "OBV_SLOPE_10D",
    "PRICE_POSITION_20D",
    "PRICE_POSITION_120D",
    "PV_CORR_20D",
    "SHARPE_RATIO_20D",
    "SLOPE_20D",
    "UP_DOWN_VOL_RATIO_20D",
    "VOL_RATIO_20D",
    "VORTEX_14D",
]

NON_OHLCV_FACTORS = [
    "SHARE_CHG_5D",
    "SHARE_CHG_10D",
    "SHARE_CHG_20D",
    "SHARE_ACCEL",
    "MARGIN_CHG_10D",
    "MARGIN_BUY_RATIO",
]


def load_factor_directions() -> dict:
    """Load factor direction metadata from PreciseFactorLibrary."""
    from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary

    lib = PreciseFactorLibrary()
    directions = {}
    for name in OHLCV_FACTORS + NON_OHLCV_FACTORS:
        meta = lib.get_metadata(name)
        if meta is not None:
            directions[name] = meta.direction
        else:
            # Default: assume high_is_good unless name suggests otherwise
            directions[name] = "high_is_good"
    return directions


def load_ohlcv_data(config: dict) -> dict:
    """Load OHLCV price data via DataLoader."""
    from etf_strategy.core.data_loader import DataLoader

    loader = DataLoader(
        data_dir=config["data"]["data_dir"],
        cache_dir=config["data"]["cache_dir"],
    )
    prices = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    return prices


def compute_ohlcv_factors(prices: dict) -> dict:
    """Compute all OHLCV factors using PreciseFactorLibrary."""
    from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary

    lib = PreciseFactorLibrary()
    all_factors_df = lib.compute_all_factors(prices)

    # Extract individual factor DataFrames from multi-level column index
    factors = {}
    available_factors = all_factors_df.columns.get_level_values(0).unique()
    for name in OHLCV_FACTORS:
        if name in available_factors:
            factors[name] = all_factors_df[name]
        else:
            print(f"  WARNING: OHLCV factor {name} not found in library output")
    return factors


def load_non_ohlcv_factors() -> dict:
    """Load pre-computed non-OHLCV factors from parquet files."""
    factors_dir = BASE_DIR / "results" / "non_ohlcv_factors"
    factors = {}
    for name in NON_OHLCV_FACTORS:
        fpath = factors_dir / f"{name}.parquet"
        if fpath.exists():
            df = pd.read_parquet(fpath)
            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, format="%Y%m%d")
            factors[name] = df
        else:
            print(f"  WARNING: Non-OHLCV factor {name} not found at {fpath}")
    return factors


def standardize_factors(raw_factors: dict) -> dict:
    """Cross-section standardize all factors using CrossSectionProcessor."""
    from etf_strategy.core.cross_section_processor import CrossSectionProcessor

    processor = CrossSectionProcessor(verbose=False)
    return processor.process_all_factors(raw_factors)


def compute_forward_returns(close: pd.DataFrame, freq: int = 5) -> pd.DataFrame:
    """Compute forward returns: close[t+freq] / close[t] - 1."""
    return close.shift(-freq) / close - 1


def compute_topk_metrics(
    factor_panel: pd.DataFrame,
    fwd_ret_panel: pd.DataFrame,
    direction: str,
    k_precision: int = 2,
    k_hit: int = 5,
    k_boundary: int = 10,
) -> dict:
    """Compute Top-K precision metrics for a single factor.

    Parameters
    ----------
    factor_panel : DataFrame (date x symbol), standardized factor values
    fwd_ret_panel : DataFrame (date x symbol), forward returns
    direction : 'high_is_good', 'low_is_good', or 'neutral'
    k_precision : K for top-K precision/recall
    k_hit : K for hit rate (actual top-K)
    k_boundary : K for boundary IC (factor's top-K)

    Returns
    -------
    dict with per-date metrics lists and summary stats
    """
    common_dates = factor_panel.index.intersection(fwd_ret_panel.index)
    common_cols = factor_panel.columns.intersection(fwd_ret_panel.columns)

    # Storage
    dates_used = []
    top2_prec_list = []
    top5_hit_list = []
    boundary_ic_list = []
    full_ic_list = []

    for dt in common_dates:
        f_vals = factor_panel.loc[dt, common_cols]
        r_vals = fwd_ret_panel.loc[dt, common_cols]

        # Drop NaN from both
        mask = f_vals.notna() & r_vals.notna()
        n_valid = mask.sum()
        # Need enough ETFs for meaningful metrics
        if n_valid < max(k_boundary, 10):
            continue

        f_clean = f_vals[mask]
        r_clean = r_vals[mask]

        # Determine ranking direction
        # For "high_is_good": higher factor value = buy → rank descending
        # For "low_is_good": lower factor value = buy → rank ascending
        # For "neutral": use absolute IC to decide
        if direction == "high_is_good":
            factor_ascending = False
        elif direction == "low_is_good":
            factor_ascending = True
        else:
            # neutral: compute IC sign to decide
            ic_sign, _ = stats.spearmanr(f_clean.values, r_clean.values)
            factor_ascending = ic_sign < 0  # if negative IC, low factor = good return

        # Factor's top-K (best candidates according to factor)
        if factor_ascending:
            factor_top_k_idx = f_clean.nsmallest(k_precision).index
            factor_top_boundary_idx = f_clean.nsmallest(k_boundary).index
        else:
            factor_top_k_idx = f_clean.nlargest(k_precision).index
            factor_top_boundary_idx = f_clean.nlargest(k_boundary).index

        # Actual top-K (best performers by forward return)
        actual_top_k = r_clean.nlargest(k_precision).index
        actual_top_hit = r_clean.nlargest(k_hit).index

        # 1. TOP_2_PRECISION: |factor_top2 ∩ actual_top2| / 2
        overlap = len(set(factor_top_k_idx) & set(actual_top_k))
        top2_prec_list.append(overlap / k_precision)

        # 2. TOP_5_HIT_RATE: at least 1 of factor_top2 in actual_top5
        hit = len(set(factor_top_k_idx) & set(actual_top_hit)) > 0
        top5_hit_list.append(float(hit))

        # 3. BOUNDARY_IC: Spearman IC on factor's top-K boundary
        f_boundary = f_clean.loc[factor_top_boundary_idx]
        r_boundary = r_clean.loc[factor_top_boundary_idx]
        if len(f_boundary) >= 5:
            bic, _ = stats.spearmanr(f_boundary.values, r_boundary.values)
            if not np.isnan(bic):
                boundary_ic_list.append(bic)

        # 4. FULL SPEARMAN IC
        fic, _ = stats.spearmanr(f_clean.values, r_clean.values)
        if not np.isnan(fic):
            full_ic_list.append(fic)

        dates_used.append(dt)

    return {
        "dates": dates_used,
        "top2_precision": top2_prec_list,
        "top5_hit_rate": top5_hit_list,
        "boundary_ic": boundary_ic_list,
        "full_ic": full_ic_list,
    }


def summarize_metrics(metrics: dict, train_end: pd.Timestamp, ho_start: pd.Timestamp) -> dict:
    """Summarize metrics into train/HO/full periods."""
    dates = pd.DatetimeIndex(metrics["dates"])

    def _split_and_mean(values, dates_idx):
        s = pd.Series(values, index=dates_idx)
        train_s = s[s.index <= train_end]
        ho_s = s[s.index >= ho_start]
        return {
            "full": s.mean() if len(s) > 0 else np.nan,
            "train": train_s.mean() if len(train_s) > 0 else np.nan,
            "ho": ho_s.mean() if len(ho_s) > 0 else np.nan,
            "n_full": len(s),
            "n_train": len(train_s),
            "n_ho": len(ho_s),
        }

    # top2_precision and top5_hit_rate use same dates
    prec = _split_and_mean(metrics["top2_precision"], dates)
    hit = _split_and_mean(metrics["top5_hit_rate"], dates)

    # boundary_ic and full_ic may have slightly different counts (NaN filtering)
    bic_dates = dates[: len(metrics["boundary_ic"])]
    fic_dates = dates[: len(metrics["full_ic"])]

    # Actually boundary_ic and full_ic may differ in count from top2/top5
    # Use the same dates but note that boundary_ic list may be shorter
    # We need to properly align - the safest is to use the dates_used
    # and filter boundary_ic/full_ic separately

    # Rebuild with proper indexing
    bic_vals = metrics["boundary_ic"]
    fic_vals = metrics["full_ic"]

    # These are aligned with dates_used (same length as top2_precision)
    # except boundary_ic may skip some days. Let's use a simpler approach.
    bic_s = pd.Series(bic_vals)
    fic_s = pd.Series(fic_vals)

    # For boundary_ic, we need the actual date alignment.
    # Since compute_topk_metrics appends boundary_ic only when len>=5 and not NaN,
    # it could be shorter. Let's fix this by using a different approach.
    # We'll just compute overall stats for boundary_ic and full_ic.

    # Actually, let's rebuild: full_ic is always appended when not NaN.
    # boundary_ic is appended when len(boundary)>=5 and not NaN.
    # Both may have fewer entries than dates_used.
    # For simplicity, compute bulk stats.

    # For full_ic, rebuild with dates
    # Actually, let me just compute mean IC for train/HO separately
    # by using the full_ic aligned with dates

    # The safest approach: full_ic and boundary_ic track which dates they belong to
    # But our current implementation doesn't track that separately.
    # Let's approximate: assume full_ic entries correspond to dates_used in order
    # (since both skip NaN dates in the same loop, full_ic should match 1:1)

    # full_ic: appended when not NaN, but could skip some dates
    # For a robust solution, let's just compute overall stats
    full_ic_mean = np.mean(fic_vals) if len(fic_vals) > 0 else np.nan
    full_ic_ir = np.mean(fic_vals) / np.std(fic_vals) if len(fic_vals) > 1 and np.std(fic_vals) > 1e-10 else np.nan
    boundary_ic_mean = np.mean(bic_vals) if len(bic_vals) > 0 else np.nan
    abs_full_ic = abs(full_ic_mean) if not np.isnan(full_ic_mean) else np.nan

    return {
        "top2_prec_full": prec["full"],
        "top2_prec_train": prec["train"],
        "top2_prec_ho": prec["ho"],
        "top5_hit_full": hit["full"],
        "top5_hit_train": hit["train"],
        "top5_hit_ho": hit["ho"],
        "boundary_ic_full": boundary_ic_mean,
        "full_ic_mean": full_ic_mean,
        "full_ic_ir": full_ic_ir,
        "abs_full_ic": abs_full_ic,
        "n_days_full": prec["n_full"],
        "n_days_train": prec["n_train"],
        "n_days_ho": prec["n_ho"],
    }


def main():
    import yaml

    print("=" * 80)
    print("TOP-K PRECISION METRICS FOR FACTOR EVALUATION")
    print("=" * 80)

    # Load config
    config_path = BASE_DIR / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 1. Load factor directions
    print("\n[1/6] Loading factor direction metadata...")
    directions = load_factor_directions()
    for name, d in sorted(directions.items()):
        print(f"  {name:35s} {d}")

    # 2. Load OHLCV data and compute OHLCV factors
    print("\n[2/6] Loading OHLCV data and computing factors...")
    prices = load_ohlcv_data(config)
    close = prices["close"]
    # ffill + fillna for late-IPO NaN (same as production pipeline)
    close_filled = close.ffill().fillna(1.0)
    print(f"  {len(close.columns)} ETFs, {len(close.index)} trading days")
    print(f"  Date range: {close.index[0].strftime('%Y-%m-%d')} ~ {close.index[-1].strftime('%Y-%m-%d')}")

    ohlcv_factors = compute_ohlcv_factors(prices)
    print(f"  Computed {len(ohlcv_factors)} OHLCV factors")

    # 3. Load non-OHLCV factors
    print("\n[3/6] Loading non-OHLCV factors...")
    non_ohlcv_factors = load_non_ohlcv_factors()
    print(f"  Loaded {len(non_ohlcv_factors)} non-OHLCV factors")

    # Combine all raw factors
    all_raw_factors = {}
    all_raw_factors.update(ohlcv_factors)
    all_raw_factors.update(non_ohlcv_factors)

    # Align non-OHLCV factors to same index/columns as OHLCV
    ref_index = close.index
    ref_cols = close.columns
    for name in list(all_raw_factors.keys()):
        df = all_raw_factors[name]
        # Reindex to common dates and columns
        common_cols = df.columns.intersection(ref_cols)
        df = df.reindex(index=ref_index, columns=ref_cols)
        all_raw_factors[name] = df

    print(f"  Total: {len(all_raw_factors)} factors")

    # 4. Cross-section standardize
    print("\n[4/6] Cross-section standardizing all factors...")
    std_factors = standardize_factors(all_raw_factors)
    print(f"  Standardized {len(std_factors)} factors")

    # 5. Compute forward returns
    print("\n[5/6] Computing 5-day forward returns...")
    fwd_ret = compute_forward_returns(close_filled, freq=5)
    n_valid = fwd_ret.notna().sum().sum()
    print(f"  Valid forward return observations: {n_valid:,}")

    # 6. Compute Top-K metrics for each factor
    print("\n[6/6] Computing Top-K precision metrics...")
    print("-" * 80)

    results = []
    for name in sorted(std_factors.keys()):
        factor_panel = std_factors[name]
        direction = directions.get(name, "high_is_good")

        metrics = compute_topk_metrics(
            factor_panel=factor_panel,
            fwd_ret_panel=fwd_ret,
            direction=direction,
            k_precision=2,
            k_hit=5,
            k_boundary=10,
        )

        summary = summarize_metrics(metrics, TRAIN_END, HO_START)
        summary["factor"] = name
        summary["direction"] = direction
        summary["is_non_ohlcv"] = name in NON_OHLCV_FACTORS
        results.append(summary)

        print(
            f"  {name:35s}  "
            f"Top2P={summary['top2_prec_ho']:.3f}  "
            f"Top5H={summary['top5_hit_ho']:.3f}  "
            f"BndIC={summary['boundary_ic_full']:.4f}  "
            f"IC={summary['full_ic_mean']:+.4f}  "
            f"n={summary['n_days_full']}"
        )

    # Build results DataFrame
    results_df = pd.DataFrame(results).set_index("factor")

    # Rank by HO Top2 Precision descending
    results_df = results_df.sort_values("top2_prec_ho", ascending=False)

    # Compute IC rank and Top2 Precision rank
    results_df["ic_rank"] = results_df["abs_full_ic"].rank(ascending=False).astype(int)
    results_df["top2_prec_ho_rank"] = results_df["top2_prec_ho"].rank(ascending=False).astype(int)
    results_df["rank_delta"] = results_df["ic_rank"] - results_df["top2_prec_ho_rank"]

    # Print main results table
    print("\n" + "=" * 80)
    print("RESULTS: SORTED BY HO TOP-2 PRECISION")
    print("=" * 80)

    display_cols = [
        "direction",
        "is_non_ohlcv",
        "top2_prec_train",
        "top2_prec_ho",
        "top5_hit_train",
        "top5_hit_ho",
        "boundary_ic_full",
        "full_ic_mean",
        "full_ic_ir",
        "ic_rank",
        "top2_prec_ho_rank",
        "rank_delta",
        "n_days_ho",
    ]

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 25)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results_df[display_cols].to_string())

    # Identify "rescued" factors: Top2 precision rank >= 5 positions better than IC rank
    print("\n" + "=" * 80)
    print("RESCUED FACTORS: Top2_Precision_Rank >= 5 positions better than IC_Rank")
    print("(rank_delta >= 5 means factor is much better for top-2 selection than IC suggests)")
    print("=" * 80)

    rescued = results_df[results_df["rank_delta"] >= 5]
    if len(rescued) > 0:
        for name in rescued.index:
            row = rescued.loc[name]
            marker = " [NON-OHLCV]" if row["is_non_ohlcv"] else ""
            print(
                f"  {name:35s}{marker}  "
                f"IC_Rank={row['ic_rank']:.0f} -> Top2P_Rank={row['top2_prec_ho_rank']:.0f}  "
                f"(+{row['rank_delta']:.0f} positions)  "
                f"HO_Top2P={row['top2_prec_ho']:.3f}  IC={row['full_ic_mean']:+.4f}"
            )
    else:
        print("  No rescued factors found.")

    # Demoted factors (good IC but poor top-2 precision)
    print("\n" + "=" * 80)
    print("DEMOTED FACTORS: IC_Rank >= 5 positions better than Top2_Precision_Rank")
    print("(rank_delta <= -5 means factor's IC overstates its top-2 selection ability)")
    print("=" * 80)

    demoted = results_df[results_df["rank_delta"] <= -5]
    if len(demoted) > 0:
        for name in demoted.index:
            row = demoted.loc[name]
            marker = " [NON-OHLCV]" if row["is_non_ohlcv"] else ""
            print(
                f"  {name:35s}{marker}  "
                f"IC_Rank={row['ic_rank']:.0f} -> Top2P_Rank={row['top2_prec_ho_rank']:.0f}  "
                f"({row['rank_delta']:.0f} positions)  "
                f"HO_Top2P={row['top2_prec_ho']:.3f}  IC={row['full_ic_mean']:+.4f}"
            )
    else:
        print("  No demoted factors found.")

    # Non-OHLCV factor focus
    print("\n" + "=" * 80)
    print("NON-OHLCV FACTOR FOCUS")
    print("=" * 80)

    non_ohlcv_df = results_df[results_df["is_non_ohlcv"]]
    if len(non_ohlcv_df) > 0:
        focus_cols = [
            "top2_prec_train",
            "top2_prec_ho",
            "top5_hit_ho",
            "boundary_ic_full",
            "full_ic_mean",
            "ic_rank",
            "top2_prec_ho_rank",
            "rank_delta",
        ]
        print(non_ohlcv_df[focus_cols].to_string())
    else:
        print("  No non-OHLCV factors available.")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Random baseline for Top-2 precision with N ETFs
    n_etfs = len(close.columns)
    random_top2_prec = 2 / n_etfs  # E[|random_top2 ∩ actual_top2|] / 2
    random_top5_hit = 1 - (1 - 5 / n_etfs) ** 2  # P(at least 1 hit)
    print(f"  Random baseline (N={n_etfs} ETFs):")
    print(f"    Top-2 Precision: {random_top2_prec:.4f}")
    print(f"    Top-5 Hit Rate:  {random_top5_hit:.4f}")

    ho_results = results_df.sort_values("top2_prec_ho", ascending=False)
    print(f"\n  Best HO Top-2 Precision:")
    for i, (name, row) in enumerate(ho_results.head(5).iterrows()):
        marker = " *" if row["is_non_ohlcv"] else ""
        print(f"    {i+1}. {name:35s} {row['top2_prec_ho']:.4f}{marker}")

    print(f"\n  Best HO Top-5 Hit Rate:")
    ho_hit = results_df.sort_values("top5_hit_ho", ascending=False)
    for i, (name, row) in enumerate(ho_hit.head(5).iterrows()):
        marker = " *" if row["is_non_ohlcv"] else ""
        print(f"    {i+1}. {name:35s} {row['top5_hit_ho']:.4f}{marker}")

    # IC vs Top-2 Precision correlation
    ic_top2_corr = results_df[["abs_full_ic", "top2_prec_ho"]].dropna().corr().iloc[0, 1]
    print(f"\n  Correlation between |IC| and HO Top-2 Precision: {ic_top2_corr:.3f}")

    recommendation = (
        "STRONG" if ic_top2_corr < 0.5 else "MODERATE" if ic_top2_corr < 0.75 else "WEAK"
    )
    print(f"  Recommendation: {recommendation} case for replacing IC with Top-K metrics")
    if ic_top2_corr < 0.5:
        print("  -> IC and Top-K rankings diverge significantly. Top-K adds real information.")
    elif ic_top2_corr < 0.75:
        print("  -> IC and Top-K are moderately correlated. Top-K provides some additional signal.")
    else:
        print("  -> IC and Top-K rankings are highly aligned. Top-K adds limited incremental value.")

    # Save results
    results_df.to_csv(OUT_DIR / "topk_metrics_all.csv")
    if len(rescued) > 0:
        rescued.to_csv(OUT_DIR / "rescued_factors.csv")
    if len(demoted) > 0:
        demoted.to_csv(OUT_DIR / "demoted_factors.csv")
    print(f"\n  Results saved to {OUT_DIR}/")

    return results_df


if __name__ == "__main__":
    results = main()
