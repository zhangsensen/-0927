"""
Liquidity-Filtered Vectorized Backtest
Runs VEC backtest on a subset of liquid ETFs.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import logging
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.utils.rebalance import generate_rebalance_schedule
from batch_vec_backtest import run_vec_backtest

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- LIQUIDITY WHITELIST ---
# Hardcoded list of liquid ETFs (ADDV > 50M approx)
# Based on user suggestion + common knowledge of China ETF market
LIQUID_ETFS = [
    "510300",  # HS300
    "510500",  # ZZ500
    "510050",  # SZ50
    "513100",  # Nasdaq (QDII) - High Premium Risk but Liquid
    "513500",  # SP500 (QDII)
    "512480",  # Semiconductor
    "512760",  # Chip
    "515030",  # New Energy Car
    "512010",  # Pharma
    "512880",  # Securities
    "512660",  # Military
    "512690",  # Alcohol
    "515790",  # PV
    "512170",  # Medical
    "515000",  # Tech
    "588000",  # KC50
    "159915",  # CYB
    "159949",  # CYB50
    "518880",  # Gold
    "513050",  # KWEB (QDII)
    "513330",  # HSTECH (QDII)
]


def main():
    print("=" * 80)
    print("🚀 LIQUIDITY-FILTERED VECTORIZED BACKTEST")
    print("=" * 80)

    # 1. Load Configuration
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override with optimized parameters
    FREQ = 3
    POS_SIZE = 2

    # 2. Load Data
    print("\nLoading Data...")
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )

    # Load ALL first, then filter
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # Filter for Liquid ETFs
    # Check which ones exist in data
    available_liquid = [c for c in LIQUID_ETFS if c in ohlcv["close"].columns]
    print(f"Liquid ETFs found: {len(available_liquid)} / {len(LIQUID_ETFS)}")
    print(f"Missing: {set(LIQUID_ETFS) - set(available_liquid)}")

    if len(available_liquid) < 5:
        print("⚠️ Too few liquid ETFs found! Aborting.")
        return

    # Subset Data
    ohlcv_liquid = {k: v[available_liquid] for k, v in ohlcv.items()}

    # 3. Compute Factors
    print("Computing Factors on Liquid Subset...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv_liquid)
    factor_names_list = sorted(
        raw_factors_df.columns.get_level_values(0).unique().tolist()
    )
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    # 4. Prepare Backtest Data
    first_factor = std_factors[factor_names_list[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()

    # Pre-compute 3D factor array
    # Shape: (Time, Assets, Factors)
    # We need a mapping from factor name to index
    factor_name_to_idx = {name: i for i, name in enumerate(factor_names_list)}

    T, N = first_factor.shape
    F = len(factor_names_list)

    factors_3d = np.full((T, N, F), np.nan)
    for i, fname in enumerate(factor_names_list):
        factors_3d[:, :, i] = std_factors[fname].values

    # Prepare Price Arrays
    close_prices = ohlcv_liquid["close"].values
    open_prices = ohlcv_liquid["open"].values
    high_prices = ohlcv_liquid["high"].values
    low_prices = ohlcv_liquid["low"].values

    # Rebalance Schedule
    rebalance_schedule = generate_rebalance_schedule(len(dates), 252, FREQ)

    # 5. Load Combos to Test
    # We use the combos from the latest WFO run
    results_dir = ROOT / "results"
    wfo_dirs = sorted(
        [d for d in results_dir.glob("run_*") if d.is_dir() and not d.is_symlink()]
    )
    latest_wfo = wfo_dirs[-1]
    wfo_path = latest_wfo / "all_combos.parquet"
    wfo_df = pd.read_parquet(wfo_path)

    combos = wfo_df["combo"].unique().tolist()
    print(f"Testing {len(combos)} combos on {len(available_liquid)} ETFs...")

    # 6. Run Backtest Loop
    results = []

    # Dummy arrays for unused features
    timing_arr = np.ones(T)  # Scalar timing per day
    atr_arr = np.zeros((T, N))
    individual_trend_arr = np.ones((T, N), dtype=bool)

    # Profit Ladder (Disabled)
    profit_ladder_thresholds = np.array([np.inf, np.inf, np.inf])
    profit_ladder_stops = np.array([0.0, 0.0, 0.0])
    profit_ladder_multipliers = np.array([0.0, 0.0, 0.0])

    for combo_str in tqdm(combos):
        factors = [f.strip() for f in combo_str.split("+")]

        # Get indices
        try:
            f_indices = np.array(
                [factor_name_to_idx[f] for f in factors], dtype=np.int64
            )
        except KeyError:
            continue

        # Run Kernel
        # Note: run_vec_backtest is a wrapper, we can call it or call kernel directly.
        # Let's call run_vec_backtest from batch_vec_backtest.py if possible,
        # but it might expect different args.
        # Actually, let's use the kernel directly or the wrapper if it's exposed.
        # batch_vec_backtest.py has `vec_backtest_kernel`.

        from batch_vec_backtest import vec_backtest_kernel, stable_topk_indices

        # We need to slice factors_3d for this combo?
        # No, the kernel takes full factors_3d and factor_indices.
        # Wait, let's check kernel signature in batch_vec_backtest.py
        # def vec_backtest_kernel(factors_3d, ..., factor_indices, ...)
        # Yes.

        # But factors_3d in kernel expects (Time, Assets, Factors_in_Combo)?
        # Let's check batch_vec_backtest.py line 308:
        # factors_3d = np.stack([std_factors[f].values for f in factor_names_in_combo], axis=-1)
        # So it expects only the relevant factors.

        combo_factors_3d = factors_3d[:, :, f_indices]
        # Now factor_indices passed to kernel should be... wait.
        # If we pass combo_factors_3d, then factor_indices are just 0, 1, 2...
        # The kernel sums them up.

        # Actually, the kernel signature:
        # vec_backtest_kernel(factors_3d, ..., factor_indices, ...)
        # Inside kernel:
        # for k in range(K):
        #    f_idx = factor_indices[k]
        #    val = factors_3d[t, n, f_idx]

        # So if we pass the FULL factors_3d, we pass real indices.
        # If we pass sliced factors_3d, we pass 0..K-1.
        # Passing full factors_3d is more efficient if we don't want to allocate memory every loop.
        # But `factors_3d` is (T, N, F).
        # Let's pass full factors_3d and real indices.

        equity_curve, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = vec_backtest_kernel(
            factors_3d,  # Full
            close_prices,
            open_prices,
            high_prices,
            low_prices,
            timing_arr,
            np.ones(factors_3d.shape[0]),  # ✅ v3.1: Vol Regime (Default 1.0)
            f_indices,  # Real indices
            rebalance_schedule,
            POS_SIZE,
            1_000_000.0,  # Capital
            0.0002,  # Commission
            0.0,
            0,
            False,  # Vol target
            False,
            0.0,
            atr_arr,
            0.0,
            False,  # Stop loss
            individual_trend_arr,
            False,  # Trend filter
            profit_ladder_thresholds,
            profit_ladder_stops,
            profit_ladder_multipliers,
            0.0,
            0.0,
            0,  # Circuit breaker
            0,  # Cooldown
            1.0,  # Leverage cap
        )

        # Calculate Metrics
        # Filter zeros
        valid_equity = equity_curve[equity_curve > 0]
        if len(valid_equity) > 0:
            total_ret = (valid_equity[-1] / valid_equity[0]) - 1

            # Sharpe
            ret_series = pd.Series(valid_equity).pct_change().dropna()
            sharpe = (
                ret_series.mean() / ret_series.std() * np.sqrt(252)
                if ret_series.std() > 0
                else 0
            )

            # MaxDD
            peak = np.maximum.accumulate(valid_equity)
            dd = (peak - valid_equity) / peak
            max_dd = np.max(dd)

            # Recent Metrics (2025-01 to 2025-05)
            # Need to map dates to indices
            # dates is DatetimeIndex
            recent_mask = (dates >= pd.Timestamp("2025-01-01")) & (
                dates <= pd.Timestamp("2025-05-31")
            )
            # Get indices where mask is true
            # But equity_curve is aligned with dates? Yes.

            recent_equity = pd.Series(equity_curve, index=dates)[recent_mask]
            recent_equity = recent_equity[recent_equity > 0]

            if len(recent_equity) > 0:
                recent_ret = (recent_equity.iloc[-1] / recent_equity.iloc[0]) - 1
                r_peak = recent_equity.cummax()
                r_dd = (r_peak - recent_equity) / r_peak
                recent_mdd = r_dd.max()
            else:
                recent_ret = -999
                recent_mdd = 999

        else:
            total_ret = -1.0
            sharpe = 0.0
            max_dd = 1.0
            recent_ret = -999
            recent_mdd = 999

        results.append(
            {
                "combo": combo_str,
                "vec_return": total_ret,
                "vec_sharpe_ratio": sharpe,
                "vec_max_drawdown": max_dd,
                "Recent_Ret_5M": recent_ret,
                "Recent_MDD_5M": recent_mdd,
            }
        )

    # Save Results
    res_df = pd.DataFrame(results)
    out_dir = (
        ROOT / "results" / f"vec_liquid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "liquid_results.csv", index=False)
    print(f"Results saved to {out_dir}")

    # --- SELECTION & REPORTING ---
    print("\n--- LIQUIDITY-FILTERED SELECTION ---")

    # Filter: Recent_Ret > 0 and Recent_MDD < 0.08
    filtered = res_df[
        (res_df["Recent_Ret_5M"] > 0) & (res_df["Recent_MDD_5M"] < 0.08)
    ].copy()

    print(f"Total Combos: {len(res_df)}")
    print(f"Passed Filter: {len(filtered)}")

    if len(filtered) > 0:
        # Sort by Composite Score (Simplified: Rank(Ret) + Rank(Sharpe) + Rank(RecentRet))
        # Or just use Total Return for now as per user request "Top 5"
        # User asked for "Combo, 5Y_Return, Sharpe, Recent_Ret_5M"
        # Let's sort by Total Return for simplicity or Sharpe?
        # "Top 1 收益率相比之前的 75% 下降了多少" implies we look at Return.

        filtered = filtered.sort_values("vec_return", ascending=False)

        top5 = filtered.head(5)
        print("\nTop 5 Liquid Strategies:")
        print(
            top5[
                ["combo", "vec_return", "vec_sharpe_ratio", "Recent_Ret_5M"]
            ].to_markdown(index=False, floatfmt=".2%")
        )

        top1 = top5.iloc[0]
        print(f"\nTop 1 Return: {top1['vec_return']:.2%}")
        print(f"Previous Top 1 Return: 75.09%")
        print(f"Liquidity Premium Cost: {75.09 - top1['vec_return']*100:.2f} pp")

    else:
        print("No strategies passed the filter!")


if __name__ == "__main__":
    main()
