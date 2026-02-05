import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
)

# Import the kernel directly
from scripts.batch_vec_backtest import vec_backtest_kernel

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("🚀 FULL SPACE V3 (VOLATILITY ADAPTIVE) BACKTEST")
    print("=" * 80)

    # 1. Load Configuration
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Parameters
    FREQ = 3
    POS_SIZE = 2
    INITIAL_CAPITAL = 1_000_000.0
    COMMISSION = 0.0002
    LOOKBACK = 252
    CASH_RATE_DAILY = 0.02 / 252

    # 2. Load Data
    print("\nLoading Data...")
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    # Use full date range for this run
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # 3. Compute Factors
    print("Computing Factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = sorted(
        raw_factors_df.columns.get_level_values(0).unique().tolist()
    )
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    # 4. Prepare 3D Factor Array
    print("Preparing Factor Matrix...")
    # Ensure all factors are aligned
    first_factor = std_factors[factor_names_list[0]]
    dates = first_factor.index
    tickers = first_factor.columns.tolist()
    T, N = first_factor.shape

    # Map factor name to index
    factor_map = {name: i for i, name in enumerate(factor_names_list)}

    # Stack: (Time, Assets, Factors)
    factors_3d = np.zeros((T, N, len(factor_names_list)), dtype=np.float64)
    for i, name in enumerate(factor_names_list):
        factors_3d[:, :, i] = std_factors[name].values

    # 5. Compute Volatility Regime & Exposure
    print("Computing Volatility Regime...")
    if "510300" in tickers:
        hs300 = ohlcv["close"]["510300"]
    else:
        print("Warning: 510300 not found, using first ticker as proxy.")
        hs300 = ohlcv["close"].iloc[:, 0]

    rets = hs300.pct_change()
    hv = rets.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
    hv_5d = hv.shift(5)
    regime_vol = (hv + hv_5d) / 2
    regime_vol = regime_vol.fillna(0.0)

    # Map to Exposure
    exposure = np.ones(T, dtype=np.float64)
    vol_vals = regime_vol.values

    # < 15%: 1.0
    # 15-25%: 1.0
    # 25-30%: 0.7
    mask_yellow = (vol_vals >= 25) & (vol_vals < 30)
    exposure[mask_yellow] = 0.7
    # 30-40%: 0.4
    mask_orange = (vol_vals >= 30) & (vol_vals < 40)
    exposure[mask_orange] = 0.4
    # > 40%: 0.1
    mask_red = vol_vals >= 40
    exposure[mask_red] = 0.1

    # 6. Prepare Timing
    print("Preparing Timing Signal...")
    timing_module = LightTimingModule()
    timing_signals = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr = shift_timing_signal(timing_signals)

    # Apply Exposure to Timing
    # timing_arr is (T,)
    # exposure is (T,)
    # Ensure alignment (dates match)
    # Both are derived from ohlcv which is aligned
    # timing_v3 = timing_arr * exposure # Deprecated: Passed separately to kernel

    # 7. Prepare Other Kernel Inputs
    close_arr = ohlcv["close"].values
    open_arr = ohlcv["open"].values
    high_arr = ohlcv["high"].values
    low_arr = ohlcv["low"].values

    rebalance_schedule = generate_rebalance_schedule(T, LOOKBACK, FREQ)

    # Dummy inputs for unused features
    atr_arr = np.zeros((T, N))
    individual_trend_arr = np.ones((T, N), dtype=bool)
    profit_ladder_thresholds = np.array([np.inf, np.inf, np.inf])
    profit_ladder_stops = np.array([0.0, 0.0, 0.0])
    profit_ladder_multipliers = np.array([0.0, 0.0, 0.0])

    # 8. Load Combinations
    print("Loading Combinations...")

    # DIAGNOSTICS
    print(f"Factors 3D shape: {factors_3d.shape}")
    print(f"Factors NaN count: {np.isnan(factors_3d).sum()}")
    print(f"Timing mean: {timing_arr.mean()}")
    print(f"Exposure mean: {exposure.mean()}")
    print(f"Close prices NaN count: {np.isnan(close_arr).sum()}")
    print(f"Rebalance schedule length: {len(rebalance_schedule)}")

    combos_df = pd.read_csv(
        ROOT / "results/ARCHIVE_unified_wfo_43etf_best/all_combos.csv"
    )
    # combos_df = combos_df.head(100) # Debug: Run first 100

    results = []

    print(f"Running Backtest for {len(combos_df)} strategies...")

    for idx, row in tqdm(combos_df.iterrows(), total=len(combos_df)):
        combo_str = row["combo"]
        factors = [f.strip() for f in combo_str.split("+")]

        # Get indices
        try:
            indices = np.array([factor_map[f] for f in factors], dtype=np.int64)
        except KeyError as e:
            print(f"Skipping {combo_str}: Factor {e} not found")
            continue

        # Run Kernel
        # Note: vec_backtest_kernel returns a tuple. The first element is equity_curve.
        kernel_res = vec_backtest_kernel(
            factors_3d,
            close_arr,
            open_arr,
            high_arr,
            low_arr,
            timing_arr,  # Original Timing
            exposure,  # Vol Regime
            indices,
            rebalance_schedule,
            POS_SIZE,
            INITIAL_CAPITAL,
            COMMISSION,
            0.20,
            20,
            False,  # Dynamic Leverage (Disabled)
            False,
            0.0,
            atr_arr,
            3.0,
            False,  # Stop Loss (Disabled)
            individual_trend_arr,
            False,  # Individual Trend (Disabled)
            profit_ladder_thresholds,
            profit_ladder_stops,
            profit_ladder_multipliers,  # Profit Ladder (Disabled)
            0.0,
            0.0,
            5,  # Circuit Breaker (Disabled)
            0,  # Cooldown
            1.0,  # Leverage Cap
        )

        equity_raw = kernel_res[0]

        # Fix equity_raw before start_day
        # Find first non-zero index
        non_zero_idx = np.argmax(equity_raw > 0)
        if non_zero_idx > 0:
            equity_raw[:non_zero_idx] = INITIAL_CAPITAL

        # Post-process Equity (Add Cash Return)
        equity_adj = np.zeros_like(equity_raw)
        equity_adj[0] = equity_raw[0]

        # Vectorized adjustment
        # r_raw = equity_raw[t] / equity_raw[t-1] - 1
        # r_adj = r_raw + (1 - timing_v3[t]) * CASH_RATE_DAILY
        # Note: timing_v3[t] is the exposure for day t

        # Avoid division by zero
        equity_raw_safe = equity_raw.copy()
        equity_raw_safe[equity_raw_safe == 0] = 1.0  # Should not happen after fix

        r_raw = np.zeros(T)
        r_raw[1:] = equity_raw[1:] / equity_raw_safe[:-1] - 1

        # Cash weight = 1 - timing_v3
        # We use timing_v3 directly as exposure
        # timing_v3 = timing_arr * exposure
        current_exposure = timing_arr * exposure
        cash_weight = 1.0 - current_exposure
        cash_ret = cash_weight * CASH_RATE_DAILY

        r_adj = r_raw + cash_ret

        # Rebuild equity
        equity_adj = np.cumprod(1 + r_adj) * INITIAL_CAPITAL
        # Fix first element
        equity_adj[0] = INITIAL_CAPITAL

        # Calculate Metrics
        # 1. Annual Return
        # Use last value
        total_ret = equity_adj[-1] / INITIAL_CAPITAL - 1
        ann_ret = (1 + total_ret) ** (252 / T) - 1

        # 2. Max Drawdown
        eq_s = pd.Series(equity_adj, index=dates)
        dd = (eq_s / eq_s.cummax() - 1).min()

        # 3. Sharpe
        daily_ret = eq_s.pct_change().fillna(0)
        sharpe = (
            daily_ret.mean() / daily_ret.std() * np.sqrt(252)
            if daily_ret.std() > 0
            else 0
        )

        # 4. Calmar
        calmar = ann_ret / abs(dd) if dd != 0 else 0

        # 5. Specific Years
        def get_year_ret(year):
            try:
                y_eq = eq_s[str(year)]
                if len(y_eq) > 0:
                    return y_eq.iloc[-1] / y_eq.iloc[0] - 1
                return np.nan
            except:
                return np.nan

        ret_2022 = get_year_ret(2022)
        ret_2024 = get_year_ret(2024)
        ret_2025 = get_year_ret(2025)

        results.append(
            {
                "combo": combo_str,
                "ann_return": ann_ret,
                "max_dd": dd,
                "sharpe": sharpe,
                "calmar": calmar,
                "ret_2022": ret_2022,
                "ret_2024": ret_2024,
                "ret_2025": ret_2025,
                "combo_size": len(factors),
            }
        )

    # Save Results
    res_df = pd.DataFrame(results)
    out_path = ROOT / "results/full_space_v3_results.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\n✅ Saved V3 results to {out_path}")

    # Print Top 5
    print("\n🏆 Top 5 Strategies (by Sharpe):")
    print(res_df.sort_values("sharpe", ascending=False).head(5).to_markdown())


if __name__ == "__main__":
    main()
