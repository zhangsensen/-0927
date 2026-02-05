"""
Multi-Strategy Robustness Validation
Systematic validation of all liquid strategies that passed the initial filter.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import logging
from numba import njit
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.utils.rebalance import (
    generate_rebalance_schedule,
    shift_timing_signal,
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Numba Kernels (Copied & Modified) ---


@njit(cache=True)
def stable_topk_indices(scores, k):
    """Stable Top-K Indices"""
    N = len(scores)
    result = np.empty(k, dtype=np.int64)
    used = np.zeros(N, dtype=np.bool_)

    for i in range(k):
        best_idx = -1
        best_score = -np.inf
        for n in range(N):
            if used[n]:
                continue
            if scores[n] > best_score or (
                scores[n] == best_score and (best_idx < 0 or n < best_idx)
            ):
                best_score = scores[n]
                best_idx = n
        if best_idx < 0 or best_score == -np.inf:
            return result[:i]
        result[i] = best_idx
        used[best_idx] = True
    return result


@njit(cache=True)
def validation_backtest_kernel(
    factors_3d,
    close_prices,
    open_prices,
    high_prices,
    low_prices,
    timing_arr,
    factor_indices,
    rebalance_schedule,
    pos_size,
    initial_capital,
    commission_rate,
    target_vol,
    vol_window,
    dynamic_leverage_enabled,
    use_atr_stop,
    trailing_stop_pct,
    atr_arr,
    atr_multiplier,
    stop_on_rebalance_only,
    individual_trend_arr,
    individual_trend_enabled,
    profit_ladder_thresholds,
    profit_ladder_stops,
    profit_ladder_multipliers,
    circuit_breaker_day,
    circuit_breaker_total,
    circuit_recovery_days,
    cooldown_days,
    leverage_cap,
):
    T, N, _ = factors_3d.shape

    cash = initial_capital
    holdings = np.full(N, -1.0)
    entry_prices = np.zeros(N)
    high_water_marks = np.zeros(N)

    current_stop_pcts = np.full(N, trailing_stop_pct)
    current_atr_mults = np.full(N, atr_multiplier)

    circuit_breaker_active = False
    circuit_breaker_countdown = 0
    cooldown_remaining = np.zeros(N, dtype=np.int64)

    wins = 0
    losses = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0

    combined_score = np.empty(N)
    target_set = np.zeros(N, dtype=np.bool_)
    buy_order = np.empty(pos_size, dtype=np.int64)
    new_targets = np.empty(pos_size, dtype=np.int64)
    target_value_total = 0.0
    filled_value_total = 0.0
    target_shares_total = 0.0
    filled_shares_total = 0.0

    equity_curve = np.zeros(T)
    holdings_history = np.zeros(
        (T, N), dtype=np.float64
    )  # ✅ NEW: Track holdings history

    peak_equity = initial_capital
    max_drawdown = 0.0
    prev_equity = initial_capital
    rebal_ptr = 0

    returns_buffer = np.zeros(vol_window)
    buffer_ptr = 0
    buffer_filled = 0
    current_leverage = 1.0
    leverage_sum = 0.0
    leverage_count = 0

    welford_mean = 0.0
    welford_m2 = 0.0
    welford_count = 0

    start_day = rebalance_schedule[0] if len(rebalance_schedule) > 0 else 252

    for t in range(start_day, T):
        # 1. Update Equity (Open)
        current_equity = cash
        for n in range(N):
            if holdings[n] > 0.0:
                price = close_prices[t - 1, n]
                if not np.isnan(price):
                    current_equity += holdings[n] * price

        equity_curve[t] = current_equity

        if t > start_day:
            daily_return = (
                (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            )
            welford_count += 1
            delta = daily_return - welford_mean
            welford_mean += delta / welford_count
            delta2 = daily_return - welford_mean
            welford_m2 += delta * delta2

            if dynamic_leverage_enabled:
                returns_buffer[buffer_ptr] = daily_return
                buffer_ptr = (buffer_ptr + 1) % vol_window
                if buffer_filled < vol_window:
                    buffer_filled += 1

        if current_equity > peak_equity:
            peak_equity = current_equity
        current_dd = (
            (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
        )
        if current_dd > max_drawdown:
            max_drawdown = current_dd

        prev_equity = current_equity

        is_rebalance_day = False
        if rebal_ptr < len(rebalance_schedule) and rebalance_schedule[rebal_ptr] == t:
            is_rebalance_day = True
            rebal_ptr += 1

        # Circuit Breaker Logic
        if circuit_breaker_day > 0.0 or circuit_breaker_total > 0.0:
            if t > start_day:
                day_return = (
                    (current_equity - prev_equity) / prev_equity
                    if prev_equity > 0
                    else 0.0
                )
                if circuit_breaker_day > 0.0 and day_return < -circuit_breaker_day:
                    circuit_breaker_active = True
                    circuit_breaker_countdown = circuit_recovery_days
                if circuit_breaker_total > 0.0 and current_dd > circuit_breaker_total:
                    circuit_breaker_active = True
                    circuit_breaker_countdown = circuit_recovery_days

            if circuit_breaker_active:
                if circuit_breaker_countdown > 0:
                    circuit_breaker_countdown -= 1
                else:
                    circuit_breaker_active = False

        # Cooldown Logic
        for n in range(N):
            if cooldown_remaining[n] > 0:
                cooldown_remaining[n] -= 1

        # Stop Loss Logic
        should_check_stop = (use_atr_stop and atr_multiplier > 0.0) or (
            not use_atr_stop and trailing_stop_pct > 0.0
        )
        if should_check_stop and (not stop_on_rebalance_only or is_rebalance_day):
            for n in range(N):
                if holdings[n] > 0.0:
                    prev_hwm = high_water_marks[n]
                    current_return = (
                        (prev_hwm - entry_prices[n]) / entry_prices[n]
                        if entry_prices[n] > 0
                        else 0.0
                    )
                    for ladder_idx in range(3):
                        if current_return >= profit_ladder_thresholds[ladder_idx]:
                            if use_atr_stop:
                                if (
                                    profit_ladder_multipliers[ladder_idx]
                                    < current_atr_mults[n]
                                ):
                                    current_atr_mults[n] = profit_ladder_multipliers[
                                        ladder_idx
                                    ]
                            else:
                                if (
                                    profit_ladder_stops[ladder_idx]
                                    < current_stop_pcts[n]
                                ):
                                    current_stop_pcts[n] = profit_ladder_stops[
                                        ladder_idx
                                    ]

                    if use_atr_stop:
                        prev_atr = atr_arr[t - 1, n] if t > 0 else 0.0
                        if np.isnan(prev_atr) or prev_atr <= 0.0:
                            curr_high = high_prices[t, n]
                            if (
                                not np.isnan(curr_high)
                                and curr_high > high_water_marks[n]
                            ):
                                high_water_marks[n] = curr_high
                            continue
                        stop_price = prev_hwm - (current_atr_mults[n] * prev_atr)
                    else:
                        stop_price = prev_hwm * (1.0 - current_stop_pcts[n])

                    curr_low = low_prices[t, n]
                    curr_open = open_prices[t, n]

                    if not np.isnan(curr_low) and curr_low < stop_price:
                        if not np.isnan(curr_open) and curr_open < stop_price:
                            exec_price = curr_open
                        else:
                            exec_price = stop_price

                        if not np.isnan(curr_low):
                            exec_price = max(exec_price, curr_low)

                        proceeds = holdings[n] * exec_price * (1.0 - commission_rate)
                        cash += proceeds

                        pnl = (exec_price - entry_prices[n]) / entry_prices[n]
                        if pnl > 0.0:
                            wins += 1
                            total_win_pnl += pnl
                        else:
                            losses += 1
                            total_loss_pnl += abs(pnl)

                        holdings[n] = -1.0
                        entry_prices[n] = 0.0
                        high_water_marks[n] = 0.0
                        current_stop_pcts[n] = trailing_stop_pct
                        current_atr_mults[n] = atr_multiplier
                        cooldown_remaining[n] = cooldown_days
                    else:
                        curr_high = high_prices[t, n]
                        if not np.isnan(curr_high) and curr_high > high_water_marks[n]:
                            high_water_marks[n] = curr_high

        # Rebalance Logic
        if is_rebalance_day:
            if dynamic_leverage_enabled and buffer_filled >= vol_window // 2:
                n_samples = buffer_filled
                sum_val = 0.0
                sum_sq = 0.0
                for i in range(n_samples):
                    val = returns_buffer[i]
                    sum_val += val
                    sum_sq += val * val
                mean_ret = sum_val / n_samples
                variance = (sum_sq / n_samples) - (mean_ret * mean_ret)
                if variance > 0:
                    daily_std = np.sqrt(variance)
                    realized_vol = daily_std * np.sqrt(252.0)
                    if realized_vol > 0.0001:
                        current_leverage = min(1.0, target_vol / realized_vol)
                    else:
                        current_leverage = 1.0
                else:
                    current_leverage = 1.0
            else:
                current_leverage = 1.0

            leverage_sum += current_leverage
            leverage_count += 1

            valid = 0
            for n in range(N):
                score = 0.0
                has_value = False
                for idx in factor_indices:
                    val = factors_3d[t - 1, n, idx]
                    if not np.isnan(val):
                        score += val
                        has_value = True

                if has_value and score != 0.0:
                    combined_score[n] = score
                    valid += 1
                else:
                    combined_score[n] = -np.inf

            for n in range(N):
                target_set[n] = False

            buy_count = 0
            if valid >= pos_size:
                top_indices = stable_topk_indices(combined_score, pos_size)
                for k in range(len(top_indices)):
                    idx = top_indices[k]
                    if combined_score[idx] == -np.inf:
                        break
                    if (
                        individual_trend_enabled
                        and not individual_trend_arr[t - 1, idx]
                    ):
                        continue
                    target_set[idx] = True
                    buy_order[buy_count] = idx
                    buy_count += 1

            effective_leverage = min(current_leverage, leverage_cap)
            timing_ratio = timing_arr[t] * effective_leverage

            if circuit_breaker_active:
                timing_ratio = 0.0

            for n in range(N):
                if holdings[n] > 0.0 and not target_set[n]:
                    price = close_prices[t, n]
                    proceeds = holdings[n] * price * (1.0 - commission_rate)
                    cash += proceeds

                    pnl = (price - entry_prices[n]) / entry_prices[n]
                    if pnl > 0.0:
                        wins += 1
                        total_win_pnl += pnl
                    else:
                        losses += 1
                        total_loss_pnl += abs(pnl)

                    holdings[n] = -1.0
                    entry_prices[n] = 0.0
                    high_water_marks[n] = 0.0
                    current_stop_pcts[n] = trailing_stop_pct
                    current_atr_mults[n] = atr_multiplier

            current_value = cash
            kept_value = 0.0
            for n in range(N):
                if holdings[n] > 0.0:
                    val = holdings[n] * close_prices[t, n]
                    current_value += val
                    kept_value += val

            new_count = 0
            for k in range(buy_count):
                idx = buy_order[k]
                if holdings[idx] < 0.0 and cooldown_remaining[idx] == 0:
                    new_targets[new_count] = idx
                    new_count += 1

            if new_count > 0:
                target_exposure = current_value * timing_ratio
                available_for_new = target_exposure - kept_value
                if available_for_new < 0.0:
                    available_for_new = 0.0

                target_pos_value = (
                    available_for_new / new_count / (1.0 + commission_rate)
                )

                if target_pos_value > 0.0:
                    for k in range(new_count):
                        idx = new_targets[k]
                        price = close_prices[t, idx]
                        if np.isnan(price) or price <= 0.0:
                            continue

                        shares = target_pos_value / price
                        cost = shares * price * (1.0 + commission_rate)
                        target_value_total += target_pos_value
                        target_shares_total += shares

                        if cash >= cost - 1e-5 and cost > 0.0:
                            actual_cost = cost if cost <= cash else cash
                            actual_shares = actual_cost / (
                                price * (1.0 + commission_rate)
                            )
                            filled_shares_total += actual_shares
                            filled_value_total += actual_shares * price
                            cash -= actual_cost
                            holdings[idx] = shares
                            entry_prices[idx] = price
                            high_water_marks[idx] = price
                            current_stop_pcts[idx] = trailing_stop_pct
                            current_atr_mults[idx] = atr_multiplier

        # ✅ NEW: Record Holdings
        holdings_history[t] = holdings

    # Final Value Calculation
    final_value = cash
    for n in range(N):
        if holdings[n] > 0.0:
            price = close_prices[T - 1, n]
            if np.isnan(price):
                price = entry_prices[n]
            final_value += holdings[n] * price

    if final_value != prev_equity and prev_equity > 0:
        daily_return = (final_value - prev_equity) / prev_equity
        welford_count += 1
        delta = daily_return - welford_mean
        welford_mean += delta / welford_count
        delta2 = daily_return - welford_mean
        welford_m2 += delta * delta2

    if final_value > peak_equity:
        peak_equity = final_value
    final_dd = (peak_equity - final_value) / peak_equity if peak_equity > 0 else 0.0
    if final_dd > max_drawdown:
        max_drawdown = final_dd

    num_trades = wins + losses
    total_return = (final_value - initial_capital) / initial_capital

    trading_days = T - start_day
    years = trading_days / 252.0 if trading_days > 0 else 1.0
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    if welford_count > 1:
        daily_variance = welford_m2 / (welford_count - 1)
        daily_std = np.sqrt(daily_variance) if daily_variance > 0 else 0.0
        annual_volatility = daily_std * np.sqrt(252.0)
    else:
        annual_volatility = 0.0

    sharpe_ratio = (
        annual_return / annual_volatility if annual_volatility > 0.0001 else 0.0
    )

    return (
        equity_curve,
        holdings_history,  # ✅ NEW
        total_return,
        num_trades,
        filled_value_total,
        max_drawdown,
        annual_return,
        sharpe_ratio,
    )


# --- Main Logic ---


def main():
    # 1. Find Latest Results
    results_dir = ROOT / "results"
    liquid_dirs = sorted(results_dir.glob("vec_liquid_*"))
    if not liquid_dirs:
        print("No liquid results found.")
        return
    latest_dir = liquid_dirs[-1]
    csv_path = latest_dir / "liquid_results.csv"
    print(f"Loading results from {csv_path}")

    df = pd.read_csv(csv_path)

    # 2. Filter Strategies
    # Regime Fitness Filter: Recent_Ret_5M > 0 and Recent_MDD_5M < 8%
    filtered = df[(df["Recent_Ret_5M"] > 0) & (df["Recent_MDD_5M"] < 0.08)].copy()

    print(f"Total Strategies: {len(df)}")
    print(f"Filtered Strategies: {len(filtered)}")

    if len(filtered) == 0:
        print("No strategies passed the filter.")
        return

    # 3. Load Data
    print("Loading Market Data...")
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from etf_strategy.core.data_loader import DataLoader

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )

    ohlcv_data = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # Filter Liquid ETFs
    with open(ROOT / "scripts/run_liquid_vec_backtest.py", "r") as f:
        content = f.read()
        # Extract LIQUID_ETFS
        import ast

        tree = ast.parse(content)
        whitelist = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "LIQUID_ETFS":
                        whitelist = ast.literal_eval(node.value)
                        break
        if not whitelist:
            # Fallback hardcoded list if extraction fails
            whitelist = [
                "510300",
                "510500",
                "510050",
                "513100",
                "513500",
                "512880",
                "512000",
                "512660",
                "512010",
                "512800",
                "512690",
                "512480",
                "512100",
                "512070",
                "515000",
                "588000",
                "159915",
                "159949",
                "518880",
                "513050",
                "513330",
            ]

    # Handle Panel Data (Dict of DataFrames)
    all_tickers = ohlcv_data["close"].columns
    tickers = sorted([t for t in all_tickers if t in whitelist])
    print(f"Liquid Tickers: {len(tickers)}")

    # Align Data
    dates = ohlcv_data["close"].index
    T = len(dates)
    N = len(tickers)

    close_prices = ohlcv_data["close"][tickers].values
    open_prices = ohlcv_data["open"][tickers].values
    high_prices = ohlcv_data["high"][tickers].values
    low_prices = ohlcv_data["low"][tickers].values
    volume_arr = ohlcv_data["volume"][tickers].values

    # Calculate ADV (Amount = Close * Volume * 100)
    # Note: Volume is likely in Hands (100 shares) for China ETFs
    amount_arr = close_prices * volume_arr * 100
    # Rolling 20D ADV
    adv_20d = np.zeros_like(amount_arr)
    for i in range(N):
        s = pd.Series(amount_arr[:, i])
        adv_20d[:, i] = s.rolling(20, min_periods=1).mean().fillna(0).values

    # 4. Compute Factors
    print("Computing Factors...")
    lib = PreciseFactorLibrary()

    # Filter data for liquid tickers
    liquid_data = {k: v[tickers] for k, v in ohlcv_data.items()}

    # Compute all factors at once
    raw_factors_df = lib.compute_all_factors(liquid_data)
    factor_names = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    F = len(factor_names)

    # Construct 3D Factor Array (T, N, F)
    factors_3d = np.full((T, N, F), np.nan)

    # Need to standardize factors?
    # run_liquid_vec_backtest.py does CrossSectionProcessor.
    # We should too.
    from etf_strategy.core.cross_section_processor import CrossSectionProcessor

    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    for j, f_name in enumerate(factor_names):
        factors_3d[:, :, j] = std_factors[f_name].values

    # 5. Validation Loop
    print(f"Validating {len(filtered)} strategies...")

    # Common Params
    FREQ = 3
    POS_SIZE = 2
    rebalance_schedule = generate_rebalance_schedule(len(dates), 252, FREQ)
    timing_arr = np.ones(T)  # Always bullish for now (or load market timing if needed)
    # Note: Original script used market_timing.py. Here we assume full exposure for simplicity
    # OR we should replicate the timing logic.
    # run_liquid_vec_backtest.py used:
    # timing_signals = calculate_market_timing(...)
    # Let's replicate that.
    from etf_strategy.core.market_timing import LightTimingModule

    # Need close prices DataFrame
    close_df = pd.DataFrame(close_prices, index=dates, columns=tickers)
    timing_module = LightTimingModule()
    timing_signals = timing_module.compute_position_ratios(close_df)
    timing_arr = shift_timing_signal(timing_signals)

    # Prepare Results List
    validation_results = []

    # Pre-compile kernel with dummy run
    print("Compiling Kernel...")
    _ = validation_backtest_kernel(
        factors_3d[:, :, :1],
        close_prices,
        open_prices,
        high_prices,
        low_prices,
        timing_arr,
        np.array([0]),
        rebalance_schedule,
        POS_SIZE,
        1000000.0,
        0.0002,
        0.0,
        0,
        False,
        False,
        0.0,
        np.zeros((T, N)),
        0.0,
        False,
        np.ones((T, N), dtype=bool),
        False,
        np.array([0.15, 0.30, np.inf]),
        np.array([0.05, 0.03, 0.08]),
        np.array([2.0, 1.5, 3.0]),
        0.0,
        0.0,
        0,
        0,
        1.0,
    )

    # Define Years for Analysis
    years = sorted(list(set(dates.year)))
    year_masks = {y: (dates.year == y) for y in years}

    for idx, row in tqdm(filtered.iterrows(), total=len(filtered)):
        combo_str = row["combo"]
        f_list = [f.strip() for f in combo_str.split("+")]
        f_indices = np.array(
            [factor_names.index(f) for f in f_list if f in factor_names]
        )

        if len(f_indices) == 0:
            continue

        # Run Backtest
        (
            equity_curve,
            holdings_hist,
            total_ret,
            num_trades,
            filled_val,
            max_dd,
            ann_ret,
            sharpe,
        ) = validation_backtest_kernel(
            factors_3d,
            close_prices,
            open_prices,
            high_prices,
            low_prices,
            timing_arr,
            f_indices,
            rebalance_schedule,
            POS_SIZE,
            1000000.0,
            0.0002,
            0.0,
            0,
            False,
            False,
            0.0,
            np.zeros((T, N)),
            0.0,
            False,
            np.ones((T, N), dtype=bool),
            False,
            np.array([0.15, 0.30, np.inf]),
            np.array([0.05, 0.03, 0.08]),
            np.array([2.0, 1.5, 3.0]),
            0.0,
            0.0,
            0,
            0,
            1.0,
        )

        # --- Metrics Calculation ---

        res = {
            "combo": combo_str,
            "total_return": total_ret,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "ann_return": ann_ret,
            "num_trades": num_trades,
        }

        # 1. Yearly Stats
        valid_equity = equity_curve
        # Handle zeros at start (before lookback)
        start_idx = np.argmax(valid_equity > 0)

        for y in years:
            mask = year_masks[y]
            # Ensure we are within valid range
            mask[:start_idx] = False

            if not np.any(mask):
                res[f"ret_{y}"] = 0.0
                res[f"mdd_{y}"] = 0.0
                continue

            y_equity = valid_equity[mask]
            if len(y_equity) > 0:
                # Return
                # Need prev day equity for first day of year
                first_idx = np.where(mask)[0][0]
                prev_eq = valid_equity[first_idx - 1] if first_idx > 0 else 1000000.0
                if prev_eq <= 0.0001:
                    prev_eq = 1000000.0  # Fallback to initial capital if previous day was 0 (warmup)
                y_ret = (y_equity[-1] / prev_eq) - 1

                # MDD
                peak = np.maximum.accumulate(y_equity)
                dd = (peak - y_equity) / peak
                y_mdd = np.max(dd)

                res[f"ret_{y}"] = y_ret
                res[f"mdd_{y}"] = y_mdd
            else:
                res[f"ret_{y}"] = 0.0
                res[f"mdd_{y}"] = 0.0

        # 2. Cost Sensitivity
        # Turnover Rate (Annualized)
        # Avg Equity
        avg_equity = (
            np.mean(valid_equity[start_idx:])
            if len(valid_equity[start_idx:]) > 0
            else 1000000.0
        )
        trading_years = (T - start_idx) / 252.0
        turnover_ann = (
            (filled_val / avg_equity) / trading_years if trading_years > 0 else 0.0
        )

        res["turnover_ann"] = turnover_ann

        # Sensitivity: Impact of +5bp cost (0.07% total vs 0.02%)
        # Actually base is 0.02% (2bp). User asked for 0.07, 0.10, 0.12.
        # Let's calc impact of moving from 0.02% to 0.12% (+10bp per side -> +20bp roundtrip? No, commission is usually one-side)
        # Commission rate passed is 0.0002 (2bp).
        # If we increase to 0.0012 (12bp), delta is 0.0010 (10bp).
        # Cost Impact = Turnover * Delta_Cost
        # Note: filled_val is one-side value.
        # Total Cost Increase = filled_val * 0.0010
        # Return Impact = Total Cost Increase / Initial Capital
        # Ann Return Impact approx = Turnover_Ann * 0.0010

        cost_delta = 0.0010  # 10bp increase
        ret_impact = turnover_ann * cost_delta
        res["cost_sensitivity"] = (
            ret_impact / abs(ann_ret) if abs(ann_ret) > 0.01 else 1.0
        )

        # 3. Capacity / Holdings
        # Top Symbol Weight
        # holdings_hist: (T, N) > 0
        held_days = holdings_hist > 0
        symbol_days = np.sum(held_days, axis=0)
        total_days = np.sum(symbol_days)
        top_symbol_weight = np.max(symbol_days) / total_days if total_days > 0 else 0.0
        res["top_symbol_weight"] = top_symbol_weight

        # Avg ADV Weighted
        # For each day t, calculate average ADV of held assets
        # Then average over time
        # Vectorized:
        # held_adv = adv_20d * held_days
        # daily_adv_sum = np.sum(held_adv, axis=1)
        # daily_count = np.sum(held_days, axis=1)
        # daily_avg_adv = np.divide(daily_adv_sum, daily_count, out=np.zeros_like(daily_adv_sum), where=daily_count>0)
        # avg_adv_weighted = np.mean(daily_avg_adv[daily_count>0])

        held_adv = adv_20d * held_days
        daily_adv_sum = np.sum(held_adv, axis=1)
        daily_count = np.sum(held_days, axis=1)

        # Avoid div by zero
        valid_days = daily_count > 0
        if np.any(valid_days):
            daily_avg = daily_adv_sum[valid_days] / daily_count[valid_days]
            avg_adv_weighted = np.mean(daily_avg)
        else:
            avg_adv_weighted = 0.0

        res["avg_adv_weighted"] = avg_adv_weighted

        # --- Scoring & Labeling ---
        status = "PASS"
        reasons = []

        # FAIL Conditions
        if res.get("mdd_2022", 0) > 0.30 or res.get("mdd_2025", 0) > 0.30:
            status = "FAIL"
            reasons.append("High Bear MDD")

        if res["cost_sensitivity"] > 0.50:
            status = "FAIL"
            reasons.append("High Cost Sens")

        if res["avg_adv_weighted"] < 30_000_000:  # 30M
            status = "FAIL"
            reasons.append("Low Liquidity")

        # WARNING Conditions
        if status != "FAIL":
            if res["top_symbol_weight"] > 0.50:
                status = "WARNING"
                reasons.append("Concentrated")

            # Check consistency: if only 1 year is good
            # Simple check: count years with Return > 5%
            good_years = sum(1 for y in years if res.get(f"ret_{y}", 0) > 0.05)
            if good_years < 2:
                status = "WARNING"
                reasons.append("Inconsistent")

        res["status"] = status
        res["reasons"] = "; ".join(reasons)

        validation_results.append(res)

    # 6. Save Results
    val_df = pd.DataFrame(validation_results)
    out_dir = ROOT / "results" / "multi_strategy_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_out = out_dir / "strategy_summary.csv"
    val_df.to_csv(csv_out, index=False)
    print(f"Saved summary to {csv_out}")

    # 7. Generate Report
    report_path = out_dir / "VALIDATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# 🛡️ Multi-Strategy Robustness Validation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Source**: `{csv_path}`\n\n")

        # Overview
        counts = val_df["status"].value_counts()
        f.write("## 📊 Overview\n")
        f.write(f"- **Total Validated**: {len(val_df)}\n")
        f.write(f"- **PASS**: {counts.get('PASS', 0)}\n")
        f.write(f"- **WARNING**: {counts.get('WARNING', 0)}\n")
        f.write(f"- **FAIL**: {counts.get('FAIL', 0)}\n\n")

        # Top PASS Strategies
        pass_df = val_df[val_df["status"] == "PASS"].copy()
        if not pass_df.empty:
            # Sort by Sharpe for now, or a composite score
            pass_df = pass_df.sort_values("sharpe", ascending=False)
            top10 = pass_df.head(10)

            f.write("## 🏆 Top 10 PASS Strategies\n")
            cols = [
                "combo",
                "ann_return",
                "sharpe",
                "max_dd",
                "cost_sensitivity",
                "avg_adv_weighted",
            ]
            # Format columns
            disp_df = top10[cols].copy()
            disp_df["ann_return"] = disp_df["ann_return"].apply(lambda x: f"{x:.2%}")
            disp_df["max_dd"] = disp_df["max_dd"].apply(lambda x: f"{x:.2%}")
            disp_df["cost_sensitivity"] = disp_df["cost_sensitivity"].apply(
                lambda x: f"{x:.2%}"
            )
            disp_df["avg_adv_weighted"] = disp_df["avg_adv_weighted"].apply(
                lambda x: f"{x/1e6:.1f}M"
            )
            disp_df["sharpe"] = disp_df["sharpe"].apply(lambda x: f"{x:.3f}")

            f.write(disp_df.to_markdown(index=False))
            f.write("\n\n")

            # Detailed Year Stats for Top 5
            f.write("## 📅 Yearly Performance (Top 5 PASS)\n")
            for idx, row in top10.head(5).iterrows():
                f.write(f"### `{row['combo']}`\n")
                f.write(
                    f"- **Ann Ret**: {row['ann_return']:.2%} | **Sharpe**: {row['sharpe']:.3f} | **MaxDD**: {row['max_dd']:.2%}\n"
                )
                f.write(
                    f"- **Liquidity**: {row['avg_adv_weighted']/1e6:.1f}M | **Concentration**: {row['top_symbol_weight']:.1%}\n"
                )

                # Yearly Table
                y_data = []
                for y in years:
                    y_data.append(
                        {
                            "Year": y,
                            "Return": row.get(f"ret_{y}", 0),
                            "MaxDD": row.get(f"mdd_{y}", 0),
                        }
                    )
                y_df = pd.DataFrame(y_data)
                y_df["Return"] = y_df["Return"].apply(lambda x: f"{x:.2%}")
                y_df["MaxDD"] = y_df["MaxDD"].apply(lambda x: f"{x:.2%}")
                f.write(y_df.to_markdown(index=False))
                f.write("\n\n")
        else:
            f.write("No strategies passed the validation criteria.\n")

    print(f"Report generated at {report_path}")


if __name__ == "__main__":
    main()
