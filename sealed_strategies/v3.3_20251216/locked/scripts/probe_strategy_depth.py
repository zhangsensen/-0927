"""
策略深度探测脚本 (Deep Probe)
Strategy: ADX_14D + PRICE_POSITION_120D + PRICE_POSITION_20D
Version: v1.0
Date: 2025-12-10
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit

# Ensure src is in path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.utils.rebalance import generate_rebalance_schedule

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
FREQ = 3
POS_SIZE = 2
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252

# --- HELPER FUNCTIONS ---


@njit
def stable_topk_indices(scores, k):
    """稳定排序：按 score 降序，score 相同时按 ETF 索引升序（tie-breaker）。

    返回 top-k 的索引数组。这确保 numba 和 Python 行为一致。
    """
    N = len(scores)
    # 创建 (score, -index) 对，使得相同 score 时较小 index 排在前面
    # Numba 不支持复杂排序，手动实现选择排序（k 很小，O(kN) 可接受）
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
            # 不够 k 个有效值
            return result[:i]
        result[i] = best_idx
        used[best_idx] = True
    return result


@njit
def calc_price_position_numba(close, high, low, window):
    """
    Calculate Price Position manually for variable windows.
    PP = (Close - Min(Low, w)) / (Max(High, w) - Min(Low, w))
    """
    T, N = close.shape
    pp = np.full((T, N), np.nan)

    for n in range(N):
        for t in range(window, T):
            # Window slice: [t-window+1 : t+1]
            # Note: t is inclusive index in python slice end is exclusive
            # slice indices: t-window+1 to t+1
            start_idx = t - window + 1
            end_idx = t + 1

            w_high = high[start_idx:end_idx, n]
            w_low = low[start_idx:end_idx, n]

            # Check for NaNs in window
            if np.isnan(w_high).any() or np.isnan(w_low).any():
                continue

            h_max = np.max(w_high)
            l_min = np.min(w_low)

            denom = h_max - l_min
            if denom > 1e-9:
                pp[t, n] = (close[t, n] - l_min) / denom
            else:
                pp[t, n] = 0.5  # Default if flat

    return pp


@njit
def simple_vec_backtest(
    combined_score,
    close_prices,
    rebalance_schedule,
    pos_size,
    initial_capital,
    commission_rate,
):
    """
    Simplified Vectorized Backtest for Probe.
    No stop loss, no complex timing, just rank and hold.
    """
    T, N = combined_score.shape
    cash = initial_capital
    holdings = np.zeros(N)  # Shares held

    equity_curve = np.zeros(T)
    equity_curve[: rebalance_schedule[0]] = initial_capital

    rebal_ptr = 0

    # Track holdings for liquidity analysis
    # holdings_history[t, n] = shares
    holdings_history = np.zeros((T, N))

    for t in range(rebalance_schedule[0], T):
        # 1. Calculate Equity (Before Rebalance)
        current_equity = cash
        for n in range(N):
            if holdings[n] > 0:
                current_equity += holdings[n] * close_prices[t, n]

        equity_curve[t] = current_equity

        # 2. Check Rebalance
        if rebal_ptr < len(rebalance_schedule) and rebalance_schedule[rebal_ptr] == t:
            # Rebalance Day
            rebal_ptr += 1

            # Sell All
            for n in range(N):
                if holdings[n] > 0:
                    cash += holdings[n] * close_prices[t, n] * (1 - commission_rate)
                    holdings[n] = 0

            # Select Top K
            scores = combined_score[t]
            # Filter NaNs
            valid_mask = ~np.isnan(scores)
            if np.sum(valid_mask) >= pos_size:
                # Use stable sort logic or simple argsort
                # We want descending sort
                # Fill NaNs with -inf for sorting
                safe_scores = scores.copy()
                safe_scores[~valid_mask] = -np.inf

                # Get top indices
                top_indices = stable_topk_indices(safe_scores, pos_size)

                # Buy
                target_val = cash / pos_size
                for idx in top_indices:
                    price = close_prices[t, idx]
                    if not np.isnan(price) and price > 0:
                        shares = target_val / (price * (1 + commission_rate))
                        cost = shares * price * (1 + commission_rate)
                        cash -= cost
                        holdings[idx] = shares

        # Record holdings
        holdings_history[t] = holdings

    return equity_curve, holdings_history


def calculate_metrics(equity_curve):
    """Calculate Sharpe and Total Return"""
    # Filter zeros (pre-start)
    valid_equity = equity_curve[equity_curve > 0]
    if len(valid_equity) < 2:
        return 0.0, 0.0

    ret_series = pd.Series(valid_equity).pct_change().dropna()
    if len(ret_series) == 0:
        return 0.0, 0.0

    total_ret = (valid_equity[-1] / valid_equity[0]) - 1
    sharpe = (
        ret_series.mean() / ret_series.std() * np.sqrt(252)
        if ret_series.std() > 0
        else 0
    )

    return total_ret, sharpe


def calculate_max_dd(equity_curve):
    """Calculate Max Drawdown"""
    valid_equity = equity_curve[equity_curve > 0]
    if len(valid_equity) == 0:
        return 0.0

    peak = np.maximum.accumulate(valid_equity)
    dd = (peak - valid_equity) / peak
    return np.max(dd)


def run_probe():
    logger.info("Starting Deep Probe...")

    # 1. Load Data
    data_dir = project_root / "raw" / "ETF" / "daily"
    loader = DataLoader(data_dir=str(data_dir))
    ohlcv_dict = loader.load_ohlcv()

    # Align Data
    # ohlcv_dict structure: {'close': df, 'high': df, ...}
    # The previous code assumed ohlcv_dict[code] -> df, but DataLoader returns dict of dfs

    close = ohlcv_dict["close"].ffill()
    high = ohlcv_dict["high"].ffill()
    low = ohlcv_dict["low"].ffill()
    volume = ohlcv_dict["volume"].fillna(0)

    dates = close.index
    codes = close.columns.tolist()

    close_np = close.values
    high_np = high.values
    low_np = low.values

    # 2. Base Factor: ADX_14D
    logger.info("Calculating Base Factor: ADX_14D")
    lib = PreciseFactorLibrary()
    # We can use the library's internal batch method if accessible, or compute_all and extract
    # compute_all is safer
    all_factors = lib.compute_all_factors(ohlcv_dict)
    adx_df = all_factors["ADX_14D"]
    # Align columns to codes list
    adx_np = adx_df[codes].values

    # Rank ADX
    adx_rank = pd.DataFrame(adx_np).rank(axis=1, pct=True).values

    # Rebalance Schedule
    # generate_rebalance_schedule(total_periods, lookback_window, freq)
    rebalance_schedule = generate_rebalance_schedule(len(dates), LOOKBACK, FREQ)

    # --- PART 1: PARAMETER SENSITIVITY ---
    logger.info("--- PART 1: PARAMETER SENSITIVITY ---")

    long_windows = [100, 110, 120, 130, 140]
    short_windows = [15, 18, 20, 22, 25]

    sensitivity_results = []

    for w_long in long_windows:
        for w_short in short_windows:
            # Calc PP
            pp_long = calc_price_position_numba(close_np, high_np, low_np, w_long)
            pp_short = calc_price_position_numba(close_np, high_np, low_np, w_short)

            # Rank
            pp_long_rank = pd.DataFrame(pp_long).rank(axis=1, pct=True).values
            pp_short_rank = pd.DataFrame(pp_short).rank(axis=1, pct=True).values

            # Combine
            combined = adx_rank + pp_long_rank + pp_short_rank

            # Backtest
            equity, _ = simple_vec_backtest(
                combined,
                close_np,
                rebalance_schedule,
                POS_SIZE,
                INITIAL_CAPITAL,
                COMMISSION_RATE,
            )

            # Metrics
            ret, sharpe = calculate_metrics(equity)

            sensitivity_results.append(
                {"w_long": w_long, "w_short": w_short, "return": ret, "sharpe": sharpe}
            )
            # print(f"L={w_long}, S={w_short} -> Ret={ret:.2%}, Sharpe={sharpe:.2f}")

    sens_df = pd.DataFrame(sensitivity_results)

    # --- PART 2: HISTORICAL CRISIS REPLAY ---
    logger.info("--- PART 2: HISTORICAL CRISIS REPLAY ---")

    # Base Parameters: 120, 20
    base_pp_long = calc_price_position_numba(close_np, high_np, low_np, 120)
    base_pp_short = calc_price_position_numba(close_np, high_np, low_np, 20)

    base_pp_long_rank = pd.DataFrame(base_pp_long).rank(axis=1, pct=True).values
    base_pp_short_rank = pd.DataFrame(base_pp_short).rank(axis=1, pct=True).values

    base_combined = adx_rank + base_pp_long_rank + base_pp_short_rank

    base_equity, base_holdings = simple_vec_backtest(
        base_combined,
        close_np,
        rebalance_schedule,
        POS_SIZE,
        INITIAL_CAPITAL,
        COMMISSION_RATE,
    )

    equity_series = pd.Series(base_equity, index=dates)
    # Filter out zeros
    equity_series = equity_series[equity_series > 0]

    crisis_periods = {
        "2020 Pandemic": ("2020-02-20", "2020-03-31"),
        "2022 Bear Market": ("2022-03-01", "2022-04-30"),
        "2024 Micro-cap Crash": ("2024-01-01", "2024-02-08"),
    }

    crisis_results = []
    for name, (start, end) in crisis_periods.items():
        # Slice
        sub_equity = equity_series.loc[start:end]
        if len(sub_equity) > 0:
            period_ret = (sub_equity.iloc[-1] / sub_equity.iloc[0]) - 1

            # MaxDD in period
            peak = sub_equity.cummax()
            dd = (peak - sub_equity) / peak
            max_dd = dd.max()

            crisis_results.append(
                {
                    "Scenario": name,
                    "Start": start,
                    "End": end,
                    "Return": period_ret,
                    "MaxDD": max_dd,
                }
            )
        else:
            crisis_results.append(
                {
                    "Scenario": name,
                    "Start": start,
                    "End": end,
                    "Return": 0.0,
                    "MaxDD": 0.0,
                }
            )

    crisis_df = pd.DataFrame(crisis_results)

    # --- PART 3: POSITION CHARACTERISTICS ---
    logger.info("--- PART 3: POSITION CHARACTERISTICS ---")

    # Calculate Daily Turnover Amount for all assets
    # Amount = Close * Volume
    # Note: Volume in China is usually shares (100), Amount is value.
    # But here we have raw volume. Let's assume Volume is shares.
    # Actually, let's check if we have 'amount' column in data loader? No, usually just OHLCV.
    # We approximate Amount = Close * Volume.

    daily_amount = close * volume  # DataFrame (T, N)

    # We need to know which assets were held each day
    # base_holdings is (T, N) shares held
    # We want to know the liquidity of the assets *selected*, not the amount we held.
    # So if holdings[t, n] > 0, we check daily_amount[t, n]

    held_mask = base_holdings > 0

    # Extract amounts for held assets
    held_amounts = daily_amount.values[held_mask]

    # Binning
    # < 50M (50,000,000)
    # 50M - 200M
    # > 200M

    bins = [0, 50_000_000, 200_000_000, float("inf")]
    labels = ["<50M", "50M-200M", ">200M"]

    cats = pd.cut(held_amounts, bins=bins, labels=labels)
    # Handle pandas version differences for Categorical.value_counts
    try:
        counts = cats.value_counts(normalize=True)
    except TypeError:
        # Older pandas or specific behavior
        counts = pd.Series(cats).value_counts(normalize=True)

    # --- REPORT GENERATION ---
    print("\n" + "=" * 40)
    print("      STRATEGY DEEP PROBE REPORT      ")
    print("=" * 40 + "\n")

    print("### 1. Parameter Sensitivity (Heatmap Data)")
    print(
        sens_df.pivot(index="w_long", columns="w_short", values="sharpe").to_markdown()
    )
    print("\n(Values are Sharpe Ratio)")

    # Check Plateau
    base_sharpe = sens_df[(sens_df["w_long"] == 120) & (sens_df["w_short"] == 20)][
        "sharpe"
    ].values[0]
    min_sharpe = sens_df["sharpe"].min()
    drop = (base_sharpe - min_sharpe) / base_sharpe
    print(f"\nBase Sharpe (120, 20): {base_sharpe:.4f}")
    print(f"Min Sharpe in Range: {min_sharpe:.4f}")
    print(f"Max Performance Drop: {drop:.2%}")

    print("\n### 2. Historical Crisis Replay")
    print(crisis_df.to_markdown(index=False, floatfmt=".2%"))

    print("\n### 3. Position Liquidity Profile")
    print(counts.to_markdown(floatfmt=".2%"))

    # Final Conclusion Logic
    print("\n### 4. Final Conclusion")

    is_robust = True
    reasons = []

    if drop > 0.20:
        is_robust = False
        reasons.append("Parameter Sensitivity Drop > 20% (Overfitting Risk)")

    # Crisis Check: If any MaxDD > 20% in crisis
    if crisis_df["MaxDD"].max() > 0.20:
        is_robust = False
        reasons.append("Historical Crisis MaxDD > 20%")

    # Liquidity Check: If > 50% in <50M
    if counts.get("<50M", 0) > 0.5:
        is_robust = False
        reasons.append("High Exposure to Illiquid Assets (>50% in <50M)")

    if is_robust:
        print("**STATUS: PASS**")
        print(
            "Strategy shows robust characteristics across parameters and historical crises."
        )
    else:
        print("**STATUS: WARNING / FAIL**")
        for r in reasons:
            print(f"- {r}")


if __name__ == "__main__":
    run_probe()
