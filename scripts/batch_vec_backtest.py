#!/usr/bin/env python3
"""
批量 VEC 回测：遍历 WFO 输出的全部组合，逐个用向量化引擎回测并保存结果。

✅ P0 修正: 删除所有硬编码常量，强制从配置文件读取
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from numba import njit
from joblib import Parallel, delayed

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import load_frozen_config
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule, DualTimingModule
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
    ensure_price_views,
)
from etf_strategy.regime_gate import compute_regime_gate_arr, gate_stats
from etf_strategy.core.hysteresis import apply_hysteresis
from etf_strategy.core.utils.position_sizing import (
    parse_dynamic_pos_config,
    resolve_pos_size_for_day,
)
from aligned_metrics import compute_aligned_metrics

# ✅ P0: 删除硬编码 - 所有参数必须从配置文件读取
# FREQ = 8  # DELETED
# POS_SIZE = 3  # DELETED
# INITIAL_CAPITAL = 1_000_000.0  # DELETED
# COMMISSION_RATE = 0.0002  # DELETED
# LOOKBACK = 252  # DELETED


@njit(cache=True)
def calculate_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14
) -> np.ndarray:
    """计算 ATR (Average True Range) 使用 Wilder 平滑法。

    ATR = Wilder Smoothed(TR)
    TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)

    Args:
        high: (T, N) 最高价矩阵
        low: (T, N) 最低价矩阵
        close: (T, N) 收盘价矩阵
        window: ATR 周期，默认 14

    Returns:
        (T, N) ATR 矩阵，前 window 行为 NaN
    """
    T, N = high.shape
    atr = np.full((T, N), np.nan)

    # True Range 计算
    tr = np.zeros((T, N))
    for t in range(1, T):
        for n in range(N):
            h = high[t, n]
            l = low[t, n]
            prev_c = close[t - 1, n]

            if np.isnan(h) or np.isnan(l) or np.isnan(prev_c):
                tr[t, n] = np.nan
            else:
                tr[t, n] = max(h - l, abs(h - prev_c), abs(l - prev_c))

    # Wilder 平滑 (指数平滑)
    # 初始 ATR = 前 window 个 TR 的简单平均
    # 后续 ATR = (prev_ATR * (window-1) + TR) / window
    alpha = 1.0 / window

    for n in range(N):
        # 找到第一个有效的 TR 开始位置
        first_valid = -1
        for t in range(1, T):
            if not np.isnan(tr[t, n]):
                first_valid = t
                break

        if first_valid < 0 or first_valid + window > T:
            continue  # 数据不足

        # 初始 ATR: 简单平均
        initial_sum = 0.0
        count = 0
        for t in range(first_valid, first_valid + window):
            if not np.isnan(tr[t, n]):
                initial_sum += tr[t, n]
                count += 1

        if count < window // 2:
            continue  # 有效数据不足

        prev_atr = initial_sum / count
        atr[first_valid + window - 1, n] = prev_atr

        # Wilder 平滑
        for t in range(first_valid + window, T):
            if np.isnan(tr[t, n]):
                atr[t, n] = prev_atr  # 保持前值
            else:
                # Wilder: ATR = prev_ATR + alpha * (TR - prev_ATR)
                curr_atr = prev_atr + alpha * (tr[t, n] - prev_atr)
                atr[t, n] = curr_atr
                prev_atr = curr_atr

    return atr


@njit(cache=True)
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


@njit(cache=True)  # ✅ 稳定排序已修复，可安全启用缓存
def vec_backtest_kernel(
    factors_3d,
    close_prices,
    open_prices,  # ✅ 添加开盘价
    high_prices,  # ✅ 添加最高价 (用于止损)
    low_prices,  # ✅ 添加最低价 (用于止损)
    timing_arr,
    vol_regime_arr,  # ✅ v3.1: 波动率体制数组 (T,)
    factor_indices,
    rebalance_schedule,  # ✅ 改用预生成的调仓日程数组
    pos_size,
    # ✅ v4.0: 动态持仓数组 (len=rebalance_schedule), -1表示使用固定pos_size
    dynamic_pos_size_arr,
    initial_capital,
    cost_arr,  # ✅ Exp2: per-ETF cost array (N,), replaces scalar commission_rate
    # ✅ P2: 动态降权参数 (已禁用 - 零杠杆原则)
    target_vol,
    vol_window,
    dynamic_leverage_enabled,
    # ✅ P3: 止损模式选择
    use_atr_stop,  # True = ATR 动态止损, False = 固定百分比止损
    trailing_stop_pct,  # Fixed 模式使用
    atr_arr,  # ATR 矩阵 (T, N)，ATR 模式使用
    atr_multiplier,  # ATR 倍数，ATR 模式使用
    stop_on_rebalance_only,  # ✅ NEW: True = 仅在调仓日检查止损
    # ✅ v3.0: 个股趋势过滤 (MA20)
    individual_trend_arr,  # ✅ NEW: Boolean 矩阵 (T, N)，True=趋势OK可买入
    individual_trend_enabled,  # ✅ NEW: 是否启用个股趋势过滤
    # ✅ P4: 阶梯止盈参数 (最多支持 3 级阶梯)
    profit_ladder_thresholds,  # shape (3,): [0.15, 0.30, inf]
    profit_ladder_stops,  # shape (3,): [0.05, 0.03, 0.08] (Fixed 模式)
    profit_ladder_multipliers,  # shape (3,): [2.0, 1.5, 3.0] (ATR 模式)
    # ✅ P5: 熔断机制参数
    circuit_breaker_day,  # 单日最大跌幅 (e.g. 0.05)
    circuit_breaker_total,  # 总最大回撤 (e.g. 0.20)
    circuit_recovery_days,  # 熔断后冷却天数
    # ✅ P6: 止损冷却期
    cooldown_days,  # 单标的止损后冷却天数
    # ✅ P7: 杠杆上限（零杠杆原则）
    leverage_cap,  # 最大仓位上限 (1.0 = 无杠杆)
    # ✅ Exp1: T+1 Open 执行模式
    use_t1_open,  # True = 用明天的 open 成交, False = 当天 close (COC)
    # ✅ Exp4: Hysteresis + minimum holding period
    delta_rank,  # float: rank01 gap threshold for swap (0 = disabled)
    min_hold_days,  # int: minimum holding days before allowing sell (0 = disabled)
):
    T, N, _ = factors_3d.shape

    cash = initial_capital
    holdings = np.full(N, -1.0)
    entry_prices = np.zeros(N)
    high_water_marks = np.zeros(N)  # ✅ 追踪持仓最高价

    # ✅ P4: 阶梯止盈 - 追踪当前生效的止损率/倍数
    current_stop_pcts = np.full(N, trailing_stop_pct)  # Fixed 模式: 百分比
    current_atr_mults = np.full(N, atr_multiplier)  # ATR 模式: 倍数

    # ✅ P5: 熔断机制状态
    circuit_breaker_active = False
    circuit_breaker_countdown = 0

    # ✅ P6: 冷却期追踪 (每个标的的剩余冷却天数)
    cooldown_remaining = np.zeros(N, dtype=np.int64)

    # ✅ Exp4: 持有天数追踪
    hold_days_arr = np.zeros(N, dtype=np.int64)

    # ✅ Exp1: T+1 Open pending 状态
    pend_active = False
    pend_target = np.zeros(N, dtype=np.bool_)
    pend_buy = np.empty(pos_size, dtype=np.int64)
    pend_buy_cnt = 0
    pend_timing = 1.0

    # ✅ Exp2: turnover & commission tracking
    total_commission_paid = 0.0
    total_turnover_value = 0.0

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

    # ✅ P0: Daily Equity Curve (NEW RETURN VALUE)
    # We need to return the equity curve for further analysis (e.g. Recent_Ret)
    # But numba function can only return fixed tuple.
    # Let's add equity_curve to return tuple.
    # But wait, equity_curve is an array, others are scalars.
    # Numba handles this fine.

    # Reconstruct equity curve from daily returns? No, we didn't store it.
    # We need to store it.
    # We have welford, but not the curve.
    # Let's add equity_curve array.

    # Wait, the caller expects 20 values?
    # ✅ P0: Daily Equity 追踪 (用于 MaxDD, Sharpe, Vol 计算)
    equity_curve = np.full(
        T, initial_capital, dtype=np.float64
    )  # ✅ NEW: Store equity curve (init with capital)

    # Welford 在线算法变量（用于计算 daily return 的均值和方差）
    welford_count = 0
    welford_m2 = 0.0

    # 历史最高净值（用于计算 MaxDD）
    peak_equity = initial_capital
    max_drawdown = 0.0

    # 前一日净值（用于计算 daily return）
    prev_equity = initial_capital

    # 调仓日索引指针
    rebal_ptr = 0

    # ✅ P2: 动态降权 - 环形缓冲区存储滚动日收益率
    returns_buffer = np.zeros(vol_window)
    buffer_ptr = 0
    buffer_filled = 0  # 缓冲区已填充的数量
    current_leverage = 1.0  # 当前动态杠杆率
    leverage_sum = 0.0  # 用于计算平均 leverage
    leverage_count = 0

    # Welford stats initialization
    welford_mean = 0.0
    welford_m2 = 0.0
    welford_count = 0
    prev_equity = initial_capital

    # ✅ P0: 遍历每一天以追踪 daily equity（从 LOOKBACK 开始，与 rebalance_schedule 一致）
    start_day = rebalance_schedule[0] if len(rebalance_schedule) > 0 else 252

    for t in range(start_day, T):
        # 计算当日净值（开盘时刻）
        current_equity = cash
        for n in range(N):
            if holdings[n] > 0.0:
                price = close_prices[t - 1, n]  # 用前一日收盘价估算当日开盘净值
                if not np.isnan(price):
                    current_equity += holdings[n] * price

        # 计算 daily return 并更新 Welford 统计量
        equity_curve[t] = current_equity  # ✅ NEW: Record equity

        if t > start_day:
            daily_return = (
                (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            )
            welford_count += 1
            delta = daily_return - welford_mean
            welford_mean += delta / welford_count
            delta2 = daily_return - welford_mean
            welford_m2 += delta * delta2

            # ✅ P2: 更新环形缓冲区
            if dynamic_leverage_enabled:
                returns_buffer[buffer_ptr] = daily_return
                buffer_ptr = (buffer_ptr + 1) % vol_window
                if buffer_filled < vol_window:
                    buffer_filled += 1

        # 更新历史最高净值和最大回撤
        if current_equity > peak_equity:
            peak_equity = current_equity
        current_dd = (
            (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
        )
        if current_dd > max_drawdown:
            max_drawdown = current_dd

        prev_equity = current_equity

        # 检查是否为调仓日
        is_rebalance_day = False
        if rebal_ptr < len(rebalance_schedule) and rebalance_schedule[rebal_ptr] == t:
            is_rebalance_day = True
            rebal_ptr += 1

        # --- 2. 日内止损逻辑 (High/Low时刻) ---
        # ✅ P3: 即使是调仓日，买入后也可能当天触发止损 (如果买入价是 Open/Close，这里假设 Close 买入则当天不触发，除非 Low < Close)
        # 但 VEC 逻辑是 Close 买入，所以当天 Low 已经是过去式了？
        # 不，VEC 这里的逻辑是：
        # 如果是调仓日：
        #   1. 卖出旧的 (Close)
        #   2. 买入新的 (Close)
        #   这意味着调仓日的 High/Low 无法用于新持仓的止损（因为是在 Close 买入的）。
        #   但对于旧持仓（如果没卖出），或者非调仓日的持仓，需要检查止损。

        # 如果是调仓日，且持仓被卖出了，holdings[n] 已经是 -1，不会触发止损。
        # 如果是调仓日，且持仓保留了，holdings[n] > 0，需要检查止损。
        # 如果是非调仓日，holdings[n] > 0，需要检查止损。

        # 关键点：调仓日的执行价格是 Close。
        # 如果我们在 Close 买入，那么当天的 High/Low 对新持仓无效。
        # 如果我们在 Close 卖出，那么当天的 High/Low 对旧持仓有效吗？
        # 如果 Low < Stop，应该在盘中触发止损，而不是等到 Close 正常调仓卖出。
        # 所以止损检查应该在调仓逻辑之前？
        # 如果止损触发了，就卖出，然后调仓逻辑会看到 holdings[n] = -1 (或者 0)，然后可能又买回来？
        # 这是一个复杂点。
        # 简化逻辑：
        # 1. 每天先检查止损。如果触发，强制卖出。
        # 2. 然后执行调仓逻辑。如果止损卖出了，调仓逻辑可能会再次买入（如果它还在 Top K）。
        #    这符合逻辑：止损是风险控制，如果信号依然强，可能会买回（但通常有冷却期，这里暂不实现冷却）。
        #    或者，如果止损了，当天就不买回？

        # 让我们把止损逻辑放在调仓逻辑之前。

        # 修正循环结构：
        # for t in range(start_day, T):
        #    1. Update Equity (Open)
        #    2. Check Stop Loss (High/Low) -> Sell if triggered
        #    3. Check Rebalance (Close) -> Sell/Buy

        # 但是 VEC 原有逻辑是混合的。
        # 让我们在调仓逻辑 *之前* 插入止损检查。

        # ✅ P5: 熔断检查 - 单日跌幅和总回撤
        if circuit_breaker_day > 0.0 or circuit_breaker_total > 0.0:
            # 计算单日跌幅
            if t > start_day:
                day_return = (
                    (current_equity - prev_equity) / prev_equity
                    if prev_equity > 0
                    else 0.0
                )
                # 单日熔断检查
                if circuit_breaker_day > 0.0 and day_return < -circuit_breaker_day:
                    circuit_breaker_active = True
                    circuit_breaker_countdown = circuit_recovery_days
                # 总回撤熔断检查
                if circuit_breaker_total > 0.0 and current_dd > circuit_breaker_total:
                    circuit_breaker_active = True
                    circuit_breaker_countdown = circuit_recovery_days

            # 熔断恢复倒计时
            if circuit_breaker_active:
                if circuit_breaker_countdown > 0:
                    circuit_breaker_countdown -= 1
                else:
                    circuit_breaker_active = False

        # ✅ P6: 冷却期倒计时
        for n in range(N):
            if cooldown_remaining[n] > 0:
                cooldown_remaining[n] -= 1

        # ✅ Exp4: 持有天数递增 (每个交易日递增, 在买卖逻辑前)
        for n in range(N):
            if holdings[n] > 0.0:
                hold_days_arr[n] += 1

        # ✅ Exp1: 执行 T+1 Open 挂单 (在当天开盘执行昨日的调仓决策)
        if use_t1_open and pend_active:
            # --- 卖出不在目标集的持仓 (at open[t]) ---
            for n in range(N):
                if holdings[n] > 0.0 and not pend_target[n]:
                    price = open_prices[t, n]
                    if np.isnan(price) or price <= 0.0:
                        price = close_prices[t - 1, n]
                    sell_cost = holdings[n] * price * cost_arr[n]
                    total_commission_paid += sell_cost
                    total_turnover_value += holdings[n] * price
                    proceeds = holdings[n] * price - sell_cost
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
                    hold_days_arr[n] = 0  # ✅ Exp4

            # --- 计算开盘时组合价值 (用 open[t] 估值) ---
            pend_val = cash
            pend_kept = 0.0
            for n in range(N):
                if holdings[n] > 0.0:
                    p = open_prices[t, n]
                    if np.isnan(p):
                        p = close_prices[t - 1, n]
                    v = holdings[n] * p
                    pend_val += v
                    pend_kept += v

            # --- 买入新标的 (at open[t]) ---
            pend_new_cnt = 0
            for k in range(pend_buy_cnt):
                idx = pend_buy[k]
                if holdings[idx] < 0.0 and cooldown_remaining[idx] == 0:
                    new_targets[pend_new_cnt] = idx
                    pend_new_cnt += 1

            if pend_new_cnt > 0:
                target_exposure = pend_val * pend_timing
                available_for_new = target_exposure - pend_kept
                if available_for_new < 0.0:
                    available_for_new = 0.0
                tpv = available_for_new / pend_new_cnt
                if tpv > 0.0:
                    for k in range(pend_new_cnt):
                        idx = new_targets[k]
                        price = open_prices[t, idx]
                        if np.isnan(price) or price <= 0.0:
                            continue
                        effective_tpv = tpv / (1.0 + cost_arr[idx])
                        shares = effective_tpv / price
                        cost = shares * price * (1.0 + cost_arr[idx])
                        target_value_total += tpv
                        target_shares_total += shares
                        if cash >= cost - 1e-5 and cost > 0.0:
                            actual_cost = cost if cost <= cash else cash
                            actual_shares = actual_cost / (
                                price * (1.0 + cost_arr[idx])
                            )
                            buy_comm = actual_shares * price * cost_arr[idx]
                            total_commission_paid += buy_comm
                            total_turnover_value += actual_shares * price
                            filled_shares_total += actual_shares
                            filled_value_total += actual_shares * price
                            cash -= actual_cost
                            holdings[idx] = shares
                            entry_prices[idx] = price
                            high_water_marks[idx] = price
                            current_stop_pcts[idx] = trailing_stop_pct
                            current_atr_mults[idx] = atr_multiplier
                            hold_days_arr[idx] = 1  # ✅ Exp4
            pend_active = False

        # ✅ P3/P4: 动态止损 + 阶梯止盈
        # 🔧 修复1: HWM 滞后更新 - 止损检查使用昨日 HWM，避免前视偏差
        # 🔧 修复2: 精确执行价 - 使用止损价或开盘价（如跳空低开）
        # ✅ v1.2: 支持 ATR 动态止损模式
        # ✅ v1.3: 支持仅在调仓日检查止损（与策略节奏一致）
        should_check_stop = (use_atr_stop and atr_multiplier > 0.0) or (
            not use_atr_stop and trailing_stop_pct > 0.0
        )

        # ⚠️ 核心修改: 只在调仓日检查止损（如启用）
        if should_check_stop and (not stop_on_rebalance_only or is_rebalance_day):
            for n in range(N):
                if holdings[n] > 0.0:
                    # 🔧 修复1: 先用昨日 HWM 计算止损线，再更新 HWM
                    # 这样避免"先跌后涨"导致的未来函数问题
                    prev_hwm = high_water_marks[n]  # 昨日 HWM

                    # ✅ P4: 阶梯止盈 - 根据收益率动态调整止损参数
                    # 使用昨日 HWM 计算收益率
                    current_return = (
                        (prev_hwm - entry_prices[n]) / entry_prices[n]
                        if entry_prices[n] > 0
                        else 0.0
                    )
                    for ladder_idx in range(3):
                        if current_return >= profit_ladder_thresholds[ladder_idx]:
                            # ATR 模式: 收紧倍数
                            if use_atr_stop:
                                if (
                                    profit_ladder_multipliers[ladder_idx]
                                    < current_atr_mults[n]
                                ):
                                    current_atr_mults[n] = profit_ladder_multipliers[
                                        ladder_idx
                                    ]
                            # Fixed 模式: 收紧百分比
                            else:
                                if (
                                    profit_ladder_stops[ladder_idx]
                                    < current_stop_pcts[n]
                                ):
                                    current_stop_pcts[n] = profit_ladder_stops[
                                        ladder_idx
                                    ]

                    # ✅ 计算止损价 (ATR 模式 vs Fixed 模式)
                    if use_atr_stop:
                        # ATR 模式: stop_price = HWM - (multiplier × ATR)
                        # 使用 t-1 日的 ATR (避免前视偏差)
                        prev_atr = atr_arr[t - 1, n] if t > 0 else 0.0
                        if np.isnan(prev_atr) or prev_atr <= 0.0:
                            # ATR 无效时跳过止损检查
                            curr_high = high_prices[t, n]
                            if (
                                not np.isnan(curr_high)
                                and curr_high > high_water_marks[n]
                            ):
                                high_water_marks[n] = curr_high
                            continue
                        stop_price = prev_hwm - (current_atr_mults[n] * prev_atr)
                    else:
                        # Fixed 模式: stop_price = HWM × (1 - stop_pct)
                        stop_price = prev_hwm * (1.0 - current_stop_pcts[n])

                    curr_low = low_prices[t, n]
                    curr_open = open_prices[t, n]

                    if not np.isnan(curr_low) and curr_low < stop_price:
                        # 触发止损
                        # 🔧 修复2: 精确执行价格
                        # 情况1: 跳空低开 (Open < Stop) → 按 Open 执行
                        # 情况2: 盘中触发 (Open >= Stop, Low < Stop) → 按 Stop 执行
                        if not np.isnan(curr_open) and curr_open < stop_price:
                            exec_price = curr_open  # 跳空低开，按开盘价止损
                        else:
                            exec_price = stop_price  # 盘中触发，按止损价执行

                        # 确保执行价不低于 Low (防止数据异常)
                        if not np.isnan(curr_low):
                            exec_price = max(exec_price, curr_low)

                        sl_cost = holdings[n] * exec_price * cost_arr[n]
                        total_commission_paid += sl_cost
                        total_turnover_value += holdings[n] * exec_price
                        proceeds = holdings[n] * exec_price - sl_cost
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
                        current_stop_pcts[n] = trailing_stop_pct  # 重置止损率
                        current_atr_mults[n] = atr_multiplier  # 重置 ATR 倍数
                        # ✅ P6: 设置冷却期
                        cooldown_remaining[n] = cooldown_days
                        hold_days_arr[n] = 0  # ✅ Exp4
                    else:
                        # 未触发止损，更新 HWM 供明日使用
                        curr_high = high_prices[t, n]
                        if not np.isnan(curr_high) and curr_high > high_water_marks[n]:
                            high_water_marks[n] = curr_high

        # --- 3. 调仓逻辑 (Close时刻) ---
        if is_rebalance_day:
            # ✅ P2: 在调仓日计算动态 leverage
            if (
                dynamic_leverage_enabled and buffer_filled >= vol_window // 2
            ):  # 至少有半窗口数据
                # 计算缓冲区内的标准差
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
                    realized_vol = daily_std * np.sqrt(252.0)  # 年化
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

            # ✅ v4.0: 动态持仓数
            effective_pos_size = pos_size
            if dynamic_pos_size_arr[rebal_ptr - 1] > 0:
                effective_pos_size = dynamic_pos_size_arr[rebal_ptr - 1]

            buy_count = 0
            if valid >= effective_pos_size:
                # ✅ 使用稳定排序，确保 numba/Python 结果一致
                top_indices = stable_topk_indices(combined_score, effective_pos_size)
                for k in range(len(top_indices)):
                    idx = top_indices[k]
                    if combined_score[idx] == -np.inf:
                        break
                    target_set[idx] = True
                    buy_order[buy_count] = idx
                    buy_count += 1

            # ✅ Exp4: Apply hysteresis filter (max 1 swap per rebalance)
            if (delta_rank > 0.0 or min_hold_days > 0) and buy_count > 0:
                h_mask = np.zeros(N, dtype=np.bool_)
                for n in range(N):
                    h_mask[n] = holdings[n] > 0.0
                target_mask = apply_hysteresis(
                    combined_score, h_mask, hold_days_arr,
                    top_indices, effective_pos_size,
                    delta_rank, min_hold_days,
                )
                # Overwrite target_set and buy_order from hysteresis result
                for n in range(N):
                    target_set[n] = target_mask[n]
                buy_count = 0
                for n in range(N):
                    if target_set[n]:
                        buy_order[buy_count] = n
                        buy_count += 1

            # ✅ 改用 t-1 日择时信号 (timing_arr 已在 main 中 shift(1)，故此处用 t 即为 t-1 日信号)
            # ✅ P2: 应用动态 leverage，受 leverage_cap 限制
            # ✅ P7: 零杠杆原则 - 确保不超过 leverage_cap
            effective_leverage = min(current_leverage, leverage_cap)
            timing_ratio = timing_arr[t] * effective_leverage

            # ✅ P5: 如果熔断激活，不允许新建仓位（只允许平仓）
            if circuit_breaker_active:
                timing_ratio = 0.0  # 清空目标仓位

            # ✅ Exp1: T+1 Open → 存储 pending, 下一天用 open 成交
            if use_t1_open:
                for n in range(N):
                    pend_target[n] = target_set[n]
                for k in range(buy_count):
                    pend_buy[k] = buy_order[k]
                pend_buy_cnt = buy_count
                pend_timing = timing_ratio
                pend_active = True
            else:
                # --- COC 模式: 当天 close 即时成交 (原逻辑) ---
                for n in range(N):
                    if holdings[n] > 0.0 and not target_set[n]:
                        price = close_prices[t, n]
                        coc_sell_cost = holdings[n] * price * cost_arr[n]
                        total_commission_paid += coc_sell_cost
                        total_turnover_value += holdings[n] * price
                        proceeds = holdings[n] * price - coc_sell_cost
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
                        hold_days_arr[n] = 0  # ✅ Exp4

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

                    target_pos_value = available_for_new / new_count

                    if target_pos_value > 0.0:
                        for k in range(new_count):
                            idx = new_targets[k]
                            price = close_prices[t, idx]
                            if np.isnan(price) or price <= 0.0:
                                continue

                            effective_tpv_coc = target_pos_value / (1.0 + cost_arr[idx])
                            shares = effective_tpv_coc / price
                            cost = shares * price * (1.0 + cost_arr[idx])
                            target_value_total += target_pos_value
                            target_shares_total += shares

                            if cash >= cost - 1e-5 and cost > 0.0:
                                actual_cost = cost if cost <= cash else cash
                                actual_shares = actual_cost / (
                                    price * (1.0 + cost_arr[idx])
                                )
                                coc_buy_comm = actual_shares * price * cost_arr[idx]
                                total_commission_paid += coc_buy_comm
                                total_turnover_value += actual_shares * price
                                filled_shares_total += actual_shares
                                filled_value_total += actual_shares * price
                                cash -= actual_cost
                                holdings[idx] = shares
                                entry_prices[idx] = price
                                high_water_marks[idx] = price
                                current_stop_pcts[idx] = trailing_stop_pct
                                current_atr_mults[idx] = atr_multiplier
                                hold_days_arr[idx] = 1  # ✅ Exp4
    final_value = cash
    for n in range(N):
        if holdings[n] > 0.0:
            # ✅ 改用收盘价计算持仓市值（与 BT getvalue() 对齐）
            # 注意: 不扣清仓手续费，因为 BT 的 getvalue() 也不扣
            price = close_prices[T - 1, n]
            if np.isnan(price):
                price = entry_prices[n]

            final_value += holdings[n] * price  # 与 BT 对齐：mark-to-market

            # ⚠️ 注意: 最终清仓不计入交易次数统计
            # BT 的 getvalue() 只是 mark-to-market，不产生实际交易
            # 所以这里只计算浮盈浮亏用于 win_rate 和 profit_factor（仅作参考）
            # 不再增加 wins/losses 计数

    # ✅ P0: 最后一天的 daily return 也要纳入统计
    if final_value != prev_equity and prev_equity > 0:
        daily_return = (final_value - prev_equity) / prev_equity
        welford_count += 1
        delta = daily_return - welford_mean
        welford_mean += delta / welford_count
        delta2 = daily_return - welford_mean
        welford_m2 += delta * delta2

    # 更新最终的 peak 和 MaxDD
    if final_value > peak_equity:
        peak_equity = final_value
    final_dd = (peak_equity - final_value) / peak_equity if peak_equity > 0 else 0.0
    if final_dd > max_drawdown:
        max_drawdown = final_dd

    num_trades = wins + losses
    total_return = (final_value - initial_capital) / initial_capital
    win_rate = wins / num_trades if num_trades > 0 else 0.0

    if losses > 0:
        avg_win = total_win_pnl / max(wins, 1)
        avg_loss = total_loss_pnl / losses
        profit_factor = avg_win / max(avg_loss, 0.0001)
    else:
        # ✅ P3: Semantically correct — infinite profit factor when no losses
        profit_factor = np.inf

    # ✅ P0: 计算风险指标
    # 年化收益率（假设 252 交易日/年）
    trading_days = T - start_day
    years = trading_days / 252.0 if trading_days > 0 else 1.0

    # 🛡️ Safety Check: Handle potential overflow or invalid values
    try:
        annual_return = (
            (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        )
        if np.isinf(annual_return) or np.isnan(annual_return):
            annual_return = -0.99
    except:
        annual_return = -0.99

    # 年化波动率（daily std * sqrt(252)）
    if welford_count > 1:
        daily_variance = welford_m2 / (welford_count - 1)
        daily_std = np.sqrt(daily_variance) if daily_variance > 0 else 0.0
        annual_volatility = daily_std * np.sqrt(252.0)
    else:
        annual_volatility = 0.0

    # Sharpe Ratio（假设无风险利率为 0）
    # 🛡️ Safety Check: Clamp Sharpe to reasonable range
    sharpe_ratio = (
        annual_return / annual_volatility if annual_volatility > 0.0001 else 0.0
    )
    if np.isinf(sharpe_ratio) or np.isnan(sharpe_ratio):
        sharpe_ratio = 0.0
    elif sharpe_ratio > 20.0:  # Cap extreme sharpe
        sharpe_ratio = 20.0
    elif sharpe_ratio < -20.0:
        sharpe_ratio = -20.0

    # Calmar Ratio
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0.0001 else 0.0
    if np.isinf(calmar_ratio) or np.isnan(calmar_ratio):
        calmar_ratio = 0.0

    # ✅ P2: 平均动态 leverage
    avg_leverage = leverage_sum / leverage_count if leverage_count > 0 else 1.0

    return (
        equity_curve,  # ✅ NEW: Return equity curve
        total_return,
        win_rate,
        profit_factor,
        num_trades,
        target_value_total,
        filled_value_total,
        target_shares_total,
        filled_shares_total,
        # ✅ P0: 新增风险指标
        max_drawdown,
        annual_return,
        annual_volatility,
        sharpe_ratio,
        calmar_ratio,
        # ✅ P2: 动态降权诊断
        avg_leverage,
        # ✅ Exp2: 成本追踪
        total_commission_paid,
        total_turnover_value,
    )


def run_vec_backtest(
    factors_3d,
    close_prices,
    open_prices,
    high_prices,
    low_prices,
    timing_arr,
    factor_indices,
    # ✅ P0: 添加配置参数
    freq,
    pos_size,
    initial_capital,
    commission_rate,  # ✅ Exp2: legacy scalar, used when cost_arr not provided
    lookback,
    cost_arr=None,  # ✅ Exp2: per-ETF cost array (N,), overrides commission_rate
    target_vol=0.20,
    vol_window=20,
    dynamic_leverage_enabled=False,
    vol_regime_arr=None,  # ✅ v3.1: 波动率体制数组
    # ✅ v4.0: 动态持仓
    dynamic_pos_size_arr=None,
    # ✅ v1.2: ATR 动态止损参数
    use_atr_stop=False,
    trailing_stop_pct=0.0,
    atr_arr=None,  # ATR 矩阵 (T, N)
    atr_multiplier=3.0,
    stop_on_rebalance_only=False,  # ✅ v1.3: 仅在调仓日检查止损
    # ✅ v3.0: 个股趋势过滤参数
    individual_trend_arr=None,  # ✅ NEW: 趋势状态矩阵 (T, N)
    individual_trend_enabled=False,  # ✅ NEW: 是否启用个股趋势过滤
    # ✅ P4: 阶梯止盈参数
    profit_ladders=None,
    # ✅ P5: 熔断机制参数
    circuit_breaker_day=0.0,
    circuit_breaker_total=0.0,
    circuit_recovery_days=5,
    # ✅ P6: 冷却期参数
    cooldown_days=0,
    # ✅ P7: 杠杆上限
    leverage_cap=1.0,
    # ✅ Exp1: T+1 Open 执行模式
    use_t1_open=False,
    # ✅ Exp4: Hysteresis + minimum holding period
    delta_rank=0.0,  # rank01 gap threshold (0 = disabled)
    min_hold_days=0,  # minimum holding days (0 = disabled)
):
    """运行单个策略的 VEC 回测，并返回舍入诊断信息和风险指标。

    ✅ P0 修正: 所有策略参数从外部传入，不使用默认值
    ✅ Exp1: use_t1_open=True 时用 open[t+1] 成交 (T+1 Open 执行模式)
    """
    factor_indices_arr = np.array(factor_indices, dtype=np.int64)
    T, N = factors_3d.shape[:2]

    # ✅ Exp2: 构建 per-ETF 成本数组
    if cost_arr is not None:
        cost_arr_internal = np.asarray(cost_arr, dtype=np.float64)
    else:
        # Legacy fallback: uniform commission_rate for all ETFs
        cost_arr_internal = np.full(N, commission_rate, dtype=np.float64)

    # ✅ v1.2: 处理 ATR 矩阵
    if atr_arr is None:
        # 如果没有传入 ATR，创建一个全零矩阵 (不会被使用，因为 use_atr_stop=False)
        atr_arr_internal = np.zeros((T, N), dtype=np.float64)
    else:
        atr_arr_internal = atr_arr

    # ✅ v3.0: 处理个股趋势矩阵
    if individual_trend_arr is None:
        # 如果没有传入，创建全 True 矩阵（所有标的都可买入）
        individual_trend_arr_internal = np.ones((T, N), dtype=np.bool_)
    else:
        individual_trend_arr_internal = individual_trend_arr

    # ✅ P4: 处理阶梯止盈参数 (最多支持 3 级)
    if profit_ladders is None or len(profit_ladders) == 0:
        # 默认不启用阶梯止盈
        profit_ladder_thresholds = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        profit_ladder_stops = np.array(
            [trailing_stop_pct, trailing_stop_pct, trailing_stop_pct], dtype=np.float64
        )
        profit_ladder_multipliers = np.array(
            [atr_multiplier, atr_multiplier, atr_multiplier], dtype=np.float64
        )
    else:
        thresholds = []
        stops = []
        multipliers = []
        for ladder in profit_ladders[:3]:  # 最多取前 3 级
            thresholds.append(ladder.get("threshold", np.inf))
            stops.append(ladder.get("new_stop", trailing_stop_pct))
            multipliers.append(ladder.get("new_multiplier", atr_multiplier))
        # 补齐到 3 级
        while len(thresholds) < 3:
            thresholds.append(np.inf)
            stops.append(trailing_stop_pct)
            multipliers.append(atr_multiplier)
        profit_ladder_thresholds = np.array(thresholds, dtype=np.float64)
        profit_ladder_stops = np.array(stops, dtype=np.float64)
        profit_ladder_multipliers = np.array(multipliers, dtype=np.float64)

    # ✅ 使用 ensure_price_views 验证和回退 open_prices
    _, open_arr, close_arr = ensure_price_views(
        close_prices,
        open_prices,
        copy_if_missing=True,
        warn_if_copied=True,
        validate=True,
        min_valid_index=lookback,  # 使用传入的 lookback
    )

    # 简单的视图确保 (High/Low)
    high_arr = high_prices
    low_arr = low_prices

    # ✅ 使用共享 helper 生成调仓日程（与 BT 引擎一致）
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T,
        lookback_window=lookback,  # 使用传入的 lookback
        freq=freq,  # 使用传入的 freq
    )
    # ✅ v3.1: 处理 Vol Regime
    if vol_regime_arr is None:
        vol_regime_arr = np.ones(T, dtype=np.float64)

    start_day = rebalance_schedule[0] if len(rebalance_schedule) > 0 else lookback

    # ✅ v4.0: 动态持仓数组
    if dynamic_pos_size_arr is None:
        # 全部使用 -1 表示使用固定 pos_size
        dynamic_pos_size_arr_internal = np.full(
            len(rebalance_schedule), -1, dtype=np.int64
        )
    else:
        dynamic_pos_size_arr_internal = np.asarray(
            dynamic_pos_size_arr, dtype=np.int64
        )

    (
        equity_curve,
        total_return,
        win_rate,
        profit_factor,
        num_trades,
        target_value_total,
        filled_value_total,
        target_shares_total,
        filled_shares_total,
        # ✅ P0: 新增风险指标
        max_drawdown,
        annual_return,
        annual_volatility,
        sharpe_ratio,
        calmar_ratio,
        # ✅ P2: 动态降权诊断
        avg_leverage,
        # ✅ Exp2: 成本追踪
        total_commission_paid,
        total_turnover_value,
    ) = vec_backtest_kernel(
        factors_3d,
        close_arr,
        open_arr,
        high_arr,
        low_arr,
        timing_arr,
        vol_regime_arr,  # ✅ v3.1
        factor_indices_arr,
        rebalance_schedule,  # ✅ 传入调仓日程数组
        pos_size,  # 使用传入的 pos_size
        dynamic_pos_size_arr_internal,  # ✅ v4.0: 动态持仓
        initial_capital,  # 使用传入的 initial_capital
        cost_arr_internal,  # ✅ Exp2: per-ETF cost array
        # ✅ P2: 动态降权参数 (已禁用 - 零杠杆原则)
        target_vol,
        vol_window,
        dynamic_leverage_enabled,
        # ✅ v1.2: ATR 动态止损参数
        use_atr_stop,
        trailing_stop_pct,
        atr_arr_internal,
        atr_multiplier,
        stop_on_rebalance_only,  # ✅ v1.3: 传入调仓日止损标志
        # ✅ v3.0: 个股趋势过滤参数
        individual_trend_arr_internal,
        individual_trend_enabled,
        # ✅ P4: 阶梯止盈参数
        profit_ladder_thresholds,
        profit_ladder_stops,
        profit_ladder_multipliers,
        # ✅ P5: 熔断机制参数
        circuit_breaker_day,
        circuit_breaker_total,
        circuit_recovery_days,
        # ✅ P6: 冷却期参数
        cooldown_days,
        # ✅ P7: 杠杆上限
        leverage_cap,
        # ✅ Exp1: T+1 Open
        use_t1_open,
        # ✅ Exp4: Hysteresis
        delta_rank,
        min_hold_days,
    )

    rounding_diag = {
        "target_value_total": target_value_total,
        "filled_value_total": filled_value_total,
        "target_shares_total": target_shares_total,
        "filled_shares_total": filled_shares_total,
    }

    aligned_metrics = compute_aligned_metrics(equity_curve, start_idx=start_day)

    # ✅ Exp2: 计算换手率和成本拖拽
    trading_days_total = T - start_day
    years_total = trading_days_total / 252.0 if trading_days_total > 0 else 1.0
    turnover_ann = total_turnover_value / (initial_capital * years_total) if years_total > 0 else 0.0
    final_value_est = initial_capital * (1.0 + total_return)
    gross_pnl = final_value_est - initial_capital
    cost_drag = total_commission_paid / max(gross_pnl, 1.0) if gross_pnl > 0 else 0.0

    # ✅ P0: 风险指标字典
    risk_metrics = {
        "max_drawdown": max_drawdown,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "calmar_ratio": calmar_ratio,
        # ✅ P2: 动态降权诊断
        "avg_leverage": avg_leverage,
        # ✅ 对齐后的统一指标
        "aligned_return": aligned_metrics["aligned_return"],
        "aligned_sharpe": aligned_metrics["aligned_sharpe"],
        # ✅ Exp2: 成本诊断
        "turnover_ann": turnover_ann,
        "cost_drag": cost_drag,
        "total_commission_paid": total_commission_paid,
    }

    return (
        equity_curve,
        total_return,
        win_rate,
        profit_factor,
        num_trades,
        rounding_diag,
        risk_metrics,
    )


def main():
    print("=" * 80)
    print("批量 VEC 回测：遍历全部 WFO 组合")
    print("=" * 80)

    # 0. ✅ P0: 立即加载配置文件
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))
    print(f"🔒 参数冻结校验通过 (version={frozen.version})")

    # ✅ P0: 从配置读取回测参数（强制依赖配置，无默认值）
    backtest_config = config.get("backtest", {})
    FREQ = backtest_config.get("freq")
    POS_SIZE = backtest_config.get("pos_size")
    LOOKBACK = backtest_config.get("lookback")
    INITIAL_CAPITAL = float(backtest_config.get("initial_capital"))
    COMMISSION_RATE = float(backtest_config.get("commission_rate"))

    # 验证必需参数
    if FREQ is None or POS_SIZE is None or LOOKBACK is None:
        raise ValueError("配置文件缺少必需参数: freq, pos_size, lookback")

    print(f"✅ 回测参数 (从配置读取):")
    print(f"   FREQ: {FREQ}")
    print(f"   POS_SIZE: {POS_SIZE}")
    print(f"   LOOKBACK: {LOOKBACK}")
    print(f"   INITIAL_CAPITAL: {INITIAL_CAPITAL:,.0f}")
    print(f"   COMMISSION_RATE: {COMMISSION_RATE*10000:.1f} bp")

    # ✅ Exp1: 执行模型
    from etf_strategy.core.execution_model import load_execution_model
    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open
    print(f"   EXECUTION_MODEL: {exec_model.mode}")

    # ✅ Exp2: 成本模型
    from etf_strategy.core.cost_model import load_cost_model, build_cost_array
    from etf_strategy.core.frozen_params import FrozenETFPool
    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)
    print(f"   COST_MODEL: mode={cost_model.mode}, tier={cost_model.tier}")

    # 1. 加载 WFO 结果
    # ✅ 优先查找 run_* 目录 (True WFO)，排除 symlink
    wfo_dirs = sorted(
        [
            d
            for d in (ROOT / "results").glob("run_*")
            if d.is_dir() and not d.is_symlink()
        ]
    )

    if not wfo_dirs:
        wfo_dirs = sorted((ROOT / "results").glob("unified_wfo_*"))

    if not wfo_dirs:
        print("❌ 未找到 WFO 结果目录")
        return
    latest_wfo = wfo_dirs[-1]

    # 优先加载 top100_by_ic.parquet (WFO 输出)，其次 all_combos.parquet
    combos_path = latest_wfo / "top100_by_ic.parquet"
    if not combos_path.exists():
        combos_path = latest_wfo / "all_combos.parquet"

    if not combos_path.exists():
        print(f"❌ 未找到 {combos_path}")
        return

    df_combos = pd.read_parquet(combos_path)
    print(f"✅ 加载 WFO 结果 ({latest_wfo.name})：{len(df_combos)} 个组合")

    # 2. 加载数据
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # 3. 计算因子 (带缓存)
    factor_cache = FactorCache(cache_dir=Path(config["data"].get("cache_dir") or ".cache"))
    cached = factor_cache.get_or_compute(
        ohlcv=ohlcv,
        config=config,
        data_dir=loader.data_dir,
    )
    std_factors = cached["std_factors"]
    factor_names = cached["factor_names"]
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    T = len(dates)
    N = len(etf_codes)

    # ✅ Exp2: 构建 per-ETF 成本数组
    COST_ARR = build_cost_array(cost_model, list(etf_codes), qdii_set)
    tier = cost_model.active_tier
    print(f"   COST_ARR: A股={tier.a_share*10000:.0f}bp, QDII={tier.qdii*10000:.0f}bp")
    factors_3d = cached["factors_3d"]
    # 价格数据处理：仅 ffill（bfill 会将未来价格回填到过去，造成 lookahead bias）
    # ✅ FIX: 部分 ETF 在 lookback 后才上市，ffill 无法填充上市前的 NaN
    # 用 1.0 兜底填充（上市前 ETF 的 factor score 也是 NaN，不会被选中交易，填充值不影响结果）
    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    # ✅ P1: 从配置文件读取择时参数
    # ═══════════════════════════════════════════════════════════════════════════
    # 择时配置 v3.0 - 双重择时机制
    # ═══════════════════════════════════════════════════════════════════════════
    timing_config = config.get("backtest", {}).get("timing", {})
    timing_type = timing_config.get("type", "light_timing")

    # 旧版参数 (兼容 light_timing)
    extreme_threshold = timing_config.get("extreme_threshold", -0.4)
    extreme_position = timing_config.get("extreme_position", 0.3)

    # ✅ v3.0: 双重择时参数
    index_timing_config = timing_config.get("index_timing", {})
    individual_timing_config = timing_config.get("individual_timing", {})

    index_timing_enabled = index_timing_config.get("enabled", False)
    individual_timing_enabled = individual_timing_config.get("enabled", False)

    if timing_type == "dual_timing" and (
        index_timing_enabled or individual_timing_enabled
    ):
        # ✅ 使用双重择时模块
        print(f"✅ 择时模式: dual_timing (大盘 + 个股)")

        dual_timing = DualTimingModule(
            index_ma_window=index_timing_config.get("window", 200),
            bear_position=index_timing_config.get("bear_position", 0.1),
            index_symbol=index_timing_config.get("symbol", "market_avg"),
            individual_ma_window=individual_timing_config.get("window", 20),
        )

        # 计算所有择时信号
        timing_signals = dual_timing.compute_all_signals(ohlcv["close"])

        # 层级 1: 大盘择时 (仓位系数)
        if index_timing_enabled:
            timing_arr_raw = (
                timing_signals["index_timing"].reindex(dates).fillna(1.0).values
            )
            stats = timing_signals["stats"]
            print(
                f"   大盘择时 (MA{index_timing_config.get('window', 200)}): "
                f"熊市 {stats['bear_days']:.0f} 天 ({stats['bear_ratio']:.1f}%), "
                f"熊市仓位 {index_timing_config.get('bear_position', 0.1)*100:.0f}%"
            )
        else:
            timing_arr_raw = np.ones(T, dtype=np.float64)
            print(f"   大盘择时: 禁用")

        # 层级 2: 个股趋势 (状态矩阵)
        if individual_timing_enabled:
            individual_trend_df = timing_signals["individual_trend"].reindex(dates)
            individual_trend_arr = individual_trend_df[etf_codes].values.astype(
                np.bool_
            )
            trend_ok_pct = timing_signals["stats"]["avg_trend_ok_pct"]
            print(
                f"   个股趋势 (MA{individual_timing_config.get('window', 20)}): "
                f"平均 {trend_ok_pct:.1f}% 标的趋势良好"
            )
        else:
            individual_trend_arr = None
            individual_timing_enabled = False
            print(f"   个股趋势: 禁用")
    else:
        # ✅ 兼容旧版 light_timing
        print(f"✅ 择时模式: light_timing (旧版兼容)")
        print(f"   参数: threshold={extreme_threshold}, position={extreme_position}")

        timing_module = LightTimingModule(
            extreme_threshold=extreme_threshold,
            extreme_position=extreme_position,
        )
        timing_arr_raw = (
            timing_module.compute_position_ratios(ohlcv["close"])
            .reindex(dates)
            .fillna(1.0)
            .values
        )
        individual_trend_arr = None
        individual_timing_enabled = False

    # ✅ 使用共享 helper shift_timing_signal: t 日调仓用 t-1 日的择时信号
    timing_arr = shift_timing_signal(timing_arr_raw)

    # ✅ v3.2: Regime gate（可选），通过缩放 timing_arr 实现统一降仓/停跑
    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=config.get("backtest", {})
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64)).astype(
        np.float64
    )
    if bool(config.get("backtest", {}).get("regime_gate", {}).get("enabled", False)):
        s = gate_stats(gate_arr)
        print(
            f"✅ Regime gate enabled: mean={s['mean']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}"
        )

    # ✅ v3.2: vol_regime 已由 compute_regime_gate_arr() 统一处理并融入 timing_arr
    # 不再单独计算 vol_regime_arr（kernel 从未读取该参数），避免维护两套相同逻辑
    vol_regime_arr = None  # run_vec_backtest 默认填充 ones

    print(f"✅ 数据加载完成：{T} 天 × {N} 只 ETF × {len(factor_names)} 个因子")

    # ✅ P2: 从配置文件读取动态降权参数 (已禁用 - 零杠杆原则)
    risk_config = config.get("backtest", {}).get("risk_control", {})
    dyn_lev_config = risk_config.get("dynamic_leverage", {})
    dynamic_leverage_enabled = dyn_lev_config.get("enabled", False)
    target_vol = dyn_lev_config.get("target_vol", 0.20)
    vol_window = dyn_lev_config.get("vol_window", 20)

    # ✅ v1.2: 止损模式选择 ("atr" 或 "fixed")
    stop_method = risk_config.get("stop_method", "fixed")
    use_atr_stop = stop_method == "atr"

    # Fixed 模式参数
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.08)

    # ATR 模式参数
    atr_window = risk_config.get("atr_window", 14)
    atr_multiplier = risk_config.get("atr_multiplier", 3.0)

    # ✅ v1.3: 调仓日止损标志
    stop_on_rebalance_only = risk_config.get("stop_check_on_rebalance_only", False)

    # ✅ v1.2: 计算 ATR 矩阵 (如果使用 ATR 模式)
    if use_atr_stop:
        print(
            f"✅ 止损模式: ATR 动态 (window={atr_window}, multiplier={atr_multiplier}x)"
        )
        atr_arr = calculate_atr(
            high_prices, low_prices, close_prices, window=atr_window
        )
        # 统计 ATR 有效率
        valid_atr_pct = np.sum(~np.isnan(atr_arr)) / atr_arr.size * 100
        avg_atr = np.nanmean(atr_arr)
        print(f"   ATR 有效率: {valid_atr_pct:.1f}%, 平均 ATR: {avg_atr:.4f}")
    else:
        print(f"✅ 止损模式: Fixed 百分比 ({trailing_stop_pct*100:.1f}%)")
        atr_arr = None

    # ✅ v1.3: 显示止损检查时机
    print(f"✅ 止损检查时机: {'仅调仓日' if stop_on_rebalance_only else '每日'}")

    # ✅ P4: 从配置文件读取阶梯止盈参数
    profit_ladders = risk_config.get("profit_ladders", [])

    # ✅ P5: 从配置文件读取熔断机制参数
    circuit_breaker_config = risk_config.get("circuit_breaker", {})
    circuit_breaker_day = circuit_breaker_config.get("max_drawdown_day", 0.0)
    circuit_breaker_total = circuit_breaker_config.get("max_drawdown_total", 0.0)
    circuit_recovery_days = circuit_breaker_config.get("recovery_days", 5)

    # ✅ P6: 从配置文件读取冷却期参数
    cooldown_days = risk_config.get("cooldown_days", 0)

    # ✅ P7: 从配置文件读取杠杆上限 (零杠杆原则)
    leverage_cap = risk_config.get("leverage_cap", 1.0)

    print(
        f"✅ 动态降权: enabled={dynamic_leverage_enabled}, target_vol={target_vol}, vol_window={vol_window}"
    )
    print(f"✅ 阶梯止盈: {len(profit_ladders)} 级")
    for i, ladder in enumerate(profit_ladders):
        if use_atr_stop:
            print(
                f"   [{i+1}] 收益>{ladder.get('threshold', 0)*100:.0f}% → ATR倍数={ladder.get('new_multiplier', atr_multiplier):.1f}x"
            )
        else:
            print(
                f"   [{i+1}] 收益>{ladder.get('threshold', 0)*100:.0f}% → 止损={ladder.get('new_stop', 0)*100:.1f}%"
            )
    print(
        f"✅ 熔断机制: 单日={circuit_breaker_day*100:.1f}%, 总回撤={circuit_breaker_total*100:.1f}%, 恢复={circuit_recovery_days}天"
    )
    print(f"✅ 冷却期: {cooldown_days} 天")
    print(f"✅ 杠杆上限: {leverage_cap} (零杠杆原则)")

    # ✅ v4.0: 动态持仓数组
    dps_config = parse_dynamic_pos_config(backtest_config)
    if dps_config["enabled"]:
        # Pre-compute dynamic pos_size for each rebalance day
        _rebalance_schedule = generate_rebalance_schedule(T, LOOKBACK, FREQ)
        # gate_arr has been computed and applied to timing_arr already;
        # we need the raw gate values (before shift/multiply) for pos sizing.
        # Recompute raw gate for position sizing lookup.
        _raw_gate = compute_regime_gate_arr(
            ohlcv["close"], dates, backtest_config=config.get("backtest", {})
        )
        dynamic_pos_size_arr = np.array(
            [
                resolve_pos_size_for_day(dps_config, float(_raw_gate[rb_idx]))
                for rb_idx in _rebalance_schedule
            ],
            dtype=np.int64,
        )
        print(
            f"✅ 动态持仓: enabled, pos_size分布: "
            f"{dict(zip(*np.unique(dynamic_pos_size_arr, return_counts=True)))}"
        )
    else:
        dynamic_pos_size_arr = None
        print(f"✅ 动态持仓: disabled (固定 pos_size={POS_SIZE})")

    # 4. 批量回测
    results = []
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    combo_strings = df_combos["combo"].tolist()
    combo_indices = [
        [factor_index_map[f.strip()] for f in combo.split(" + ")]
        for combo in combo_strings
    ]

    # 定义单个combo回测函数（闭包捕获共享数据）
    def _backtest_one_combo(combo_str, factor_indices):
        _, ret, wr, pf, trades, rounding, risk = run_vec_backtest(
            factors_3d,
            close_prices,
            open_prices,
            high_prices,
            low_prices,
            timing_arr,
            factor_indices,
            # ✅ P0: 传入配置参数
            freq=FREQ,
            pos_size=POS_SIZE,
            initial_capital=INITIAL_CAPITAL,
            commission_rate=COMMISSION_RATE,
            lookback=LOOKBACK,
            cost_arr=COST_ARR,  # ✅ Exp2: per-ETF 成本数组
            # ✅ P2: 动态降权参数 (已禁用 - 零杠杆原则)
            target_vol=target_vol,
            vol_window=vol_window,
            dynamic_leverage_enabled=dynamic_leverage_enabled,
            vol_regime_arr=vol_regime_arr,  # ✅ v3.1: 传入波动率体制
            # ✅ v4.0: 动态持仓
            dynamic_pos_size_arr=dynamic_pos_size_arr,
            # ✅ v1.2: ATR 动态止损参数
            use_atr_stop=use_atr_stop,
            trailing_stop_pct=trailing_stop_pct,
            atr_arr=atr_arr,
            atr_multiplier=atr_multiplier,
            stop_on_rebalance_only=stop_on_rebalance_only,  # ✅ v1.3: 传入调仓日止损标志
            # ✅ v3.0: 个股趋势过滤参数
            individual_trend_arr=individual_trend_arr,
            individual_trend_enabled=individual_timing_enabled,
            # ✅ P4: 阶梯止盈参数
            profit_ladders=profit_ladders,
            # ✅ P5: 熔断机制参数
            circuit_breaker_day=circuit_breaker_day,
            circuit_breaker_total=circuit_breaker_total,
            circuit_recovery_days=circuit_recovery_days,
            # ✅ P6: 冷却期参数
            cooldown_days=cooldown_days,
            # ✅ P7: 杠杆上限
            leverage_cap=leverage_cap,
            # ✅ Exp1: T+1 Open
            use_t1_open=USE_T1_OPEN,
        )
        return {
            "combo": combo_str,
            "vec_return": ret,
            "vec_win_rate": wr,
            "vec_profit_factor": pf,
            "vec_trades": trades,
            # ✅ P0: 风险指标
            "vec_max_drawdown": risk["max_drawdown"],
            "vec_annual_return": risk["annual_return"],
            "vec_annual_volatility": risk["annual_volatility"],
            "vec_sharpe_ratio": risk["sharpe_ratio"],
            "vec_calmar_ratio": risk["calmar_ratio"],
            "vec_aligned_return": risk["aligned_return"],
            "vec_aligned_sharpe": risk["aligned_sharpe"],
            # ✅ P2: 动态降权诊断
            "vec_avg_leverage": risk["avg_leverage"],
            # ✅ Exp2: 成本诊断
            "vec_turnover_ann": risk["turnover_ann"],
            "vec_cost_drag": risk["cost_drag"],
            # 诊断信息
            "vec_target_value": rounding["target_value_total"],
            "vec_filled_value": rounding["filled_value_total"],
            "vec_target_shares": rounding["target_shares_total"],
            "vec_filled_shares": rounding["filled_shares_total"],
            "vec_value_gap": rounding["target_value_total"]
            - rounding["filled_value_total"],
            "vec_share_gap": rounding["target_shares_total"]
            - rounding["filled_shares_total"],
        }

    # ✅ 并行回测（Numba JIT 释放 GIL，线程池避免序列化开销）
    import os as _os
    n_vec_jobs = int(_os.environ.get("VEC_N_JOBS", min((_os.cpu_count() or 8) // 2, 16)))
    n_combos = len(combo_strings)
    print(f"\n🚀 并行 VEC 回测: {n_combos} 组合 (n_jobs={n_vec_jobs})")
    results = Parallel(n_jobs=n_vec_jobs, prefer="threads")(
        delayed(_backtest_one_combo)(cs, fi)
        for cs, fi in tqdm(
            zip(combo_strings, combo_indices),
            total=n_combos,
            desc="VEC 回测",
        )
    )

    # 5. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"vec_full_backtest_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results)
    df_results.to_parquet(output_dir / "vec_all_combos.parquet", index=False)

    print(f"\n✅ VEC 批量回测完成")
    print(f"   输出目录: {output_dir}")
    print(f"   组合数: {len(df_results)}")


if __name__ == "__main__":
    main()
