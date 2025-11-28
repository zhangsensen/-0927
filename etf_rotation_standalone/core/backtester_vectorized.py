"""
向量化回测引擎 | Vectorized Backtester

基于 Numba 加速的向量化回测引擎，支持：
- 多因子组合打分
- 择时信号集成
- 交易成本模拟
- 严格的防前视偏差处理

作者: Linus
日期: 2025-11-28
"""

import numpy as np
from numba import njit
from .utils.rebalance import generate_rebalance_schedule, ensure_price_views

# 默认参数
DEFAULT_FREQ = 8
DEFAULT_POS_SIZE = 3
DEFAULT_INITIAL_CAPITAL = 1_000_000.0
DEFAULT_COMMISSION_RATE = 0.0002
DEFAULT_LOOKBACK = 252


@njit(cache=True)
def vec_backtest_kernel(
    factors_3d,
    close_prices,
    open_prices,
    timing_arr,
    risk_off_prices,  # New argument: Risk-Off Asset Prices (e.g., Gold)
    factor_indices,
    rebalance_schedule,
    pos_size,
    initial_capital,
    commission_rate,
):
    """
    Numba 加速的回测核心逻辑 (支持 RORO)
    """
    T, N, _ = factors_3d.shape

    cash = initial_capital
    holdings = np.full(N, -1.0)
    entry_prices = np.zeros(N)
    
    # Risk-Off Asset State
    risk_off_holdings = 0.0
    risk_off_entry_price = 0.0

    wins = 0
    losses = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0

    combined_score = np.empty(N)
    target_set = np.zeros(N, dtype=np.bool_)
    buy_order = np.empty(pos_size, dtype=np.int64)
    new_targets = np.empty(pos_size, dtype=np.int64)

    # 使用预生成的调仓日程
    for i in range(len(rebalance_schedule)):
        t = rebalance_schedule[i]
        if t >= T:
            break

        # 1. 计算组合得分
        valid = 0
        for n in range(N):
            score = 0.0
            has_value = False
            for idx in factor_indices:
                val = factors_3d[t - 1, n, idx]  # 使用 t-1 日因子值（防前视）
                if not np.isnan(val):
                    score += val
                    has_value = True

            if has_value and score != 0.0:
                combined_score[n] = score
                valid += 1
            else:
                combined_score[n] = -np.inf

        # 2. 选股
        for n in range(N):
            target_set[n] = False

        buy_count = 0
        if valid >= pos_size:
            sorted_idx = np.argsort(combined_score)
            for k in range(pos_size):
                idx = sorted_idx[N - 1 - k]
                if combined_score[idx] == -np.inf:
                    break
                target_set[idx] = True
                buy_order[buy_count] = idx
                buy_count += 1

        # 3. 获取择时信号 (t-1 日信号已在外部 shift)
        timing_ratio = timing_arr[t]

        # 4. 卖出逻辑 (Equity)
        for n in range(N):
            if holdings[n] > 0.0 and not target_set[n]:
                # 使用收盘价卖出 (Cheat-On-Close)
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

        # 5. 计算当前总资产价值 (Equity + Risk-Off + Cash)
        equity_value = 0.0
        kept_equity_value = 0.0
        for n in range(N):
            if holdings[n] > 0.0:
                val = holdings[n] * close_prices[t, n]
                equity_value += val
                kept_equity_value += val
        
        risk_off_price = risk_off_prices[t]
        risk_off_value = 0.0
        if not np.isnan(risk_off_price) and risk_off_price > 0:
            risk_off_value = risk_off_holdings * risk_off_price
            
        total_value = cash + equity_value + risk_off_value
        
        # 6. RORO 资产配置
        target_equity = total_value * timing_ratio
        target_risk_off = total_value * (1.0 - timing_ratio)
        
        # 6.1 调整 Risk-Off 资产 (Gold)
        if not np.isnan(risk_off_price) and risk_off_price > 0:
            risk_off_diff = target_risk_off - risk_off_value
            
            # 卖出 Gold
            if risk_off_diff < -1e-5:
                sell_val = -risk_off_diff
                # 确保不超过持有量
                if sell_val > risk_off_value:
                    sell_val = risk_off_value
                
                shares_to_sell = sell_val / risk_off_price
                proceeds = sell_val * (1.0 - commission_rate)
                
                cash += proceeds
                risk_off_holdings -= shares_to_sell
                risk_off_value -= sell_val # Update for equity calc
                
            # 买入 Gold
            elif risk_off_diff > 1e-5:
                buy_val = risk_off_diff
                # 确保不超过现金
                if buy_val > cash:
                    buy_val = cash
                
                if buy_val > 0:
                    cost = buy_val
                    shares_to_buy = (cost / (1.0 + commission_rate)) / risk_off_price
                    
                    cash -= cost
                    risk_off_holdings += shares_to_buy
                    risk_off_value += buy_val # Update for equity calc (though not used)

        # 6.2 买入 Equity
        new_count = 0
        for k in range(buy_count):
            idx = buy_order[k]
            if holdings[idx] < 0.0:
                new_targets[new_count] = idx
                new_count += 1

        if new_count > 0:
            # 重新计算可用资金 (因为刚才可能买卖了 Gold)
            # 目标权益仓位 - 已持有权益仓位
            available_for_new = target_equity - kept_equity_value
            
            # 确保不超过现金
            if available_for_new > cash:
                available_for_new = cash
                
            if available_for_new < 0.0:
                available_for_new = 0.0

            target_pos_value = available_for_new / new_count / (1.0 + commission_rate)

            if target_pos_value > 0.0:
                for k in range(new_count):
                    idx = new_targets[k]
                    price = close_prices[t, idx]
                    if np.isnan(price) or price <= 0.0:
                        continue

                    shares = target_pos_value / price
                    cost = shares * price * (1.0 + commission_rate)

                    if cash >= cost - 1e-5:
                        actual_cost = cost
                        if actual_cost > cash:
                            actual_cost = cash
                        cash -= actual_cost
                        holdings[idx] = shares
                        entry_prices[idx] = price

    # 7. 最终清算
    final_value = cash
    
    # 清算 Equity
    for n in range(N):
        if holdings[n] > 0.0:
            price = close_prices[T - 1, n]
            if np.isnan(price):
                price = entry_prices[n]

            final_value += holdings[n] * price * (1.0 - commission_rate)

            pnl = (price - entry_prices[n]) / entry_prices[n]
            if pnl > 0.0:
                wins += 1
                total_win_pnl += pnl
            else:
                losses += 1
                total_loss_pnl += abs(pnl)
                
    # 清算 Risk-Off Asset
    if risk_off_holdings > 0:
        price = risk_off_prices[T - 1]
        if np.isnan(price) or price <= 0:
            # Fallback to last valid? For now just assume 0 if invalid at end
            price = 0.0 
            # In practice, should find last valid price. 
            # But let's assume data is clean or ffilled.
        
        final_value += risk_off_holdings * price * (1.0 - commission_rate)

    num_trades = wins + losses
    total_return = (final_value - initial_capital) / initial_capital
    win_rate = wins / num_trades if num_trades > 0 else 0.0

    if losses > 0:
        avg_win = total_win_pnl / max(wins, 1)
        avg_loss = total_loss_pnl / losses
        profit_factor = avg_win / max(avg_loss, 0.0001)
    else:
        profit_factor = 0.0

    return total_return, win_rate, profit_factor, num_trades


def run_vec_backtest(
    factors_3d: np.ndarray,
    close_prices: np.ndarray,
    open_prices: np.ndarray,
    timing_arr: np.ndarray,
    factor_indices: list,
    risk_off_prices: np.ndarray = None, # New optional argument
    freq: int = DEFAULT_FREQ,
    pos_size: int = DEFAULT_POS_SIZE,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    lookback: int = DEFAULT_LOOKBACK,
):
    """
    运行单个策略的 VEC 回测
    """
    factor_indices_arr = np.array(factor_indices, dtype=np.int64)
    T = factors_3d.shape[0]
    
    # Handle Risk-Off Prices
    if risk_off_prices is None:
        # Default to 0s (Cash mode)
        risk_off_prices_arr = np.zeros(T)
    else:
        risk_off_prices_arr = risk_off_prices
        if len(risk_off_prices_arr) != T:
             raise ValueError(f"Risk-off prices length {len(risk_off_prices_arr)} != T {T}")
    
    # 验证价格视图
    _, open_arr, close_arr = ensure_price_views(
        close_prices,
        open_prices,
        copy_if_missing=True,
        warn_if_copied=True,
        validate=True,
        min_valid_index=lookback,
    )
    
    # 生成调仓日程
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T,
        lookback_window=lookback,
        freq=freq,
    )
    
    return vec_backtest_kernel(
        factors_3d,
        close_arr,
        open_arr,
        timing_arr,
        risk_off_prices_arr,
        factor_indices_arr,
        rebalance_schedule,
        pos_size,
        initial_capital,
        commission_rate,
    )
