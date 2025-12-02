"""
向量化回测引擎 | Vectorized Backtester

基于 Numba 加速的向量化回测引擎，支持：
- 多因子组合打分
- 择时信号集成
- 交易成本模拟
- 严格的防前视偏差处理
- 风险指标计算 (MaxDD, Sharpe, Calmar)

作者: Linus
日期: 2025-11-28
更新: 2025-11-29 (加入风险指标)
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
TRADING_DAYS_PER_YEAR = 252


@njit(cache=True)
def vec_backtest_kernel_with_risk(
    factors_3d,
    close_prices,
    open_prices,
    timing_arr,
    risk_off_prices,
    factor_indices,
    rebalance_schedule,
    pos_size,
    initial_capital,
    commission_rate,
):
    """
    Numba 加速的回测核心逻辑 (支持 RORO + 风险指标计算)
    
    返回:
        total_return: 总收益率
        win_rate: 胜率
        profit_factor: 盈亏比
        num_trades: 交易次数
        max_drawdown: 最大回撤 (0-1之间，如0.2表示20%)
        sharpe_ratio: 夏普比率 (年化)
        annual_return: 年化收益率
        volatility: 年化波动率
    """
    T, N, _ = factors_3d.shape

    cash = initial_capital
    holdings = np.full(N, -1.0)
    entry_prices = np.zeros(N)
    
    # Risk-Off Asset State
    risk_off_holdings = 0.0

    wins = 0
    losses = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0
    
    # Risk metrics tracking
    peak_value = initial_capital
    max_drawdown = 0.0
    
    # 存储每个调仓周期的收益率 (用于计算波动率和夏普)
    # 预分配足够大的数组
    period_returns = np.zeros(len(rebalance_schedule) + 1)
    prev_value = initial_capital
    n_periods = 0

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
                val = factors_3d[t - 1, n, idx]
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

        # 3. 获取择时信号
        timing_ratio = timing_arr[t]

        # 4. 卖出逻辑 (Equity)
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

        # 5. 计算当前总资产价值
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
        
        # === 风险指标更新 ===
        # 更新最大回撤
        if total_value > peak_value:
            peak_value = total_value
        current_dd = (peak_value - total_value) / peak_value
        if current_dd > max_drawdown:
            max_drawdown = current_dd
            
        # 记录周期收益率 (用于夏普计算)
        if prev_value > 0:
            period_ret = (total_value - prev_value) / prev_value
            period_returns[n_periods] = period_ret
            n_periods += 1
        prev_value = total_value
        # === 风险指标更新结束 ===
        
        # 6. RORO 资产配置
        target_equity = total_value * timing_ratio
        target_risk_off = total_value * (1.0 - timing_ratio)
        
        # 6.1 调整 Risk-Off 资产 (Gold)
        if not np.isnan(risk_off_price) and risk_off_price > 0:
            risk_off_diff = target_risk_off - risk_off_value
            
            if risk_off_diff < -1e-5:
                sell_val = -risk_off_diff
                if sell_val > risk_off_value:
                    sell_val = risk_off_value
                
                shares_to_sell = sell_val / risk_off_price
                proceeds = sell_val * (1.0 - commission_rate)
                
                cash += proceeds
                risk_off_holdings -= shares_to_sell
                risk_off_value -= sell_val
                
            elif risk_off_diff > 1e-5:
                buy_val = risk_off_diff
                if buy_val > cash:
                    buy_val = cash
                
                if buy_val > 0:
                    cost = buy_val
                    shares_to_buy = (cost / (1.0 + commission_rate)) / risk_off_price
                    
                    cash -= cost
                    risk_off_holdings += shares_to_buy

        # 6.2 买入 Equity
        new_count = 0
        for k in range(buy_count):
            idx = buy_order[k]
            if holdings[idx] < 0.0:
                new_targets[new_count] = idx
                new_count += 1

        if new_count > 0:
            available_for_new = target_equity - kept_equity_value
            
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
                
    if risk_off_holdings > 0:
        price = risk_off_prices[T - 1]
        if np.isnan(price) or price <= 0:
            price = 0.0
        
        final_value += risk_off_holdings * price * (1.0 - commission_rate)

    # 最终回撤检查
    if final_value > peak_value:
        peak_value = final_value
    final_dd = (peak_value - final_value) / peak_value
    if final_dd > max_drawdown:
        max_drawdown = final_dd

    # 8. 计算指标
    num_trades = wins + losses
    total_return = (final_value - initial_capital) / initial_capital
    win_rate = wins / num_trades if num_trades > 0 else 0.0

    if losses > 0:
        avg_win = total_win_pnl / max(wins, 1)
        avg_loss = total_loss_pnl / losses
        profit_factor = avg_win / max(avg_loss, 0.0001)
    else:
        profit_factor = 0.0
    
    # 计算波动率和夏普比率
    # 回测跨越的交易日数
    start_day = rebalance_schedule[0] if len(rebalance_schedule) > 0 else 0
    trading_days = T - start_day
    
    if n_periods > 1 and trading_days > 0:
        # 计算周期收益的均值和标准差
        mean_ret = 0.0
        for j in range(n_periods):
            mean_ret += period_returns[j]
        mean_ret /= n_periods
        
        var_ret = 0.0
        for j in range(n_periods):
            diff = period_returns[j] - mean_ret
            var_ret += diff * diff
        var_ret /= (n_periods - 1)
        std_ret = np.sqrt(var_ret)
        
        # 年化收益率
        years = trading_days / 252.0
        annual_return = (1 + total_return) ** (1.0 / years) - 1 if years > 0 else total_return
        
        # 年化波动率: 假设调仓周期平均间隔
        avg_period_days = trading_days / n_periods if n_periods > 0 else 1.0
        periods_per_year = 252.0 / avg_period_days
        volatility = std_ret * np.sqrt(periods_per_year)
        
        # 夏普比率 (假设无风险利率为 0)
        sharpe_ratio = annual_return / max(volatility, 0.0001)
    else:
        annual_return = total_return
        volatility = 0.0
        sharpe_ratio = 0.0

    return total_return, win_rate, profit_factor, num_trades, max_drawdown, sharpe_ratio, annual_return, volatility


@njit(cache=True)
def vec_backtest_kernel(
    factors_3d,
    close_prices,
    open_prices,
    timing_arr,
    risk_off_prices,
    factor_indices,
    rebalance_schedule,
    pos_size,
    initial_capital,
    commission_rate,
):
    """
    简化版内核 (只返回基本指标，用于向后兼容)
    """
    ret, wr, pf, trades, mdd, sharpe, ann_ret, vol = vec_backtest_kernel_with_risk(
        factors_3d, close_prices, open_prices, timing_arr, risk_off_prices,
        factor_indices, rebalance_schedule, pos_size, initial_capital, commission_rate
    )
    return ret, wr, pf, trades


@njit(cache=True)
def vec_backtest_detailed_kernel(
    factors_3d,
    close_prices,
    open_prices,
    timing_arr,
    risk_off_prices,
    factor_indices,
    rebalance_schedule,
    pos_size,
    initial_capital,
    commission_rate,
):
    """
    Detailed version of the backtest kernel that returns daily time series.
    """
    T, N, _ = factors_3d.shape

    cash = initial_capital
    holdings = np.full(N, -1.0)
    entry_prices = np.zeros(N)
    
    risk_off_holdings = 0.0
    
    daily_total_value = np.zeros(T)
    daily_cash = np.zeros(T)
    daily_equity_value = np.zeros(T)
    daily_risk_off_value = np.zeros(T)
    
    daily_total_value[:] = initial_capital
    daily_cash[:] = initial_capital

    combined_score = np.empty(N)
    target_set = np.zeros(N, dtype=np.bool_)
    buy_order = np.empty(pos_size, dtype=np.int64)
    new_targets = np.empty(pos_size, dtype=np.int64)

    next_rebalance_idx = 0
    start_idx = rebalance_schedule[0] if len(rebalance_schedule) > 0 else T
    
    for t in range(start_idx, T):
        is_rebalance = False
        if next_rebalance_idx < len(rebalance_schedule) and rebalance_schedule[next_rebalance_idx] == t:
            is_rebalance = True
            next_rebalance_idx += 1
            
        if is_rebalance:
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
                sorted_idx = np.argsort(combined_score)
                for k in range(pos_size):
                    idx = sorted_idx[N - 1 - k]
                    if combined_score[idx] == -np.inf:
                        break
                    target_set[idx] = True
                    buy_order[buy_count] = idx
                    buy_count += 1

            timing_ratio = timing_arr[t]

            for n in range(N):
                if holdings[n] > 0.0 and not target_set[n]:
                    price = close_prices[t, n]
                    proceeds = holdings[n] * price * (1.0 - commission_rate)
                    cash += proceeds
                    holdings[n] = -1.0
                    entry_prices[n] = 0.0

            equity_val = 0.0
            kept_equity_val = 0.0
            for n in range(N):
                if holdings[n] > 0.0:
                    val = holdings[n] * close_prices[t, n]
                    equity_val += val
                    kept_equity_val += val
            
            risk_off_price = risk_off_prices[t]
            risk_off_val = 0.0
            if not np.isnan(risk_off_price) and risk_off_price > 0:
                risk_off_val = risk_off_holdings * risk_off_price
                
            total_val = cash + equity_val + risk_off_val
            
            target_equity = total_val * timing_ratio
            target_risk_off = total_val * (1.0 - timing_ratio)
            
            if not np.isnan(risk_off_price) and risk_off_price > 0:
                risk_off_diff = target_risk_off - risk_off_val
                
                if risk_off_diff < -1e-5:
                    sell_val = -risk_off_diff
                    if sell_val > risk_off_val: sell_val = risk_off_val
                    shares_to_sell = sell_val / risk_off_price
                    cash += sell_val * (1.0 - commission_rate)
                    risk_off_holdings -= shares_to_sell
                    
                elif risk_off_diff > 1e-5:
                    buy_val = risk_off_diff
                    if buy_val > cash: buy_val = cash
                    if buy_val > 0:
                        cost = buy_val
                        shares_to_buy = (cost / (1.0 + commission_rate)) / risk_off_price
                        cash -= cost
                        risk_off_holdings += shares_to_buy

            new_count = 0
            for k in range(buy_count):
                idx = buy_order[k]
                if holdings[idx] < 0.0:
                    new_targets[new_count] = idx
                    new_count += 1

            if new_count > 0:
                available_for_new = target_equity - kept_equity_val
                if available_for_new > cash: available_for_new = cash
                if available_for_new < 0.0: available_for_new = 0.0

                target_pos_value = available_for_new / new_count / (1.0 + commission_rate)

                if target_pos_value > 0.0:
                    for k in range(new_count):
                        idx = new_targets[k]
                        price = close_prices[t, idx]
                        if np.isnan(price) or price <= 0.0: continue
                        shares = target_pos_value / price
                        cost = shares * price * (1.0 + commission_rate)
                        if cash >= cost - 1e-5:
                            actual_cost = min(cost, cash)
                            cash -= actual_cost
                            holdings[idx] = shares
                            entry_prices[idx] = price

        current_equity_val = 0.0
        for n in range(N):
            if holdings[n] > 0.0:
                price = close_prices[t, n]
                if np.isnan(price): price = entry_prices[n]
                current_equity_val += holdings[n] * price
        
        current_risk_off_val = 0.0
        ro_price = risk_off_prices[t]
        if risk_off_holdings > 0 and not np.isnan(ro_price) and ro_price > 0:
            current_risk_off_val = risk_off_holdings * ro_price
            
        daily_total_value[t] = cash + current_equity_val + current_risk_off_val
        daily_cash[t] = cash
        daily_equity_value[t] = current_equity_val
        daily_risk_off_value[t] = current_risk_off_val

    return daily_total_value, daily_cash, daily_equity_value, daily_risk_off_value


def run_vec_backtest(
    factors_3d: np.ndarray,
    close_prices: np.ndarray,
    open_prices: np.ndarray,
    timing_arr: np.ndarray,
    factor_indices: list,
    risk_off_prices: np.ndarray = None,
    freq: int = DEFAULT_FREQ,
    pos_size: int = DEFAULT_POS_SIZE,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    lookback: int = DEFAULT_LOOKBACK,
):
    """
    运行单个策略的 VEC 回测 (向后兼容接口)
    """
    factor_indices_arr = np.array(factor_indices, dtype=np.int64)
    T = factors_3d.shape[0]
    
    if risk_off_prices is None:
        risk_off_prices_arr = np.zeros(T)
    else:
        risk_off_prices_arr = risk_off_prices
        if len(risk_off_prices_arr) != T:
             raise ValueError(f"Risk-off prices length {len(risk_off_prices_arr)} != T {T}")
    
    _, open_arr, close_arr = ensure_price_views(
        close_prices,
        open_prices,
        copy_if_missing=True,
        warn_if_copied=True,
        validate=True,
        min_valid_index=lookback,
    )
    
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


def run_vec_backtest_with_risk(
    factors_3d: np.ndarray,
    close_prices: np.ndarray,
    open_prices: np.ndarray,
    timing_arr: np.ndarray,
    factor_indices: list,
    risk_off_prices: np.ndarray = None,
    freq: int = DEFAULT_FREQ,
    pos_size: int = DEFAULT_POS_SIZE,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    lookback: int = DEFAULT_LOOKBACK,
):
    """
    运行单个策略的 VEC 回测，返回完整风险指标
    
    Returns:
        tuple: (total_return, win_rate, profit_factor, num_trades, 
                max_drawdown, sharpe_ratio, annual_return, volatility)
    """
    factor_indices_arr = np.array(factor_indices, dtype=np.int64)
    T = factors_3d.shape[0]
    
    if risk_off_prices is None:
        risk_off_prices_arr = np.zeros(T)
    else:
        risk_off_prices_arr = risk_off_prices
        if len(risk_off_prices_arr) != T:
             raise ValueError(f"Risk-off prices length {len(risk_off_prices_arr)} != T {T}")
    
    _, open_arr, close_arr = ensure_price_views(
        close_prices,
        open_prices,
        copy_if_missing=True,
        warn_if_copied=True,
        validate=True,
        min_valid_index=lookback,
    )
    
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T,
        lookback_window=lookback,
        freq=freq,
    )
    
    return vec_backtest_kernel_with_risk(
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


def run_detailed_backtest(
    factors_3d: np.ndarray,
    close_prices: np.ndarray,
    open_prices: np.ndarray,
    timing_arr: np.ndarray,
    factor_indices: list,
    risk_off_prices: np.ndarray = None,
    freq: int = DEFAULT_FREQ,
    pos_size: int = DEFAULT_POS_SIZE,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    lookback: int = DEFAULT_LOOKBACK,
):
    """
    运行详细回测，返回每日资产曲线
    """
    factor_indices_arr = np.array(factor_indices, dtype=np.int64)
    T = factors_3d.shape[0]
    
    if risk_off_prices is None:
        risk_off_prices_arr = np.zeros(T)
    else:
        risk_off_prices_arr = risk_off_prices
    
    _, open_arr, close_arr = ensure_price_views(
        close_prices,
        open_prices,
        copy_if_missing=True,
        warn_if_copied=True,
        validate=True,
        min_valid_index=lookback,
    )
    
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T,
        lookback_window=lookback,
        freq=freq,
    )
    
    return vec_backtest_detailed_kernel(
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
