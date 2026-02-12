"""VEC-BT Hybrid Engine: Numba-accelerated backtest with integer-lot sizing.

Combines VEC speed (Numba @njit kernel) with BT realism (integer-lot sizing,
cash constraints, SPLIT_MARKET costs). Designed to replace Backtrader as
ground truth. Per-combo speedup is unverified — needs benchmark.

Key differences from VEC kernel:
  - Integer-lot sizing: shares = floor(float_shares / lot_size) * lot_size
  - Cash-constrained: orders rejected when insufficient cash (margin_failures)
  - Sell-first-then-buy ordering (matches BT GenericStrategy)
  - Per-ETF commission via cost_arr (SPLIT_MARKET model)

Key differences from BT engine:
  - Pure Numba: no Python/Backtrader overhead
  - Deterministic: same inputs always produce same outputs
  - Expected faster than Backtrader per combo (unverified, needs benchmark)

Execution model:
  T1_OPEN:  signal at close(t) -> fill at open(t+1)
  COC:      signal at close(t) -> fill at close(t)   [legacy]

Usage:
    from etf_strategy.core.vec_bt_hybrid import vec_bt_hybrid_kernel, run_hybrid_backtest
"""

from __future__ import annotations

import numpy as np
from numba import njit

from etf_strategy.core.hysteresis import apply_hysteresis


@njit(cache=True)
def _stable_topk(scores, k):
    """Stable top-k selection: descending by score, ascending index as tiebreaker.

    Returns array of indices (length <= k). Skips -inf scores.
    """
    N = len(scores)
    result = np.empty(k, dtype=np.int64)
    used = np.zeros(N, dtype=np.bool_)
    count = 0

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
            return result[:count]
        result[count] = best_idx
        used[best_idx] = True
        count += 1
    return result[:count]


@njit(cache=True)
def _execute_sells(
    holdings,
    entry_prices,
    hold_days_arr,
    cost_arr,
    target_mask,
    exec_prices,
    cash,
    total_commission_paid,
    wins,
    losses,
    total_win_pnl,
    total_loss_pnl,
):
    """Sell all held positions NOT in target_mask. Returns updated state scalars."""
    N = len(holdings)
    for n in range(N):
        if holdings[n] > 0 and not target_mask[n]:
            price = exec_prices[n]
            if np.isnan(price) or price <= 0.0:
                continue
            sell_comm = holdings[n] * price * cost_arr[n]
            total_commission_paid += sell_comm
            proceeds = holdings[n] * price - sell_comm
            cash += proceeds

            if entry_prices[n] > 0.0:
                pnl = (price - entry_prices[n]) / entry_prices[n]
            else:
                pnl = 0.0
            if pnl > 0.0:
                wins += 1
                total_win_pnl += pnl
            else:
                losses += 1
                total_loss_pnl += abs(pnl)

            holdings[n] = 0
            entry_prices[n] = 0.0
            hold_days_arr[n] = 0
    return cash, total_commission_paid, wins, losses, total_win_pnl, total_loss_pnl


@njit(cache=True)
def _execute_buys_integer_lots(
    holdings,
    entry_prices,
    hold_days_arr,
    cost_arr,
    target_mask,
    exec_prices,
    cash,
    total_commission_paid,
    margin_failures,
    available_for_new,
    pos_size,
    lot_size,
    init_hold_days,
):
    """Buy new targets (not already held) using integer-lot sizing.

    Returns updated state scalars.
    """
    N = len(holdings)

    # Collect new buy targets
    new_targets = np.empty(pos_size, dtype=np.int64)
    new_count = 0
    for n in range(N):
        if target_mask[n] and holdings[n] == 0:
            if new_count < pos_size:
                new_targets[new_count] = n
                new_count += 1

    if new_count == 0 or available_for_new <= 0.0:
        return cash, total_commission_paid, margin_failures

    tpv = available_for_new / new_count

    for k_idx in range(new_count):
        idx = new_targets[k_idx]
        price = exec_prices[idx]
        if np.isnan(price) or price <= 0.0:
            margin_failures += 1
            continue

        # Integer lot sizing: floor to nearest lot_size
        effective_tpv = tpv / (1.0 + cost_arr[idx])
        float_shares = effective_tpv / price
        int_shares = (int(float_shares) // lot_size) * lot_size

        if int_shares <= 0:
            # Try max affordable
            max_affordable = cash / (price * (1.0 + cost_arr[idx]))
            int_shares = (int(max_affordable) // lot_size) * lot_size
            if int_shares <= 0:
                margin_failures += 1
                continue

        total_cost = int_shares * price * (1.0 + cost_arr[idx])

        # Reduce lots until affordable
        while total_cost > cash + 1e-5 and int_shares > 0:
            int_shares -= lot_size
            total_cost = int_shares * price * (1.0 + cost_arr[idx])
        if int_shares <= 0:
            margin_failures += 1
            continue

        buy_comm = int_shares * price * cost_arr[idx]
        total_commission_paid += buy_comm
        cash -= total_cost
        holdings[idx] = int_shares
        entry_prices[idx] = price
        hold_days_arr[idx] = init_hold_days

    return cash, total_commission_paid, margin_failures


@njit(cache=True)
def vec_bt_hybrid_kernel(
    factors_3d,  # (T, N, F) float64: factor score tensor
    close_prices,  # (T, N) float64: close prices
    open_prices,  # (T, N) float64: open prices
    timing_arr,  # (T,) float64: shifted timing signal (already t-1)
    cost_arr,  # (N,) float64: per-ETF one-way transaction cost (decimal)
    factor_indices,  # (n_factors,) int64: which factors to sum
    rebalance_schedule,  # (n_rebal,) int32: bar indices for rebalancing
    pos_size,  # int: number of positions to hold
    initial_capital,  # float: starting cash
    lot_size,  # int: shares per lot (100 for A-share ETFs)
    use_t1_open,  # bool: T+1 Open execution mode
    delta_rank,  # float: hysteresis rank gap (0 = disabled)
    min_hold_days,  # int: min holding period (0 = disabled)
    dynamic_leverage_enabled,  # bool
    target_vol,  # float: target annualized vol for dynamic leverage
    vol_window,  # int: lookback for vol calculation
    leverage_cap,  # float: max leverage (1.0 = no leverage)
):
    """Numba kernel that matches BT ground truth with integer-lot sizing.

    Returns
    -------
    tuple of:
        equity_curve : (T,) float64
        total_return : float
        win_rate : float
        profit_factor : float
        num_trades : int (as int64)
        max_drawdown : float
        annual_return : float
        annual_volatility : float
        sharpe_ratio : float
        calmar_ratio : float
        total_commission_paid : float
        margin_failures : int (as int64)
    """
    T, N, _ = factors_3d.shape

    # --- State ---
    cash = initial_capital
    holdings = np.zeros(N, dtype=np.int64)  # share counts (integer lots)
    hold_days_arr = np.zeros(N, dtype=np.int64)  # execution hold days
    entry_prices = np.zeros(N, dtype=np.float64)

    # Exp4.1: Signal portfolio decoupled from execution (matches BT GenericStrategy)
    # Hysteresis decisions use signal state, not execution state.
    # This prevents integer-lot rounding from cascading into hysteresis divergence.
    signal_portfolio = np.zeros(N, dtype=np.bool_)  # like BT's _signal_portfolio
    signal_hold_days = np.zeros(N, dtype=np.int64)  # like BT's _signal_hold_days

    equity_curve = np.full(T, initial_capital, dtype=np.float64)

    # T1_OPEN pending state
    pend_active = False
    pend_target = np.zeros(N, dtype=np.bool_)
    pend_timing = 1.0

    # Welford online stats for daily returns
    welford_mean = 0.0
    welford_m2 = 0.0
    welford_count = 0
    prev_equity = initial_capital
    peak_equity = initial_capital
    max_drawdown = 0.0

    # Dynamic leverage ring buffer
    returns_buffer = np.zeros(vol_window, dtype=np.float64)
    buffer_ptr = 0
    buffer_filled = 0
    current_leverage = 1.0

    # Trade tracking
    wins = 0
    losses = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0
    total_commission_paid = 0.0
    margin_failures = 0

    # Scratch arrays (avoid allocation in loop)
    combined_score = np.empty(N, dtype=np.float64)
    target_mask = np.zeros(N, dtype=np.bool_)
    exec_prices = np.empty(N, dtype=np.float64)

    rebal_ptr = 0
    start_day = int(rebalance_schedule[0]) if len(rebalance_schedule) > 0 else 252

    for t in range(start_day, T):
        # ===== 1. Compute equity at start of day (using prev close) =====
        current_equity = cash
        for n in range(N):
            if holdings[n] > 0:
                price = close_prices[t - 1, n] if t > 0 else close_prices[0, n]
                if not np.isnan(price):
                    current_equity += float(holdings[n]) * price
        equity_curve[t] = current_equity

        # ===== 2. Daily return and Welford update =====
        if t > start_day:
            dr = (
                (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            )
            welford_count += 1
            delta = dr - welford_mean
            welford_mean += delta / welford_count
            delta2 = dr - welford_mean
            welford_m2 += delta * delta2

            if dynamic_leverage_enabled:
                returns_buffer[buffer_ptr] = dr
                buffer_ptr = (buffer_ptr + 1) % vol_window
                if buffer_filled < vol_window:
                    buffer_filled += 1

        # Peak/drawdown tracking
        if current_equity > peak_equity:
            peak_equity = current_equity
        dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd
        prev_equity = current_equity

        # ===== 3. Increment hold_days for held positions =====
        for n in range(N):
            if holdings[n] > 0:
                hold_days_arr[n] += 1
        # Exp4.1: Increment signal hold_days for signal portfolio
        for n in range(N):
            if signal_portfolio[n]:
                signal_hold_days[n] += 1

        # ===== 4. Execute T1_OPEN pending orders from yesterday =====
        if use_t1_open and pend_active:
            # Build exec prices from open[t] with close[t-1] fallback
            for n in range(N):
                p = open_prices[t, n]
                if np.isnan(p) or p <= 0.0:
                    p = close_prices[t - 1, n]
                exec_prices[n] = p

            # Sell non-targets
            cash, total_commission_paid, wins, losses, total_win_pnl, total_loss_pnl = (
                _execute_sells(
                    holdings,
                    entry_prices,
                    hold_days_arr,
                    cost_arr,
                    pend_target,
                    exec_prices,
                    cash,
                    total_commission_paid,
                    wins,
                    losses,
                    total_win_pnl,
                    total_loss_pnl,
                )
            )

            # Compute portfolio value at open for buy sizing
            pend_val = cash
            pend_kept = 0.0
            for n in range(N):
                if holdings[n] > 0:
                    v = float(holdings[n]) * exec_prices[n]
                    pend_val += v
                    pend_kept += v

            target_exposure = pend_val * pend_timing
            available = target_exposure - pend_kept
            if available < 0.0:
                available = 0.0

            # Buy new targets
            cash, total_commission_paid, margin_failures = _execute_buys_integer_lots(
                holdings,
                entry_prices,
                hold_days_arr,
                cost_arr,
                pend_target,
                exec_prices,
                cash,
                total_commission_paid,
                margin_failures,
                available,
                pos_size,
                lot_size,
                0,  # init_hold_days=0 for T1_OPEN (count starts next bar)
            )
            pend_active = False

        # ===== 5. Check rebalance day =====
        is_rebalance = False
        if rebal_ptr < len(rebalance_schedule) and rebalance_schedule[rebal_ptr] == t:
            is_rebalance = True
            rebal_ptr += 1

        if is_rebalance:
            # --- Dynamic leverage ---
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

            # --- Compute combined scores from t-1 factors ---
            valid_count = 0
            for n in range(N):
                score = 0.0
                n_valid_f = 0
                for fi in range(len(factor_indices)):
                    fidx = factor_indices[fi]
                    val = factors_3d[t - 1, n, fidx]
                    if not np.isnan(val):
                        score += val
                        n_valid_f += 1
                if n_valid_f > 0:
                    combined_score[n] = score / n_valid_f
                    valid_count += 1
                else:
                    combined_score[n] = -np.inf

            # --- Compute top-k and apply hysteresis ---
            for n in range(N):
                target_mask[n] = False

            if valid_count >= pos_size:
                top_indices = _stable_topk(combined_score, pos_size)

                # Exp4.1: Use signal portfolio state for hysteresis (not execution state)
                h_mask = np.zeros(N, dtype=np.bool_)
                for n in range(N):
                    h_mask[n] = signal_portfolio[n]

                if (delta_rank > 0.0 or min_hold_days > 0) and len(
                    top_indices
                ) >= pos_size:
                    hyst_result = apply_hysteresis(
                        combined_score,
                        h_mask,
                        signal_hold_days,
                        top_indices,
                        pos_size,
                        delta_rank,
                        min_hold_days,
                    )
                    for n in range(N):
                        target_mask[n] = hyst_result[n]
                else:
                    for i in range(len(top_indices)):
                        idx = top_indices[i]
                        if combined_score[idx] > -np.inf:
                            target_mask[idx] = True

                # Exp4.1: Update signal portfolio from hysteresis decision
                init_days = 0 if use_t1_open else 1
                for n in range(N):
                    if target_mask[n] and not signal_portfolio[n]:
                        # New entry: init hold days
                        signal_hold_days[n] = init_days
                    elif not target_mask[n] and signal_portfolio[n]:
                        # Exit: reset hold days
                        signal_hold_days[n] = 0
                    signal_portfolio[n] = target_mask[n]

            # --- Compute timing ratio ---
            effective_leverage = min(current_leverage, leverage_cap)
            timing_ratio = timing_arr[t] * effective_leverage

            if use_t1_open:
                # Store pending targets for execution at next day's open
                for n in range(N):
                    pend_target[n] = target_mask[n]
                pend_timing = timing_ratio
                pend_active = True
            else:
                # ===== COC mode: execute at close[t] =====
                for n in range(N):
                    exec_prices[n] = close_prices[t, n]

                # Sell non-targets
                (
                    cash,
                    total_commission_paid,
                    wins,
                    losses,
                    total_win_pnl,
                    total_loss_pnl,
                ) = _execute_sells(
                    holdings,
                    entry_prices,
                    hold_days_arr,
                    cost_arr,
                    target_mask,
                    exec_prices,
                    cash,
                    total_commission_paid,
                    wins,
                    losses,
                    total_win_pnl,
                    total_loss_pnl,
                )

                # Compute portfolio value for buy sizing
                current_value = cash
                kept_value = 0.0
                for n in range(N):
                    if holdings[n] > 0:
                        p = close_prices[t, n]
                        if not np.isnan(p):
                            v = float(holdings[n]) * p
                            current_value += v
                            kept_value += v

                target_exposure = current_value * timing_ratio
                available = target_exposure - kept_value
                if available < 0.0:
                    available = 0.0

                # Buy new targets
                cash, total_commission_paid, margin_failures = (
                    _execute_buys_integer_lots(
                        holdings,
                        entry_prices,
                        hold_days_arr,
                        cost_arr,
                        target_mask,
                        exec_prices,
                        cash,
                        total_commission_paid,
                        margin_failures,
                        available,
                        pos_size,
                        lot_size,
                        1,  # init_hold_days=1 for COC (count starts immediately)
                    )
                )

    # ===== Final equity (mark-to-market at last close, no liquidation cost) =====
    final_value = cash
    for n in range(N):
        if holdings[n] > 0:
            price = close_prices[T - 1, n]
            if np.isnan(price):
                price = entry_prices[n]
            final_value += float(holdings[n]) * price

    # Final daily return
    if final_value != prev_equity and prev_equity > 0:
        dr = (final_value - prev_equity) / prev_equity
        welford_count += 1
        delta = dr - welford_mean
        welford_mean += delta / welford_count
        delta2 = dr - welford_mean
        welford_m2 += delta * delta2

    if final_value > peak_equity:
        peak_equity = final_value
    final_dd = (peak_equity - final_value) / peak_equity if peak_equity > 0 else 0.0
    if final_dd > max_drawdown:
        max_drawdown = final_dd

    # ===== Compute summary metrics =====
    num_trades = wins + losses
    total_return = (final_value - initial_capital) / initial_capital
    win_rate = float(wins) / float(num_trades) if num_trades > 0 else 0.0

    if losses > 0:
        avg_win = total_win_pnl / max(wins, 1)
        avg_loss = total_loss_pnl / losses
        profit_factor = avg_win / max(avg_loss, 0.0001)
    else:
        profit_factor = np.inf if wins > 0 else 0.0

    trading_days = T - start_day
    years = float(trading_days) / 252.0 if trading_days > 0 else 1.0

    annual_return = 0.0
    if years > 0:
        base = 1.0 + total_return
        if base > 0.0:
            annual_return = base ** (1.0 / years) - 1.0
        else:
            annual_return = -0.99

    annual_volatility = 0.0
    if welford_count > 1:
        daily_variance = welford_m2 / (welford_count - 1)
        if daily_variance > 0:
            daily_std = np.sqrt(daily_variance)
            annual_volatility = daily_std * np.sqrt(252.0)

    sharpe_ratio = 0.0
    if annual_volatility > 0.0001:
        sharpe_ratio = annual_return / annual_volatility
    if np.isinf(sharpe_ratio) or np.isnan(sharpe_ratio):
        sharpe_ratio = 0.0

    calmar_ratio = 0.0
    if max_drawdown > 0.0001:
        calmar_ratio = annual_return / max_drawdown
    if np.isinf(calmar_ratio) or np.isnan(calmar_ratio):
        calmar_ratio = 0.0

    return (
        equity_curve,
        total_return,
        win_rate,
        profit_factor,
        np.int64(num_trades),
        max_drawdown,
        annual_return,
        annual_volatility,
        sharpe_ratio,
        calmar_ratio,
        total_commission_paid,
        np.int64(margin_failures),
    )


def run_hybrid_backtest(
    factors_3d: np.ndarray,
    close_prices: np.ndarray,
    open_prices: np.ndarray,
    timing_arr: np.ndarray,
    cost_arr: np.ndarray,
    factor_indices: list[int] | np.ndarray,
    rebalance_schedule: np.ndarray,
    pos_size: int = 2,
    initial_capital: float = 1_000_000.0,
    lot_size: int = 100,
    use_t1_open: bool = True,
    delta_rank: float = 0.0,
    min_hold_days: int = 0,
    dynamic_leverage_enabled: bool = False,
    target_vol: float = 0.20,
    vol_window: int = 20,
    leverage_cap: float = 1.0,
) -> dict:
    """Python wrapper for vec_bt_hybrid_kernel with aligned_metrics computation.

    Returns a dict with all metrics, compatible with BT batch output format.
    """
    factor_indices_arr = np.asarray(factor_indices, dtype=np.int64)

    result = vec_bt_hybrid_kernel(
        factors_3d,
        close_prices,
        open_prices,
        timing_arr,
        cost_arr,
        factor_indices_arr,
        rebalance_schedule,
        pos_size,
        initial_capital,
        lot_size,
        use_t1_open,
        delta_rank,
        min_hold_days,
        dynamic_leverage_enabled,
        target_vol,
        vol_window,
        leverage_cap,
    )

    (
        equity_curve,
        total_return,
        win_rate,
        profit_factor,
        num_trades,
        max_drawdown,
        annual_return,
        annual_volatility,
        sharpe_ratio,
        calmar_ratio,
        total_commission_paid,
        margin_failures,
    ) = result

    # Compute aligned metrics from equity curve
    start_idx = int(rebalance_schedule[0]) if len(rebalance_schedule) > 0 else 0
    eq = equity_curve[start_idx:]
    eq_valid = eq[np.isfinite(eq)]
    aligned_return = 0.0
    aligned_sharpe = 0.0
    if len(eq_valid) > 1:
        aligned_return = (
            (eq_valid[-1] - eq_valid[0]) / eq_valid[0] if eq_valid[0] > 0 else 0.0
        )
        daily_rets = np.diff(eq_valid) / eq_valid[:-1]
        valid_rets = daily_rets[np.isfinite(daily_rets)]
        if len(valid_rets) > 1:
            mean_r = float(np.mean(valid_rets))
            std_r = float(np.std(valid_rets, ddof=1))
            if std_r > 1e-12:
                aligned_sharpe = mean_r / std_r * np.sqrt(252.0)

    return {
        "equity_curve": equity_curve,
        "total_return": float(total_return),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "num_trades": int(num_trades),
        "max_drawdown": float(max_drawdown),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "calmar_ratio": float(calmar_ratio),
        "aligned_return": float(aligned_return),
        "aligned_sharpe": float(aligned_sharpe),
        "total_commission_paid": float(total_commission_paid),
        "margin_failures": int(margin_failures),
    }
