"""
Full Universe Post-Selection & Robustness Analysis
1. Filter Production Candidates
2. Correlation Analysis & Portfolio Construction
3. Rolling Pseudo-OOS Validation
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from numba import njit
from datetime import datetime
import itertools

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.utils.rebalance import generate_rebalance_schedule, shift_timing_signal
from etf_strategy.core.market_timing import LightTimingModule

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Numba Kernels (Copied from multi_strategy_validation.py) ---

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
            if scores[n] > best_score or (scores[n] == best_score and (best_idx < 0 or n < best_idx)):
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
            daily_return = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
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
        current_dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
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
                day_return = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
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
        should_check_stop = (use_atr_stop and atr_multiplier > 0.0) or (not use_atr_stop and trailing_stop_pct > 0.0)
        if should_check_stop and (not stop_on_rebalance_only or is_rebalance_day):
            for n in range(N):
                if holdings[n] > 0.0:
                    prev_hwm = high_water_marks[n]
                    current_return = (prev_hwm - entry_prices[n]) / entry_prices[n] if entry_prices[n] > 0 else 0.0
                    for ladder_idx in range(3):
                        if current_return >= profit_ladder_thresholds[ladder_idx]:
                            if use_atr_stop:
                                if profit_ladder_multipliers[ladder_idx] < current_atr_mults[n]:
                                    current_atr_mults[n] = profit_ladder_multipliers[ladder_idx]
                            else:
                                if profit_ladder_stops[ladder_idx] < current_stop_pcts[n]:
                                    current_stop_pcts[n] = profit_ladder_stops[ladder_idx]
                    
                    if use_atr_stop:
                        prev_atr = atr_arr[t - 1, n] if t > 0 else 0.0
                        if np.isnan(prev_atr) or prev_atr <= 0.0:
                            curr_high = high_prices[t, n]
                            if not np.isnan(curr_high) and curr_high > high_water_marks[n]:
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
                    if individual_trend_enabled and not individual_trend_arr[t - 1, idx]:
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

                target_pos_value = available_for_new / new_count / (1.0 + commission_rate)

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
                            actual_shares = actual_cost / (price * (1.0 + commission_rate))
                            filled_shares_total += actual_shares
                            filled_value_total += actual_shares * price
                            cash -= actual_cost
                            holdings[idx] = shares
                            entry_prices[idx] = price
                            high_water_marks[idx] = price
                            current_stop_pcts[idx] = trailing_stop_pct
                            current_atr_mults[idx] = atr_multiplier
        
    # Final Value Calculation
    final_value = cash
    for n in range(N):
        if holdings[n] > 0.0:
            price = close_prices[T - 1, n]
            if np.isnan(price):
                price = entry_prices[n]
            final_value += holdings[n] * price

    return equity_curve

# --- Main Logic ---

def main():
    # 1. Load Strategy Summary
    summary_path = ROOT / "results/multi_strategy_validation/strategy_summary.csv"
    if not summary_path.exists():
        print("Strategy summary not found!")
        return
    
    df = pd.read_csv(summary_path)
    print(f"Loaded {len(df)} strategies.")
    
    # 2. Filter Production Candidates
    # Criteria:
    # - status in [PASS, WARNING]
    # - ann_return >= 0.16
    # - max_dd <= 0.28
    # - cost_sensitivity <= 0.35
    # - ret_2022 > -0.05
    # - ret_2025 > 0
    # - avg_adv_weighted >= 800e6
    # - top_symbol_weight <= 0.40
    
    candidates = df[
        (df['status'].isin(['PASS', 'WARNING'])) &
        (df['ann_return'] >= 0.16) &
        (df['max_dd'] <= 0.28) &
        (df['cost_sensitivity'] <= 0.35) &
        (df['ret_2022'] > -0.05) &
        (df['ret_2025'] > 0) &
        (df['avg_adv_weighted'] >= 800e6) &
        (df['top_symbol_weight'] <= 0.40)
    ].copy()
    
    print(f"Production Candidates: {len(candidates)}")
    
    out_dir = ROOT / "results/postselection"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_dir / "prod_candidates_full.csv", index=False)
    
    if len(candidates) == 0:
        print("No candidates found. Exiting.")
        return

    # 3. Load Market Data & Compute Factors (for re-running backtest)
    print("Loading Market Data & Computing Factors...")
    config_path = ROOT / 'configs/combo_wfo_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from etf_strategy.core.data_loader import DataLoader
    loader = DataLoader(
        data_dir=config['data'].get('data_dir'),
        cache_dir=config['data'].get('cache_dir'),
    )
    
    ohlcv_data = loader.load_ohlcv(
        etf_codes=config['data']['symbols'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
    )
    
    # Filter Liquid ETFs
    with open(ROOT / "scripts/run_liquid_vec_backtest.py", "r") as f:
        content = f.read()
        import ast
        tree = ast.parse(content)
        whitelist = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'LIQUID_ETFS':
                        whitelist = ast.literal_eval(node.value)
                        break
        if not whitelist:
            whitelist = [
                '510300', '510500', '510050', '513100', '513500',
                '512880', '512000', '512660', '512010', '512800',
                '512690', '512480', '512100', '512070', '515000',
                '588000', '159915', '159949', '518880', '513050', '513330'
            ]

    all_tickers = ohlcv_data['close'].columns
    tickers = sorted([t for t in all_tickers if t in whitelist])
    
    dates = ohlcv_data['close'].index
    T = len(dates)
    N = len(tickers)
    
    close_prices = ohlcv_data['close'][tickers].values
    open_prices = ohlcv_data['open'][tickers].values
    high_prices = ohlcv_data['high'][tickers].values
    low_prices = ohlcv_data['low'][tickers].values
    
    # Factors
    lib = PreciseFactorLibrary()
    liquid_data = {k: v[tickers] for k, v in ohlcv_data.items()}
    raw_factors_df = lib.compute_all_factors(liquid_data)
    factor_names = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    F = len(factor_names)
    
    factors_3d = np.full((T, N, F), np.nan)
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    for j, f_name in enumerate(factor_names):
        factors_3d[:, :, j] = std_factors[f_name].values
        
    # Timing
    close_df = pd.DataFrame(close_prices, index=dates, columns=tickers)
    timing_module = LightTimingModule()
    timing_signals = timing_module.compute_position_ratios(close_df)
    timing_arr = shift_timing_signal(timing_signals)
    
    # Rebalance Schedule
    FREQ = 3
    POS_SIZE = 2
    rebalance_schedule = generate_rebalance_schedule(len(dates), 252, FREQ)
    
    # 4. Compute Daily Returns for Candidates
    print(f"Computing daily returns for {len(candidates)} candidates...")
    
    daily_returns_dict = {}
    
    # Pre-compile
    _ = validation_backtest_kernel(
        factors_3d[:,:,:1], close_prices, open_prices, high_prices, low_prices, timing_arr,
        np.array([0]), rebalance_schedule, POS_SIZE, 1000000.0, 0.0002,
        0.0, 0, False, False, 0.0, np.zeros((T,N)), 0.0, False,
        np.ones((T,N), dtype=bool), False,
        np.array([0.15, 0.30, np.inf]), np.array([0.05, 0.03, 0.08]), np.array([2.0, 1.5, 3.0]),
        0.0, 0.0, 0, 0, 1.0
    )
    
    for idx, row in tqdm(candidates.iterrows(), total=len(candidates)):
        combo_str = row['combo']
        f_list = [f.strip() for f in combo_str.split('+')]
        f_indices = np.array([factor_names.index(f) for f in f_list if f in factor_names])
        
        equity_curve = validation_backtest_kernel(
            factors_3d, close_prices, open_prices, high_prices, low_prices, timing_arr,
            f_indices, rebalance_schedule, POS_SIZE, 1000000.0, 0.0002,
            0.0, 0, False, False, 0.0, np.zeros((T,N)), 0.0, False,
            np.ones((T,N), dtype=bool), False,
            np.array([0.15, 0.30, np.inf]), np.array([0.05, 0.03, 0.08]), np.array([2.0, 1.5, 3.0]),
            0.0, 0.0, 0, 0, 1.0
        )
        
        # Convert to daily returns
        # Handle zeros at start
        valid_idx = np.argmax(equity_curve > 0)
        eq_series = pd.Series(equity_curve, index=dates)
        # Replace 0 with NaN before pct_change to avoid inf
        eq_series[eq_series <= 0] = np.nan
        ret_series = eq_series.pct_change().fillna(0.0)
        daily_returns_dict[combo_str] = ret_series.values

    daily_returns_df = pd.DataFrame(daily_returns_dict, index=dates)
    
    # 5. Correlation Analysis
    print("Calculating Correlation Matrix...")
    corr_matrix = daily_returns_df.corr()
    
    # 6. Portfolio Construction
    print("Constructing Portfolios...")
    
    # 6.1 Single Strategies (Top 5 by Sharpe)
    top_singles = candidates.sort_values('sharpe', ascending=False).head(5)
    
    # 6.2 Pair Strategies
    # For each candidate, find min corr partner
    pairs = []
    for combo in candidates['combo']:
        corrs = corr_matrix[combo]
        # Filter self
        corrs = corrs[corrs.index != combo]
        if len(corrs) == 0:
            continue
        min_corr_combo = corrs.idxmin()
        min_corr = corrs.min()
        
        if min_corr > -0.5: # Only if correlation is reasonable (not perfectly inverse)
             # Calculate Portfolio Return
             ret_a = daily_returns_df[combo]
             ret_b = daily_returns_df[min_corr_combo]
             port_ret = 0.5 * ret_a + 0.5 * ret_b
             
             ann_ret = port_ret.mean() * 252
             ann_vol = port_ret.std() * np.sqrt(252)
             sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
             
             # MaxDD
             cum_ret = (1 + port_ret).cumprod()
             peak = cum_ret.cummax()
             dd = (peak - cum_ret) / peak
             max_dd = dd.max()
             
             pairs.append({
                 'combo_a': combo,
                 'combo_b': min_corr_combo,
                 'correlation': min_corr,
                 'ann_return': ann_ret,
                 'sharpe': sharpe,
                 'max_dd': max_dd
             })
             
    if pairs:
        pairs_df = pd.DataFrame(pairs).sort_values('sharpe', ascending=False).drop_duplicates(subset=['sharpe'])
        pairs_df.to_csv(out_dir / "combo2_summary.csv", index=False)
    else:
        print("No valid pairs found.")
        pairs_df = pd.DataFrame(columns=['combo_a', 'combo_b', 'correlation', 'ann_return', 'sharpe', 'max_dd'])
    
    # 6.3 Triple Strategies (Heuristic)
    # Take Top 20 singles, try to find triples with low pairwise corr
    top20 = candidates.sort_values('sharpe', ascending=False).head(20)['combo'].tolist()
    triples = []
    
    for c1, c2, c3 in itertools.combinations(top20, 3):
        corr12 = corr_matrix.loc[c1, c2]
        corr13 = corr_matrix.loc[c1, c3]
        corr23 = corr_matrix.loc[c2, c3]
        
        if max(corr12, corr13, corr23) < 0.7:
            ret_a = daily_returns_df[c1]
            ret_b = daily_returns_df[c2]
            ret_c = daily_returns_df[c3]
            port_ret = (ret_a + ret_b + ret_c) / 3.0
            
            ann_ret = port_ret.mean() * 252
            ann_vol = port_ret.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            
            cum_ret = (1 + port_ret).cumprod()
            peak = cum_ret.cummax()
            dd = (peak - cum_ret) / peak
            max_dd = dd.max()
            
            triples.append({
                'combo_1': c1,
                'combo_2': c2,
                'combo_3': c3,
                'max_corr': max(corr12, corr13, corr23),
                'ann_return': ann_ret,
                'sharpe': sharpe,
                'max_dd': max_dd
            })
            
    if triples:
        triples_df = pd.DataFrame(triples).sort_values('sharpe', ascending=False)
        triples_df.to_csv(out_dir / "combo3_summary.csv", index=False)
    else:
        print("No valid triples found.")
        triples_df = pd.DataFrame(columns=['combo_1', 'combo_2', 'combo_3', 'max_corr', 'ann_return', 'sharpe', 'max_dd'])
    
    # 7. Rolling Pseudo-OOS
    print("Running Rolling Pseudo-OOS...")
    
    windows = [
        {'train_start': '2020-01-01', 'train_end': '2022-12-31', 'test_start': '2023-01-01', 'test_end': '2023-12-31'},
        {'train_start': '2021-01-01', 'train_end': '2023-12-31', 'test_start': '2024-01-01', 'test_end': '2024-12-31'},
        {'train_start': '2022-01-01', 'train_end': '2024-12-31', 'test_start': '2025-01-01', 'test_end': '2025-12-31'}, # Until now
    ]
    
    oos_returns = []
    
    for w in windows:
        train_mask = (daily_returns_df.index >= w['train_start']) & (daily_returns_df.index <= w['train_end'])
        test_mask = (daily_returns_df.index >= w['test_start']) & (daily_returns_df.index <= w['test_end'])
        
        if not any(test_mask):
            continue
            
        train_rets = daily_returns_df[train_mask]
        test_rets = daily_returns_df[test_mask]
        
        # Select Top 5 Strategies in Train Period (by Sharpe)
        train_stats = []
        for col in train_rets.columns:
            r = train_rets[col]
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            
            # Apply basic filters (relaxed)
            cum_ret = (1 + r).cumprod()
            peak = cum_ret.cummax()
            dd = (peak - cum_ret) / peak
            max_dd = dd.max()
            
            if ann_ret >= 0.10 and max_dd <= 0.35:
                train_stats.append({'combo': col, 'sharpe': sharpe})
        
        if not train_stats:
            # Fallback: just take top sharpe
            for col in train_rets.columns:
                r = train_rets[col]
                sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
                train_stats.append({'combo': col, 'sharpe': sharpe})
                
        train_stats_df = pd.DataFrame(train_stats).sort_values('sharpe', ascending=False)
        top_combos = train_stats_df.head(5)['combo'].tolist()
        
        # Form Portfolio for Test Period
        port_test_ret = test_rets[top_combos].mean(axis=1)
        oos_returns.append(port_test_ret)
        
    if oos_returns:
        full_oos_ret = pd.concat(oos_returns)
        full_oos_ret = full_oos_ret[~full_oos_ret.index.duplicated(keep='first')].sort_index()
        
        # Calculate Metrics
        ann_ret = full_oos_ret.mean() * 252
        ann_vol = full_oos_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        cum_ret = (1 + full_oos_ret).cumprod()
        peak = cum_ret.cummax()
        dd = (peak - cum_ret) / peak
        max_dd = dd.max()
        
        print("\n--- Rolling Pseudo-OOS Results (2023-2025) ---")
        print(f"Annualized Return: {ann_ret:.2%}")
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        
        # Save OOS Curve
        full_oos_ret.to_csv(out_dir / "rolling_oos_curve.csv")
        
    # 8. Generate Report
    report_path = out_dir / "VALIDATION_REPORT_postselection.md"
    with open(report_path, "w") as f:
        f.write("# 🛡️ Full Universe Post-Selection Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Candidates**: {len(candidates)}\n\n")
        
        f.write("## 1. Top Single Strategies\n")
        f.write(top_singles[['combo', 'ann_return', 'sharpe', 'max_dd', 'ret_2022', 'ret_2025']].to_markdown(index=False, floatfmt=".2%"))
        f.write("\n\n")
        
        f.write("## 2. Top Pair Portfolios\n")
        f.write(pairs_df.head(5)[['combo_a', 'combo_b', 'correlation', 'ann_return', 'sharpe', 'max_dd']].to_markdown(index=False, floatfmt=".2%"))
        f.write("\n\n")
        
        f.write("## 3. Top Triple Portfolios\n")
        f.write(triples_df.head(5)[['combo_1', 'combo_2', 'combo_3', 'max_corr', 'ann_return', 'sharpe', 'max_dd']].to_markdown(index=False, floatfmt=".2%"))
        f.write("\n\n")
        
        if oos_returns:
            f.write("## 4. Rolling Pseudo-OOS (2023-2025)\n")
            f.write(f"- **Annualized Return**: {ann_ret:.2%}\n")
            f.write(f"- **Sharpe Ratio**: {sharpe:.3f}\n")
            f.write(f"- **Max Drawdown**: {max_dd:.2%}\n")
            
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()
