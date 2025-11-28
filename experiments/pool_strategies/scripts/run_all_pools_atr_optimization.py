#!/usr/bin/env python3
"""
全子池 WFO + VEC：因子 × ATR 动态风控联合优化
================================================================================
使用 ATR 动态止损止盈，而不是固定百分比：

搜索空间:
- 因子组合: C(18,2) + C(18,3) + C(18,4) + C(18,5) = 12,597
- ATR 周期: [10, 14, 20]
- 止损倍数: [1.5, 2.0, 2.5, 3.0] × ATR
- 止盈倍数: [2.0, 3.0, 4.0, 5.0] × ATR
- 跟踪止损倍数: [1.0, 1.5, 2.0] × ATR

总策略数: 12,597 × 3 × 4 × 4 × 3 = 1,814,832 (每池)

用法:
    uv run python scripts/run_all_pools_atr_optimization.py
================================================================================
"""

import sys
from pathlib import Path
from datetime import datetime
from itertools import combinations, product
import logging
import json
import time

import numpy as np
import pandas as pd
import yaml
from numba import njit, prange

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "etf_rotation_optimized"))

from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary
from core.cross_section_processor import CrossSectionProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# 参数配置
# =============================================================================

COMBO_SIZES = [2, 3, 4, 5]
LOOKBACK = 252
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000
COMMISSION_RATE = 0.0002

# ATR 风控参数搜索空间
ATR_PERIODS = [10, 14, 20]
STOP_LOSS_ATR_MULT = [1.5, 2.0, 2.5, 3.0]      # 止损 = 入场价 - N × ATR
TAKE_PROFIT_ATR_MULT = [2.0, 3.0, 4.0, 5.0]    # 止盈 = 入场价 + N × ATR  
TRAILING_STOP_ATR_MULT = [1.0, 1.5, 2.0]       # 跟踪止损 = 最高价 - N × ATR


# =============================================================================
# Numba: ATR 计算
# =============================================================================

@njit(cache=True)
def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """计算 ATR"""
    T = len(close)
    tr = np.zeros(T)
    atr = np.zeros(T)
    
    # True Range
    tr[0] = high[0] - low[0]
    for t in range(1, T):
        hl = high[t] - low[t]
        hc = abs(high[t] - close[t-1])
        lc = abs(low[t] - close[t-1])
        tr[t] = max(hl, hc, lc)
    
    # ATR (EMA)
    atr[period-1] = np.mean(tr[:period])
    alpha = 2.0 / (period + 1)
    for t in range(period, T):
        atr[t] = alpha * tr[t] + (1 - alpha) * atr[t-1]
    
    return atr


@njit(cache=True)
def compute_all_atr(
    high: np.ndarray,   # [T, N]
    low: np.ndarray,
    close: np.ndarray,
    periods: np.ndarray,
) -> np.ndarray:
    """计算所有 ATR 周期 [T, N, n_periods]"""
    T, N = close.shape
    n_periods = len(periods)
    result = np.zeros((T, N, n_periods))
    
    for n in range(N):
        for p_idx in range(n_periods):
            result[:, n, p_idx] = compute_atr(high[:, n], low[:, n], close[:, n], periods[p_idx])
    
    return result


# =============================================================================
# Numba: VEC 回测 (ATR 动态风控)
# =============================================================================

@njit(cache=True)
def _vec_backtest_atr(
    close_prices: np.ndarray,      # [T, N]
    atr_all: np.ndarray,           # [T, N, n_periods]
    factors_3d: np.ndarray,        # [T, N, F]
    factor_indices: np.ndarray,
    n_factors: int,
    rebalance_days: np.ndarray,
    pos_size: int,
    initial_capital: float,
    commission_rate: float,
    atr_period_idx: int,           # ATR 周期索引
    sl_mult: float,                # 止损倍数
    tp_mult: float,                # 止盈倍数
    ts_mult: float,                # 跟踪止损倍数
) -> tuple:
    """ATR 动态风控的 VEC 回测"""
    T, N = close_prices.shape
    
    cash = initial_capital
    holdings = np.zeros(N)
    entry_prices = np.zeros(N)
    entry_atr = np.zeros(N)        # 入场时的 ATR
    highest_prices = np.zeros(N)
    
    daily_values = np.zeros(T)
    daily_values[0] = initial_capital
    
    total_trades = 0
    wins = 0
    
    for t in range(1, T):
        # 更新最高价
        for n in range(N):
            if holdings[n] > 0 and close_prices[t, n] > highest_prices[n]:
                highest_prices[n] = close_prices[t, n]
        
        # ATR 动态风控检查
        for n in range(N):
            if holdings[n] > 0:
                price = close_prices[t, n]
                atr = entry_atr[n]  # 使用入场时的 ATR
                
                # 止损价 = 入场价 - sl_mult × ATR
                stop_loss_price = entry_prices[n] - sl_mult * atr
                # 止盈价 = 入场价 + tp_mult × ATR
                take_profit_price = entry_prices[n] + tp_mult * atr
                # 跟踪止损价 = 最高价 - ts_mult × ATR
                trailing_stop_price = highest_prices[n] - ts_mult * atr
                
                should_exit = False
                
                # 触发止损
                if price <= stop_loss_price:
                    should_exit = True
                # 触发止盈
                elif price >= take_profit_price:
                    should_exit = True
                # 触发跟踪止损 (只有盈利时才检查)
                elif price > entry_prices[n] and price <= trailing_stop_price:
                    should_exit = True
                
                if should_exit:
                    sell_value = holdings[n] * price
                    commission = sell_value * commission_rate
                    cash += sell_value - commission
                    
                    if price > entry_prices[n]:
                        wins += 1
                    total_trades += 1
                    
                    holdings[n] = 0
                    entry_prices[n] = 0
                    entry_atr[n] = 0
                    highest_prices[n] = 0
        
        # 计算净值
        pv = cash
        for n in range(N):
            if holdings[n] > 0:
                pv += holdings[n] * close_prices[t, n]
        daily_values[t] = pv
        
        # 调仓检查
        is_rebal = False
        for r in range(len(rebalance_days)):
            if rebalance_days[r] == t:
                is_rebal = True
                break
        
        if not is_rebal:
            continue
        
        # 卖出剩余持仓
        for n in range(N):
            if holdings[n] > 0:
                sell_value = holdings[n] * close_prices[t, n]
                commission = sell_value * commission_rate
                cash += sell_value - commission
                
                if close_prices[t, n] > entry_prices[n]:
                    wins += 1
                total_trades += 1
                
                holdings[n] = 0
                entry_prices[n] = 0
                entry_atr[n] = 0
                highest_prices[n] = 0
        
        # 计算因子得分
        scores = np.zeros(N)
        for n in range(N):
            s = 0.0
            cnt = 0
            for i in range(n_factors):
                f_idx = factor_indices[i]
                v = factors_3d[t-1, n, f_idx]
                if not np.isnan(v):
                    s += v
                    cnt += 1
            scores[n] = s / cnt if cnt > 0 else -1e9
        
        # 选股买入
        top_indices = np.argsort(scores)[::-1][:pos_size]
        capital_per = cash / pos_size
        
        for idx in top_indices:
            if scores[idx] <= -1e8:
                continue
            price = close_prices[t, idx]
            if price <= 0:
                continue
            
            # 获取当前 ATR
            current_atr = atr_all[t, idx, atr_period_idx]
            if current_atr <= 0 or np.isnan(current_atr):
                continue
            
            shares = int(capital_per / price / 100) * 100
            if shares <= 0:
                continue
            buy_value = shares * price
            commission = buy_value * commission_rate
            if buy_value + commission > cash:
                continue
            
            cash -= buy_value + commission
            holdings[idx] = shares
            entry_prices[idx] = price
            entry_atr[idx] = current_atr
            highest_prices[idx] = price
    
    # 计算指标
    final_value = daily_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    max_dd = 0.0
    peak = daily_values[0]
    for t in range(T):
        if daily_values[t] > peak:
            peak = daily_values[t]
        dd = (peak - daily_values[t]) / peak
        if dd > max_dd:
            max_dd = dd
    
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    
    years = T / 252.0
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    returns = np.zeros(T - 1)
    for t in range(1, T):
        if daily_values[t-1] > 0:
            returns[t-1] = (daily_values[t] - daily_values[t-1]) / daily_values[t-1]
    
    ret_std = np.std(returns)
    sharpe = np.mean(returns) / ret_std * np.sqrt(252) if ret_std > 0 else 0.0
    
    return total_return, annual_return, max_dd, sharpe, win_rate, total_trades


@njit(parallel=True, cache=True)
def _run_all_strategies_atr(
    close_prices: np.ndarray,
    atr_all: np.ndarray,
    factors_3d: np.ndarray,
    all_combo_indices: np.ndarray,
    combo_sizes: np.ndarray,
    n_combos: int,
    risk_params: np.ndarray,  # [n_risk, 4] - (atr_period_idx, sl_mult, tp_mult, ts_mult)
    n_risk_combos: int,
    rebalance_days: np.ndarray,
    pos_size: int,
    initial_capital: float,
    commission_rate: float,
) -> np.ndarray:
    """并行运行所有策略"""
    total_strategies = n_combos * n_risk_combos
    results = np.zeros((total_strategies, 6))
    
    for i in prange(total_strategies):
        combo_idx = i // n_risk_combos
        risk_idx = i % n_risk_combos
        
        size = combo_sizes[combo_idx]
        factor_indices = all_combo_indices[combo_idx, :size]
        
        atr_period_idx = int(risk_params[risk_idx, 0])
        sl_mult = risk_params[risk_idx, 1]
        tp_mult = risk_params[risk_idx, 2]
        ts_mult = risk_params[risk_idx, 3]
        
        (tr, ar, mdd, sharpe, wr, nt) = _vec_backtest_atr(
            close_prices, atr_all, factors_3d, factor_indices, size,
            rebalance_days, pos_size, initial_capital, commission_rate,
            atr_period_idx, sl_mult, tp_mult, ts_mult
        )
        
        results[i, 0] = tr
        results[i, 1] = ar
        results[i, 2] = mdd
        results[i, 3] = sharpe
        results[i, 4] = wr
        results[i, 5] = nt
    
    return results


# =============================================================================
# 单池处理
# =============================================================================

def process_single_pool(
    pool_name: str,
    pool_symbols: list,
    ohlcv: dict,
    factor_names: list,
    output_dir: Path,
) -> dict:
    """处理单个池"""
    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info(f"📊 处理池: {pool_name} ({len(pool_symbols)} ETFs)")
    logger.info(f"{'='*80}")
    
    valid_symbols = [s for s in pool_symbols if s in ohlcv["close"].columns]
    if len(valid_symbols) < 3:
        logger.warning(f"   ⚠️ 有效 ETF 不足 3 个，跳过")
        return None
    
    logger.info(f"   有效 ETF: {len(valid_symbols)}")
    
    pool_ohlcv = {key: df[valid_symbols] for key, df in ohlcv.items()}
    
    # 计算因子
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(pool_ohlcv)
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    T = len(pool_ohlcv["close"])
    N = len(valid_symbols)
    F = len(factor_names)
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = pool_ohlcv["close"].ffill().bfill().values.astype(np.float64)
    high_prices = pool_ohlcv["high"].ffill().bfill().values.astype(np.float64)
    low_prices = pool_ohlcv["low"].ffill().bfill().values.astype(np.float64)
    
    # 计算所有 ATR 周期
    atr_periods = np.array(ATR_PERIODS, dtype=np.int64)
    atr_all = compute_all_atr(high_prices, low_prices, close_prices, atr_periods)
    logger.info(f"   ATR 计算完成: 周期 {ATR_PERIODS}")
    
    # 生成因子组合
    all_combos = []
    for size in COMBO_SIZES:
        combos = list(combinations(range(F), size))
        all_combos.extend([(c, size) for c in combos])
    
    n_combos = len(all_combos)
    
    max_combo_size = max(COMBO_SIZES)
    all_combo_indices = np.full((n_combos, max_combo_size), -1, dtype=np.int64)
    combo_sizes_arr = np.zeros(n_combos, dtype=np.int64)
    
    for i, (combo, size) in enumerate(all_combos):
        combo_sizes_arr[i] = size
        for j, idx in enumerate(combo):
            all_combo_indices[i, j] = idx
    
    # 生成 ATR 风控参数组合
    risk_combos = []
    for atr_idx, atr_period in enumerate(ATR_PERIODS):
        for sl in STOP_LOSS_ATR_MULT:
            for tp in TAKE_PROFIT_ATR_MULT:
                for ts in TRAILING_STOP_ATR_MULT:
                    risk_combos.append((atr_idx, sl, tp, ts))
    
    n_risk_combos = len(risk_combos)
    risk_params = np.array(risk_combos, dtype=np.float64)
    
    total_strategies = n_combos * n_risk_combos
    logger.info(f"   因子组合: {n_combos} | ATR风控组合: {n_risk_combos} | 总策略: {total_strategies:,}")
    
    # 调仓日
    rebalance_days = np.array([t for t in range(LOOKBACK, T, FREQ)], dtype=np.int64)
    pool_pos_size = min(POS_SIZE, N - 1) if N > 1 else 1
    
    # 运行所有策略
    t0 = time.time()
    results = _run_all_strategies_atr(
        close_prices, atr_all, factors_3d, all_combo_indices, combo_sizes_arr, n_combos,
        risk_params, n_risk_combos, rebalance_days, pool_pos_size,
        INITIAL_CAPITAL, COMMISSION_RATE
    )
    elapsed = time.time() - t0
    logger.info(f"   运行耗时: {elapsed:.1f}s ({total_strategies/elapsed:.0f} 策略/秒)")
    
    # 整合结果
    records = []
    for i in range(total_strategies):
        combo_idx = i // n_risk_combos
        risk_idx = i % n_risk_combos
        
        combo, size = all_combos[combo_idx]
        combo_names = [factor_names[idx] for idx in combo]
        combo_str = " + ".join(combo_names)
        
        atr_idx, sl_mult, tp_mult, ts_mult = risk_combos[risk_idx]
        atr_period = ATR_PERIODS[atr_idx]
        
        tr, ar, mdd, sharpe, wr, nt = results[i]
        
        records.append({
            "pool": pool_name,
            "combo": combo_str,
            "combo_size": size,
            "atr_period": atr_period,
            "stop_loss_atr": sl_mult,
            "take_profit_atr": tp_mult,
            "trailing_stop_atr": ts_mult,
            "total_return": tr,
            "annual_return": ar,
            "max_drawdown": mdd,
            "sharpe_ratio": sharpe,
            "win_rate": wr,
            "num_trades": int(nt),
        })
    
    df = pd.DataFrame(records)
    df = df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    
    # 保存
    pool_dir = output_dir / pool_name
    pool_dir.mkdir(parents=True, exist_ok=True)
    df.head(10000).to_parquet(pool_dir / "top10000.parquet", index=False)
    df.head(100).to_csv(pool_dir / "top100.csv", index=False)
    
    # 最优策略
    best = df.iloc[0]
    best_info = {
        "pool": pool_name,
        "n_symbols": len(valid_symbols),
        "symbols": valid_symbols,
        "total_strategies": total_strategies,
        "best_combo": best["combo"],
        "best_factors": best["combo"].split(" + "),
        "best_atr_params": {
            "atr_period": int(best["atr_period"]),
            "stop_loss_atr": float(best["stop_loss_atr"]),
            "take_profit_atr": float(best["take_profit_atr"]),
            "trailing_stop_atr": float(best["trailing_stop_atr"]),
        },
        "total_return": float(best["total_return"]),
        "annual_return": float(best["annual_return"]),
        "max_drawdown": float(best["max_drawdown"]),
        "sharpe_ratio": float(best["sharpe_ratio"]),
        "win_rate": float(best["win_rate"]),
    }
    
    with open(pool_dir / "best_strategy.json", "w") as f:
        json.dump(best_info, f, indent=2, ensure_ascii=False)
    
    rp = best_info["best_atr_params"]
    logger.info(f"   🏆 最优策略:")
    logger.info(f"      因子: {best['combo']}")
    logger.info(f"      ATR风控: ATR({rp['atr_period']}) | 止损{rp['stop_loss_atr']}×ATR | 止盈{rp['take_profit_atr']}×ATR | 跟踪{rp['trailing_stop_atr']}×ATR")
    logger.info(f"      收益: {best['total_return']*100:.1f}% | 夏普: {best['sharpe_ratio']:.2f} | 回撤: {best['max_drawdown']*100:.1f}%")
    
    return best_info


# =============================================================================
# 主函数
# =============================================================================

def main():
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 80)
    logger.info("🚀 全子池 WFO + VEC：因子 × ATR 动态风控联合优化")
    logger.info("=" * 80)
    logger.info(f"   ATR 周期: {ATR_PERIODS}")
    logger.info(f"   止损倍数: {STOP_LOSS_ATR_MULT} × ATR")
    logger.info(f"   止盈倍数: {TAKE_PROFIT_ATR_MULT} × ATR")
    logger.info(f"   跟踪止损: {TRAILING_STOP_ATR_MULT} × ATR")
    
    config_pools_path = ROOT / "configs/etf_pools.yaml"
    config_wfo_path = ROOT / "configs/combo_wfo_config.yaml"
    
    with open(config_pools_path) as f:
        config_pools = yaml.safe_load(f)
    with open(config_wfo_path) as f:
        config_wfo = yaml.safe_load(f)
    
    pools = config_pools["pools"]
    logger.info(f"📊 共 {len(pools)} 个子池")
    
    all_symbols = set()
    for pool_name, pool_info in pools.items():
        all_symbols.update(pool_info["symbols"])
    all_symbols = sorted(all_symbols)
    
    logger.info("📊 加载数据...")
    loader = DataLoader(
        data_dir=config_wfo["data"].get("data_dir"),
        cache_dir=config_wfo["data"].get("cache_dir"),
    )
    
    ohlcv = loader.load_ohlcv(
        etf_codes=all_symbols,
        start_date=config_wfo["data"]["start_date"],
        end_date=config_wfo["data"]["end_date"],
    )
    
    factor_lib = PreciseFactorLibrary()
    sample_ohlcv = {k: v.iloc[:10] for k, v in ohlcv.items()}
    sample_factors = factor_lib.compute_all_factors(sample_ohlcv)
    factor_names = sorted(sample_factors.columns.get_level_values(0).unique().tolist())
    
    output_dir = ROOT / "results" / f"atr_optimization_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_best = {}
    for pool_name, pool_info in pools.items():
        best_info = process_single_pool(
            pool_name=pool_name,
            pool_symbols=pool_info["symbols"],
            ohlcv=ohlcv,
            factor_names=factor_names,
            output_dir=output_dir,
        )
        if best_info:
            all_best[pool_name] = best_info
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ 全部完成 | 总耗时: {elapsed:.1f}秒")
    logger.info("=" * 80)
    
    logger.info("")
    logger.info("🏆 各池最优策略汇总（因子 + ATR 动态风控）")
    logger.info("-" * 150)
    logger.info(f"{'池名':<18} {'收益':>8} {'夏普':>6} {'回撤':>6} {'ATR':>5} {'止损':>6} {'止盈':>6} {'跟踪':>6} {'最优因子组合'}")
    logger.info("-" * 150)
    
    for pool_name, info in sorted(all_best.items(), key=lambda x: -x[1]["sharpe_ratio"]):
        rp = info["best_atr_params"]
        logger.info(
            f"{pool_name:<18} {info['total_return']*100:>7.1f}% {info['sharpe_ratio']:>6.2f} "
            f"{info['max_drawdown']*100:>5.1f}% {rp['atr_period']:>5} "
            f"{rp['stop_loss_atr']:>5.1f}× {rp['take_profit_atr']:>5.1f}× {rp['trailing_stop_atr']:>5.1f}× "
            f"{info['best_combo']}"
        )
    
    # 保存汇总
    summary = {
        "timestamp": timestamp,
        "atr_search_space": {
            "atr_periods": ATR_PERIODS,
            "stop_loss_mult": STOP_LOSS_ATR_MULT,
            "take_profit_mult": TAKE_PROFIT_ATR_MULT,
            "trailing_stop_mult": TRAILING_STOP_ATR_MULT,
        },
        "elapsed_seconds": elapsed,
        "n_pools": len(all_best),
        "pools": all_best,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    best_config = {
        "timestamp": timestamp,
        "pool_strategies": {
            pool: {
                "factors": info["best_factors"],
                "atr_params": info["best_atr_params"],
            }
            for pool, info in all_best.items()
        },
    }
    
    with open(output_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    with open(ROOT / "results" / "atr_optimization_best_config_latest.json", "w") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    logger.info("")
    logger.info(f"📁 结果已保存至: {output_dir}")
    
    return summary


if __name__ == "__main__":
    main()
