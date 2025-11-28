#!/usr/bin/env python3
"""
全子池完整 WFO + VEC 流程 (带止损止盈)
================================================================================
对所有 7 个子池执行完整的 WFO + VEC 扫描，包含风控：

- 止损: -8% 触发卖出
- 止盈: +20% 触发卖出
- 跟踪止损: 从高点回撤 -10% 触发卖出

用法:
    uv run python scripts/run_all_pools_with_risk.py
================================================================================
"""

import sys
from pathlib import Path
from datetime import datetime
from itertools import combinations
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

# 风控参数
STOP_LOSS = -0.08       # 止损: -8%
TAKE_PROFIT = 0.20      # 止盈: +20%
TRAILING_STOP = -0.10   # 跟踪止损: 从高点回撤 -10%


# =============================================================================
# Numba 加速：WFO IC 计算 (与之前相同)
# =============================================================================

@njit(cache=True)
def _compute_spearman_ic(scores: np.ndarray, returns: np.ndarray) -> float:
    mask = ~(np.isnan(scores) | np.isnan(returns))
    n_valid = np.sum(mask)
    if n_valid < 3:
        return np.nan
    
    s = scores[mask]
    r = returns[mask]
    
    s_rank = np.argsort(np.argsort(s)).astype(np.float64)
    r_rank = np.argsort(np.argsort(r)).astype(np.float64)
    
    s_mean = np.mean(s_rank)
    r_mean = np.mean(r_rank)
    
    num = np.sum((s_rank - s_mean) * (r_rank - r_mean))
    s_std = np.sqrt(np.sum((s_rank - s_mean) ** 2))
    r_std = np.sqrt(np.sum((r_rank - r_mean) ** 2))
    
    if s_std > 0 and r_std > 0:
        return num / (s_std * r_std)
    return np.nan


@njit(cache=True)
def _compute_combo_icir(factors_3d, factor_indices, forward_returns, lookback):
    T, N, _ = factors_3d.shape
    
    ic_list = []
    for t in range(lookback, T):
        combo_score = np.zeros(N)
        for n in range(N):
            s = 0.0
            cnt = 0
            for f_idx in factor_indices:
                v = factors_3d[t-1, n, f_idx]
                if not np.isnan(v):
                    s += v
                    cnt += 1
            combo_score[n] = s / cnt if cnt > 0 else np.nan
        
        ic = _compute_spearman_ic(combo_score, forward_returns[t])
        if not np.isnan(ic):
            ic_list.append(ic)
    
    n_valid = len(ic_list)
    if n_valid < 20:
        return 0.0, 0.0, 0.0, n_valid
    
    ic_arr = np.array(ic_list)
    ic_mean = np.mean(ic_arr)
    ic_std = np.std(ic_arr)
    icir = ic_mean / ic_std if ic_std > 0.001 else 0.0
    
    return ic_mean, ic_std, icir, n_valid


@njit(parallel=True, cache=True)
def _compute_all_combos_icir(factors_3d, all_combo_indices, combo_sizes, forward_returns, lookback):
    n_combos = all_combo_indices.shape[0]
    results = np.zeros((n_combos, 4))
    
    for i in prange(n_combos):
        size = combo_sizes[i]
        factor_indices = all_combo_indices[i, :size]
        ic_mean, ic_std, icir, n_valid = _compute_combo_icir(
            factors_3d, factor_indices, forward_returns, lookback
        )
        results[i, 0] = ic_mean
        results[i, 1] = ic_std
        results[i, 2] = icir
        results[i, 3] = n_valid
    
    return results


# =============================================================================
# Numba 加速：VEC 回测 (带止损止盈)
# =============================================================================

@njit(cache=True)
def _vec_backtest_with_risk(
    close_prices: np.ndarray,      # [T, N]
    factors_3d: np.ndarray,        # [T, N, F]
    factor_indices: np.ndarray,
    rebalance_days: np.ndarray,
    pos_size: int,
    initial_capital: float,
    commission_rate: float,
    stop_loss: float,              # 止损阈值 (负数)
    take_profit: float,            # 止盈阈值 (正数)
    trailing_stop: float,          # 跟踪止损阈值 (负数)
) -> tuple:
    """带止损止盈的 VEC 回测"""
    T, N = close_prices.shape
    
    cash = initial_capital
    holdings = np.zeros(N)           # 持仓股数
    entry_prices = np.zeros(N)       # 入场价格
    highest_prices = np.zeros(N)     # 持仓期间最高价 (用于跟踪止损)
    
    daily_values = np.zeros(T)
    daily_values[0] = initial_capital
    
    total_trades = 0
    wins = 0
    stop_loss_count = 0
    take_profit_count = 0
    trailing_stop_count = 0
    
    for t in range(1, T):
        # 1. 更新持仓最高价
        for n in range(N):
            if holdings[n] > 0:
                if close_prices[t, n] > highest_prices[n]:
                    highest_prices[n] = close_prices[t, n]
        
        # 2. 检查止损/止盈/跟踪止损
        for n in range(N):
            if holdings[n] > 0:
                current_price = close_prices[t, n]
                pnl_pct = (current_price - entry_prices[n]) / entry_prices[n]
                trailing_dd = (current_price - highest_prices[n]) / highest_prices[n]
                
                should_exit = False
                exit_reason = 0  # 1=止损, 2=止盈, 3=跟踪止损
                
                # 止损
                if pnl_pct <= stop_loss:
                    should_exit = True
                    exit_reason = 1
                # 止盈
                elif pnl_pct >= take_profit:
                    should_exit = True
                    exit_reason = 2
                # 跟踪止损 (只有盈利时才触发)
                elif pnl_pct > 0 and trailing_dd <= trailing_stop:
                    should_exit = True
                    exit_reason = 3
                
                if should_exit:
                    sell_value = holdings[n] * current_price
                    commission = sell_value * commission_rate
                    cash += sell_value - commission
                    
                    if current_price > entry_prices[n]:
                        wins += 1
                    total_trades += 1
                    
                    if exit_reason == 1:
                        stop_loss_count += 1
                    elif exit_reason == 2:
                        take_profit_count += 1
                    else:
                        trailing_stop_count += 1
                    
                    holdings[n] = 0
                    entry_prices[n] = 0
                    highest_prices[n] = 0
        
        # 3. 计算当前资产价值
        pv = cash
        for n in range(N):
            if holdings[n] > 0:
                pv += holdings[n] * close_prices[t, n]
        daily_values[t] = pv
        
        # 4. 检查是否调仓日
        is_rebal = False
        for r in range(len(rebalance_days)):
            if rebalance_days[r] == t:
                is_rebal = True
                break
        
        if not is_rebal:
            continue
        
        # === 调仓 ===
        # 5. 卖出所有剩余持仓
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
                highest_prices[n] = 0
        
        # 6. 计算因子得分
        scores = np.zeros(N)
        for n in range(N):
            s = 0.0
            cnt = 0
            for f_idx in factor_indices:
                v = factors_3d[t-1, n, f_idx]
                if not np.isnan(v):
                    s += v
                    cnt += 1
            scores[n] = s / cnt if cnt > 0 else -1e9
        
        # 7. 选 Top N
        top_indices = np.argsort(scores)[::-1][:pos_size]
        
        # 8. 买入
        capital_per_etf = cash / pos_size
        for idx in top_indices:
            if scores[idx] <= -1e8:
                continue
            
            price = close_prices[t, idx]
            if price <= 0:
                continue
            
            shares = int(capital_per_etf / price / 100) * 100
            if shares <= 0:
                continue
            
            buy_value = shares * price
            commission = buy_value * commission_rate
            
            if buy_value + commission > cash:
                continue
            
            cash -= buy_value + commission
            holdings[idx] = shares
            entry_prices[idx] = price
            highest_prices[idx] = price
    
    # 计算指标
    final_value = daily_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    # 最大回撤
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
    
    # 夏普
    returns = np.zeros(T - 1)
    for t in range(1, T):
        if daily_values[t-1] > 0:
            returns[t-1] = (daily_values[t] - daily_values[t-1]) / daily_values[t-1]
    
    ret_mean = np.mean(returns)
    ret_std = np.std(returns)
    sharpe = ret_mean / ret_std * np.sqrt(252) if ret_std > 0 else 0.0
    
    # 返回: 总收益, 年化, 最大回撤, 夏普, 胜率, 交易数, 止损次数, 止盈次数, 跟踪止损次数
    return (total_return, annual_return, max_dd, sharpe, win_rate, total_trades,
            stop_loss_count, take_profit_count, trailing_stop_count)


@njit(parallel=True, cache=True)
def _vec_backtest_all_combos_with_risk(
    close_prices: np.ndarray,
    factors_3d: np.ndarray,
    all_combo_indices: np.ndarray,
    combo_sizes: np.ndarray,
    rebalance_days: np.ndarray,
    pos_size: int,
    initial_capital: float,
    commission_rate: float,
    stop_loss: float,
    take_profit: float,
    trailing_stop: float,
) -> np.ndarray:
    """并行 VEC 回测所有组合 (带风控)"""
    n_combos = all_combo_indices.shape[0]
    results = np.zeros((n_combos, 9))
    
    for i in prange(n_combos):
        size = combo_sizes[i]
        factor_indices = all_combo_indices[i, :size]
        
        (tr, ar, mdd, sharpe, wr, nt, sl_cnt, tp_cnt, ts_cnt) = _vec_backtest_with_risk(
            close_prices, factors_3d, factor_indices, rebalance_days,
            pos_size, initial_capital, commission_rate,
            stop_loss, take_profit, trailing_stop
        )
        
        results[i, 0] = tr
        results[i, 1] = ar
        results[i, 2] = mdd
        results[i, 3] = sharpe
        results[i, 4] = wr
        results[i, 5] = nt
        results[i, 6] = sl_cnt
        results[i, 7] = tp_cnt
        results[i, 8] = ts_cnt
    
    return results


# =============================================================================
# 单池处理函数
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
    
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(pool_ohlcv)
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    T = len(pool_ohlcv["close"])
    N = len(valid_symbols)
    F = len(factor_names)
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = pool_ohlcv["close"].ffill().bfill().values
    
    forward_returns = np.zeros((T, N))
    for t in range(T - 1):
        for n in range(N):
            if close_prices[t, n] > 0:
                forward_returns[t + 1, n] = (close_prices[t + 1, n] - close_prices[t, n]) / close_prices[t, n]
            else:
                forward_returns[t + 1, n] = np.nan
    
    all_combos = []
    for size in COMBO_SIZES:
        combos = list(combinations(range(F), size))
        all_combos.extend([(c, size) for c in combos])
    
    n_combos = len(all_combos)
    logger.info(f"   组合数: {n_combos}")
    
    max_combo_size = max(COMBO_SIZES)
    all_combo_indices = np.full((n_combos, max_combo_size), -1, dtype=np.int64)
    combo_sizes_arr = np.zeros(n_combos, dtype=np.int64)
    
    for i, (combo, size) in enumerate(all_combos):
        combo_sizes_arr[i] = size
        for j, idx in enumerate(combo):
            all_combo_indices[i, j] = idx
    
    # WFO
    t0 = time.time()
    icir_results = _compute_all_combos_icir(
        factors_3d, all_combo_indices, combo_sizes_arr, forward_returns, LOOKBACK
    )
    wfo_time = time.time() - t0
    logger.info(f"   WFO: {wfo_time:.2f}s")
    
    # VEC (带风控)
    rebalance_days = np.array([t for t in range(LOOKBACK, T, FREQ)], dtype=np.int64)
    pool_pos_size = min(POS_SIZE, N - 1) if N > 1 else 1
    
    t0 = time.time()
    vec_results = _vec_backtest_all_combos_with_risk(
        close_prices, factors_3d, all_combo_indices, combo_sizes_arr,
        rebalance_days, pool_pos_size, INITIAL_CAPITAL, COMMISSION_RATE,
        STOP_LOSS, TAKE_PROFIT, TRAILING_STOP
    )
    vec_time = time.time() - t0
    logger.info(f"   VEC (带风控): {vec_time:.2f}s")
    
    # 整合结果
    results = []
    for i, (combo, size) in enumerate(all_combos):
        combo_names = [factor_names[idx] for idx in combo]
        combo_str = " + ".join(combo_names)
        
        ic_mean, ic_std, icir, n_valid = icir_results[i]
        tr, ar, mdd, sharpe, wr, nt, sl_cnt, tp_cnt, ts_cnt = vec_results[i]
        
        results.append({
            "pool": pool_name,
            "combo": combo_str,
            "combo_size": size,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "total_return": tr,
            "annual_return": ar,
            "max_drawdown": mdd,
            "sharpe_ratio": sharpe,
            "win_rate": wr,
            "num_trades": int(nt),
            "stop_loss_count": int(sl_cnt),
            "take_profit_count": int(tp_cnt),
            "trailing_stop_count": int(ts_cnt),
        })
    
    df = pd.DataFrame(results)
    
    # 按夏普排序 (风控后更关注风险调整收益)
    df["rank_sharpe"] = df["sharpe_ratio"].rank(ascending=False)
    df["rank_return"] = df["total_return"].rank(ascending=False)
    df["rank_mdd"] = df["max_drawdown"].rank(ascending=True)
    df["composite_score"] = (df["rank_sharpe"] + df["rank_return"] + df["rank_mdd"]) / 3
    df = df.sort_values("composite_score").reset_index(drop=True)
    
    pool_dir = output_dir / pool_name
    pool_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pool_dir / "all_results.parquet", index=False)
    df.head(100).to_csv(pool_dir / "top100.csv", index=False)
    
    best = df.iloc[0]
    best_info = {
        "pool": pool_name,
        "n_symbols": len(valid_symbols),
        "symbols": valid_symbols,
        "n_combos": n_combos,
        "best_combo": best["combo"],
        "best_factors": best["combo"].split(" + "),
        "total_return": float(best["total_return"]),
        "annual_return": float(best["annual_return"]),
        "max_drawdown": float(best["max_drawdown"]),
        "sharpe_ratio": float(best["sharpe_ratio"]),
        "win_rate": float(best["win_rate"]),
        "stop_loss_count": int(best["stop_loss_count"]),
        "take_profit_count": int(best["take_profit_count"]),
        "trailing_stop_count": int(best["trailing_stop_count"]),
    }
    
    with open(pool_dir / "best_strategy.json", "w") as f:
        json.dump(best_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"   🏆 最优: {best['combo']}")
    logger.info(f"      收益: {best['total_return']*100:.1f}% | 夏普: {best['sharpe_ratio']:.2f} | 回撤: {best['max_drawdown']*100:.1f}%")
    logger.info(f"      止损: {int(best['stop_loss_count'])}次 | 止盈: {int(best['take_profit_count'])}次 | 跟踪止损: {int(best['trailing_stop_count'])}次")
    
    return best_info


# =============================================================================
# 主函数
# =============================================================================

def main():
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 80)
    logger.info("🚀 全子池完整 WFO + VEC 流程 (带止损止盈)")
    logger.info(f"   止损: {STOP_LOSS*100:.0f}% | 止盈: {TAKE_PROFIT*100:.0f}% | 跟踪止损: {TRAILING_STOP*100:.0f}%")
    logger.info("=" * 80)
    
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
    
    output_dir = ROOT / "results" / f"all_pools_risk_{timestamp}"
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
    logger.info("🏆 各池最优策略汇总 (带风控)")
    logger.info("-" * 120)
    logger.info(f"{'池名':<18} {'收益':>8} {'夏普':>6} {'回撤':>6} {'止损':>5} {'止盈':>5} {'跟踪':>5} {'最优因子组合'}")
    logger.info("-" * 120)
    
    for pool_name, info in sorted(all_best.items(), key=lambda x: -x[1]["sharpe_ratio"]):
        logger.info(
            f"{pool_name:<18} {info['total_return']*100:>7.1f}% {info['sharpe_ratio']:>6.2f} "
            f"{info['max_drawdown']*100:>5.1f}% {info['stop_loss_count']:>5} {info['take_profit_count']:>5} "
            f"{info['trailing_stop_count']:>5} {info['best_combo']}"
        )
    
    summary = {
        "timestamp": timestamp,
        "risk_params": {
            "stop_loss": STOP_LOSS,
            "take_profit": TAKE_PROFIT,
            "trailing_stop": TRAILING_STOP,
        },
        "elapsed_seconds": elapsed,
        "n_pools": len(all_best),
        "pools": all_best,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    best_config = {
        "timestamp": timestamp,
        "risk_params": {
            "stop_loss": STOP_LOSS,
            "take_profit": TAKE_PROFIT,
            "trailing_stop": TRAILING_STOP,
        },
        "pool_factors": {
            pool: info["best_factors"] for pool, info in all_best.items()
        },
    }
    
    with open(output_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    with open(ROOT / "results" / "all_pools_risk_best_config_latest.json", "w") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    logger.info("")
    logger.info(f"📁 结果已保存至: {output_dir}")
    
    return summary


if __name__ == "__main__":
    main()
