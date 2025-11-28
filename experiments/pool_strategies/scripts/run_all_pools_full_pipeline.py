#!/usr/bin/env python3
"""
全子池完整 WFO + VEC 流程 | All Pools Full WFO + VEC Pipeline
================================================================================
对所有 7 个子池执行完整的 WFO + VEC 扫描：

1. 遍历 7 个子池
2. 每个子池：18 因子 → 12,597 组合
3. WFO: 计算每个组合的 ICIR
4. VEC: 对每个组合跑向量化回测
5. 输出每个子池的最优策略

用法:
    uv run python scripts/run_all_pools_full_pipeline.py
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

COMBO_SIZES = [2, 3, 4, 5]  # 因子组合大小
LOOKBACK = 252
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000
COMMISSION_RATE = 0.0002


# =============================================================================
# Numba 加速函数（从 run_single_pool_full_pipeline.py 复制）
# =============================================================================

@njit(cache=True)
def _compute_spearman_ic(scores: np.ndarray, returns: np.ndarray) -> float:
    """计算单日 Spearman IC"""
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
def _compute_combo_icir(
    factors_3d: np.ndarray,
    factor_indices: np.ndarray,
    forward_returns: np.ndarray,
    lookback: int,
) -> tuple:
    """计算单个组合的 ICIR"""
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
def _compute_all_combos_icir(
    factors_3d: np.ndarray,
    all_combo_indices: np.ndarray,
    combo_sizes: np.ndarray,
    forward_returns: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """并行计算所有组合的 ICIR"""
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


@njit(cache=True)
def _vec_backtest_single_combo(
    close_prices: np.ndarray,
    open_prices: np.ndarray,
    low_prices: np.ndarray,
    high_prices: np.ndarray,
    factors_3d: np.ndarray,
    factor_indices: np.ndarray,
    rebalance_days: np.ndarray,
    pos_size: int,
    initial_capital: float,
    commission_rate: float,
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.15,
) -> tuple:
    """单个组合的 VEC 回测 (含止损止盈)"""
    T, N = close_prices.shape
    
    cash = initial_capital
    holdings = np.zeros(N)
    entry_prices = np.zeros(N)
    
    daily_values = np.zeros(T)
    daily_values[0] = initial_capital
    
    total_trades = 0
    wins = 0
    
    # 记录上次调仓日，用于防止止损后立即买入
    last_rebal_idx = -1
    
    for t in range(1, T):
        # 1. 更新净值
        pv = cash
        for n in range(N):
            if holdings[n] > 0:
                pv += holdings[n] * close_prices[t, n]
        daily_values[t] = pv
        
        # 2. 检查止损止盈 (每日检查)
        for n in range(N):
            if holdings[n] > 0:
                entry = entry_prices[n]
                
                # 止损
                if low_prices[t, n] < entry * (1 - stop_loss_pct):
                    # 假设以止损价成交
                    stop_price = entry * (1 - stop_loss_pct)
                    # 如果开盘就低开，则以开盘价成交
                    if open_prices[t, n] < stop_price:
                        stop_price = open_prices[t, n]
                    
                    sell_value = holdings[n] * stop_price
                    commission = sell_value * commission_rate
                    cash += sell_value - commission
                    
                    total_trades += 1 # 止损算一次交易
                    # 亏损交易
                    
                    holdings[n] = 0
                    entry_prices[n] = 0
                    continue

                # 止盈
                if high_prices[t, n] > entry * (1 + take_profit_pct):
                    # 假设以止盈价成交
                    tp_price = entry * (1 + take_profit_pct)
                    # 如果开盘就高开，则以开盘价成交
                    if open_prices[t, n] > tp_price:
                        tp_price = open_prices[t, n]
                        
                    sell_value = holdings[n] * tp_price
                    commission = sell_value * commission_rate
                    cash += sell_value - commission
                    
                    total_trades += 1
                    wins += 1 # 止盈算盈利
                    
                    holdings[n] = 0
                    entry_prices[n] = 0
                    continue

        # 3. 检查是否调仓日
        is_rebal = False
        for r in range(len(rebalance_days)):
            if rebalance_days[r] == t:
                is_rebal = True
                last_rebal_idx = r
                break
        
        if not is_rebal:
            continue
        
        # 4. 调仓逻辑
        
        # 卖出 (先卖出非目标或需要调整的)
        # 这里简化：先全卖再全买 (或者只卖出不在 Top N 的)
        # 为了效率，我们重新计算得分并调整
        
        # 计算得分
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
        
        # 选 Top N
        top_indices = np.argsort(scores)[::-1][:pos_size]
        target_set = set()
        for idx in top_indices:
            if scores[idx] > -1e8:
                target_set.add(idx)
        
        # 卖出不在目标池的
        for n in range(N):
            if holdings[n] > 0:
                if n not in target_set:
                    sell_value = holdings[n] * close_prices[t, n]
                    commission = sell_value * commission_rate
                    cash += sell_value - commission
                    
                    if close_prices[t, n] > entry_prices[n]:
                        wins += 1
                    total_trades += 1
                    
                    holdings[n] = 0
                    entry_prices[n] = 0
        
        # 买入目标池的 (如果未持有)
        # 简单均分资金模型
        current_holdings_count = 0
        for n in range(N):
            if holdings[n] > 0:
                current_holdings_count += 1
        
        slots_available = pos_size - current_holdings_count
        if slots_available > 0 and cash > 0:
            capital_per_slot = cash / slots_available
            
            for idx in top_indices:
                if scores[idx] <= -1e8:
                    continue
                
                if holdings[idx] == 0: # 只买未持有的
                    price = close_prices[t, idx]
                    if price <= 0:
                        continue
                    
                    shares = int(capital_per_slot / price / 100) * 100
                    if shares <= 0:
                        continue
                    
                    buy_value = shares * price
                    commission = buy_value * commission_rate
                    
                    if buy_value + commission > cash:
                        continue
                    
                    cash -= buy_value + commission
                    holdings[idx] = shares
                    entry_prices[idx] = price
                    
                    slots_available -= 1
                    if slots_available == 0:
                        break
    
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
    
    ret_mean = np.mean(returns)
    ret_std = np.std(returns)
    sharpe = ret_mean / ret_std * np.sqrt(252) if ret_std > 0 else 0.0
    
    return total_return, annual_return, max_dd, sharpe, win_rate, total_trades


@njit(parallel=True, cache=True)
def _vec_backtest_all_combos(
    close_prices: np.ndarray,
    open_prices: np.ndarray,
    low_prices: np.ndarray,
    high_prices: np.ndarray,
    factors_3d: np.ndarray,
    all_combo_indices: np.ndarray,
    combo_sizes: np.ndarray,
    rebalance_days: np.ndarray,
    pos_size: int,
    initial_capital: float,
    commission_rate: float,
) -> np.ndarray:
    """并行 VEC 回测所有组合"""
    n_combos = all_combo_indices.shape[0]
    results = np.zeros((n_combos, 6))
    
    for i in prange(n_combos):
        size = combo_sizes[i]
        factor_indices = all_combo_indices[i, :size]
        
        tr, ar, mdd, sharpe, wr, nt = _vec_backtest_single_combo(
            close_prices, open_prices, low_prices, high_prices,
            factors_3d, factor_indices, rebalance_days,
            pos_size, initial_capital, commission_rate
        )
        
        results[i, 0] = tr
        results[i, 1] = ar
        results[i, 2] = mdd
        results[i, 3] = sharpe
        results[i, 4] = wr
        results[i, 5] = nt
    
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
    
    # 过滤有效符号
    valid_symbols = [s for s in pool_symbols if s in ohlcv["close"].columns]
    if len(valid_symbols) < 3:
        logger.warning(f"   ⚠️ 有效 ETF 不足 3 个，跳过")
        return None
    
    logger.info(f"   有效 ETF: {len(valid_symbols)}")
    
    # 提取池数据
    pool_ohlcv = {key: df[valid_symbols] for key, df in ohlcv.items()}
    
    # 计算因子
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(pool_ohlcv)
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    
    # 横截面标准化
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # 准备数据
    T = len(pool_ohlcv["close"])
    N = len(valid_symbols)
    F = len(factor_names)
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = pool_ohlcv["close"].ffill().bfill().values
    open_prices = pool_ohlcv["open"].ffill().bfill().values
    low_prices = pool_ohlcv["low"].ffill().bfill().values
    high_prices = pool_ohlcv["high"].ffill().bfill().values
    
    # 计算未来收益
    forward_returns = np.zeros((T, N))
    for t in range(T - 1):
        for n in range(N):
            if close_prices[t, n] > 0:
                forward_returns[t + 1, n] = (close_prices[t + 1, n] - close_prices[t, n]) / close_prices[t, n]
            else:
                forward_returns[t + 1, n] = np.nan
    
    # 生成组合
    all_combos = []
    for size in COMBO_SIZES:
        combos = list(combinations(range(F), size))
        all_combos.extend([(c, size) for c in combos])
    
    n_combos = len(all_combos)
    logger.info(f"   组合数: {n_combos}")
    
    # 准备 Numba 数据
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
    
    # VEC
    rebalance_days = np.array([t for t in range(LOOKBACK, T, FREQ)], dtype=np.int64)
    
    # 动态调整 pos_size
    pool_pos_size = min(POS_SIZE, N - 1) if N > 1 else 1
    
    t0 = time.time()
    vec_results = _vec_backtest_all_combos(
        close_prices, open_prices, low_prices, high_prices,
        factors_3d, all_combo_indices, combo_sizes_arr,
        rebalance_days, pool_pos_size, INITIAL_CAPITAL, COMMISSION_RATE
    )
    vec_time = time.time() - t0
    logger.info(f"   VEC: {vec_time:.2f}s")
    
    # 整合结果
    results = []
    for i, (combo, size) in enumerate(all_combos):
        combo_names = [factor_names[idx] for idx in combo]
        combo_str = " + ".join(combo_names)
        
        ic_mean, ic_std, icir, n_valid = icir_results[i]
        tr, ar, mdd, sharpe, wr, nt = vec_results[i]
        
        results.append({
            "pool": pool_name,
            "combo": combo_str,
            "combo_size": size,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "ic_valid_days": int(n_valid),
            "total_return": tr,
            "annual_return": ar,
            "max_drawdown": mdd,
            "sharpe_ratio": sharpe,
            "win_rate": wr,
            "num_trades": int(nt),
        })
    
    df = pd.DataFrame(results)

    # 多维排名：同时考虑 ICIR、收益、夏普、回撤
    df["rank_icir"] = df["icir"].rank(ascending=False, method="min")
    df["rank_return"] = df["total_return"].rank(ascending=False, method="min")
    df["rank_sharpe"] = df["sharpe_ratio"].rank(ascending=False, method="min")
    df["rank_mdd"] = df["max_drawdown"].rank(ascending=True, method="min")

    df["composite_rank"] = (
        df["rank_icir"]
        + df["rank_return"]
        + df["rank_sharpe"]
        + df["rank_mdd"]
    ) / 4.0

    df = df.sort_values(["composite_rank", "total_return"], ascending=[True, False]).reset_index(drop=True)
    df["final_rank"] = df.index + 1
    
    # 保存
    pool_dir = output_dir / pool_name
    pool_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pool_dir / "all_results.parquet", index=False)
    df.head(100).to_csv(pool_dir / "top100.csv", index=False)
    
    # 最优策略
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
        "icir": float(best["icir"]),
        "rank_icir": int(best["rank_icir"]),
        "rank_return": int(best["rank_return"]),
        "rank_sharpe": int(best["rank_sharpe"]),
        "rank_mdd": int(best["rank_mdd"]),
        "composite_rank": float(best["composite_rank"]),
        "final_rank": int(best["final_rank"]),
        "wfo_time": wfo_time,
        "vec_time": vec_time,
    }
    
    with open(pool_dir / "best_strategy.json", "w") as f:
        json.dump(best_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"   🏆 最优: {best['combo']}")
    logger.info(
        f"      收益: {best['total_return']*100:.1f}% | 夏普: {best['sharpe_ratio']:.2f} "
        f"| 回撤: {best['max_drawdown']*100:.1f}% | 复合排名: {best['composite_rank']:.1f}"
    )
    
    return best_info


# =============================================================================
# 主函数
# =============================================================================

def main():
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 80)
    logger.info("🚀 全子池完整 WFO + VEC 流程")
    logger.info("=" * 80)
    
    # 加载配置
    config_pools_path = ROOT / "configs/etf_pools.yaml"
    config_wfo_path = ROOT / "configs/combo_wfo_config.yaml"
    
    with open(config_pools_path) as f:
        config_pools = yaml.safe_load(f)
    with open(config_wfo_path) as f:
        config_wfo = yaml.safe_load(f)
    
    pools = config_pools["pools"]
    logger.info(f"📊 共 {len(pools)} 个子池")
    
    # 收集所有符号
    all_symbols = set()
    for pool_name, pool_info in pools.items():
        all_symbols.update(pool_info["symbols"])
    all_symbols = sorted(all_symbols)
    logger.info(f"   总 ETF 数: {len(all_symbols)}")
    
    # 加载所有数据
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
    
    # 获取因子名
    factor_lib = PreciseFactorLibrary()
    sample_ohlcv = {k: v.iloc[:10] for k, v in ohlcv.items()}
    sample_factors = factor_lib.compute_all_factors(sample_ohlcv)
    factor_names = sorted(sample_factors.columns.get_level_values(0).unique().tolist())
    logger.info(f"   因子数: {len(factor_names)}")
    
    # 输出目录
    output_dir = ROOT / "results" / f"all_pools_full_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个池
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
    
    # 汇总
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ 全部完成 | 总耗时: {elapsed:.1f}秒")
    logger.info("=" * 80)
    
    logger.info("")
    logger.info("🏆 各池最优策略汇总")
    logger.info("-" * 120)
    logger.info(f"{'池名':<20} {'收益':>10} {'夏普':>8} {'回撤':>8} {'复合':>8} {'最优因子组合'}")
    logger.info("-" * 120)
    
    for pool_name, info in sorted(all_best.items(), key=lambda x: x[1]["composite_rank"]):
        logger.info(
            f"{pool_name:<20} {info['total_return']*100:>9.1f}% {info['sharpe_ratio']:>8.2f} "
            f"{info['max_drawdown']*100:>7.1f}% {info['composite_rank']:>8.1f} {info['best_combo']}"
        )
    
    # 保存汇总
    summary = {
        "timestamp": timestamp,
        "elapsed_seconds": elapsed,
        "n_pools": len(all_best),
        "pools": all_best,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 保存最优配置（供后续使用）
    best_config = {
        "timestamp": timestamp,
        "pool_factors": {
            pool: info["best_factors"] for pool, info in all_best.items()
        },
        "pool_returns": {
            pool: info["total_return"] for pool, info in all_best.items()
        },
        "pool_sharpes": {
            pool: info["sharpe_ratio"] for pool, info in all_best.items()
        },
        "pool_max_drawdowns": {
            pool: info["max_drawdown"] for pool, info in all_best.items()
        },
        "pool_composite_ranks": {
            pool: info["composite_rank"] for pool, info in all_best.items()
        },
    }
    
    with open(output_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    # 同时保存到固定位置
    with open(ROOT / "results" / "all_pools_best_config_latest.json", "w") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    logger.info("")
    logger.info(f"📁 结果已保存至: {output_dir}")
    logger.info(f"📁 最优配置: results/all_pools_best_config_latest.json")
    
    return summary


if __name__ == "__main__":
    main()
