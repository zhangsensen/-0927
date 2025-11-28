#!/usr/bin/env python3
"""
单池完整 WFO + VEC 流程 | Single Pool Full WFO + VEC Pipeline
================================================================================
选择 EQUITY_GROWTH 池（17 个 ETF），执行完整流程：

1. 加载 17 个 ETF 数据
2. 计算 18 个因子
3. 生成 1万+ 因子组合
4. WFO: 计算每个组合的 IC/ICIR
5. VEC: 对每个组合跑向量化回测
6. 输出完整排名

用法:
    uv run python scripts/run_single_pool_full_pipeline.py
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
from tqdm import tqdm

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

POOL_NAME = "EQUITY_GROWTH"
COMBO_SIZES = [2, 3, 4, 5]  # 因子组合大小 (完整版)
LOOKBACK = 252
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000
COMMISSION_RATE = 0.0002
MIN_VALID_DAYS = 50


# =============================================================================
# Numba 加速：WFO IC 计算
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
    n_factors = len(factor_indices)
    
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


# =============================================================================
# Numba 加速：VEC 回测
# =============================================================================

@njit(cache=True)
def _vec_backtest_single_combo(
    close_prices: np.ndarray,  # [T, N]
    factors_3d: np.ndarray,    # [T, N, F]
    factor_indices: np.ndarray,
    rebalance_days: np.ndarray,
    pos_size: int,
    initial_capital: float,
    commission_rate: float,
) -> tuple:
    """单个组合的 VEC 回测"""
    T, N = close_prices.shape
    n_factors = len(factor_indices)
    
    cash = initial_capital
    holdings = np.zeros(N)
    entry_prices = np.zeros(N)
    
    daily_values = np.zeros(T)
    daily_values[0] = initial_capital
    
    total_trades = 0
    wins = 0
    
    for t in range(1, T):
        # 计算当前资产价值
        pv = cash
        for n in range(N):
            if holdings[n] > 0:
                pv += holdings[n] * close_prices[t, n]
        daily_values[t] = pv
        
        # 检查是否调仓日
        is_rebal = False
        for r in range(len(rebalance_days)):
            if rebalance_days[r] == t:
                is_rebal = True
                break
        
        if not is_rebal:
            continue
        
        # === 调仓 ===
        # 1. 卖出所有
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
        
        # 2. 计算因子得分
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
        
        # 3. 选 Top N
        # 简单排序选股
        top_indices = np.argsort(scores)[::-1][:pos_size]
        
        # 4. 买入
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
    
    # 胜率
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    
    # 年化收益
    years = T / 252.0
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    # 夏普 (简化)
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
            close_prices, factors_3d, factor_indices, rebalance_days,
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
# 主函数
# =============================================================================

def main():
    start_time = datetime.now()
    
    logger.info("=" * 80)
    logger.info(f"🚀 单池完整 WFO + VEC 流程 | Pool: {POOL_NAME}")
    logger.info("=" * 80)
    
    # 1. 加载池配置
    config_pools_path = ROOT / "configs/etf_pools.yaml"
    config_wfo_path = ROOT / "configs/combo_wfo_config.yaml"
    
    with open(config_pools_path) as f:
        config_pools = yaml.safe_load(f)
    with open(config_wfo_path) as f:
        config_wfo = yaml.safe_load(f)
    
    pool_symbols = config_pools["pools"][POOL_NAME]["symbols"]
    logger.info(f"📊 池 {POOL_NAME}: {len(pool_symbols)} 个 ETF")
    
    # 2. 加载数据
    logger.info("📊 加载数据...")
    loader = DataLoader(
        data_dir=config_wfo["data"].get("data_dir"),
        cache_dir=config_wfo["data"].get("cache_dir"),
    )
    
    ohlcv = loader.load_ohlcv(
        etf_codes=pool_symbols,
        start_date=config_wfo["data"]["start_date"],
        end_date=config_wfo["data"]["end_date"],
    )
    
    # 过滤有效符号
    valid_symbols = [s for s in pool_symbols if s in ohlcv["close"].columns]
    logger.info(f"   有效 ETF: {len(valid_symbols)}")
    
    # 3. 计算因子
    logger.info("🔧 计算 18 个因子...")
    pool_ohlcv = {key: df[valid_symbols] for key, df in ohlcv.items()}
    
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(pool_ohlcv)
    
    factor_names = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    
    # 4. 横截面标准化
    logger.info("📐 横截面标准化...")
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # 5. 准备数据
    T = len(ohlcv["close"])
    N = len(valid_symbols)
    F = len(factor_names)
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = ohlcv["close"][valid_symbols].ffill().bfill().values
    
    # 计算未来收益
    forward_returns = np.zeros((T, N))
    for t in range(T - 1):
        for n in range(N):
            if close_prices[t, n] > 0:
                forward_returns[t + 1, n] = (close_prices[t + 1, n] - close_prices[t, n]) / close_prices[t, n]
            else:
                forward_returns[t + 1, n] = np.nan
    
    logger.info(f"   数据: {T}天 × {N}只ETF × {F}个因子")
    
    # 6. 生成所有因子组合
    logger.info("🔢 生成因子组合...")
    all_combos = []
    for size in COMBO_SIZES:
        combos = list(combinations(range(F), size))
        all_combos.extend([(c, size) for c in combos])
        logger.info(f"   {size}-因子组合: {len(combos)}")
    
    n_combos = len(all_combos)
    logger.info(f"   总计: {n_combos} 个组合")
    
    # 准备 Numba 数据
    max_combo_size = max(COMBO_SIZES)
    all_combo_indices = np.full((n_combos, max_combo_size), -1, dtype=np.int64)
    combo_sizes_arr = np.zeros(n_combos, dtype=np.int64)
    
    for i, (combo, size) in enumerate(all_combos):
        combo_sizes_arr[i] = size
        for j, idx in enumerate(combo):
            all_combo_indices[i, j] = idx
    
    # 7. WFO: 计算 IC/ICIR
    logger.info("")
    logger.info("=" * 80)
    logger.info("⚡ Phase 1: WFO 因子质量评估 (IC/ICIR)")
    logger.info("=" * 80)
    
    t0 = time.time()
    
    # 预热
    _ = _compute_all_combos_icir(
        factors_3d[:100], all_combo_indices[:10], combo_sizes_arr[:10],
        forward_returns[:100], 50
    )
    
    # 正式计算
    icir_results = _compute_all_combos_icir(
        factors_3d, all_combo_indices, combo_sizes_arr, forward_returns, LOOKBACK
    )
    
    logger.info(f"   WFO 完成，耗时: {time.time() - t0:.2f}秒")
    
    # 8. VEC: 回测所有组合
    logger.info("")
    logger.info("=" * 80)
    logger.info("⚡ Phase 2: VEC 策略收益评估")
    logger.info("=" * 80)
    
    # 生成调仓日
    rebalance_days = np.array([
        t for t in range(LOOKBACK, T, FREQ)
    ], dtype=np.int64)
    logger.info(f"   调仓日: {len(rebalance_days)} 次, 频率: {FREQ}天")
    
    t0 = time.time()
    
    # 预热
    _ = _vec_backtest_all_combos(
        close_prices[:100], factors_3d[:100], all_combo_indices[:10],
        combo_sizes_arr[:10], rebalance_days[:5], POS_SIZE, INITIAL_CAPITAL, COMMISSION_RATE
    )
    
    # 正式计算
    vec_results = _vec_backtest_all_combos(
        close_prices, factors_3d, all_combo_indices, combo_sizes_arr,
        rebalance_days, POS_SIZE, INITIAL_CAPITAL, COMMISSION_RATE
    )
    
    logger.info(f"   VEC 完成，耗时: {time.time() - t0:.2f}秒")
    
    # 9. 整合结果
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 整合结果")
    logger.info("=" * 80)
    
    results = []
    for i, (combo, size) in enumerate(all_combos):
        combo_names = [factor_names[idx] for idx in combo]
        combo_str = " + ".join(combo_names)
        
        ic_mean, ic_std, icir, n_valid = icir_results[i]
        tr, ar, mdd, sharpe, wr, nt = vec_results[i]
        
        results.append({
            "combo": combo_str,
            "combo_size": size,
            # WFO 指标
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "ic_valid_days": int(n_valid),
            # VEC 指标
            "total_return": tr,
            "annual_return": ar,
            "max_drawdown": mdd,
            "sharpe_ratio": sharpe,
            "win_rate": wr,
            "num_trades": int(nt),
        })
    
    df = pd.DataFrame(results)
    
    # 10. 多维排名
    df["rank_icir"] = df["icir"].rank(ascending=False)
    df["rank_return"] = df["total_return"].rank(ascending=False)
    df["rank_sharpe"] = df["sharpe_ratio"].rank(ascending=False)
    df["rank_mdd"] = df["max_drawdown"].rank(ascending=True)  # 回撤越小越好
    
    # 综合得分 (简单平均排名)
    df["composite_rank"] = (
        df["rank_icir"] + df["rank_return"] + df["rank_sharpe"] + df["rank_mdd"]
    ) / 4
    
    df = df.sort_values("composite_rank").reset_index(drop=True)
    df["final_rank"] = range(1, len(df) + 1)
    
    # 11. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"single_pool_full_{POOL_NAME}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_dir / "all_results.parquet", index=False)
    df.to_csv(output_dir / "all_results.csv", index=False)
    df.head(100).to_csv(output_dir / "top100.csv", index=False)
    
    # 保存配置
    run_config = {
        "timestamp": timestamp,
        "pool_name": POOL_NAME,
        "symbols": valid_symbols,
        "n_symbols": len(valid_symbols),
        "n_factors": F,
        "factor_names": factor_names,
        "combo_sizes": COMBO_SIZES,
        "n_combos": n_combos,
        "parameters": {
            "lookback": LOOKBACK,
            "freq": FREQ,
            "pos_size": POS_SIZE,
            "initial_capital": INITIAL_CAPITAL,
            "commission_rate": COMMISSION_RATE,
        }
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    
    # 12. 输出结果
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ 完成 | 总耗时: {elapsed:.1f}秒 | 组合数: {n_combos}")
    logger.info("=" * 80)
    
    logger.info("")
    logger.info("🏆 TOP 20 综合排名 (WFO + VEC)")
    logger.info("-" * 120)
    
    cols = ["final_rank", "combo", "icir", "total_return", "sharpe_ratio", "max_drawdown"]
    print(df[cols].head(20).to_string(index=False))
    
    logger.info("")
    logger.info("📊 统计摘要")
    logger.info("-" * 80)
    logger.info(f"   总收益 - 均值: {df['total_return'].mean():.2%}, 最大: {df['total_return'].max():.2%}, 最小: {df['total_return'].min():.2%}")
    logger.info(f"   夏普比 - 均值: {df['sharpe_ratio'].mean():.2f}, 最大: {df['sharpe_ratio'].max():.2f}")
    logger.info(f"   ICIR   - 均值: {df['icir'].mean():.4f}, 最大: {df['icir'].max():.4f}")
    
    # ICIR vs Return 相关性
    corr = df["icir"].corr(df["total_return"])
    logger.info(f"   ICIR vs Return 相关性: {corr:.4f}")
    
    logger.info("")
    logger.info(f"📁 结果已保存至: {output_dir}")
    
    return df, output_dir


if __name__ == "__main__":
    main()
