#!/usr/bin/env python3
"""
WFO 因子组合优化器（IC/ICIR 排序）

核心原则：
- WFO 负责因子质量评估（IC/ICIR 排序）
- VEC/BT 负责收益评估（策略表现）
- 职责分离，不重叠

排序逻辑：
- 主指标：ICIR = IC_mean / IC_std（信息比率）
- IC_mean：因子组合得分 vs 未来收益的 Spearman 相关系数均值
- IC_std：IC 的标准差（稳定性）

用法: uv run python etf_strategy/run_unified_wfo.py
"""

import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from numba import njit, prange

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# 核心参数
# ============================================================================
LOOKBACK = 252        # 回测起点（跳过前252天热身）
MIN_VALID_DAYS = 20   # IC 计算最少有效天数


@njit(cache=True)
def _compute_spearman_ic_single_day(scores: np.ndarray, returns: np.ndarray) -> float:
    """
    计算单日的 Spearman IC
    
    Args:
        scores: (N,) 因子组合得分
        returns: (N,) 未来收益
    
    Returns:
        IC 值（如果无效返回 NaN）
    """
    # 去除 NaN
    mask = ~(np.isnan(scores) | np.isnan(returns))
    n_valid = np.sum(mask)
    
    if n_valid < 3:
        return np.nan
    
    s = scores[mask]
    r = returns[mask]
    
    # 计算秩
    s_rank = np.argsort(np.argsort(s)).astype(np.float64)
    r_rank = np.argsort(np.argsort(r)).astype(np.float64)
    
    # Spearman 相关系数
    s_mean = np.mean(s_rank)
    r_mean = np.mean(r_rank)
    
    numerator = np.sum((s_rank - s_mean) * (r_rank - r_mean))
    s_std = np.sqrt(np.sum((s_rank - s_mean) ** 2))
    r_std = np.sqrt(np.sum((r_rank - r_mean) ** 2))
    
    if s_std > 0 and r_std > 0:
        return numerator / (s_std * r_std)
    return np.nan


@njit(cache=True)
def _compute_combo_ic_series(
    factors_3d: np.ndarray,
    factor_indices: np.ndarray,
    forward_returns: np.ndarray,
    lookback: int,
    min_valid_days: int,
) -> tuple:
    """
    计算单个因子组合的 IC 时间序列
    
    Args:
        factors_3d: (T, N, F) 因子数据
        factor_indices: 因子索引数组
        forward_returns: (T, N) 未来收益
        lookback: 跳过的天数
        min_valid_days: 最少有效天数
    
    Returns:
        (ic_mean, ic_std, ic_ir, n_valid_days)
    """
    T, N, _ = factors_3d.shape
    n_factors = len(factor_indices)
    
    ic_values = np.zeros(T - lookback)
    valid_count = 0
    
    for t in range(lookback, T):
        # 计算因子组合得分（T-1 时刻的因子值）
        combo_score = np.zeros(N)
        for n in range(N):
            score = 0.0
            for i in range(n_factors):
                f_idx = factor_indices[i]
                val = factors_3d[t-1, n, f_idx]
                if not np.isnan(val):
                    score += val
            combo_score[n] = score
        
        # 计算 IC（因子得分 vs 未来收益）
        ic = _compute_spearman_ic_single_day(combo_score, forward_returns[t])
        ic_values[t - lookback] = ic
        if not np.isnan(ic):
            valid_count += 1
    
    # 计算统计量
    if valid_count >= min_valid_days:
        # 去除 NaN 计算均值和标准差
        valid_ics = np.zeros(valid_count)
        idx = 0
        for i in range(len(ic_values)):
            if not np.isnan(ic_values[i]):
                valid_ics[idx] = ic_values[i]
                idx += 1
        
        ic_mean = np.mean(valid_ics)
        ic_std = np.std(valid_ics)
        
        if ic_std > 0.001:
            ic_ir = ic_mean / ic_std
        else:
            ic_ir = 0.0
        
        return ic_mean, ic_std, ic_ir, valid_count
    
    return 0.0, 0.0, 0.0, valid_count


@njit(parallel=True, cache=True)
def _compute_all_combo_ics(
    factors_3d: np.ndarray,
    all_combo_indices: np.ndarray,
    combo_sizes: np.ndarray,
    forward_returns: np.ndarray,
    lookback: int,
    min_valid_days: int,
) -> np.ndarray:
    """
    并行计算所有因子组合的 IC/ICIR
    
    Args:
        factors_3d: (T, N, F) 因子数据
        all_combo_indices: (n_combos, max_combo_size) 因子索引，-1 表示无效
        combo_sizes: (n_combos,) 每个组合的实际大小
        forward_returns: (T, N) 未来收益
        lookback: 跳过的天数
        min_valid_days: 最少有效天数
    
    Returns:
        (n_combos, 4) 数组，列为 [ic_mean, ic_std, ic_ir, n_valid]
    """
    n_combos = all_combo_indices.shape[0]
    results = np.zeros((n_combos, 4))
    
    for i in prange(n_combos):
        size = combo_sizes[i]
        factor_indices = all_combo_indices[i, :size]
        
        ic_mean, ic_std, ic_ir, n_valid = _compute_combo_ic_series(
            factors_3d, factor_indices, forward_returns, lookback, min_valid_days
        )
        
        results[i, 0] = ic_mean
        results[i, 1] = ic_std
        results[i, 2] = ic_ir
        results[i, 3] = n_valid
    
    return results


def run_unified_wfo():
    """主函数"""
    start_time = datetime.now()
    
    logger.info("=" * 80)
    logger.info("🎯 WFO 因子组合优化器（IC/ICIR 排序）")
    logger.info("=" * 80)
    logger.info("核心原则: WFO 评估因子质量 → VEC/BT 评估策略收益")
    logger.info("排序指标: ICIR = IC_mean / IC_std（信息比率）")
    logger.info("")
    
    # 1. 加载配置
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. 加载数据
    logger.info("📊 加载数据...")
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    # 3. 计算因子
    logger.info("🔧 计算因子...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    
    factor_names = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    
    # 4. 横截面标准化
    logger.info("📐 横截面标准化...")
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # 5. 准备数据
    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    T, N = first_factor.shape
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    
    # 6. 计算未来收益（T+1 收益，用于 IC 计算）
    logger.info("📈 计算未来收益...")
    forward_returns = np.zeros((T, N))
    for t in range(T - 1):
        for n in range(N):
            if close_prices[t, n] > 0 and not np.isnan(close_prices[t + 1, n]):
                forward_returns[t + 1, n] = (close_prices[t + 1, n] - close_prices[t, n]) / close_prices[t, n]
            else:
                forward_returns[t + 1, n] = np.nan
    
    logger.info(f"   数据: {T}天 × {N}只ETF × {len(factor_names)}个因子")
    
    # 7. 生成所有组合
    combo_sizes_config = config["combo_wfo"]["combo_sizes"]
    all_combos = []
    for size in combo_sizes_config:
        combos = list(combinations(range(len(factor_names)), size))
        all_combos.extend([(c, size) for c in combos])
        logger.info(f"   {size}-因子组合: {len(combos)}")
    logger.info(f"   总计: {len(all_combos)} 个组合")
    
    # 8. 准备 Numba 数据结构
    max_combo_size = max(combo_sizes_config)
    n_combos = len(all_combos)
    all_combo_indices = np.full((n_combos, max_combo_size), -1, dtype=np.int64)
    combo_sizes = np.zeros(n_combos, dtype=np.int64)
    
    for i, (combo, size) in enumerate(all_combos):
        combo_sizes[i] = size
        for j, idx in enumerate(combo):
            all_combo_indices[i, j] = idx
    
    # 9. 计算所有组合的 IC/ICIR
    logger.info("")
    logger.info("⚡ 计算 IC/ICIR（因子质量评估）")
    logger.info("-" * 80)
    
    # 预热 Numba
    _ = _compute_all_combo_ics(
        factors_3d[:100],
        all_combo_indices[:10],
        combo_sizes[:10],
        forward_returns[:100],
        50,
        MIN_VALID_DAYS,
    )
    
    # 正式计算
    from tqdm import tqdm
    import time
    
    logger.info("   并行计算中...")
    t0 = time.time()
    ic_results = _compute_all_combo_ics(
        factors_3d,
        all_combo_indices,
        combo_sizes,
        forward_returns,
        LOOKBACK,
        MIN_VALID_DAYS,
    )
    logger.info(f"   IC 计算完成，耗时: {time.time() - t0:.2f}秒")
    
    # 10. 构建结果 DataFrame
    results = []
    for i, (combo, size) in enumerate(all_combos):
        combo_str = " + ".join([factor_names[idx] for idx in combo])
        ic_mean, ic_std, ic_ir, n_valid = ic_results[i]
        
        if n_valid >= MIN_VALID_DAYS:
            results.append({
                "combo": combo_str,
                "combo_size": size,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": ic_ir,
                "n_valid_days": int(n_valid),
            })
    
    # 11. 排序（主指标：ICIR）
    df = pd.DataFrame(results)
    df = df.sort_values("icir", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    
    # 12. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir.parent / "results" / f"unified_wfo_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_dir / "all_combos.parquet", index=False)
    df.head(100).to_parquet(output_dir / "top100.parquet", index=False)
    df.to_csv(output_dir / "all_combos.csv", index=False)
    
    # 保存因子
    factors_dir = output_dir / "factors"
    factors_dir.mkdir(exist_ok=True)
    for fname in factor_names:
        std_factors[fname].to_parquet(factors_dir / f"{fname}.parquet")
    
    # 保存配置
    run_config = {
        "timestamp": timestamp,
        "rule": "IC/ICIR-based (Factor Quality)",
        "ranking_metric": "ICIR = IC_mean / IC_std",
        "parameters": {
            "lookback": LOOKBACK,
            "min_valid_days": MIN_VALID_DAYS,
        },
        "data": config["data"],
        "note": "WFO 评估因子质量，VEC/BT 评估策略收益",
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    
    # 13. 输出结果
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ 完成 | 耗时: {elapsed:.1f}秒 | 有效组合: {len(df)}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("🏆 TOP 20 因子组合（按 ICIR 排序）")
    logger.info("-" * 80)
    print(f"{'Rank':>4} | {'ICIR':>8} | {'IC_mean':>8} | {'IC_std':>8} | {'Days':>5} | Combo")
    print("-" * 100)
    
    for _, row in df.head(20).iterrows():
        print(f"{row['rank']:>4} | {row['icir']:>8.4f} | "
              f"{row['ic_mean']:>8.4f} | {row['ic_std']:>8.4f} | "
              f"{row['n_valid_days']:>5} | {row['combo'][:50]}")
    
    logger.info("")
    logger.info(f"📁 输出目录: {output_dir}")
    logger.info("")
    logger.info("💡 下一步: 用 VEC/BT 评估 Top-N 组合的策略收益")
    logger.info("   uv run python scripts/batch_vec_backtest.py")
    
    return df, output_dir


if __name__ == "__main__":
    run_unified_wfo()
