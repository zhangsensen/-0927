#!/usr/bin/env python3
"""
分池 WFO 优化器 | Pool-Specific WFO Optimizer
================================================================================
核心思路（来自用户洞察）：
    "把 18 个因子分别在 7 个子池里面去应用，
     然后就能知道这些 7 个子池里面的 WFO 策略"

关键修复：
1. 每个池独立加载数据和计算因子
2. 池内横截面标准化（不是全量标准化）
3. 分类因子正确合并到对应池
4. 输出格式与 VEC 回测兼容

输出：
- results/pool_wfo_{timestamp}/
  ├── pool_results.json      # 每个池的最优因子组合
  ├── pool_metrics.json      # 每个池的 IC/ICIR 指标
  └── best_config.json       # 供 VEC 回测使用的统一配置
================================================================================
"""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from numba import njit, prange

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from etf_rotation_optimized.core.data_loader import DataLoader
from etf_rotation_optimized.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_rotation_optimized.core.cross_section_processor import CrossSectionProcessor
from etf_rotation_optimized.core.category_factors import CategoryFactorManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Numba 加速函数
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
    
    numerator = np.sum((s_rank - s_mean) * (r_rank - r_mean))
    s_std = np.sqrt(np.sum((s_rank - s_mean) ** 2))
    r_std = np.sqrt(np.sum((r_rank - r_mean) ** 2))
    
    if s_std > 0 and r_std > 0:
        return numerator / (s_std * r_std)
    return np.nan


@njit(cache=True)
def _compute_combo_icir(
    factors_3d: np.ndarray,
    factor_indices: np.ndarray,
    forward_returns: np.ndarray,
    lookback: int,
) -> Tuple[float, float, float, int]:
    """计算因子组合的 IC/ICIR"""
    T, N, _ = factors_3d.shape
    n_factors = len(factor_indices)
    
    ic_values = []
    
    for t in range(lookback, T):
        # 因子组合得分（等权相加）
        combo_score = np.zeros(N)
        valid_count = 0
        
        for n in range(N):
            score = 0.0
            n_valid_factors = 0
            for f_idx in factor_indices:
                val = factors_3d[t-1, n, f_idx]
                if not np.isnan(val):
                    score += val
                    n_valid_factors += 1
            
            if n_valid_factors > 0:
                combo_score[n] = score / n_valid_factors  # 平均
                valid_count += 1
            else:
                combo_score[n] = np.nan
        
        # 需要至少 3 个有效资产才能计算 IC
        if valid_count >= 3:
            ic = _compute_spearman_ic(combo_score, forward_returns[t])
            if not np.isnan(ic):
                ic_values.append(ic)
    
    n_valid = len(ic_values)
    if n_valid < 20:
        return 0.0, 0.0, 0.0, n_valid
    
    # 计算统计量
    ic_arr = np.array(ic_values)
    ic_mean = np.mean(ic_arr)
    ic_std = np.std(ic_arr)
    
    if ic_std > 0.001:
        icir = ic_mean / ic_std
    else:
        icir = 0.0
    
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
# 单池优化器
# =============================================================================

class PoolOptimizer:
    """单个池的 WFO 优化器"""
    
    def __init__(
        self,
        pool_name: str,
        symbols: List[str],
        ohlcv: Dict[str, pd.DataFrame],
        lookback: int = 252,
    ):
        self.pool_name = pool_name
        self.symbols = symbols
        self.ohlcv = ohlcv
        self.lookback = lookback
        
        self.factor_lib = PreciseFactorLibrary()
        self.category_mgr = CategoryFactorManager()
        self.processor = CrossSectionProcessor(verbose=False)
        
        self.factors_3d = None
        self.factor_names = []
        self.forward_returns = None
        
    def prepare_data(self):
        """准备数据：计算因子并标准化"""
        logger.info(f"  📊 池 {self.pool_name}: {len(self.symbols)} 个 ETF")
        
        # 1. 提取池内数据
        close_df = self.ohlcv["close"][self.symbols].ffill().bfill()
        
        # 2. 计算通用因子（18个）
        pool_ohlcv = {
            key: df[self.symbols] if isinstance(df, pd.DataFrame) else df
            for key, df in self.ohlcv.items()
        }
        
        raw_factors_df = self.factor_lib.compute_all_factors(pool_ohlcv)
        
        # 3. 计算分类因子（如果适用）
        category_factors = self._compute_category_factors(pool_ohlcv)
        
        # 4. 合并所有因子
        if not category_factors.empty:
            # 将分类因子转换为与通用因子相同的格式
            cat_factor_dict = self._reshape_category_factors(category_factors)
            
            # 合并
            all_factor_names = list(raw_factors_df.columns.get_level_values(0).unique())
            all_factor_names.extend(cat_factor_dict.keys())
            
            factor_dict = {fname: raw_factors_df[fname] for fname in raw_factors_df.columns.get_level_values(0).unique()}
            factor_dict.update(cat_factor_dict)
        else:
            factor_dict = {fname: raw_factors_df[fname] for fname in raw_factors_df.columns.get_level_values(0).unique()}
        
        # 5. 池内横截面标准化
        std_factors = self.processor.process_all_factors(factor_dict)
        
        # 6. 构建 3D 数组 [T, N, F]
        self.factor_names = sorted(std_factors.keys())
        T = len(close_df)
        N = len(self.symbols)
        F = len(self.factor_names)
        
        self.factors_3d = np.zeros((T, N, F))
        for f_idx, fname in enumerate(self.factor_names):
            self.factors_3d[:, :, f_idx] = std_factors[fname].values
        
        # 7. 计算前向收益
        self.forward_returns = close_df.pct_change(fill_method=None).shift(-1).values
        
        logger.info(f"     ✅ 因子: {F} 个, 时间: {T} 天")
        
    def _compute_category_factors(self, pool_ohlcv: Dict) -> pd.DataFrame:
        """计算分类专属因子"""
        pool_upper = self.pool_name.upper()
        
        if pool_upper == "BOND":
            return self.category_mgr.bond_factors.compute_all(pool_ohlcv, self.symbols)
        elif pool_upper == "COMMODITY":
            market_proxy = "510300" if "510300" in self.ohlcv["close"].columns else None
            return self.category_mgr.commodity_factors.compute_all(pool_ohlcv, self.symbols, market_proxy)
        elif pool_upper == "QDII":
            return self.category_mgr.qdii_factors.compute_all(pool_ohlcv, self.symbols)
        else:
            return pd.DataFrame()
    
    def _reshape_category_factors(self, cat_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """将分类因子从 (factor, symbol) MultiIndex 转换为字典格式"""
        result = {}
        
        if cat_df.empty:
            return result
        
        # cat_df 的列是 MultiIndex (factor, symbol)
        for factor_name in cat_df.columns.get_level_values(0).unique():
            factor_data = cat_df[factor_name]
            result[factor_name] = factor_data
        
        return result
    
    def run_optimization(
        self,
        combo_sizes: List[int] = [2, 3, 4],
        max_combos: int = 3000,
    ) -> Dict:
        """执行 WFO 优化"""
        if self.factors_3d is None:
            self.prepare_data()
        
        n_factors = len(self.factor_names)
        
        # 生成所有组合
        all_combos = []
        for size in combo_sizes:
            if size > n_factors:
                continue
            combos = list(combinations(range(n_factors), size))
            all_combos.extend([(c, size) for c in combos])
        
        # 随机采样（如果太多）
        if len(all_combos) > max_combos:
            import random
            random.seed(42)
            all_combos = random.sample(all_combos, max_combos)
            logger.info(f"     🎲 采样至 {max_combos} 个组合")
        
        n_combos = len(all_combos)
        if n_combos == 0:
            logger.warning(f"     ⚠️ 池 {self.pool_name} 组合数为 0")
            return {"best_factors": [], "icir": 0.0, "ic_mean": 0.0}
        
        # 准备 Numba 输入
        max_size = max(combo_sizes)
        all_combo_indices = np.full((n_combos, max_size), -1, dtype=np.int64)
        combo_sizes_arr = np.zeros(n_combos, dtype=np.int64)
        
        for i, (combo, size) in enumerate(all_combos):
            combo_sizes_arr[i] = size
            for j, idx in enumerate(combo):
                all_combo_indices[i, j] = idx
        
        # 计算 ICIR
        logger.info(f"     ⚡ 计算 {n_combos} 个组合的 ICIR...")
        results = _compute_all_combos_icir(
            self.factors_3d,
            all_combo_indices,
            combo_sizes_arr,
            self.forward_returns,
            self.lookback,
        )
        
        # 整理结果
        valid_results = []
        for i, (combo, size) in enumerate(all_combos):
            ic_mean, ic_std, icir, n_valid = results[i]
            if n_valid >= 20 and icir > 0:  # 只保留正 ICIR 的组合
                combo_names = [self.factor_names[idx] for idx in combo]
                valid_results.append({
                    "factors": combo_names,
                    "icir": float(icir),
                    "ic_mean": float(ic_mean),
                    "ic_std": float(ic_std),
                    "n_valid": int(n_valid),
                })
        
        if not valid_results:
            logger.warning(f"     ⚠️ 池 {self.pool_name} 没有有效组合")
            # 回退到默认因子
            default_factors = self._get_default_factors()
            return {
                "best_factors": default_factors,
                "icir": 0.0,
                "ic_mean": 0.0,
                "fallback": True,
            }
        
        # 排序选择最佳
        valid_results.sort(key=lambda x: x["icir"], reverse=True)
        best = valid_results[0]
        
        logger.info(f"     🏆 最佳: {best['factors']} (ICIR: {best['icir']:.3f})")
        
        return {
            "best_factors": best["factors"],
            "icir": best["icir"],
            "ic_mean": best["ic_mean"],
            "top5": valid_results[:5],
        }
    
    def _get_default_factors(self) -> List[str]:
        """获取回退默认因子"""
        pool_upper = self.pool_name.upper()
        
        if pool_upper == "BOND":
            return ["MOM_20D", "SHARPE_RATIO_20D"]
        elif pool_upper == "COMMODITY":
            return ["MOM_20D", "SLOPE_20D"]
        elif pool_upper == "QDII":
            return ["MOM_20D", "ADX_14D", "SHARPE_RATIO_20D"]
        else:
            # 权益类默认：Rank 3
            return ["ADX_14D", "PRICE_POSITION_20D", "SHARPE_RATIO_20D", "SLOPE_20D"]


# =============================================================================
# 主函数
# =============================================================================

def run_pool_wfo():
    """执行分池 WFO 优化"""
    logger.info("=" * 80)
    logger.info("🌊 分池 WFO 优化器 | Pool-Specific WFO Optimizer")
    logger.info("=" * 80)
    
    # 1. 加载配置
    config_wfo_path = ROOT / "configs/combo_wfo_config.yaml"
    config_pools_path = ROOT / "configs/etf_pools.yaml"
    
    with open(config_wfo_path) as f:
        config_wfo = yaml.safe_load(f)
    with open(config_pools_path) as f:
        config_pools = yaml.safe_load(f)
    
    # 2. 加载全量数据（一次性）
    logger.info("\n📊 加载数据...")
    
    all_symbols = []
    for pool in config_pools["pools"].values():
        all_symbols.extend(pool["symbols"])
    all_symbols = sorted(list(set(all_symbols)))
    
    loader = DataLoader(
        data_dir=config_wfo["data"].get("data_dir"),
        cache_dir=config_wfo["data"].get("cache_dir"),
    )
    
    ohlcv = loader.load_ohlcv(
        etf_codes=all_symbols,
        start_date=config_wfo["data"]["start_date"],
        end_date=config_wfo["data"]["end_date"],
    )
    
    logger.info(f"✅ 数据加载完成: {len(all_symbols)} ETF, {len(ohlcv['close'])} 天")
    
    # 3. 对每个池执行优化
    logger.info("\n🚀 开始分池优化...")
    
    pool_results = {}
    target_pools = [p for p in config_pools["pools"] if p != "A_SHARE_LIVE"]
    
    for pool_name in target_pools:
        pool_config = config_pools["pools"][pool_name]
        symbols = [s for s in pool_config["symbols"] if s in ohlcv["close"].columns]
        
        if len(symbols) < 2:
            logger.warning(f"⚠️ 跳过池 {pool_name}: 有效 ETF 数 < 2")
            continue
        
        optimizer = PoolOptimizer(
            pool_name=pool_name,
            symbols=symbols,
            ohlcv=ohlcv,
            lookback=252,
        )
        
        result = optimizer.run_optimization(
            combo_sizes=[2, 3, 4],
            max_combos=2000,
        )
        
        pool_results[pool_name] = result
    
    # 4. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"pool_wfo_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4.1 保存详细结果
    with open(output_dir / "pool_results.json", "w") as f:
        json.dump(pool_results, f, indent=2, ensure_ascii=False)
    
    # 4.2 生成 VEC 兼容的配置
    best_config = {
        "timestamp": timestamp,
        "pool_factors": {
            pool_name: result["best_factors"]
            for pool_name, result in pool_results.items()
        },
        "pool_weights": {
            pool_name: config_pools["capital_constraints"].get(pool_name, {}).get("target_capital", 0.1)
            for pool_name in pool_results
        },
    }
    
    with open(output_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    # 4.3 同时保存到 latest
    latest_file = ROOT / "results" / "pool_wfo_best_config_latest.json"
    with open(latest_file, "w") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    # 5. 打印汇总
    logger.info("\n" + "=" * 80)
    logger.info("📊 优化结果汇总")
    logger.info("=" * 80)
    
    for pool_name, result in pool_results.items():
        factors = result.get("best_factors", [])
        icir = result.get("icir", 0)
        fallback = result.get("fallback", False)
        status = "⚠️ 回退" if fallback else "✅"
        
        logger.info(f"\n{pool_name}:")
        logger.info(f"  {status} 因子: {factors}")
        logger.info(f"  ICIR: {icir:.3f}")
    
    logger.info(f"\n💾 结果已保存至: {output_dir}")
    logger.info(f"💾 最新配置: {latest_file}")
    
    return pool_results


if __name__ == "__main__":
    run_pool_wfo()
