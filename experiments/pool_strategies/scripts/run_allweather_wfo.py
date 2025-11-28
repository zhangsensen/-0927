#!/usr/bin/env python3
"""
全天候策略专用 WFO 优化器 | All-Weather WFO Optimizer
================================================================================
功能：
1. 针对全天候策略的 7 个子池分别运行 WFO 优化
2. 为每个子池寻找最佳的因子组合 (基于 ICIR)
3. 生成 unified_config.json 供回测引擎使用

子池定义：
- EQUITY_BROAD, EQUITY_GROWTH, EQUITY_CYCLICAL, EQUITY_DEFENSIVE
- BOND, COMMODITY, QDII

输出：
- results/allweather_wfo_YYYYMMDD_HHMMSS/best_combos.json
================================================================================
"""

import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple

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

# ============================================================================
# Numba 加速函数 (从 run_unified_wfo.py 移植)
# ============================================================================

@njit(cache=True)
def _compute_spearman_ic_single_day(scores: np.ndarray, returns: np.ndarray) -> float:
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
def _compute_combo_ic_series(
    factors_3d: np.ndarray,
    factor_indices: np.ndarray,
    forward_returns: np.ndarray,
    lookback: int,
    min_valid_days: int,
) -> Tuple[float, float, float, int]:
    T, N, _ = factors_3d.shape
    n_factors = len(factor_indices)
    
    ic_values = np.zeros(T - lookback)
    valid_count = 0
    
    for t in range(lookback, T):
        combo_score = np.zeros(N)
        for n in range(N):
            score = 0.0
            valid_factors = 0
            for i in range(n_factors):
                f_idx = factor_indices[i]
                val = factors_3d[t-1, n, f_idx]
                if not np.isnan(val):
                    score += val
                    valid_factors += 1
            
            # 只有当至少有一个有效因子时才计算得分
            if valid_factors > 0:
                combo_score[n] = score
            else:
                combo_score[n] = np.nan
        
        ic = _compute_spearman_ic_single_day(combo_score, forward_returns[t])
        ic_values[t - lookback] = ic
        if not np.isnan(ic):
            valid_count += 1
    
    if valid_count >= min_valid_days:
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

# ============================================================================
# 主逻辑
# ============================================================================

class AllWeatherWFO:
    def __init__(self):
        self.config_wfo = self._load_yaml(ROOT / "configs/combo_wfo_config.yaml")
        self.config_pools = self._load_yaml(ROOT / "configs/etf_pools.yaml")
        
        self.loader = DataLoader(
            data_dir=self.config_wfo["data"].get("data_dir"),
            cache_dir=self.config_wfo["data"].get("cache_dir"),
        )
        self.factor_lib = PreciseFactorLibrary()
        self.category_factor_mgr = CategoryFactorManager()
        self.processor = CrossSectionProcessor(verbose=False)
        
        self.ohlcv = None
        self.factors_dict = {}
        self.factor_names = []
        self.etf_codes = []
        
    def _load_yaml(self, path: Path) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)
            
    def load_and_prepare_data(self):
        """加载数据并计算所有因子"""
        logger.info("📊 加载全量数据...")
        
        # 1. 获取所有池的并集符号
        all_symbols = []
        for pool in self.config_pools["pools"].values():
            all_symbols.extend(pool["symbols"])
        all_symbols = sorted(list(set(all_symbols)))
        
        # 2. 加载 OHLCV
        self.ohlcv = self.loader.load_ohlcv(
            etf_codes=all_symbols,
            start_date=self.config_wfo["data"]["start_date"],
            end_date=self.config_wfo["data"]["end_date"],
        )
        self.etf_codes = self.ohlcv["close"].columns.tolist()
        
        # 3. 计算通用因子
        logger.info("🔧 计算通用因子...")
        raw_factors_df = self.factor_lib.compute_all_factors(self.ohlcv)
        
        # 4. 计算分类因子 (Bond, Commodity, QDII)
        logger.info("🔧 计算分类因子...")
        bond_symbols = self.config_pools["pools"]["BOND"]["symbols"]
        comm_symbols = self.config_pools["pools"]["COMMODITY"]["symbols"]
        qdii_symbols = self.config_pools["pools"]["QDII"]["symbols"]
        
        market_proxy = "510300.SH" if "510300.SH" in self.ohlcv["close"].columns else self.ohlcv["close"].columns[0]
        
        bond_factors = self.category_factor_mgr.compute_factors_for_pool("BOND", self.ohlcv, bond_symbols)
        comm_factors = self.category_factor_mgr.compute_factors_for_pool("COMMODITY", self.ohlcv, comm_symbols, market_proxy)
        qdii_factors = self.category_factor_mgr.compute_factors_for_pool("QDII", self.ohlcv, qdii_symbols)
        
        # 5. 合并所有因子
        all_dfs = [raw_factors_df]
        if not bond_factors.empty: all_dfs.append(bond_factors)
        if not comm_factors.empty: all_dfs.append(comm_factors)
        if not qdii_factors.empty: all_dfs.append(qdii_factors)
        
        combined_df = pd.concat(all_dfs, axis=1)
        
        # 6. 标准化
        logger.info("📐 标准化因子...")
        factor_names_list = combined_df.columns.get_level_values(0).unique().tolist()
        raw_factors_dict = {fname: combined_df[fname] for fname in factor_names_list}
        
        processed_factors = self.processor.process_all_factors(raw_factors_dict)
        
        # 关键修复：确保所有因子 DataFrame 都对齐到完整的 etf_codes
        # 这样后续使用 symbol_indices 切片时才不会越界
        self.factors_dict = {}
        for fname, df in processed_factors.items():
            # Reindex columns to match self.etf_codes, filling with NaN
            aligned_df = df.reindex(columns=self.etf_codes)
            self.factors_dict[fname] = aligned_df
            
        self.factor_names = sorted(self.factors_dict.keys())
        
        logger.info(f"✅ 数据准备完成: {len(self.etf_codes)} ETFs, {len(self.factor_names)} Factors")

    def run_pool_optimization(self, pool_name: str, pool_config: Dict) -> Dict:
        """对单个池运行 WFO 优化"""
        logger.info(f"🚀 优化子池: {pool_name}")
        
        pool_symbols = pool_config["symbols"]
        # 过滤出存在的 symbols
        valid_symbols = [s for s in pool_symbols if s in self.etf_codes]
        if not valid_symbols:
            logger.warning(f"⚠️ 池 {pool_name} 没有有效的数据，跳过")
            return None
            
        # 1. 准备该池的数据切片
        # 找出 valid_symbols 在 self.etf_codes 中的索引
        symbol_indices = [self.etf_codes.index(s) for s in valid_symbols]
        
        # 准备因子 3D 数组 (T, N_subset, F)
        # 注意：我们需要筛选出对该池有意义的因子
        # 比如 Bond 池不需要看 Equity 因子，反之亦然
        # 但为了简单，我们先使用所有因子，IC 会自动过滤无效值 (NaN)
        
        # 优化：只选择在该池上有非 NaN 值的因子
        # 这一步可以显著减少计算量
        relevant_factors = []
        for fname in self.factor_names:
            # 检查该因子在这些 symbol 上是否全为 NaN
            f_data = self.factors_dict[fname].values[:, symbol_indices]
            if not np.isnan(f_data).all():
                relevant_factors.append(fname)
        
        logger.info(f"   有效因子数: {len(relevant_factors)} / {len(self.factor_names)}")
        
        factors_3d = np.stack([
            self.factors_dict[f].values[:, symbol_indices] 
            for f in relevant_factors
        ], axis=-1)
        
        # 准备收益率
        close_prices = self.ohlcv["close"][valid_symbols].ffill().bfill().values
        T, N = close_prices.shape
        forward_returns = np.zeros((T, N))
        for t in range(T - 1):
            for n in range(N):
                if close_prices[t, n] > 0:
                    forward_returns[t + 1, n] = (close_prices[t + 1, n] - close_prices[t, n]) / close_prices[t, n]
                else:
                    forward_returns[t + 1, n] = np.nan
                    
        # 2. 生成组合
        combo_sizes = [2, 3, 4] # 限制组合大小以加快速度
        all_combos = []
        for size in combo_sizes:
            combos = list(combinations(range(len(relevant_factors)), size))
            all_combos.extend([(c, size) for c in combos])
            
        # 限制组合数量 (随机采样如果太多)
        MAX_COMBOS = 5000
        if len(all_combos) > MAX_COMBOS:
            import random
            random.seed(42)
            all_combos = random.sample(all_combos, MAX_COMBOS)
            logger.info(f"   组合过多，采样至 {MAX_COMBOS} 个")
            
        # 3. 运行 IC 计算
        n_combos = len(all_combos)
        max_size = max(combo_sizes)
        all_combo_indices = np.full((n_combos, max_size), -1, dtype=np.int64)
        combo_sizes_arr = np.zeros(n_combos, dtype=np.int64)
        
        for i, (combo, size) in enumerate(all_combos):
            combo_sizes_arr[i] = size
            for j, idx in enumerate(combo):
                all_combo_indices[i, j] = idx
                
        ic_results = _compute_all_combo_ics(
            factors_3d,
            all_combo_indices,
            combo_sizes_arr,
            forward_returns,
            lookback=252,
            min_valid_days=20
        )
        
        # 4. 整理结果
        results = []
        for i, (combo, size) in enumerate(all_combos):
            ic_mean, ic_std, ic_ir, n_valid = ic_results[i]
            if n_valid >= 20:
                combo_names = [relevant_factors[idx] for idx in combo]
                results.append({
                    "combo": combo_names,
                    "icir": ic_ir,
                    "ic_mean": ic_mean
                })
        
        # 5. 选出最佳
        if not results:
            logger.warning(f"⚠️ 池 {pool_name} 没有找到有效组合")
            return None
            
        results.sort(key=lambda x: x["icir"], reverse=True)
        best = results[0]
        logger.info(f"   🏆 最佳组合: {best['combo']} (ICIR: {best['icir']:.2f})")
        
        return best["combo"]

    def run(self):
        self.load_and_prepare_data()
        
        best_combos = {}
        
        # 遍历所有池
        pools = self.config_pools["pools"]
        # 排除 A_SHARE_LIVE
        target_pools = [p for p in pools if p != "A_SHARE_LIVE"]
        
        for pool_name in target_pools:
            pool_config = pools[pool_name]
            best_combo = self.run_pool_optimization(pool_name, pool_config)
            if best_combo:
                best_combos[pool_name] = best_combo
                
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "results" / f"allweather_wfo_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "best_combos.json"
        with open(output_file, "w") as f:
            json.dump(best_combos, f, indent=2)
            
        logger.info(f"✅ 优化完成，结果已保存至: {output_file}")
        
        # 同时更新 latest 软链接或固定路径供回测脚本读取
        latest_file = ROOT / "results" / "allweather_best_combos_latest.json"
        with open(latest_file, "w") as f:
            json.dump(best_combos, f, indent=2)
            
        return best_combos

if __name__ == "__main__":
    optimizer = AllWeatherWFO()
    optimizer.run()
