#!/usr/bin/env python3
"""
全天候策略引擎 | All-Weather Strategy Engine
================================================================================
实现分池选股 + 波动率体制切换的完整全天候策略

核心架构：
1. 7个子池独立运行WFO/VEC选股
2. 波动率体制动态调整各池权重
3. 每个池使用类别专用因子

资金分配（基准）：
- EQUITY_BROAD: 20%
- EQUITY_GROWTH: 15%
- EQUITY_CYCLICAL: 10%
- EQUITY_DEFENSIVE: 5%
- BOND: 20%
- COMMODITY: 15%
- QDII: 15%

波动率体制切换：
- 高波动（VIX等效 > 25）：增配BOND/COMMODITY，减配EQUITY
- 低波动（VIX等效 < 15）：增配EQUITY，减配BOND

================================================================================
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from numba import njit
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.category_factors import CategoryFactorManager, BondFactors, CommodityFactors, QDIIFactors
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule, ensure_price_views


# =============================================================================
# 配置与常量
# =============================================================================

FREQ = 8  # 调仓频率（交易日）
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252

# 风控参数
STOP_LOSS_PCT = 0.12
DD_LIMIT_SOFT = 0.15
DD_LIMIT_HARD = 0.25


@dataclass
class PoolConfig:
    """子池配置"""
    name: str
    symbols: List[str]
    target_weight: float
    pos_size: int
    factor_type: str  # 'equity', 'bond', 'commodity', 'qdii'
    

# =============================================================================
# 波动率体制切换器
# =============================================================================

class TrendRegimeSwitch:
    """
    趋势体制切换器 (Based on LightTimingModule)
    
    使用 MA200 + 动量 + 黄金走势判断市场环境：
    - 牛市 (Bull): 100% 权益
    - 熊市 (Bear): 30% 权益 / 70% 避险
    """
    
    def __init__(self):
        self.timing_module = LightTimingModule(extreme_threshold=-0.4, extreme_position=0.3)
    
    def compute_equity_ratio(self, close_df: pd.DataFrame) -> pd.Series:
        """
        计算目标权益仓位比例
        
        Returns:
            pd.Series: 1.0 (Bull) or 0.3 (Bear)
        """
        # 尝试使用 510300 (沪深300) 或 510050 (上证50) 作为市场基准
        market_symbol = '510300.SH'
        if market_symbol not in close_df.columns:
            market_symbol = '510300' # 尝试不带后缀
            
        gold_symbol = '518880.SH'
        if gold_symbol not in close_df.columns:
            gold_symbol = '518880'
            
        return self.timing_module.compute_position_ratios(
            close_df, 
            market_symbol=market_symbol, 
            gold_symbol=gold_symbol
        )
    
    def adjust_weights(
        self,
        base_weights: Dict[str, float],
        equity_ratio: float
    ) -> Dict[str, float]:
        """
        根据权益比例调整权重 (优化版)
        
        策略：
        1. A股权益 (Equity): 受 equity_ratio 直接控制 (Bull: High, Bear: Low)
        2. QDII: 作为替代资产，在 A股 Bear 时增配，Bull 时保持标配
        3. 避险 (Bond/Comm): 在 A股 Bear 时作为主要避风港
        
        Args:
            base_weights: 基准权重
            equity_ratio: 目标权益比例 (0.3 ~ 1.0)
            
        Returns:
            Dict: 调整后的权重
        """
        # 分类
        equity_pools = ['EQUITY_BROAD', 'EQUITY_GROWTH', 'EQUITY_CYCLICAL', 'EQUITY_DEFENSIVE']
        qdii_pools = ['QDII']
        safe_pools = ['BOND', 'COMMODITY']
        
        adjusted = {}
        
        # 1. A股权益处理
        # Bull (Ratio=1.0): 1.6x (80% Total) - Max Aggression
        # Bear (Ratio=0.3): 0.6x (30% Total) - Defensive but participating
        if equity_ratio > 0.5:
            equity_scale = 1.6
        else:
            equity_scale = 0.6
        
        for p in equity_pools:
            if p in base_weights:
                adjusted[p] = base_weights[p] * equity_scale
                
        # 2. QDII 处理
        # Bull (Ratio=1.0): 1.33x (20% Total)
        # Bear (Ratio=0.3): 1.33x (20% Total) - Always hold US Tech
        qdii_scale = 1.33
        
        for p in qdii_pools:
            if p in base_weights:
                adjusted[p] = base_weights[p] * qdii_scale
                
        # 3. 避险资产处理
        # Bull (Ratio=1.0): 0.0x (No Bonds in Bull Market)
        # Bear (Ratio=0.3): Fill the rest (~50%)
        
        # 先计算目前已分配的权重
        current_total = sum(adjusted.values())
        remaining = 1.0 - current_total
        
        # 计算避险资产的基准总权重
        safe_base_total = sum(base_weights.get(p, 0) for p in safe_pools)
        
        # 3. 避险资产处理
        # Bull (equity_ratio > 0.5): 剩余仓位分配给避险（通常很少）
        # Bear (equity_ratio <= 0.5): 剩余仓位分配给避险（约50%）
        
        if safe_base_total > 0:
            # 确保 remaining 非负
            remaining = max(0.0, remaining)
            
            if remaining < 0.01:  # 几乎没有剩余空间（牛市情况）
                for p in safe_pools:
                    adjusted[p] = 0.0
            else:
                # 按基准权重比例分配剩余空间
                safe_scale = remaining / safe_base_total
                for p in safe_pools:
                    if p in base_weights:
                        adjusted[p] = base_weights[p] * safe_scale
        else:
            for p in safe_pools:
                adjusted[p] = 0.0
                
        # 4. 最终归一化 (防止浮点误差)
        total = sum(adjusted.values())
        if total > 0:
            for k in adjusted:
                adjusted[k] /= total
                
        return adjusted


# =============================================================================
# 分池选股引擎
# =============================================================================

class PoolSelector:
    """
    单个池的选股器
    
    支持：
    - 权益类：使用 PreciseFactorLibrary 的 18 个因子
    - 债券类：使用 BondFactors
    - 商品类：使用 CommodityFactors
    - QDII类：使用 QDIIFactors
    """
    
    def __init__(
        self,
        pool_config: PoolConfig,
        factors_3d: np.ndarray,
        factor_names: List[str],
        close_prices: np.ndarray,
        etf_code_to_idx: Dict[str, int],
    ):
        self.config = pool_config
        self.factors_3d = factors_3d
        self.factor_names = factor_names
        self.close_prices = close_prices
        self.etf_code_to_idx = etf_code_to_idx
        
        # 获取池内ETF的索引
        self.pool_indices = []
        for sym in pool_config.symbols:
            if sym in etf_code_to_idx:
                self.pool_indices.append(etf_code_to_idx[sym])
            else:
                logger.warning(f"池 {pool_config.name} 中的 {sym} 不在数据中")
        
        self.pool_indices = np.array(self.pool_indices, dtype=np.int64)
    
    def select_top_n(
        self,
        t: int,
        factor_indices: np.ndarray,
        n: int = None
    ) -> List[int]:
        """
        在时刻 t 选出池内得分最高的 n 个ETF
        
        Args:
            t: 当前时间索引
            factor_indices: 使用的因子索引
            n: 选择数量（默认使用池配置）
            
        Returns:
            List[int]: 选中ETF的全局索引
        """
        if n is None:
            n = self.config.pos_size
        
        if len(self.pool_indices) == 0:
            return []
        
        # 计算池内每个ETF的综合得分
        scores = []
        for idx in self.pool_indices:
            score = 0.0
            valid = False
            for f_idx in factor_indices:
                val = self.factors_3d[t - 1, idx, f_idx]
                if not np.isnan(val):
                    score += val
                    valid = True
            
            if valid:
                scores.append((idx, score))
            else:
                scores.append((idx, -np.inf))
        
        # 排序选出 Top N
        scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        for idx, score in scores[:n]:
            if score > -np.inf:
                selected.append(idx)
        
        return selected


# =============================================================================
# Numba 加速的回测核心
# =============================================================================

@njit(cache=True)
def allweather_backtest_kernel(
    close_prices: np.ndarray,        # [T, N]
    open_prices: np.ndarray,         # [T, N]
    low_prices: np.ndarray,          # [T, N]
    high_prices: np.ndarray,         # [T, N]
    pool_selections: np.ndarray,     # [num_rebal, num_pools, max_pos] 每个调仓日每个池选中的ETF索引
    pool_weights: np.ndarray,        # [num_rebal, num_pools] 每个调仓日每个池的权重
    pool_pos_sizes: np.ndarray,      # [num_pools] 每个池的持仓数
    rebalance_schedule: np.ndarray,  # [num_rebal] 调仓日索引
    initial_capital: float,
    commission_rate: float,
    stop_loss_pct: float,
    take_profit_pct: float = 0.15,
) -> Tuple[float, float, float, int, float, np.ndarray]:
    """
    全天候策略回测核心
    
    Returns:
        total_return, win_rate, profit_factor, num_trades, max_drawdown, daily_values
    """
    T, N = close_prices.shape
    num_rebal = len(rebalance_schedule)
    num_pools = pool_weights.shape[1]
    
    cash = initial_capital
    holdings = np.full(N, -1.0)  # -1 表示不持仓
    entry_prices = np.zeros(N)
    
    peak_value = initial_capital
    max_drawdown = 0.0
    
    wins = 0
    losses = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0
    
    # 记录每日净值
    daily_values = np.zeros(T)
    daily_values[0] = initial_capital
    
    for i in range(num_rebal):
        t = rebalance_schedule[i]
        
        if i < num_rebal - 1:
            next_t = rebalance_schedule[i + 1]
        else:
            next_t = T
        
        if t >= T:
            break
        
        # 1. 计算当前组合价值
        portfolio_value = cash
        for n in range(N):
            if holdings[n] > 0.0:
                portfolio_value += holdings[n] * close_prices[t, n]
        
        # 2. 更新峰值和回撤
        if portfolio_value > peak_value:
            peak_value = portfolio_value
        
        dd = 1.0 - portfolio_value / peak_value
        if dd > max_drawdown:
            max_drawdown = dd
        
        # 3. 确定本期目标持仓
        target_set = np.zeros(N, dtype=np.bool_)
        target_value = np.zeros(N)
        
        for p in range(num_pools):
            pool_weight = pool_weights[i, p]
            pool_pos_size = pool_pos_sizes[p]
            pool_capital = portfolio_value * pool_weight
            
            if pool_capital < 1000 or pool_pos_size == 0:
                continue
            
            per_pos_capital = pool_capital / pool_pos_size
            
            for j in range(pool_pos_size):
                idx = pool_selections[i, p, j]
                if idx >= 0 and idx < N:
                    target_set[idx] = True
                    target_value[idx] = per_pos_capital
        
        # 4. 卖出逻辑：卖出不在目标中的持仓
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
        
        # 5. 买入逻辑：买入新目标
        for n in range(N):
            if target_set[n] and holdings[n] < 0.0:
                price = close_prices[t, n]
                if np.isnan(price) or price <= 0:
                    continue
                
                target_cost = target_value[n] * (1.0 + commission_rate)
                if target_cost > cash:
                    target_cost = cash
                
                if target_cost > 0:
                    shares = target_cost / (price * (1.0 + commission_rate))
                    actual_cost = shares * price * (1.0 + commission_rate)
                    
                    if cash >= actual_cost - 1e-5:
                        cash -= actual_cost
                        holdings[n] = shares
                        entry_prices[n] = price
        
        # 6. 止损止盈检查
        check_start = t + 1
        check_end = min(next_t, T)
        
        for n in range(N):
            if holdings[n] > 0.0:
                entry = entry_prices[n]
                stop_price = entry * (1.0 - stop_loss_pct)
                tp_price = entry * (1.0 + take_profit_pct)
                
                for day in range(check_start, check_end):
                    # 止损
                    if low_prices[day, n] < stop_price:
                        # 假设以止损价成交
                        exec_price = stop_price
                        if open_prices[day, n] < stop_price:
                            exec_price = open_prices[day, n]
                            
                        proceeds = holdings[n] * exec_price * (1.0 - commission_rate)
                        cash += proceeds
                        
                        pnl = (exec_price - entry) / entry
                        losses += 1
                        total_loss_pnl += abs(pnl)
                        
                        holdings[n] = -1.0
                        entry_prices[n] = 0.0
                        break
                    
                    # 止盈
                    if high_prices[day, n] > tp_price:
                        # 假设以止盈价成交
                        exec_price = tp_price
                        if open_prices[day, n] > tp_price:
                            exec_price = open_prices[day, n]
                            
                        proceeds = holdings[n] * exec_price * (1.0 - commission_rate)
                        cash += proceeds
                        
                        pnl = (exec_price - entry) / entry
                        wins += 1
                        total_win_pnl += pnl
                        
                        holdings[n] = -1.0
                        entry_prices[n] = 0.0
                        break
        
        # 7. 记录每日净值
        for day in range(t, min(next_t, T)):
            day_value = cash
            for n in range(N):
                if holdings[n] > 0.0:
                    day_value += holdings[n] * close_prices[day, n]
            daily_values[day] = day_value
    
    # 最终清算
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
    
    daily_values[T - 1] = final_value
    
    # 计算指标
    num_trades = wins + losses
    total_return = (final_value - initial_capital) / initial_capital
    win_rate = wins / num_trades if num_trades > 0 else 0.0
    
    if losses > 0 and wins > 0:
        avg_win = total_win_pnl / wins
        avg_loss = total_loss_pnl / losses
        profit_factor = (avg_win * wins) / (avg_loss * losses)
    else:
        profit_factor = 0.0
    
    return total_return, win_rate, profit_factor, num_trades, max_drawdown, daily_values


# =============================================================================
# 主引擎
# =============================================================================

class AllWeatherEngine:
    """
    全天候策略引擎
    
    整合：
    1. 数据加载
    2. 因子计算（分类因子）
    3. 分池选股
    4. 波动率体制切换
    5. VEC回测
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or str(ROOT / "configs/etf_pools.yaml")
        self.pool_config = self._load_pool_config()
        
        # 初始化组件
        self.factor_lib = PreciseFactorLibrary()
        self.category_factor_mgr = CategoryFactorManager()
        self.trend_regime_switch = TrendRegimeSwitch()
        self.processor = CrossSectionProcessor(verbose=False)
        self.timing_module = LightTimingModule()
        
        # 数据存储
        self.ohlcv = None
        self.factors_3d = None
        self.factor_names = None
        self.etf_codes = None
        self.dates = None
    
    def _load_pool_config(self) -> Dict:
        """加载池配置"""
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def get_pool_configs(self) -> List[PoolConfig]:
        """
        获取所有池配置
        
        重要：排除避险资产池（BOND, COMMODITY），它们不参与因子选股
        只返回权益池和QDII池
        """
        pools = self.pool_config.get("pools", {})
        capital_constraints = self.pool_config.get("capital_constraints", {})
        
        # 池类型映射
        pool_type_map = {
            "EQUITY_BROAD": "equity",
            "EQUITY_GROWTH": "equity",
            "EQUITY_CYCLICAL": "equity",
            "EQUITY_DEFENSIVE": "equity",
            "BOND": "bond",
            "COMMODITY": "commodity",
            "QDII": "qdii",
        }
        
        configs = []
        for pool_name, pool_data in pools.items():
            if pool_name == "A_SHARE_LIVE":  # 跳过实盘精选池
                continue
            
            symbols = pool_data.get("symbols", [])
            target_weight = capital_constraints.get(pool_name, {}).get("target_capital", 0.1)
            
            # 根据池类型和资产数量确定持仓数
            factor_type = pool_type_map.get(pool_name, "equity")
            
            if factor_type == "bond":
                pos_size = min(2, len(symbols))  # 债券池最多2个
            elif factor_type == "commodity":
                pos_size = min(2, len(symbols))  # 商品池最多2个
            else:
                pos_size = min(3, len(symbols))  # 权益和QDII最多3个
            
            configs.append(PoolConfig(
                name=pool_name,
                symbols=symbols,
                target_weight=target_weight,
                pos_size=pos_size,
                factor_type=factor_type,
            ))
        
        return configs
    
    def load_data(self, data_config: Dict):
        """加载数据"""
        loader = DataLoader(
            data_dir=data_config.get("data_dir"),
            cache_dir=data_config.get("cache_dir")
        )
        
        self.ohlcv = loader.load_ohlcv(
            etf_codes=data_config["symbols"],
            start_date=data_config["start_date"],
            end_date=data_config["end_date"]
        )
        
        logger.info(f"✅ 数据加载完成: {len(data_config['symbols'])} 个ETF")
    
    def compute_factors(self):
        """计算所有因子（通用 + 分类）"""
        # 1. 计算通用因子（18个）- 对所有ETF计算
        logger.info("正在计算通用因子...")
        raw_factors_df = self.factor_lib.compute_all_factors(self.ohlcv)
        
        # 2. 计算分类因子 - 对特定池计算
        logger.info("正在计算分类因子...")
        
        # 获取各类别的符号列表
        bond_symbols = [s for p in self.get_pool_configs() if p.factor_type == 'bond' for s in p.symbols]
        comm_symbols = [s for p in self.get_pool_configs() if p.factor_type == 'commodity' for s in p.symbols]
        qdii_symbols = [s for p in self.get_pool_configs() if p.factor_type == 'qdii' for s in p.symbols]
        
        # 市场基准（用于商品避险评分等）
        market_proxy = "510300.SH" if "510300.SH" in self.ohlcv["close"].columns else self.ohlcv["close"].columns[0]
        
        # 计算各类别因子
        bond_factors = self.category_factor_mgr.compute_factors_for_pool("BOND", self.ohlcv, bond_symbols)
        comm_factors = self.category_factor_mgr.compute_factors_for_pool("COMMODITY", self.ohlcv, comm_symbols, market_proxy)
        qdii_factors = self.category_factor_mgr.compute_factors_for_pool("QDII", self.ohlcv, qdii_symbols)
        
        # 3. 合并所有因子
        # raw_factors_df 是 (T, N*F_common)
        # category_factors 是 (T, N_subset*F_cat)
        # 我们需要将它们合并到一个大的 DataFrame 中
        
        all_dfs = [raw_factors_df]
        if not bond_factors.empty: all_dfs.append(bond_factors)
        if not comm_factors.empty: all_dfs.append(comm_factors)
        if not qdii_factors.empty: all_dfs.append(qdii_factors)
        
        # 合并 (按列索引自动对齐，缺失值为NaN)
        combined_df = pd.concat(all_dfs, axis=1)
        
        # 拆分为字典格式供处理器使用
        factor_names_list = combined_df.columns.get_level_values(0).unique().tolist()
        raw_factors = {fname: combined_df[fname] for fname in factor_names_list}
        
        # 4. 标准化
        # 注意：分类因子只在特定资产上有值，标准化时会自动忽略NaN
        std_factors = self.processor.process_all_factors(raw_factors)
        
        # 5. 构建 3D 数组
        self.factor_names = sorted(std_factors.keys())
        first_factor = std_factors[self.factor_names[0]]
        self.dates = first_factor.index
        
        # 关键修复：确保所有因子DataFrame具有相同的列（ETF代码）
        # 使用所有ETF代码的并集
        all_etf_codes = sorted(self.ohlcv["close"].columns.tolist())
        self.etf_codes = all_etf_codes
        
        aligned_factors = []
        for f in self.factor_names:
            df = std_factors[f]
            # Reindex columns to include all ETFs, filling with NaN
            df_aligned = df.reindex(columns=all_etf_codes)
            aligned_factors.append(df_aligned.values)
            
        self.factors_3d = np.stack(aligned_factors, axis=-1)
        
        logger.info(f"✅ 因子计算完成: {len(self.factor_names)} 个因子 (通用+分类), {len(self.etf_codes)} 个ETF")
    
    def get_equity_pool_symbols(self) -> List[str]:
        """
        获取所有权益池的ETF列表
        
        重要：排除 BOND 和 COMMODITY 池的资产，避免双重计算
        """
        pools = self.pool_config.get("pools", {})
        equity_symbols = []
        
        for pool_name, pool_data in pools.items():
            if pool_name.startswith("EQUITY_") or pool_name == "QDII":
                equity_symbols.extend(pool_data.get("symbols", []))
        
        return list(set(equity_symbols))
    
    def get_safe_asset_symbols(self) -> List[str]:
        """获取避险资产列表（BOND + COMMODITY）"""
        pools = self.pool_config.get("pools", {})
        safe_symbols = []
        
        for pool_name in ["BOND", "COMMODITY"]:
            if pool_name in pools:
                safe_symbols.extend(pools[pool_name].get("symbols", []))
        
        return list(set(safe_symbols))
    
    def run_backtest(
        self,
        factor_map: Dict[str, List[str]] = None,
        use_regime_switch: bool = True,
    ) -> Dict:
        """
        运行单次全天候回测
        
        Args:
            factor_map: 池名称到因子列表的映射
                {'EQUITY_BROAD': [...], 'BOND': [...], ...}
            use_regime_switch: 是否使用波动率体制切换
            
        Returns:
            Dict: 回测结果
        """
        T = self.factors_3d.shape[0]
        N = len(self.etf_codes)
        
        # 默认因子映射 (按类型)
        default_type_map = {
            "equity": ["ADX_14D", "PRICE_POSITION_20D", "SHARPE_RATIO_20D", "SLOPE_20D"],
            "bond": ["YIELD_MOMENTUM_20D", "DURATION_PROXY_60D", "BOND_MOMENTUM_SCORE"],
            "commodity": ["USD_INVERSE_MOM_20D", "COMMODITY_TREND_20D", "GOLD_SAFE_HAVEN_SCORE"],
            "qdii": ["QDII_MOMENTUM_20D", "FX_ADJUSTED_MOM", "QDII_VOL_RATIO"]
        }
        
        # 如果未提供 factor_map，尝试加载最新的 WFO 结果
        if factor_map is None:
            wfo_result_path = ROOT / "results/all_pools_best_config_latest.json"
            if wfo_result_path.exists():
                logger.info(f"📂 加载 WFO 优化结果: {wfo_result_path}")
                with open(wfo_result_path) as f:
                    data = json.load(f)
                    factor_map = data.get("pool_factors", {})
            else:
                # 尝试加载旧版结果作为回退
                legacy_path = ROOT / "results/allweather_best_combos_latest.json"
                if legacy_path.exists():
                    logger.info(f"📂 加载旧版 WFO 结果: {legacy_path}")
                    with open(legacy_path) as f:
                        factor_map = json.load(f)
                else:
                    logger.warning("⚠️ 未找到 WFO 结果，使用默认因子")
                    factor_map = {}

        # 补全缺失的池 (使用默认类型因子)
        pool_configs = self.get_pool_configs()
        for pc in pool_configs:
            if pc.name not in factor_map:
                # 如果 WFO 没结果 (如 Commodity)，回退到默认
                defaults = default_type_map.get(pc.factor_type, default_type_map['equity'])
                factor_map[pc.name] = defaults
                logger.info(f"   池 {pc.name} 使用默认因子: {defaults}")
            else:
                logger.info(f"   池 {pc.name} 使用 WFO 因子: {factor_map[pc.name]}")
        
        # 准备价格数据
        close_prices = self.ohlcv["close"][self.etf_codes].ffill().bfill().values
        open_prices = self.ohlcv["open"][self.etf_codes].ffill().bfill().values
        low_prices = self.ohlcv["low"][self.etf_codes].ffill().bfill().values
        high_prices = self.ohlcv["high"][self.etf_codes].ffill().bfill().values
        
        # 构建 ETF 代码到索引的映射
        etf_code_to_idx = {code: i for i, code in enumerate(self.etf_codes)}
        
        # 获取因子索引映射
        factor_name_to_idx = {name: idx for idx, name in enumerate(self.factor_names)}
        
        # 预计算每个池的因子索引数组
        pool_factor_indices = {}
        for pc in pool_configs:
            f_list = factor_map[pc.name]
            indices = []
            for fname in f_list:
                if fname in factor_name_to_idx:
                    indices.append(factor_name_to_idx[fname])
                else:
                    logger.warning(f"因子 {fname} 未找到，将被忽略")
            pool_factor_indices[pc.name] = np.array(indices, dtype=np.int64)
        
        # 计算趋势体制 (LightTiming)
        market_proxy = "510300.SH" if "510300.SH" in self.etf_codes else self.etf_codes[0]
        equity_ratio_series = self.trend_regime_switch.compute_equity_ratio(self.ohlcv["close"])
        
        # 生成调仓日程
        rebalance_schedule = generate_rebalance_schedule(
            total_periods=T,
            lookback_window=LOOKBACK,
            freq=FREQ
        )
        num_rebal = len(rebalance_schedule)
        
        # 构建池选择器
        pool_selectors = []
        for pc in pool_configs:
            selector = PoolSelector(
                pool_config=pc,
                factors_3d=self.factors_3d,
                factor_names=self.factor_names,
                close_prices=close_prices,
                etf_code_to_idx=etf_code_to_idx,
            )
            pool_selectors.append(selector)
        
        # 预计算每个调仓日的选股结果和权重
        num_pools = len(pool_configs)
        max_pos = max(pc.pos_size for pc in pool_configs)
        pool_selections = np.full((num_rebal, num_pools, max_pos), -1, dtype=np.int64)
        pool_weights = np.zeros((num_rebal, num_pools))
        pool_pos_sizes = np.array([pc.pos_size for pc in pool_configs], dtype=np.int64)
        
        # 基准权重
        base_weights = {pc.name: pc.target_weight for pc in pool_configs}
        
        for i, t in enumerate(rebalance_schedule):
            if t >= T:
                continue
            
            # 1. 获取当前权益比例
            if t > 0:
                equity_ratio = equity_ratio_series.iloc[t-1] if use_regime_switch else 1.0
            else:
                equity_ratio = 1.0
            
            # 2. 调整权重
            adjusted_weights = self.trend_regime_switch.adjust_weights(base_weights, equity_ratio)
            
            # 3. 每个池选股
            for p, (pc, selector) in enumerate(zip(pool_configs, pool_selectors)):
                pool_weights[i, p] = adjusted_weights.get(pc.name, 0.0)
                
                # 获取该池对应的因子索引
                f_indices = pool_factor_indices[pc.name]
                
                # 选出池内 Top N
                selected = selector.select_top_n(t, f_indices)
                for j, idx in enumerate(selected):
                    if j < max_pos:
                        pool_selections[i, p, j] = idx
        
        # 运行回测核心
        total_return, win_rate, profit_factor, num_trades, max_drawdown, daily_values = allweather_backtest_kernel(
            close_prices=close_prices,
            open_prices=open_prices,
            low_prices=low_prices,
            high_prices=high_prices,
            pool_selections=pool_selections,
            pool_weights=pool_weights,
            pool_pos_sizes=pool_pos_sizes,
            rebalance_schedule=rebalance_schedule,
            initial_capital=INITIAL_CAPITAL,
            commission_rate=COMMISSION_RATE,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=10.0, # 禁用止盈 (让利润奔跑)
        )
        
        # 计算夏普比率（修复除零问题）
        daily_values_valid = daily_values[daily_values > 0]
        if len(daily_values_valid) > 1:
            daily_returns = np.diff(daily_values_valid) / daily_values_valid[:-1]
            daily_returns = daily_returns[np.isfinite(daily_returns)]
            if len(daily_returns) > 0 and np.std(daily_returns) > 1e-10:
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe = 0.0
        
        # 记录使用的因子描述
        combo_desc = " + ".join([f"{k}:{','.join(v)}" for k, v in factor_map.items()])
        
        return {
            "combo": combo_desc,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": num_trades,
            "daily_values": daily_values,
        }


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 80)
    print("🌤️ 全天候策略引擎 | All-Weather Strategy Engine")
    print("=" * 80)
    
    # 1. 加载配置
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. 初始化引擎
    engine = AllWeatherEngine()
    
    # 3. 加载数据
    engine.load_data(config["data"])
    
    # 4. 计算因子
    engine.compute_factors()
    
    # 5. 显示池配置
    print("\n📊 子池配置:")
    print("-" * 60)
    for pc in engine.get_pool_configs():
        print(f"  {pc.name:20} | 权重: {pc.target_weight*100:.0f}% | 持仓: {pc.pos_size} | 类型: {pc.factor_type}")
    
    # 6. 运行示例回测
    print("\n🔬 运行示例回测 (使用 WFO 优化因子)...")
    
    # 运行有/无体制切换的对比 (传入 None 以触发 WFO 结果加载)
    result_with_switch = engine.run_backtest(None, use_regime_switch=True)
    result_no_switch = engine.run_backtest(None, use_regime_switch=False)
    
    print("\n📈 回测结果:")
    print("-" * 60)
    print(f"因子组合: WFO Optimized + Trend Following (LightTiming)")
    print(f"\n{'指标':<20} {'有趋势择时':<15} {'无趋势择时':<15}")
    print("-" * 50)
    print(f"{'总收益':<20} {result_with_switch['total_return']*100:>12.1f}% {result_no_switch['total_return']*100:>12.1f}%")
    print(f"{'最大回撤':<20} {result_with_switch['max_drawdown']*100:>12.1f}% {result_no_switch['max_drawdown']*100:>12.1f}%")
    print(f"{'夏普比率':<20} {result_with_switch['sharpe']:>12.2f} {result_no_switch['sharpe']:>12.2f}")
    print(f"{'胜率':<20} {result_with_switch['win_rate']*100:>12.1f}% {result_no_switch['win_rate']*100:>12.1f}%")
    print(f"{'交易次数':<20} {result_with_switch['num_trades']:>12} {result_no_switch['num_trades']:>12}")
    
    # 7. 保存结果
    output_dir = ROOT / "results" / f"allweather_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每日净值
    daily_df = pd.DataFrame({
        "date": engine.dates,
        "value_with_timing": result_with_switch["daily_values"],
        "value_no_timing": result_no_switch["daily_values"],
    })
    daily_df.to_csv(output_dir / "daily_values.csv", index=False)
    
    print(f"\n✅ 结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()
