"""
FactorEngine适配器 - 为hk_midfreq提供统一因子计算

确保回测阶段使用与因子生成阶段完全相同的计算逻辑
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from factor_system.factor_engine.batch_calculator import BatchFactorCalculator

logger = logging.getLogger(__name__)


class BacktestFactorAdapter:
    """
    回测因子适配器
    
    为回测系统提供统一的因子计算接口，
    确保与因子生成阶段使用相同的FactorEngine
    """
    
    _instance: Optional["BacktestFactorAdapter"] = None
    _calculator: Optional[BatchFactorCalculator] = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化适配器"""
        if self._calculator is None:
            self._initialize_calculator()
    
    def _initialize_calculator(self):
        """初始化批量计算器（使用统一API）"""
        try:
            from hk_midfreq.config import PathConfig
            from factor_system.factor_engine import api
            
            # 创建PathConfig实例
            path_config = PathConfig()
            
            # 直接使用统一API获取引擎（单例模式）
            self._engine = api.get_engine(
                raw_data_dir=path_config.raw_data_dir,
                registry_file=path_config.factor_registry_path,
            )
            
            # 向后兼容：保留_calculator引用
            # 但实际计算直接用api模块
            self._calculator = None
            
            logger.info(f"BacktestFactorAdapter初始化完成（使用统一API），{len(self._engine.registry.factors)}个因子可用")
        except Exception as e:
            logger.error(f"初始化BacktestFactorAdapter失败: {e}")
            raise
    
    def calculate_factor(
        self,
        factor_name: str,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        计算单个因子（使用统一API）
        
        Args:
            factor_name: 因子名称（如 "RSI", "STOCH", "WILLR"）
            symbol: 股票代码
            timeframe: 时间框架
            data: OHLCV数据
        
        Returns:
            因子值Series
        """
        from factor_system.factor_engine import api
        
        if self._engine is None:
            raise RuntimeError("FactorEngine未初始化")
        
        # 提取因子ID（去除后缀数字）
        factor_id = self._extract_factor_id(factor_name)
        
        # 检查因子是否可用
        available_factors = list(self._engine.registry.factors.keys())
        if factor_id not in available_factors:
            raise api.UnknownFactorError(factor_id, available_factors)
        
        # 计算因子
        try:
            start_date = data.index.min()
            end_date = data.index.max()
            
            # 使用统一API计算
            factors_df = api.calculate_factors(
                factor_ids=[factor_id],
                symbols=[symbol],
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
            )
            
            # 提取单symbol结果
            if isinstance(factors_df.index, pd.MultiIndex):
                factors_df = factors_df.xs(symbol, level='symbol')
            
            if factor_id in factors_df.columns:
                return factors_df[factor_id]
            else:
                logger.warning(f"因子{factor_id}未在结果中找到")
                return pd.Series(index=data.index, dtype=float)
        
        except api.UnknownFactorError:
            raise  # 重新抛出，让上层处理
        except Exception as e:
            logger.error(f"计算因子{factor_name}失败: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def _extract_factor_id(self, factor_name: str) -> str:
        """
        从因子名称提取因子ID
        
        例如: "RSI_14" -> "RSI", "STOCH_14_3" -> "STOCH"
        """
        # 常见模式
        if "_" in factor_name:
            parts = factor_name.split("_")
            # 如果第一部分是字母，返回第一部分
            if parts[0].isalpha():
                return parts[0]
            # 否则尝试找到第一个纯字母部分
            for part in parts:
                if part.isalpha():
                    return part
        
        # 没有下划线，直接返回
        return factor_name
    
    def get_engine(self):
        """获取底层引擎实例"""
        if self._engine is None:
            self._initialize_calculator()
        return self._engine
    
    def get_calculator(self):
        """
        获取底层计算器实例（已废弃，使用get_engine）
        
        ⚠️ 此方法仅为向后兼容保留，推荐直接使用 factor_engine.api
        """
        logger.warning("get_calculator已废弃，推荐使用 factor_engine.api 模块")
        return self.get_engine()


# 全局单例
_global_adapter: Optional[BacktestFactorAdapter] = None


def get_factor_adapter() -> BacktestFactorAdapter:
    """获取全局因子适配器"""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = BacktestFactorAdapter()
    return _global_adapter


def calculate_factor_for_backtest(
    factor_name: str,
    symbol: str,
    timeframe: str,
    data: pd.DataFrame,
) -> pd.Series:
    """
    便捷函数：为回测计算因子
    
    Args:
        factor_name: 因子名称
        symbol: 股票代码
        timeframe: 时间框架
        data: OHLCV数据
    
    Returns:
        因子值Series
    """
    adapter = get_factor_adapter()
    return adapter.calculate_factor(factor_name, symbol, timeframe, data)

