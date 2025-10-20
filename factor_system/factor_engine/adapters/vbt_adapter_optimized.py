#!/usr/bin/env python3
"""
VBT适配器优化版 - 从参数爆炸转向智能选择
基于IC/IR动态选择最优参数，而非穷举所有组合
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import vectorbt as vbt
import talib

from factor_system.factor_engine.adapters.smart_indicator_adapter import (
    SmartIndicatorAdapter, IndicatorType, IndicatorConfig
)

logger = logging.getLogger(__name__)


class VBTIndicatorAdapterOptimized:
    """VBT指标适配器优化版 - 智能参数选择"""
    
    def __init__(self, 
                 price_field: str = "close",
                 engine_version: str = "2.0.0",
                 enable_smart_selection: bool = True,
                 ic_threshold: float = 0.02):
        self.price_field = price_field
        self.engine_version = engine_version
        self.enable_smart_selection = enable_smart_selection
        self.ic_threshold = ic_threshold
        self.indicators_computed = 0
        
        # 初始化智能适配器
        if enable_smart_selection:
            self.smart_adapter = SmartIndicatorAdapter(
                price_field=price_field,
                min_ic_threshold=ic_threshold
            )
        
        # 核心参数配置 - 基于市场经验
        self.core_parameters = self._build_core_parameters()
        
    def _build_core_parameters(self) -> Dict[str, List[Dict]]:
        """构建核心参数配置 - 避免参数爆炸"""
        
        return {
            # 趋势类 - 核心周期
            "MA": [
                {"window": 20, "rationale": "短期趋势，月度周期"},
                {"window": 60, "rationale": "中期趋势，季度周期"},
                {"window": 252, "rationale": "长期趋势，年度周期"}
            ],
            "EMA": [
                {"window": 12, "rationale": "快速EMA，2周周期"},
                {"window": 26, "rationale": "慢速EMA，1月周期，MACD标准"}
            ],
            "MACD": [
                {"fast": 12, "slow": 26, "signal": 9, "rationale": "经典MACD参数"}
            ],
            
            # 动量类 - 基于市场微观结构
            "RSI": [
                {"window": 9, "rationale": "短期动量，敏感"},
                {"window": 14, "rationale": "经典RSI，平衡性"}
            ],
            "STOCH": [
                {"k_window": 14, "d_window": 3, "rationale": "经典随机指标，只保留D线"}
            ],
            
            # 波动率类 - 去除数学重复
            "ATR": [
                {"window": 14, "rationale": "经典ATR，2周周期"}
            ],
            "BBANDS": [
                {"window": 20, "alpha": 2.0, "rationale": "经典布林带，月度周期"}
            ],
            
            # 价格位置 - 核心窗口
            "PRICE_POSITION": [
                {"window": 20, "rationale": "月度价格位置"},
                {"window": 60, "rationale": "季度价格位置"}
            ]
        }
    
    def compute_all_indicators(self, df: pd.DataFrame, 
                             target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        计算优化后的指标 - 智能参数选择
        
        Args:
            df: OHLCV数据
            target: 预测目标（可选，用于参数优化）
            
        Returns:
            优化指标DataFrame
        """
        logger.info(f"VBT优化适配器开始计算，输入: {df.shape}")
        
        if self.enable_smart_selection and target is not None:
            # 使用智能适配器进行参数优化
            return self.smart_adapter.compute_optimized_indicators(df, target)
        else:
            # 使用核心参数配置
            return self._compute_with_core_parameters(df)
    
    def _compute_with_core_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用核心参数计算指标"""
        
        # 提取价格数据
        open_price = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        volume = df["volume"].values
        
        factors = {}
        
        # ===== VectorBT指标（智能参数） =====
        factors.update(self._compute_vbt_smart(close, high, low, volume))
        
        # ===== TA-Lib指标（去重版本） =====
        factors.update(self._compute_talib_smart(open_price, high, low, close, volume))
        
        # ===== 自定义指标（核心参数） =====
        factors.update(self._compute_custom_smart(close, high, low, volume))
        
        # 转换为DataFrame
        result_df = pd.DataFrame(factors, index=df.index)
        
        # 添加date列
        if "date" in df.columns:
            result_df["date"] = df["date"].values
            
        self.indicators_computed = len(factors)
        logger.info(f"✅ VBT优化适配器完成: {self.indicators_computed} 个智能指标")
        
        return result_df
    
    def _compute_vbt_smart(self, close: np.ndarray, high: np.ndarray, 
                          low: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """智能VBT指标计算 - 核心参数"""
        factors = {}
        
        # MA - 只选3个核心周期
        for config in self.core_parameters["MA"]:
            window = config["window"]
            try:
                ma = vbt.MA.run(close, window=window)
                factors[f"VBT_MA_{window}"] = ma.ma.values
                logger.debug(f"  MA_{window}: {config['rationale']}")
            except Exception as e:
                logger.warning(f"MA_{window} 计算失败: {e}")
        
        # EMA - 只选2个核心周期  
        for config in self.core_parameters["EMA"]:
            window = config["window"]
            try:
                ema = vbt.MA.run(close, window=window, ewm=True)
                factors[f"VBT_EMA_{window}"] = ema.ma.values
                logger.debug(f"  EMA_{window}: {config['rationale']}")
            except Exception as e:
                logger.warning(f"EMA_{window} 计算失败: {e}")
        
        # MACD - 只保留经典参数
        for config in self.core_parameters["MACD"]:
            fast, slow, signal = config["fast"], config["slow"], config["signal"]
            try:
                macd = vbt.MACD.run(close, fast_window=fast, slow_window=slow, signal_window=signal)
                # 只保留MACD线，去除SIGNAL和HIST重复
                factors[f"VBT_MACD"] = macd.macd.values
                logger.debug(f"  MACD: {config['rationale']}")
            except Exception as e:
                logger.warning(f"MACD 计算失败: {e}")
        
        # RSI - 只选2个核心周期
        for config in self.core_parameters["RSI"]:
            window = config["window"]
            try:
                rsi = vbt.RSI.run(close, window=window)
                factors[f"VBT_RSI_{window}"] = rsi.rsi.values
                logger.debug(f"  RSI_{window}: {config['rationale']}")
            except Exception as e:
                logger.warning(f"RSI_{window} 计算失败: {e}")
        
        # STOCH - 只保留D线，经典参数
        for config in self.core_parameters["STOCH"]:
            k_window, d_window = config["k_window"], config["d_window"]
            try:
                stoch = vbt.STOCH.run(high, low, close, k_window=k_window, d_window=d_window)
                # 只保留D线，K线高度相关
                factors[f"VBT_STOCH_D"] = stoch.percent_d.values
                logger.debug(f"  STOCH_D: {config['rationale']}")
            except Exception as e:
                logger.warning(f"STOCH 计算失败: {e}")
        
        # ATR - 经典参数
        for config in self.core_parameters["ATR"]:
            window = config["window"]
            try:
                atr = vbt.ATR.run(high, low, close, window=window)
                factors[f"VBT_ATR"] = atr.atr.values
                logger.debug(f"  ATR: {config['rationale']}")
            except Exception as e:
                logger.warning(f"ATR 计算失败: {e}")
        
        logger.info(f"VBT智能指标: {len(factors)} 个")
        return factors
    
    def _compute_talib_smart(self, open_price: np.ndarray, high: np.ndarray,
                            low: np.ndarray, close: np.ndarray, 
                            volume: np.ndarray) -> Dict[str, np.ndarray]:
        """智能TA-Lib指标计算 - 避免与VBT重复"""
        factors = {}
        
        # 只添加VBT没有的独特指标
        try:
            # 价格变换（VBT没有）
            factors["TA_TYPPRICE"] = talib.TYPPRICE(high, low, close)
            factors["TA_WCLPRICE"] = talib.WCLPRICE(high, low, close)
            
            # 周期指标（VBT没有）
            factors["TA_HT_DCPERIOD"] = talib.HT_DCPERIOD(close)
            factors["TA_HT_TRENDMODE"] = talib.HT_TRENDMODE(close)
            
            # 成交量指标（VBT没有详细版本）
            factors["TA_AD"] = talib.AD(high, low, close, volume)
            factors["TA_ADOSC"] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
            
            # 形态识别（选择性添加）
            factors["TA_CDL_DOJI"] = talib.CDLDOJI(open_price, high, low, close)
            
        except Exception as e:
            logger.warning(f"TA-Lib智能指标计算失败: {e}")
        
        logger.info(f"TA-Lib独特指标: {len(factors)} 个")
        return factors
    
    def _compute_custom_smart(self, close: np.ndarray, high: np.ndarray,
                             low: np.ndarray, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """智能自定义指标计算"""
        factors = {}
        
        # 收益率 - 核心周期
        for period in [1, 5, 20]:
            ret = np.roll(close, -period) / close - 1
            factors[f"RETURN_{period}d"] = ret
        
        # 波动率 - 核心窗口
        for window in [20, 60]:
            log_returns = pd.Series(close).pct_change()
            vol = log_returns.rolling(window).std()
            factors[f"VOLATILITY_{window}d"] = vol.values
        
        # 价格位置 - 核心窗口
        for window in [20, 60]:
            rolling_high = pd.Series(high).rolling(window).max()
            rolling_low = pd.Series(low).rolling(window).min()
            factors[f"PRICE_POSITION_{window}d"] = (close - rolling_low) / (
                rolling_high - rolling_low + 1e-10
            )
        
        # 成交量比率 - 核心窗口
        for window in [5, 20]:
            vol_ma = pd.Series(volume).rolling(window).mean()
            factors[f"VOLUME_RATIO_{window}d"] = volume / (vol_ma + 1e-10)
        
        logger.info(f"自定义智能指标: {len(factors)} 个")
        return factors
    
    def get_indicator_stats(self) -> Dict[str, any]:
        """获取指标统计信息"""
        return {
            "total_indicators": self.indicators_computed,
            "smart_selection_enabled": self.enable_smart_selection,
            "optimization_threshold": self.ic_threshold,
            "core_parameters_used": len(self.core_parameters),
            "reduction_ratio": f"{((370 - self.indicators_computed) / 370 * 100):.1f}%"
        }