"""
严格时序安全数据提供者包装器

确保所有资金流数据执行T+1滞后，防止未来信息泄露
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from factor_system.factor_engine.providers.base import DataProvider

logger = logging.getLogger(__name__)


class StrictTemporalProvider(DataProvider):
    """
    严格时序安全包装器
    
    职责：
    - 包装任何数据提供者
    - 自动对资金流字段执行shift(1)
    - 标记temporal_safe=True
    - 确保无未来信息泄露
    """

    def __init__(self, wrapped_provider: DataProvider, money_flow_columns: Optional[List[str]] = None):
        """
        初始化严格时序安全包装器
        
        Args:
            wrapped_provider: 被包装的数据提供者
            money_flow_columns: 需要T+1处理的资金流列名（None则自动检测）
        """
        self.wrapped_provider = wrapped_provider
        self.money_flow_columns = money_flow_columns or self._default_money_flow_columns()
        
        logger.info(f"初始化StrictTemporalProvider，监控{len(self.money_flow_columns)}个资金流字段")

    def _default_money_flow_columns(self) -> List[str]:
        """默认的资金流字段列表"""
        return [
            # 原始字段
            "buy_small_amount", "sell_small_amount",
            "buy_medium_amount", "sell_medium_amount",
            "buy_large_amount", "sell_large_amount",
            "buy_super_large_amount", "sell_super_large_amount",
            "turnover_amount", "net_mf_amount",
            # 衍生字段
            "main_net", "retail_net", "total_net",
            # 标准化字段
            "main_net_winsorized", "main_net_zscore",
            "retail_net_winsorized", "retail_net_zscore",
            "total_net_winsorized", "total_net_zscore",
        ]

    def load_price_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        加载价格数据并应用时序安全处理
        
        Args:
            symbols: 股票代码列表
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            时序安全的数据DataFrame
        """
        # 调用被包装的提供者
        data = self.wrapped_provider.load_price_data(symbols, timeframe, start_date, end_date)
        
        if data.empty:
            return data
        
        # 应用T+1时序安全处理
        data = self._apply_temporal_safety(data)
        
        return data

    def _apply_temporal_safety(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        应用T+1时序安全处理
        
        Args:
            data: 原始数据
            
        Returns:
            时序安全的数据
        """
        # 检测数据中存在的资金流字段
        money_flow_cols_present = [col for col in self.money_flow_columns if col in data.columns]
        
        if not money_flow_cols_present:
            # 没有资金流字段，直接返回
            return data
        
        logger.debug(f"对{len(money_flow_cols_present)}个资金流字段执行T+1滞后")
        
        # 处理MultiIndex情况
        if isinstance(data.index, pd.MultiIndex):
            # 按symbol分组处理
            result_dfs = []
            for symbol in data.index.get_level_values('symbol').unique():
                symbol_data = data.xs(symbol, level='symbol').copy()
                
                # 对资金流字段执行shift(1)
                for col in money_flow_cols_present:
                    symbol_data[col] = symbol_data[col].shift(1)
                
                # 添加symbol回去
                symbol_data['symbol'] = symbol
                symbol_data = symbol_data.set_index('symbol', append=True)
                result_dfs.append(symbol_data)
            
            data = pd.concat(result_dfs)
        else:
            # 单个时间序列，直接shift
            for col in money_flow_cols_present:
                data[col] = data[col].shift(1)
        
        # 添加时序安全标记
        data['temporal_safe'] = True
        
        return data

    def load_fundamental_data(
        self,
        symbols: List[str],
        fields: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """加载基本面数据（直接转发）"""
        return self.wrapped_provider.load_fundamental_data(symbols, fields, start_date, end_date)

    def get_trading_calendar(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[datetime]:
        """获取交易日历（直接转发）"""
        return self.wrapped_provider.get_trading_calendar(market, start_date, end_date)
