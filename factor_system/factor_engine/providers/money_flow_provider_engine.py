"""
资金流数据提供者 - 集成到FactorEngine

专门为因子引擎设计的资金流数据提供者，支持T+1时序安全
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from factor_system.factor_engine.providers.base import DataProvider

logger = logging.getLogger(__name__)


class MoneyFlowDataProvider(DataProvider):
    """
    资金流数据提供者

    职责：
    - 加载资金流数据
    - 合并价格数据（用于需要价格的资金流因子）
    - 确保T+1时序安全
    """

    def __init__(
        self,
        money_flow_dir: Path,
        price_dir: Optional[Path] = None,
        enforce_t_plus_1: bool = True,
    ):
        """
        初始化资金流数据提供者

        Args:
            money_flow_dir: 资金流数据目录
            price_dir: 价格数据目录（可选）
            enforce_t_plus_1: 是否强制T+1时序安全
        """
        super().__init__()
        self.money_flow_dir = Path(money_flow_dir)
        self.price_dir = Path(price_dir) if price_dir else None
        self.enforce_t_plus_1 = enforce_t_plus_1

        logger.info(f"初始化MoneyFlowDataProvider: {self.money_flow_dir}")

    def load_money_flow_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        加载资金流数据

        Args:
            symbols: 股票代码列表
            timeframe: 时间框架（应为"1day"）
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            资金流数据DataFrame，MultiIndex(symbol, date)
        """
        if timeframe not in ("1day", "daily"):
            raise ValueError(f"资金流因子只支持日线数据，当前timeframe={timeframe}")

        all_data = []

        for symbol in symbols:
            try:
                # 加载单个股票的资金流数据
                df = self._load_single_symbol_money_flow(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    # 添加symbol列用于MultiIndex (symbol, date)
                    df["symbol"] = symbol
                    df = df.reset_index().set_index(["symbol", "trade_date"])
                    all_data.append(df)

                logger.debug(f"✅ 加载{symbol}资金流数据: {len(df)}天")

            except Exception as e:
                logger.error(f"❌ 加载{symbol}资金流数据失败: {e}")
                continue

        if not all_data:
            logger.warning("未加载到任何资金流数据")
            return pd.DataFrame()

        # 合并所有股票的数据
        result = pd.concat(all_data)

        logger.info(f"资金流数据加载完成: {result.shape}")
        return result

    def load_price_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        加载价格数据（如果需要的话）

        Args:
            symbols: 股票代码列表
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            价格数据DataFrame
        """
        # 对于资金流因子，主要使用资金流数据
        # 只有少数因子（如Flow_Price_Divergence）需要价格数据
        # 这里返回空DataFrame，让因子引擎自己处理数据合并
        logger.debug("资金流数据提供者不直接提供价格数据")
        return pd.DataFrame()

    def _load_single_symbol_money_flow(
        self, symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        加载单个股票的资金流数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            资金流数据DataFrame或None
        """
        # 支持多种文件名格式（优先匹配无下划线格式）
        possible_names = [
            f"{symbol}_moneyflow.parquet",
            f"{symbol}_money_flow.parquet",
            f"{symbol}.parquet",
        ]

        file_path = None
        for name in possible_names:
            path = self.money_flow_dir / name
            if path.exists():
                file_path = path
                break

        if not file_path:
            logger.warning(f"未找到{symbol}的资金流数据文件")
            return None

        # 加载数据
        df = pd.read_parquet(file_path)

        # 处理日期格式兼容性
        if df["trade_date"].dtype == 'object':
            start_date_str = start_date.replace('-', '')
            end_date_str = end_date.replace('-', '')
            df = df[
                (df["trade_date"] >= start_date_str) & (df["trade_date"] <= end_date_str)
            ].copy()
        else:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[
                (df["trade_date"] >= start_dt) & (df["trade_date"] <= end_dt)
            ].copy()

        if df.empty:
            logger.warning(f"{symbol}在指定日期范围内无资金流数据")
            return None

        # 转换日期并设置索引
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df.set_index("trade_date", inplace=True)
        df = df.sort_index()

        # 字段映射
        rename_map = {
            "buy_sm_amount": "buy_small_amount",
            "sell_sm_amount": "sell_small_amount",
            "buy_md_amount": "buy_medium_amount",
            "sell_md_amount": "sell_medium_amount",
            "buy_lg_amount": "buy_large_amount",
            "sell_lg_amount": "sell_large_amount",
            "buy_elg_amount": "buy_super_large_amount",
            "sell_elg_amount": "sell_super_large_amount",
        }
        df = df.rename(columns=rename_map)

        # 计算衍生字段
        df = self._calculate_derived_fields(df)

        # T+1时序安全处理
        if self.enforce_t_plus_1:
            df = self._apply_t_plus_1_lag(df)

        # 添加元信息
        df["data_source"] = "money_flow"
        df["temporal_safe"] = self.enforce_t_plus_1

        return df

    def _calculate_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算衍生字段并转换单位
        
        注意：TuShare原始数据单位是"万元"，需要转换为"元"（×10000）
        """
        # 单位转换：万元 -> 元
        amount_cols = [
            "buy_small_amount", "sell_small_amount",
            "buy_medium_amount", "sell_medium_amount",
            "buy_large_amount", "sell_large_amount",
            "buy_super_large_amount", "sell_super_large_amount",
        ]
        
        for col in amount_cols:
            if col in df.columns:
                df[col] = df[col] * 10000
        
        # 成交额（已转换为元）
        df["turnover_amount"] = (
            df.get("buy_small_amount", 0) + df.get("sell_small_amount", 0) +
            df.get("buy_medium_amount", 0) + df.get("sell_medium_amount", 0) +
            df.get("buy_large_amount", 0) + df.get("sell_large_amount", 0) +
            df.get("buy_super_large_amount", 0) + df.get("sell_super_large_amount", 0)
        )

        # 主力净额（已转换为元）
        df["main_net"] = (
            df.get("buy_large_amount", 0) + df.get("buy_super_large_amount", 0) -
            df.get("sell_large_amount", 0) - df.get("sell_super_large_amount", 0)
        )

        # 散户净额（已转换为元）
        df["retail_net"] = df.get("buy_small_amount", 0) - df.get("sell_small_amount", 0)
        
        # 添加单位标记
        df["value_unit"] = "yuan"

        return df

    def _apply_t_plus_1_lag(self, df: pd.DataFrame) -> pd.DataFrame:
        """T+1时序安全处理"""
        money_flow_cols = [
            "buy_small_amount", "sell_small_amount",
            "buy_medium_amount", "sell_medium_amount",
            "buy_large_amount", "sell_large_amount",
            "buy_super_large_amount", "sell_super_large_amount",
            "turnover_amount", "main_net", "retail_net", "total_net",
        ]

        # 向量化滞后处理
        for col in money_flow_cols:
            if col in df.columns:
                df[col] = df[col].shift(1)

        return df

    def load_fundamental_data(
        self,
        symbols: List[str],
        fields: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        加载基本面数据（资金流提供者不支持）

        Args:
            symbols: 股票代码列表
            fields: 字段列表
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            空DataFrame（资金流提供者不支持基本面数据）
        """
        logger.debug("资金流数据提供者不支持基本面数据")
        return pd.DataFrame()

    def get_trading_calendar(
        self,
        market: str,
        start_date: str,
        end_date: str,
    ) -> List[str]:
        """
        获取交易日历（资金流提供者不支持）

        Args:
            market: 市场代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            空列表（资金流提供者不支持交易日历）
        """
        logger.debug("资金流数据提供者不支持交易日历")
        return []