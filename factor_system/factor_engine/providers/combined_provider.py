"""
组合数据提供者 - 价格 + 资金流统一接口
透明合并价格数据和资金流数据，确保T+1时序安全
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from factor_system.factor_engine.providers.base import DataProvider

logger = logging.getLogger(__name__)


class CombinedMoneyFlowProvider(DataProvider):
    """
    组合数据提供者 - 价格 + 资金流

    特性：
    1. 透明合并价格和资金流数据
    2. 自动对齐索引（symbol, date）
    3. T+1时序安全验证
    4. 缺失数据日志记录
    """

    def __init__(
        self,
        price_provider: DataProvider,
        money_flow_dir: Path,
        enforce_t_plus_1: bool = True,
    ):
        """
        初始化组合提供者

        Args:
            price_provider: 价格数据提供者
            money_flow_dir: 资金流数据根目录（如raw/SH/money_flow）
            enforce_t_plus_1: 是否强制T+1时序安全
        """
        self.price_provider = price_provider
        self.money_flow_base_dir = Path(money_flow_dir)
        self.enforce_t_plus_1 = enforce_t_plus_1

        # 市场目录映射
        self.market_money_flow_dirs = {
            "SH": self.money_flow_base_dir.parent / "SH/money_flow",
            "SZ": self.money_flow_base_dir.parent / "SZ/money_flow",
        }

        logger.info("CombinedMoneyFlowProvider初始化完成")
        logger.info(f"  价格提供者: {type(price_provider).__name__}")
        logger.info(f"  资金流基础目录: {self.money_flow_base_dir}")
        logger.info(f"  T+1时序安全: {enforce_t_plus_1}")

    def load_price_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        加载价格数据，日线时自动合并资金流

        Args:
            symbols: 股票代码列表
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            MultiIndex DataFrame，日线时包含资金流字段
        """
        # 1. 加载价格数据
        logger.info(f"加载价格数据: {len(symbols)} 个标的, {timeframe}")
        price_data = self.price_provider.load_price_data(
            symbols, timeframe, start_date, end_date
        )

        if price_data.empty:
            logger.warning("价格数据为空")
            return price_data

        logger.info(f"✅ 价格数据: {price_data.shape}")

        # 2. 如果是日线，合并资金流数据
        if timeframe == "daily":
            logger.info("检测到日线请求，尝试合并资金流数据...")

            try:
                # 按市场分组加载资金流
                money_flow_data = self._load_money_flow_by_market(
                    symbols, start_date, end_date
                )

                if not money_flow_data.empty:
                    # 合并数据
                    combined_data = self._align_and_merge(price_data, money_flow_data)
                    logger.info(f"✅ 合并完成: {combined_data.shape}")
                    return combined_data
                else:
                    logger.warning("资金流数据为空，仅返回价格数据")

            except Exception as e:
                logger.error(f"资金流数据加载失败: {e}，仅返回价格数据")

        return price_data

    def _load_money_flow_by_market(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        按市场分组加载资金流数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            合并的资金流数据
        """
        from factor_system.factor_engine.providers.money_flow_provider_engine import (
            MoneyFlowDataProvider,
        )

        # 按市场分组
        market_symbols = {"SH": [], "SZ": []}
        for symbol in symbols:
            if symbol.endswith(".SH"):
                market_symbols["SH"].append(symbol)
            elif symbol.endswith(".SZ"):
                market_symbols["SZ"].append(symbol)
            else:
                logger.warning(f"未知市场后缀: {symbol}，跳过资金流加载")

        all_mf_data = []

        for market, market_syms in market_symbols.items():
            if not market_syms:
                continue

            # 选择目录
            mf_dir = self.market_money_flow_dirs.get(market)
            if not mf_dir or not mf_dir.exists():
                # 回退到基础目录
                logger.info(f"{market} 目录不存在，回退到基础目录")
                mf_dir = self.money_flow_base_dir

            if not mf_dir.exists():
                logger.warning(f"资金流目录不存在: {mf_dir}，跳过 {market} 市场")
                continue

            logger.info(f"加载 {market} 市场资金流: {len(market_syms)} 个标的")
            logger.debug(f"  目录: {mf_dir}")
            logger.debug(f"  时间窗: {start_date.date()} ~ {end_date.date()}")

            try:
                provider = MoneyFlowDataProvider(
                    money_flow_dir=mf_dir,
                    enforce_t_plus_1=self.enforce_t_plus_1,
                )

                mf_data = provider.load_money_flow_data(
                    market_syms,
                    "daily",
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                )

                if not mf_data.empty:
                    all_mf_data.append(mf_data)
                    logger.info(f"  ✅ {market}: {mf_data.shape}")
                else:
                    logger.warning(f"  ⚠️ {market}: 数据为空")
                    logger.debug(f"    可能原因: 无文件/日期范围外/字段缺失")

            except FileNotFoundError as e:
                logger.error(f"  ❌ {market} 文件不存在: {e}")
                logger.debug(f"    目录: {mf_dir}")
                logger.debug(f"    标的: {market_syms}")
            except Exception as e:
                logger.error(f"  ❌ {market} 加载失败: {type(e).__name__}: {e}")
                logger.debug(f"    目录: {mf_dir}")
                logger.debug(f"    时间窗: {start_date.date()} ~ {end_date.date()}")

        if all_mf_data:
            return pd.concat(all_mf_data)
        else:
            return pd.DataFrame()

    def _align_and_merge(
        self,
        price_data: pd.DataFrame,
        money_flow_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        对齐并合并价格和资金流数据

        Args:
            price_data: 价格数据 (symbol, datetime) × OHLCV
            money_flow_data: 资金流数据 (symbol, trade_date) × 资金流字段

        Returns:
            合并后的DataFrame
        """
        logger.info("开始对齐和合并数据...")

        # 验证索引结构
        if not isinstance(price_data.index, pd.MultiIndex):
            raise ValueError("价格数据索引必须是MultiIndex(symbol, datetime)")

        if not isinstance(money_flow_data.index, pd.MultiIndex):
            raise ValueError("资金流数据索引必须是MultiIndex(symbol, trade_date)")

        # 统一索引名称
        price_data.index.names = ["symbol", "datetime"]
        money_flow_data.index.names = ["symbol", "datetime"]

        # 日期归一化：去除时分秒，防止时区/时间戳错配
        logger.debug("归一化日期索引到00:00:00...")
        price_data.index = price_data.index.set_levels(
            price_data.index.levels[1].normalize(), level=1
        )
        money_flow_data.index = money_flow_data.index.set_levels(
            money_flow_data.index.levels[1].normalize(), level=1
        )

        # 记录原始形状
        logger.info(f"  价格数据: {price_data.shape}")
        logger.info(f"  资金流数据: {money_flow_data.shape}")

        # 合并（左连接，保留所有价格数据）
        combined = price_data.join(money_flow_data, how="left")

        # 检测资金流列（排除元数据列）
        mf_cols = [
            c
            for c in money_flow_data.columns
            if c not in ["value_unit", "temporal_safe"]
        ]

        # 如果存在资金流列，添加元数据列
        if mf_cols:
            # 单位保险丝：验证或添加 value_unit
            if "value_unit" in combined.columns:
                unit_values = combined["value_unit"].dropna().unique()
                if len(unit_values) > 0:
                    if "yuan" not in unit_values:
                        logger.error(
                            f"⚠️ 单位异常：value_unit={unit_values}，预期'yuan'"
                        )
                        raise ValueError(
                            "资金流数据单位不是'yuan'，可能存在二次缩放风险"
                        )
                    else:
                        logger.info("✅ 单位验证通过: value_unit='yuan'")
            else:
                # 主动添加元数据列
                combined["value_unit"] = "yuan"
                logger.info("✅ 已添加元数据: value_unit='yuan'")

            # 添加时序安全标记
            if "temporal_safe" not in combined.columns:
                combined["temporal_safe"] = True
                logger.info("✅ 已添加元数据: temporal_safe=True")

        # 统计合并结果
        mf_cols = [c for c in money_flow_data.columns if c != "value_unit"]
        for col in mf_cols[:5]:  # 只显示前5个
            non_null = combined[col].notna().sum()
            total = len(combined)
            pct = non_null / total * 100
            logger.info(f"  {col}: {non_null}/{total} ({pct:.1f}%)")

        # T+1验证（可选）
        if self.enforce_t_plus_1:
            self._verify_t_plus_1(combined, mf_cols)

        return combined

    def _verify_t_plus_1(self, data: pd.DataFrame, mf_cols: pd.Index):
        """验证T+1时序安全"""
        # 检查每个股票的第一天资金流数据是否为NaN
        for symbol in data.index.get_level_values("symbol").unique():
            symbol_data = data.xs(symbol, level="symbol")
            if not symbol_data.empty:
                first_mf_value = symbol_data[mf_cols[0]].iloc[0]
                if pd.notna(first_mf_value):
                    logger.warning(
                        f"⚠️ {symbol}: 第一天资金流数据不为NaN，可能存在前视偏差"
                    )

    def load_fundamental_data(self, *args, **kwargs) -> pd.DataFrame:
        """加载基本面数据（委托给价格提供者）"""
        return self.price_provider.load_fundamental_data(*args, **kwargs)

    def get_trading_calendar(self, *args, **kwargs) -> List:
        """获取交易日历（委托给价格提供者）"""
        return self.price_provider.get_trading_calendar(*args, **kwargs)
