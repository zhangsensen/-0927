"""
增强的因子引擎 - 支持资金流因子

集成资金流因子到统一的因子计算引擎中
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from factor_system.factor_engine.core.engine import FactorEngine
from factor_system.factor_engine.providers.money_flow_provider_engine import (
    MoneyFlowDataProvider,
)

logger = logging.getLogger(__name__)


class EnhancedFactorEngine(FactorEngine):
    """
    增强因子引擎，支持技术和资金流因子混合计算

    新增功能：
    - 多数据源支持（价格 + 资金流）
    - 资金流因子集成
    - 跨数据源因子计算
    """

    def __init__(
        self,
        data_provider,
        money_flow_provider: Optional[MoneyFlowDataProvider] = None,
        registry=None,
        cache_config=None,
    ):
        """
        初始化增强因子引擎

        Args:
            data_provider: 价格数据提供者
            money_flow_provider: 资金流数据提供者
            registry: 因子注册表（应已注册资金流因子）
            cache_config: 缓存配置
        """
        super().__init__(data_provider, registry, cache_config)

        self.money_flow_provider = money_flow_provider
        if self.money_flow_provider is None:
            # 默认创建资金流数据提供者
            self.money_flow_provider = MoneyFlowDataProvider(
                money_flow_dir="raw/SH/money_flow", enforce_t_plus_1=True
            )

        logger.info("初始化EnhancedFactorEngine，支持资金流因子")

    def calculate_mixed_factors(
        self,
        technical_factors: List[str],
        money_flow_factors: List[str],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        计算混合因子（技术和资金流因子）

        Args:
            technical_factors: 技术因子列表
            money_flow_factors: 资金流因子列表
            symbols: 股票代码列表
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        Returns:
            合并的因子数据DataFrame
        """
        logger.info(
            f"开始计算混合因子: {len(technical_factors)}个技术因子, "
            f"{len(money_flow_factors)}个资金流因子, "
            f"{len(symbols)}个标的"
        )

        all_factors = []
        factor_data_source_map = {}

        # 1. 收集所有因子并标记数据源
        for factor_id in technical_factors:
            all_factors.append(factor_id)
            factor_data_source_map[factor_id] = "price"

        for factor_id in money_flow_factors:
            all_factors.append(factor_id)
            factor_data_source_map[factor_id] = "money_flow"

        # 2. 计算因子（分数据源处理）
        price_factors = [f for f in all_factors if factor_data_source_map[f] == "price"]
        money_flow_factors_list = [
            f for f in all_factors if factor_data_source_map[f] == "money_flow"
        ]

        results = []

        # 计算技术因子（基于价格数据）
        if price_factors:
            try:
                price_result = self.calculate_factors(
                    price_factors, symbols, timeframe, start_date, end_date, use_cache
                )
                if not price_result.empty:
                    results.append(price_result)
            except Exception as e:
                logger.error(f"技术因子计算失败: {e}")

        # 计算资金流因子（基于资金流数据）
        if money_flow_factors_list:
            try:
                money_flow_result = self._calculate_money_flow_factors(
                    money_flow_factors_list, symbols, start_date, end_date, use_cache
                )
                if not money_flow_result.empty:
                    results.append(money_flow_result)
            except Exception as e:
                logger.error(f"资金流因子计算失败: {e}")

        # 3. 合并结果
        if not results:
            logger.warning("所有因子计算失败")
            return pd.DataFrame()

        # 合并所有结果
        merged_result = results[0]
        for result in results[1:]:
            # 按(symbol, timestamp)对齐合并
            merged_result = merged_result.join(result, how="outer")

        logger.info(f"混合因子计算完成: {merged_result.shape}")

        # 4. 返回用户请求的因子
        requested_factors = technical_factors + money_flow_factors
        available_columns = [
            col for col in requested_factors if col in merged_result.columns
        ]

        if available_columns:
            return merged_result[available_columns]
        else:
            logger.warning("请求的因子在计算结果中未找到")
            return pd.DataFrame()

    def _calculate_money_flow_factors(
        self,
        factor_ids: List[str],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        计算资金流因子

        Args:
            factor_ids: 资金流因子ID列表
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        Returns:
            资金流因子数据DataFrame
        """
        logger.info(f"计算资金流因子: {len(factor_ids)}个因子, {len(symbols)}个标的")

        # 加载资金流数据
        money_flow_data = self.money_flow_provider.load_money_flow_data(
            symbols,
            "1day",
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        if money_flow_data.empty:
            logger.warning("资金流数据为空")
            return pd.DataFrame()

        # 筛选时间范围（MultiIndex: symbol, date）
        date_index = money_flow_data.index.get_level_values("trade_date")
        mask = (date_index >= pd.to_datetime(start_date)) & (
            date_index <= pd.to_datetime(end_date)
        )
        money_flow_data = money_flow_data[mask]

        if money_flow_data.empty:
            logger.warning("筛选后资金流数据为空")
            return pd.DataFrame()

        # 计算因子（简化版，每个股票分别计算）
        results = []

        for symbol in symbols:
            try:
                # 提取单个股票的数据
                symbol_data = money_flow_data.xs(symbol, level="symbol")

                # 计算因子
                symbol_results = {}
                for factor_id in factor_ids:
                    try:
                        # 获取因子实例
                        factor = self.registry.get_factor(factor_id)

                        # 计算因子值
                        factor_values = factor.calculate(symbol_data)

                        if isinstance(factor_values, pd.Series):
                            symbol_results[factor_id] = factor_values
                        else:
                            logger.warning(
                                f"因子{factor_id}返回类型错误: {type(factor_values)}"
                            )
                            symbol_results[factor_id] = pd.Series(
                                index=symbol_data.index, dtype=float
                            )

                    except Exception as e:
                        logger.error(f"计算因子{factor_id}失败: {e}")
                        symbol_results[factor_id] = pd.Series(
                            index=symbol_data.index, dtype=float
                        )

                # 合并单个股票的结果
                if symbol_results:
                    symbol_df = pd.DataFrame(symbol_results)
                    symbol_df["symbol"] = symbol
                    symbol_df = symbol_df.set_index("symbol", append=True)
                    results.append(symbol_df)

            except Exception as e:
                logger.error(f"处理股票{symbol}失败: {e}")
                continue

        if not results:
            logger.warning("资金流因子计算结果为空")
            return pd.DataFrame()

        # 合并所有股票的结果
        final_result = pd.concat(results)

        logger.info(f"资金流因子计算完成: {final_result.shape}")
        return final_result

    def calculate_technical_factors(
        self,
        factor_ids: List[str],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """计算纯技术因子（便捷方法）"""
        return self.calculate_factors(
            factor_ids, symbols, timeframe, start_date, end_date, use_cache
        )

    def calculate_money_flow_factors(
        self,
        factor_ids: List[str],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """计算纯资金流因子（便捷方法）"""
        return self._calculate_money_flow_factors(
            factor_ids, symbols, start_date, end_date, use_cache
        )
