#!/usr/bin/env python3
"""
资金流数据提供者适配器
将MoneyFlowProvider适配到FactorEngine接口
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from factor_system.factor_engine.providers.base import DataProvider
from factor_system.factor_engine.providers.money_flow_provider import MoneyFlowProvider

logger = logging.getLogger(__name__)


class MoneyFlowAdapter:
    """
    资金流数据提供者适配器

    职责：
    - 适配MoneyFlowProvider到FactorEngine接口
    - 提供统一的数据格式
    - 处理多股票数据合并
    """

    def __init__(
        self,
        money_flow_dir: Path,
        price_dir: Optional[Path] = None,
        enforce_t_plus_1: bool = True,
    ):
        """
        初始化适配器

        Args:
            money_flow_dir: 资金流数据目录
            price_dir: 价格数据目录（可选）
            enforce_t_plus_1: 是否强制T+1滞后
        """
        self.money_flow_dir = Path(money_flow_dir)
        self.price_dir = Path(price_dir) if price_dir else None
        self.enforce_t_plus_1 = enforce_t_plus_1

        # 初始化资金流提供者
        self.provider = MoneyFlowProvider(
            data_dir=self.money_flow_dir, enforce_t_plus_1=self.enforce_t_plus_1
        )

        logger.info(f"MoneyFlowAdapter初始化完成: {money_flow_dir}")

    def load_data(
        self, symbols: List[str], start_date: str, end_date: str, timeframe: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        加载多股票资金流数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            timeframe: 时间框架（暂只支持1d）

        Returns:
            {symbol: DataFrame} 的字典
        """
        logger.info(f"加载资金流数据: {len(symbols)}个股票, {start_date} 到 {end_date}")

        results = {}

        for symbol in symbols:
            try:
                # 构建资金流文件名，支持多种格式
                possible_files = [
                    f"{symbol}_money_flow.parquet",
                    f"{symbol}_moneyflow.parquet",
                    f"{symbol}_moneyflow.parquet",
                ]

                # 尝试不同的文件名格式
                loaded = False
                for filename in possible_files:
                    file_path = self.money_flow_dir / filename
                    if file_path.exists():
                        df = self.provider.load_money_flow(symbol, start_date, end_date)
                        if not df.empty:
                            results[symbol] = df
                            logger.debug(
                                f"✅ {symbol}: {len(df)}行数据 (文件: {filename})"
                            )
                            loaded = True
                            break

                if not loaded:
                    logger.warning(f"⚠️ {symbol}: 无可用数据文件")

            except Exception as e:
                logger.error(f"❌ {symbol}: 加载失败 - {e}")
                continue

        logger.info(f"成功加载 {len(results)}/{len(symbols)} 个股票的资金流数据")
        return results

    def get_available_symbols(self) -> List[str]:
        """
        获取可用的股票代码列表

        Returns:
            股票代码列表
        """
        if not self.money_flow_dir.exists():
            logger.warning(f"资金流数据目录不存在: {self.money_flow_dir}")
            return []

        # 扫描parquet文件
        symbols = []
        for file_path in self.money_flow_dir.glob("*.parquet"):
            # 提取股票代码：600036.SH_money_flow.parquet -> 600036.SH
            filename = file_path.stem
            if filename.endswith("_money_flow"):
                symbol = filename[:-12]  # 移除"_money_flow"后缀
                symbols.append(symbol)

        logger.info(f"发现 {len(symbols)} 个资金流数据文件")
        return sorted(symbols)

    def validate_data_coverage(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> Dict[str, Dict]:
        """
        验证数据覆盖情况

        Returns:
            {symbol: {start_date, end_date, count, coverage_ratio}} 的字典
        """
        logger.info("验证资金流数据覆盖情况...")

        coverage_info = {}
        target_start = pd.to_datetime(start_date)
        target_end = pd.to_datetime(end_date)
        target_days = (target_end - target_start).days + 1

        for symbol in symbols:
            try:
                df = self.provider.load_money_flow(symbol, start_date, end_date)

                if df.empty:
                    coverage_info[symbol] = {
                        "start_date": None,
                        "end_date": None,
                        "count": 0,
                        "coverage_ratio": 0.0,
                        "status": "no_data",
                    }
                else:
                    actual_start = df.index.min()
                    actual_end = df.index.max()
                    actual_count = len(df)
                    coverage_ratio = (
                        actual_count / target_days if target_days > 0 else 0
                    )

                    coverage_info[symbol] = {
                        "start_date": actual_start,
                        "end_date": actual_end,
                        "count": actual_count,
                        "coverage_ratio": coverage_ratio,
                        "status": "good" if coverage_ratio > 0.8 else "poor",
                    }

            except Exception as e:
                coverage_info[symbol] = {
                    "start_date": None,
                    "end_date": None,
                    "count": 0,
                    "coverage_ratio": 0.0,
                    "status": "error",
                    "error": str(e),
                }

        # 统计覆盖率
        good_symbols = sum(
            1 for info in coverage_info.values() if info.get("status") == "good"
        )
        logger.info(
            f"数据覆盖率: {good_symbols}/{len(symbols)} ({good_symbols/len(symbols)*100:.1f}%) 良好"
        )

        return coverage_info

    def get_factor_columns(self) -> List[str]:
        """
        获取可用的资金流因子列

        Returns:
            因子列名列表
        """
        # 这里返回我们实现的12个因子
        factor_columns = [
            # 核心因子（8个）
            "MainNetInflow_Rate",
            "LargeOrder_Ratio",
            "SuperLargeOrder_Ratio",
            "OrderConcentration",
            "MoneyFlow_Hierarchy",
            "MoneyFlow_Consensus",
            "MainFlow_Momentum",
            "Flow_Price_Divergence",
            # 增强因子（4个）
            "Institutional_Absorption",
            "Flow_Tier_Ratio_Delta",
            "Flow_Reversal_Ratio",
            "Northbound_NetInflow_Rate",
        ]

        return factor_columns
