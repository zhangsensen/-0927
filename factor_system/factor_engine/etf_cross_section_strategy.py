#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面策略引擎
基于多因子模型的ETF选择和配置策略
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from factor_system.factor_engine.factors.etf_cross_section import ETFCrossSectionFactors
from factor_system.factor_engine.providers.etf_cross_section_provider import (
    ETFCrossSectionDataManager,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """策略配置"""

    # 基础配置
    start_date: str
    end_date: str
    etf_universe: Optional[List[str]] = None
    rebalance_freq: str = "M"  # M=月度, W=周度, Q=季度

    # 选择配置
    top_n: int = 8  # 选择前N只ETF
    min_score: float = 0.0  # 最小综合得分

    # 权重配置
    weight_method: str = "equal"  # equal=等权, score=因子加权, inverse_vol=波动率倒数
    max_single_weight: float = 0.25  # 单只ETF最大权重

    # 风险控制
    max_sector_weight: float = 0.40  # 单行业最大权重
    min_liquidity_score: float = 0.0  # 最小流动性得分
    lookback_days: int = 252  # 回测窗口

    # 因子权重
    momentum_weight: float = 0.4
    quality_weight: float = 0.3
    liquidity_weight: float = 0.2
    technical_weight: float = 0.1


class ETFCrossSectionStrategy:
    """ETF横截面策略引擎"""

    def __init__(self, config: StrategyConfig):
        """
        初始化策略引擎

        Args:
            config: 策略配置
        """
        self.config = config
        self.data_manager = ETFCrossSectionDataManager()
        self.factor_calculator = ETFCrossSectionFactors(self.data_manager)
        self.factor_data: Optional[pd.DataFrame] = None
        self.portfolio_history: List[Dict] = []

    def load_factor_data(self) -> bool:
        """
        加载因子数据

        Returns:
            是否成功加载
        """
        try:
            logger.info(
                f"开始加载因子数据: {self.config.start_date} ~ {self.config.end_date}"
            )

            # 确定ETF投资域
            if self.config.etf_universe is None:
                self.config.etf_universe = self.data_manager.get_etf_universe(
                    min_trading_days=self.config.lookback_days
                )

            logger.info(f"ETF投资域: {len(self.config.etf_universe)} 只")

            # 计算因子数据
            self.factor_data = self.factor_calculator.calculate_all_factors(
                self.config.start_date, self.config.end_date, self.config.etf_universe
            )

            if self.factor_data.empty:
                logger.error("因子数据加载失败")
                return False

            logger.info(f"因子数据加载成功: {len(self.factor_data)} 条记录")
            return True

        except Exception as e:
            logger.error(f"因子数据加载失败: {e}")
            return False

    def get_rebalance_dates(self) -> List[datetime]:
        """
        获取调仓日期列表

        Returns:
            调仓日期列表
        """
        if self.factor_data is None or self.factor_data.empty:
            return []

        # 获取所有交易日期
        all_dates = sorted(self.factor_data["date"].unique())
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)

        # 过滤日期范围
        valid_dates = [date for date in all_dates if start_date <= date <= end_date]

        if not valid_dates:
            return []

        # 根据调仓频率选择日期
        rebalance_dates = []

        if self.config.rebalance_freq == "M":
            # 月度调仓：每月最后一个交易日
            for year in range(valid_dates[0].year, valid_dates[-1].year + 1):
                for month in range(1, 13):
                    month_dates = [
                        d for d in valid_dates if d.year == year and d.month == month
                    ]
                    if month_dates:
                        rebalance_dates.append(max(month_dates))

        elif self.config.rebalance_freq == "W":
            # 周度调仓：每周最后一个交易日
            for date in valid_dates:
                if date.weekday() == 4:  # 周五
                    rebalance_dates.append(date)

        elif self.config.rebalance_freq == "Q":
            # 季度调仓：每季度最后一个交易日
            for year in range(valid_dates[0].year, valid_dates[-1].year + 1):
                for quarter in [3, 6, 9, 12]:
                    quarter_dates = [
                        d
                        for d in valid_dates
                        if d.year == year
                        and d.month <= quarter
                        and d.month > quarter - 3
                    ]
                    if quarter_dates:
                        rebalance_dates.append(max(quarter_dates))

        # 去重并排序
        rebalance_dates = sorted(list(set(rebalance_dates)))

        logger.info(f"调仓日期: {len(rebalance_dates)} 个")
        return rebalance_dates

    def select_etfs_at_date(self, date: datetime) -> List[Tuple[str, float]]:
        """
        在指定日期选择ETF

        Args:
            date: 调仓日期

        Returns:
            选择的ETF列表 [(etf_code, weight), ...]
        """
        if self.factor_data is None:
            return []

        # 获取该日期的因子数据
        date_factors = self.factor_data[self.factor_data["date"] == date].copy()

        if date_factors.empty:
            logger.warning(f"日期 {date} 没有因子数据")
            return []

        # 基础筛选
        # 1. 最小综合得分（调整为负值允许）
        if "composite_score" in date_factors.columns:
            date_factors = date_factors[
                date_factors["composite_score"] >= self.config.min_score
            ]

        # 2. 最小流动性要求（调整为0，允许所有有流动性的ETF）
        if "liquidity_score" in date_factors.columns:
            date_factors = date_factors[date_factors["liquidity_score"] >= 0]

        if date_factors.empty:
            logger.warning(f"日期 {date} 筛选后无符合条件的ETF")
            return []

        # 按综合得分排序
        if "composite_score" in date_factors.columns:
            date_factors = date_factors.sort_values("composite_score", ascending=False)

        # 选择前N只
        selected_etfs = date_factors.head(self.config.top_n)

        # 计算权重
        weights = self._calculate_weights(selected_etfs)

        # 应用风险控制
        weights = self._apply_risk_controls(weights, date)

        logger.info(f"日期 {date} 选择了 {len(weights)} 只ETF")
        return weights

    def _calculate_weights(
        self, selected_etfs: pd.DataFrame
    ) -> List[Tuple[str, float]]:
        """
        计算权重

        Args:
            selected_etfs: 选择的ETF DataFrame

        Returns:
            权重列表 [(etf_code, weight), ...]
        """
        weights = []

        if self.config.weight_method == "equal":
            # 等权重
            equal_weight = 1.0 / len(selected_etfs)
            for _, row in selected_etfs.iterrows():
                weights.append((row["etf_code"], equal_weight))

        elif self.config.weight_method == "score":
            # 因子加权
            if "composite_score" in selected_etfs.columns:
                total_score = selected_etfs["composite_score"].sum()
                for _, row in selected_etfs.iterrows():
                    weight = row["composite_score"] / total_score
                    weights.append((row["etf_code"], weight))

        elif self.config.weight_method == "inverse_vol":
            # 波动率倒数加权
            if "volatility_1y" in selected_etfs.columns:
                # 避免除零
                vols = selected_etfs["volatility_1y"].replace(0, np.nan).dropna()
                if len(vols) > 0:
                    inv_vols = 1.0 / vols
                    total_inv_vol = inv_vols.sum()

                    for _, row in selected_etfs.iterrows():
                        if row["volatility_1y"] > 0:
                            weight = (1.0 / row["volatility_1y"]) / total_inv_vol
                            weights.append((row["etf_code"], weight))

        # 归一化权重
        if weights:
            total_weight = sum(weight for _, weight in weights)
            if total_weight > 0:
                weights = [(etf, weight / total_weight) for etf, weight in weights]

        return weights

    def _apply_risk_controls(
        self, weights: List[Tuple[str, float]], date: datetime
    ) -> List[Tuple[str, float]]:
        """
        应用风险控制

        Args:
            weights: 原始权重
            date: 调仓日期

        Returns:
            调整后的权重
        """
        if not weights:
            return weights

        # 1. 单只ETF最大权重限制
        adjusted_weights = []
        for etf_code, weight in weights:
            adjusted_weight = min(weight, self.config.max_single_weight)
            adjusted_weights.append((etf_code, adjusted_weight))

        # 2. 重新归一化
        total_weight = sum(weight for _, weight in adjusted_weights)
        if total_weight > 0:
            adjusted_weights = [
                (etf, weight / total_weight) for etf, weight in adjusted_weights
            ]

        # TODO: 3. 行业权重控制（需要ETF分类数据）

        return adjusted_weights

    def run_backtest(self) -> Dict:
        """
        运行回测

        Returns:
            回测结果
        """
        logger.info("开始运行ETF横截面策略回测")

        # 加载因子数据
        if not self.load_factor_data():
            return {"success": False, "error": "因子数据加载失败"}

        # 获取调仓日期
        rebalance_dates = self.get_rebalance_dates()
        if not rebalance_dates:
            return {"success": False, "error": "无调仓日期"}

        # 运行回测
        portfolio_returns = []
        portfolio_weights = []

        for i, rebalance_date in enumerate(rebalance_dates):
            logger.info(f"处理调仓日期 {i+1}/{len(rebalance_dates)}: {rebalance_date}")

            # 选择ETF
            selected_etfs = self.select_etfs_at_date(rebalance_date)
            if not selected_etfs:
                continue

            # 记录组合
            portfolio_record = {
                "date": rebalance_date,
                "etfs": selected_etfs,
                "etf_codes": [etf for etf, _ in selected_etfs],
                "weights": [weight for _, weight in selected_etfs],
            }
            self.portfolio_history.append(portfolio_record)

            portfolio_weights.append(portfolio_record)

        # 计算组合收益
        performance = self._calculate_performance(portfolio_weights)

        result = {
            "success": True,
            "config": self.config,
            "portfolio_history": portfolio_weights,
            "performance": performance,
            "rebalance_count": len(rebalance_dates),
        }

        logger.info(f"回测完成: {len(portfolio_weights)} 次调仓")
        return result

    def _calculate_performance(self, portfolio_weights: List[Dict]) -> Dict:
        """
        计算组合表现

        Args:
            portfolio_weights: 组合权重历史

        Returns:
            表现统计
        """
        if not portfolio_weights:
            return {}

        # 获取价格数据
        all_etfs = list(
            set([etf for record in portfolio_weights for etf, _ in record["etfs"]])
        )
        price_data = self.data_manager.get_time_series_data(
            self.config.start_date, self.config.end_date, all_etfs
        )

        if price_data.empty:
            return {}

        # 计算组合净值
        portfolio_value = 1.0
        portfolio_values = [portfolio_value]
        portfolio_dates = []

        for i, record in enumerate(portfolio_weights):
            current_date = record["date"]
            portfolio_dates.append(current_date)

            if i == 0:
                continue

            prev_date = portfolio_weights[i - 1]["date"]
            prev_etfs = {
                etf: weight for etf, weight in portfolio_weights[i - 1]["etfs"]
            }

            # 计算期间收益
            period_return = 0.0
            valid_returns = []

            for etf, weight in prev_etfs.items():
                etf_data = price_data[price_data["etf_code"] == etf]
                etf_data = etf_data[
                    (etf_data["trade_date"] >= prev_date)
                    & (etf_data["trade_date"] <= current_date)
                ]

                if len(etf_data) >= 2:
                    start_price = etf_data.iloc[0]["close"]
                    end_price = etf_data.iloc[-1]["close"]
                    etf_return = (end_price / start_price) - 1
                    period_return += weight * etf_return
                    valid_returns.append(etf_return)

            if valid_returns:
                portfolio_value *= 1 + period_return
                portfolio_values.append(portfolio_value)

        # 计算统计指标
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]

            performance = {
                "total_return": portfolio_values[-1] - 1,
                "annualized_return": (portfolio_values[-1] ** (252 / len(returns))) - 1,
                "volatility": np.std(returns) * np.sqrt(252),
                "sharpe_ratio": (
                    (np.mean(returns) / np.std(returns) * np.sqrt(252))
                    if np.std(returns) > 0
                    else 0
                ),
                "max_drawdown": self._calculate_max_drawdown(portfolio_values),
                "portfolio_values": portfolio_values,
                "portfolio_dates": portfolio_dates,
            }
        else:
            performance = {"portfolio_values": portfolio_values}

        return performance

    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """计算最大回撤"""
        if len(values) < 2:
            return 0.0

        peak = values[0]
        max_dd = 0.0

        for value in values[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    def get_latest_portfolio(self) -> Optional[List[Tuple[str, float]]]:
        """
        获取最新组合

        Returns:
            最新ETF组合权重
        """
        if not self.portfolio_history:
            return None

        latest_record = self.portfolio_history[-1]
        return latest_record["etfs"]

    def export_results(self, filepath: str):
        """
        导出回测结果

        Args:
            filepath: 导出文件路径
        """
        if not self.portfolio_history:
            logger.warning("没有可导出的结果")
            return

        # 转换为DataFrame
        results_data = []
        for record in self.portfolio_history:
            for etf, weight in record["etfs"]:
                results_data.append(
                    {"date": record["date"], "etf_code": etf, "weight": weight}
                )

        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filepath, index=False)
        logger.info(f"结果已导出到: {filepath}")


# 便捷函数
def run_etf_cross_section_strategy(
    start_date: str,
    end_date: str,
    top_n: int = 8,
    rebalance_freq: str = "M",
    weight_method: str = "equal",
) -> Dict:
    """
    运行ETF横截面策略的便捷函数

    Args:
        start_date: 开始日期
        end_date: 结束日期
        top_n: 选择ETF数量
        rebalance_freq: 调仓频率
        weight_method: 权重方法

    Returns:
        回测结果
    """
    config = StrategyConfig(
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
        rebalance_freq=rebalance_freq,
        weight_method=weight_method,
    )

    strategy = ETFCrossSectionStrategy(config)
    return strategy.run_backtest()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 简单回测测试
    config = StrategyConfig(
        start_date="2024-01-01",
        end_date="2025-10-14",
        top_n=5,
        rebalance_freq="M",
        weight_method="equal",
    )

    strategy = ETFCrossSectionStrategy(config)
    result = strategy.run_backtest()

    if result["success"]:
        print("回测成功！")
        performance = result.get("performance", {})
        print(f"总收益: {performance.get('total_return', 0):.2%}")
        print(f"年化收益: {performance.get('annualized_return', 0):.2%}")
        print(f"夏普比率: {performance.get('sharpe_ratio', 0):.2f}")
        print(f"最大回撤: {performance.get('max_drawdown', 0):.2%}")
        print(f"调仓次数: {result['rebalance_count']}")

        # 显示最新组合
        latest_portfolio = strategy.get_latest_portfolio()
        if latest_portfolio:
            print("\n最新组合:")
            for etf, weight in latest_portfolio:
                print(f"  {etf}: {weight:.2%}")
    else:
        print(f"回测失败: {result.get('error', '未知错误')}")
