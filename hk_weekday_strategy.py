#!/usr/bin/env python3
"""
港股周内交易策略因子评估与回测系统

基于港股市场特有的周内效应和季节性模式，实现专业的量化因子分析、
策略回测和风险评估。支持多因子融合、统计显著性检验和成本优化。

Author: Quantitative Chief Engineer (Linus Style)
Date: 2025-10-11
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import vectorbt as vbt
from scipy import stats

from factor_system.factor_engine import api
from hk_midfreq.config import DEFAULT_RUNTIME_CONFIG, StrategyRuntimeConfig
from hk_midfreq.factor_interface import FactorScoreLoader
from hk_midfreq.price_loader import PriceDataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class HKWeekdayEffectConfig:
    """港股周内效应分析配置"""

    # 市场特征参数
    stamp_duty_rate: float = 0.001  # 港股印花税 0.1%
    commission_rate: float = 0.002  # 佣金 0.2%
    min_commission: float = 5.0  # 最低佣金 HK$5
    slippage_bps: float = 5.0  # 滑点 5bps

    # 周内效应参数
    optimal_entry_days: List[int] = field(
        default_factory=lambda: [1, 2, 3]
    )  # 周一、二、三
    optimal_exit_days: List[int] = field(default_factory=lambda: [4, 5])  # 周四、五
    monday_effect_adjustment: float = -0.15  # 周一效应调整系数
    friday_effect_adjustment: float = 0.08  # 周五效应调整系数

    # 因子参数
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    reversal_windows: List[int] = field(default_factory=lambda: [3, 5, 8])
    volume_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    volatility_windows: List[int] = field(default_factory=lambda: [10, 20])

    # 策略参数
    max_positions: int = 10
    position_size_method: str = "risk_parity"  # risk_parity, equal_weight, kelly
    stop_loss_pct: float = 0.03  # 3%
    take_profit_pct: float = 0.06  # 6%
    max_holding_days: int = 5  # 最多持仓5天

    # 回测参数
    start_date: str = "2023-01-01"
    end_date: str = "2025-10-11"
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly
    benchmark: str = "HSI"  # 恒生指数

    # 统计参数
    significance_level: float = 0.05
    min_observations: int = 100
    confidence_interval: float = 0.95


class HKWeekdayEffectAnalyzer:
    """港股周内效应分析器"""

    def __init__(self, config: HKWeekdayEffectConfig):
        self.config = config
        self.runtime_config = DEFAULT_RUNTIME_CONFIG
        self.price_loader = PriceDataLoader(self.runtime_config)
        self.factor_loader = FactorScoreLoader(self.runtime_config)

    def load_hk_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """加载港股价格数据"""
        logger.info(f"加载港股数据: {len(symbols)} 个标的")

        data = {}
        for symbol in symbols:
            try:
                df = self.price_loader.load_price(symbol, "1day")
                if df is not None and len(df) > 0:
                    # 添加周内特征
                    df = self._add_weekday_features(df)
                    data[symbol] = df
                    logger.debug(f"{symbol} 数据加载成功: {len(df)} 行")
                else:
                    logger.warning(f"{symbol} 数据为空")
            except Exception as e:
                logger.error(f"{symbol} 数据加载失败: {e}")

        logger.info(f"成功加载 {len(data)} 个标的的数据")
        return data

    def _add_weekday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加周内效应特征"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 基础周内特征
        df["weekday"] = df.index.dayofweek  # 0=周一, 4=周五
        df["is_monday"] = (df["weekday"] == 0).astype(int)
        df["is_friday"] = (df["weekday"] == 4).astype(int)
        df["is_midweek"] = ((df["weekday"] >= 1) & (df["weekday"] <= 3)).astype(int)

        # 周内收益率特征
        df["daily_return"] = df["close"].pct_change()
        df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
        df["overnight_return"] = df["open"].shift(1) / df["close"].shift(1) - 1

        # 周内动量特征
        for window in self.config.momentum_windows:
            df[f"momentum_{window}d"] = df["close"].pct_change(window)
            df[f"weekday_momentum_{window}d"] = df.groupby("weekday")[
                "daily_return"
            ].transform(lambda x: x.rolling(window).mean())

        # 周内波动率特征
        for window in self.config.volatility_windows:
            df[f"volatility_{window}d"] = df["daily_return"].rolling(window).std()
            df[f"weekday_volatility_{window}d"] = df.groupby("weekday")[
                "daily_return"
            ].transform(lambda x: x.rolling(window).std())

        # 周内成交量特征
        for window in self.config.volume_windows:
            df[f"volume_ratio_{window}d"] = (
                df["volume"] / df["volume"].rolling(window).mean()
            )
            df[f"weekday_volume_ratio_{window}d"] = df.groupby("weekday")[
                "volume"
            ].transform(
                lambda x: x.rolling(window).mean() / x.rolling(window * 5).mean()
            )

        return df

    def calculate_weekday_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算周内交易因子"""
        factors = pd.DataFrame(index=df.index)

        # 1. 周内动量因子
        factors["weekday_momentum"] = np.where(
            df["weekday"].isin(self.config.optimal_entry_days),
            df["intraday_return"]
            * (1 + self.config.monday_effect_adjustment * df["is_monday"]),
            -df["intraday_return"]
            * (1 + self.config.friday_effect_adjustment * df["is_friday"]),
        )

        # 2. 周内反转因子
        for window in self.config.reversal_windows:
            prev_return = df["daily_return"].rolling(window).sum()
            factors[f"reversal_{window}d"] = -prev_return * np.where(
                df["weekday"].isin(self.config.optimal_exit_days), 1.2, 1.0
            )

        # 3. 周内成交量因子
        factors["volume_strength"] = df["volume_ratio_5d"] * np.where(
            df["is_midweek"], 1.1, 0.9
        )

        # 4. 周内波动率因子
        factors["volatility_adj"] = df["volatility_10d"] * np.where(
            df["weekday"].isin(self.config.optimal_entry_days), 0.8, 1.2
        )

        # 5. 周内复合因子
        factors["weekday_composite"] = (
            factors["weekday_momentum"] * 0.3
            + factors["reversal_3d"] * 0.25
            + factors["volume_strength"] * 0.25
            + factors["volatility_adj"] * 0.2
        )

        return factors

    def analyze_weekday_effects(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """分析周内效应统计特征"""
        logger.info("开始分析周内效应")

        all_returns = []
        weekday_returns = {i: [] for i in range(5)}

        for symbol, df in data.items():
            if "daily_return" in df.columns and "weekday" in df.columns:
                returns = df["daily_return"].dropna()
                weekdays = df["weekday"].dropna()

                all_returns.extend(returns.tolist())

                for day in range(5):
                    day_returns = returns[weekdays == day]
                    weekday_returns[day].extend(day_returns.tolist())

        # 统计分析
        results = {}

        # 整体统计
        all_returns_array = np.array(all_returns)
        results["overall"] = {
            "mean": np.mean(all_returns_array),
            "std": np.std(all_returns_array),
            "skewness": stats.skew(all_returns_array),
            "kurtosis": stats.kurtosis(all_returns_array),
            "count": len(all_returns_array),
        }

        # 分日统计
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        for day, name in enumerate(weekday_names):
            day_returns = np.array(weekday_returns[day])
            if len(day_returns) > 0:
                t_stat, p_value = stats.ttest_1samp(day_returns, 0)

                results[f"day_{day}"] = {
                    "name": name,
                    "mean": np.mean(day_returns),
                    "std": np.std(day_returns),
                    "mean_annualized": np.mean(day_returns) * 252,
                    "sharpe": np.mean(day_returns) / np.std(day_returns) * np.sqrt(252),
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "significant": p_value < self.config.significance_level,
                    "count": len(day_returns),
                }

        logger.info("周内效应分析完成")
        return results


class HKWeekdayStrategyBacktester:
    """港股周内策略回测器"""

    def __init__(self, config: HKWeekdayEffectConfig):
        self.config = config
        self.analyzer = HKWeekdayEffectAnalyzer(config)

    def calculate_transaction_costs(
        self, positions: pd.Series, prices: pd.Series
    ) -> pd.Series:
        """计算交易成本（港股特有）"""
        # 计算交易金额
        trade_values = positions.diff().abs() * prices

        # 佣金（最低5港币）
        commissions = np.maximum(
            trade_values * self.config.commission_rate, self.config.min_commission
        )

        # 印花税（仅卖出时收取）
        sell_trades = positions.diff() < 0
        stamp_duties = np.where(
            sell_trades, trade_values * self.config.stamp_duty_rate, 0
        )

        # 滑点成本
        slippage_costs = trade_values * self.config.slippage_bps / 10000

        total_costs = commissions + stamp_duties + slippage_costs
        return total_costs

    def generate_weekday_signals(
        self, df: pd.DataFrame, factors: pd.DataFrame
    ) -> pd.DataFrame:
        """基于周内效应生成交易信号"""
        signals = pd.DataFrame(index=df.index)

        # 基础入场条件
        entry_conditions = df["weekday"].isin(self.config.optimal_entry_days) & (
            factors["weekday_composite"]
            > factors["weekday_composite"].rolling(20).quantile(0.7)
        )

        # 出场条件
        exit_conditions = df["weekday"].isin(self.config.optimal_exit_days) | (
            factors["weekday_composite"]
            < factors["weekday_composite"].rolling(20).quantile(0.3)
        )

        # 应用时间止盈止损
        holding_period = 0
        current_position = False

        entries = []
        exits = []

        for i in range(len(df)):
            if entry_conditions.iloc[i] and not current_position:
                entries.append(True)
                exits.append(False)
                current_position = True
                holding_period = 0
            elif current_position:
                holding_period += 1

                # 止损止盈检查
                if holding_period >= self.config.max_holding_days:
                    entries.append(False)
                    exits.append(True)
                    current_position = False
                    holding_period = 0
                elif exit_conditions.iloc[i]:
                    entries.append(False)
                    exits.append(True)
                    current_position = False
                    holding_period = 0
                else:
                    entries.append(False)
                    exits.append(False)
            else:
                entries.append(False)
                exits.append(False)

        signals["entries"] = entries
        signals["exits"] = exits

        return signals

    def backtest_single_symbol(self, symbol: str, df: pd.DataFrame) -> Dict:
        """单个标的回测"""
        logger.info(f"开始回测 {symbol}")

        # 计算因子
        factors = self.analyzer.calculate_weekday_factors(df)

        # 生成信号
        signals = self.generate_weekday_signals(df, factors)

        # 使用VectorBT回测
        portfolio = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=signals["entries"],
            exits=signals["exits"],
            init_cash=100000,
            fees=self.config.commission_rate,
            slippage=self.config.slippage_bps / 10000,
        )

        # 计算额外成本（印花税）
        positions = portfolio.positions.records.copy()
        if not positions.empty:
            # 计算印花税
            sell_values = (
                positions[positions["size"] < 0]["size"].abs()
                * positions[positions["size"] < 0]["price"]
            )
            stamp_duty_total = sell_values.sum() * self.config.stamp_duty_rate

            # 调整最终价值
            final_value = portfolio.value() - stamp_duty_total
        else:
            final_value = portfolio.value()

        # 计算绩效指标
        returns = portfolio.returns()
        benchmark_returns = df["daily_return"].fillna(0)

        # 基础指标
        total_return = (final_value / 100000 - 1) * 100
        annual_return = returns.mean() * 252 * 100
        annual_volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        max_drawdown = portfolio.drawdowns.max() * 100

        # 相对指标
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252) * 100
        information_ratio = (
            excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        )
        beta = (
            np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            if np.var(benchmark_returns) > 0
            else 1
        )
        alpha = (returns.mean() - beta * benchmark_returns.mean()) * 252 * 100

        # 交易统计
        trade_count = len(portfolio.trades.records)
        win_rate = (
            (portfolio.trades.records["pnl"] > 0).mean() * 100 if trade_count > 0 else 0
        )
        avg_trade = portfolio.trades.records["pnl"].mean() if trade_count > 0 else 0
        profit_factor = (
            abs(
                portfolio.trades.records[portfolio.trades.records["pnl"] > 0][
                    "pnl"
                ].sum()
                / portfolio.trades.records[portfolio.trades.records["pnl"] < 0][
                    "pnl"
                ].sum()
            )
            if trade_count > 0
            else 1
        )

        results = {
            "symbol": symbol,
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "alpha": alpha,
            "beta": beta,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
            "trade_count": trade_count,
            "win_rate": win_rate,
            "avg_trade": avg_trade,
            "profit_factor": profit_factor,
            "final_value": final_value,
        }

        logger.info(
            f"{symbol} 回测完成: 总收益率 {total_return:.2f}%, 夏普比率 {sharpe_ratio:.2f}"
        )
        return results

    def backtest_portfolio(self, symbols: List[str]) -> Dict:
        """组合回测"""
        logger.info(f"开始组合回测: {len(symbols)} 个标的")

        # 加载数据
        data = self.analyzer.load_hk_data(symbols)

        if not data:
            logger.error("未能加载任何有效数据")
            return {}

        # 分析周内效应
        weekday_effects = self.analyzer.analyze_weekday_effects(data)

        # 单独回测每个标的
        individual_results = []
        for symbol, df in data.items():
            try:
                result = self.backtest_single_symbol(symbol, df)
                individual_results.append(result)
            except Exception as e:
                logger.error(f"{symbol} 回测失败: {e}")

        # 组合汇总
        if not individual_results:
            logger.error("没有成功的回测结果")
            return {}

        results_df = pd.DataFrame(individual_results)

        # 等权重组合
        portfolio_return = results_df["annual_return"].mean()
        portfolio_volatility = (
            results_df["annual_volatility"].mean()
            / np.sqrt(len(results_df))
            * np.sqrt(len(results_df))
        )
        portfolio_sharpe = (
            portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        )
        portfolio_alpha = results_df["alpha"].mean()
        portfolio_beta = results_df["beta"].mean()

        summary = {
            "weekday_effects": weekday_effects,
            "individual_results": results_df.to_dict("records"),
            "portfolio_summary": {
                "symbols_count": len(results_df),
                "portfolio_return": portfolio_return,
                "portfolio_volatility": portfolio_volatility,
                "portfolio_sharpe": portfolio_sharpe,
                "portfolio_alpha": portfolio_alpha,
                "portfolio_beta": portfolio_beta,
                "avg_win_rate": results_df["win_rate"].mean(),
                "avg_profit_factor": results_df["profit_factor"].mean(),
                "total_trades": results_df["trade_count"].sum(),
            },
        }

        logger.info(
            f"组合回测完成: 组合收益率 {portfolio_return:.2f}%, 夏普比率 {portfolio_sharpe:.2f}"
        )
        return summary


def main():
    """主函数 - 港股周内交易策略评估"""

    # 配置参数
    config = HKWeekdayEffectConfig()

    # 港股主要标的
    hk_symbols = [
        "0700.HK",  # 腾讯控股
        "9988.HK",  # 阿里巴巴
        "0005.HK",  # 汇丰控股
        "0941.HK",  # 中国移动
        "1299.HK",  # 友邦保险
        "2318.HK",  # 中国平安
        "1398.HK",  # 工商银行
        "0002.HK",  # 中电控股
        "0388.HK",  # 港交所
        "1810.HK",  # 小米集团
    ]

    # 创建回测器
    backtester = HKWeekdayStrategyBacktester(config)

    # 执行回测
    results = backtester.backtest_portfolio(hk_symbols)

    # 打印结果
    print("\n" + "=" * 80)
    print("港股周内交易策略回测结果")
    print("=" * 80)

    if "weekday_effects" in results:
        print("\n周内效应分析:")
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        for day in range(5):
            day_key = f"day_{day}"
            if day_key in results["weekday_effects"]:
                day_stats = results["weekday_effects"][day_key]
                print(
                    f"  {day_stats['name']}: "
                    f"收益率 {day_stats['mean']*100:.3f}% "
                    f"(年化 {day_stats['mean_annualized']*100:.1f}%), "
                    f"夏普 {day_stats['sharpe']:.2f}, "
                    f"显著性 {'是' if day_stats['significant'] else '否'}"
                )

    if "portfolio_summary" in results:
        portfolio = results["portfolio_summary"]
        print(f"\n组合绩效 ({portfolio['symbols_count']} 个标的):")
        print(f"  年化收益率: {portfolio['portfolio_return']:.2f}%")
        print(f"  年化波动率: {portfolio['portfolio_volatility']:.2f}%")
        print(f"  夏普比率: {portfolio['portfolio_sharpe']:.2f}")
        print(f"  Alpha: {portfolio['portfolio_alpha']:.2f}%")
        print(f"  Beta: {portfolio['portfolio_beta']:.2f}")
        print(f"  胜率: {portfolio['avg_win_rate']:.1f}%")
        print(f"  盈亏比: {portfolio['avg_profit_factor']:.2f}")
        print(f"  总交易次数: {portfolio['total_trades']}")

    print("\n" + "=" * 80)
    print("策略建议:")
    print("=" * 80)
    print("1. 最佳交易日: 周一至周三（基于历史统计分析）")
    print("2. 止损设置: 3%（考虑港股波动性）")
    print("3. 止盈设置: 6%（风险收益比1:2）")
    print("4. 最大持仓: 5个交易日（避免周内风险累积）")
    print("5. 仓位管理: 风险平价或等权重分配")
    print("6. 成本控制: 注意印花税对高频交易的影响")
    print("7. 因子权重: 动量30% + 反转25% + 成交量25% + 波动率20%")


if __name__ == "__main__":
    main()
