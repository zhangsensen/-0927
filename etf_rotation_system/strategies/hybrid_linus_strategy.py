"""
Linus混合策略回测引擎
战略层(月度,长周期因子) + 战术层(周度,短周期因子)

设计原理:
1. 战略层: 低换手,持有趋势强劲ETF (70%资金)
2. 战术层: 高灵敏,捕捉短期alpha (30%资金)
3. 两层独立调仓,组合再平衡
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class HybridLinusStrategy:
    """混合策略引擎"""

    def __init__(self, config_path: str):
        """初始化

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.strategic_config = self.config["strategic"]
        self.tactical_config = self.config["tactical"]
        self.costs = self.config["costs"]

        # 初始化结果容器
        self.strategic_signals = None
        self.tactical_signals = None
        self.combined_positions = None
        self.trades = []
        self.metrics = {}

    def _load_config(self, path: str) -> dict:
        """加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_factor_data(self, panel_path: str) -> pd.DataFrame:
        """加载因子面板

        Args:
            panel_path: 面板文件路径 (Parquet)

        Returns:
            因子数据 (trade_date, code, factor1, factor2, ...)
        """
        df = pd.read_parquet(panel_path)
        logger.info(f"✅ 加载因子面板: {len(df)} 行, {df['code'].nunique()} 只标的")
        return df

    def generate_strategic_signals(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """生成战略层信号

        Args:
            factor_df: 因子数据

        Returns:
            信号DataFrame (trade_date, code, score, rank)
        """
        factors = self.strategic_config["factors"]
        weights = self.strategic_config["factor_weights"]
        top_n = self.strategic_config["top_n"]

        signals = []

        for date in factor_df["trade_date"].unique():
            date_df = factor_df[factor_df["trade_date"] == date].copy()

            # 计算综合分数
            date_df["score"] = 0.0
            for factor in factors:
                if factor in date_df.columns:
                    # 标准化因子值
                    factor_values = date_df[factor].values
                    if len(factor_values) > 1:
                        factor_std = (factor_values - np.nanmean(factor_values)) / (
                            np.nanstd(factor_values) + 1e-8
                        )
                    else:
                        factor_std = factor_values

                    # 加权累加
                    date_df["score"] += weights.get(factor, 0) * factor_std

            # 排名
            date_df["rank"] = date_df["score"].rank(ascending=False, method="first")

            # Top N
            top_df = date_df.nsmallest(top_n, "rank")[
                ["trade_date", "code", "score", "rank"]
            ]
            signals.append(top_df)

        result = pd.concat(signals, ignore_index=True)
        logger.info(f"🎯 战略层信号: {len(result)} 条, Top {top_n}")
        return result

    def generate_tactical_signals(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """生成战术层信号

        Args:
            factor_df: 因子数据

        Returns:
            信号DataFrame (trade_date, code, score, rank)
        """
        factors = self.tactical_config["factors"]
        weights = self.tactical_config["factor_weights"]
        top_n = self.tactical_config["top_n"]

        signals = []

        for date in factor_df["trade_date"].unique():
            date_df = factor_df[factor_df["trade_date"] == date].copy()

            # 计算综合分数
            date_df["score"] = 0.0
            for factor in factors:
                if factor in date_df.columns:
                    # 标准化
                    factor_values = date_df[factor].values
                    if len(factor_values) > 1:
                        factor_std = (factor_values - np.nanmean(factor_values)) / (
                            np.nanstd(factor_values) + 1e-8
                        )
                    else:
                        factor_std = factor_values

                    # 加权
                    date_df["score"] += weights.get(factor, 0) * factor_std

            # 排名
            date_df["rank"] = date_df["score"].rank(ascending=False, method="first")

            # Top N
            top_df = date_df.nsmallest(top_n, "rank")[
                ["trade_date", "code", "score", "rank"]
            ]
            signals.append(top_df)

        result = pd.concat(signals, ignore_index=True)
        logger.info(f"⚡ 战术层信号: {len(result)} 条, Top {top_n}")
        return result

    def combine_positions(
        self,
        strategic_signals: pd.DataFrame,
        tactical_signals: pd.DataFrame,
        price_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """组合两层持仓

        Args:
            strategic_signals: 战略层信号
            tactical_signals: 战术层信号
            price_df: 价格数据 (trade_date, code, close)

        Returns:
            持仓DataFrame (trade_date, code, strategic_weight, tactical_weight, total_weight)
        """
        strategic_weight = self.strategic_config["portfolio_weight"]
        tactical_weight = self.tactical_config["portfolio_weight"]

        # 按日期合并
        all_dates = sorted(
            set(strategic_signals["trade_date"].unique())
            | set(tactical_signals["trade_date"].unique())
        )

        positions = []

        for date in all_dates:
            # 战略层持仓
            strategic_holdings = strategic_signals[
                strategic_signals["trade_date"] == date
            ][["code"]].copy()
            strategic_holdings["strategic_weight"] = strategic_weight / len(
                strategic_holdings
            )

            # 战术层持仓
            tactical_holdings = tactical_signals[
                tactical_signals["trade_date"] == date
            ][["code"]].copy()
            tactical_holdings["tactical_weight"] = tactical_weight / len(
                tactical_holdings
            )

            # 合并
            date_positions = pd.merge(
                strategic_holdings, tactical_holdings, on="code", how="outer"
            ).fillna(0)

            date_positions["trade_date"] = date
            date_positions["total_weight"] = (
                date_positions["strategic_weight"] + date_positions["tactical_weight"]
            )

            positions.append(date_positions)

        result = pd.concat(positions, ignore_index=True)
        logger.info(f"📊 组合持仓: {len(result)} 条")
        return result

    def calculate_returns(
        self, positions: pd.DataFrame, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """计算收益率

        Args:
            positions: 持仓数据
            price_df: 价格数据 (trade_date, code, close, ret_1d)

        Returns:
            收益DataFrame (trade_date, portfolio_ret, cumulative_ret)
        """
        # 合并价格
        merged = pd.merge(
            positions,
            price_df[["trade_date", "code", "ret_1d"]],
            on=["trade_date", "code"],
            how="left",
        )

        # 计算每日组合收益
        daily_ret = (
            merged.groupby("trade_date")
            .apply(lambda x: (x["total_weight"] * x["ret_1d"]).sum())
            .reset_index(name="portfolio_ret")
        )

        # 累计收益
        daily_ret["cumulative_ret"] = (1 + daily_ret["portfolio_ret"]).cumprod() - 1

        logger.info(f"📈 计算收益: {len(daily_ret)} 天")
        return daily_ret

    def calculate_metrics(self, returns_df: pd.DataFrame) -> Dict:
        """计算性能指标

        Args:
            returns_df: 收益数据

        Returns:
            指标字典
        """
        rets = returns_df["portfolio_ret"].values
        cum_ret = returns_df["cumulative_ret"].iloc[-1]

        # 年化收益
        n_days = len(rets)
        annual_ret = (1 + cum_ret) ** (252 / n_days) - 1

        # 年化波动
        annual_vol = np.std(rets) * np.sqrt(252)

        # Sharpe
        rf = self.config["backtest"]["risk_free_rate"]
        sharpe = (annual_ret - rf) / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cumulative = (1 + returns_df["portfolio_ret"]).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_dd = drawdowns.min()

        # Calmar
        calmar = annual_ret / abs(max_dd) if max_dd < 0 else 0

        # 胜率
        win_rate = (rets > 0).sum() / len(rets)

        metrics = {
            "cumulative_return": cum_ret,
            "annual_return": annual_ret,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "n_days": n_days,
        }

        logger.info(f"✅ 性能指标计算完成")
        return metrics

    def run_backtest(
        self, factor_df: pd.DataFrame, price_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """运行完整回测

        Args:
            factor_df: 因子数据
            price_df: 价格数据

        Returns:
            (收益曲线, 性能指标)
        """
        logger.info("=" * 60)
        logger.info("🚀 Linus混合策略回测启动")
        logger.info("=" * 60)

        # 1. 生成信号
        self.strategic_signals = self.generate_strategic_signals(factor_df)
        self.tactical_signals = self.generate_tactical_signals(factor_df)

        # 2. 组合持仓
        self.combined_positions = self.combine_positions(
            self.strategic_signals, self.tactical_signals, price_df
        )

        # 3. 计算收益
        returns_df = self.calculate_returns(self.combined_positions, price_df)

        # 4. 计算指标
        self.metrics = self.calculate_metrics(returns_df)

        # 5. 输出结果
        logger.info("=" * 60)
        logger.info("📊 回测结果汇总")
        logger.info("=" * 60)
        logger.info(f"累计收益: {self.metrics['cumulative_return']:.2%}")
        logger.info(f"年化收益: {self.metrics['annual_return']:.2%}")
        logger.info(f"年化波动: {self.metrics['annual_volatility']:.2%}")
        logger.info(f"Sharpe比率: {self.metrics['sharpe_ratio']:.3f}")
        logger.info(f"最大回撤: {self.metrics['max_drawdown']:.2%}")
        logger.info(f"Calmar比率: {self.metrics['calmar_ratio']:.3f}")
        logger.info(f"胜率: {self.metrics['win_rate']:.2%}")
        logger.info("=" * 60)

        return returns_df, self.metrics

    def save_results(self, output_dir: str):
        """保存结果

        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存信号
        if self.config["output"]["save_signals"]:
            self.strategic_signals.to_csv(
                output_path / "strategic_signals.csv", index=False
            )
            self.tactical_signals.to_csv(
                output_path / "tactical_signals.csv", index=False
            )
            logger.info(f"💾 信号已保存")

        # 保存持仓
        if self.config["output"]["save_positions"]:
            self.combined_positions.to_csv(output_path / "positions.csv", index=False)
            logger.info(f"💾 持仓已保存")

        # 保存指标
        if self.config["output"]["save_metrics"]:
            metrics_df = pd.DataFrame([self.metrics])
            metrics_df.to_csv(output_path / "metrics.csv", index=False)
            logger.info(f"💾 指标已保存")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # 测试
    config_path = "../config/hybrid_strategy_config.yaml"
    strategy = HybridLinusStrategy(config_path)

    # 加载数据 (示例)
    # factor_df = strategy.load_factor_data("path/to/panel.parquet")
    # price_df = pd.read_parquet("path/to/prices.parquet")

    # returns_df, metrics = strategy.run_backtest(factor_df, price_df)
    # strategy.save_results("data/results/hybrid_strategy")

    logger.info("✅ 混合策略引擎初始化完成")
