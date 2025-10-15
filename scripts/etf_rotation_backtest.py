#!/usr/bin/env python3
"""ETF轮动回测引擎 - 2020-2025全周期

执行口径：
- T日截面 → T+1开盘建仓 → 次月末平仓
- 费用：万2.5（双边5bp）+ 10bp滑点
- |Δ权重|≤1%忽略

输出：
- 年化/回撤/夏普/月胜率/换手
- 极端月归因
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class ETFRotationBacktest:
    """ETF轮动回测引擎"""

    def __init__(
        self,
        commission_rate=0.00025,  # 万2.5（单边）
        slippage=0.001,  # 10bp滑点
        min_weight_change=0.01,
    ):  # 最小权重变化1%
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.min_weight_change = min_weight_change

    def load_data(self, panel_file, price_file, production_factors_file):
        """加载数据"""
        logger.info("加载数据...")

        # 加载因子面板
        self.panel = pd.read_parquet(panel_file)
        logger.info(f"  因子面板: {self.panel.shape}")

        # 加载价格数据（从面板中提取或单独加载）
        # 这里假设价格数据在原始数据目录
        self.prices = self._load_prices(price_file)
        logger.info(f"  价格数据: {self.prices.shape}")

        # 加载生产因子列表
        with open(production_factors_file) as f:
            self.production_factors = [line.strip() for line in f if line.strip()]
        logger.info(f"  生产因子: {len(self.production_factors)}个")

        # 筛选生产因子
        available_factors = [
            f for f in self.production_factors if f in self.panel.columns
        ]
        self.panel = self.panel[available_factors]
        logger.info(f"  可用因子: {len(available_factors)}个")

    def _load_prices(self, price_dir):
        """加载价格数据"""
        from pathlib import Path

        logger.info("  从原始数据目录加载价格...")

        price_dir = Path(price_dir)
        if not price_dir.exists():
            logger.error(f"  价格目录不存在: {price_dir}")
            return None

        # 加载所有ETF的价格数据
        all_prices = []
        for file in price_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(file)
                # 标准化列名
                df["symbol"] = df["ts_code"]
                df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                df["volume"] = df["vol"]

                # 选择需要的列
                df = df[
                    [
                        "symbol",
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "amount",
                    ]
                ]
                all_prices.append(df)
            except Exception as e:
                logger.warning(f"  加载{file.name}失败: {e}")

        if len(all_prices) == 0:
            logger.error("  未能加载任何价格数据")
            return None

        # 合并所有数据
        prices = pd.concat(all_prices, ignore_index=True)
        prices = prices.set_index(["symbol", "date"]).sort_index()

        logger.info(
            f"  ✅ 加载完成: {len(prices)}行，{prices.index.get_level_values('symbol').nunique()}个ETF"
        )

        return prices

    def calculate_scores(self):
        """计算综合得分（简单等权排名）"""
        logger.info("\n计算因子得分...")

        # 对每个因子进行排名（百分位）
        scores = self.panel.groupby(level="date").rank(pct=True)

        # 等权平均
        self.scores = scores.mean(axis=1)

        logger.info(f"  得分范围: [{self.scores.min():.4f}, {self.scores.max():.4f}]")

    def generate_signals(self, top_n=5, rebalance_freq="ME"):
        """生成交易信号

        Args:
            top_n: 每期持仓数量
            rebalance_freq: 调仓频率（'ME'=月末）
        """
        logger.info(f"\n生成交易信号（Top {top_n}，{rebalance_freq}调仓）...")

        # 按月分组
        monthly_scores = self.scores.groupby(
            [pd.Grouper(level="date", freq=rebalance_freq), pd.Grouper(level="symbol")]
        ).last()

        # 每月选择Top N
        signals = []
        for date, group in monthly_scores.groupby(level=0):
            top_symbols = (
                group.nlargest(top_n).index.get_level_values("symbol").tolist()
            )

            for symbol in top_symbols:
                signals.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "weight": 1.0 / top_n,  # 等权
                        "score": group.loc[(date, symbol)],
                    }
                )

        self.signals = pd.DataFrame(signals)
        logger.info(f"  信号数量: {len(self.signals)}")
        logger.info(f"  调仓次数: {self.signals['date'].nunique()}")

    def run_backtest(self, exec_mode: str = "next_open_close"):
        """运行回测"""
        logger.info("\n运行回测...")
        logger.info(f"执行口径: {exec_mode}")

        # 初始化
        initial_capital = 1000000  # 100万
        positions = {}  # {symbol: shares}
        cash = initial_capital
        cumulative_transaction_cost = 0.0  # 累计交易成本
        records = []

        # 获取所有调仓日期
        rebalance_dates = sorted(self.signals["date"].unique())

        for i, rebal_date in enumerate(rebalance_dates):
            logger.info(f"\n  [{i+1}/{len(rebalance_dates)}] 调仓日期: {rebal_date}")

            # 获取目标持仓
            target_signals = self.signals[self.signals["date"] == rebal_date]
            target_weights = dict(
                zip(target_signals["symbol"], target_signals["weight"])
            )

            # T+1执行（下一个交易日开盘）
            execution_date = self._get_next_trading_day(rebal_date)
            if execution_date is None:
                logger.warning("    无法找到执行日期，跳过")
                continue

            # 获取执行价格（根据执行口径）
            exec_price_field = "open" if exec_mode == "next_open_close" else "close"
            execution_prices = self._get_prices(execution_date, exec_price_field)
            if execution_prices is None or len(execution_prices) == 0:
                logger.warning("    无法获取价格，跳过")
                continue

            # 清算旧持仓（转为现金）
            for symbol, shares in positions.items():
                if symbol in execution_prices:
                    cash += shares * execution_prices[symbol]

            # 当前组合价值 = 现金
            current_value = cash

            # 执行调仓（建立新持仓）
            new_positions = {}
            transaction_cost = 0

            for symbol, target_weight in target_weights.items():
                if symbol not in execution_prices:
                    continue

                target_value = current_value * target_weight
                target_shares = (
                    int(target_value / execution_prices[symbol] / 100) * 100
                )  # 整手

                if target_shares == 0:
                    continue

                # 买入
                trade_value = target_shares * execution_prices[symbol]
                cost = trade_value * (self.commission_rate + self.slippage)
                transaction_cost += cost

                cash -= trade_value + cost
                new_positions[symbol] = target_shares

            positions = new_positions
            cumulative_transaction_cost += transaction_cost  # 累加成本

            # 使用执行日收盘价估值组合（现金 + Σ(份额×收盘价)）
            close_prices = self._get_prices(execution_date, "close") or {}
            portfolio_value_eod = cash + sum(
                positions.get(sym, 0)
                * close_prices.get(sym, execution_prices.get(sym, 0.0))
                for sym in positions.keys()
            )

            # 记录（增加持仓快照）
            records.append(
                {
                    "date": execution_date,
                    "portfolio_value": portfolio_value_eod,
                    "cash": cash,
                    "transaction_cost": transaction_cost,
                    "cumulative_cost": cumulative_transaction_cost,  # 累计成本
                    "num_positions": len(positions),
                    "positions": positions.copy(),  # 持仓份额快照
                    "execution_prices": execution_prices.copy(),  # 建仓价格
                }
            )

            logger.info(f"    组合价值: {portfolio_value_eod:,.0f}")
            logger.info(f"    持仓数: {len(positions)}")
            logger.info(f"    交易成本: {transaction_cost:,.0f}")

        self.backtest_results = pd.DataFrame(records)
        logger.info(f"\n✅ 回测完成，共{len(records)}个调仓周期")

    def _get_next_trading_day(self, date):
        """获取下一个交易日"""
        dates = self.panel.index.get_level_values("date").unique()
        future_dates = dates[dates > date]
        return future_dates[0] if len(future_dates) > 0 else None

    def _get_prices(self, date, price_type="open"):
        """获取指定日期的价格"""
        try:
            # 从已加载的价格数据中提取
            if self.prices is None:
                return None

            # 标准化日期（去除时间部分）
            date_normalized = pd.Timestamp(date).normalize()

            # 获取指定日期的价格
            try:
                date_prices = self.prices.xs(date_normalized, level="date")
            except KeyError:
                # 如果精确日期不存在，尝试找最近的日期
                available_dates = self.prices.index.get_level_values("date").unique()
                closest_date = available_dates[available_dates >= date_normalized][0]
                date_prices = self.prices.xs(closest_date, level="date")

            # 返回字典 {symbol: price}
            return dict(zip(date_prices.index, date_prices[price_type]))
        except Exception as e:
            logger.warning(f"  获取{date}价格失败: {e}")
            return None

    def generate_daily_equity_curve(self):
        """生成日频权益曲线（真实持仓逐日标价）"""
        logger.info("\n生成日频权益曲线...")

        if self.backtest_results is None or len(self.backtest_results) == 0:
            logger.warning("  ⚠️  无回测结果，跳过")
            return None

        # 获取所有交易日
        all_dates = sorted(self.prices.index.get_level_values("date").unique())

        # 初始化日频记录
        daily_records = []

        # 遍历调仓周期
        for i in range(len(self.backtest_results)):
            rebal_record = self.backtest_results.iloc[i]
            rebal_date = rebal_record["date"]

            # 确定持仓区间：当前调仓日 → 下一调仓日（或结束）
            if i < len(self.backtest_results) - 1:
                next_rebal_date = self.backtest_results.iloc[i + 1]["date"]
            else:
                next_rebal_date = all_dates[-1] + pd.Timedelta(days=1)  # 包含最后一天

            # 获取该区间的持仓份额（从调仓记录提取）
            positions = rebal_record.get("positions", {})
            cash = rebal_record["cash"]
            cumulative_cost = rebal_record.get("cumulative_cost", 0.0)  # 累计成本

            if not positions:
                # 无持仓，跳过
                continue

            # 遍历该区间的每个交易日
            period_dates = [d for d in all_dates if rebal_date <= d < next_rebal_date]

            for date in period_dates:
                # 获取当日收盘价
                try:
                    date_prices = self.prices.xs(date, level="date")["close"]
                except KeyError:
                    continue

                # 计算持仓市值：Σ(份额 × 当日收盘价)
                holdings_value = 0.0
                for symbol, shares in positions.items():
                    if symbol in date_prices.index:
                        holdings_value += shares * date_prices[symbol]

                # 组合总价值 = 持仓市值 + 现金 - 累计成本
                portfolio_value = holdings_value + cash - cumulative_cost

                daily_records.append(
                    {
                        "date": date,
                        "portfolio_value": portfolio_value,
                        "holdings_value": holdings_value,
                        "cash": cash,
                    }
                )

        if not daily_records:
            logger.warning("  ⚠️  无有效日频数据")
            return None

        self.daily_equity = (
            pd.DataFrame(daily_records)
            .drop_duplicates("date")
            .set_index("date")
            .sort_index()
        )
        logger.info(f"  ✅ 生成{len(self.daily_equity)}个交易日的权益曲线")
        logger.info(f"  起始价值: {self.daily_equity['portfolio_value'].iloc[0]:,.0f}")
        logger.info(f"  结束价值: {self.daily_equity['portfolio_value'].iloc[-1]:,.0f}")

        return self.daily_equity

    def calculate_metrics(self):
        """计算回测指标"""
        logger.info("\n计算回测指标...")

        # 优先使用日频权益曲线（如果存在）
        if hasattr(self, "daily_equity") and self.daily_equity is not None:
            df = self.daily_equity.reset_index()
            df["returns"] = df["portfolio_value"].pct_change()
            logger.info("  使用日频权益曲线计算指标")
        else:
            df = self.backtest_results.copy()
            df["returns"] = df["portfolio_value"].pct_change()
            logger.info("  使用调仓点权益计算指标（稀疏）")

        # 年化收益
        total_return = (
            df["portfolio_value"].iloc[-1] / df["portfolio_value"].iloc[0]
        ) - 1
        years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1

        # 最大回撤
        cummax = df["portfolio_value"].cummax()
        drawdown = (df["portfolio_value"] - cummax) / cummax
        max_drawdown = drawdown.min()

        # 夏普比率（假设无风险利率3%）
        risk_free_rate = 0.03
        excess_returns = df["returns"].dropna() - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

        # 月胜率
        monthly_returns = (
            df.set_index("date")["returns"]
            .resample("ME")
            .apply(lambda x: (1 + x).prod() - 1)
        )
        win_rate = (monthly_returns > 0).sum() / len(monthly_returns)

        # 换手率
        total_cost = self.backtest_results["transaction_cost"].sum()
        avg_value = df["portfolio_value"].mean()
        turnover = total_cost / avg_value / years

        # 极端月识别（收益率 top3 / bottom3）
        monthly_returns_sorted = monthly_returns.sort_values()
        # 转为可JSON序列化的字典（日期键转字符串）
        worst3 = monthly_returns_sorted.head(3)
        best3 = monthly_returns_sorted.tail(3)
        extreme_months = {
            "worst_3": {
                str(k.date() if hasattr(k, "date") else k): float(v)
                for k, v in worst3.items()
            },
            "best_3": {
                str(k.date() if hasattr(k, "date") else k): float(v)
                for k, v in best3.items()
            },
        }

        metrics = {
            "年化收益": f"{annual_return:.2%}",
            "最大回撤": f"{max_drawdown:.2%}",
            "夏普比率": f"{sharpe_ratio:.2f}",
            "月胜率": f"{win_rate:.2%}",
            "年化换手": f"{turnover:.2f}",
            "总收益": f"{total_return:.2%}",
            "回测年数": f"{years:.2f}",
            "极端月": extreme_months,
        }

        logger.info("\n回测指标:")
        for key, value in metrics.items():
            if key != "极端月":
                logger.info(f"  {key}: {value}")

        logger.info("\n极端月:")
        logger.info("  最差3个月:")
        for date, ret in extreme_months["worst_3"].items():
            logger.info(f"    {date}: {ret:.2%}")
        logger.info("  最佳3个月:")
        for date, ret in extreme_months["best_3"].items():
            logger.info(f"    {date}: {ret:.2%}")

        return metrics


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("ETF轮动回测 - 2020-2025全周期")
    logger.info("=" * 80)

    # 解析参数
    parser = argparse.ArgumentParser(description="ETF轮动回测")
    parser.add_argument(
        "--exec",
        dest="exec_mode",
        default="next_open_close",
        choices=["next_open_close", "next_close_close"],
        help="执行口径",
    )
    parser.add_argument("--start", default=None, help="起始日期(YYYY-MM-DD，可选)")
    parser.add_argument("--end", default=None, help="结束日期(YYYY-MM-DD，可选)")
    parser.add_argument("--panel-file", required=True, help="因子面板文件路径")
    parser.add_argument("--price-dir", default="raw/ETF/daily", help="价格数据目录")
    parser.add_argument("--production-factors", required=True, help="生产因子列表文件")
    parser.add_argument(
        "--output-dir", default="factor_output/etf_rotation_production", help="输出目录"
    )
    args = parser.parse_args()

    # 初始化回测引擎
    backtest = ETFRotationBacktest(
        commission_rate=0.00025,  # 万2.5
        slippage=0.001,  # 10bp
        min_weight_change=0.01,  # 1%
    )

    # 文件路径
    panel_file = Path(args.panel_file)
    price_dir = Path(args.price_dir)
    production_factors_file = Path(args.production_factors)

    # 检查文件
    if not panel_file.exists():
        logger.error(f"❌ 面板文件不存在: {panel_file}")
        return False

    if not production_factors_file.exists():
        logger.error(f"❌ 生产因子文件不存在: {production_factors_file}")
        return False

    try:
        # 加载数据
        backtest.load_data(panel_file, price_dir, production_factors_file)

        # 计算得分
        backtest.calculate_scores()

        # 生成信号
        backtest.generate_signals(top_n=5, rebalance_freq="ME")

        # 运行回测
        backtest.run_backtest(exec_mode=args.exec_mode)

        # 生成日频权益曲线
        backtest.generate_daily_equity_curve()

        # 计算指标
        metrics = backtest.calculate_metrics()

        # 落盘报告
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        backtest.backtest_results.to_parquet(out_dir / "backtest_results.parquet")
        if hasattr(backtest, "daily_equity") and backtest.daily_equity is not None:
            try:
                backtest.daily_equity.to_parquet(out_dir / "daily_equity.parquet")
            except Exception:
                pass
        with open(out_dir / "backtest_metrics.json", "w") as f:
            json.dump(
                {"exec_mode": args.exec_mode, "metrics": metrics},
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info("\n" + "=" * 80)
        logger.info("✅ 回测框架已完成")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"❌ 回测失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
