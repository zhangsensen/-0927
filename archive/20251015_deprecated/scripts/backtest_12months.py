#!/usr/bin/env python3
"""ETF轮动12个月回测（修正时序：T截面 → T+1开盘 → 下月末收盘）"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from etf_rotation.portfolio import PortfolioBuilder
from etf_rotation.scorer import ETFScorer
from etf_rotation.universe_manager import ETFUniverseManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_monthly_rotation(
    trade_dates: list, panel_file: str, config_path: str, config_file: str = None
):
    """运行月度轮动"""
    # 加载配置
    config_file = config_file or config_path  # 使用传入的config_file或默认的config_path
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # 加载面板
    panel = pd.read_parquet(panel_file)
    logger.info(f"面板形状: {panel.shape}")

    # 初始化模块
    universe_mgr = ETFUniverseManager("etf_download_manager/config/etf_config.yaml")

    # 根据配置结构初始化评分器
    if "factor_weights" in config:
        # 旧式配置（权重直接定义）
        scorer = ETFScorer(weights=config["factor_weights"])
    else:
        # 新式配置（从因子集文件加载）
        scorer = ETFScorer(
            config_file=config_file,
            factor_selection=config.get("factor_selection", "equal_weight"),
            correlation_threshold=config.get("correlation_threshold", 0.9),
        )

    portfolio_builder = PortfolioBuilder(
        n_holdings=config["portfolio"]["n_holdings"],
        max_single=config["portfolio"]["max_single"],
    )

    # 月度轮动
    results = []
    for trade_date in trade_dates:
        logger.info(f"\n{'='*60}")
        logger.info(f"月度轮动: {trade_date}")
        logger.info(f"{'='*60}")

        try:
            # 1. 宇宙锁定
            monthly_universe = universe_mgr.get_monthly_universe(
                trade_date, min_amount_20d=config["liquidity"]["min_amount_20d"]
            )

            # 2. 提取横截面
            trade_date_dt = pd.to_datetime(trade_date, format="%Y%m%d")
            available_dates = pd.to_datetime(panel.index.get_level_values(0).unique())

            # 重要：在trade_date当天，只能看到前一天的数据
            signal_date = trade_date_dt - pd.Timedelta(days=1)
            closest_date = available_dates[available_dates <= signal_date].max()

            cross_section = panel.loc[closest_date, :]
            available_etfs = [
                etf for etf in monthly_universe if etf in cross_section.index
            ]
            universe_panel = cross_section.loc[available_etfs, :]

            logger.debug(
                f"{trade_date}: 信号日期{signal_date}, 面板日期{closest_date}, "
                f"候选{len(monthly_universe)}->评分{len(universe_panel)}"
            )

            # 3. 评分
            scored = scorer.score(universe_panel)

            # 4. 组合构建
            portfolio = portfolio_builder.build(scored)

            logger.info(
                f"候选: {len(monthly_universe)} → 评分: {len(scored)} → 持仓: {len(portfolio)}"
            )

            # 记录结果
            results.append(
                {
                    "trade_date": trade_date,
                    "universe_size": len(monthly_universe),
                    "scored_size": len(scored),
                    "portfolio_size": len(portfolio),
                    "portfolio": portfolio,
                }
            )

        except Exception as e:
            logger.error(f"轮动失败: {e}")
            results.append(
                {
                    "trade_date": trade_date,
                    "universe_size": 0,
                    "scored_size": 0,
                    "portfolio_size": 0,
                    "portfolio": {},
                    "error": str(e),
                }
            )

    return results


def next_trading_day(
    d: pd.Timestamp, trading_calendar: pd.DatetimeIndex
) -> pd.Timestamp:
    """获取下一个交易日"""
    future_days = trading_calendar[trading_calendar > d]
    return future_days[0] if len(future_days) > 0 else None


def calculate_performance(results: list, panel_file: str):
    """计算回测绩效（正确时序：T日截面 → T+1开盘买入 → 下月末收盘卖出）"""
    logger.info(f"\n{'='*60}")
    logger.info("回测绩效统计（修正时序）")
    logger.info(f"{'='*60}")

    # 加载所有ETF价格数据（包含open和close）
    etf_prices = {}
    etf_dir = Path("raw/ETF/daily")
    for etf_file in etf_dir.glob("*.parquet"):
        etf_code = etf_file.stem.split("_")[0]
        df = pd.read_parquet(etf_file)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.set_index("trade_date")[["open", "close"]]
        etf_prices[etf_code] = df

    logger.info(f"加载价格数据: {len(etf_prices)} 只ETF")

    # 构建交易日历（所有ETF的并集）
    all_dates = set()
    for df in etf_prices.values():
        all_dates.update(df.index)
    trading_calendar = pd.DatetimeIndex(sorted(all_dates))
    logger.info(f"交易日历: {len(trading_calendar)} 个交易日")

    # 计算月度收益
    monthly_returns = []
    monthly_details = []

    for i in range(len(results) - 1):
        current = results[i]
        next_result = results[i + 1]

        if not current["portfolio"]:
            monthly_returns.append(0.0)
            continue

        # 决策日T
        decision_date = pd.to_datetime(current["trade_date"], format="%Y%m%d")

        # 入场日：T+1开盘（严格>T的下一个交易日）
        entry_date = next_trading_day(decision_date, trading_calendar)

        # 出场日：下个月末收盘（下一个决策日的前一天收盘）
        next_decision_date = pd.to_datetime(next_result["trade_date"], format="%Y%m%d")
        exit_date = next_trading_day(next_decision_date, trading_calendar)

        if entry_date is None or exit_date is None:
            logger.warning(f"{decision_date}: 无法确定交易日期")
            monthly_returns.append(0.0)
            continue

        # 计算组合收益
        portfolio_return = 0.0
        valid_weights = 0.0
        turnover = 0.0

        for etf, weight in current["portfolio"].items():
            try:
                if etf not in etf_prices:
                    logger.warning(f"{etf} 价格数据不存在")
                    continue

                prices = etf_prices[etf]

                # 入场价格：entry_date开盘
                if entry_date not in prices.index:
                    # 找最近的有效交易日（不能早于entry_date）
                    valid_dates = prices.index[prices.index >= entry_date]
                    if len(valid_dates) == 0:
                        logger.warning(f"{etf} 入场日期{entry_date}无数据")
                        continue
                    entry_actual = valid_dates[0]
                else:
                    entry_actual = entry_date

                # 出场价格：exit_date前一天收盘（因为exit_date是下一轮的T+1）
                exit_close_date = exit_date - pd.Timedelta(days=1)
                valid_exit_dates = prices.index[prices.index <= exit_close_date]
                if len(valid_exit_dates) == 0:
                    logger.warning(f"{etf} 出场日期{exit_close_date}无数据")
                    continue
                exit_actual = valid_exit_dates[-1]

                entry_price = prices.loc[entry_actual, "open"]
                exit_price = prices.loc[exit_actual, "close"]

                etf_return = (exit_price - entry_price) / entry_price
                portfolio_return += weight * etf_return
                valid_weights += weight
                turnover += weight  # 月度换手（简化：假设全部调仓）

                logger.debug(
                    f"{etf}: {entry_actual.date()}开{entry_price:.4f} → "
                    f"{exit_actual.date()}收{exit_price:.4f} = {etf_return:+.2%}"
                )
            except Exception as e:
                logger.warning(f"{etf} 收益计算失败: {e}")
                continue

        # 归一化
        if valid_weights > 0:
            portfolio_return = portfolio_return / valid_weights

        # 扣除交易成本（佣金万2.5 + 滑点10bp）
        transaction_cost = turnover * (0.00025 + 0.0010)
        net_return = portfolio_return - transaction_cost

        monthly_returns.append(net_return)
        monthly_details.append(
            {
                "decision_date": decision_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "gross_return": portfolio_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "turnover": turnover,
            }
        )

        logger.info(
            f"{decision_date.date()}: 毛收益{portfolio_return:+.2%}, "
            f"成本{transaction_cost:.2%}, 净收益{net_return:+.2%}"
        )

    # 统计指标
    if monthly_returns:
        returns_series = pd.Series(monthly_returns)
        cumulative_return = (1 + returns_series).prod() - 1
        annual_return = (1 + cumulative_return) ** (12 / len(monthly_returns)) - 1
        volatility = returns_series.std() * (12**0.5)
        sharpe = annual_return / volatility if volatility > 0 else 0
        max_drawdown = (
            returns_series.cumsum() - returns_series.cumsum().cummax()
        ).min()
        win_rate = (returns_series > 0).sum() / len(returns_series)

        logger.info(f"回测月数: {len(monthly_returns)}")
        logger.info(f"累计收益: {cumulative_return:.2%}")
        logger.info(f"年化收益: {annual_return:.2%}")
        logger.info(f"年化波动: {volatility:.2%}")
        logger.info(f"夏普比率: {sharpe:.2f}")
        logger.info(f"最大回撤: {max_drawdown:.2%}")
        logger.info(f"月胜率: {win_rate:.2%}")

        # 月度收益明细
        logger.info("\n月度收益明细（净收益，已扣成本）:")
        for detail in monthly_details:
            logger.info(
                f"  {detail['decision_date'].date()}: "
                f"毛{detail['gross_return']:+.2%} - 成本{detail['transaction_cost']:.2%} "
                f"= 净{detail['net_return']:+.2%}"
            )

        # 平均换手与成本
        avg_turnover = sum(d["turnover"] for d in monthly_details) / len(
            monthly_details
        )
        avg_cost = sum(d["transaction_cost"] for d in monthly_details) / len(
            monthly_details
        )
        logger.info(f"\n平均月度换手: {avg_turnover:.2%}")
        logger.info(f"平均月度成本: {avg_cost:.2%}")
        logger.info(f"年化成本: {avg_cost * 12:.2%}")

        return {
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_turnover": avg_turnover,
            "avg_cost": avg_cost,
        }

    return None


def main():
    parser = argparse.ArgumentParser(description="ETF轮动12个月回测")
    parser.add_argument("--start-date", help="开始日期(YYYYMMDD)", default="20241001")
    parser.add_argument("--end-date", help="结束日期(YYYYMMDD)", default="20250930")
    parser.add_argument(
        "--factor-set", choices=["core", "extended"], default="core", help="因子集选择"
    )
    parser.add_argument(
        "--panel-file",
        default="factor_output/etf_rotation/panel_corrected_20240101_20251014.parquet",
        help="因子面板文件路径",
    )
    parser.add_argument("--config-dir", default="etf_rotation/configs", help="配置目录")
    args = parser.parse_args()

    # 根据因子集选择配置文件
    if args.factor_set == "extended":
        config_file = f"{args.config_dir}/extended_scoring.yaml"
        panel_file = args.panel_file  # 使用指定的面板文件
    else:
        config_file = f"{args.config_dir}/scoring.yaml"
        panel_file = args.panel_file

    # 定义回测月份（2024年1月-10月）
    trade_dates = [
        "20240131",
        "20240229",
        "20240328",
        "20240430",
        "20240531",
        "20240628",
        "20240731",
        "20240830",
        "20240930",
        "20241031",
        "20241130",
    ]

    logger.info(f"使用因子集: {args.factor_set}")
    logger.info(f"配置文件: {config_file}")
    logger.info(f"面板文件: {panel_file}")

    # 运行轮动
    results = run_monthly_rotation(
        trade_dates,
        panel_file,
        config_file,
        config_file,
    )

    # 计算绩效
    performance = calculate_performance(results, panel_file)

    # 保存结果
    output_dir = Path("rotation_output/backtest")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(
        [
            {
                "trade_date": r["trade_date"],
                "universe_size": r["universe_size"],
                "scored_size": r["scored_size"],
                "portfolio_size": r["portfolio_size"],
            }
            for r in results
        ]
    )
    # 添加因子集信息到文件名
    suffix = f"_{args.factor_set}" if args.factor_set == "extended" else ""

    results_df.to_csv(output_dir / f"backtest_summary{suffix}.csv", index=False)

    if performance:
        perf_df = pd.DataFrame([performance])
        perf_df.to_csv(output_dir / f"performance_metrics{suffix}.csv", index=False)

    logger.info(f"\n✅ 回测完成，结果已保存至 {output_dir}")


if __name__ == "__main__":
    main()
