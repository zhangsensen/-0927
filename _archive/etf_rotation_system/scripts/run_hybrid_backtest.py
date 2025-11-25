"""
快速验证: Linus优化后8因子性能
对比3种策略:
1. 战略层单独 (月度,长周期因子,低换手)
2. 战术层单独 (周度,短周期因子,高换手)
3. 混合策略 (70%战略 + 30%战术)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

import logging

from strategies.hybrid_linus_strategy import HybridLinusStrategy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_latest_panel() -> pd.DataFrame:
    """加载最新因子面板"""
    panel_dir = Path("/Users/zhangshenshen/深度量化0927/etf_cross_section_results")

    # 查找最新panel
    panel_files = sorted(panel_dir.glob("panel_*.parquet"))
    if not panel_files:
        raise FileNotFoundError("未找到因子面板文件")

    latest_panel = panel_files[-1]
    logger.info(f"📁 加载面板: {latest_panel.name}")

    df = pd.read_parquet(latest_panel)
    logger.info(
        f"✅ 数据: {len(df)} 行, {df['code'].nunique()} 只, {df['trade_date'].nunique()} 天"
    )

    return df


def prepare_price_data(factor_df: pd.DataFrame) -> pd.DataFrame:
    """准备价格数据

    Args:
        factor_df: 因子面板 (包含close)

    Returns:
        价格数据 (trade_date, code, close, ret_1d)
    """
    price_df = factor_df[["trade_date", "code", "close"]].copy()

    # 计算日收益率
    price_df = price_df.sort_values(["code", "trade_date"])
    price_df["ret_1d"] = price_df.groupby("code")["close"].pct_change()
    price_df["ret_1d"] = price_df["ret_1d"].fillna(0)

    logger.info(f"✅ 价格数据准备完成: {len(price_df)} 行")
    return price_df


def run_strategic_only(factor_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
    """仅战略层回测

    Args:
        factor_df: 因子数据
        price_df: 价格数据

    Returns:
        性能指标
    """
    logger.info("\n" + "=" * 60)
    logger.info("🎯 战略层单独回测 (月度调仓)")
    logger.info("=" * 60)

    # 临时配置: 100%战略层
    config_path = "/Users/zhangshenshen/深度量化0927/etf_rotation_system/config/hybrid_strategy_config.yaml"

    strategy = HybridLinusStrategy(config_path)
    strategy.strategic_config["portfolio_weight"] = 1.0  # 100%
    strategy.tactical_config["portfolio_weight"] = 0.0  # 0%

    # 战略层信号
    signals = strategy.generate_strategic_signals(factor_df)

    # 计算收益 (简化版,等权)
    returns = []
    for date in signals["trade_date"].unique():
        holdings = signals[signals["trade_date"] == date]["code"].tolist()

        # 获取下一日收益
        date_price = price_df[
            (price_df["trade_date"] == date) & (price_df["code"].isin(holdings))
        ]

        if len(date_price) > 0:
            avg_ret = date_price["ret_1d"].mean()
            returns.append({"trade_date": date, "ret": avg_ret})

    ret_df = pd.DataFrame(returns)
    ret_df["cumulative"] = (1 + ret_df["ret"]).cumprod() - 1

    # 计算指标
    cum_ret = ret_df["cumulative"].iloc[-1] if len(ret_df) > 0 else 0
    annual_ret = (1 + cum_ret) ** (252 / len(ret_df)) - 1 if len(ret_df) > 0 else 0
    annual_vol = ret_df["ret"].std() * np.sqrt(252) if len(ret_df) > 1 else 0
    sharpe = (annual_ret - 0.02) / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    cumulative = (1 + ret_df["ret"]).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_dd = drawdowns.min()

    metrics = {
        "strategy": "战略层",
        "cumulative_return": cum_ret,
        "annual_return": annual_ret,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "n_trades": len(signals) / signals["trade_date"].nunique(),  # 平均持仓数
    }

    logger.info(f"累计收益: {cum_ret:.2%}")
    logger.info(f"年化收益: {annual_ret:.2%}")
    logger.info(f"Sharpe: {sharpe:.3f}")
    logger.info(f"最大回撤: {max_dd:.2%}")

    return metrics


def run_tactical_only(factor_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
    """仅战术层回测"""
    logger.info("\n" + "=" * 60)
    logger.info("⚡ 战术层单独回测 (周度调仓)")
    logger.info("=" * 60)

    config_path = "/Users/zhangshenshen/深度量化0927/etf_rotation_system/config/hybrid_strategy_config.yaml"
    strategy = HybridLinusStrategy(config_path)
    strategy.strategic_config["portfolio_weight"] = 0.0  # 0%
    strategy.tactical_config["portfolio_weight"] = 1.0  # 100%

    signals = strategy.generate_tactical_signals(factor_df)

    # 计算收益
    returns = []
    for date in signals["trade_date"].unique():
        holdings = signals[signals["trade_date"] == date]["code"].tolist()

        date_price = price_df[
            (price_df["trade_date"] == date) & (price_df["code"].isin(holdings))
        ]

        if len(date_price) > 0:
            avg_ret = date_price["ret_1d"].mean()
            returns.append({"trade_date": date, "ret": avg_ret})

    ret_df = pd.DataFrame(returns)
    ret_df["cumulative"] = (1 + ret_df["ret"]).cumprod() - 1

    # 指标
    cum_ret = ret_df["cumulative"].iloc[-1] if len(ret_df) > 0 else 0
    annual_ret = (1 + cum_ret) ** (252 / len(ret_df)) - 1 if len(ret_df) > 0 else 0
    annual_vol = ret_df["ret"].std() * np.sqrt(252) if len(ret_df) > 1 else 0
    sharpe = (annual_ret - 0.02) / annual_vol if annual_vol > 0 else 0

    cumulative = (1 + ret_df["ret"]).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_dd = drawdowns.min()

    metrics = {
        "strategy": "战术层",
        "cumulative_return": cum_ret,
        "annual_return": annual_ret,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "n_trades": len(signals) / signals["trade_date"].nunique(),
    }

    logger.info(f"累计收益: {cum_ret:.2%}")
    logger.info(f"年化收益: {annual_ret:.2%}")
    logger.info(f"Sharpe: {sharpe:.3f}")
    logger.info(f"最大回撤: {max_dd:.2%}")

    return metrics


def run_hybrid(factor_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
    """混合策略回测"""
    logger.info("\n" + "=" * 60)
    logger.info("🚀 混合策略回测 (70%战略 + 30%战术)")
    logger.info("=" * 60)

    config_path = "/Users/zhangshenshen/深度量化0927/etf_rotation_system/config/hybrid_strategy_config.yaml"
    strategy = HybridLinusStrategy(config_path)

    # 完整回测
    returns_df, metrics = strategy.run_backtest(factor_df, price_df)

    # 保存结果
    output_dir = "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/hybrid_strategy"
    strategy.save_results(output_dir)

    metrics["strategy"] = "混合策略"
    return metrics


def main():
    """主函数"""
    logger.info("🚀 Linus优化后8因子性能验证")

    # 1. 加载数据
    factor_df = load_latest_panel()
    price_df = prepare_price_data(factor_df)

    # 2. 三种策略回测
    results = []

    try:
        strategic_metrics = run_strategic_only(factor_df, price_df)
        results.append(strategic_metrics)
    except Exception as e:
        logger.error(f"战略层回测失败: {e}")

    try:
        tactical_metrics = run_tactical_only(factor_df, price_df)
        results.append(tactical_metrics)
    except Exception as e:
        logger.error(f"战术层回测失败: {e}")

    try:
        hybrid_metrics = run_hybrid(factor_df, price_df)
        results.append(hybrid_metrics)
    except Exception as e:
        logger.error(f"混合策略回测失败: {e}")

    # 3. 对比汇总
    if results:
        logger.info("\n" + "=" * 80)
        logger.info("📊 三种策略性能对比")
        logger.info("=" * 80)

        comparison = pd.DataFrame(results)
        comparison = comparison[
            [
                "strategy",
                "cumulative_return",
                "annual_return",
                "sharpe_ratio",
                "max_drawdown",
                "n_trades",
            ]
        ]

        print(comparison.to_string(index=False))

        # 保存
        save_path = "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/strategy_comparison.csv"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(save_path, index=False)
        logger.info(f"\n✅ 对比结果已保存: {save_path}")


if __name__ == "__main__":
    main()
