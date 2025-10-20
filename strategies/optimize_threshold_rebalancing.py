#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阈值调仓参数优化：暴力测试寻找最优阈值和持有期

目标：通过VectorBT暴力测试，找到最优的threshold和holding_period组合
- 解决当前37.69%年化换手率过高的问题
- 最大化净收益（考虑交易成本）
- 控制回撤在合理范围内

测试参数：
- threshold_range: [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
- holding_period_range: [1, 3, 5, 7, 10, 15, 20] 天
- 总测试组合：63个
"""

import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import vectorbt as vbt

    _HAS_VECTORBT = True
except ImportError:
    vbt = None
    _HAS_VECTORBT = False


class ThresholdOptimizationEngine:
    """阈值调仓优化引擎

    核心功能：
    1. 基于现有因子得分计算权重
    2. 应用不同阈值和持有期策略
    3. 计算考虑交易成本的回测结果
    4. 输出最优参数组合
    """

    def __init__(
        self,
        factor_scores: np.ndarray,  # (n_dates, n_etfs) 因子得分
        price_data: pd.DataFrame,  # (date x symbol) 价格数据
        trading_costs: float = 0.0014,  # 单边交易成本
        init_cash: float = 1_000_000.0,
    ):
        self.factor_scores = factor_scores
        self.price_data = price_data
        self.trading_costs = trading_costs
        self.init_cash = init_cash

        self.n_dates, self.n_etfs = factor_scores.shape
        self.symbols = price_data.columns.tolist()
        self.dates = price_data.index.tolist()

        print(f"🚀 优化引擎初始化: {self.n_dates}天 × {self.n_etfs}个标的")
        print(f"💰 交易成本: {self.trading_costs:.4f} (单边)")

    def calculate_baseline_weights(self, top_n: int = 8) -> np.ndarray:
        """计算基准权重（每日调仓，无阈值限制）

        Args:
            top_n: 选股数量

        Returns:
            weights: (n_dates, n_etfs) 权重矩阵
        """
        weights = np.zeros((self.n_dates, self.n_etfs), dtype=np.float32)

        for t in range(self.n_dates):
            # 获取当日因子得分
            scores_t = self.factor_scores[t]

            # 找到Top-N标的
            top_indices = np.argpartition(-scores_t, top_n)[:top_n]

            # 等权分配
            weights[t, top_indices] = 1.0 / top_n

        return weights

    def apply_threshold_strategy(
        self, threshold: float, holding_period: int, top_n: int = 8
    ) -> np.ndarray:
        """应用阈值调仓策略

        Args:
            threshold: 调仓阈值（信号变化幅度）
            holding_period: 最小持有期（天）
            top_n: 选股数量

        Returns:
            weights: (n_dates, n_etfs) 权重矩阵
        """
        # 1. 计算基准权重
        baseline_weights = self.calculate_baseline_weights(top_n)

        # 2. 应用阈值策略
        optimized_weights = np.zeros_like(baseline_weights)
        optimized_weights[0] = baseline_weights[0]  # 第一天使用基准权重

        # 持有期跟踪
        days_held = np.zeros(self.n_etfs, dtype=int)

        for t in range(1, self.n_dates):
            # 检查是否满足调仓条件
            should_rebalance = False

            # 条件1: 持有期已满
            if np.any(days_held >= holding_period):
                should_rebalance = True

            # 条件2: 信号变化超过阈值（检查持仓变化）
            baseline_positions = set(np.where(baseline_weights[t] > 0)[0])
            current_positions = set(np.where(optimized_weights[t - 1] > 0)[0])

            # 计算持仓变化比例
            position_change = (
                len(baseline_positions.symmetric_difference(current_positions)) / top_n
            )
            if position_change > threshold:
                should_rebalance = True

            if should_rebalance:
                # 更新权重
                optimized_weights[t] = baseline_weights[t]
                # 重置持有期
                current_positions = optimized_weights[t] > 0
                new_positions = baseline_weights[t] > 0

                # 继续持有的持仓
                continued_positions = current_positions & new_positions
                days_held[continued_positions] += 1

                # 新开仓的持仓
                new_opened = ~current_positions & new_positions
                days_held[new_opened] = 1

                # 清仓的持仓
                closed = current_positions & ~new_positions
                days_held[closed] = 0
            else:
                # 保持原权重
                optimized_weights[t] = optimized_weights[t - 1]
                days_held[optimized_weights[t - 1] > 0] += 1

        return optimized_weights

    def run_vectorbt_backtest(self, weights: np.ndarray) -> Dict[str, float]:
        """使用VectorBT运行回测

        Args:
            weights: (n_dates, n_etfs) 权重矩阵

        Returns:
            回测指标字典
        """
        # 转换为DataFrame
        weights_df = pd.DataFrame(weights, index=self.dates, columns=self.symbols)

        # 计算收益率
        returns = self.price_data.pct_change().fillna(0.0)

        # 滞后权重（避免前视偏差）
        lagged_weights = weights_df.shift(1).fillna(0.0)

        # 计算组合收益
        portfolio_returns = (lagged_weights * returns).sum(axis=1)

        # 计算换手率
        weight_changes = weights_df.diff().abs().sum(axis=1)
        turnover = 0.5 * weight_changes

        # 计算交易成本
        trading_costs = self.trading_costs * turnover
        net_returns = portfolio_returns - trading_costs

        # 计算权益曲线
        equity_curve = (1.0 + net_returns).cumprod()

        # 计算指标
        total_return = equity_curve.iloc[-1] - 1.0
        n_years = len(equity_curve) / 252.0
        annual_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0

        # 最大回撤
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()

        # 夏普比率
        sharpe = (
            net_returns.mean() / net_returns.std() * np.sqrt(252)
            if net_returns.std() > 0
            else 0.0
        )

        # 胜率
        win_rate = (net_returns > 0).mean()

        # 年化换手率
        annual_turnover = turnover.mean() * 252

        # 交易成本（年化）
        annual_costs = trading_costs.mean() * 252

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "annual_turnover": annual_turnover,
            "annual_costs": annual_costs,
            "net_annual_return": annual_return,  # net_return after costs
            "equity_curve": equity_curve,
            "returns_series": net_returns,
        }

    def optimize_parameters(
        self,
        threshold_range: List[float],
        holding_period_range: List[int],
        top_n: int = 8,
    ) -> List[Dict]:
        """暴力优化参数

        Args:
            threshold_range: 阈值范围
            holding_period_range: 持有期范围
            top_n: 选股数量

        Returns:
            优化结果列表
        """
        results = []
        param_combinations = list(
            itertools.product(threshold_range, holding_period_range)
        )

        print(f"🎯 开始参数优化: {len(param_combinations)} 个组合")
        print(f"📊 阈值范围: {threshold_range}")
        print(f"📅 持有期范围: {holding_period_range}")
        print(f"🎪 选股数量: {top_n}")

        with tqdm(total=len(param_combinations), desc="参数优化") as pbar:
            for threshold, holding_period in param_combinations:
                try:
                    # 应用策略
                    weights = self.apply_threshold_strategy(
                        threshold=threshold, holding_period=holding_period, top_n=top_n
                    )

                    # 运行回测
                    metrics = self.run_vectorbt_backtest(weights)

                    # 记录结果
                    result = {
                        "threshold": threshold,
                        "holding_period": holding_period,
                        "top_n": top_n,
                        **metrics,
                    }

                    # 移除大数据对象，保留关键指标
                    for key in ["equity_curve", "returns_series"]:
                        if key in result:
                            del result[key]

                    results.append(result)

                    # 实时显示最优结果
                    if len(results) == 1 or result["sharpe"] > max(
                        r["sharpe"] for r in results
                    ):
                        print(f"\n🏆 新的最优组合 (Sharpe: {result['sharpe']:.4f}):")
                        print(f"   阈值: {threshold}, 持有期: {holding_period}天")
                        print(f"   年化收益: {result['annual_return']:.4f}")
                        print(f"   最大回撤: {result['max_drawdown']:.4f}")
                        print(f"   换手率: {result['annual_turnover']:.2f}")
                        print(f"   年化成本: {result['annual_costs']:.4f}")

                except Exception as e:
                    print(f"❌ 参数组合 ({threshold}, {holding_period}) 失败: {e}")
                    continue

                pbar.update(1)

        return results


def main():
    """主函数"""
    # 解析参数
    parser = argparse.ArgumentParser(description="阈值调仓参数优化")
    parser.add_argument(
        "--factor-panel",
        default="../factor_output/etf_rotation_production/panel_FULL_20200102_20251014.parquet",
    )
    parser.add_argument("--data-dir", default="../raw/ETF/daily")
    parser.add_argument("--factors", nargs="+", default=None)
    parser.add_argument("--top-factors-json", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--trading-costs", type=float, default=0.0014)
    parser.add_argument("--output", default="results/threshold_optimization")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    print("=" * 80)
    print("🎯 阈值调仓参数优化 - 暴力测试版")
    print("=" * 80)

    # 1. 加载数据
    print("📊 加载数据...")

    # 加载价格数据
    data_path = Path(args.data_dir)
    if not data_path.is_absolute():
        # 相对路径：相对于项目根目录
        data_path = Path(__file__).parent / args.data_dir

    print(f"📂 使用数据路径: {data_path}")
    print(f"📂 路径存在: {data_path.exists()}")

    price_files = list(data_path.glob("*.parquet"))
    price_dfs = []
    for fp in price_files:
        try:
            # 检查文件是否包含必要的列
            df = pd.read_parquet(fp)
            if "close" in df.columns and "trade_date" in df.columns:
                # 提取symbol：从文件名中提取（格式：515030.SH_daily_...）
                symbol = fp.stem.split("_")[0]
                df_selected = df[["trade_date", "close"]].copy()
                df_selected["symbol"] = symbol
                price_dfs.append(df_selected)
            else:
                print(f"⚠️ 跳过文件 {fp}: 缺少必要的列 (close, trade_date)")
        except Exception as e:
            print(f"⚠️ 跳过文件 {fp}: {e}")

    if not price_dfs:
        raise FileNotFoundError(f"在 {data_path} 中找不到有效的价格文件")

    prices = pd.concat(price_dfs, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["trade_date"])
    price_pivot = prices.pivot(
        index="date", columns="symbol", values="close"
    ).sort_index()
    price_pivot = price_pivot.ffill().dropna(how="all")

    print(f"✅ 价格数据: {len(price_pivot)}天 × {len(price_pivot.columns)}个标的")

    # 2. 加载因子数据
    if args.factors:
        factors = args.factors
    else:
        # 使用经过因子筛选验证的Top-20有效因子
        factors = [
            "PRICE_POSITION_60",
            "VBT_STOCH_K_20_3",
            "VBT_STOCH_K_20_5",
            "PRICE_POSITION_20",
            "VBT_STOCH_D_20_3",
            "VBT_STOCH_D_20_5",
            "TA_EMA_252",
            "TA_ADOSC",
            "PRICE_POSITION_30",
            "TA_EMA_200",
            "VBT_MA_252",
            "TA_SMA_252",
            "VBT_MA_200",
            "TA_SMA_200",
            "VBT_STOCH_K_14_3",
            "VBT_STOCH_K_14_5",
            "TA_RSI_20",
            "TA_RSI_30",
            "TA_RSI_24",
            "VBT_MA_150",
        ]
        print(f"📊 使用因子筛选验证的Top-20有效因子")

    # 加载因子面板
    factor_panel_path = Path(args.factor_panel)
    if not factor_panel_path.is_absolute():
        factor_panel_path = Path(__file__).parent / args.factor_panel

    print(f"📊 使用因子面板: {factor_panel_path}")
    print(f"📊 面板存在: {factor_panel_path.exists()}")

    factor_panel = pd.read_parquet(factor_panel_path)
    factor_panel = factor_panel[factors].copy()

    # 标准化因子
    def normalize_factors(panel, method="zscore"):
        grouped = panel.groupby(level="date")
        if method == "zscore":

            def _zscore(df):
                return (df - df.mean()) / df.std(ddof=0)

            normalized = grouped.transform(_zscore)
        else:
            normalized = grouped.rank(pct=True) - 0.5
        return normalized.fillna(0.0)

    normalized_panel = normalize_factors(factor_panel)

    # 转换为numpy数组
    factor_scores = (
        normalized_panel.unstack(level="symbol")
        .reindex(index=price_pivot.index, columns=price_pivot.columns)
        .fillna(0.0)
        .values
    )

    print(f"✅ 因子数据: {len(factors)}个因子")

    # 3. 初始化优化引擎
    engine = ThresholdOptimizationEngine(
        factor_scores=factor_scores,
        price_data=price_pivot,
        trading_costs=args.trading_costs,
    )

    # 4. 定义参数范围
    threshold_range = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    holding_period_range = [1, 3, 5, 7, 10, 15, 20]

    # 4.1 调试：先测试基准策略
    print("\n🔍 调试：测试基准策略（无阈值限制）")
    baseline_weights = engine.calculate_baseline_weights(top_n=8)
    baseline_metrics = engine.run_vectorbt_backtest(baseline_weights)
    print(f"基准策略换手率: {baseline_metrics['annual_turnover']:.2f}")
    print(f"基准策略年化收益: {baseline_metrics['annual_return']:.4f}")
    print(f"基准策略夏普: {baseline_metrics['sharpe']:.4f}")

    # 检查每日调仓变化
    changes = []
    for t in range(1, engine.n_dates):
        pos1 = set(np.where(baseline_weights[t - 1] > 0)[0])
        pos2 = set(np.where(baseline_weights[t] > 0)[0])
        change = len(pos1.symmetric_difference(pos2)) / 8
        changes.append(change)
    print(f"每日平均持仓变化: {np.mean(changes):.3f}")
    print(f"持仓变化标准差: {np.std(changes):.3f}")
    print(f"最大持仓变化: {np.max(changes):.3f}")
    print()

    # 5. 运行优化
    start_time = time.time()
    results = engine.optimize_parameters(
        threshold_range=threshold_range,
        holding_period_range=holding_period_range,
        top_n=8,
    )
    optimization_time = time.time() - start_time

    print(f"\n✅ 优化完成: {len(results)} 个有效结果，耗时 {optimization_time:.2f}秒")

    # 6. 分析结果
    if not results:
        print("❌ 没有有效的优化结果")
        return

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 按夏普比率排序
    results_df = results_df.sort_values("sharpe", ascending=False)

    # 7. 输出结果
    print("\n" + "=" * 100)
    print("🏆 TOP 10 最优参数组合 (按夏普比率排序)")
    print("=" * 100)

    display_cols = [
        "threshold",
        "holding_period",
        "sharpe",
        "annual_return",
        "max_drawdown",
        "annual_turnover",
        "annual_costs",
    ]

    top_results = results_df.head(10)[display_cols]
    print(top_results.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    # 8. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细结果
    results_file = output_dir / f"threshold_optimization_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)

    # 保存最优参数
    best_params = results_df.iloc[0]
    best_params_file = output_dir / f"best_params_{timestamp}.json"
    best_params_dict = best_params.to_dict()
    with open(best_params_file, "w", encoding="utf-8") as f:
        json.dump(best_params_dict, f, indent=2, ensure_ascii=False)

    print(f"\n📁 详细结果已保存到: {results_file}")
    print(f"🏆 最优参数已保存到: {best_params_file}")

    # 9. 总结报告
    print("\n" + "=" * 80)
    print("📊 优化总结报告")
    print("=" * 80)

    baseline_result = results_df[results_df["threshold"] == 0.02].iloc[
        0
    ]  # 最接近无限制的

    print(f"🎯 最优组合:")
    print(f"   阈值: {best_params['threshold']:.3f}")
    print(f"   持有期: {best_params['holding_period']}天")
    print(f"   夏普比率: {best_params['sharpe']:.4f}")
    print(f"   年化收益: {best_params['annual_return']:.4f}")
    print(f"   最大回撤: {best_params['max_drawdown']:.4f}")
    print(f"   年化换手率: {best_params['annual_turnover']:.2f}")
    print(f"   年化成本: {best_params['annual_costs']:.4f}")
    print(f"   净收益: {best_params['net_annual_return']:.4f}")

    print(f"\n📈 相比基准改善:")
    print(
        f"   换手率: {baseline_result['annual_turnover']:.2f} → {best_params['annual_turnover']:.2f}"
    )
    print(
        f"   交易成本: {baseline_result['annual_costs']:.4f} → {best_params['annual_costs']:.4f}"
    )
    print(f"   夏普比率: {baseline_result['sharpe']:.4f} → {best_params['sharpe']:.4f}")

    turnover_reduction = (
        1 - best_params["annual_turnover"] / baseline_result["annual_turnover"]
    ) * 100
    cost_reduction = (
        1 - best_params["annual_costs"] / baseline_result["annual_costs"]
    ) * 100

    print(f"\n💰 成本节约:")
    print(f"   换手率降低: {turnover_reduction:.1f}%")
    print(f"   交易成本降低: {cost_reduction:.1f}%")


if __name__ == "__main__":
    main()
