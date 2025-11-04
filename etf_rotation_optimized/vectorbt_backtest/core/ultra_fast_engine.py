#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超高性能并行回测引擎 - V2
使用预计算和智能权重生成
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config_loader import BacktestConfig
from .data_loader import DataLoader
from .factor_validator import FactorValidator
from .vectorized_engine import UltraFastVectorEngine
from .weight_generator import SmartWeightGenerator


class UltraFastBacktestEngine:
    """超高性能回测引擎 V2"""

    def __init__(self, config: BacktestConfig, logger: logging.Logger = None):
        # 规范化配置中的相对路径为绝对路径（以vectorbt_backtest为基准）
        base_dir = Path(__file__).resolve().parent.parent  # core/ -> vectorbt_backtest/
        self.config = config.get_absolute_paths(base_dir)
        self.logger = logger or logging.getLogger(__name__)

        # 加载数据
        self.logger.info("加载数据...")
        data_loader = DataLoader(
            factor_dir=self.config.factor_dir,
            ohlcv_dir=self.config.ohlcv_dir,
            logger=self.logger,
        )
        factors_dict, close_df = data_loader.load_all()

        # 初始化向量化引擎（所有数据预处理在这里完成）
        self.logger.info("初始化向量化引擎...")
        self.engine = UltraFastVectorEngine(
            factors_dict=factors_dict,
            close_df=close_df,
            init_cash=self.config.init_cash,
            fees=self.config.fees,
            logger=self.logger,
        )

        # 因子有效性检验
        self.logger.info("检验因子有效性...")
        self.factor_validator = FactorValidator(factors_dict, close_df, self.logger)
        self.factor_stats = self.factor_validator.validate_all_factors(
            significance_threshold=self.config.ic_significant_threshold
        )
        self.selected_factors = self.factor_validator.select_best_factors(
            self.factor_stats,
            min_ir=self.config.min_ic_ir,
            min_positive_rate=self.config.min_positive_rate,
            min_significant_rate=self.config.min_significant_rate,
            min_abs_ic=self.config.min_abs_ic,
            min_observations=self.config.min_observations,
            max_correlation=self.config.max_factor_correlation,
            fdr_q=self.config.fdr_q,
            fallback_top_k=self.config.fallback_top_k,
        )

        if not self.selected_factors:
            self.logger.warning("因子筛选结果为空，回退启用全部因子。")
            self.selected_factors = list(self.engine.factor_names)

        self.selected_factor_indices = np.array(
            [self.engine.factor_names.index(name) for name in self.selected_factors],
            dtype=int,
        )

        # 更新因子权重生成器（仅针对筛选后的因子维度）
        self.weight_gen = SmartWeightGenerator(n_factors=len(self.selected_factors))

        selection_trace = getattr(self.factor_validator, "selection_trace", {})
        trace_msg = (
            f"初选{selection_trace.get('initial_pass', len(self.selected_factors))}/"
            f"{selection_trace.get('total_factors', self.engine.n_factors)}, "
            f"FDR后{selection_trace.get('fdr_pass', len(self.selected_factors))}, "
            f"回退={'是' if selection_trace.get('fallback_used') else '否'}"
        )
        self.logger.info(
            f"因子筛选完成：最终{len(self.selected_factors)}个因子。详情: {trace_msg}"
        )
        preview_count = min(10, len(self.selected_factors))
        if preview_count > 0:
            preview_factors = ", ".join(self.selected_factors[:preview_count])
            self.logger.info(f"有效因子示例（前{preview_count}）: {preview_factors}")
        self.logger.info(
            f"引擎就绪：{len(self.selected_factors)}/{self.engine.n_factors}有效因子 × {self.engine.n_dates}天"
        )

    def generate_weight_combinations(self, n_combinations: int) -> np.ndarray:
        """
        生成智能权重组合 - 使用高性能采样

        Args:
            n_combinations: 目标组合数

        Returns:
            weight_matrix: (n_combinations, n_factors) 权重矩阵
        """
        self.logger.info(f"生成 {n_combinations} 个高性能权重组合...")

        # 优化策略：基于因子IC值的智能采样
        strategy_mix = {
            "ic_weighted_dirichlet": 0.5,  # 50% IC加权Dirichlet
            "sobol": 0.25,  # 25% Sobol（空间探索）
            "sparse_dirichlet": 0.15,  # 15% 稀疏Dirichlet
            "correlation_adjusted": 0.1,  # 10% 相关性调整
        }

        # 计算IC分数（按筛选因子顺序对齐）
        ic_scores = None
        if (
            hasattr(self, "factor_stats")
            and self.factor_stats
            and self.selected_factors
        ):
            name_to_ic = {
                name: abs(stats.get("ic_mean", 0.0))
                for name, stats in self.factor_stats.items()
            }
            ic_scores = np.array(
                [name_to_ic.get(name, 0.0) for name in self.selected_factors],
                dtype=float,
            )
            # 若全部为0，退化为None
            if not np.any(ic_scores > 0):
                ic_scores = None

        # 一次性生成所有权重（零循环！）
        selected_weight_matrix = self.weight_gen.generate_mixed_strategy_weights(
            n_combinations=n_combinations,
            strategy_mix=strategy_mix,
            ic_scores=ic_scores,
        )

        # 映射回完整因子空间
        weight_matrix = np.zeros((len(selected_weight_matrix), self.engine.n_factors))
        weight_matrix[:, self.selected_factor_indices] = selected_weight_matrix

        # 去重（基于权重的唯一性）
        unique_weights = np.unique(np.round(weight_matrix, 4), axis=0)

        importance = self.weight_gen.get_factor_importance()
        if len(importance) == len(self.selected_factors):
            importance_pairs = sorted(
                zip(self.selected_factors, importance), key=lambda x: x[1], reverse=True
            )
            top_importance = {
                name: round(score, 4) for name, score in importance_pairs[:10]
            }
            self.logger.info(f"生成 {len(unique_weights)} 个唯一权重组合")
            self.logger.info(f"因子重要性（前10）: {top_importance}")
        else:
            self.logger.info(f"生成 {len(unique_weights)} 个唯一权重组合")

        return unique_weights

    def run_backtest(self, n_weight_combinations: int = 10000) -> pd.DataFrame:
        """
        运行超高性能回测

        Args:
            n_weight_combinations: 权重组合数（默认1万）

        Returns:
            results_df: 结果DataFrame
        """
        # 生成权重组合
        weight_matrix = self.generate_weight_combinations(n_weight_combinations)
        n_actual_combinations = len(weight_matrix)

        # 计算总策略数
        n_strategies = (
            n_actual_combinations
            * len(self.config.top_n_list)
            * len(self.config.rebalance_freq_list)
        )

        self.logger.info(f"开始回测：{n_strategies} 个策略")
        self.logger.info(f"  - 权重组合: {n_actual_combinations}")
        self.logger.info(f"  - Top-N: {self.config.top_n_list}")
        self.logger.info(f"  - 调仓频率: {self.config.rebalance_freq_list}")

        # 批量回测（完全向量化，无多进程开销）
        results = self.engine.batch_backtest(
            weight_matrix=weight_matrix,
            top_n_list=self.config.top_n_list,
            rebalance_freq_list=self.config.rebalance_freq_list,
        )

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 按Sharpe排序
        results_df = results_df.sort_values(
            "sharpe_ratio", ascending=False
        ).reset_index(drop=True)

        self.logger.info(f"回测完成：{len(results_df)} 个策略")
        return results_df

    def save_results(
        self, results_df: pd.DataFrame, output_dir: Path, top_k: int = 100
    ):
        """保存结果"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存排名表（不含权益曲线）
        summary_df = results_df.drop(columns=["equity_curve"], errors="ignore").copy()
        summary_path = output_dir / "backtest_results_ranking.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"排名表保存至: {summary_path}")

        # 保存Top-K权益曲线
        top_strategies = results_df.head(top_k)
        equity_curves = {}

        for idx, row in top_strategies.iterrows():
            # 简化策略名称
            active_factors = "_".join(
                [f[:6] for f in row["factors"][:3]]
            )  # 只取前3个因子
            strategy_name = f"S{row['strategy_id']}_{active_factors}_N{row['top_n']}_F{row['rebalance_freq']}"

            # 转换权益曲线
            if isinstance(row["equity_curve"], np.ndarray):
                equity_curves[strategy_name] = row["equity_curve"]
            elif isinstance(row["equity_curve"], dict):
                equity_curves[strategy_name] = pd.Series(row["equity_curve"])

        if equity_curves:
            equity_df = pd.DataFrame(equity_curves, index=self.engine.dates)
            equity_path = output_dir / f"top{top_k}_equity_curves.csv"
            equity_df.to_csv(equity_path, encoding="utf-8-sig")
            self.logger.info(f"Top-{top_k} 权益曲线保存至: {equity_path}")

        # 生成报告
        report_path = output_dir / "backtest_summary_report.md"
        self._generate_report(results_df, report_path, top_k=10)
        self.logger.info(f"摘要报告保存至: {report_path}")

    def _generate_report(
        self, results_df: pd.DataFrame, report_path: Path, top_k: int = 10
    ):
        """生成报告"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 超高性能向量化回测报告\n\n")
            f.write(f"## 概览\n\n")
            f.write(f"- **总策略数**: {len(results_df)}\n")
            f.write(f"- **最佳Sharpe**: {results_df['sharpe_ratio'].max():.4f}\n")
            f.write(f"- **最佳年化收益**: {results_df['annual_return'].max():.2%}\n")
            f.write(f"- **最小最大回撤**: {results_df['max_drawdown'].max():.2%}\n\n")

            if {"avg_turnover", "avg_cost"}.issubset(results_df.columns):
                f.write(
                    f"- **平均日换手率**: {results_df['avg_turnover'].mean():.4f}\n"
                )
                f.write(f"- **平均日成本拖累**: {results_df['avg_cost'].mean():.6f}\n")
                f.write(
                    f"- **Top10平均换手率**: {results_df.head(top_k)['avg_turnover'].mean():.4f}\n"
                )
                f.write(
                    f"- **Top10总成本拖累**: {results_df.head(top_k)['total_cost'].mean():.4f}\n\n"
                )

            selection_trace = getattr(self.factor_validator, "selection_trace", {})
            if selection_trace:
                f.write("## 因子筛选摘要\n\n")
                f.write(
                    f"- 候选因子总数: {selection_trace.get('total_factors', len(self.engine.factor_names))}\n"
                )
                f.write(
                    f"- 初选通过: {selection_trace.get('initial_pass', len(self.selected_factors))}\n"
                )

                f.write(
                    f"- FDR后通过: {selection_trace.get('fdr_pass', len(self.selected_factors))}\n"
                )
                f.write(
                    f"- 是否触发回退: {'是' if selection_trace.get('fallback_used') else '否'}\n"
                )
                if selection_trace.get("fallback_candidates"):
                    fallback_preview = ", ".join(
                        selection_trace["fallback_candidates"][:10]
                    )
                    f.write(f"- 回退候选示例: {fallback_preview}\n")
                if selection_trace.get("final_factors"):
                    final_preview = ", ".join(selection_trace["final_factors"][:10])
                    f.write(f"- 最终因子示例: {final_preview}\n")
                f.write("\n")

            f.write(f"## Top-{top_k} 策略\n\n")
            top_columns = [
                "strategy_id",
                "factors",
                "weights",
                "top_n",
                "rebalance_freq",
                "sharpe_ratio",
                "annual_return",
                "max_drawdown",
                "calmar_ratio",
            ]
            if {"avg_turnover", "avg_cost"}.issubset(results_df.columns):
                top_columns.extend(["avg_turnover", "avg_cost"])
            top_df = results_df.head(top_k)[top_columns]
            f.write(top_df.to_markdown(index=False))
            f.write("\n\n")

            f.write(f"## 因子重要性分析\n\n")
            # 统计Top-100中各因子出现频率
            factor_counts = {}
            for _, row in results_df.head(100).iterrows():
                for factor, weight in zip(row["factors"], row["weights"]):
                    if factor not in factor_counts:
                        factor_counts[factor] = {"count": 0, "total_weight": 0}
                    factor_counts[factor]["count"] += 1
                    factor_counts[factor]["total_weight"] += abs(weight)

            # 排序
            sorted_factors = sorted(
                factor_counts.items(),
                key=lambda x: (x[1]["count"], x[1]["total_weight"]),
                reverse=True,
            )

            f.write("| 因子 | 出现次数 | 累计权重 | 平均权重 |\n")
            f.write("|------|----------|----------|----------|\n")
            for factor, stats in sorted_factors[:20]:
                avg_weight = stats["total_weight"] / stats["count"]
                f.write(
                    f"| {factor} | {stats['count']} | {stats['total_weight']:.4f} | {avg_weight:.4f} |\n"
                )

            f.write("\n\n")
            f.write(f"## 性能分布\n\n")
            f.write(f"### Sharpe比率\n")
            f.write(f"- 均值: {results_df['sharpe_ratio'].mean():.4f}\n")
            f.write(f"- 中位数: {results_df['sharpe_ratio'].median():.4f}\n")
            f.write(f"- 75分位: {results_df['sharpe_ratio'].quantile(0.75):.4f}\n")
            f.write(f"- 90分位: {results_df['sharpe_ratio'].quantile(0.90):.4f}\n\n")

            if {"avg_turnover", "avg_cost"}.issubset(results_df.columns):
                f.write(f"### 换手率\n")
                f.write(f"- 均值: {results_df['avg_turnover'].mean():.4f}\n")
                f.write(f"- 中位数: {results_df['avg_turnover'].median():.4f}\n")
                f.write(f"- 90分位: {results_df['avg_turnover'].quantile(0.90):.4f}\n")
                f.write(
                    f"- 总换手（Top10均值）: {results_df.head(top_k)['total_turnover'].mean():.4f}\n\n"
                )

                f.write(f"### 成本拖累\n")
                f.write(f"- 均值: {results_df['avg_cost'].mean():.6f}\n")
                f.write(f"- 中位数: {results_df['avg_cost'].median():.6f}\n")
                f.write(f"- 90分位: {results_df['avg_cost'].quantile(0.90):.6f}\n")
                f.write(
                    f"- 总成本（Top10均值）: {results_df.head(top_k)['total_cost'].mean():.4f}\n\n"
                )
