#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行回测引擎 - 暴力测试所有因子组合
"""

import itertools
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config_loader import BacktestConfig
from .data_loader import DataLoader
from .signal_generator import VectorizedSignalGenerator


def _run_single_backtest(args: Tuple) -> Dict:
    """
    单次回测任务（用于多进程）

    Args:
        args: (strategy_id, factor_combo, weights, top_n, rebalance_freq,
               factors_dict, close_df, init_cash, fees)

    Returns:
        result: 包含策略信息和性能指标的字典
    """
    (
        strategy_id,
        factor_combo,
        weights,
        top_n,
        rebalance_freq,
        factors_dict,
        close_df,
        init_cash,
        fees,
    ) = args

    try:
        # 创建信号生成器
        generator = VectorizedSignalGenerator()

        # 构建权重字典
        weight_dict = {factor: w for factor, w in zip(factor_combo, weights)}

        # 生成综合得分
        composite_score = generator.generate_composite_score(
            factors_dict=factors_dict, factor_weights=weight_dict, method="weighted_sum"
        )

        # 生成交易信号
        signals = generator.generate_topn_signals(
            composite_score=composite_score, top_n=top_n, rebalance_freq=rebalance_freq
        )

        # 回测
        equity_curve, holdings = generator.backtest_portfolio(
            signals=signals, close_df=close_df, init_cash=init_cash, fees=fees
        )

        # 计算指标
        metrics = generator.calculate_metrics(equity_curve)

        # 返回结果
        result = {
            "strategy_id": strategy_id,
            "factors": list(factor_combo),
            "weights": list(weights),
            "top_n": top_n,
            "rebalance_freq": rebalance_freq,
            "equity_curve": equity_curve.to_dict(),  # 序列化为dict
            **metrics,
        }

        return result

    except Exception as e:
        return {
            "strategy_id": strategy_id,
            "factors": list(factor_combo),
            "weights": list(weights),
            "top_n": top_n,
            "rebalance_freq": rebalance_freq,
            "error": str(e),
        }


class ParallelBacktestEngine:
    """并行回测引擎"""

    def __init__(self, config: BacktestConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # 加载数据
        self.logger.info("加载数据...")
        data_loader = DataLoader(
            factor_dir=config.factor_dir, ohlcv_dir=config.ohlcv_dir, logger=self.logger
        )
        self.factors_dict, self.close_df = data_loader.load_all()
        self.logger.info(
            f"加载完成：{len(self.factors_dict)} 个因子，{len(self.close_df)} 交易日"
        )

    def enumerate_strategies(self) -> List[Tuple]:
        """
        枚举所有策略组合

        Returns:
            strategies: [(strategy_id, factor_combo, weights, top_n, rebalance_freq), ...]
        """
        strategies = []
        strategy_id = 0

        available_factors = list(self.factors_dict.keys())
        self.logger.info(f"可用因子: {available_factors}")

        # 遍历因子组合大小
        for combo_size in self.config.combination_sizes:
            if combo_size > len(available_factors):
                self.logger.warning(f"跳过组合大小 {combo_size}（超过可用因子数）")
                continue

            # 生成所有因子组合
            factor_combos = list(itertools.combinations(available_factors, combo_size))
            self.logger.info(f"生成 {len(factor_combos)} 个 {combo_size}-因子组合")

            # 获取权重网格（根据组合大小过滤）
            weight_grid = [w for w in self.config.weight_grid if len(w) == combo_size]
            if not weight_grid:
                # 如果没有匹配的权重，使用等权
                weight_grid = [[1.0 / combo_size] * combo_size]
                self.logger.warning(f"未找到 {combo_size}-因子权重网格，使用等权")

            self.logger.info(f"使用 {len(weight_grid)} 组权重方案")

            # 遍历组合
            for factor_combo in factor_combos:
                for weights in weight_grid:
                    for top_n in self.config.top_n_list:
                        for rebalance_freq in self.config.rebalance_freq_list:
                            strategies.append(
                                (
                                    strategy_id,
                                    factor_combo,
                                    tuple(weights),
                                    top_n,
                                    rebalance_freq,
                                )
                            )
                            strategy_id += 1

        self.logger.info(f"总共枚举 {len(strategies)} 个策略")
        return strategies

    def run_parallel_backtest(
        self, strategies: List[Tuple], n_workers: int = None
    ) -> pd.DataFrame:
        """
        并行执行回测

        Args:
            strategies: 策略列表
            n_workers: 进程数（默认使用CPU核心数-1）

        Returns:
            results_df: 结果DataFrame
        """
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)

        self.logger.info(f"启动并行回测：{n_workers} 个进程")

        # 准备任务参数（添加共享数据）
        tasks = [
            (
                sid,
                fc,
                w,
                tn,
                rf,
                self.factors_dict,
                self.close_df,
                self.config.init_cash,
                self.config.fees,
            )
            for sid, fc, w, tn, rf in strategies
        ]

        # 并行执行
        results = []
        with mp.Pool(processes=n_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_run_single_backtest, tasks),
                total=len(tasks),
                desc="回测进度",
            ):
                results.append(result)

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 过滤错误（检查error列是否存在）
        if "error" in results_df.columns:
            errors = results_df[results_df["error"].notna()]
            if len(errors) > 0:
                self.logger.warning(f"{len(errors)} 个策略失败")
                self.logger.debug(
                    f"失败策略: {errors[['strategy_id', 'factors', 'error']].to_dict('records')}"
                )
                results_df = results_df[results_df["error"].isna()].copy()
                results_df = results_df.drop(columns=["error"])

        # 按Sharpe排序
        results_df = results_df.sort_values(
            "sharpe_ratio", ascending=False
        ).reset_index(drop=True)

        self.logger.info(f"回测完成：{len(results_df)} 个成功策略")
        return results_df

    def save_results(
        self, results_df: pd.DataFrame, output_dir: Path, top_k: int = 100
    ):
        """
        保存结果

        Args:
            results_df: 结果DataFrame
            output_dir: 输出目录
            top_k: 保存Top-K策略的权益曲线
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整排名（不含权益曲线）
        summary_df = results_df.drop(columns=["equity_curve"], errors="ignore").copy()
        summary_path = output_dir / "backtest_results_ranking.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"排名表保存至: {summary_path}")

        # 保存Top-K权益曲线
        top_strategies = results_df.head(top_k)
        equity_curves = {}

        for idx, row in top_strategies.iterrows():
            strategy_name = f"S{row['strategy_id']}_{'_'.join(row['factors'])}_N{row['top_n']}_F{row['rebalance_freq']}"
            equity_curves[strategy_name] = pd.Series(row["equity_curve"])

        equity_df = pd.DataFrame(equity_curves)
        equity_path = output_dir / f"top{top_k}_equity_curves.csv"
        equity_df.to_csv(equity_path, encoding="utf-8-sig")
        self.logger.info(f"Top-{top_k} 权益曲线保存至: {equity_path}")

        # 生成摘要报告
        report_path = output_dir / "backtest_summary_report.md"
        self._generate_report(results_df, report_path, top_k=10)
        self.logger.info(f"摘要报告保存至: {report_path}")

    def _generate_report(
        self, results_df: pd.DataFrame, report_path: Path, top_k: int = 10
    ):
        """生成Markdown报告"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 向量化回测结果报告\n\n")
            f.write(f"## 概览\n\n")
            f.write(f"- **总策略数**: {len(results_df)}\n")
            f.write(f"- **最佳Sharpe**: {results_df['sharpe_ratio'].max():.4f}\n")
            f.write(f"- **最佳年化收益**: {results_df['annual_return'].max():.2%}\n")
            f.write(f"- **最小最大回撤**: {results_df['max_drawdown'].max():.2%}\n\n")

            f.write(f"## Top-{top_k} 策略\n\n")
            top_df = results_df.head(top_k)[
                [
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
            ]
            f.write(top_df.to_markdown(index=False))
            f.write("\n\n")

            f.write(f"## 性能分布\n\n")
            f.write(f"### Sharpe比率分布\n")
            f.write(f"- 均值: {results_df['sharpe_ratio'].mean():.4f}\n")
            f.write(f"- 中位数: {results_df['sharpe_ratio'].median():.4f}\n")
            f.write(f"- 75分位: {results_df['sharpe_ratio'].quantile(0.75):.4f}\n")
            f.write(f"- 90分位: {results_df['sharpe_ratio'].quantile(0.90):.4f}\n\n")

            f.write(f"### 年化收益分布\n")
            f.write(f"- 均值: {results_df['annual_return'].mean():.2%}\n")
            f.write(f"- 中位数: {results_df['annual_return'].median():.2%}\n")
            f.write(f"- 75分位: {results_df['annual_return'].quantile(0.75):.2%}\n")
            f.write(f"- 90分位: {results_df['annual_return'].quantile(0.90):.2%}\n\n")
