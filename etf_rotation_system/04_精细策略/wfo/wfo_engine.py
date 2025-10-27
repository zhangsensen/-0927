#!/usr/bin/env python3
"""
WFO引擎
负责WFO周期管理、滚动验证的核心逻辑
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .wfo_config import WFOConfig, WFOPeriod, add_months

logger = logging.getLogger(__name__)


class WFOEngine:
    """WFO引擎 - 管理整个WFO流程"""

    def __init__(self, config: WFOConfig):
        """
        初始化WFO引擎

        Args:
            config: WFO配置
        """
        self.config = config
        self.periods: List[WFOPeriod] = []
        self.results = {
            "config": config.to_dict(),
            "periods": [],
            "summary": {},
            "overfit_analysis": {},
        }

        logger.info(f"WFO引擎初始化完成")
        logger.info(f"训练窗口: {config.train_window_months}月")
        logger.info(f"测试窗口: {config.test_window_months}月")
        logger.info(f"步进: {config.step_months}月")

    def generate_periods(self) -> List[WFOPeriod]:
        """
        生成所有WFO周期

        Returns:
            WFO周期列表
        """
        if self.config.data_start_date is None or self.config.data_end_date is None:
            raise ValueError("必须设置data_start_date和data_end_date")

        periods = []
        period_id = 1
        current_start = self.config.data_start_date

        while True:
            # 计算训练窗口
            train_start = current_start
            train_end = add_months(train_start, self.config.train_window_months)

            # 计算测试窗口（测试开始日期应该在训练结束后）
            test_start = train_end + pd.Timedelta(days=1)
            test_end = add_months(test_start, self.config.test_window_months)

            # 检查是否超出数据范围
            if test_end > self.config.data_end_date:
                break

            # 创建周期
            period = WFOPeriod(
                period_id=period_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            periods.append(period)

            logger.info(
                f"Period {period_id}: Train[{train_start.date()} ~ {train_end.date()}] "
                f"Test[{test_start.date()} ~ {test_end.date()}]"
            )

            # 前进一步
            current_start = add_months(current_start, self.config.step_months)
            period_id += 1

        self.periods = periods
        logger.info(f"总共生成 {len(periods)} 个WFO周期")

        return periods

    def run_wfo_pipeline(
        self,
        backtest_runner,
        optimizer,
        analyzer,
        base_strategies: List[Dict],
    ) -> Dict:
        """
        运行完整WFO流程

        Args:
            backtest_runner: 回测运行器
            optimizer: 优化器
            analyzer: 分析器
            base_strategies: 基础策略集合（从VBT结果提取）

        Returns:
            WFO结果字典
        """
        logger.info("=" * 80)
        logger.info("开始WFO流程")
        logger.info("=" * 80)

        if not self.periods:
            logger.warning("未生成WFO周期，先调用generate_periods()")
            self.generate_periods()

        # 遍历每个周期
        for period in self.periods:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Period {period.period_id}/{len(self.periods)}")
            logger.info(f"{'=' * 80}")

            try:
                # 1. IS阶段：优化
                is_results = self._run_in_sample_phase(
                    period, backtest_runner, optimizer, base_strategies
                )
                period.is_results = is_results

                # 2. 选择最优策略
                best_strategies = self._select_best_strategies(
                    is_results, self.config.top_n_strategies
                )
                period.best_strategy = best_strategies[0] if best_strategies else None

                # 3. OOS阶段：验证
                oos_results = self._run_out_of_sample_phase(
                    period, backtest_runner, best_strategies
                )
                period.oos_results = oos_results

                # 4. 保存周期结果
                self.results["periods"].append(
                    {
                        "period": period.to_dict(),
                        "is_best_sharpe": is_results.get("best_sharpe", 0),
                        "oos_avg_sharpe": oos_results.get("avg_sharpe", 0),
                        "overfit_ratio": is_results.get("best_sharpe", 0)
                        / max(oos_results.get("avg_sharpe", 0.01), 0.01),
                    }
                )

                logger.info(f"Period {period.period_id} 完成:")
                logger.info(f"  IS最佳Sharpe: {is_results.get('best_sharpe', 0):.3f}")
                logger.info(f"  OOS平均Sharpe: {oos_results.get('avg_sharpe', 0):.3f}")
                logger.info(
                    f"  过拟合比: {self.results['periods'][-1]['overfit_ratio']:.3f}"
                )

            except Exception as e:
                logger.error(f"Period {period.period_id} 执行失败: {e}")
                continue

        # 5. 汇总分析
        self.results["summary"] = self._aggregate_results()

        # 6. 过拟合分析
        self.results["overfit_analysis"] = analyzer.analyze_overfitting(self.periods)

        logger.info("\n" + "=" * 80)
        logger.info("WFO流程完成")
        logger.info("=" * 80)

        return self.results

    def _run_in_sample_phase(
        self,
        period: WFOPeriod,
        backtest_runner,
        optimizer,
        base_strategies: List[Dict],
    ) -> Dict:
        """
        运行IS阶段（样本内优化）

        Args:
            period: 当前周期
            backtest_runner: 回测运行器
            optimizer: 优化器
            base_strategies: 基础策略

        Returns:
            IS结果字典
        """
        logger.info(
            f"IS阶段: 训练 [{period.train_start.date()} ~ {period.train_end.date()}]"
        )

        # 在训练窗口上回测所有候选策略
        results = backtest_runner.run_batch_backtest(
            strategies=base_strategies,
            start_date=period.train_start,
            end_date=period.train_end,
        )

        # 排序并返回
        sorted_results = sorted(
            results, key=lambda x: x.get("sharpe_ratio", 0), reverse=True
        )

        return {
            "all_results": sorted_results,
            "best_sharpe": sorted_results[0]["sharpe_ratio"] if sorted_results else 0,
            "best_strategy": sorted_results[0] if sorted_results else None,
        }

    def _select_best_strategies(self, is_results: Dict, top_n: int) -> List[Dict]:
        """
        从IS结果中选择最优策略

        Args:
            is_results: IS结果
            top_n: 选择数量

        Returns:
            最优策略列表
        """
        all_results = is_results.get("all_results", [])

        # 过滤：IS Sharpe必须达到阈值
        filtered = [
            s
            for s in all_results
            if s.get("sharpe_ratio", 0) >= self.config.min_is_sharpe
        ]

        # 取Top-N
        selected = filtered[:top_n]

        logger.info(f"从 {len(all_results)} 个策略中筛选 {len(selected)} 个进入OOS")

        return selected

    def _run_out_of_sample_phase(
        self,
        period: WFOPeriod,
        backtest_runner,
        strategies: List[Dict],
    ) -> Dict:
        """
        运行OOS阶段（样本外验证）

        Args:
            period: 当前周期
            backtest_runner: 回测运行器
            strategies: 待验证策略

        Returns:
            OOS结果字典
        """
        logger.info(
            f"OOS阶段: 验证 [{period.test_start.date()} ~ {period.test_end.date()}]"
        )

        if not strategies:
            logger.warning("没有策略进入OOS阶段")
            return {"all_results": [], "avg_sharpe": 0}

        # 在测试窗口上回测选定策略
        results = backtest_runner.run_batch_backtest(
            strategies=strategies,
            start_date=period.test_start,
            end_date=period.test_end,
        )

        # 计算平均表现
        avg_sharpe = (
            sum(r.get("sharpe_ratio", 0) for r in results) / len(results)
            if results
            else 0
        )

        return {
            "all_results": results,
            "avg_sharpe": avg_sharpe,
            "best_sharpe": (
                max(r.get("sharpe_ratio", 0) for r in results) if results else 0
            ),
        }

    def _aggregate_results(self) -> Dict:
        """汇总所有周期结果"""
        if not self.results["periods"]:
            return {}

        is_sharpes = [p["is_best_sharpe"] for p in self.results["periods"]]
        oos_sharpes = [p["oos_avg_sharpe"] for p in self.results["periods"]]
        overfit_ratios = [p["overfit_ratio"] for p in self.results["periods"]]

        # 计算统计量
        import numpy as np

        summary = {
            "total_periods": len(self.periods),
            "is_sharpe_mean": np.mean(is_sharpes),
            "is_sharpe_std": np.std(is_sharpes),
            "oos_sharpe_mean": np.mean(oos_sharpes),
            "oos_sharpe_std": np.std(oos_sharpes),
            "avg_overfit_ratio": np.mean(overfit_ratios),
            "max_overfit_ratio": np.max(overfit_ratios),
            "oos_win_rate": sum(
                1 for s in oos_sharpes if s >= self.config.min_oos_sharpe
            )
            / len(oos_sharpes),
            "performance_decay": (
                (np.mean(is_sharpes) - np.mean(oos_sharpes)) / np.mean(is_sharpes)
                if np.mean(is_sharpes) > 0
                else 0
            ),
        }

        logger.info("\n" + "=" * 80)
        logger.info("WFO汇总结果:")
        logger.info(f"  总周期数: {summary['total_periods']}")
        logger.info(
            f"  IS平均Sharpe: {summary['is_sharpe_mean']:.3f} ± {summary['is_sharpe_std']:.3f}"
        )
        logger.info(
            f"  OOS平均Sharpe: {summary['oos_sharpe_mean']:.3f} ± {summary['oos_sharpe_std']:.3f}"
        )
        logger.info(f"  平均过拟合比: {summary['avg_overfit_ratio']:.3f}")
        logger.info(f"  OOS胜率: {summary['oos_win_rate']*100:.1f}%")
        logger.info(f"  性能衰减: {summary['performance_decay']*100:.1f}%")
        logger.info("=" * 80)

        return summary

    def save_results(self, output_path: Path):
        """保存WFO结果"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"WFO结果已保存: {output_path}")
