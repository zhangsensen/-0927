"""
约束集成WFO优化器 | Constrained Walk-Forward Optimizer

将因子约束条件集成到WFO框架中:
  1. IS阶段应用因子约束进行筛选
  2. OOS阶段评估选中因子性能
  3. 输出约束应用日志和报告

使用流程:
  1. 创建约束选择器
  2. 使用WFO进行前向回测
  3. 在每个IS窗口应用约束进行因子筛选
  4. 生成完整报告

作者: Step 5 Constrained WFO
日期: 2025-10-26
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .factor_selector import FactorSelector, create_default_selector
from .ic_calculator import ICCalculator
from .walk_forward_optimizer import WalkForwardOptimizer

logger = logging.getLogger(__name__)


@dataclass
class ConstraintApplicationReport:
    """约束应用报告"""

    window_index: int
    """窗口索引"""

    is_start: int
    """IS窗口起始日期"""

    is_end: int
    """IS窗口结束日期"""

    oos_start: int
    """OOS窗口起始日期"""

    oos_end: int
    """OOS窗口结束日期"""

    is_ic_stats: Dict[str, float]
    """IS阶段IC统计"""

    candidate_factors: List[str]
    """候选因子列表"""

    selected_factors: List[str]
    """筛选后因子列表"""

    constraint_violations: List[str]
    """约束违反记录"""

    oos_performance: Dict[str, float]
    """OOS阶段性能"""

    selection_ic_mean: float
    """选中因子的平均IC"""

    oos_ic_mean: float
    """OOS阶段平均IC"""


class ConstrainedWalkForwardOptimizer:
    """
    约束集成WFO优化器

    在WFO框架中集成因子约束条件:
    - IS阶段计算IC并应用约束筛选
    - OOS阶段评估选中因子性能
    """

    def __init__(self, selector: Optional[FactorSelector] = None, verbose: bool = True):
        """
        初始化约束WFO优化器

        Args:
            selector: 因子选择器，若为None则创建默认选择器
            verbose: 是否打印详细日志
        """
        self.selector = selector or create_default_selector()
        self.ic_calculator = ICCalculator()
        self.wfo = WalkForwardOptimizer()
        self.verbose = verbose
        self.window_reports = []

        if verbose:
            logging.basicConfig(level=logging.INFO)

    def run_constrained_wfo(
        self,
        factors_data: np.ndarray,
        returns: np.ndarray,
        factor_names: Optional[List[str]] = None,
        is_period: int = 100,
        oos_period: int = 20,
        step_size: int = 20,
        target_factor_count: int = 5,
    ) -> Tuple[pd.DataFrame, List[ConstraintApplicationReport]]:
        """
        运行约束条件集成的WFO优化

        Args:
            factors_data: 因子数据 (time_steps, num_assets, num_factors)
            returns: 收益率 (time_steps, num_assets)
            factor_names: 因子名称列表
            is_period: IS窗口长度
            oos_period: OOS窗口长度
            step_size: 滑动步长
            target_factor_count: 目标因子数量

        Returns:
            (前向性能汇总DataFrame, 窗口应用报告列表)
        """
        num_time_steps, num_assets, num_factors = factors_data.shape

        if factor_names is None:
            factor_names = [f"FACTOR_{i}" for i in range(num_factors)]

        if self.verbose:
            logger.info(f"开始约束WFO优化")
            logger.info(
                f"数据形状: {num_time_steps} 日期 × {num_assets} 资产 × {num_factors} 因子"
            )
            logger.info(
                f"IS周期: {is_period}, OOS周期: {oos_period}, 步长: {step_size}"
            )

        # 划分窗口
        windows = self._partition_windows(
            num_time_steps, is_period, oos_period, step_size
        )

        forward_performances = []

        for window_idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
            if self.verbose:
                logger.info(
                    f"\n【窗口 {window_idx + 1}/{len(windows)}】IS: [{is_start}, {is_end}), OOS: [{oos_start}, {oos_end})"
                )

            # 心跳机制 - 防止长时间无响应
            if window_idx % 10 == 0 and window_idx > 0:
                logger.info(
                    f"🔄 进度: {window_idx}/{len(windows)} 窗口完成 ({window_idx/len(windows)*100:.1f}%)"
                )

            # IS阶段
            is_factors = factors_data[is_start:is_end]
            is_returns = returns[is_start:is_end]

            # 计算IC
            ic_scores = self._compute_window_ic(is_factors, is_returns, factor_names)

            # 应用约束筛选
            selected_factors, selection_report = self.selector.select_factors(
                ic_scores, target_count=target_factor_count
            )

            if self.verbose:
                logger.info(f"筛选: {len(ic_scores)} → {len(selected_factors)} 因子")

            # OOS阶段
            oos_factors = factors_data[oos_start:oos_end]
            oos_returns = returns[oos_start:oos_end]

            if len(selected_factors) > 0:
                # 计算OOS性能
                oos_ic_scores = self._compute_window_ic(
                    oos_factors, oos_returns, factor_names
                )

                selected_oos_ics = {
                    f: oos_ic_scores.get(f, 0.0) for f in selected_factors
                }

                oos_ic_mean = np.mean(list(selected_oos_ics.values()))
            else:
                oos_ic_mean = 0.0
                selected_oos_ics = {}

            # 创建窗口报告
            report = ConstraintApplicationReport(
                window_index=window_idx,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
                is_ic_stats={
                    "mean": np.mean(list(ic_scores.values())),
                    "median": np.median(list(ic_scores.values())),
                    "std": np.std(list(ic_scores.values())),
                    "min": np.min(list(ic_scores.values())),
                    "max": np.max(list(ic_scores.values())),
                },
                candidate_factors=list(ic_scores.keys()),
                selected_factors=selected_factors,
                constraint_violations=self._extract_violations(selection_report),
                oos_performance=selected_oos_ics,
                selection_ic_mean=(
                    np.mean([ic_scores[f] for f in selected_factors])
                    if selected_factors
                    else 0.0
                ),
                oos_ic_mean=oos_ic_mean,
            )

            self.window_reports.append(report)

            forward_performances.append(
                {
                    "window": window_idx,
                    "is_start": is_start,
                    "is_end": is_end,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                    "is_ic_mean": report.is_ic_stats["mean"],
                    "selected_factor_count": len(selected_factors),
                    "selected_factors": ",".join(selected_factors),
                    "selection_ic_mean": report.selection_ic_mean,
                    "oos_ic_mean": report.oos_ic_mean,
                    "ic_drop": report.selection_ic_mean - report.oos_ic_mean,
                }
            )

        # 汇总结果
        forward_df = pd.DataFrame(forward_performances)

        if self.verbose:
            self._print_summary(forward_df)

        return forward_df, self.window_reports

    def _partition_windows(
        self, num_time_steps: int, is_period: int, oos_period: int, step_size: int
    ) -> List[Tuple[int, int, int, int]]:
        """划分IS/OOS窗口"""
        windows = []

        for start in range(0, num_time_steps - is_period - oos_period + 1, step_size):
            is_start = start
            is_end = start + is_period
            oos_start = is_end
            oos_end = oos_start + oos_period

            if oos_end <= num_time_steps:
                windows.append((is_start, is_end, oos_start, oos_end))

        return windows

    def _compute_window_ic(
        self, factors: np.ndarray, returns: np.ndarray, factor_names: List[str]
    ) -> Dict[str, float]:
        """计算窗口内因子IC"""
        ic_scores = {}

        for i, name in enumerate(factor_names):
            factor_series = factors[:, :, i]

            # 计算平均的IC (对所有资产)
            ics = []
            for j in range(factor_series.shape[1]):
                factor_col = factor_series[:, j]
                returns_col = returns[:, j]

                # 计算相关系数
                corr = np.corrcoef(factor_col, returns_col)[0, 1]
                if not np.isnan(corr):
                    ics.append(corr)

            # 取平均IC
            ic_scores[name] = np.mean(ics) if ics else 0.0

        return ic_scores

    def _extract_violations(self, report: Any) -> List[str]:
        """提取约束违反记录"""
        violations = []

        if hasattr(report, "violations"):
            for v in report.violations:
                if isinstance(v, dict):
                    violations.append(
                        f"{v.get('type', 'unknown')}: {v.get('reason', '')}"
                    )
                else:
                    violations.append(str(v))

        return violations

    def _print_summary(self, forward_df: pd.DataFrame):
        """打印汇总结果"""
        print("\n" + "=" * 80)
        print("约束WFO优化结果汇总")
        print("=" * 80)

        if len(forward_df) > 0:
            print(f"\n窗口总数: {len(forward_df)}")
            print(f"\n每窗口平均:")
            print(f"  IS IC均值:        {forward_df['is_ic_mean'].mean():.6f}")
            print(
                f"  选中因子数:       {forward_df['selected_factor_count'].mean():.1f}"
            )
            print(f"  选中因子IC:       {forward_df['selection_ic_mean'].mean():.6f}")
            print(f"  OOS IC:           {forward_df['oos_ic_mean'].mean():.6f}")
            print(f"  IC衰减幅度:       {forward_df['ic_drop'].mean():.6f}")

            print(f"\nIC衰减分布:")
            print(f"  最小: {forward_df['ic_drop'].min():.6f}")
            print(f"  最大: {forward_df['ic_drop'].max():.6f}")
            print(f"  标准差: {forward_df['ic_drop'].std():.6f}")

        print("\n" + "=" * 80)


def main():
    """示例用法"""
    np.random.seed(42)

    # 生成测试数据
    num_time_steps = 500
    num_assets = 30
    num_factors = 10

    # 创建因子数据（高斯分布）
    factors = np.random.randn(num_time_steps, num_assets, num_factors) * 0.1

    # 创建收益率（与因子相关联）
    returns = np.zeros((num_time_steps, num_assets))
    for i in range(num_time_steps):
        # 前3个因子与收益率高度相关
        returns[i] = 0.3 * np.mean(factors[i, :, 0:3], axis=1) + 0.1 * np.random.randn(
            num_assets
        )

    factor_names = [f"FACTOR_{i}" for i in range(num_factors)]

    # 创建优化器
    selector = create_default_selector()
    optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=True)

    # 运行约束WFO
    forward_df, window_reports = optimizer.run_constrained_wfo(
        factors_data=factors,
        returns=returns,
        factor_names=factor_names,
        is_period=100,
        oos_period=20,
        step_size=20,
        target_factor_count=5,
    )

    print("\n前向回测汇总:")
    print(forward_df.to_string())


if __name__ == "__main__":
    main()
