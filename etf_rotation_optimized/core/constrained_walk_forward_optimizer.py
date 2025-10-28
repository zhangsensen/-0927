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

        # Factor Momentum: 记录历史 OOS IC（key=因子名, value=OOS IC列表）
        self.historical_oos_ics: Dict[str, List[float]] = {}

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

            # IS阶段：因子提前1天切片以实现T-1→T对齐
            # 因子: [is_start-1, is_end-1) 长度 = is_end - is_start
            # 收益: [is_start, is_end)     长度 = is_end - is_start
            # 配对关系: factors[t] 预测 returns[t]
            is_factor_start = max(0, is_start - 1)
            is_factor_end = max(0, is_end - 1)  # 注意：slice的右边界是开区间
            is_factors = factors_data[is_factor_start:is_factor_end]
            is_returns = returns[is_start:is_end]

            # 轻量日志：确认 T-1 切片已生效
            if window_idx == 0 or (self.verbose and window_idx % 10 == 0):
                logger.debug(
                    f"[窗口{window_idx+1}] IS 使用 T-1 切片: "
                    f"因子[{is_factor_start}:{is_factor_end}), 收益[{is_start}:{is_end})"
                )

            # 计算IC（因子与收益索引已对齐）
            ic_scores = self._compute_window_ic(is_factors, is_returns, factor_names)

            # 计算因子相关矩阵（用于相关性去重）
            factor_correlations = self._compute_factor_correlations(
                is_factors, factor_names
            )

            meta_cfg = (
                self.selector.constraints.get("meta_factor_weighting", {})
                if hasattr(self.selector, "constraints")
                else {}
            )
            use_meta = (
                bool(meta_cfg.get("enabled", False))
                and meta_cfg.get("mode", "") == "icir"
            )
            factor_icir: Dict[str, float] = {}
            if use_meta and self.historical_oos_ics:
                k = int(meta_cfg.get("windows", 20))
                min_w = int(meta_cfg.get("min_windows", 5))
                std_floor = float(meta_cfg.get("std_floor", 0.005))
                for name in factor_names:
                    hist = self.historical_oos_ics.get(name, [])
                    if len(hist) >= min_w:
                        arr = np.array(hist[-k:]) if k > 0 else np.array(hist)
                        m = float(np.mean(arr))
                        s = float(np.std(arr))
                        s = s if s >= std_floor else std_floor
                        factor_icir[name] = m / s
                    else:
                        factor_icir[name] = 0.0

            selected_factors, selection_report = self.selector.select_factors(
                ic_scores,
                factor_correlations=factor_correlations,
                target_count=target_factor_count,
                historical_oos_ics=self.historical_oos_ics,
                factor_icir=factor_icir if use_meta else None,
            )

            if self.verbose:
                logger.info(f"筛选: {len(ic_scores)} → {len(selected_factors)} 因子")

            # OOS阶段：因子提前1天切片以实现T-1→T对齐
            oos_factor_start = max(0, oos_start - 1)
            oos_factor_end = max(0, oos_end - 1)
            oos_factors = factors_data[oos_factor_start:oos_factor_end]
            oos_returns = returns[oos_start:oos_end]

            # 防御性初始化，避免空选择时未定义
            oos_ic_scores = {}

            if len(selected_factors) > 0:
                # 计算OOS性能（因子与收益索引已对齐）
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
                # ⚠️ Linus Fix: 使用排序后的候选列表（反映Meta Factor调整）
                candidate_factors=selection_report.candidate_factors,
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

            # Factor Momentum: 记录当前窗口【所有因子】的 OOS IC（解决未入选因子 ICIR=0 锁死问题）
            # 🔪 Linus Fix: 不只记录已选因子，记录全部因子，给未来翻盘机会
            for factor_name in factor_names:
                oos_ic = oos_ic_scores.get(factor_name, 0.0)  # 未计算的默认0
                if factor_name not in self.historical_oos_ics:
                    self.historical_oos_ics[factor_name] = []
                self.historical_oos_ics[factor_name].append(oos_ic)

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

    def _compute_factor_correlations(
        self, factors: np.ndarray, factor_names: List[str]
    ) -> Dict[Tuple[str, str], float]:
        """
        计算因子相关矩阵（用于相关性去重）

        参数:
            factors: shape (n_days, n_assets, n_factors)
            factor_names: 因子名称列表

        返回:
            {(factor1, factor2): correlation} 字典
        """
        from scipy.stats import spearmanr

        n_days, n_assets, n_factors = factors.shape

        # 展平为 (n_days * n_assets, n_factors) 以计算因子间相关
        factors_flat = factors.reshape(-1, n_factors)  # (n_days*n_assets, n_factors)

        # 计算 Spearman 相关矩阵（因子维度）
        corr_matrix, _ = spearmanr(factors_flat, axis=0, nan_policy="omit")

        # 转为字典格式 {(f1, f2): corr}
        # ⚠️ Linus Fix: 使用字母排序key，与factor_selector查找逻辑一致
        correlations = {}
        for i in range(n_factors):
            for j in range(i + 1, n_factors):  # 只存储上三角（避免重复）
                # 使用字母排序确保key规范一致
                key = tuple(sorted([factor_names[i], factor_names[j]]))
                correlations[key] = corr_matrix[i, j]

        return correlations

    def _compute_window_ic(
        self, factors: np.ndarray, returns: np.ndarray, factor_names: List[str]
    ) -> Dict[str, float]:
        """计算窗口内因子IC（与Step4一致的口径）

        定义：日频横截面 Spearman IC，使用 T-1 因子预测 T 日收益。
        做法：按日在资产维度做秩相关，得到每日IC后取均值。
        注意：输入的 factors/returns 应已错位对齐（factors 比 returns 提前1天切片）。
             函数会自动处理长度不等（当窗口起点无法再提前时）的情况。
        """
        from scipy.stats import spearmanr

        ic_scores = {}

        n_factors_days, n_assets, n_factor_cols = factors.shape
        n_returns_days = returns.shape[0]

        # 取两者的最小长度，避免越界
        n_days = min(n_factors_days, n_returns_days)

        # 轻量校验日志：边界保护触发提示（仅在长度不等时记录一次）
        if n_factors_days != n_returns_days:
            logger.debug(
                f"[边界保护] 因子天数({n_factors_days}) != 收益天数({n_returns_days})，"
                f"使用 min={n_days}（T-1 对齐的边界情况）"
            )

        for i, name in enumerate(factor_names):
            factor_ts = factors[:, :, i]  # (n_days, n_assets)

            daily_ics: List[float] = []
            # 直接逐日配对（外部已保证因子提前1天切片）
            for t in range(n_days):
                signal_t = factor_ts[t, :]
                ret_t = returns[t, :]

                # 有效掩码
                valid_mask = ~(np.isnan(signal_t) | np.isnan(ret_t))
                if valid_mask.sum() < 2:
                    continue

                ic, _ = spearmanr(signal_t[valid_mask], ret_t[valid_mask])
                if not np.isnan(ic):
                    daily_ics.append(float(ic))

            ic_scores[name] = float(np.mean(daily_ics)) if daily_ics else 0.0

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
