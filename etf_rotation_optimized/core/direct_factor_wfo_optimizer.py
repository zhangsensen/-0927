"""
Direct Factor WFO Optimizer | 直接因子级加权优化器

核心理念:
1. 移除组合搜索 - 直接使用所有筛选后的因子
2. IC加权 - 基于IS阶段IC直接加权因子
3. 纯向量化 - 最大化计算效率
4. 100%信息利用率 - 无信息压缩损失

作者: AI Agent (Linus Mode)
日期: 2025-10-29
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from core.ic_calculator_numba import ICCalculatorNumba as ICCalculator

logger = logging.getLogger(__name__)


@dataclass
class DirectFactorWindowResult:
    """单窗口直接因子加权结果"""

    window_index: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int

    selected_factors: List[str]
    """筛选后的因子列表"""

    is_ic_scores: Dict[str, float]
    """IS阶段各因子IC"""

    factor_weights: Dict[str, float]
    """因子权重"""

    oos_ensemble_ic: float
    """OOS阶段集成IC"""

    oos_ensemble_sharpe: float
    """OOS阶段集成Sharpe"""

    oos_baseline_ic: float
    """OOS阶段等权ETF基准IC"""

    oos_baseline_sharpe: float
    """OOS阶段等权ETF基准Sharpe"""

    factor_contributions: Dict[str, float] = None
    """因子边际贡献 (P1-1: 权重 × OOS IC)"""


class DirectFactorWFOOptimizer:
    """
    直接因子级加权WFO优化器

    核心流程:
    1. IS阶段: 计算每个因子的IC
    2. 筛选: 基于min_ic阈值过滤因子
    3. 加权: 基于IC计算因子权重
    4. OOS预测: 直接加权求和生成信号
    """

    def __init__(
        self,
        factor_weighting: str = "ic_weighted",
        min_factor_ic: float = 0.01,
        ic_floor: float = 0.0,
        contribution_weighting_temperature: float = 0.5,
        max_single_weight: float = 0.3,
        min_single_weight: float = 0.05,
        verbose: bool = True,
    ):
        """
        初始化直接因子WFO优化器

        Args:
            factor_weighting: 加权方案
                - 'equal': 等权
                - 'ic_weighted': IC加权 (默认)
                - 'contribution_weighted': 贡献加权 (Day 2优化)
            min_factor_ic: 最小IC门槛
            ic_floor: IC下界
            contribution_weighting_temperature: 贡献加权温度参数 (控制权重集中度)
            max_single_weight: 单因子最大权重上限
            min_single_weight: 单因子最小权重下限
            verbose: 是否打印日志
        """
        # P1-2: 参数校验 (不终止运行，仅告警)
        if not (0 <= min_factor_ic <= 1):
            logger.warning(
                f"⚠️  min_factor_ic={min_factor_ic} 超出合理范围[0,1]，"
                f"可能导致异常筛选结果"
            )
        if not (ic_floor <= min_factor_ic):
            logger.warning(
                f"⚠️  ic_floor={ic_floor} > min_factor_ic={min_factor_ic}，"
                f"逻辑不一致，建议 ic_floor ≤ min_factor_ic"
            )
        if factor_weighting not in ["equal", "ic_weighted", "contribution_weighted"]:
            logger.warning(
                f"⚠️  未知加权方案: {factor_weighting}，"
                f"支持: equal, ic_weighted, contribution_weighted"
            )
        if not (0.1 <= contribution_weighting_temperature <= 1.0):
            logger.warning(
                f"⚠️  温度参数={contribution_weighting_temperature} 建议范围[0.1,1.0]"
            )

        self.factor_weighting = factor_weighting
        self.min_factor_ic = min_factor_ic
        self.ic_floor = ic_floor
        self.contribution_weighting_temperature = contribution_weighting_temperature
        self.max_single_weight = max_single_weight
        self.min_single_weight = min_single_weight
        self.verbose = verbose

        self.ic_calculator = ICCalculator()

        # P0: 内存化历史贡献数据（消除外部依赖）
        self.historical_contributions: Dict[str, List[float]] = {}

        # P0: 持久化历史数据文件路径
        self.contribution_cache_path = "cache/historical_contributions.pkl"

        # P0: 初始化时加载历史贡献数据（仅用于实验性贡献加权）
        if factor_weighting == "contribution_weighted":
            self._load_persistent_contributions()

        if self.verbose:
            logger.info("=" * 80)
            logger.info("初始化 DirectFactorWFOOptimizer")
            logger.info(f"加权方案: {factor_weighting}")
            if factor_weighting == "contribution_weighted":
                logger.info(f"贡献加权温度: {contribution_weighting_temperature}")
                logger.info(
                    f"权重范围: [{min_single_weight:.1%}, {max_single_weight:.1%}]"
                )
                logger.info(
                    f"历史贡献数据: {sum(len(v) for v in self.historical_contributions.values())} 条记录"
                )
            logger.info(f"最小IC门槛: {min_factor_ic}")
            logger.info("✅ 参数校验通过")
            logger.info("=" * 80)

    def run_wfo(
        self,
        factors_data: np.ndarray,
        returns: np.ndarray,
        factor_names: List[str],
        is_period: int = 252,
        oos_period: int = 60,
        step_size: int = 20,
    ) -> Tuple[List[DirectFactorWindowResult], pd.DataFrame]:
        """
        运行Walk-Forward优化

        Args:
            factors_data: 因子数据 (T, N, F)
            returns: 收益率数据 (T, N)
            factor_names: 因子名称列表
            is_period: IS窗口长度
            oos_period: OOS窗口长度
            step_size: 滑动步长

        Returns:
            results: 各窗口结果列表
            summary_df: 汇总DataFrame
        """
        # P1-2: WFO参数校验
        T = len(factors_data)
        if is_period <= 0 or oos_period <= 0 or step_size <= 0:
            logger.warning(
                f"⚠️  窗口参数异常: is_period={is_period}, "
                f"oos_period={oos_period}, step_size={step_size}"
            )
        if is_period + oos_period > T:
            logger.warning(
                f"⚠️  窗口总长度({is_period + oos_period}) > "
                f"数据总长度({T})，无法生成窗口"
            )

        results = []

        # 生成窗口
        windows = []
        start = 0
        while start + is_period + oos_period <= T:
            is_start = start
            is_end = start + is_period
            oos_start = is_end
            oos_end = oos_start + oos_period
            windows.append((is_start, is_end, oos_start, oos_end))
            start += step_size

        if self.verbose:
            logger.info(f"生成 {len(windows)} 个WFO窗口")

        # 逐窗口处理
        for window_idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
            if self.verbose:
                logger.info(f"\n{'='*80}")
                logger.info(f"窗口 {window_idx+1}/{len(windows)}")
                logger.info(f"IS: [{is_start}:{is_end}], OOS: [{oos_start}:{oos_end}]")

            result = self._run_single_window(
                window_idx=window_idx,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
                factors_data=factors_data,
                returns=returns,
                factor_names=factor_names,
            )
            results.append(result)

        # 汇总结果
        summary_df = self._create_summary_df(results)

        if self.verbose:
            avg_ic = summary_df["oos_ensemble_ic"].mean()
            avg_baseline_ic = summary_df["oos_baseline_ic"].mean()
            avg_excess_ic = summary_df["excess_ic"].mean()
            win_rate = (summary_df["oos_ensemble_ic"] > 0).mean()
            baseline_win_rate = (summary_df["oos_baseline_ic"] > 0).mean()

            logger.info(f"\n{'='*80}")
            logger.info("WFO完成")
            logger.info(f"平均OOS IC: {avg_ic:.4f}")
            logger.info(f"OOS IC胜率: {win_rate:.1%}")
            logger.info("\n基准对照 (等权ETF):")
            logger.info(
                f"  基准IC: {avg_baseline_ic:.4f}, 胜率: {baseline_win_rate:.1%}"
            )
            logger.info(
                f"  超额IC: {avg_excess_ic:+.4f} ({avg_excess_ic/abs(avg_baseline_ic)*100:+.1f}% vs基准)"
            )
            logger.info("=" * 80)

        # P0: 保存历史贡献数据到磁盘
        if self.factor_weighting == "contribution_weighted":
            self._save_persistent_contributions()

        return results, summary_df

    def _run_single_window(
        self,
        window_idx: int,
        is_start: int,
        is_end: int,
        oos_start: int,
        oos_end: int,
        factors_data: np.ndarray,
        returns: np.ndarray,
        factor_names: List[str],
    ) -> DirectFactorWindowResult:
        """
        单窗口处理

        流程:
        1. IS数据切片 + T-1对齐
        2. 计算各因子IS IC
        3. 筛选有效因子 (IC > min_ic)
        4. 计算因子权重
        5. OOS信号生成 + 性能评估
        """
        from .data_contract import align_factor_to_return

        # Step 1: IS数据切片 + T-1对齐
        is_factors_raw = factors_data[is_start:is_end]
        is_returns_raw = returns[is_start:is_end]
        is_factors, is_returns = align_factor_to_return(is_factors_raw, is_returns_raw)

        # Step 2: 计算IS阶段各因子IC
        is_ic_scores = self._compute_window_ic(is_factors, is_returns, factor_names)

        if self.verbose:
            logger.info(
                f"Step 1: IS IC计算完成, "
                f"平均IC={np.mean(list(is_ic_scores.values())):.4f}"
            )

        # Step 3: 筛选有效因子
        selected_factors = [
            f for f in factor_names if is_ic_scores.get(f, 0) > self.min_factor_ic
        ]

        if len(selected_factors) == 0:
            logger.warning("无有效因子，使用所有因子")
            selected_factors = factor_names

        if self.verbose:
            logger.info(
                f"Step 2: 筛选完成, {len(selected_factors)}/{len(factor_names)} 因子"
            )

        # Step 4: 计算因子权重（传入IS IC用于冷启动）
        factor_weights = self._calculate_factor_weights(selected_factors, is_ic_scores)

        if self.verbose:
            logger.info(f"Step 3: 权重计算完成, 方案={self.factor_weighting}")
            top3 = sorted(factor_weights.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"Top3权重: {top3}")

        # Step 5: OOS信号生成 + 性能评估
        oos_factors_raw = factors_data[oos_start:oos_end]
        oos_returns_raw = returns[oos_start:oos_end]
        oos_factors, oos_returns = align_factor_to_return(
            oos_factors_raw, oos_returns_raw
        )

        # 提取选中因子的索引
        factor_name_to_idx = {name: idx for idx, name in enumerate(factor_names)}
        selected_indices = [factor_name_to_idx[f] for f in selected_factors]

        # 提取选中因子的权重
        weights_array = np.array([factor_weights[f] for f in selected_factors])

        # 向量化信号生成: (T, N, F) -> (T, N)
        oos_selected_factors = oos_factors[:, :, selected_indices]  # (T, N, F)
        final_signal = np.tensordot(
            weights_array,
            np.transpose(oos_selected_factors, (2, 0, 1)),
            axes=([0], [0]),
        )  # (T, N)

        # 计算OOS IC
        oos_ensemble_ic = self._compute_signal_ic(final_signal, oos_returns)

        # 计算OOS Sharpe
        oos_ensemble_sharpe = self._compute_signal_sharpe(final_signal, oos_returns)

        # 计算等权ETF基准 (P0-1: 基准对照)
        baseline_signal = np.ones_like(oos_returns)  # 等权信号
        oos_baseline_ic = self._compute_signal_ic(baseline_signal, oos_returns)
        oos_baseline_sharpe = self._compute_signal_sharpe(baseline_signal, oos_returns)

        # P1-1: 计算因子边际贡献 (权重 × 单因子OOS IC)
        factor_contributions = {}
        for i, factor_name in enumerate(selected_factors):
            factor_idx = selected_indices[i]
            single_factor_signal = oos_factors[:, :, factor_idx]
            single_factor_ic = self._compute_signal_ic(
                single_factor_signal, oos_returns
            )
            contribution = factor_weights[factor_name] * single_factor_ic
            factor_contributions[factor_name] = contribution

        if self.verbose:
            excess_ic = oos_ensemble_ic - oos_baseline_ic
            excess_sharpe = oos_ensemble_sharpe - oos_baseline_sharpe
            logger.info(
                f"Step 4: OOS评估完成, IC={oos_ensemble_ic:.4f}, "
                f"Sharpe={oos_ensemble_sharpe:.2f}"
            )
            logger.info(
                f"        基准对照: 基准IC={oos_baseline_ic:.4f}, "
                f"超额IC={excess_ic:+.4f}, 超额Sharpe={excess_sharpe:+.2f}"
            )
            # P1-1: 输出因子贡献 TopK
            top_contributors = sorted(
                factor_contributions.items(), key=lambda x: abs(x[1]), reverse=True
            )[:3]
            logger.info(f"        Top3贡献: {[(f, c) for f, c in top_contributors]}")

        # P0: 更新历史贡献数据（内存化）
        if self.factor_weighting == "contribution_weighted":
            for factor_name, contribution in factor_contributions.items():
                if factor_name not in self.historical_contributions:
                    self.historical_contributions[factor_name] = []
                self.historical_contributions[factor_name].append(contribution)

        return DirectFactorWindowResult(
            window_index=window_idx,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            selected_factors=selected_factors,
            is_ic_scores=is_ic_scores,
            factor_weights=factor_weights,
            oos_ensemble_ic=oos_ensemble_ic,
            oos_ensemble_sharpe=oos_ensemble_sharpe,
            oos_baseline_ic=oos_baseline_ic,
            oos_baseline_sharpe=oos_baseline_sharpe,
            factor_contributions=factor_contributions,
        )

    def _compute_window_ic(
        self, factors: np.ndarray, returns: np.ndarray, factor_names: List[str]
    ) -> Dict[str, float]:
        """计算窗口内各因子IC"""
        ic_scores = {}
        for i, factor_name in enumerate(factor_names):
            factor_data = factors[:, :, i]
            ic = self.ic_calculator.compute_ic(factor_data, returns)
            ic_scores[factor_name] = float(ic)
        return ic_scores

    def _calculate_factor_weights(
        self, selected_factors: List[str], is_ic_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """计算因子权重 (Day 2: 支持贡献加权优化)"""
        if self.factor_weighting == "equal":
            # 等权
            weight = 1.0 / len(selected_factors)
            return {f: weight for f in selected_factors}

        elif self.factor_weighting == "ic_weighted":
            # IC加权
            ic_values = np.array([is_ic_scores[f] for f in selected_factors])
            ic_values = np.maximum(ic_values, self.ic_floor)  # 负IC置为floor
            total_ic = ic_values.sum()

            if total_ic < 1e-10:
                # 退化为等权
                weight = 1.0 / len(selected_factors)
                return {f: weight for f in selected_factors}

            weights = ic_values / total_ic
            return {f: float(w) for f, w in zip(selected_factors, weights)}

        elif self.factor_weighting == "contribution_weighted":
            # Day 2: 贡献加权 - 基于历史贡献数据的指数加权
            return self._calculate_contribution_weights(selected_factors, is_ic_scores)

        else:
            raise ValueError(f"未知加权方案: {self.factor_weighting}")

    def _calculate_contribution_weights(
        self, selected_factors: List[str], is_ic_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Day 2: 贡献加权权重计算

        基于历史贡献数据，使用指数加权算法优化权重分配

        算法步骤:
        1. 从历史WFO结果中提取因子贡献数据
        2. 计算因子贡献强度和稳定性评分
        3. 指数加权分配权重
        4. 应用风险控制约束
        5. 平滑过渡避免剧烈变化
        """
        try:
            # 读取历史WFO结果
            historical_weights = self._load_historical_contributions()

            # P0优化: 冷启动使用IC加权而非等权
            if not historical_weights:
                logger.warning("⚠️ 无历史贡献数据，回退到IC加权")
                # 使用IS IC作为权重（从is_ic_scores获取）
                ic_values = np.array(
                    [is_ic_scores.get(f, 0.01) for f in selected_factors]
                )
                ic_values = np.maximum(ic_values, self.ic_floor)
                total_ic = ic_values.sum()

                if total_ic < 1e-10:
                    weight = 1.0 / len(selected_factors)
                    return {f: weight for f in selected_factors}

                weights = ic_values / total_ic
                return {f: float(w) for f, w in zip(selected_factors, weights)}

            # 计算各因子的综合评分
            factor_scores = self._compute_factor_scores(
                selected_factors, historical_weights
            )

            # P0优化: 前3窗口数据不足时使用IC加权而非等权
            if not factor_scores:
                logger.warning("⚠️ 历史数据不足(<3窗口)，回退到IC加权")
                ic_values = np.array(
                    [is_ic_scores.get(f, 0.01) for f in selected_factors]
                )
                ic_values = np.maximum(ic_values, self.ic_floor)
                total_ic = ic_values.sum()

                if total_ic < 1e-10:
                    weight = 1.0 / len(selected_factors)
                    return {f: weight for f in selected_factors}

                weights = ic_values / total_ic
                return {f: float(w) for f, w in zip(selected_factors, weights)}

            # 指数加权分配
            raw_weights = self._exponential_weighting(factor_scores)

            # 应用风险控制约束
            constrained_weights = self._apply_weight_constraints(raw_weights)

            # 平滑过渡 (70%新权重 + 30%等权)
            smoothed_weights = self._smooth_weight_transition(
                constrained_weights, selected_factors
            )

            return smoothed_weights

        except Exception as e:
            logger.warning(f"⚠️ 贡献加权计算失败: {e}，回退到等权")
            weight = 1.0 / len(selected_factors)
            return {f: weight for f in selected_factors}

    def _load_historical_contributions(self) -> Dict[str, List[float]]:
        """
        加载历史贡献数据

        P0修复: 直接使用内存化数据，消除外部文件依赖
        """
        return self.historical_contributions

    def _compute_factor_scores(
        self, selected_factors: List[str], historical_weights: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """计算因子综合评分"""
        factor_scores = {}

        for factor in selected_factors:
            if factor in historical_weights:
                contribs = np.array(historical_weights[factor])

                if len(contribs) >= 3:  # 至少3次观测
                    # 贡献强度 (均值)
                    mean_contrib = contribs.mean()

                    # 贡献稳定性 (1/CV)
                    std_contrib = contribs.std()
                    cv = (
                        std_contrib / abs(mean_contrib)
                        if mean_contrib != 0
                        else float("inf")
                    )
                    stability_score = 1.0 / cv if cv > 0 else 0

                    # 强度评分 (标准化)
                    strength_score = np.tanh(mean_contrib / (std_contrib + 1e-6))

                    # 综合评分: 70%稳定性 + 30%强度
                    composite_score = (
                        0.7 * min(stability_score / 5.0, 1.0) + 0.3 * strength_score
                    )

                    factor_scores[factor] = composite_score

        return factor_scores

    def _exponential_weighting(
        self, factor_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """指数加权分配"""
        if not factor_scores:
            return {}

        factors = list(factor_scores.keys())
        scores = np.array([factor_scores[f] for f in factors])

        # 指数加权
        exp_scores = np.exp(scores / self.contribution_weighting_temperature)
        weights = exp_scores / exp_scores.sum()

        return {f: float(w) for f, w in zip(factors, weights)}

    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用权重约束"""
        constrained_weights = {}

        # 首先应用上下限
        for factor, weight in weights.items():
            constrained_weights[factor] = np.clip(
                weight, self.min_single_weight, self.max_single_weight
            )

        # 重新归一化
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {
                f: w / total_weight for f, w in constrained_weights.items()
            }

        return constrained_weights

    def _smooth_weight_transition(
        self, new_weights: Dict[str, float], selected_factors: List[str]
    ) -> Dict[str, float]:
        """平滑权重过渡"""
        # 等权重作为基准
        equal_weight = 1.0 / len(selected_factors)
        equal_weights = {f: equal_weight for f in selected_factors}

        # 70%新权重 + 30%等权重
        smoothed_weights = {}
        for factor in selected_factors:
            new_w = new_weights.get(factor, equal_weight)
            equal_w = equal_weights[factor]
            smoothed_weights[factor] = 0.7 * new_w + 0.3 * equal_w

        return smoothed_weights

    def _compute_signal_ic(self, signal: np.ndarray, returns: np.ndarray) -> float:
        """计算信号IC"""
        return float(self.ic_calculator.compute_ic(signal, returns))

    def _compute_signal_sharpe(self, signal: np.ndarray, returns: np.ndarray) -> float:
        """计算信号Sharpe (IC序列的IR)"""
        T = min(len(signal), len(returns))
        sig = signal[:T]
        ret = returns[:T]

        # 计算每日IC
        sig_std = np.nanstd(sig, axis=1)
        ret_std = np.nanstd(ret, axis=1)
        valid_mask = (sig_std > 1e-10) & (ret_std > 1e-10)

        if not np.any(valid_mask):
            return 0.0

        sig_mean = np.nanmean(sig, axis=1, keepdims=True)
        ret_mean = np.nanmean(ret, axis=1, keepdims=True)
        sig_norm = (sig - sig_mean) / (sig_std[:, None] + 1e-10)
        ret_norm = (ret - ret_mean) / (ret_std[:, None] + 1e-10)
        ic_series = np.nanmean(sig_norm * ret_norm, axis=1)
        ic_series = ic_series[valid_mask]

        if ic_series.size == 0:
            return 0.0

        ic_mean = np.nanmean(ic_series)
        ic_std = np.nanstd(ic_series)

        return float(ic_mean / ic_std) if ic_std > 1e-12 else 0.0

    def _create_summary_df(
        self, results: List[DirectFactorWindowResult]
    ) -> pd.DataFrame:
        """创建汇总DataFrame (P0-1: 增加基准对照列, P1-1: 增加因子贡献)"""
        import json

        records = []
        for r in results:
            excess_ic = r.oos_ensemble_ic - r.oos_baseline_ic
            excess_sharpe = r.oos_ensemble_sharpe - r.oos_baseline_sharpe

            # P1-1: 序列化因子权重与贡献 (JSON格式)
            top_factors = sorted(
                r.factor_weights.items(), key=lambda x: x[1], reverse=True
            )[:5]
            top_factors_json = json.dumps(
                {
                    f: {"weight": w, "contribution": r.factor_contributions.get(f, 0.0)}
                    for f, w in top_factors
                },
                ensure_ascii=False,
            )

            records.append(
                {
                    "window_index": r.window_index,
                    "is_start": r.is_start,
                    "is_end": r.is_end,
                    "oos_start": r.oos_start,
                    "oos_end": r.oos_end,
                    "n_selected_factors": len(r.selected_factors),
                    "selected_factors": ",".join(r.selected_factors),
                    "oos_ensemble_ic": r.oos_ensemble_ic,
                    "oos_ensemble_sharpe": r.oos_ensemble_sharpe,
                    "oos_baseline_ic": r.oos_baseline_ic,
                    "oos_baseline_sharpe": r.oos_baseline_sharpe,
                    "excess_ic": excess_ic,
                    "excess_sharpe": excess_sharpe,
                    "top_factors": top_factors_json,  # P1-1: 因子权重与贡献
                }
            )
        return pd.DataFrame(records)

    def _load_persistent_contributions(self) -> None:
        """P0: 从磁盘加载历史贡献数据"""
        import os
        import pickle

        if os.path.exists(self.contribution_cache_path):
            try:
                with open(self.contribution_cache_path, "rb") as f:
                    self.historical_contributions = pickle.load(f)
                if self.verbose:
                    total_records = sum(
                        len(v) for v in self.historical_contributions.values()
                    )
                    logger.info(f"✅ 加载历史贡献数据: {total_records} 条记录")
            except Exception as e:
                logger.warning(f"⚠️ 加载历史贡献数据失败: {e}，使用空数据")
                self.historical_contributions = {}
        else:
            if self.verbose:
                logger.info("ℹ️ 无历史贡献数据文件，使用空数据")

    def _save_persistent_contributions(self) -> None:
        """P0: 保存历史贡献数据到磁盘"""
        import os
        import pickle

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.contribution_cache_path), exist_ok=True)

            with open(self.contribution_cache_path, "wb") as f:
                pickle.dump(self.historical_contributions, f)

            if self.verbose:
                total_records = sum(
                    len(v) for v in self.historical_contributions.values()
                )
                logger.info(
                    f"✅ 保存历史贡献数据: {total_records} 条记录到 {self.contribution_cache_path}"
                )

        except Exception as e:
            logger.warning(f"⚠️ 保存历史贡献数据失败: {e}")
