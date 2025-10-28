"""
Ensemble WFO Optimizer | 集成Walk-Forward优化器

基于ConstrainedWalkForwardOptimizer扩展,实现:
1. 智能因子组合采样 (1000个5因子组合)
2. 批量向量化评估
3. Top10集成预测
4. 抗过拟合机制

作者: AI Agent
日期: 2025-01-XX
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.constrained_walk_forward_optimizer import (
    ConstrainedWalkForwardOptimizer,
    ConstraintApplicationReport,
)
from core.ensemble_sampler import EnsembleSampler
from core.factor_weighting import FactorWeighting
from core.ic_calculator import ICCalculator

logger = logging.getLogger(__name__)


@dataclass
class EnsembleWindowResult:
    """单窗口集成结果"""

    window_index: int
    """窗口索引"""

    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    """时间范围"""

    n_sampled_combos: int
    """采样的组合数量"""

    is_ic_scores: Dict[str, float]
    """IS阶段各因子IC"""

    top10_combos: List[Tuple[str, ...]]
    """Top10最优组合"""

    top10_is_ics: List[float]
    """Top10 IS阶段IC"""

    oos_ensemble_ic: float
    """OOS阶段集成IC"""

    oos_ensemble_sharpe: float
    """OOS阶段集成Sharpe"""

    oos_single_ics: Dict[Tuple[str, ...], float]
    """OOS阶段各组合IC (用于验证)"""


class EnsembleWFOOptimizer(ConstrainedWalkForwardOptimizer):
    """
    集成Walk-Forward优化器

    继承ConstrainedWalkForwardOptimizer,增强为:
    - 每个窗口采样1000个5因子组合
    - 批量评估所有组合的IS性能
    - 选择Top10组合进行OOS集成
    - 使用梯度衰减权重抗过拟合
    """

    def __init__(
        self,
        constraints_config: Dict,
        n_samples: int = 1000,
        combo_size: int = 5,
        top_k: int = 10,
        weighting_scheme: str = "gradient_decay",
        random_seed: int = 42,
        verbose: bool = True,
    ):
        """
        初始化集成WFO优化器

        Args:
            constraints_config: 约束配置 (family_quotas + mutual_exclusions)
            n_samples: 每窗口采样组合数 (默认1000)
            combo_size: 每组合因子数 (默认5)
            top_k: 集成的最优组合数 (默认10, 抗过拟合)
            weighting_scheme: 加权方案 ('equal'/'ic_weighted'/'gradient_decay')
            random_seed: 随机种子 (确保可复现)
            verbose: 是否打印日志
        """
        # 调用父类初始化
        super().__init__(selector=None, verbose=verbose)

        # 集成组件
        self.sampler = EnsembleSampler(constraints_config, random_seed=random_seed)
        self.weighter = FactorWeighting()
        self.ic_calculator = ICCalculator()

        # 超参数
        self.n_samples = n_samples
        self.combo_size = combo_size
        self.top_k = top_k
        self.weighting_scheme = weighting_scheme
        self.random_seed = random_seed

        # 结果存储
        self.window_results: List[EnsembleWindowResult] = []

        if verbose:
            logger.info("=" * 80)
            logger.info("🚀 Ensemble WFO Optimizer 初始化")
            logger.info("=" * 80)
            logger.info(f"采样策略: {n_samples}个组合, 每组合{combo_size}因子")
            logger.info(f"集成策略: Top{top_k}, 权重方案={weighting_scheme}")
            logger.info(f"随机种子: {random_seed}")
            logger.info("=" * 80)

    def run_ensemble_wfo(
        self,
        factors_data: np.ndarray,
        returns: np.ndarray,
        factor_names: List[str],
        is_period: int = 100,
        oos_period: int = 20,
        step_size: int = 20,
    ) -> pd.DataFrame:
        """
        运行集成Walk-Forward优化

        Args:
            factors_data: 因子数据 (time_steps, num_assets, num_factors)
            returns: 收益率 (time_steps, num_assets)
            factor_names: 因子名称列表
            is_period: IS窗口长度
            oos_period: OOS窗口长度
            step_size: 滑动步长

        Returns:
            汇总DataFrame: 每窗口OOS性能指标
        """
        num_time_steps, num_assets, num_factors = factors_data.shape

        if self.verbose:
            logger.info("\n" + "=" * 80)
            logger.info("开始 Ensemble WFO 优化")
            logger.info("=" * 80)
            logger.info(
                f"数据形状: {num_time_steps} 日期 × {num_assets} 资产 × {num_factors} 因子"
            )
            logger.info(
                f"窗口设置: IS={is_period}, OOS={oos_period}, step={step_size}"
            )

        # 划分窗口
        windows = self._partition_windows(
            num_time_steps, is_period, oos_period, step_size
        )

        if self.verbose:
            logger.info(f"总窗口数: {len(windows)}")
            logger.info("=" * 80)

        # 逐窗口优化
        for window_idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
            if self.verbose:
                logger.info(
                    f"\n{'─'*80}\n【窗口 {window_idx+1}/{len(windows)}】"
                    f"IS: [{is_start}, {is_end}), OOS: [{oos_start}, {oos_end})\n{'─'*80}"
                )

            # 进度心跳
            if window_idx % 10 == 0 and window_idx > 0:
                logger.info(
                    f"🔄 进度: {window_idx}/{len(windows)} 窗口完成 "
                    f"({window_idx/len(windows)*100:.1f}%)"
                )

            # 运行单窗口优化
            window_result = self._run_single_window(
                factors_data,
                returns,
                factor_names,
                is_start,
                is_end,
                oos_start,
                oos_end,
                window_idx,
            )

            self.window_results.append(window_result)

            if self.verbose:
                logger.info(
                    f"✓ 窗口{window_idx+1} 完成: "
                    f"OOS IC={window_result.oos_ensemble_ic:.4f}, "
                    f"Sharpe={window_result.oos_ensemble_sharpe:.2f}"
                )

        # 生成汇总报告
        summary_df = self._generate_summary_report()

        if self.verbose:
            logger.info("\n" + "=" * 80)
            logger.info("Ensemble WFO 优化完成")
            logger.info("=" * 80)
            logger.info(f"总窗口数: {len(self.window_results)}")
            logger.info(f"平均OOS IC: {summary_df['oos_ensemble_ic'].mean():.4f}")
            logger.info(
                f"平均OOS Sharpe: {summary_df['oos_ensemble_sharpe'].mean():.2f}"
            )
            logger.info("=" * 80)

        return summary_df

    def _run_single_window(
        self,
        factors_data: np.ndarray,
        returns: np.ndarray,
        factor_names: List[str],
        is_start: int,
        is_end: int,
        oos_start: int,
        oos_end: int,
        window_idx: int,
    ) -> EnsembleWindowResult:
        """
        运行单窗口优化 - 核心6步流程

        Steps:
        1. IS数据切片 (T-1对齐)
        2. 计算IS IC评分
        3. 智能采样1000个组合
        4. 批量评估所有组合IS性能
        5. 选择Top10组合
        6. OOS集成预测 + 性能评估

        Returns:
            单窗口集成结果
        """
        # Step 1: IS数据切片 (T-1对齐: 因子[t-1]预测收益[t])
        is_factor_start = max(0, is_start - 1)
        is_factor_end = max(0, is_end - 1)
        is_factors = factors_data[is_factor_start:is_factor_end]
        is_returns = returns[is_start:is_end]

        if self.verbose and window_idx == 0:
            logger.debug(
                f"IS切片: 因子[{is_factor_start}:{is_factor_end}), "
                f"收益[{is_start}:{is_end})"
            )

        # Step 2: 计算IS阶段各因子IC
        is_ic_scores = self._compute_window_ic(is_factors, is_returns, factor_names)

        if self.verbose:
            logger.info(
                f"Step 2: IS IC计算完成, "
                f"平均IC={np.mean(list(is_ic_scores.values())):.4f}"
            )

        # Step 3: 智能采样1000个组合
        sampled_combos = self.sampler.sample_combinations(
            n_samples=self.n_samples,
            factor_pool=factor_names,
            ic_scores=is_ic_scores,
            combo_size=self.combo_size,
        )

        if self.verbose:
            logger.info(f"Step 3: 采样完成, {len(sampled_combos)} 个组合")

        # Step 4: 批量评估所有组合的IS性能
        combo_is_ics = self._batch_evaluate_combos(
            sampled_combos, is_factors, is_returns, factor_names
        )

        if self.verbose:
            logger.info(
                f"Step 4: 批量评估完成, "
                f"IS IC范围=[{min(combo_is_ics):.4f}, {max(combo_is_ics):.4f}]"
            )

        # Step 5: 选择Top10组合 (抗过拟合)
        top_indices = np.argsort(combo_is_ics)[-self.top_k :][::-1]
        top10_combos = [sampled_combos[i] for i in top_indices]
        top10_is_ics = [combo_is_ics[i] for i in top_indices]

        if self.verbose:
            logger.info(
                f"Step 5: Top{self.top_k}选择完成, "
                f"IS IC范围=[{min(top10_is_ics):.4f}, {max(top10_is_ics):.4f}]"
            )

        # Step 6: OOS集成预测
        oos_factor_start = max(0, oos_start - 1)
        oos_factor_end = max(0, oos_end - 1)
        oos_factors = factors_data[oos_factor_start:oos_factor_end]
        oos_returns = returns[oos_start:oos_end]

        # 6.1 计算每个Top10组合的OOS预测信号
        ensemble_signals = []
        oos_single_ics = {}

        for combo in top10_combos:
            # 获取该组合的因子索引
            combo_indices = [factor_names.index(f) for f in combo]

            # 提取该组合的因子数据: (T, N, K) → (T, N)
            combo_factors = oos_factors[:, :, combo_indices]

            # 等权合并为单一信号 (简化版, 可用加权方案)
            combo_signal = np.mean(combo_factors, axis=2)  # (T, N)

            ensemble_signals.append(combo_signal)

            # 计算该组合的OOS IC (用于验证)
            combo_ic = self._compute_signal_ic(combo_signal, oos_returns)
            oos_single_ics[combo] = combo_ic

        # 6.2 集成预测: 使用梯度衰减权重
        ensemble_signals_array = np.stack(
            ensemble_signals, axis=0
        )  # (top_k, T, N)

        # 计算集成权重 (基于IS IC排序)
        # 使用简化方式: 直接根据IC计算权重,无需FactorWeighting
        if self.weighting_scheme == "equal":
            combo_weights = np.ones(self.top_k) / self.top_k
        elif self.weighting_scheme == "ic_weighted":
            ic_array = np.array(top10_is_ics)
            combo_weights = ic_array / ic_array.sum()
        elif self.weighting_scheme == "gradient_decay":
            # 梯度衰减: w_i = exp(-0.5*i) / Z
            ranks = np.arange(self.top_k)
            weights = np.exp(-0.5 * ranks)
            combo_weights = weights / weights.sum()
        else:
            combo_weights = np.ones(self.top_k) / self.top_k

        # 加权集成: ensemble_signal = Σ(w_i * signal_i)
        final_signal = np.tensordot(
            combo_weights, ensemble_signals_array, axes=([0], [0])
        )  # (T, N)

        # 6.3 计算OOS性能
        oos_ensemble_ic = self._compute_signal_ic(final_signal, oos_returns)

        # 计算Sharpe (信号的时序稳定性)
        ic_series = []
        T_oos = min(len(final_signal), len(oos_returns))
        
        for t in range(T_oos):
            if np.std(final_signal[t]) > 0 and np.std(oos_returns[t]) > 0:
                ic_t = np.corrcoef(final_signal[t], oos_returns[t])[0, 1]
                if not np.isnan(ic_t):
                    ic_series.append(ic_t)

        oos_ensemble_sharpe = (
            np.mean(ic_series) / np.std(ic_series) if len(ic_series) > 0 else 0.0
        )

        if self.verbose:
            logger.info(
                f"Step 6: OOS集成完成, IC={oos_ensemble_ic:.4f}, "
                f"Sharpe={oos_ensemble_sharpe:.2f}"
            )

        # 返回窗口结果
        return EnsembleWindowResult(
            window_index=window_idx,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            n_sampled_combos=len(sampled_combos),
            is_ic_scores=is_ic_scores,
            top10_combos=top10_combos,
            top10_is_ics=top10_is_ics,
            oos_ensemble_ic=oos_ensemble_ic,
            oos_ensemble_sharpe=oos_ensemble_sharpe,
            oos_single_ics=oos_single_ics,
        )

    def _batch_evaluate_combos(
        self,
        combos: List[Tuple[str, ...]],
        is_factors: np.ndarray,
        is_returns: np.ndarray,
        factor_names: List[str],
    ) -> List[float]:
        """
        批量评估所有组合的IS IC性能

        Args:
            combos: 因子组合列表
            is_factors: IS因子数据 (T, N, K)
            is_returns: IS收益数据 (T, N)
            factor_names: 因子名称列表

        Returns:
            每个组合的IS IC列表
        """
        combo_ics = []

        for combo in combos:
            # 获取组合因子索引
            combo_indices = [factor_names.index(f) for f in combo]

            # 提取组合因子: (T, N, K) → (T, N)
            combo_factors = is_factors[:, :, combo_indices]
            combo_signal = np.mean(combo_factors, axis=2)  # 等权合并

            # 计算IC
            ic = self._compute_signal_ic(combo_signal, is_returns)
            combo_ics.append(ic)

        return combo_ics

    def _compute_signal_ic(
        self, signal: np.ndarray, returns: np.ndarray
    ) -> float:
        """
        计算信号与收益的IC (Information Coefficient)

        Args:
            signal: 预测信号 (T, N)
            returns: 实际收益 (T, N)

        Returns:
            平均IC
        """
        ic_series = []

        T = min(len(signal), len(returns))  # 防止长度不一致

        for t in range(T):
            if np.std(signal[t]) > 0 and np.std(returns[t]) > 0:
                ic_t = np.corrcoef(signal[t], returns[t])[0, 1]
                if not np.isnan(ic_t):
                    ic_series.append(ic_t)

        return np.mean(ic_series) if len(ic_series) > 0 else 0.0

    def _generate_summary_report(self) -> pd.DataFrame:
        """
        生成汇总报告DataFrame

        Returns:
            每窗口性能指标汇总
        """
        records = []

        for result in self.window_results:
            records.append(
                {
                    "window_index": result.window_index,
                    "is_start": result.is_start,
                    "is_end": result.is_end,
                    "oos_start": result.oos_start,
                    "oos_end": result.oos_end,
                    "n_sampled_combos": result.n_sampled_combos,
                    "top10_mean_is_ic": np.mean(result.top10_is_ics),
                    "oos_ensemble_ic": result.oos_ensemble_ic,
                    "oos_ensemble_sharpe": result.oos_ensemble_sharpe,
                    "top10_combos": str(result.top10_combos[:3]),  # 前3个组合
                }
            )

        return pd.DataFrame(records)

    def save_results(self, output_dir: Path):
        """
        保存结果到文件

        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存汇总报告
        summary_df = self._generate_summary_report()
        summary_path = output_dir / "ensemble_wfo_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        if self.verbose:
            logger.info(f"✓ 汇总报告已保存: {summary_path}")

        # 2. 保存详细窗口结果 (JSON格式)
        import json

        detailed_results = []
        for result in self.window_results:
            detailed_results.append(
                {
                    "window_index": result.window_index,
                    "time_range": {
                        "is_start": result.is_start,
                        "is_end": result.is_end,
                        "oos_start": result.oos_start,
                        "oos_end": result.oos_end,
                    },
                    "sampling": {"n_combos": result.n_sampled_combos},
                    "top10_combos": [list(c) for c in result.top10_combos],
                    "top10_is_ics": result.top10_is_ics,
                    "oos_metrics": {
                        "ensemble_ic": result.oos_ensemble_ic,
                        "ensemble_sharpe": result.oos_ensemble_sharpe,
                    },
                }
            )

        detailed_path = output_dir / "ensemble_wfo_detailed.json"
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        if self.verbose:
            logger.info(f"✓ 详细结果已保存: {detailed_path}")
