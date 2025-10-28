#!/usr/bin/env python3
"""
因子验证标准框架（Base Class）

用途：
- 提供统一的因子评估接口
- 严格执行横截面 Spearman + T-1 对齐
- 自动输出准入门槛报告

使用方法：
1. 继承 FactorValidator 类
2. 实现 compute_factor() 方法
3. 调用 evaluate() 获取报告
"""

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class FactorValidator(ABC):
    """因子验证基础类"""

    # 准入门槛（可在子类中覆盖）
    MIN_OOS_IC = 0.010  # OOS 平均 RankIC 下限
    MAX_DECAY_RATIO = 0.50  # IS→OOS 衰减比例上限
    MAX_FAILURE_RATIO = 0.30  # 失败窗口比例上限
    MAX_TOP3_CORR = 0.70  # 与 Top3 因子相关性上限

    # WFO 配置（沿用系统标准）
    IS_WINDOW = 252
    OOS_WINDOW = 60
    STEP = 20

    def __init__(self, ohlcv_dir: str, existing_factors_dir: str):
        """
        Args:
            ohlcv_dir: OHLCV 数据目录（parquet 格式）
            existing_factors_dir: 现有标准化因子目录
        """
        self.ohlcv_dir = Path(ohlcv_dir)
        self.existing_factors_dir = Path(existing_factors_dir)

        # 加载数据
        self._load_data()

        logger.info(f"✅ 因子验证器初始化完成")
        logger.info(f"  - IS窗口: {self.IS_WINDOW}d")
        logger.info(f"  - OOS窗口: {self.OOS_WINDOW}d")
        logger.info(f"  - 步长: {self.STEP}d")

    def _load_data(self):
        """加载 OHLCV、收益率与 Top3 因子"""
        logger.info("📊 加载数据...")

        # 加载 OHLCV
        self.close = pd.read_parquet(self.ohlcv_dir / "close.parquet")
        self.high = pd.read_parquet(self.ohlcv_dir / "high.parquet")
        self.low = pd.read_parquet(self.ohlcv_dir / "low.parquet")
        self.volume = pd.read_parquet(self.ohlcv_dir / "volume.parquet")
        self.open = pd.read_parquet(self.ohlcv_dir / "open.parquet")

        # 计算收益率（T-1 对齐）
        self.returns = self.close.pct_change(fill_method=None)

        # 加载 Top3 稳定因子（用于冗余检查）
        self.top3_factors = {
            "CALMAR_RATIO_60D": pd.read_parquet(
                self.existing_factors_dir / "CALMAR_RATIO_60D.parquet"
            ),
            "PRICE_POSITION_120D": pd.read_parquet(
                self.existing_factors_dir / "PRICE_POSITION_120D.parquet"
            ),
            "CMF_20D": pd.read_parquet(self.existing_factors_dir / "CMF_20D.parquet"),
        }

        logger.info(f"  - OHLCV: {self.close.shape}")
        logger.info(f"  - 收益率: {self.returns.shape}")
        logger.info(f"  - Top3 因子已加载")

    @abstractmethod
    def compute_factor(self) -> pd.DataFrame:
        """
        计算因子值（子类必须实现）

        Returns:
            pd.DataFrame:
                - Index: 时间序列（与 OHLCV 对齐）
                - Columns: 资产代码
                - Values: 横截面标准化后的因子值

        注意：
            1. 必须进行横截面标准化（每日去均值/标准差）
            2. 数据对齐：确保 index 与 self.close.index 一致
            3. NaN 处理：允许前期有 NaN（窗口预热期）
        """
        pass

    def _cross_sectional_standardize(self, factor: pd.DataFrame) -> pd.DataFrame:
        """横截面标准化（每日去均值/标准差）"""
        factor_std = factor.sub(factor.mean(axis=1), axis=0).div(
            factor.std(axis=1) + 1e-8, axis=0
        )
        return factor_std

    def _compute_cross_sectional_ic(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        window_start: int,
        window_end: int,
    ) -> float:
        """
        计算窗口内横截面 Spearman IC（严格 T-1 对齐）

        Args:
            factor: 因子值（横截面）
            returns: 收益率（横截面）
            window_start: 窗口起始索引
            window_end: 窗口结束索引（不含）

        Returns:
            平均 IC 值
        """
        # T-1 对齐：因子 [start-1, end-1)，收益 [start, end)
        factor_start = max(0, window_start - 1)
        factor_end = max(0, window_end - 1)

        factor_slice = factor.iloc[factor_start:factor_end]
        return_slice = returns.iloc[window_start:window_end]

        # 长度保护
        n_days = min(len(factor_slice), len(return_slice))

        daily_ics = []
        for t in range(n_days):
            factor_t = factor_slice.iloc[t].values
            return_t = return_slice.iloc[t].values

            # 有效掩码
            valid_mask = ~(np.isnan(factor_t) | np.isnan(return_t))
            if valid_mask.sum() < 5:
                continue

            ic, _ = spearmanr(factor_t[valid_mask], return_t[valid_mask])
            if not np.isnan(ic):
                daily_ics.append(float(ic))

        return np.mean(daily_ics) if daily_ics else 0.0

    def _run_wfo_evaluation(self, factor: pd.DataFrame, factor_name: str) -> Dict:
        """
        WFO 评估（沿用系统配置）

        Returns:
            {
                'factor_name': str,
                'n_windows': int,
                'is_ic_mean': float,
                'oos_ic_mean': float,
                'ic_decay': float,
                'decay_ratio': float,
                'failure_ratio': float,
                'windows': List[Dict]
            }
        """
        logger.info(f"\n📈 WFO 评估: {factor_name}")

        n_days = len(self.returns)
        windows = []

        # 生成 WFO 窗口
        for start in range(0, n_days - self.IS_WINDOW - self.OOS_WINDOW + 1, self.STEP):
            is_start = start
            is_end = start + self.IS_WINDOW
            oos_start = is_end
            oos_end = min(oos_start + self.OOS_WINDOW, n_days)

            # IS IC
            is_ic = self._compute_cross_sectional_ic(
                factor, self.returns, is_start, is_end
            )

            # OOS IC
            oos_ic = self._compute_cross_sectional_ic(
                factor, self.returns, oos_start, oos_end
            )

            windows.append(
                {
                    "is_start": is_start,
                    "is_end": is_end,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                    "is_ic": is_ic,
                    "oos_ic": oos_ic,
                    "ic_decay": is_ic - oos_ic,
                }
            )

        # 汇总统计
        is_ics = [w["is_ic"] for w in windows]
        oos_ics = [w["oos_ic"] for w in windows]
        decays = [w["ic_decay"] for w in windows]

        is_ic_mean = np.mean(is_ics)
        oos_ic_mean = np.mean(oos_ics)
        ic_decay = np.mean(decays)

        # 衰减比例（避免除零）
        decay_ratio = (
            abs(ic_decay / (is_ic_mean + 1e-8)) if abs(is_ic_mean) > 1e-8 else 999.0
        )

        # 失败窗口比例（OOS IC < 0）
        failure_count = sum(1 for ic in oos_ics if ic < 0)
        failure_ratio = failure_count / len(windows) if windows else 1.0

        logger.info(f"  - 窗口数: {len(windows)}")
        logger.info(f"  - IS IC 均值: {is_ic_mean:.4f}")
        logger.info(f"  - OOS IC 均值: {oos_ic_mean:.4f}")
        logger.info(f"  - IC 衰减: {ic_decay:.4f}")
        logger.info(f"  - 衰减比例: {decay_ratio:.2%}")
        logger.info(f"  - 失败窗口: {failure_ratio:.2%}")

        return {
            "factor_name": factor_name,
            "n_windows": len(windows),
            "is_ic_mean": is_ic_mean,
            "oos_ic_mean": oos_ic_mean,
            "ic_decay": ic_decay,
            "decay_ratio": decay_ratio,
            "failure_ratio": failure_ratio,
            "windows": windows,
        }

    def _check_correlation_with_top3(self, factor: pd.DataFrame) -> Dict[str, float]:
        """检查与 Top3 稳定因子的相关性（横截面时间序列相关）"""
        logger.info("🔍 检查与 Top3 因子的相关性...")

        correlations = {}

        for name, top3_factor in self.top3_factors.items():
            # 对齐索引
            common_idx = factor.index.intersection(top3_factor.index)

            if len(common_idx) < 100:
                correlations[name] = np.nan
                continue

            factor_aligned = factor.loc[common_idx]
            top3_aligned = top3_factor.loc[common_idx]

            # 全面展开成向量（时间 × 资产）
            factor_vec = factor_aligned.values.flatten()
            top3_vec = top3_aligned.values.flatten()

            # 有效掩码
            valid_mask = ~(np.isnan(factor_vec) | np.isnan(top3_vec))

            if valid_mask.sum() < 100:
                correlations[name] = np.nan
                continue

            corr, _ = spearmanr(factor_vec[valid_mask], top3_vec[valid_mask])
            correlations[name] = corr

            logger.info(f"  - {name}: {corr:.4f}")

        return correlations

    def evaluate(self, factor_name: str) -> Dict:
        """
        完整评估流程（主方法）

        Args:
            factor_name: 因子名称

        Returns:
            评估报告字典（包含是否通过准入门槛）
        """
        logger.info("=" * 80)
        logger.info(f"🚀 开始评估因子: {factor_name}")
        logger.info("=" * 80)

        # 计算因子
        factor_data = self.compute_factor()

        # WFO 评估
        wfo_result = self._run_wfo_evaluation(factor_data, factor_name)

        # Top3 相关性
        correlations = self._check_correlation_with_top3(factor_data)
        median_corr = np.nanmedian(list(correlations.values()))

        # 准入判定
        pass_oos_ic = wfo_result["oos_ic_mean"] >= self.MIN_OOS_IC
        pass_decay = wfo_result["decay_ratio"] <= self.MAX_DECAY_RATIO
        pass_failure = wfo_result["failure_ratio"] <= self.MAX_FAILURE_RATIO
        pass_corr = median_corr < self.MAX_TOP3_CORR

        all_pass = pass_oos_ic and pass_decay and pass_failure and pass_corr

        result = {
            "factor_name": factor_name,
            "oos_ic_mean": wfo_result["oos_ic_mean"],
            "is_ic_mean": wfo_result["is_ic_mean"],
            "ic_decay_ratio": wfo_result["decay_ratio"],
            "failure_ratio": wfo_result["failure_ratio"],
            "top3_median_corr": median_corr,
            "pass_oos_ic": pass_oos_ic,
            "pass_decay": pass_decay,
            "pass_failure": pass_failure,
            "pass_corr": pass_corr,
            "PASS_ALL": all_pass,
            "top3_correlations": correlations,
            "wfo_windows": wfo_result["windows"],
        }

        # 输出报告
        self._print_report(result)

        return result

    def _print_report(self, result: Dict):
        """打印评估报告"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 评估结果")
        logger.info("=" * 80)
        logger.info(f"因子名称: {result['factor_name']}")
        logger.info(f"  - IS IC 均值: {result['is_ic_mean']:.4f}")
        logger.info(f"  - OOS IC 均值: {result['oos_ic_mean']:.4f}")
        logger.info(f"  - IC 衰减比: {result['ic_decay_ratio']:.2%}")
        logger.info(f"  - 失败窗口率: {result['failure_ratio']:.2%}")
        logger.info(f"  - Top3 中位相关: {result['top3_median_corr']:.4f}")

        logger.info("\n" + "=" * 80)
        logger.info("🎯 准入门槛检查")
        logger.info("=" * 80)
        logger.info(
            f"  {'✅' if result['pass_oos_ic'] else '❌'} OOS IC ≥ {self.MIN_OOS_IC}: {result['oos_ic_mean']:.4f}"
        )
        logger.info(
            f"  {'✅' if result['pass_decay'] else '❌'} 衰减比 ≤ {self.MAX_DECAY_RATIO:.0%}: {result['ic_decay_ratio']:.2%}"
        )
        logger.info(
            f"  {'✅' if result['pass_failure'] else '❌'} 失败率 ≤ {self.MAX_FAILURE_RATIO:.0%}: {result['failure_ratio']:.2%}"
        )
        logger.info(
            f"  {'✅' if result['pass_corr'] else '❌'} Top3相关 < {self.MAX_TOP3_CORR}: {result['top3_median_corr']:.4f}"
        )

        logger.info("\n" + "=" * 80)
        if result["PASS_ALL"]:
            logger.info("✅ 通过准入门槛！")
        else:
            logger.info("❌ 未通过准入门槛")
            reasons = []
            if not result["pass_oos_ic"]:
                reasons.append(
                    f"OOS IC不足({result['oos_ic_mean']:.4f}<{self.MIN_OOS_IC})"
                )
            if not result["pass_decay"]:
                reasons.append(
                    f"衰减过大({result['ic_decay_ratio']:.2%}>{self.MAX_DECAY_RATIO:.0%})"
                )
            if not result["pass_failure"]:
                reasons.append(
                    f"失败率高({result['failure_ratio']:.2%}>{self.MAX_FAILURE_RATIO:.0%})"
                )
            if not result["pass_corr"]:
                reasons.append(
                    f"相关度高({result['top3_median_corr']:.4f}>{self.MAX_TOP3_CORR})"
                )
            logger.info(f"原因: {', '.join(reasons)}")
        logger.info("=" * 80)


class BatchFactorValidator:
    """批量因子验证器（支持多个因子一次性评估）"""

    def __init__(self, ohlcv_dir: str, existing_factors_dir: str):
        self.ohlcv_dir = ohlcv_dir
        self.existing_factors_dir = existing_factors_dir

    def evaluate_batch(
        self, validators: List[FactorValidator], factor_names: List[str]
    ) -> pd.DataFrame:
        """
        批量评估多个因子

        Args:
            validators: FactorValidator 实例列表
            factor_names: 对应的因子名称列表

        Returns:
            pd.DataFrame: 汇总结果表
        """
        results = []

        for validator, factor_name in zip(validators, factor_names):
            result = validator.evaluate(factor_name)
            results.append(
                {
                    "factor_name": result["factor_name"],
                    "oos_ic_mean": result["oos_ic_mean"],
                    "ic_decay_ratio": result["ic_decay_ratio"],
                    "failure_ratio": result["failure_ratio"],
                    "top3_median_corr": result["top3_median_corr"],
                    "pass_oos_ic": result["pass_oos_ic"],
                    "pass_decay": result["pass_decay"],
                    "pass_failure": result["pass_failure"],
                    "pass_corr": result["pass_corr"],
                    "PASS_ALL": result["PASS_ALL"],
                }
            )

        df = pd.DataFrame(results)

        # 输出汇总报告
        logger.info("\n\n" + "=" * 80)
        logger.info("📊 批量评估结果汇总")
        logger.info("=" * 80)
        logger.info(f"\n{df.to_string(index=False)}")

        passed = df[df["PASS_ALL"] == True]
        rejected = df[df["PASS_ALL"] == False]

        logger.info("\n" + "=" * 80)
        logger.info(f"✅ 通过准入: {len(passed)} 个")
        logger.info(f"❌ 未通过: {len(rejected)} 个")
        logger.info("=" * 80)

        return df
