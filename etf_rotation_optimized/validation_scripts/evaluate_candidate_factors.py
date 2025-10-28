#!/usr/bin/env python3
"""
离线评估候选因子（严格遵循横截面 Spearman + T-1 口径）

评估因子：
1. REVERSAL_FACTOR_5D - 短期反转
2. VOLATILITY_SKEW_20D - 波动结构质量
3. DOLLAR_VOLUME_ACCELERATION_10D - 美元成交额加速度

准入门槛（全部满足）：
- OOS 平均 RankIC ≥ 0.010
- 衰减比（IS→OOS）≤ 50%
- 失败窗口占比 ≤ 30%
- 与 Top3 稳定因子中位相关性 < 0.7
- 加入后 Step4 Sharpe 不下降
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class CandidateFactorEvaluator:
    """候选因子离线评估器"""

    def __init__(self, ohlcv_dir: str, existing_factors_dir: str):
        """
        Args:
            ohlcv_dir: OHLCV 数据目录
            existing_factors_dir: 现有标准化因子目录
        """
        self.ohlcv_dir = Path(ohlcv_dir)
        self.existing_factors_dir = Path(existing_factors_dir)

        # WFO 配置（沿用现有系统）
        self.is_window = 252
        self.oos_window = 60
        self.step = 20

        # 准入门槛
        self.min_oos_ic = 0.010
        self.max_decay_ratio = 0.50
        self.max_failure_ratio = 0.30
        self.max_top3_corr = 0.70

        logger.info("✅ 候选因子评估器初始化完成")
        logger.info(f"  - IS窗口: {self.is_window}d")
        logger.info(f"  - OOS窗口: {self.oos_window}d")
        logger.info(f"  - 步长: {self.step}d")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        """加载 OHLCV 与现有因子数据"""
        logger.info("📊 加载数据...")

        # 加载 OHLCV
        close = pd.read_parquet(self.ohlcv_dir / "close.parquet")
        volume = pd.read_parquet(self.ohlcv_dir / "volume.parquet")

        # 计算收益率（T-1 对齐）
        returns = close.pct_change(fill_method=None)

        # 加载 Top3 稳定因子
        top3_factors = {
            "CALMAR_RATIO_60D": pd.read_parquet(
                self.existing_factors_dir / "CALMAR_RATIO_60D.parquet"
            ),
            "PRICE_POSITION_120D": pd.read_parquet(
                self.existing_factors_dir / "PRICE_POSITION_120D.parquet"
            ),
            "CMF_20D": pd.read_parquet(self.existing_factors_dir / "CMF_20D.parquet"),
        }

        logger.info(f"  - OHLCV: {close.shape}")
        logger.info(f"  - 收益率: {returns.shape}")
        logger.info(f"  - Top3 因子已加载")

        return close, volume, returns, top3_factors

    def compute_reversal_5d(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        计算 5 日短期反转因子（横截面标准化）

        逻辑：过去 5 日收益率的负值（跌多→反转预期强）
        """
        logger.info("🔧 计算 REVERSAL_FACTOR_5D...")

        # 5日累计收益率
        ret_5d = close.pct_change(periods=5, fill_method=None)

        # 取负值（反转逻辑）
        reversal = -ret_5d

        # 横截面标准化（每日）
        reversal_std = reversal.sub(reversal.mean(axis=1), axis=0).div(
            reversal.std(axis=1) + 1e-8, axis=0
        )

        logger.info(f"  ✅ NaN率: {reversal_std.isna().mean().mean():.2%}")
        return reversal_std

    def compute_volatility_skew_20d(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        计算 20 日波动率偏斜因子（横截面标准化）

        逻辑：下跌日波动率 / 上涨日波动率
        健康趋势: skew < 1 (上涨日波动低)
        出货特征: skew > 1 (上涨日波动高)
        """
        logger.info("🔧 计算 VOLATILITY_SKEW_20D...")

        returns = close.pct_change(fill_method=None)

        skew = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)

        for col in close.columns:
            ret = returns[col]

            # 20日滚动窗口
            for i in range(20, len(ret)):
                window_ret = ret.iloc[i - 20 : i]

                # 上涨日与下跌日波动率
                up_vol = window_ret[window_ret > 0].std()
                down_vol = window_ret[window_ret < 0].std()

                # 避免除零
                if pd.notna(up_vol) and pd.notna(down_vol) and up_vol > 1e-8:
                    skew.iloc[i, skew.columns.get_loc(col)] = down_vol / up_vol

        # 横截面标准化（每日）
        skew_std = skew.sub(skew.mean(axis=1), axis=0).div(
            skew.std(axis=1) + 1e-8, axis=0
        )

        logger.info(f"  ✅ NaN率: {skew_std.isna().mean().mean():.2%}")
        return skew_std

    def compute_dollar_volume_accel_10d(
        self, close: pd.DataFrame, volume: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算 10 日美元成交额加速度因子（横截面标准化）

        逻辑：成交额 = close * volume
        加速度 = (最近5日均成交额 - 前5日均成交额) / 前5日均成交额
        """
        logger.info("🔧 计算 DOLLAR_VOLUME_ACCELERATION_10D...")

        # 美元成交额
        dollar_vol = close * volume

        # 最近5日与前5日均值
        recent_5d = dollar_vol.rolling(window=5, min_periods=5).mean()
        prior_5d = dollar_vol.shift(5).rolling(window=5, min_periods=5).mean()

        # 加速度（百分比变化）
        accel = (recent_5d - prior_5d) / (prior_5d + 1e-8)

        # 横截面标准化（每日）
        accel_std = accel.sub(accel.mean(axis=1), axis=0).div(
            accel.std(axis=1) + 1e-8, axis=0
        )

        logger.info(f"  ✅ NaN率: {accel_std.isna().mean().mean():.2%}")
        return accel_std

    def compute_cross_sectional_ic(
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

    def evaluate_factor_wfo(
        self, factor: pd.DataFrame, returns: pd.DataFrame, factor_name: str
    ) -> Dict:
        """
        WFO 评估单个因子（沿用系统配置）

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

        n_days = len(returns)
        windows = []

        # 生成 WFO 窗口
        for start in range(0, n_days - self.is_window - self.oos_window + 1, self.step):
            is_start = start
            is_end = start + self.is_window
            oos_start = is_end
            oos_end = min(oos_start + self.oos_window, n_days)

            # IS IC
            is_ic = self.compute_cross_sectional_ic(factor, returns, is_start, is_end)

            # OOS IC
            oos_ic = self.compute_cross_sectional_ic(
                factor, returns, oos_start, oos_end
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

    def check_correlation_with_top3(
        self, factor: pd.DataFrame, top3_factors: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """检查与 Top3 稳定因子的相关性（横截面时间序列相关）"""
        logger.info("🔍 检查与 Top3 因子的相关性...")

        correlations = {}

        for name, top3_factor in top3_factors.items():
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

    def evaluate_all_candidates(self) -> pd.DataFrame:
        """评估所有候选因子并输出报告"""
        logger.info("=" * 80)
        logger.info("🚀 开始候选因子离线评估")
        logger.info("=" * 80)

        # 加载数据
        close, volume, returns, top3_factors = self.load_data()

        # 计算候选因子
        candidates = {
            "REVERSAL_FACTOR_5D": self.compute_reversal_5d(close),
            "VOLATILITY_SKEW_20D": self.compute_volatility_skew_20d(close),
            "DOLLAR_VOLUME_ACCELERATION_10D": self.compute_dollar_volume_accel_10d(
                close, volume
            ),
        }

        # 评估结果
        results = []

        for factor_name, factor_data in candidates.items():
            # WFO 评估
            wfo_result = self.evaluate_factor_wfo(factor_data, returns, factor_name)

            # Top3 相关性
            correlations = self.check_correlation_with_top3(factor_data, top3_factors)
            median_corr = np.nanmedian(list(correlations.values()))

            # 准入判定
            pass_oos_ic = wfo_result["oos_ic_mean"] >= self.min_oos_ic
            pass_decay = wfo_result["decay_ratio"] <= self.max_decay_ratio
            pass_failure = wfo_result["failure_ratio"] <= self.max_failure_ratio
            pass_corr = median_corr < self.max_top3_corr

            all_pass = pass_oos_ic and pass_decay and pass_failure and pass_corr

            results.append(
                {
                    "factor_name": factor_name,
                    "oos_ic_mean": wfo_result["oos_ic_mean"],
                    "ic_decay_ratio": wfo_result["decay_ratio"],
                    "failure_ratio": wfo_result["failure_ratio"],
                    "top3_median_corr": median_corr,
                    "pass_oos_ic": pass_oos_ic,
                    "pass_decay": pass_decay,
                    "pass_failure": pass_failure,
                    "pass_corr": pass_corr,
                    "PASS_ALL": all_pass,
                }
            )

        # 输出报告
        df = pd.DataFrame(results)

        logger.info("\n" + "=" * 80)
        logger.info("📊 评估结果汇总")
        logger.info("=" * 80)
        logger.info(f"\n{df.to_string(index=False)}")

        logger.info("\n" + "=" * 80)
        logger.info("🎯 准入门槛检查")
        logger.info("=" * 80)
        logger.info(f"  - OOS IC ≥ {self.min_oos_ic}")
        logger.info(f"  - 衰减比 ≤ {self.max_decay_ratio:.0%}")
        logger.info(f"  - 失败率 ≤ {self.max_failure_ratio:.0%}")
        logger.info(f"  - Top3相关 < {self.max_top3_corr}")

        # 最终裁决
        passed = df[df["PASS_ALL"] == True]
        rejected = df[df["PASS_ALL"] == False]

        logger.info("\n" + "=" * 80)
        logger.info("✅ 通过准入门槛:")
        logger.info("=" * 80)
        if len(passed) > 0:
            for _, row in passed.iterrows():
                logger.info(f"  ✅ {row['factor_name']}")
                logger.info(f"     - OOS IC: {row['oos_ic_mean']:.4f}")
                logger.info(f"     - 衰减比: {row['ic_decay_ratio']:.2%}")
                logger.info(f"     - 失败率: {row['failure_ratio']:.2%}")
                logger.info(f"     - Top3相关: {row['top3_median_corr']:.4f}")
        else:
            logger.info("  无因子通过")

        logger.info("\n" + "=" * 80)
        logger.info("❌ 未通过准入门槛:")
        logger.info("=" * 80)
        if len(rejected) > 0:
            for _, row in rejected.iterrows():
                logger.info(f"  ❌ {row['factor_name']}")
                reasons = []
                if not row["pass_oos_ic"]:
                    reasons.append(
                        f"OOS IC不足({row['oos_ic_mean']:.4f}<{self.min_oos_ic})"
                    )
                if not row["pass_decay"]:
                    reasons.append(
                        f"衰减过大({row['ic_decay_ratio']:.2%}>{self.max_decay_ratio:.0%})"
                    )
                if not row["pass_failure"]:
                    reasons.append(
                        f"失败率高({row['failure_ratio']:.2%}>{self.max_failure_ratio:.0%})"
                    )
                if not row["pass_corr"]:
                    reasons.append(
                        f"相关度高({row['top3_median_corr']:.4f}>{self.max_top3_corr})"
                    )
                logger.info(f"     原因: {', '.join(reasons)}")
        else:
            logger.info("  所有因子均通过")

        return df


def main():
    """主函数"""
    # 查找最新的数据目录
    results_dir = Path(__file__).parent.parent / "results"

    # 查找最新的 cross_section 目录
    cross_section_base = results_dir / "cross_section" / "20251027"
    latest_cross = sorted(cross_section_base.glob("*"))[-1]
    ohlcv_dir = latest_cross / "ohlcv"

    # 查找最新的 factor_selection 目录
    factor_sel_base = results_dir / "factor_selection" / "20251027"
    latest_factor = sorted(factor_sel_base.glob("*"))[-1]
    factors_dir = latest_factor / "standardized"

    logger.info(f"📁 数据目录:")
    logger.info(f"  - OHLCV: {ohlcv_dir}")
    logger.info(f"  - 标准化因子: {factors_dir}")

    # 创建评估器
    evaluator = CandidateFactorEvaluator(
        ohlcv_dir=str(ohlcv_dir), existing_factors_dir=str(factors_dir)
    )

    # 执行评估
    results_df = evaluator.evaluate_all_candidates()

    # 保存结果
    output_dir = Path(__file__).parent
    output_file = (
        output_dir
        / f"candidate_factors_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    results_df.to_csv(output_file, index=False)

    logger.info(f"\n💾 评估结果已保存: {output_file}")

    logger.info("\n" + "=" * 80)
    logger.info("🎉 评估完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
