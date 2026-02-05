"""
因子稳定性测试器 v1.0
================================================================================
评估因子在不同时间窗口的稳定性，淘汰不稳定因子

核心思想：
- 稳定因子：IC在不同时期保持一致性（低方差）
- 不稳定因子：IC波动大，过拟合训练期噪音

测试方法：
1. WFO滚动窗口计算每期IC
2. 计算IC序列的标准差、偏度
3. 计算正向IC期数占比
4. 综合评分排序

过滤规则：
- IC均值 < 0.03: 预测力太弱
- IC标准差 > 0.15: 波动过大
- 正向占比 < 0.55: 不稳定
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FactorStabilityMetrics:
    """因子稳定性指标"""

    factor_name: str
    ic_mean: float  # IC均值
    ic_std: float  # IC标准差
    ic_skew: float  # IC偏度
    positive_rate: float  # 正向IC占比
    stability_score: float  # 综合稳定性得分
    is_stable: bool  # 是否稳定

    def __repr__(self):
        return (
            f"FactorStability(factor={self.factor_name}, "
            f"ic_mean={self.ic_mean:.4f}, ic_std={self.ic_std:.4f}, "
            f"positive_rate={self.positive_rate:.2%}, "
            f"stability_score={self.stability_score:.4f}, "
            f"stable={self.is_stable})"
        )


class FactorStabilityTester:
    """
    因子稳定性测试器

    评分公式：
    stability_score = 0.4 * ic_mean + 0.3 * positive_rate - 0.3 * ic_std

    稳定性标准：
    - ic_mean >= 0.03
    - ic_std <= 0.15
    - positive_rate >= 0.55
    """

    def __init__(
        self,
        ic_mean_threshold: float = 0.03,
        ic_std_threshold: float = 0.15,
        positive_rate_threshold: float = 0.55,
    ):
        """
        参数:
            ic_mean_threshold: IC均值最低阈值
            ic_std_threshold: IC标准差最高阈值
            positive_rate_threshold: 正向IC占比最低阈值
        """
        self.ic_mean_threshold = ic_mean_threshold
        self.ic_std_threshold = ic_std_threshold
        self.positive_rate_threshold = positive_rate_threshold

    def test_factor_stability(
        self,
        factor_values: np.ndarray,
        returns: np.ndarray,
        window: int = 180,
        step: int = 60,
        min_samples: int = 20,
    ) -> FactorStabilityMetrics:
        """
        测试单个因子的稳定性

        参数:
            factor_values: (T, N) 因子值
            returns: (T, N) 收益率
            window: 滚动窗口大小
            step: 滚动步长
            min_samples: 最小样本数

        返回:
            FactorStabilityMetrics: 稳定性指标
        """
        T, N = factor_values.shape

        # 滚动计算IC
        ic_series = []

        for start_idx in range(0, T - window, step):
            end_idx = start_idx + window

            # 窗口内数据
            window_factors = factor_values[start_idx:end_idx]
            window_returns = returns[start_idx:end_idx]

            # 计算IC (Spearman)
            ic = self._compute_window_ic(window_factors, window_returns, min_samples)
            if not np.isnan(ic):
                ic_series.append(ic)

        if len(ic_series) < 3:
            # 数据不足，标记为不稳定
            return FactorStabilityMetrics(
                factor_name="unknown",
                ic_mean=0.0,
                ic_std=999.0,
                ic_skew=0.0,
                positive_rate=0.0,
                stability_score=-999.0,
                is_stable=False,
            )

        ic_array = np.array(ic_series)

        # 计算统计量
        ic_mean = np.mean(ic_array)
        ic_std = np.std(ic_array)
        ic_skew = stats.skew(ic_array)
        positive_rate = np.mean(ic_array > 0)

        # 综合评分
        stability_score = 0.4 * ic_mean + 0.3 * positive_rate - 0.3 * ic_std

        # 判断稳定性
        is_stable = (
            ic_mean >= self.ic_mean_threshold
            and ic_std <= self.ic_std_threshold
            and positive_rate >= self.positive_rate_threshold
        )

        return FactorStabilityMetrics(
            factor_name="unknown",
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_skew=ic_skew,
            positive_rate=positive_rate,
            stability_score=stability_score,
            is_stable=is_stable,
        )

    def _compute_window_ic(
        self, factors: np.ndarray, returns: np.ndarray, min_samples: int
    ) -> float:
        """
        计算窗口内的平均IC

        参数:
            factors: (T, N) 窗口内因子
            returns: (T, N) 窗口内收益
            min_samples: 最小样本数

        返回:
            float: 平均IC
        """
        T, N = factors.shape
        ic_list = []

        for t in range(T):
            f = factors[t]
            r = returns[t]

            # 过滤NaN
            mask = ~(np.isnan(f) | np.isnan(r))
            if np.sum(mask) < min_samples:
                continue

            f_valid = f[mask]
            r_valid = r[mask]

            # Spearman IC
            try:
                ic, _ = stats.spearmanr(f_valid, r_valid)
                if not np.isnan(ic):
                    ic_list.append(ic)
            except:
                continue

        if len(ic_list) == 0:
            return np.nan

        return np.mean(ic_list)

    def test_all_factors(
        self,
        factors_dict: Dict[str, np.ndarray],
        returns: np.ndarray,
        window: int = 180,
        step: int = 60,
    ) -> Tuple[List[FactorStabilityMetrics], pd.DataFrame]:
        """
        测试所有因子的稳定性

        参数:
            factors_dict: {factor_name: (T, N) array}
            returns: (T, N) 收益率
            window: 滚动窗口
            step: 滚动步长

        返回:
            metrics_list: 稳定性指标列表
            summary_df: 汇总DataFrame
        """
        logger.info(f"🔬 开始因子稳定性测试: {len(factors_dict)} 个因子")

        metrics_list = []

        for factor_name, factor_values in factors_dict.items():
            metrics = self.test_factor_stability(
                factor_values, returns, window=window, step=step
            )
            metrics.factor_name = factor_name
            metrics_list.append(metrics)

            logger.info(
                f"  {factor_name:40} | IC={metrics.ic_mean:+.4f}±{metrics.ic_std:.4f} | "
                f"正向={metrics.positive_rate:.1%} | 稳定={metrics.is_stable}"
            )

        # 按稳定性得分排序
        metrics_list.sort(key=lambda x: x.stability_score, reverse=True)

        # 汇总DataFrame
        summary_df = pd.DataFrame(
            [
                {
                    "factor": m.factor_name,
                    "ic_mean": m.ic_mean,
                    "ic_std": m.ic_std,
                    "ic_skew": m.ic_skew,
                    "positive_rate": m.positive_rate,
                    "stability_score": m.stability_score,
                    "is_stable": m.is_stable,
                }
                for m in metrics_list
            ]
        )

        # 统计
        stable_count = sum(m.is_stable for m in metrics_list)
        logger.info(f"\n✅ 稳定性测试完成:")
        logger.info(
            f"  稳定因子: {stable_count} / {len(metrics_list)} ({stable_count/len(metrics_list):.1%})"
        )
        logger.info(f"  不稳定因子: {len(metrics_list) - stable_count}")

        return metrics_list, summary_df

    def filter_stable_factors(
        self,
        factors_dict: Dict[str, np.ndarray],
        metrics_list: List[FactorStabilityMetrics],
    ) -> Dict[str, np.ndarray]:
        """
        过滤出稳定因子

        参数:
            factors_dict: 原始因子字典
            metrics_list: 稳定性指标列表

        返回:
            stable_factors_dict: 稳定因子字典
        """
        stable_factors = {}

        for metrics in metrics_list:
            if metrics.is_stable:
                if metrics.factor_name in factors_dict:
                    stable_factors[metrics.factor_name] = factors_dict[
                        metrics.factor_name
                    ]

        logger.info(
            f"📊 过滤结果: {len(stable_factors)} / {len(factors_dict)} 因子通过稳定性测试"
        )

        return stable_factors


if __name__ == "__main__":
    """测试代码"""
    import yaml
    from pathlib import Path
    from etf_strategy.core.data_loader import DataLoader
    from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
    from etf_strategy.core.cross_section_processor import CrossSectionProcessor

    print("=" * 80)
    print("🔬 因子稳定性测试")
    print("=" * 80)

    # 加载配置
    ROOT = Path(__file__).parent.parent.parent.parent
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 加载数据
    data_loader = DataLoader(
        data_dir=config["data"]["data_dir"],
        cache_dir=config["data"]["cache_dir"],
    )

    ohlcv_data = data_loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date="2020-01-01",
        end_date="2025-04-30",  # 只用训练期
    )

    print(
        f"✅ 数据加载: {len(ohlcv_data['close'])} 天 × {len(config['data']['symbols'])} ETF"
    )

    # 计算因子
    factor_lib = PreciseFactorLibrary()
    factors_df = factor_lib.compute_all_factors(ohlcv_data)

    # 准备收益率
    returns = ohlcv_data["close"].pct_change().values

    # 转换为dict (直接使用原始因子，不做横截面处理以避免兼容性问题)
    factors_dict = {}
    for factor_name in factors_df.columns.levels[0]:
        factors_dict[factor_name] = factors_df[factor_name].values

    # 测试稳定性
    tester = FactorStabilityTester()
    metrics_list, summary_df = tester.test_all_factors(
        factors_dict, returns, window=180, step=60
    )

    # 显示结果
    print("\n" + "=" * 80)
    print("📊 因子稳定性排名")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # 过滤稳定因子
    stable_factors = tester.filter_stable_factors(factors_dict, metrics_list)

    print("\n✅ 测试完成")
    print(f"稳定因子: {list(stable_factors.keys())}")
