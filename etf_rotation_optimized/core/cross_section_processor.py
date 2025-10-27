"""
跨截面标准化处理模块 | Cross-Section Standardization Processor

功能:
  1. 对无界因子执行日期内横截面 Z-score 标准化
  2. 对无界因子执行 Winsorize 极值截断 (2.5%, 97.5%)
  3. 对有界因子直接透传，不做任何处理
  4. 严格保留 NaN，不做任何填充

工作流:
  原始因子矩阵 (date × symbol × factor)
     ↓
  CrossSectionProcessor.process_all_factors()
     ↓
  标准化因子矩阵 (mean≈0, std≈1, 无极端值, NaN保留)
     ↓
  Step 4: IC计算 & WFO筛选

作者: Step 3 Implementation
日期: 2025-10-26
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FactorMetadata:
    """因子元数据"""

    name: str
    description: str
    bounded: bool
    bounds: Optional[Tuple[float, float]] = None

    def __repr__(self):
        bound_str = (
            f" [{self.bounds[0]:.1f}, {self.bounds[1]:.1f}]"
            if self.bounded
            else " (无界)"
        )
        return f"{self.name}: {self.description}{bound_str}"


class CrossSectionProcessor:
    """
    跨截面标准化处理器

    负责将原始因子矩阵进行标准化处理，包括：
    1. Z-score 标准化 (横截面)
    2. Winsorize 极值截断 (无界因子)
    3. 有界因子透传保护
    4. NaN 严格保留

    属性:
        lower_percentile (float): 下截断分位数 (默认 2.5%)
        upper_percentile (float): 上截断分位数 (默认 97.5%)
        verbose (bool): 是否输出详细信息
        factor_metadata (Dict): 因子元数据
        processing_report (Dict): 处理报告
    """

    # 有界因子名单
    BOUNDED_FACTORS = {
        "PRICE_POSITION_20D",
        "PRICE_POSITION_120D",
        "PV_CORR_20D",
        "RSI_14",
    }

    # 有界因子的值域
    FACTOR_BOUNDS = {
        "PRICE_POSITION_20D": (0.0, 1.0),
        "PRICE_POSITION_120D": (0.0, 1.0),
        "PV_CORR_20D": (-1.0, 1.0),
        "RSI_14": (0.0, 100.0),
    }

    def __init__(
        self,
        lower_percentile: float = 2.5,
        upper_percentile: float = 97.5,
        verbose: bool = True,
    ):
        """
        初始化跨截面处理器

        参数:
            lower_percentile: 下截断分位数 (默认 2.5%)
            upper_percentile: 上截断分位数 (默认 97.5%)
            verbose: 是否输出详细信息
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.verbose = verbose

        # 构建因子元数据
        self.factor_metadata = self._build_metadata()

        # 处理报告
        self.processing_report = {
            "timestamp": None,
            "factors_processed": [],
            "standardization_stats": {},
            "winsorization_stats": {},
            "bounded_factors_passed": [],
            "nan_stats": {},
            "warnings": [],
        }

    def _build_metadata(self) -> Dict[str, FactorMetadata]:
        """构建因子元数据"""
        metadata = {
            # 无界因子
            "MOM_20D": FactorMetadata("MOM_20D", "20日动量百分比", bounded=False),
            "SLOPE_20D": FactorMetadata("SLOPE_20D", "20日线性回归斜率", bounded=False),
            "RET_VOL_20D": FactorMetadata(
                "RET_VOL_20D", "20日收益波动率", bounded=False
            ),
            "MAX_DD_60D": FactorMetadata("MAX_DD_60D", "60日最大回撤", bounded=False),
            "VOL_RATIO_20D": FactorMetadata(
                "VOL_RATIO_20D", "20日成交量比率", bounded=False
            ),
            "VOL_RATIO_60D": FactorMetadata(
                "VOL_RATIO_60D", "60日成交量比率", bounded=False
            ),
            # 有界因子
            "PRICE_POSITION_20D": FactorMetadata(
                "PRICE_POSITION_20D", "20日价格位置", bounded=True, bounds=(0.0, 1.0)
            ),
            "PRICE_POSITION_120D": FactorMetadata(
                "PRICE_POSITION_120D", "120日价格位置", bounded=True, bounds=(0.0, 1.0)
            ),
            "PV_CORR_20D": FactorMetadata(
                "PV_CORR_20D", "20日价量相关性", bounded=True, bounds=(-1.0, 1.0)
            ),
            "RSI_14": FactorMetadata(
                "RSI_14", "14日相对强度", bounded=True, bounds=(0.0, 100.0)
            ),
        }
        return metadata

    def standardize_factor(self, factor: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        对单个因子执行横截面 Z-score 标准化

        参数:
            factor: 因子序列 (index: symbols, values: factor values)

        返回:
            standardized: 标准化后的因子序列
            stats: 标准化统计信息

        说明:
            - NaN 保持 NaN
            - 无效数据（inf）被标记为 NaN
            - 标准化公式: (x - mean) / std
        """
        stats = {
            "count": factor.notna().sum(),
            "original_mean": factor.mean(),
            "original_std": factor.std(),
            "nan_count": factor.isna().sum(),
            "inf_count": np.isinf(pd.to_numeric(factor, errors="coerce")).sum(),
            "standardized_mean": None,
            "standardized_std": None,
        }

        # 移除无效值 (inf)，转换为 numeric
        factor_numeric = pd.to_numeric(factor, errors="coerce")
        factor_clean = factor_numeric.replace([np.inf, -np.inf], np.nan)

        # 更新有效数据count（移除inf后）
        valid_count = factor_clean.count()

        if valid_count < 2:
            # 数据不足，无法标准化
            if self.verbose:
                print(f"⚠️ 警告: 有效数据不足 ({valid_count} < 2)，无法标准化")
            self.processing_report["warnings"].append(
                f"因子有效数据不足 (count={valid_count})"
            )
            stats["count"] = valid_count
            return factor_clean, stats

        # 计算均值和标准差
        mean = factor_clean.mean()
        std = factor_clean.std()

        if std == 0 or np.isnan(std):
            # 方差为0，无法标准化，返回原值
            if self.verbose:
                print(f"⚠️ 警告: 标准差为0，无法标准化")
            self.processing_report["warnings"].append("因子方差为0，无法标准化")
            stats["standardized_mean"] = mean
            stats["standardized_std"] = 0.0
            return factor_clean, stats

        # 执行 Z-score 标准化
        standardized = (factor_clean - mean) / std

        # 更新统计信息
        stats["standardized_mean"] = standardized.mean()
        stats["standardized_std"] = standardized.std()

        return standardized, stats

    def winsorize_factor(self, factor: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        对单个因子执行 Winsorize 极值截断

        参数:
            factor: 因子序列

        返回:
            winsorized: 截断后的因子序列
            stats: 截断统计信息

        说明:
            - 计算下上分位数，将极端值替换为分位数值
            - NaN 保持 NaN，不参与截断
            - 截断后的值被限制在 [lower, upper] 范围内
        """
        stats = {
            "count": factor.notna().sum(),
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "lower_bound": None,
            "upper_bound": None,
            "clipped_count": 0,
            "nan_count": factor.isna().sum(),
        }

        if stats["count"] < 2:
            # 数据不足，跳过截断
            if self.verbose:
                print(f"⚠️ 警告: 有效数据不足 ({stats['count']} < 2)，跳过 Winsorize")
            return factor, stats

        # 计算分位数
        lower_bound = factor.quantile(self.lower_percentile / 100)
        upper_bound = factor.quantile(self.upper_percentile / 100)

        stats["lower_bound"] = lower_bound
        stats["upper_bound"] = upper_bound

        # 计算需要被截断的值的个数
        lower_clip = (factor < lower_bound).sum()
        upper_clip = (factor > upper_bound).sum()
        stats["clipped_count"] = lower_clip + upper_clip

        # 执行截断
        winsorized = factor.clip(lower=lower_bound, upper=upper_bound)

        return winsorized, stats

    def process_factor(
        self, factor_name: str, factor: pd.Series
    ) -> Tuple[pd.Series, Dict]:
        """
        处理单个因子

        参数:
            factor_name: 因子名称
            factor: 因子序列

        返回:
            processed: 处理后的因子序列
            process_stats: 处理统计信息
        """
        process_stats = {
            "factor_name": factor_name,
            "is_bounded": factor_name in self.BOUNDED_FACTORS,
            "steps": [],
        }

        # 1. 检查有界性
        if factor_name in self.BOUNDED_FACTORS:
            # 有界因子：直接透传
            process_stats["steps"].append("passed_through")
            if self.verbose:
                print(f"  ✓ {factor_name:20s} [有界] 直接透传")
            return factor, process_stats

        # 2. 无界因子：执行标准化
        standardized, std_stats = self.standardize_factor(factor)
        process_stats["standardization_stats"] = std_stats
        process_stats["steps"].append("standardized")

        if self.verbose and std_stats["standardized_mean"] is not None:
            print(
                f"  ✓ {factor_name:20s} Z-score标准化: mean={std_stats['standardized_mean']:.4f}, std={std_stats['standardized_std']:.4f}"
            )

        # 3. 无界因子：执行 Winsorize
        winsorized, win_stats = self.winsorize_factor(standardized)
        process_stats["winsorization_stats"] = win_stats
        process_stats["steps"].append("winsorized")

        if self.verbose:
            clipped = win_stats["clipped_count"]
            total = win_stats["count"]
            pct = 100 * clipped / total if total > 0 else 0
            print(
                f"  ✓ {factor_name:20s} Winsorize [{self.lower_percentile}%, {self.upper_percentile}%]: 截断 {clipped}/{total} ({pct:.2f}%)"
            )

        return winsorized, process_stats

    def process_all_factors(
        self, factors_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        批量处理所有因子

        参数:
            factors_dict: 因子字典
                key: 因子名称
                value: 因子 DataFrame (index: date, columns: symbols)

        返回:
            processed_factors: 处理后的因子字典
        """
        self.processing_report["timestamp"] = datetime.now().isoformat()

        if self.verbose:
            print("\n" + "=" * 70)
            print("跨截面标准化处理 | Cross-Section Standardization")
            print("=" * 70)

        processed_factors = {}

        for factor_name, factor_data in factors_dict.items():
            if self.verbose:
                print(f"\n📊 处理因子: {factor_name}")

            # 初始化该因子的统计信息
            factor_results = {}
            factor_nan_stats = {
                "original_nan_count": 0,
                "final_nan_count": 0,
                "nan_preserved": True,
            }

            # 逐日期处理
            processed_dates = []
            for date in factor_data.index:
                series = factor_data.loc[date]

                # 处理该日期的因子序列
                processed_series, process_stats = self.process_factor(
                    factor_name, series
                )

                processed_dates.append(processed_series)

                # 累计 NaN 统计
                factor_nan_stats["original_nan_count"] += series.isna().sum()
                factor_nan_stats["final_nan_count"] += processed_series.isna().sum()

            # 重新组合成 DataFrame
            processed_df = pd.DataFrame(processed_dates)
            processed_df.index = factor_data.index
            processed_factors[factor_name] = processed_df

            # 记录因子处理信息
            self.processing_report["factors_processed"].append(factor_name)
            self.processing_report["nan_stats"][factor_name] = factor_nan_stats

            # 验证 NaN 是否被正确保留
            if (
                factor_nan_stats["original_nan_count"]
                == factor_nan_stats["final_nan_count"]
            ):
                if self.verbose:
                    print(
                        f"  ✓ NaN 保留正确: {factor_nan_stats['original_nan_count']} → {factor_nan_stats['final_nan_count']}"
                    )
            else:
                msg = f"⚠️ NaN 计数不匹配: {factor_nan_stats['original_nan_count']} → {factor_nan_stats['final_nan_count']}"
                if self.verbose:
                    print(msg)
                self.processing_report["warnings"].append(msg)

        if self.verbose:
            print("\n" + "=" * 70)
            print(f"✅ 处理完成: {len(processed_factors)} 个因子")
            print("=" * 70 + "\n")

        return processed_factors

    def get_metadata(self, factor_name: str) -> FactorMetadata:
        """获取因子元数据"""
        return self.factor_metadata.get(factor_name)

    def list_bounded_factors(self) -> list:
        """列出所有有界因子"""
        return list(self.BOUNDED_FACTORS)

    def list_unbounded_factors(self) -> list:
        """列出所有无界因子"""
        unbounded = set(self.factor_metadata.keys()) - self.BOUNDED_FACTORS
        return sorted(list(unbounded))

    def get_factor_bounds(self, factor_name: str) -> Optional[Tuple[float, float]]:
        """获取有界因子的值域"""
        return self.FACTOR_BOUNDS.get(factor_name)

    def get_report(self) -> Dict:
        """获取处理报告"""
        return self.processing_report

    def print_summary(self):
        """打印处理总结"""
        print("\n" + "=" * 70)
        print("处理总结 | Processing Summary")
        print("=" * 70)

        print(f"\n📊 因子处理统计:")
        print(f"  总数: {len(self.processing_report['factors_processed'])}")
        print(f"  有界因子: {len(self.BOUNDED_FACTORS)} (直接透传)")
        print(
            f"  无界因子: {len(self.processing_report['factors_processed']) - len(self.BOUNDED_FACTORS)} (标准化+截断)"
        )

        if self.processing_report["warnings"]:
            print(f"\n⚠️ 警告 ({len(self.processing_report['warnings'])}):")
            for warning in self.processing_report["warnings"]:
                print(f"  - {warning}")
        else:
            print(f"\n✅ 无警告")

        print("\n" + "=" * 70 + "\n")


def create_sample_factors() -> Dict[str, pd.DataFrame]:
    """
    创建示例因子矩阵供测试使用

    返回:
        factors: 因子字典
            key: 因子名称
            value: DataFrame (index: date, columns: symbols)
    """
    np.random.seed(42)

    dates = pd.date_range("2025-01-01", periods=20)
    symbols = [f"ETF{i:02d}" for i in range(30)]

    factors = {}

    # 无界因子
    for factor_name in [
        "MOM_20D",
        "SLOPE_20D",
        "RET_VOL_20D",
        "MAX_DD_60D",
        "VOL_RATIO_20D",
        "VOL_RATIO_60D",
    ]:
        data = (
            np.random.randn(len(dates), len(symbols)) * 10
            + np.random.randn(len(symbols)) * 5
        )
        df = pd.DataFrame(data, index=dates, columns=symbols)
        # 随机插入 NaN
        mask = np.random.random((len(dates), len(symbols))) < 0.05
        df[mask] = np.nan
        factors[factor_name] = df

    # 有界因子
    factors["PRICE_POSITION_20D"] = pd.DataFrame(
        np.random.rand(len(dates), len(symbols)), index=dates, columns=symbols
    )
    factors["PRICE_POSITION_120D"] = pd.DataFrame(
        np.random.rand(len(dates), len(symbols)), index=dates, columns=symbols
    )
    factors["PV_CORR_20D"] = pd.DataFrame(
        np.random.uniform(-1, 1, (len(dates), len(symbols))),
        index=dates,
        columns=symbols,
    )
    factors["RSI_14"] = pd.DataFrame(
        np.random.uniform(0, 100, (len(dates), len(symbols))),
        index=dates,
        columns=symbols,
    )

    return factors


if __name__ == "__main__":
    # 示例使用
    print("示例: CrossSectionProcessor 使用")

    # 创建示例因子
    factors = create_sample_factors()

    # 初始化处理器
    processor = CrossSectionProcessor(verbose=True)

    # 处理因子
    processed = processor.process_all_factors(factors)

    # 打印总结
    processor.print_summary()

    # 验证结果
    print("\n验证:")
    for factor_name, factor_df in processed.items():
        metadata = processor.get_metadata(factor_name)
        if metadata.bounded:
            print(
                f"✓ {factor_name:20s} [有界] min={factor_df.min().min():.4f}, max={factor_df.max().max():.4f}"
            )
        else:
            print(
                f"✓ {factor_name:20s} [无界] mean={factor_df.mean().mean():.4f}, std={factor_df.std().mean():.4f}"
            )
