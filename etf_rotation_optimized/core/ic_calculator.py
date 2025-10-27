"""
IC 计算模块 | IC Calculator Module

功能:
  1. 计算多种相关系数 (Pearson, Spearman, Kendall)
  2. 计算 IC 统计量 (均值、标准差、IR、t-stat、p-value)
  3. 多周期 IC 计算 (1日、5日、20日等)
  4. 详细的 IC 报告生成

工作流:
  标准化因子 + 收益率
    ↓
  按日期计算因子与收益的相关系数
    ↓
  生成 IC 时间序列
    ↓
  计算 IC 统计量 & 显著性检验
    ↓
  输出 IC 报告

作者: Step 4 IC Calculator
日期: 2025-10-26
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ICStats:
    """IC 统计数据"""

    mean: float
    std: float
    ir: float
    t_stat: float
    p_value: float
    sharpe: float
    n_obs: int
    min: float
    max: float
    median: float
    skew: float
    kurtosis: float

    def __repr__(self):
        return f"""
IC 统计量:
  均值:        {self.mean:8.4f}
  标准差:      {self.std:8.4f}
  IR (IR比):   {self.ir:8.4f}
  t-统计量:    {self.t_stat:8.4f}
  p-值:        {self.p_value:8.6f}
  Sharpe比:    {self.sharpe:8.4f}
  观察数:      {self.n_obs:6d}
  
  最小值:      {self.min:8.4f}
  中位数:      {self.median:8.4f}
  最大值:      {self.max:8.4f}
  偏度:        {self.skew:8.4f}
  峰度:        {self.kurtosis:8.4f}
"""


class ICCalculator:
    """
    IC 计算器

    负责计算因子与收益的相关系数及其统计量。

    属性:
        verbose (bool): 是否输出详细信息
        ic_data (Dict): IC 时间序列存储
        ic_stats (Dict): IC 统计量存储
    """

    def __init__(self, verbose: bool = True):
        """
        初始化 IC 计算器

        参数:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.ic_data = {}
        self.ic_stats = {}

    @staticmethod
    def compute_pearson_ic(factor_series: pd.Series, return_series: pd.Series) -> float:
        """
        计算 Pearson 相关系数 (IC)

        参数:
            factor_series: 因子值序列
            return_series: 收益率序列

        返回:
            IC 值 (Pearson 相关系数)

        说明:
            - 只使用有效数据对 (非 NaN)
            - 数据不足 (< 2) 时返回 NaN
        """
        # 移除 NaN
        valid_mask = ~(pd.isna(factor_series) | pd.isna(return_series))

        if valid_mask.sum() < 2:
            return np.nan

        valid_factor = factor_series[valid_mask]
        valid_return = return_series[valid_mask]

        return valid_factor.corr(valid_return)

    @staticmethod
    def compute_spearman_ic(
        factor_series: pd.Series, return_series: pd.Series
    ) -> float:
        """
        计算 Spearman 秩相关系数 (RankIC)

        参数:
            factor_series: 因子值序列
            return_series: 收益率序列

        返回:
            RankIC 值 (Spearman 秩相关系数)

        说明:
            - 对极端值不敏感
            - 更加鲁棒
        """
        # 移除 NaN
        valid_mask = ~(pd.isna(factor_series) | pd.isna(return_series))

        if valid_mask.sum() < 2:
            return np.nan

        valid_factor = factor_series[valid_mask].values
        valid_return = return_series[valid_mask].values

        rho, _ = stats.spearmanr(valid_factor, valid_return)

        return rho if not np.isnan(rho) else np.nan

    @staticmethod
    def compute_kendall_ic(factor_series: pd.Series, return_series: pd.Series) -> float:
        """
        计算 Kendall τ 相关系数

        参数:
            factor_series: 因子值序列
            return_series: 收益率序列

        返回:
            Kendall τ 值

        说明:
            - 非参数相关系数
            - 计算量较大，对大样本可能较慢
        """
        # 移除 NaN
        valid_mask = ~(pd.isna(factor_series) | pd.isna(return_series))

        if valid_mask.sum() < 2:
            return np.nan

        valid_factor = factor_series[valid_mask].values
        valid_return = return_series[valid_mask].values

        tau, _ = stats.kendalltau(valid_factor, valid_return)

        return tau if not np.isnan(tau) else np.nan

    def compute_ic(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        returns_df: pd.DataFrame,
        method: str = "pearson",
        forward_periods: int = 1,
    ) -> Dict[str, pd.Series]:
        """
        计算 IC 时间序列

        参数:
            factors_dict: 标准化因子矩阵
                {factor_name: DataFrame(date × symbol)}
            returns_df: 前向收益率 DataFrame(date × symbol)
            method: 相关系数方法
                'pearson': Pearson 相关系数
                'spearman': Spearman 秩相关系数
                'kendall': Kendall τ
            forward_periods: 前向天数 (用于标记收益)

        返回:
            ic_dict: {factor_name: Series(date)}
                各因子的 IC 时间序列
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"计算 IC 时间序列")
            print(f"{'='*70}")
            print(f"方法: {method}")
            print(f"前向周期: {forward_periods} 天")
            print(f"因子数: {len(factors_dict)}")
            print(f"日期数: {len(returns_df)}")

        ic_dict = {}

        # 获取相关系数计算函数
        if method == "pearson":
            ic_func = self.compute_pearson_ic
        elif method == "spearman":
            ic_func = self.compute_spearman_ic
        elif method == "kendall":
            ic_func = self.compute_kendall_ic
        else:
            raise ValueError(f"未知的相关系数方法: {method}")

        # 对每个因子计算 IC
        for factor_name, factor_df in factors_dict.items():
            ic_series = []
            dates = []

            # 对每个交易日计算 IC
            for date in factor_df.index:
                # 获取该日的因子值
                factor_values = factor_df.loc[date]

                # 查找对应的前向收益
                try:
                    return_idx = returns_df.index.get_loc(date)
                    if return_idx + forward_periods > len(returns_df):
                        # 超出范围，跳过
                        continue

                    return_values = returns_df.iloc[return_idx + forward_periods - 1]
                except (KeyError, IndexError):
                    continue

                # 计算 IC
                ic = ic_func(factor_values, return_values)

                if not np.isnan(ic):
                    ic_series.append(ic)
                    dates.append(date)

            # 创建 IC 时间序列
            ic_dict[factor_name] = pd.Series(ic_series, index=dates)

            if self.verbose:
                n_ic = len(ic_series)
                print(f"  ✓ {factor_name:20s}: {n_ic:3d} 个 IC 值")

        self.ic_data = ic_dict
        return ic_dict

    def compute_ic_stats(self, ic_series: pd.Series, label: str = "") -> ICStats:
        """
        计算 IC 统计量

        参数:
            ic_series: IC 时间序列
            label: 标签 (用于日志)

        返回:
            ICStats: IC 统计数据对象

        计算的指标:
            - 基本统计: 均值、标准差、最小/最大/中位数
            - 风险调整: IR (Information Ratio)、Sharpe 比
            - 显著性: t-统计量、p 值
            - 分布特性: 偏度、峰度
        """
        # 移除 NaN
        valid_ic = ic_series.dropna()

        if len(valid_ic) < 2:
            if self.verbose:
                print(f"⚠️ 数据不足: {label} ({len(valid_ic)} 个观察)")
            return None

        # 基本统计
        mean_ic = valid_ic.mean()
        std_ic = valid_ic.std()

        # IR (Information Ratio) = Mean IC / Std IC
        ir = mean_ic / std_ic if std_ic > 0 else 0

        # Sharpe 比 (年化)
        sharpe = ir * np.sqrt(252)  # 假设 252 个交易日

        # t-统计量
        se = std_ic / np.sqrt(len(valid_ic))
        t_stat = mean_ic / se if se > 0 else 0

        # p-值 (双尾检验)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(valid_ic) - 1))

        # 分布特性
        min_ic = valid_ic.min()
        max_ic = valid_ic.max()
        median_ic = valid_ic.median()
        skew = valid_ic.skew()
        kurtosis = valid_ic.kurtosis()

        stats_obj = ICStats(
            mean=mean_ic,
            std=std_ic,
            ir=ir,
            t_stat=t_stat,
            p_value=p_value,
            sharpe=sharpe,
            n_obs=len(valid_ic),
            min=min_ic,
            max=max_ic,
            median=median_ic,
            skew=skew,
            kurtosis=kurtosis,
        )

        if self.verbose:
            print(f"\n{label} 统计量{stats_obj}")
            # 显著性判断
            if p_value < 0.05:
                print(f"  ✅ 显著: p-value = {p_value:.4f} < 0.05")
            elif p_value < 0.10:
                print(f"  ⚠️ 弱显著: p-value = {p_value:.4f} < 0.10")
            else:
                print(f"  ❌ 不显著: p-value = {p_value:.4f} >= 0.10")

        return stats_obj

    def compute_all_ic_stats(self) -> Dict[str, ICStats]:
        """
        计算所有因子的 IC 统计量

        返回:
            {factor_name: ICStats}
        """
        ic_stats = {}

        for factor_name, ic_series in self.ic_data.items():
            stats_obj = self.compute_ic_stats(ic_series, label=f"{factor_name}")
            if stats_obj is not None:
                ic_stats[factor_name] = stats_obj

        self.ic_stats = ic_stats
        return ic_stats

    def print_ic_report(self, factor_name: str):
        """
        打印单个因子的 IC 报告

        参数:
            factor_name: 因子名称
        """
        if factor_name not in self.ic_data:
            print(f"❌ 因子 {factor_name} 未找到")
            return

        ic_series = self.ic_data[factor_name]
        stats_obj = self.ic_stats.get(factor_name)

        print(f"\n{'='*70}")
        print(f"IC 报告: {factor_name}")
        print(f"{'='*70}")

        print(f"\n基本信息:")
        print(f"  IC 观察数: {len(ic_series)}")
        print(f"  时间范围: {ic_series.index[0]} 至 {ic_series.index[-1]}")

        if stats_obj:
            print(stats_obj)

            print(f"\n结论:")
            if stats_obj.p_value < 0.05:
                print(f"  ✅ 该因子具有显著的预测能力 (p < 0.05)")
            else:
                print(f"  ❌ 该因子无显著预测能力 (p >= 0.05)")

        print(f"\n{'='*70}\n")

    def print_all_ic_reports(self):
        """打印所有因子的 IC 报告"""
        for factor_name in sorted(self.ic_data.keys()):
            self.print_ic_report(factor_name)

    def get_summary_stats(self) -> pd.DataFrame:
        """
        获取所有因子的统计量总结

        返回:
            DataFrame: 因子统计量总结表
        """
        summary_data = []

        for factor_name, stats_obj in self.ic_stats.items():
            summary_data.append(
                {
                    "因子": factor_name,
                    "平均IC": stats_obj.mean,
                    "IC标差": stats_obj.std,
                    "IR": stats_obj.ir,
                    "Sharpe": stats_obj.sharpe,
                    "t-stat": stats_obj.t_stat,
                    "p-值": stats_obj.p_value,
                    "显著": "✅" if stats_obj.p_value < 0.05 else "❌",
                    "观察数": stats_obj.n_obs,
                }
            )

        return pd.DataFrame(summary_data).sort_values("平均IC", ascending=False)


if __name__ == "__main__":
    # 示例用法
    print("IC 计算器示例")

    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=100)
    symbols = [f"ETF{i:02d}" for i in range(30)]

    # 模拟标准化因子
    factor = np.random.randn(len(dates), len(symbols))
    factors_dict = {"TEST_FACTOR": pd.DataFrame(factor, index=dates, columns=symbols)}

    # 模拟前向收益
    returns = np.random.randn(len(dates), len(symbols)) * 0.01 + factor * 0.05
    returns_df = pd.DataFrame(returns, index=dates, columns=symbols)

    # 计算 IC
    calc = ICCalculator()
    ic_dict = calc.compute_ic(factors_dict, returns_df)

    # 计算统计量
    ic_stats = calc.compute_all_ic_stats()

    # 打印报告
    calc.print_all_ic_reports()

    # 输出总结表
    print("\n统计量总结:")
    print(calc.get_summary_stats())
