"""
前向回测框架 | Walk-Forward Optimizer

功能:
  1. 滑动窗口划分 (In-Sample / Out-of-Sample)
  2. 在 IS 窗口进行因子筛选
  3. 在 OOS 窗口进行性能验证
  4. 前向性能评估 (前向 IC、Sharpe 等)
  5. 详细的 WFO 报告

工作流:
  原始数据 (2年历史)
    ↓
  划分 IS/OOS 窗口 (IS:252天, OOS:60天, Step:20天)
    ↓
  Window 1: IS[0:252] 筛选因子 → OOS[252:312] 验证
  Window 2: IS[20:272] 筛选因子 → OOS[272:332] 验证
  ...
  Window N: IS[...] 筛选因子 → OOS[...] 验证
    ↓
  汇总前向性能
    ↓
  生成 WFO 报告

参数:
  - IS_WINDOW: 252 天 (1年)
  - OOS_WINDOW: 60 天 (3个月)
  - STEP: 20 天 (月度rebalance)
  - IC_THRESHOLD: 0.05 (选择 IC > 0.05 的因子)

作者: Step 4 WFO Framework
日期: 2025-10-26
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .ic_calculator import ICCalculator


@dataclass
class WFOWindow:
    """WFO 窗口"""

    window_id: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    selected_factors: List[str]
    is_ic_stats: Dict
    oos_ic_stats: Dict
    oos_performance: Dict

    def __repr__(self):
        return f"""
窗口 {self.window_id}:
  IS 范围:  [{self.is_start}, {self.is_end})
  OOS 范围: [{self.oos_start}, {self.oos_end})
  选中因子: {len(self.selected_factors)} 个 {self.selected_factors}
  OOS Sharpe: {self.oos_performance.get('sharpe', np.nan):.4f}
  OOS 平均IC: {self.oos_performance.get('mean_ic', np.nan):.4f}
"""


@dataclass
class WFOResult:
    """WFO 最终结果"""

    windows: List[WFOWindow]
    forward_mean_ic: float
    forward_sharpe: float
    forward_ir: float
    selected_factors_freq: Dict[str, float]
    all_factors_count: int

    def __repr__(self):
        return f"""
前向回测结果总结:
  窗口数:      {len(self.windows)}
  
  前向性能:
    平均IC:    {self.forward_mean_ic:.4f}
    Sharpe:    {self.forward_sharpe:.4f}
    IR:        {self.forward_ir:.4f}
  
  因子选中频率:
    总因子数:  {self.all_factors_count}
    选中因子:  {len(self.selected_factors_freq)}
    频率范围:  {min(self.selected_factors_freq.values()):.1%} ~ {max(self.selected_factors_freq.values()):.1%}
"""


class WalkForwardOptimizer:
    """
    前向回测优化器

    属性:
        is_window: In-Sample 窗口大小 (交易日)
        oos_window: Out-of-Sample 窗口大小 (交易日)
        step: 滑动步长 (交易日)
        ic_threshold: IC 筛选阈值
        verbose: 是否输出详细信息
    """

    def __init__(
        self,
        is_window: int = 252,
        oos_window: int = 60,
        step: int = 20,
        ic_threshold: float = 0.05,
        verbose: bool = True,
    ):
        """
        初始化 WFO

        参数:
            is_window: In-Sample 窗口 (默认 252 天 ≈ 1年)
            oos_window: Out-of-Sample 窗口 (默认 60 天 ≈ 3个月)
            step: 滑动步长 (默认 20 天 ≈ 月度)
            ic_threshold: IC 筛选阈值 (默认 0.05)
            verbose: 是否输出详细信息
        """
        self.is_window = is_window
        self.oos_window = oos_window
        self.step = step
        self.ic_threshold = ic_threshold
        self.verbose = verbose
        self.windows = []

    def partition_data(self, data_df: pd.DataFrame) -> List[Tuple[int, int, int, int]]:
        """
        划分 IS/OOS 窗口

        参数:
            data_df: 数据 DataFrame (index 为日期或日期索引)

        返回:
            窗口列表: [(is_start, is_end, oos_start, oos_end), ...]

        说明:
            - IS 窗口: [is_start, is_end)
            - OOS 窗口: [oos_start, oos_end)
            - 窗口滑动步长: self.step
        """
        n_days = len(data_df)
        windows = []

        is_start = 0

        while True:
            is_end = is_start + self.is_window
            oos_start = is_end
            oos_end = oos_start + self.oos_window

            # 检查是否超出范围
            if oos_end > n_days:
                break

            windows.append((is_start, is_end, oos_start, oos_end))

            # 滑动窗口
            is_start += self.step

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"数据划分 (WFO 参数)")
            print(f"{'='*70}")
            print(f"总交易日: {n_days}")
            print(f"IS 窗口:  {self.is_window} 天")
            print(f"OOS 窗口: {self.oos_window} 天")
            print(f"滑动步长: {self.step} 天")
            print(f"窗口总数: {len(windows)}")
            print(f"\n窗口划分:")
            for i, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
                print(
                    f"  窗口 {i+1:2d}: IS [{is_s:3d}, {is_e:3d}) "
                    f"OOS [{oos_s:3d}, {oos_e:3d})"
                )

        return windows

    def select_factors(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        returns_df: pd.DataFrame,
        is_start: int,
        is_end: int,
    ) -> Tuple[List[str], Dict]:
        """
        在 IS 窗口内筛选因子

        参数:
            factors_dict: 标准化因子字典
            returns_df: 前向收益率
            is_start: IS 窗口开始索引
            is_end: IS 窗口结束索引

        返回:
            (selected_factors, ic_stats)
            - selected_factors: 选中的因子名称列表
            - ic_stats: 各因子的 IC 统计量
        """
        # 提取 IS 窗口数据
        is_factors = {}
        for name, factor_df in factors_dict.items():
            is_factors[name] = factor_df.iloc[is_start:is_end]

        is_returns = returns_df.iloc[is_start:is_end]

        # 计算 IS 内的 IC
        calc = ICCalculator(verbose=False)
        ic_dict = calc.compute_ic(is_factors, is_returns, method="pearson")
        ic_stats = calc.compute_all_ic_stats()

        # 筛选因子
        selected_factors = []
        for factor_name, stats_obj in ic_stats.items():
            if stats_obj.mean > self.ic_threshold:
                selected_factors.append(factor_name)

        return selected_factors, ic_stats

    def evaluate_factors_oos(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        returns_df: pd.DataFrame,
        selected_factors: List[str],
        oos_start: int,
        oos_end: int,
    ) -> Tuple[Dict, Dict]:
        """
        在 OOS 窗口内评估筛选后的因子

        参数:
            factors_dict: 标准化因子字典
            returns_df: 前向收益率
            selected_factors: 选中的因子名称列表
            oos_start: OOS 窗口开始索引
            oos_end: OOS 窗口结束索引

        返回:
            (oos_ic_stats, oos_performance)
        """
        # 提取 OOS 窗口数据
        oos_factors = {}
        for name in selected_factors:
            if name in factors_dict:
                oos_factors[name] = factors_dict[name].iloc[oos_start:oos_end]

        oos_returns = returns_df.iloc[oos_start:oos_end]

        # 计算 OOS 内的 IC
        calc = ICCalculator(verbose=False)
        ic_dict = calc.compute_ic(oos_factors, oos_returns, method="pearson")
        ic_stats = calc.compute_all_ic_stats()

        # 汇总性能指标
        performance = {
            "mean_ic": (
                np.mean([s.mean for s in ic_stats.values()]) if ic_stats else np.nan
            ),
            "ic_std": (
                np.mean([s.std for s in ic_stats.values()]) if ic_stats else np.nan
            ),
            "ir": np.mean([s.ir for s in ic_stats.values()]) if ic_stats else np.nan,
            "sharpe": (
                np.mean([s.sharpe for s in ic_stats.values()]) if ic_stats else np.nan
            ),
        }

        return ic_stats, performance

    def walk_forward(
        self, factors_dict: Dict[str, pd.DataFrame], returns_df: pd.DataFrame
    ) -> WFOResult:
        """
        执行前向回测

        参数:
            factors_dict: 标准化因子字典 {factor_name: DataFrame(date×symbol)}
            returns_df: 前向收益率 DataFrame(date×symbol)

        返回:
            WFOResult: 前向回测结果
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"执行前向回测 (WFO)")
            print(f"{'='*70}")

        # 划分窗口
        window_partitions = self.partition_data(returns_df)

        # 执行逐窗口回测
        self.windows = []
        selected_factors_all = {}
        oos_ic_all = []
        oos_sharpe_all = []
        oos_ir_all = []

        for window_id, (is_s, is_e, oos_s, oos_e) in enumerate(window_partitions):
            if self.verbose:
                print(
                    f"\n  处理窗口 {window_id + 1}/{len(window_partitions)}...", end=""
                )

            # 在 IS 窗口内筛选因子
            selected_factors, is_ic_stats = self.select_factors(
                factors_dict, returns_df, is_s, is_e
            )

            # 在 OOS 窗口内评估因子
            oos_ic_stats, oos_performance = self.evaluate_factors_oos(
                factors_dict, returns_df, selected_factors, oos_s, oos_e
            )

            # 记录选中的因子
            for factor in selected_factors:
                selected_factors_all[factor] = selected_factors_all.get(factor, 0) + 1

            # 记录 OOS 性能指标
            mean_ic = oos_performance.get("mean_ic")
            sharpe = oos_performance.get("sharpe")
            ir = oos_performance.get("ir")

            if mean_ic is not None and not np.isnan(mean_ic):
                oos_ic_all.append(mean_ic)
            if sharpe is not None and not np.isnan(sharpe):
                oos_sharpe_all.append(sharpe)
            if ir is not None and not np.isnan(ir):
                oos_ir_all.append(ir)

            # 创建窗口对象
            window = WFOWindow(
                window_id=window_id + 1,
                is_start=is_s,
                is_end=is_e,
                oos_start=oos_s,
                oos_end=oos_e,
                selected_factors=selected_factors,
                is_ic_stats={k: v.mean for k, v in is_ic_stats.items()},
                oos_ic_stats={k: v.mean for k, v in oos_ic_stats.items()},
                oos_performance=oos_performance,
            )
            self.windows.append(window)

            if self.verbose:
                n_factors = len(selected_factors)
                sharpe = oos_performance.get("sharpe", np.nan)
                print(f" ✓ {n_factors:2d} 个因子, OOS Sharpe={sharpe:.4f}")

        # 计算前向性能汇总
        if len(oos_ic_all) > 0:
            forward_mean_ic = np.mean(oos_ic_all)
        else:
            forward_mean_ic = 0.0  # 如果没有有效 IC，设为 0

        if len(oos_sharpe_all) > 0:
            forward_sharpe = np.mean(oos_sharpe_all)
        else:
            forward_sharpe = 0.0

        if len(oos_ir_all) > 0:
            forward_ir = np.mean(oos_ir_all)
        else:
            forward_ir = 0.0

        # 计算因子选中频率
        selected_factors_freq = {}
        total_windows = len(self.windows) if self.windows else 1
        for factor, count in selected_factors_all.items():
            selected_factors_freq[factor] = count / total_windows

        # 创建结果对象
        result = WFOResult(
            windows=self.windows,
            forward_mean_ic=forward_mean_ic,
            forward_sharpe=forward_sharpe,
            forward_ir=forward_ir,
            selected_factors_freq=selected_factors_freq,
            all_factors_count=len(factors_dict),
        )

        return result

    def print_wfo_report(self, result: WFOResult):
        """
        打印 WFO 报告

        参数:
            result: WFOResult 对象
        """
        print(f"\n{'='*70}")
        print(f"前向回测总结")
        print(f"{'='*70}")
        print(result)

        print(f"\n因子选中频率:")
        if result.selected_factors_freq:
            freq_df = pd.DataFrame(
                list(result.selected_factors_freq.items()), columns=["因子", "选中频率"]
            ).sort_values("选中频率", ascending=False)

            for _, row in freq_df.iterrows():
                factor = row["因子"]
                freq = row["选中频率"]
                bar = "█" * int(freq * 20)
                print(f"  {factor:20s}: {freq:6.1%} {bar}")
        else:
            print("  (无因子被选中)")

        print(f"\n逐窗口性能:")
        for window in result.windows:
            print(
                f"  窗口 {window.window_id:2d}: "
                f"{len(window.selected_factors):2d} 因子, "
                f"OOS IC={window.oos_performance.get('mean_ic', np.nan):7.4f}, "
                f"Sharpe={window.oos_performance.get('sharpe', np.nan):7.4f}"
            )

        print(f"\n{'='*70}\n")

    def get_summary_dataframe(self, result: WFOResult) -> pd.DataFrame:
        """
        获取 WFO 窗口总结表

        参数:
            result: WFOResult 对象

        返回:
            DataFrame: 窗口总结表
        """
        summary_data = []

        for window in result.windows:
            summary_data.append(
                {
                    "窗口": window.window_id,
                    "IS_Start": window.is_start,
                    "IS_End": window.is_end,
                    "OOS_Start": window.oos_start,
                    "OOS_End": window.oos_end,
                    "选中因子数": len(window.selected_factors),
                    "选中因子": ",".join(window.selected_factors[:3])
                    + ("..." if len(window.selected_factors) > 3 else ""),
                    "OOS_IC": window.oos_performance.get("mean_ic", np.nan),
                    "OOS_Sharpe": window.oos_performance.get("sharpe", np.nan),
                    "OOS_IR": window.oos_performance.get("ir", np.nan),
                }
            )

        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # 示例用法
    print("WFO 框架示例")

    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=600)  # 约 2.4 年
    symbols = [f"ETF{i:02d}" for i in range(30)]

    # 模拟标准化因子 (3 个)
    factors_dict = {}
    for i in range(3):
        factor_data = np.random.randn(len(dates), len(symbols))
        factors_dict[f"FACTOR_{i}"] = pd.DataFrame(
            factor_data, index=dates, columns=symbols
        )

    # 模拟前向收益 (有因子预测能力)
    base = np.random.randn(len(dates), len(symbols))
    returns_df = pd.DataFrame(
        base * 0.05 + np.random.randn(len(dates), len(symbols)) * 0.02,
        index=dates,
        columns=symbols,
    )

    # 执行 WFO
    wfo = WalkForwardOptimizer(is_window=252, oos_window=60, step=20, ic_threshold=0.05)

    result = wfo.walk_forward(factors_dict, returns_df)

    # 打印报告
    wfo.print_wfo_report(result)

    # 输出总结表
    print("\n逐窗口总结表:")
    print(wfo.get_summary_dataframe(result))
