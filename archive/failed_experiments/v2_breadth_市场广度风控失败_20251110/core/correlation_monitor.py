"""因子相关性监控模块 | Correlation Clustering Monitor

原理：
  检测选中因子间相关性热聚集，高相关→"伪分散"→减权
  
  逻辑：
    1. 计算最近窗口内因子间相关矩阵
    2. 取均值（排除对角线）
    3. 若 mean_corr > threshold → 降权
  
  风险与收益：
    - 老项目已有静态去冗（>0.8只保留IC高者）
    - 动态版改善有限，可能画蛇添足
    - 计算成本：每窗口O(F²×T)
  
  建议：
    - 只在窗口切换时检查（IS→OOS），不要每日算
    - 优先级低于广度监控
    - 阈值设保守（0.65-0.7），避免误杀正常相关

Linus判断：
  🟡 Nice to have but not critical
  建议最后加，或直接不加
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class CorrelationSignal:
    """相关性信号"""

    mean_correlation: float  # 平均相关系数
    max_correlation: float  # 最大相关系数
    correlation_penalty: float  # 建议权重惩罚（0-1）
    triggered: bool  # 是否触发高相关预警


class CorrelationMonitor:
    """
    因子相关性监控器
    
    职责：
      - 计算选中因子间相关性矩阵
      - 检测热聚集（高平均相关）
      - 输出建议权重惩罚
    
    参数：
      corr_threshold: 相关性阈值（默认0.65）
      window: 计算窗口长度（默认20）
      min_penalty: 最小惩罚系数（默认0.5，最多减权50%）
      verbose: 是否输出日志
    """

    def __init__(
        self,
        corr_threshold: float = 0.65,
        window: int = 20,
        min_penalty: float = 0.5,
        verbose: bool = True,
    ):
        if not 0 < corr_threshold < 1:
            raise ValueError(
                f"corr_threshold必须在(0,1)之间，当前值: {corr_threshold}"
            )
        if not 0 < min_penalty <= 1:
            raise ValueError(f"min_penalty必须在(0,1]之间，当前值: {min_penalty}")

        self.corr_threshold = corr_threshold
        self.window = window
        self.min_penalty = min_penalty
        self.verbose = verbose

        # 历史记录
        self.history = []

    def calculate_correlation(
        self,
        factor_data: np.ndarray,
        factor_names: Optional[List[str]] = None,
        date: Optional[str] = None,
    ) -> CorrelationSignal:
        """
        计算因子相关性并判断是否触发预警
        
        参数:
          factor_data: (T, N, F) 因子数据（最近T天，N个ETF，F个因子）
                       或 (T, F) 每日因子均值序列
          factor_names: 因子名称列表（用于日志）
          date: 日期字符串（用于日志）
        
        返回:
          CorrelationSignal: 相关性信号对象
        
        实现:
          - 使用np.corrcoef向量化计算
          - O(F²×T)复杂度
        """
        # 确保是2D数组（时间 × 因子）
        if factor_data.ndim == 3:
            # (T, N, F) → (T, F) 取横截面均值
            factor_series = np.nanmean(factor_data, axis=1)
        elif factor_data.ndim == 2:
            factor_series = factor_data
        else:
            raise ValueError(f"factor_data维度错误: {factor_data.ndim}, 期望2或3")

        # 取最近window天
        if len(factor_series) > self.window:
            factor_series = factor_series[-self.window :]

        n_factors = factor_series.shape[1]

        if n_factors < 2:
            # 单因子无相关性问题
            signal = CorrelationSignal(
                mean_correlation=0.0,
                max_correlation=0.0,
                correlation_penalty=1.0,
                triggered=False,
            )
            return signal

        # 计算相关矩阵（排除NaN）
        # 转置：np.corrcoef期望 (F, T)
        corr_matrix = np.corrcoef(factor_series.T)

        # 提取非对角线元素
        mask = ~np.eye(n_factors, dtype=bool)
        off_diag = corr_matrix[mask]

        # 计算统计量
        mean_corr = np.mean(np.abs(off_diag))  # 用绝对值（负相关也是相关）
        max_corr = np.max(np.abs(off_diag))

        # 判断是否触发
        triggered = mean_corr > self.corr_threshold

        # 计算惩罚系数
        if triggered:
            # penalty = min(1.0, threshold / mean_corr)
            # 例：mean_corr=0.8, threshold=0.65 → penalty=0.8125
            raw_penalty = self.corr_threshold / mean_corr
            correlation_penalty = max(self.min_penalty, raw_penalty)
        else:
            correlation_penalty = 1.0

        signal = CorrelationSignal(
            mean_correlation=mean_corr,
            max_correlation=max_corr,
            correlation_penalty=correlation_penalty,
            triggered=triggered,
        )

        # 记录历史
        self.history.append(
            {
                "date": date,
                "mean_corr": mean_corr,
                "max_corr": max_corr,
                "penalty": correlation_penalty,
                "triggered": triggered,
            }
        )

        # 日志输出
        if self.verbose and triggered:
            date_str = f"[{date}] " if date else ""
            factor_str = (
                f" ({', '.join(factor_names)})" if factor_names else ""
            )
            print(
                f"⚠️  {date_str}因子相关性过高{factor_str}: "
                f"mean={mean_corr:.3f} > {self.corr_threshold:.2f}, "
                f"max={max_corr:.3f}, 应用权重惩罚={correlation_penalty:.2%}"
            )

        return signal

    def get_penalty(self, factor_data: np.ndarray) -> float:
        """快速接口：只返回权重惩罚系数"""
        signal = self.calculate_correlation(factor_data)
        return signal.correlation_penalty

    def get_statistics(self) -> dict:
        """统计历史触发情况"""
        if not self.history:
            return {
                "total_checks": 0,
                "triggered_checks": 0,
                "trigger_rate": 0.0,
                "mean_correlation": 0.0,
                "max_correlation": 0.0,
            }

        mean_corrs = [h["mean_corr"] for h in self.history]
        max_corrs = [h["max_corr"] for h in self.history]
        trigger_count = sum(h["triggered"] for h in self.history)

        return {
            "total_checks": len(self.history),
            "triggered_checks": trigger_count,
            "trigger_rate": trigger_count / len(self.history),
            "mean_correlation": np.mean(mean_corrs),
            "max_correlation": np.max(max_corrs),
        }

    def reset_history(self):
        """清空历史记录"""
        self.history = []
