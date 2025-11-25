"""波动率目标模块 | Volatility Targeting

原理：
  组合年化波动率 > 阈值时，按比例降低杠杆（等价于减仓）
  
  公式：
    realized_vol = std(returns) * sqrt(252)
    if realized_vol > target_vol:
        scale = min(1.0, target_vol / realized_vol)
    else:
        scale = 1.0
  
  风险：
    - 滞后性：20D/60D是已发生波动，暴跌初期反应慢
    - 可能踏空反弹：高波动 ≠ 负收益（如2020年3月）
    - 增加交易成本：每次调整→全组合按比例调仓
  
  改进：
    - 用短期vs长期波动比值（3D vs 20D）减少滞后
    - 阈值设高（30%+），只在极端情况触发
    - 可选：用实现波动率vs隐含波动率差值

Linus判断：
  🟡 OK but risky - 必须回测验证不会在关键反弹期空仓
  建议优先级低于市场广度
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VolatilitySignal:
    """波动率信号"""

    realized_vol_20d: float  # 20日年化波动率
    realized_vol_60d: float  # 60日年化波动率
    vol_signal: float  # 取短长期较高者
    target_vol: float  # 目标波动率
    vol_scale: float  # 建议仓位比例
    triggered: bool  # 是否触发降杠杆


class VolatilityTargeting:
    """
    波动率目标管理器
    
    职责：
      - 计算组合实现波动率（20D/60D）
      - 判断是否超过目标阈值
      - 输出建议仓位scale
    
    参数：
      target_vol: 目标年化波动率（默认0.25，即25%）
      min_window: 最小计算窗口（默认20）
      max_scale: 最大仓位比例（默认1.0，不加杠杆）
      min_scale: 最小仓位比例（默认0.3，最多降至30%）
      verbose: 是否输出日志
    """

    def __init__(
        self,
        target_vol: float = 0.25,
        min_window: int = 20,
        max_scale: float = 1.0,
        min_scale: float = 0.3,
        verbose: bool = True,
    ):
        if target_vol <= 0:
            raise ValueError(f"target_vol必须>0，当前值: {target_vol}")
        if not 0 < min_scale <= max_scale <= 2.0:
            raise ValueError(
                f"scale范围错误: min_scale={min_scale}, max_scale={max_scale}"
            )

        self.target_vol = target_vol
        self.min_window = min_window
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.verbose = verbose

        # 历史记录
        self.history = []

    def calculate_volatility(
        self, portfolio_returns: np.ndarray, date: Optional[str] = None
    ) -> VolatilitySignal:
        """
        计算实现波动率并判断是否降杠杆
        
        参数:
          portfolio_returns: (T,) 组合日收益率序列（最新在最后）
          date: 日期字符串（用于日志）
        
        返回:
          VolatilitySignal: 波动率信号对象
        
        实现:
          - 向量化std计算，O(T)复杂度
          - 短期(20D) vs 长期(60D)取较高者
        """
        if portfolio_returns.ndim != 1:
            raise ValueError(
                f"portfolio_returns必须是1维数组，当前维度: {portfolio_returns.ndim}"
            )

        # 计算20日波动率
        if len(portfolio_returns) >= 20:
            vol_20d = np.std(portfolio_returns[-20:]) * np.sqrt(252)
        else:
            vol_20d = 0.0

        # 计算60日波动率
        if len(portfolio_returns) >= 60:
            vol_60d = np.std(portfolio_returns[-60:]) * np.sqrt(252)
        else:
            vol_60d = 0.0

        # 取短长期较高者（更快捕捉波动突增）
        vol_signal = max(vol_20d, vol_60d) if vol_20d > 0 or vol_60d > 0 else 0.0

        # 计算建议scale
        if vol_signal > self.target_vol and vol_signal > 0:
            raw_scale = self.target_vol / vol_signal
            vol_scale = np.clip(raw_scale, self.min_scale, self.max_scale)
            triggered = True
        else:
            vol_scale = self.max_scale
            triggered = False

        signal = VolatilitySignal(
            realized_vol_20d=vol_20d,
            realized_vol_60d=vol_60d,
            vol_signal=vol_signal,
            target_vol=self.target_vol,
            vol_scale=vol_scale,
            triggered=triggered,
        )

        # 记录历史
        self.history.append(
            {
                "date": date,
                "vol_20d": vol_20d,
                "vol_60d": vol_60d,
                "vol_scale": vol_scale,
                "triggered": triggered,
            }
        )

        # 日志输出
        if self.verbose and triggered:
            date_str = f"[{date}] " if date else ""
            print(
                f"⚠️  {date_str}波动率过高: {vol_signal:.2%} > {self.target_vol:.0%}, "
                f"降杠杆至 {vol_scale:.1%} (20D={vol_20d:.2%}, 60D={vol_60d:.2%})"
            )

        return signal

    def get_position_scale(self, portfolio_returns: np.ndarray) -> float:
        """快速接口：只返回仓位scale"""
        signal = self.calculate_volatility(portfolio_returns)
        return signal.vol_scale

    def get_statistics(self) -> dict:
        """统计历史触发情况"""
        if not self.history:
            return {
                "total_days": 0,
                "triggered_days": 0,
                "trigger_rate": 0.0,
                "mean_vol_20d": 0.0,
                "max_vol_20d": 0.0,
            }

        vol_20d_list = [h["vol_20d"] for h in self.history if h["vol_20d"] > 0]
        trigger_count = sum(h["triggered"] for h in self.history)

        return {
            "total_days": len(self.history),
            "triggered_days": trigger_count,
            "trigger_rate": trigger_count / len(self.history),
            "mean_vol_20d": np.mean(vol_20d_list) if vol_20d_list else 0.0,
            "max_vol_20d": np.max(vol_20d_list) if vol_20d_list else 0.0,
        }

    def reset_history(self):
        """清空历史记录"""
        self.history = []
