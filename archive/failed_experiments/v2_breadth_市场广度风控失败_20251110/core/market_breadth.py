"""市场广度监控模块 | Market Breadth Monitor

原理：
  检测有效因子信号占比，当大部分ETF信号失效时触发防守模式。
  
  有效信号定义：
    - 复合因子得分 > threshold（默认0，即z-score正值）
    - 占比 < breadth_floor 时触发降仓
  
  优势：
    - 比波动率更早捕捉信号崩溃
    - 直接检测模型失效而非市场波动
    - 成本低，无额外重计算
  
  集成点：
    WFO输出复合因子得分后 → 计算广度 → 决定仓位scale

Linus原则：
  - 向量化计算，O(N)复杂度
  - 明确阈值，可回测验证
  - 低侵入，不改变因子选择逻辑
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BreadthSignal:
    """市场广度信号"""

    breadth: float  # 0-1之间，有效信号占比
    valid_count: int  # 有效ETF数量
    total_count: int  # 总ETF数量
    defensive_mode: bool  # 是否触发防守模式
    position_scale: float  # 建议仓位比例


class MarketBreadthMonitor:
    """
    市场广度监控器
    
    职责：
      - 计算当日有效因子信号占比
      - 判断是否触发防守模式
      - 输出建议仓位scale
    
    参数：
      breadth_floor: 最低广度阈值（默认0.25，即25%有效）
      score_threshold: 最低有效得分（默认0.0，z-score正值）
      defensive_scale: 防守模式仓位比例（默认0.5，降至50%）
      verbose: 是否输出日志
    """

    def __init__(
        self,
        breadth_floor: float = 0.25,
        score_threshold: float = 0.0,
        defensive_scale: float = 0.5,
        verbose: bool = True,
    ):
        if not 0 < breadth_floor < 1:
            raise ValueError(f"breadth_floor必须在(0,1)之间，当前值: {breadth_floor}")
        if not 0 < defensive_scale <= 1:
            raise ValueError(
                f"defensive_scale必须在(0,1]之间，当前值: {defensive_scale}"
            )

        self.breadth_floor = breadth_floor
        self.score_threshold = score_threshold
        self.defensive_scale = defensive_scale
        self.verbose = verbose

        # 历史记录
        self.history = []

    def calculate_breadth(
        self, factor_scores: np.ndarray, date: Optional[str] = None
    ) -> BreadthSignal:
        """
        计算市场广度并判断防守模式
        
        参数:
          factor_scores: (N,) 当日复合因子得分（已标准化）
          date: 日期字符串（用于日志）
        
        返回:
          BreadthSignal: 广度信号对象
        
        实现:
          - O(N)向量化判断
          - 无循环，无.apply()
        """
        if factor_scores.ndim != 1:
            raise ValueError(
                f"factor_scores必须是1维数组，当前维度: {factor_scores.ndim}"
            )

        # 向量化判断有效信号
        valid_mask = factor_scores > self.score_threshold
        valid_count = np.sum(valid_mask)
        total_count = len(factor_scores)

        # 计算广度
        breadth = valid_count / total_count if total_count > 0 else 0.0

        # 判断防守模式
        defensive_mode = breadth < self.breadth_floor
        position_scale = self.defensive_scale if defensive_mode else 1.0

        signal = BreadthSignal(
            breadth=breadth,
            valid_count=int(valid_count),
            total_count=int(total_count),
            defensive_mode=defensive_mode,
            position_scale=position_scale,
        )

        # 记录历史
        self.history.append(
            {
                "date": date,
                "breadth": breadth,
                "defensive_mode": defensive_mode,
                "position_scale": position_scale,
            }
        )

        # 日志输出
        if self.verbose and defensive_mode:
            date_str = f"[{date}] " if date else ""
            print(
                f"⚠️  {date_str}市场广度崩溃: {breadth:.2%} "
                f"({valid_count}/{total_count}) < {self.breadth_floor:.0%}, "
                f"进入防守模式 (scale={position_scale:.1%})"
            )

        return signal

    def get_position_scale(self, factor_scores: np.ndarray) -> float:
        """
        快速接口：只返回仓位scale
        
        用于实际交易决策
        """
        signal = self.calculate_breadth(factor_scores)
        return signal.position_scale

    def get_statistics(self) -> dict:
        """
        统计历史触发情况
        
        返回:
          {
            'total_days': 总天数,
            'defensive_days': 防守天数,
            'defensive_rate': 防守占比,
            'mean_breadth': 平均广度,
            'min_breadth': 最低广度,
          }
        """
        if not self.history:
            return {
                "total_days": 0,
                "defensive_days": 0,
                "defensive_rate": 0.0,
                "mean_breadth": 0.0,
                "min_breadth": 0.0,
            }

        breadths = [h["breadth"] for h in self.history]
        defensive_count = sum(h["defensive_mode"] for h in self.history)

        return {
            "total_days": len(self.history),
            "defensive_days": defensive_count,
            "defensive_rate": defensive_count / len(self.history),
            "mean_breadth": np.mean(breadths),
            "min_breadth": np.min(breadths),
        }

    def reset_history(self):
        """清空历史记录"""
        self.history = []
