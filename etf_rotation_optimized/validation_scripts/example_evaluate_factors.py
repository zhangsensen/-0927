#!/usr/bin/env python3
"""
示例：如何使用因子验证框架评估新因子

演示三个候选因子的评估流程
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation_scripts.factor_validator import BatchFactorValidator, FactorValidator


# ==================== 示例 1: 短期反转因子 ====================
class ReversalFactor5D(FactorValidator):
    """5日短期反转因子"""

    def compute_factor(self) -> pd.DataFrame:
        """
        计算逻辑：
        - 过去 5 日收益率的负值（跌多→反转预期强）
        - 横截面标准化
        """
        # 5日累计收益率
        ret_5d = self.close.pct_change(periods=5, fill_method=None)

        # 取负值（反转逻辑）
        reversal = -ret_5d

        # 横截面标准化（每日）
        reversal_std = self._cross_sectional_standardize(reversal)

        return reversal_std


# ==================== 示例 2: 波动率偏斜因子 ====================
class VolatilitySkew20D(FactorValidator):
    """20日波动率偏斜因子"""

    def compute_factor(self) -> pd.DataFrame:
        """
        计算逻辑：
        - 下跌日波动率 / 上涨日波动率
        - 健康趋势: skew < 1 (上涨日波动低)
        - 出货特征: skew > 1 (上涨日波动高)
        """
        returns = self.close.pct_change(fill_method=None)

        skew = pd.DataFrame(
            index=self.close.index, columns=self.close.columns, dtype=float
        )

        for col in self.close.columns:
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

        # 横截面标准化
        skew_std = self._cross_sectional_standardize(skew)

        return skew_std


# ==================== 示例 3: 美元成交额加速度因子 ====================
class DollarVolumeAccel10D(FactorValidator):
    """10日美元成交额加速度因子"""

    def compute_factor(self) -> pd.DataFrame:
        """
        计算逻辑：
        - 成交额 = close * volume
        - 加速度 = (最近5日均成交额 - 前5日均成交额) / 前5日均成交额
        """
        # 美元成交额
        dollar_vol = self.close * self.volume

        # 最近5日与前5日均值
        recent_5d = dollar_vol.rolling(window=5, min_periods=5).mean()
        prior_5d = dollar_vol.shift(5).rolling(window=5, min_periods=5).mean()

        # 加速度（百分比变化）
        accel = (recent_5d - prior_5d) / (prior_5d + 1e-8)

        # 横截面标准化
        accel_std = self._cross_sectional_standardize(accel)

        return accel_std


# ==================== 主函数 ====================
def main():
    """主函数：批量评估三个候选因子"""

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

    print(f"📁 数据目录:")
    print(f"  - OHLCV: {ohlcv_dir}")
    print(f"  - 标准化因子: {factors_dir}")

    # 创建因子验证器实例
    validators = [
        ReversalFactor5D(str(ohlcv_dir), str(factors_dir)),
        VolatilitySkew20D(str(ohlcv_dir), str(factors_dir)),
        DollarVolumeAccel10D(str(ohlcv_dir), str(factors_dir)),
    ]

    factor_names = [
        "REVERSAL_FACTOR_5D",
        "VOLATILITY_SKEW_20D",
        "DOLLAR_VOLUME_ACCELERATION_10D",
    ]

    # 批量评估
    batch_validator = BatchFactorValidator(str(ohlcv_dir), str(factors_dir))
    results_df = batch_validator.evaluate_batch(validators, factor_names)

    # 保存结果
    from datetime import datetime

    output_dir = Path(__file__).parent
    output_file = (
        output_dir
        / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    results_df.to_csv(output_file, index=False)

    print(f"\n💾 评估结果已保存: {output_file}")


if __name__ == "__main__":
    main()
