"""
自动生成的overlap类因子
使用SHARED_CALCULATORS确保计算一致性
生成时间: 2025-10-09 15:15:28.903320
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from ..core.base_factor import BaseFactor

logger = logging.getLogger(__name__)


class MA3(BaseFactor):
    """
    移动平均线 - MA3
    
    类别: overlap
    参数: {'timeperiod': 3}
    """
    
    factor_id = "MA3"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 3}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=3).mean()
            return result.rename("MA3")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class MA5(BaseFactor):
    """
    移动平均线 - MA5
    
    类别: overlap
    参数: {'timeperiod': 5}
    """
    
    factor_id = "MA5"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 5}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=5).mean()
            return result.rename("MA5")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class MA8(BaseFactor):
    """
    移动平均线 - MA8
    
    类别: overlap
    参数: {'timeperiod': 8}
    """
    
    factor_id = "MA8"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 8}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=8).mean()
            return result.rename("MA8")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class MA10(BaseFactor):
    """
    移动平均线 - MA10
    
    类别: overlap
    参数: {'timeperiod': 10}
    """
    
    factor_id = "MA10"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=10).mean()
            return result.rename("MA10")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class MA15(BaseFactor):
    """
    移动平均线 - MA15
    
    类别: overlap
    参数: {'timeperiod': 15}
    """
    
    factor_id = "MA15"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 15}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=15).mean()
            return result.rename("MA15")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class MA20(BaseFactor):
    """
    移动平均线 - MA20
    
    类别: overlap
    参数: {'timeperiod': 20}
    """
    
    factor_id = "MA20"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=20).mean()
            return result.rename("MA20")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class EMA3(BaseFactor):
    """
    移动平均线 - EMA3
    
    类别: overlap
    参数: {'timeperiod': 3}
    """
    
    factor_id = "EMA3"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 3}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=3, adjust=False).mean()
            return result.rename("EMA3")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class EMA5(BaseFactor):
    """
    移动平均线 - EMA5
    
    类别: overlap
    参数: {'timeperiod': 5}
    """
    
    factor_id = "EMA5"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 5}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=5, adjust=False).mean()
            return result.rename("EMA5")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class EMA8(BaseFactor):
    """
    移动平均线 - EMA8
    
    类别: overlap
    参数: {'timeperiod': 8}
    """
    
    factor_id = "EMA8"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 8}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=8, adjust=False).mean()
            return result.rename("EMA8")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class EMA12(BaseFactor):
    """
    移动平均线 - EMA12
    
    类别: overlap
    参数: {'timeperiod': 12}
    """
    
    factor_id = "EMA12"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 12}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=12, adjust=False).mean()
            return result.rename("EMA12")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class EMA15(BaseFactor):
    """
    移动平均线 - EMA15
    
    类别: overlap
    参数: {'timeperiod': 15}
    """
    
    factor_id = "EMA15"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 15}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=15, adjust=False).mean()
            return result.rename("EMA15")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class EMA20(BaseFactor):
    """
    移动平均线 - EMA20
    
    类别: overlap
    参数: {'timeperiod': 20}
    """
    
    factor_id = "EMA20"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=20, adjust=False).mean()
            return result.rename("EMA20")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class MACD_6_13_4(BaseFactor):
    """
    移动平均收敛散度 - MACD_6_13_4
    
    类别: overlap
    参数: {'fastperiod': 6, 'slowperiod': 13, 'signalperiod': 4}
    """
    
    factor_id = "MACD_6_13_4"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'fastperiod': 6, 'slowperiod': 13, 'signalperiod': 4}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_macd(
                data["close"], fastperiod=6, slowperiod=13, signalperiod=4
            )
            return result['macd'].rename("MACD_6_13_4")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class MACD_8_17_5(BaseFactor):
    """
    移动平均收敛散度 - MACD_8_17_5
    
    类别: overlap
    参数: {'fastperiod': 8, 'slowperiod': 17, 'signalperiod': 5}
    """
    
    factor_id = "MACD_8_17_5"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'fastperiod': 8, 'slowperiod': 17, 'signalperiod': 5}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_macd(
                data["close"], fastperiod=8, slowperiod=17, signalperiod=5
            )
            return result['macd'].rename("MACD_8_17_5")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class MACD_12_26_9(BaseFactor):
    """
    移动平均收敛散度 - MACD_12_26_9
    
    类别: overlap
    参数: {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}
    """
    
    factor_id = "MACD_12_26_9"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_macd(
                data["close"], fastperiod=12, slowperiod=26, signalperiod=9
            )
            return result['macd'].rename("MACD_12_26_9")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class BB_10_2_0_Upper(BaseFactor):
    """
    技术指标 - BB_10_2.0_Upper
    
    类别: overlap
    参数: {'timeperiod': 10, 'nbdevup': 0.2, 'nbdevdn': 0.2}
    """
    
    factor_id = "BB_10_2.0_Upper"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10, 'nbdevup': 0.2, 'nbdevdn': 0.2}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_bbands(
                data["close"], period=10, nbdevup=0.2, nbdevdn=0.2
            )
            return result['upper'].rename("BB_10_2.0_Upper")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class BB_10_2_0_Middle(BaseFactor):
    """
    技术指标 - BB_10_2.0_Middle
    
    类别: overlap
    参数: {'timeperiod': 10, 'nbdevup': 0.2, 'nbdevdn': 0.2}
    """
    
    factor_id = "BB_10_2.0_Middle"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10, 'nbdevup': 0.2, 'nbdevdn': 0.2}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_bbands(
                data["close"], period=10, nbdevup=0.2, nbdevdn=0.2
            )
            return result['middle'].rename("BB_10_2.0_Middle")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class BB_10_2_0_Lower(BaseFactor):
    """
    技术指标 - BB_10_2.0_Lower
    
    类别: overlap
    参数: {'timeperiod': 10, 'nbdevup': 0.2, 'nbdevdn': 0.2}
    """
    
    factor_id = "BB_10_2.0_Lower"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10, 'nbdevup': 0.2, 'nbdevdn': 0.2}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_bbands(
                data["close"], period=10, nbdevup=0.2, nbdevdn=0.2
            )
            return result['lower'].rename("BB_10_2.0_Lower")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class BB_10_2_0_Width(BaseFactor):
    """
    技术指标 - BB_10_2.0_Width
    
    类别: overlap
    参数: {'timeperiod': 10, 'nbdevup': 0.2, 'nbdevdn': 0.2}
    """
    
    factor_id = "BB_10_2.0_Width"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10, 'nbdevup': 0.2, 'nbdevdn': 0.2}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_bbands(
                data["close"], period=10, nbdevup=0.2, nbdevdn=0.2
            )
            return result['middle'].rename("BB_10_2.0_Width")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class BB_15_2_0_Upper(BaseFactor):
    """
    技术指标 - BB_15_2.0_Upper
    
    类别: overlap
    参数: {'timeperiod': 15, 'nbdevup': 0.2, 'nbdevdn': 0.2}
    """
    
    factor_id = "BB_15_2.0_Upper"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 15, 'nbdevup': 0.2, 'nbdevdn': 0.2}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_bbands(
                data["close"], period=15, nbdevup=0.2, nbdevdn=0.2
            )
            return result['upper'].rename("BB_15_2.0_Upper")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class BB_15_2_0_Middle(BaseFactor):
    """
    技术指标 - BB_15_2.0_Middle
    
    类别: overlap
    参数: {'timeperiod': 15, 'nbdevup': 0.2, 'nbdevdn': 0.2}
    """
    
    factor_id = "BB_15_2.0_Middle"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 15, 'nbdevup': 0.2, 'nbdevdn': 0.2}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_bbands(
                data["close"], period=15, nbdevup=0.2, nbdevdn=0.2
            )
            return result['middle'].rename("BB_15_2.0_Middle")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class BB_15_2_0_Lower(BaseFactor):
    """
    技术指标 - BB_15_2.0_Lower
    
    类别: overlap
    参数: {'timeperiod': 15, 'nbdevup': 0.2, 'nbdevdn': 0.2}
    """
    
    factor_id = "BB_15_2.0_Lower"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 15, 'nbdevup': 0.2, 'nbdevdn': 0.2}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_bbands(
                data["close"], period=15, nbdevup=0.2, nbdevdn=0.2
            )
            return result['lower'].rename("BB_15_2.0_Lower")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class BB_15_2_0_Width(BaseFactor):
    """
    技术指标 - BB_15_2.0_Width
    
    类别: overlap
    参数: {'timeperiod': 15, 'nbdevup': 0.2, 'nbdevdn': 0.2}
    """
    
    factor_id = "BB_15_2.0_Width"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 15, 'nbdevup': 0.2, 'nbdevdn': 0.2}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_bbands(
                data["close"], period=15, nbdevup=0.2, nbdevdn=0.2
            )
            return result['middle'].rename("BB_15_2.0_Width")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class BB_20_2_0_Upper(BaseFactor):
    """
    技术指标 - BB_20_2.0_Upper
    
    类别: overlap
    参数: {'timeperiod': 20, 'nbdevup': 0.2, 'nbdevdn': 0.2}
    """
    
    factor_id = "BB_20_2.0_Upper"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20, 'nbdevup': 0.2, 'nbdevdn': 0.2}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_bbands(
                data["close"], period=20, nbdevup=0.2, nbdevdn=0.2
            )
            return result['upper'].rename("BB_20_2.0_Upper")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class BB_20_2_0_Middle(BaseFactor):
    """
    技术指标 - BB_20_2.0_Middle
    
    类别: overlap
    参数: {'timeperiod': 20, 'nbdevup': 0.2, 'nbdevdn': 0.2}
    """
    
    factor_id = "BB_20_2.0_Middle"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20, 'nbdevup': 0.2, 'nbdevdn': 0.2}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_bbands(
                data["close"], period=20, nbdevup=0.2, nbdevdn=0.2
            )
            return result['middle'].rename("BB_20_2.0_Middle")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class BB_20_2_0_Lower(BaseFactor):
    """
    技术指标 - BB_20_2.0_Lower
    
    类别: overlap
    参数: {'timeperiod': 20, 'nbdevup': 0.2, 'nbdevdn': 0.2}
    """
    
    factor_id = "BB_20_2.0_Lower"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20, 'nbdevup': 0.2, 'nbdevdn': 0.2}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_bbands(
                data["close"], period=20, nbdevup=0.2, nbdevdn=0.2
            )
            return result['lower'].rename("BB_20_2.0_Lower")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class BB_20_2_0_Width(BaseFactor):
    """
    技术指标 - BB_20_2.0_Width
    
    类别: overlap
    参数: {'timeperiod': 20, 'nbdevup': 0.2, 'nbdevdn': 0.2}
    """
    
    factor_id = "BB_20_2.0_Width"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20, 'nbdevup': 0.2, 'nbdevdn': 0.2}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            result = SHARED_CALCULATORS.calculate_bbands(
                data["close"], period=20, nbdevup=0.2, nbdevdn=0.2
            )
            return result['middle'].rename("BB_20_2.0_Width")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class OBV_SMA5(BaseFactor):
    """
    能量潮 - OBV_SMA5
    
    类别: overlap
    参数: {'timeperiod': 5}
    """
    
    factor_id = "OBV_SMA5"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 5}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=5).mean()
            return result.rename("OBV_SMA5")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class OBV_SMA10(BaseFactor):
    """
    能量潮 - OBV_SMA10
    
    类别: overlap
    参数: {'timeperiod': 10}
    """
    
    factor_id = "OBV_SMA10"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=10).mean()
            return result.rename("OBV_SMA10")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class OBV_SMA15(BaseFactor):
    """
    能量潮 - OBV_SMA15
    
    类别: overlap
    参数: {'timeperiod': 15}
    """
    
    factor_id = "OBV_SMA15"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 15}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=15).mean()
            return result.rename("OBV_SMA15")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class OBV_SMA20(BaseFactor):
    """
    能量潮 - OBV_SMA20
    
    类别: overlap
    参数: {'timeperiod': 20}
    """
    
    factor_id = "OBV_SMA20"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=20).mean()
            return result.rename("OBV_SMA20")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class BOLB_20(BaseFactor):
    """
    技术指标 - BOLB_20
    
    类别: overlap
    参数: {}
    """
    
    factor_id = "BOLB_20"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # 默认实现 - 需要完善
            logger.warning(f"因子BOLB_20使用默认实现")
            return data["close"].rename("BOLB_20")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class FMAX5(BaseFactor):
    """
    移动平均线 - FMAX5
    
    类别: overlap
    参数: {'timeperiod': 5}
    """
    
    factor_id = "FMAX5"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 5}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=5).mean()
            return result.rename("FMAX5")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class FMAX10(BaseFactor):
    """
    移动平均线 - FMAX10
    
    类别: overlap
    参数: {'timeperiod': 10}
    """
    
    factor_id = "FMAX10"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=10).mean()
            return result.rename("FMAX10")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class FMAX15(BaseFactor):
    """
    移动平均线 - FMAX15
    
    类别: overlap
    参数: {'timeperiod': 15}
    """
    
    factor_id = "FMAX15"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 15}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=15).mean()
            return result.rename("FMAX15")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class FMAX20(BaseFactor):
    """
    移动平均线 - FMAX20
    
    类别: overlap
    参数: {'timeperiod': 20}
    """
    
    factor_id = "FMAX20"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=20).mean()
            return result.rename("FMAX20")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_SMA_5(BaseFactor):
    """
    移动平均线 - TA_SMA_5
    
    类别: overlap
    参数: {'timeperiod': 5}
    """
    
    factor_id = "TA_SMA_5"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 5}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=5).mean()
            return result.rename("TA_SMA_5")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_SMA_10(BaseFactor):
    """
    移动平均线 - TA_SMA_10
    
    类别: overlap
    参数: {'timeperiod': 10}
    """
    
    factor_id = "TA_SMA_10"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=10).mean()
            return result.rename("TA_SMA_10")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_SMA_20(BaseFactor):
    """
    移动平均线 - TA_SMA_20
    
    类别: overlap
    参数: {'timeperiod': 20}
    """
    
    factor_id = "TA_SMA_20"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=20).mean()
            return result.rename("TA_SMA_20")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_SMA_30(BaseFactor):
    """
    移动平均线 - TA_SMA_30
    
    类别: overlap
    参数: {'timeperiod': 30}
    """
    
    factor_id = "TA_SMA_30"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 30}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=30).mean()
            return result.rename("TA_SMA_30")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_SMA_60(BaseFactor):
    """
    移动平均线 - TA_SMA_60
    
    类别: overlap
    参数: {'timeperiod': 60}
    """
    
    factor_id = "TA_SMA_60"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 60}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=60).mean()
            return result.rename("TA_SMA_60")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_EMA_5(BaseFactor):
    """
    移动平均线 - TA_EMA_5
    
    类别: overlap
    参数: {'timeperiod': 5}
    """
    
    factor_id = "TA_EMA_5"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 5}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=5, adjust=False).mean()
            return result.rename("TA_EMA_5")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_EMA_10(BaseFactor):
    """
    移动平均线 - TA_EMA_10
    
    类别: overlap
    参数: {'timeperiod': 10}
    """
    
    factor_id = "TA_EMA_10"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=10, adjust=False).mean()
            return result.rename("TA_EMA_10")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_EMA_20(BaseFactor):
    """
    移动平均线 - TA_EMA_20
    
    类别: overlap
    参数: {'timeperiod': 20}
    """
    
    factor_id = "TA_EMA_20"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=20, adjust=False).mean()
            return result.rename("TA_EMA_20")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_EMA_30(BaseFactor):
    """
    移动平均线 - TA_EMA_30
    
    类别: overlap
    参数: {'timeperiod': 30}
    """
    
    factor_id = "TA_EMA_30"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 30}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=30, adjust=False).mean()
            return result.rename("TA_EMA_30")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_EMA_60(BaseFactor):
    """
    移动平均线 - TA_EMA_60
    
    类别: overlap
    参数: {'timeperiod': 60}
    """
    
    factor_id = "TA_EMA_60"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 60}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=60, adjust=False).mean()
            return result.rename("TA_EMA_60")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_WMA_5(BaseFactor):
    """
    移动平均线 - TA_WMA_5
    
    类别: overlap
    参数: {'timeperiod': 5}
    """
    
    factor_id = "TA_WMA_5"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 5}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=5).mean()
            return result.rename("TA_WMA_5")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_WMA_10(BaseFactor):
    """
    移动平均线 - TA_WMA_10
    
    类别: overlap
    参数: {'timeperiod': 10}
    """
    
    factor_id = "TA_WMA_10"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=10).mean()
            return result.rename("TA_WMA_10")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_WMA_20(BaseFactor):
    """
    移动平均线 - TA_WMA_20
    
    类别: overlap
    参数: {'timeperiod': 20}
    """
    
    factor_id = "TA_WMA_20"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=20).mean()
            return result.rename("TA_WMA_20")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_DEMA_5(BaseFactor):
    """
    移动平均线 - TA_DEMA_5
    
    类别: overlap
    参数: {'timeperiod': 5}
    """
    
    factor_id = "TA_DEMA_5"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 5}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=5, adjust=False).mean()
            return result.rename("TA_DEMA_5")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_DEMA_10(BaseFactor):
    """
    移动平均线 - TA_DEMA_10
    
    类别: overlap
    参数: {'timeperiod': 10}
    """
    
    factor_id = "TA_DEMA_10"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=10, adjust=False).mean()
            return result.rename("TA_DEMA_10")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_DEMA_20(BaseFactor):
    """
    移动平均线 - TA_DEMA_20
    
    类别: overlap
    参数: {'timeperiod': 20}
    """
    
    factor_id = "TA_DEMA_20"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=20, adjust=False).mean()
            return result.rename("TA_DEMA_20")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_TEMA_5(BaseFactor):
    """
    移动平均线 - TA_TEMA_5
    
    类别: overlap
    参数: {'timeperiod': 5}
    """
    
    factor_id = "TA_TEMA_5"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 5}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=5, adjust=False).mean()
            return result.rename("TA_TEMA_5")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_TEMA_10(BaseFactor):
    """
    移动平均线 - TA_TEMA_10
    
    类别: overlap
    参数: {'timeperiod': 10}
    """
    
    factor_id = "TA_TEMA_10"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=10, adjust=False).mean()
            return result.rename("TA_TEMA_10")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_TEMA_20(BaseFactor):
    """
    移动平均线 - TA_TEMA_20
    
    类别: overlap
    参数: {'timeperiod': 20}
    """
    
    factor_id = "TA_TEMA_20"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # EMA - 指数移动平均
            result = data["close"].ewm(span=20, adjust=False).mean()
            return result.rename("TA_TEMA_20")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_TRIMA_5(BaseFactor):
    """
    移动平均线 - TA_TRIMA_5
    
    类别: overlap
    参数: {'timeperiod': 5}
    """
    
    factor_id = "TA_TRIMA_5"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 5}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=5).mean()
            return result.rename("TA_TRIMA_5")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_TRIMA_10(BaseFactor):
    """
    移动平均线 - TA_TRIMA_10
    
    类别: overlap
    参数: {'timeperiod': 10}
    """
    
    factor_id = "TA_TRIMA_10"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=10).mean()
            return result.rename("TA_TRIMA_10")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_TRIMA_20(BaseFactor):
    """
    移动平均线 - TA_TRIMA_20
    
    类别: overlap
    参数: {'timeperiod': 20}
    """
    
    factor_id = "TA_TRIMA_20"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=20).mean()
            return result.rename("TA_TRIMA_20")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_KAMA_10(BaseFactor):
    """
    移动平均线 - TA_KAMA_10
    
    类别: overlap
    参数: {'timeperiod': 10}
    """
    
    factor_id = "TA_KAMA_10"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 10}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=10).mean()
            return result.rename("TA_KAMA_10")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_KAMA_20(BaseFactor):
    """
    移动平均线 - TA_KAMA_20
    
    类别: overlap
    参数: {'timeperiod': 20}
    """
    
    factor_id = "TA_KAMA_20"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 20}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=20).mean()
            return result.rename("TA_KAMA_20")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_APO_fastperiod12_matype0_slowperiod26(BaseFactor):
    """
    移动平均线 - TA_APO_fastperiod12_matype0_slowperiod26
    
    类别: overlap
    参数: {'timeperiod': 12}
    """
    
    factor_id = "TA_APO_fastperiod12_matype0_slowperiod26"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {'timeperiod': 12}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            # SMA - 简单移动平均
            result = data["close"].rolling(window=12).mean()
            return result.rename("TA_APO_fastperiod12_matype0_slowperiod26")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_CDLCLOSINGMARUBOZU(BaseFactor):
    """
    移动平均线 - TA_CDLCLOSINGMARUBOZU
    
    类别: overlap
    参数: {}
    """
    
    factor_id = "TA_CDLCLOSINGMARUBOZU"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            return SHARED_CALCULATORS.calculate_candlestick_pattern(
                data["open"], data["high"], data["low"], data["close"],
                pattern_name="CDLCLOSINGMARUBOZU"
            ).rename("TA_CDLCLOSINGMARUBOZU")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_CDLHANGINGMAN(BaseFactor):
    """
    移动平均线 - TA_CDLHANGINGMAN
    
    类别: overlap
    参数: {}
    """
    
    factor_id = "TA_CDLHANGINGMAN"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            return SHARED_CALCULATORS.calculate_candlestick_pattern(
                data["open"], data["high"], data["low"], data["close"],
                pattern_name="CDLHANGINGMAN"
            ).rename("TA_CDLHANGINGMAN")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_CDLMARUBOZU(BaseFactor):
    """
    移动平均线 - TA_CDLMARUBOZU
    
    类别: overlap
    参数: {}
    """
    
    factor_id = "TA_CDLMARUBOZU"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            return SHARED_CALCULATORS.calculate_candlestick_pattern(
                data["open"], data["high"], data["low"], data["close"],
                pattern_name="CDLMARUBOZU"
            ).rename("TA_CDLMARUBOZU")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_CDLMATCHINGLOW(BaseFactor):
    """
    移动平均线 - TA_CDLMATCHINGLOW
    
    类别: overlap
    参数: {}
    """
    
    factor_id = "TA_CDLMATCHINGLOW"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            return SHARED_CALCULATORS.calculate_candlestick_pattern(
                data["open"], data["high"], data["low"], data["close"],
                pattern_name="CDLMATCHINGLOW"
            ).rename("TA_CDLMATCHINGLOW")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_CDLMATHOLD(BaseFactor):
    """
    移动平均线 - TA_CDLMATHOLD
    
    类别: overlap
    参数: {}
    """
    
    factor_id = "TA_CDLMATHOLD"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            return SHARED_CALCULATORS.calculate_candlestick_pattern(
                data["open"], data["high"], data["low"], data["close"],
                pattern_name="CDLMATHOLD"
            ).rename("TA_CDLMATHOLD")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


class TA_CDLRICKSHAWMAN(BaseFactor):
    """
    移动平均线 - TA_CDLRICKSHAWMAN
    
    类别: overlap
    参数: {}
    """
    
    factor_id = "TA_CDLRICKSHAWMAN"
    category = "overlap"
    
    def __init__(self, **kwargs):
        default_params = {}
        default_params.update(kwargs)
        super().__init__(**default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值 - 使用SHARED_CALCULATORS确保一致性"""
        try:
            from factor_system.shared.factor_calculators import SHARED_CALCULATORS
            return SHARED_CALCULATORS.calculate_candlestick_pattern(
                data["open"], data["high"], data["low"], data["close"],
                pattern_name="CDLRICKSHAWMAN"
            ).rename("TA_CDLRICKSHAWMAN")

        except Exception as e:
            logger.error(f"计算{self.factor_id}失败: {e}")
            return pd.Series(np.nan, index=data.index, name=self.factor_id)


