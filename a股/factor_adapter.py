"""
A股因子适配器 - 修复Registry实例化问题

修复要点：
1. 注册因子类而不是实例
2. 使用统一API接口
3. 简化初始化逻辑
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

# 使用统一API接口
from factor_system.factor_engine import api


class AShareFactorAdapter:
    """
    A股因子适配器 - 修复版本

    主要修复：
    - 使用统一API，避免Registry实例化问题
    - 简化因子映射逻辑
    - 增强错误处理
    """

    # 因子名称映射：A股项目 -> factor_engine
    FACTOR_MAPPING = {
        # 移动平均线
        "MA5": "SMA_5",
        "MA10": "SMA_10",
        "MA20": "SMA_20",
        "MA30": "SMA_30",
        "MA60": "SMA_60",
        "EMA5": "EMA_5",
        "EMA12": "EMA_12",
        "EMA26": "EMA_26",
        # 动量指标
        "RSI": "RSI_14_wilders",  # 使用Wilders平滑
        "MACD": "MACD_12_26_9",
        "MACD_Signal": "MACD_Signal_12_26_9",
        "MACD_Hist": "MACD_Hist_12_26_9",
        "KDJ_K": "STOCH_14_K",
        "KDJ_D": "STOCH_14_D",
        "KDJ_J": "STOCH_14_J",
        "Williams_R": "WILLR_14",
        # 波动性指标
        "ATR": "ATR_14",
        "BB_Upper": "BBANDS_Upper_20_2",
        "BB_Middle": "BBANDS_Middle_20_2",
        "BB_Lower": "BBANDS_Lower_20_2",
        # 趋势指标
        "ADX": "ADX_14",
        "DI_plus": "PLUS_DI_14",
        "DI_minus": "MINUS_DI_14",
        # 成交量指标
        "OBV": "OBV",
        "Volume_SMA": "SMA_Volume_20",
        "MFI": "MFI_14",
        # 其他指标
        "CCI": "CCI_14",
        "MOM": "MOM_10",
        "ROC": "ROC_10",
        "TRIX": "TRIX_14",
    }

    def __init__(self, data_dir: str):
        """
        初始化适配器

        Args:
            data_dir: A股数据目录路径
        """
        self.data_dir = data_dir

        print(f"✅ A股因子适配器初始化完成 (修复版本)")
        print(f"   数据目录: {data_dir}")

        # 可用因子列表
        self.available_factors = self._check_available_factors()
        print(f"   可用因子: {len(self.available_factors)}个")

    def _check_available_factors(self) -> List[str]:
        """检查factor_engine中可用的因子"""
        try:
            # 使用统一API获取可用因子
            available = api.list_available_factors()

            # 过滤出我们映射的因子
            mapped_factors = set(self.FACTOR_MAPPING.values())
            available_mapped = [f for f in available if f in mapped_factors]

            return available_mapped

        except Exception as e:
            print(f"⚠️  检查可用因子时出错: {e}")
            return []

    def get_technical_indicators(
        self,
        stock_code: str,
        timeframe: str = "1d",
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """
        获取技术指标DataFrame

        Args:
            stock_code: 股票代码 (e.g. '300450.SZ')
            timeframe: 时间框架
            lookback_days: 回看天数

        Returns:
            DataFrame with technical indicators
        """
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # 获取需要计算的因子列表（去重，只计算可用的）
        factor_ids = list(set(self.FACTOR_MAPPING.values()))
        factor_ids = [f for f in factor_ids if f in self.available_factors]

        if not factor_ids:
            print(f"⚠️  没有可用的因子")
            return pd.DataFrame()

        try:
            # 使用统一API计算因子
            factors_df = api.calculate_factors(
                factor_ids=factor_ids,
                symbols=[stock_code],
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
            )

            if factors_df.empty:
                print(f"⚠️  {stock_code} 未计算到任何因子数据")
                return pd.DataFrame()

            # 重命名列（从factor_engine名称 -> A股项目名称）
            reverse_mapping = {v: k for k, v in self.FACTOR_MAPPING.items()}

            # 只保留映射中存在的列
            available_columns = [
                col for col in factors_df.columns if col in reverse_mapping
            ]
            factors_df = factors_df[available_columns]

            # 重命名
            factors_df = factors_df.rename(columns=reverse_mapping)

            print(
                f"✅ {stock_code} 技术指标计算完成: {len(factors_df)}行 x {len(factors_df.columns)}列"
            )

            return factors_df

        except Exception as e:
            print(f"❌ {stock_code} 技术指标计算失败: {e}")
            import traceback

            traceback.print_exc()
            return pd.DataFrame()

    def add_indicators_to_dataframe(
        self,
        df: pd.DataFrame,
        stock_code: str,
    ) -> pd.DataFrame:
        """
        将技术指标添加到现有DataFrame

        Args:
            df: 原始OHLCV数据
            stock_code: 股票代码

        Returns:
            添加了技术指标的DataFrame
        """
        # 确保df有timestamp列
        if "timestamp" not in df.columns:
            if df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            else:
                raise ValueError("DataFrame必须有timestamp列或索引")

        # 获取技术指标
        indicators = self.get_technical_indicators(
            stock_code=stock_code,
            lookback_days=len(df) + 60,  # 额外60天确保充足数据
        )

        if indicators.empty:
            print(f"⚠️  {stock_code} 未获取到技术指标，返回原数据")
            return df

        # 合并到原DataFrame（按timestamp对齐）
        df_with_indicators = df.merge(
            indicators, left_on="timestamp", right_index=True, how="left"
        )

        print(
            f"✅ {stock_code} 技术指标合并完成: 总列数 {len(df_with_indicators.columns)}"
        )

        return df_with_indicators

    def calculate_single_indicator(
        self,
        stock_code: str,
        indicator_name: str,
        timeframe: str = "1d",
        lookback_days: int = 252,
    ) -> pd.Series:
        """
        计算单个技术指标

        Args:
            stock_code: 股票代码
            indicator_name: 指标名称（A股项目命名）
            timeframe: 时间框架
            lookback_days: 回看天数

        Returns:
            指标序列
        """
        if indicator_name not in self.FACTOR_MAPPING:
            raise ValueError(f"不支持的指标: {indicator_name}")

        factor_id = self.FACTOR_MAPPING[indicator_name]

        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        try:
            result = api.calculate_single_factor(
                factor_id=factor_id,
                symbol=stock_code,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

            return result

        except Exception as e:
            print(f"❌ {stock_code} {indicator_name} 计算失败: {e}")
            return pd.Series()

    def list_available_indicators(self) -> List[str]:
        """
        列出所有可用的技术指标

        Returns:
            指标名称列表（A股项目命名）
        """
        # 返回映射中且可用的指标
        available = []
        for a_share_name, factor_id in self.FACTOR_MAPPING.items():
            if factor_id in self.available_factors:
                available.append(a_share_name)
        return available

    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息

        Returns:
            缓存统计字典
        """
        try:
            return api.get_cache_stats()
        except Exception as e:
            print(f"⚠️  获取缓存统计失败: {e}")
            return {}

    def clear_cache(self):
        """清空缓存"""
        try:
            api.clear_cache()
            print("✅ 缓存已清空")
        except Exception as e:
            print(f"❌ 清空缓存失败: {e}")


# 便捷函数
def create_a_share_adapter(data_dir: str = None) -> AShareFactorAdapter:
    """
    创建A股因子适配器的便捷函数

    Args:
        data_dir: 数据目录，默认使用项目中的A股目录

    Returns:
        A股因子适配器实例
    """
    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent)

    return AShareFactorAdapter(data_dir)


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试修复后的适配器...")

    adapter = create_a_share_adapter()

    # 测试获取技术指标
    stock_code = "300450.SZ"
    indicators = adapter.get_technical_indicators(stock_code)

    if not indicators.empty:
        print(f"\n📊 {stock_code} 技术指标预览:")
        print(indicators.tail())

        print(f"\n📈 可用指标: {adapter.list_available_indicators()}")

        print(f"\n💾 缓存统计: {adapter.get_cache_stats()}")
    else:
        print(f"❌ 未能获取到{stock_code}的技术指标")
