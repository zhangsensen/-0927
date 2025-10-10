#!/usr/bin/env python3
"""
生成FactorEngine中应该保留的有效因子清单
基于factor_generation中实际存在的因子
"""


def get_valid_factors():
    """获取factor_generation中实际存在的有效因子清单"""

    # 基于enhanced_factor_calculator.py分析的实际因子清单
    valid_factors = {
        # 移动平均线系列
        "MA",
        "MA3",
        "MA5",
        "MA8",
        "MA10",
        "MA12",
        "MA15",
        "MA20",
        "MA25",
        "MA30",
        "MA40",
        "MA50",
        "MA60",
        "MA80",
        "MA100",
        "MA120",
        "MA150",
        "MA200",
        "EMA",
        "EMA3",
        "EMA5",
        "EMA8",
        "EMA12",
        "EMA15",
        "EMA20",
        "EMA26",
        "EMA30",
        "EMA40",
        "EMA50",
        "EMA60",
        # MACD指标系列
        "MACD",
        "MACD_SIGNAL",
        "MACD_HIST",
        # RSI指标系列
        "RSI",
        "RSI3",
        "RSI6",
        "RSI9",
        "RSI12",
        "RSI14",
        "RSI18",
        "RSI21",
        "RSI25",
        "RSI30",
        # 布林带系列
        "BBANDS",
        "BB_UPPER",
        "BB_MIDDLE",
        "BB_LOWER",
        "BB_WIDTH",
        # 随机指标系列
        "STOCH",
        "STOCH_K",
        "STOCH_D",
        # ATR指标系列
        "ATR",
        "ATR7",
        "ATR14",
        "ATR21",
        "ATR28",
        # 波动率指标系列
        "MSTD",
        "MSTD5",
        "MSTD10",
        "MSTD15",
        "MSTD20",
        "MSTD25",
        "MSTD30",
        # 成交量指标系列
        "OBV",
        "OBV_SMA5",
        "OBV_SMA10",
        "OBV_SMA15",
        "OBV_SMA20",
        "Volume_ratio10",
        "volume_ratio15",
        "volume_ratio20",
        "volume_ratio25",
        "volume_ratio30",
        "volume_momentum10",
        "volume_momentum15",
        "volume_momentum20",
        "volume_momentum25",
        "volume_momentum30",
        "VWAP10",
        "VWAP15",
        "VWAP20",
        "VWAP25",
        "VWAP30",
        # 威廉指标系列
        "WILLR",
        "WILLR9",
        "WILLR14",
        "WILLR18",
        "WILLR21",
        # 商品通道指标系列
        "CCI",
        "CCI10",
        "CCI14",
        "CCI20",
        # 动量指标系列
        "MOMENTUM1",
        "MOMENTUM3",
        "MOMENTUM5",
        "MOMENTUM8",
        "MOMENTUM10",
        "MOMENTUM12",
        "MOMENTUM15",
        "MOMENTUM20",
        # 位置指标系列
        "POSITION5",
        "POSITION8",
        "POSITION10",
        "POSITION12",
        "POSITION15",
        "POSITION20",
        "POSITION25",
        "POSITION30",
        # 趋势强度指标系列
        "TREND5",
        "TREND8",
        "TREND10",
        "TREND12",
        "TREND15",
        "TREND20",
        "TREND25",
        # VectorBT特殊指标
        "BOLB_20",
        # TA-Lib核心指标（实际在factor_generation中启用的）
        "SMA",
        "WMA",
        "DEMA",
        "TEMA",
        "TRIMA",
        "KAMA",
        "MAMA",
        "T3",
        "MIDPOINT",
        "MIDPRICE",
        "SAR",
        "SAREXT",
        "ADX",
        "ADXR",
        "APO",
        "AROON",
        "AROONOSC",
        "CCI",
        "CMO",
        "DX",
        "MFI",
        "MOM",
        "NATR",
        "OBV",
        "PLUS_DI",
        "PLUS_DM",
        "MINUS_DI",
        "MINUS_DM",
        "PPO",
        "ROC",
        "ROCP",
        "ROCR",
        "ROCR100",
        "STOCHF",
        "STOCHRSI",
        "TRANGE",
        "TRIX",
        "ULTOSC",
        "WILLR",
    }

    return valid_factors


def get_factor_mappings():
    """获取因子映射关系"""
    mappings = {
        # MACD系列映射
        "MACD": ["MACD_12_26_9_MACD"],
        "MACD_SIGNAL": ["MACD_12_26_9_Signal"],
        "MACD_HIST": ["MACD_12_26_9_Hist"],
        # 布林带系列映射
        "BBANDS": ["BB_20_2_0_Middle"],
        "BB_UPPER": ["BB_20_2_0_Upper"],
        "BB_MIDDLE": ["BB_20_2_0_Middle"],
        "BB_LOWER": ["BB_20_2_0_Lower"],
        "BB_WIDTH": ["BB_20_2_0_Width"],
        # 随机指标系列映射
        "STOCH": ["STOCH_14_3_K"],
        "STOCH_K": ["STOCH_14_3_K"],
        "STOCH_D": ["STOCH_14_3_D"],
        # RSI系列映射
        "RSI": ["RSI14"],
        "RSI3": ["RSI3"],
        "RSI6": ["RSI6"],
        "RSI9": ["RSI9"],
        "RSI12": ["RSI12"],
        "RSI14": ["RSI14"],
        "RSI18": ["RSI18"],
        "RSI21": ["RSI21"],
        "RSI25": ["RSI25"],
        "RSI30": ["RSI30"],
        # ATR系列映射
        "ATR": ["ATR14"],
        "ATR7": ["ATR7"],
        "ATR14": ["ATR14"],
        "ATR21": ["ATR21"],
        "ATR28": ["ATR28"],
        # 威廉指标系列映射
        "WILLR": ["WILLR14"],
        "WILLR9": ["WILLR9"],
        "WILLR14": ["WILLR14"],
        "WILLR18": ["WILLR18"],
        "WILLR21": ["WILLR21"],
        # CCI系列映射
        "CCI": ["CCI14"],
        "CCI10": ["CCI10"],
        "CCI14": ["CCI14"],
        "CCI20": ["CCI20"],
        # 移动平均线映射
        "SMA": ["MA20"],
        "EMA": ["EMA12"],
        "DEMA": ["DEMA"],
        "TEMA": ["TEMA"],
        "KAMA": ["KAMA"],
        "MAMA": ["MAMA"],
        "T3": ["T3"],
        "TRIMA": ["TRIMA"],
        "WMA": ["WMA"],
        # 其他指标映射
        "ADX": ["ADX14"],
        "ADXR": ["ADXR14"],
        "AROON": ["AROON14"],
        "AROONOSC": ["AROONOSC14"],
        "MFI": ["MFI14"],
        "NATR": ["NATR14"],
        "OBV": ["OBV"],
        "TRANGE": ["TRANGE"],
        "ULTOSC": ["ULTOSC14"],
        "MIDPOINT": ["MIDPOINT14"],
        "MIDPRICE": ["MIDPRICE14"],
        "SAR": ["SAR"],
        "SAREXT": ["SAREXT"],
    }

    return mappings


def main():
    """主函数"""
    valid_factors = get_valid_factors()
    mappings = get_factor_mappings()

    print("✅ FactorEngine允许保留的因子清单:")
    print(f"📊 总计: {len(valid_factors)} 个因子")

    # 分类显示
    ma_factors = [
        f
        for f in valid_factors
        if f.startswith(
            ("MA", "EMA", "SMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "MAMA", "T3")
        )
    ]
    macd_factors = [f for f in valid_factors if f.startswith("MACD")]
    rsi_factors = [f for f in valid_factors if f.startswith("RSI")]
    bb_factors = [f for f in valid_factors if f.startswith("BB")]
    stoch_factors = [f for f in valid_factors if f.startswith("STOCH")]
    atr_factors = [f for f in valid_factors if f.startswith("ATR")]
    mstd_factors = [f for f in valid_factors if f.startswith("MSTD")]
    volume_factors = [
        f for f in valid_factors if any(x in f for x in ["OBV", "Volume", "VWAP"])
    ]
    willr_factors = [f for f in valid_factors if f.startswith("WILLR")]
    cci_factors = [f for f in valid_factors if f.startswith("CCI")]
    momentum_factors = [f for f in valid_factors if f.startswith("MOMENTUM")]
    position_factors = [f for f in valid_factors if f.startswith("POSITION")]
    trend_factors = [f for f in valid_factors if f.startswith("TREND")]

    # 其他技术指标
    other_factors = []
    all_category_factors = (
        ma_factors
        + macd_factors
        + rsi_factors
        + bb_factors
        + stoch_factors
        + atr_factors
        + mstd_factors
        + volume_factors
        + willr_factors
        + cci_factors
        + momentum_factors
        + position_factors
        + trend_factors
    )
    for f in valid_factors:
        if f not in all_category_factors:
            other_factors.append(f)

    categories = {
        "移动平均线": ma_factors,
        "MACD指标": macd_factors,
        "RSI指标": rsi_factors,
        "布林带": bb_factors,
        "随机指标": stoch_factors,
        "ATR指标": atr_factors,
        "波动率指标": mstd_factors,
        "成交量指标": volume_factors,
        "威廉指标": willr_factors,
        "商品通道": cci_factors,
        "动量指标": momentum_factors,
        "位置指标": position_factors,
        "趋势强度": trend_factors,
        "其他技术指标": other_factors,
    }

    for category, factor_list in categories.items():
        if factor_list:
            print(f"\n📈 {category} ({len(factor_list)}个):")
            for factor in sorted(factor_list):
                print(f"  - {factor}")

    # 保存到文件
    with open(
        "/Users/zhangshenshen/深度量化0927/valid_factor_engine_factors.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("FactorEngine允许保留的因子清单（基于factor_generation一致性要求）\n")
        f.write("=" * 70 + "\n\n")

        for category, factor_list in categories.items():
            if factor_list:
                f.write(f"{category} ({len(factor_list)}个):\n")
                for factor in sorted(factor_list):
                    f.write(f"  {factor}\n")
                f.write("\n")

        f.write(f"总计: {len(valid_factors)} 个因子\n\n")

        f.write("因子映射关系:\n")
        f.write("-" * 30 + "\n")
        for factor, alternatives in mappings.items():
            f.write(f"{factor}: {alternatives}\n")

    print(f"\n✅ 有效因子清单已保存至: valid_factor_engine_factors.txt")

    return valid_factors, mappings


if __name__ == "__main__":
    valid_factors, mappings = main()
