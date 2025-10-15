#!/usr/bin/env python3
"""生成ETF扩展因子集配置（基于已知ETF兼容因子）"""

import logging
from pathlib import Path
from typing import Dict

import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_etf_extended_factors():
    """生成ETF扩展因子集（生产稳健档30-60个因子）"""

    # 基于因子ID模式手动选择ETF兼容因子
    # 排除资金流相关因子（包含money_flow等关键词）
    all_technical_factors = [
        # 动量类 (Momentum)
        "Momentum1",
        "Momentum3",
        "Momentum5",
        "Momentum8",
        "Momentum10",
        "Momentum12",
        "Momentum15",
        "Momentum20",
        "TA_MOM_10",
        "TA_ROC_10",
        "TA_ROCP_10",
        "TA_ROCR_10",
        "TA_ROCR100_10",
        # 均线类 (Moving Averages)
        "MA3",
        "MA5",
        "MA8",
        "MA10",
        "MA15",
        "MA20",
        "EMA3",
        "EMA5",
        "EMA8",
        "EMA12",
        "EMA15",
        "EMA20",
        "TA_SMA_5",
        "TA_SMA_10",
        "TA_SMA_20",
        "TA_SMA_30",
        "TA_SMA_60",
        "TA_EMA_5",
        "TA_EMA_10",
        "TA_EMA_20",
        "TA_EMA_30",
        "TA_EMA_60",
        "TA_WMA_5",
        "TA_WMA_10",
        "TA_WMA_20",
        "TA_DEMA_5",
        "TA_DEMA_10",
        "TA_DEMA_20",
        "TA_TEMA_5",
        "TA_TEMA_10",
        "TA_TEMA_20",
        "TA_T3_5",
        "TA_T3_10",
        "TA_T3_20",
        "TA_TRIMA_5",
        "TA_TRIMA_10",
        "TA_TRIMA_20",
        # 趋势类 (Trend)
        "Trend5",
        "Trend8",
        "Trend10",
        "Trend12",
        "Trend15",
        "Trend20",
        "Trend25",
        "TRENDLB3",
        "TRENDLB5",
        "TRENDLB8",
        "TRENDLB10",
        "FIXLB3",
        "FIXLB5",
        "FIXLB8",
        "FIXLB10",
        "MEANLB3",
        "MEANLB5",
        "MEANLB8",
        "MEANLB10",
        "LEXLB3",
        "LEXLB5",
        "LEXLB8",
        "LEXLB10",
        "TA_ADX_14",
        "TA_ADXR_14",
        "TA_AROON_14_up",
        "TA_AROON_14_down",
        "TA_AROONOSC_14",
        "TA_SAR",
        "TA_TRIX_14",
        # MACD类
        "MACD",
        "MACD_SIGNAL",
        "MACD_HIST",
        "MACD_12_26_9",
        "MACD_6_13_4",
        "MACD_8_17_5",
        # 摆荡指标类 (Oscillators)
        "RSI7",
        "RSI10",
        "RSI14",
        "TA_RSI_14",
        "STOCH",
        "STOCH_7_10",
        "STOCH_10_14",
        "STOCH_14_20",
        "TA_STOCH_K",
        "TA_STOCH_D",
        "TA_STOCHF_K",
        "TA_STOCHF_D",
        "TA_STOCHRSI_fastk_period5_timeperiod14_K",
        "TA_STOCHRSI_fastd_period3_fastk_period5_timeperiod14_D",
        "WILLR9",
        "WILLR14",
        "WILLR18",
        "WILLR21",
        "TA_WILLR_14",
        "CCI10",
        "CCI14",
        "CCI20",
        "TA_CCI_14",
        "STX",
        "STCX",
        "RPROB",
        "RPROBX",
        "RPROBCX",
        "RPROBNX",
        "RANDX",
        "RANDNX",
        "RAND",
        # 波动率类 (Volatility)
        "ATR7",
        "ATR10",
        "ATR14",
        "TA_MFI_14",
        "TA_ULTOSC_timeperiod17_timeperiod214_timeperiod328",
        # 布林带类 (Bollinger Bands)
        "BB_10_2.0_Lower",
        "BB_10_2.0_Middle",
        "BB_10_2.0_Upper",
        "BB_10_2.0_Width",
        "BB_15_2.0_Lower",
        "BB_15_2.0_Middle",
        "BB_15_2.0_Upper",
        "BB_15_2.0_Width",
        "BB_20_2.0_Lower",
        "BB_20_2.0_Middle",
        "BB_20_2.0_Upper",
        "BB_20_2.0_Width",
        "BOLB_20",
        # 成交量类 (Volume)
        "OBV",
        "OBV_SMA5",
        "OBV_SMA10",
        "OBV_SMA15",
        "OBV_SMA20",
        "Volume_Momentum10",
        "Volume_Momentum15",
        "Volume_Momentum20",
        "Volume_Momentum25",
        "Volume_Momentum30",
        "Volume_Ratio10",
        "Volume_Ratio15",
        "Volume_Ratio20",
        "Volume_Ratio25",
        "Volume_Ratio30",
        "VWAP10",
        "VWAP15",
        "VWAP20",
        "VWAP25",
        "VWAP30",
        # 统计类
        "FMEAN5",
        "FMEAN10",
        "FMEAN15",
        "FMEAN20",
        "FSTD5",
        "FSTD10",
        "FSTD15",
        "FSTD20",
        "FMIN5",
        "FMIN10",
        "FMIN15",
        "FMIN20",
        "FMAX5",
        "FMAX10",
        "FMAX15",
        "FMAX20",
        # 位置类
        "Position5",
        "Position8",
        "Position10",
        "Position12",
        "Position15",
        "Position20",
        "Position25",
        "Position30",
    ]

    # 移除重复项并排序
    unique_factors = sorted(list(set(all_technical_factors)))

    logger.info(f"技术因子候选池: {len(unique_factors)} 个")

    # 生产稳健选择：从每个类别选择代表性因子
    production_factors = []

    # 动量类（选择不同周期）
    momentum_factors = [
        f
        for f in unique_factors
        if any(kw in f.lower() for kw in ["momentum", "mom_", "roc", "rocp"])
    ]
    production_factors.extend(momentum_factors[:8])  # 选择前8个

    # 均线类（选择不同类型和周期）
    ma_factors = [
        f
        for f in unique_factors
        if any(kw in f.lower() for kw in ["ma", "ema", "sma"])
        and not f.startswith("TA_")
    ]
    production_factors.extend(ma_factors[:10])  # 选择前10个

    # TA均线类（选择代表性）
    ta_ma_factors = [
        f
        for f in unique_factors
        if f.startswith("TA_") and any(kw in f for kw in ["SMA", "EMA", "WMA"])
    ]
    production_factors.extend(ta_ma_factors[:8])  # 选择前8个

    # 趋势类
    trend_factors = [
        f for f in unique_factors if any(kw in f.lower() for kw in ["trend", "lb"])
    ]
    production_factors.extend(trend_factors[:6])  # 选择前6个

    # MACD类
    macd_factors = [f for f in unique_factors if "macd" in f.lower()]
    production_factors.extend(macd_factors[:4])  # 选择前4个

    # RSI类
    rsi_factors = [f for f in unique_factors if "rsi" in f.lower()]
    production_factors.extend(rsi_factors[:3])  # 选择前3个

    # 随机指标类
    stoch_factors = [
        f for f in unique_factors if any(kw in f.lower() for kw in ["stoch", "willr"])
    ]
    production_factors.extend(stoch_factors[:6])  # 选择前6个

    # 布林带类
    bb_factors = [f for f in unique_factors if "bb_" in f.lower()]
    production_factors.extend(bb_factors[:8])  # 选择前8个

    # 成交量类
    volume_factors = [
        f
        for f in unique_factors
        if any(kw in f.lower() for kw in ["obv", "volume_", "vwap"])
    ]
    production_factors.extend(volume_factors[:8])  # 选择前8个

    # 波动率类
    volatility_factors = [
        f for f in unique_factors if any(kw in f.lower() for kw in ["atr"])
    ]
    production_factors.extend(volatility_factors[:3])  # 选择前3个

    # 去重
    production_factors = sorted(list(set(production_factors)))

    logger.info(f"选择生产因子: {len(production_factors)} 个")

    return production_factors


def generate_factor_set_config():
    """生成因子集配置"""
    production_factors = generate_etf_extended_factors()

    # 分类统计
    categories = {}
    for factor in production_factors:
        if any(kw in factor.lower() for kw in ["momentum", "mom_", "roc", "rocp"]):
            categories.setdefault("momentum", []).append(factor)
        elif any(kw in factor.lower() for kw in ["ma", "ema", "sma"]):
            categories.setdefault("moving_average", []).append(factor)
        elif any(kw in factor.lower() for kw in ["trend", "lb"]):
            categories.setdefault("trend", []).append(factor)
        elif "macd" in factor.lower():
            categories.setdefault("macd", []).append(factor)
        elif "rsi" in factor.lower():
            categories.setdefault("rsi", []).append(factor)
        elif any(kw in factor.lower() for kw in ["stoch", "willr"]):
            categories.setdefault("oscillator", []).append(factor)
        elif "bb_" in factor.lower():
            categories.setdefault("bollinger", []).append(factor)
        elif any(kw in factor.lower() for kw in ["obv", "volume_", "vwap"]):
            categories.setdefault("volume", []).append(factor)
        elif "atr" in factor.lower():
            categories.setdefault("volatility", []).append(factor)
        else:
            categories.setdefault("other", []).append(factor)

    config = {
        "name": "etf_price_extended",
        "description": "ETF扩展价量因子集 - 生产稳健档",
        "total_factors": len(production_factors),
        "categories": {
            cat: {"count": len(factors), "factors": factors}
            for cat, factors in categories.items()
        },
        "production_factors": production_factors,
        "metadata": {
            "created_at": "2025-10-14",
            "data_requirements": "日线OHLCV",
            "safety": "T+1安全",
            "coverage_target": "横截面覆盖率≥80%",
            "usage": "ETF轮动策略评分",
        },
    }

    return config


def save_factor_set_yaml(config: Dict, output_path: str):
    """保存因子集配置到YAML文件"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 简化配置格式
    yaml_config = {
        "name": config["name"],
        "description": config["description"],
        "factors": config["production_factors"],
        "metadata": config["metadata"],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"因子集配置已保存: {output_file}")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("生成ETF扩展因子集配置")
    logger.info("=" * 60)

    # 生成配置
    config = generate_factor_set_config()

    # 保存YAML
    output_path = (
        "factor_system/factor_engine/factors/factor_sets/etf_price_extended.yaml"
    )
    save_factor_set_yaml(config, output_path)

    # 输出统计
    logger.info(f"\n因子集统计:")
    logger.info(f"  总因子数: {config['total_factors']}")

    logger.info(f"\n分类统计:")
    for category, data in config["categories"].items():
        logger.info(f"  {category}: {data['count']} 个")

    logger.info(f"\n生产因子列表 ({len(config['production_factors'])} 个):")
    for i, factor in enumerate(config["production_factors"], 1):
        logger.info(f"  {i:2d}. {factor}")

    logger.info("\n✅ ETF扩展因子集生成完成！")
    logger.info("\n建议下一步:")
    logger.info(
        "1. 测试因子集加载: python -c \"from factor_system.factor_engine.api import load_factor_set; print(load_factor_set('etf_price_extended'))\""
    )
    logger.info("2. 使用该因子集生产ETF面板")
    logger.info("3. 修改评分器支持配置驱动")


if __name__ == "__main__":
    main()
