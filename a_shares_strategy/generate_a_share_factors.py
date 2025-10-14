#!/usr/bin/env python3
"""
A股因子生成主入口

基于现有因子引擎，为A股市场生成技术因子。
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from a_shares_strategy.data_pipeline.loader import load_a_share_minute
from factor_system.factor_generation.enhanced_factor_calculator import (
    EnhancedFactorCalculator,
    IndicatorConfig,
    TimeFrame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_TIMEFRAMES = tuple(
    dict.fromkeys(timeframe.value for timeframe in TimeFrame)
)


def load_a_share_config(config_path: str) -> IndicatorConfig:
    """
    加载A股配置文件并转换为IndicatorConfig

    Args:
        config_path: 配置文件路径

    Returns:
        IndicatorConfig对象
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # 构建IndicatorConfig
    return IndicatorConfig(
        enable_ma=True,
        enable_ema=True,
        enable_macd=True,
        enable_rsi=True,
        enable_bbands=True,
        enable_stoch=True,
        enable_atr=True,
        enable_obv=True,
        enable_mstd=True,
        enable_manual_indicators=config_dict.get("enable_manual_indicators", True),
        enable_all_periods=config_dict.get("enable_all_periods", True),
        memory_efficient=False,
        market="A_SHARES",
    )


def generate_factors_for_symbol(
    symbol: str,
    timeframe: str = "5min",
    output_dir: str = "factor_system/factor_output/A_SHARES",
    config_path: str = "factor_system/factor_generation/config/indicator_config_a_shares.yaml",
) -> bool:
    """
    为单个A股标的生成因子

    Args:
        symbol: 股票代码（如 600036.SH）
        timeframe: 时间框架
        output_dir: 输出目录
        config_path: 配置文件路径

    Returns:
        是否成功
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        logger.error("不支持的时间框架: %s", timeframe)
        return False

    try:
        logger.info("=" * 80)
        logger.info("开始处理 %s - %s", symbol, timeframe)
        logger.info("=" * 80)

        # 加载数据
        logger.info("1. 加载原始数据...")
        df = load_a_share_minute(symbol)
        logger.info(
            "   数据形状: %s, 时间范围: %s 至 %s",
            df.shape,
            df.index[0],
            df.index[-1],
        )

        # 重采样到目标时间框架（如果需要）
        if timeframe != "1min":
            logger.info("2. 重采样到 %s...", timeframe)
            df = resample_to_timeframe(df, timeframe)
            logger.info("   重采样后形状: %s", df.shape)

        # 加载配置
        logger.info("3. 加载因子配置...")
        config = load_a_share_config(config_path)

        # 初始化因子计算器
        logger.info("4. 初始化因子计算器...")
        calculator = EnhancedFactorCalculator(config)

        # 计算因子
        logger.info("5. 计算因子...")
        tf_enum = TimeFrame(timeframe)
        factors_df = calculator.calculate_comprehensive_factors(df, tf_enum)

        if factors_df is None or factors_df.empty:
            logger.error(f"   ❌ {symbol} 因子计算失败")
            return False

        logger.info("   ✅ 生成 %d 个因子", len(factors_df.columns))

        # 保存结果
        logger.info("6. 保存因子数据...")
        output_path = Path(output_dir) / timeframe
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"{symbol}_{timeframe}_factors.parquet"
        factors_df.to_parquet(output_file)
        logger.info("   ✅ 保存至: %s", output_file)

        # 输出统计信息
        logger.info("7. 因子统计:")
        logger.info("   - 总因子数: %d", len(factors_df.columns))
        logger.info("   - 数据点数: %d", len(factors_df))
        null_ratio = factors_df.isnull().sum().sum() / factors_df.size
        logger.info("   - 空值比例: %.2f%%", null_ratio * 100)

        logger.info("✅ %s 处理完成", symbol)
        return True

    except Exception as e:
        logger.error(f"❌ {symbol} 处理失败: {e}", exc_info=True)
        return False


def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    重采样到目标时间框架

    Args:
        df: 输入DataFrame（1分钟级）
        timeframe: 目标时间框架

    Returns:
        重采样后的DataFrame
    """
    # 时间框架映射
    freq_map = {
        "1min": "1min",
        "5min": "5min",
        "15min": "15min",
        "30min": "30min",
        "60min": "60min",
        "1day": "1D",
    }

    freq = freq_map.get(timeframe, "5min")

    # OHLCV重采样规则
    resampled = df.resample(freq).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        }
    )

    # 删除全为NaN的行（resample会生成24小时网格，需过滤非交易时间）
    resampled = resampled.dropna(how="all")
    
    # 进一步过滤：只保留有实际数据的行（volume > 0）
    resampled = resampled[resampled["volume"] > 0]

    return resampled


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="A股因子生成工具")
    parser.add_argument("symbol", help="股票代码（如 600036.SH）")
    parser.add_argument(
        "--timeframe",
        "-t",
        default="5min",
        choices=SUPPORTED_TIMEFRAMES,
        help="时间框架（默认: 5min）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="factor_system/factor_output/A_SHARES",
        help="输出目录",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="factor_system/factor_generation/config/indicator_config_a_shares.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--all-timeframes",
        action="store_true",
        help="生成所有支持的时间框架因子",
    )

    args = parser.parse_args()

    timeframes = SUPPORTED_TIMEFRAMES if args.all_timeframes else (args.timeframe,)
    success = True

    for timeframe in timeframes:
        result = generate_factors_for_symbol(
            symbol=args.symbol,
            timeframe=timeframe,
            output_dir=args.output,
            config_path=args.config,
        )
        if not result:
            success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
