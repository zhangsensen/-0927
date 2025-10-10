#!/usr/bin/env python3
"""
增强版因子生成器 - 从1min数据重采样到10个时间框架并生成带时间戳的因子文件
"""

import glob
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl
import talib
import yaml

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("enhanced_factor_generation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class EnhancedFactorGenerator:
    """增强版因子生成器 - 支持1min数据重采样到10个时间框架"""

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = self.config.get("data", {}).get(
            "root_dir", "/Users/zhangshenshen/深度量化0927/raw"
        )
        self.output_dir = self.config.get("output", {}).get(
            "directory", "/Users/zhangshenshen/深度量化0927/factor_system/factor_output"
        )

        # 定义10个时间框架
        self.timeframes = [
            "1min",
            "2min",
            "3min",
            "5min",
            "15min",
            "30min",
            "60min",
            "2h",
            "4h",
            "daily",
        ]

        # 时间框架映射
        self.timeframe_mapping = {
            "1min": "1T",
            "2min": "2T",
            "3min": "3T",
            "5min": "5T",
            "15min": "15T",
            "30min": "30T",
            "60min": "1H",
            "2h": "2H",
            "4h": "4H",
            "daily": "1D",
        }

        logger.info(f"Enhanced Factor Generator 初始化完成")
        logger.info(f"批次时间戳: {self.batch_timestamp}")
        logger.info(f"数据目录: {self.data_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"支持时间框架: {len(self.timeframes)}个")

    def load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}

    def scan_1min_data(self) -> List[str]:
        """扫描所有1min数据文件"""
        pattern = os.path.join(self.data_dir, "*_1min_*.parquet")
        files = glob.glob(pattern)

        symbols = []
        for file in files:
            filename = os.path.basename(file)
            # 提取symbol: 0005HK_1min_2025-03-06_2025-09-02.parquet -> 0005HK
            parts = filename.split("_")
            if len(parts) >= 2:
                symbol = parts[0]
                symbols.append(symbol)

        symbols = sorted(list(set(symbols)))
        logger.info(f"发现 {len(symbols)} 个股票的1min数据")
        return symbols

    def load_1min_data(self, symbol: str) -> pd.DataFrame:
        """加载1分钟数据"""
        pattern = os.path.join(self.data_dir, f"{symbol}_1min_*.parquet")
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(f"未找到 {symbol} 的1min数据文件")

        file = files[0]  # 取第一个文件
        logger.info(f"加载1min数据: {file}")

        df = pd.read_parquet(file)

        # 确保数据格式正确
        if "datetime" not in df.columns:
            if "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"])
            else:
                df.index = pd.to_datetime(df.index)
                df = df.reset_index()
                df["datetime"] = pd.to_datetime(df["datetime"])

        # 确保数据格式正确并转换为float64
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")
            # 转换为float64类型以满足TALIB要求
            if col == "volume":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        # 按时间排序
        df = df.sort_values("datetime").reset_index(drop=True)

        logger.info(
            f"加载 {symbol} 1min数据: {len(df)} 行, 时间范围: {df['datetime'].min()} - {df['datetime'].max()}"
        )
        return df

    def resample_data(self, df_1min: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """重采样数据到指定时间框架"""
        if timeframe == "1min":
            return df_1min.copy()

        rule = self.timeframe_mapping[timeframe]

        # 设置datetime为索引
        df = df_1min.set_index("datetime").copy()

        # OHLCV重采样
        resampled = (
            df.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        # 重置索引
        resampled = resampled.reset_index()

        logger.info(f"重采样到 {timeframe}: {len(resampled)} 行")
        return resampled

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()

        # 基础价格列
        open_price = df["open"].values
        high_price = df["high"].values
        low_price = df["low"].values
        close_price = df["close"].values
        volume = df["volume"].values

        try:
            # 移动平均线
            df["MA5"] = talib.SMA(close_price, timeperiod=5)
            df["MA10"] = talib.SMA(close_price, timeperiod=10)
            df["MA20"] = talib.SMA(close_price, timeperiod=20)
            df["MA60"] = talib.SMA(close_price, timeperiod=60)

            # EMA
            df["EMA5"] = talib.EMA(close_price, timeperiod=5)
            df["EMA12"] = talib.EMA(close_price, timeperiod=12)
            df["EMA26"] = talib.EMA(close_price, timeperiod=26)

            # MACD
            df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = talib.MACD(
                close_price, fastperiod=12, slowperiod=26, signalperiod=9
            )

            # RSI
            df["RSI"] = talib.RSI(close_price, timeperiod=14)

            # Stochastic
            df["STOCH_K"], df["STOCH_D"] = talib.STOCH(
                high_price,
                low_price,
                close_price,
                fastk_period=9,
                slowk_period=3,
                slowd_period=3,
            )

            # Bollinger Bands
            df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = talib.BBANDS(
                close_price, timeperiod=20, nbdevup=2, nbdevdn=2
            )

            # ATR
            df["ATR"] = talib.ATR(high_price, low_price, close_price, timeperiod=14)

            # Williams %R
            df["WILLR"] = talib.WILLR(high_price, low_price, close_price, timeperiod=14)

            # CCI
            df["CCI"] = talib.CCI(high_price, low_price, close_price, timeperiod=14)

            # MFI
            df["MFI"] = talib.MFI(
                high_price, low_price, close_price, volume, timeperiod=14
            )

            # ADX
            df["ADX"] = talib.ADX(high_price, low_price, close_price, timeperiod=14)

            # OBV
            df["OBV"] = talib.OBV(close_price, volume)

            # 价格变化率
            df["ROC"] = talib.ROC(close_price, timeperiod=10)

            # 动量
            df["MOM"] = talib.MOM(close_price, timeperiod=10)

            # 标准差
            df["STDDEV"] = talib.STDDEV(close_price, timeperiod=20, nbdev=1)

            logger.info(f"成功计算技术指标: {len(df.columns)} 个字段")

        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            raise

        return df

    def save_factors(
        self, df_factors: pd.DataFrame, symbol: str, timeframe: str, market: str = "HK"
    ):
        """保存因子数据"""
        # 创建输出目录
        market_dir = os.path.join(self.output_dir, market)
        os.makedirs(market_dir, exist_ok=True)

        # 构建文件名 - 带批次时间戳
        filename = f"{symbol}_{timeframe}_factors_{self.batch_timestamp}.parquet"
        filepath = os.path.join(market_dir, filename)

        # 转换为polars并保存
        df_pl = pl.from_pandas(df_factors)
        df_pl.write_parquet(filepath, compression="snappy")

        logger.info(f"保存因子文件: {filename} ({len(df_factors)} 行)")
        return filepath

    def process_symbol(self, symbol: str, market: str = "HK") -> bool:
        """处理单个股票的所有时间框架"""
        try:
            logger.info(f"开始处理股票: {symbol}")

            # 加载1min数据
            df_1min = self.load_1min_data(symbol)

            success_count = 0
            for timeframe in self.timeframes:
                try:
                    logger.info(f"处理 {symbol} {timeframe}")

                    # 重采样
                    df_resampled = self.resample_data(df_1min, timeframe)

                    if len(df_resampled) < 50:  # 数据太少跳过
                        logger.warning(
                            f"{symbol} {timeframe} 数据不足: {len(df_resampled)} 行"
                        )
                        continue

                    # 计算技术指标
                    df_factors = self.calculate_technical_indicators(df_resampled)

                    # 保存
                    self.save_factors(df_factors, symbol, timeframe, market)
                    success_count += 1

                except Exception as e:
                    logger.error(f"处理 {symbol} {timeframe} 失败: {e}")
                    continue

            logger.info(
                f"✅ {symbol} 完成: {success_count}/{len(self.timeframes)} 个时间框架"
            )
            return True

        except Exception as e:
            logger.error(f"❌ 处理股票 {symbol} 失败: {e}")
            return False

    def process_all(self, market: str = "HK"):
        """处理所有股票"""
        logger.info(f"开始处理 {market} 市场所有股票")

        # 扫描股票
        symbols = self.scan_1min_data()
        if not symbols:
            logger.error("未找到任何1min数据文件")
            return

        logger.info(f"找到 {len(symbols)} 只股票，开始处理...")

        success_count = 0
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"进度: {i}/{len(symbols)} - {symbol}")

            if self.process_symbol(symbol, market):
                success_count += 1

        # 生成报告
        logger.info("=" * 60)
        logger.info("增强版因子生成完成报告")
        logger.info("=" * 60)
        logger.info(f"处理市场: {market}")
        logger.info(f"总股票数: {len(symbols)}")
        logger.info(f"成功处理: {success_count}")
        logger.info(f"处理失败: {len(symbols) - success_count}")
        logger.info(f"成功率: {success_count/len(symbols)*100:.1f}%")
        logger.info(f"时间框架数: {len(self.timeframes)}")
        logger.info(f"批次时间戳: {self.batch_timestamp}")
        logger.info("=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="增强版因子生成器")
    parser.add_argument(
        "--config",
        default="factor_system/factor_generation/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument("--market", default="HK", help="市场代码")

    args = parser.parse_args()

    # 创建生成器
    generator = EnhancedFactorGenerator(args.config)

    # 处理指定市场
    generator.process_all(args.market)


if __name__ == "__main__":
    main()
