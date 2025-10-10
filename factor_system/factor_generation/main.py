#!/usr/bin/env python3
"""
因子生成主程序 - 严格的配置管理
不提供任何默认值或回退逻辑
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .config import setup_logging
from .config_loader import ConfigLoader
from .enhanced_factor_calculator import EnhancedFactorCalculator

logger = logging.getLogger(__name__)


class FactorGenerationApp:
    """因子生成应用程序"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化应用

        Args:
            config_path: 配置文件路径，如果为None则使用完整配置
        """
        self.config = self._load_config(config_path)
        self.calculator = EnhancedFactorCalculator(self.config)

    def _load_config(self, config_path: Optional[str]) -> "IndicatorConfig":
        """加载配置"""
        try:
            if config_path:
                logger.info(f"从配置文件加载配置: {config_path}")
                config = ConfigLoader.load_config(config_path)
            else:
                logger.info("使用完整功能配置")
                config = ConfigLoader.create_full_config()

            # 验证配置
            ConfigLoader.validate_config(config)
            return config

        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise

    def generate_factors(
        self, symbol: str, timeframes: List[str], data_path: str
    ) -> None:
        """
        生成因子数据

        Args:
            symbol: 股票代码
            timeframes: 时间框架列表
            data_path: 数据文件路径
        """
        # 加载数据
        try:
            if data_path.endswith(".parquet"):
                df = pd.read_parquet(data_path)
            elif data_path.endswith(".csv"):
                df = pd.read_csv(data_path)
            else:
                raise ValueError(f"不支持的数据格式: {data_path}")

            logger.info(f"加载数据成功: {df.shape}")

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

        # 生成各时间框架的因子
        for timeframe in timeframes:
            try:
                logger.info(f"开始生成 {symbol} {timeframe} 时间框架的因子...")

                # 根据时间框架重采样数据
                resampled_data = self._resample_data(df, timeframe)

                if resampled_data is None or resampled_data.empty:
                    logger.warning(f"时间框架 {timeframe} 重采样后为空，跳过")
                    continue

                # 计算因子
                from .enhanced_factor_calculator import TimeFrame

                # 转换时间框架格式 - 真实的映射关系
                timeframe_mapping = {
                    "1min": "5min",  # 最小支持5min
                    "2min": "5min",
                    "2m": "5min",
                    "3min": "5min",
                    "3m": "5min",
                    "5min": "5min",
                    "5m": "5min",
                    "10min": "15min",  # 10min映射到15min计算
                    "10m": "15min",
                    "15m": "15min",
                    "15min": "15min",
                    "30m": "30min",
                    "30min": "30min",
                    "60m": "60min",
                    "60min": "60min",
                    "1h": "60min",
                    "2h": "daily",  # 2h以上映射到日线
                    "4h": "daily",
                    "daily": "daily",
                    "1day": "daily",
                }

                mapped_timeframe = timeframe_mapping.get(timeframe, "5min")
                tf_enum = TimeFrame(mapped_timeframe)
                factors = self.calculator.calculate_comprehensive_factors(
                    resampled_data, tf_enum
                )

                if factors is not None:
                    # 保存结果
                    self._save_factors(factors, symbol, timeframe)
                    logger.info(
                        f"✅ {symbol} {timeframe} 因子生成完成: {factors.shape}"
                    )
                else:
                    logger.error(f"❌ {symbol} {timeframe} 因子计算失败")

            except Exception as e:
                logger.error(f"❌ {symbol} {timeframe} 因子生成失败: {e}")
                continue

    def _resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """重采样数据到指定时间框架"""
        try:
            # 确保数据包含datetime列（支持timestamp或datetime列名）
            if "datetime" not in df.columns and "timestamp" not in df.columns:
                logger.error("数据中缺少datetime或timestamp列")
                return None

            # 处理timestamp或datetime列
            if "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"])
            else:
                df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")

            # 根据时间框架重采样 - 真实的时间框架映射
            resample_map = {
                "1min": "1min",
                "2min": "2min",
                "2m": "2min",
                "3min": "3min",
                "3m": "3min",
                "5min": "5min",
                "5m": "5min",
                "10min": "10min",
                "10m": "10min",
                "15min": "15min",
                "15m": "15min",
                "30min": "30min",
                "30m": "30min",
                "60min": "60min",
                "60m": "60min",
                "1h": "60min",
                "2h": "120min",
                "4h": "240min",
                "daily": "1D",
                "1day": "1D",
            }

            resample_freq = resample_map.get(timeframe)
            if not resample_freq:
                logger.error(f"不支持的时间框架: {timeframe}")
                return None

            # 重采样
            resampled = (
                df.resample(resample_freq)
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

            logger.info(f"重采样 {timeframe}: {len(df)} -> {len(resampled)} 行")
            return resampled

        except Exception as e:
            logger.error(f"重采样失败: {e}")
            return None

    def _save_factors(self, factors: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """保存因子数据"""
        try:
            # 从配置中获取输出路径
            import yaml

            config_path = Path(__file__).parent / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            output_dir = Path(
                config_data.get("output", {}).get("directory", "./factor_output")
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件名
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timeframe}_factors_{timestamp}.parquet"
            filepath = output_dir / filename

            # 保存
            factors.to_parquet(filepath)
            logger.info(f"因子数据已保存: {filepath}")

        except Exception as e:
            logger.error(f"保存因子数据失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="因子生成器 - 严格配置管理")
    parser.add_argument("--symbol", required=True, help="股票代码 (例如: 0700.HK)")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        required=True,
        help="时间框架 (例如: 1min 5min 15min 60min daily)",
    )
    parser.add_argument("--data-path", required=True, help="数据文件路径")
    parser.add_argument("--config", help="配置文件路径 (可选，默认使用完整配置)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging()
    logger.setLevel(args.log_level)

    try:
        # 创建应用实例
        app = FactorGenerationApp(args.config)

        # 生成因子
        app.generate_factors(args.symbol, args.timeframes, args.data_path)

        logger.info("因子生成任务完成")

    except Exception as e:
        logger.error(f"因子生成失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
