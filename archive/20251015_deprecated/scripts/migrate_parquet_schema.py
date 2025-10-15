#!/usr/bin/env python3
"""
数据迁移脚本 - 统一Parquet Schema

根本解决方案：
1. 统一所有数据文件的schema格式
2. 标准化symbol和时间框架命名
3. 强制包含必需字段：timestamp, symbol, timeframe, open, high, low, close, volume

运行方式：
python migrate_parquet_schema.py --dry-run  # 预览模式
python migrate_parquet_schema.py --execute  # 执行迁移
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ParquetSchemaMigrator:
    """Parquet文件Schema统一迁移工具"""

    def __init__(self, raw_dir: Path, backup_dir: Path):
        self.raw_dir = raw_dir
        self.backup_dir = backup_dir
        self.hk_dir = raw_dir / "HK"

        # 标准化映射
        self.timeframe_mapping = {
            "1m": "1min",
            "2m": "2min",
            "3m": "3min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "60m": "60min",
            "1h": "60min",
            "1day": "daily",
            "1d": "daily",
            "D": "daily",
        }

        # 列名映射
        self.column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Datetime": "timestamp",
            "Date": "timestamp",
            "Date_Time": "timestamp",
        }

    def parse_filename(self, filename: str) -> Dict[str, str]:
        """解析文件名提取symbol和时间框架信息"""
        # 示例: 0005HK_1day_2025-03-06_2025-09-02.parquet
        parts = filename.replace(".parquet", "").split("_")

        if len(parts) < 3:
            raise ValueError(f"无法解析文件名: {filename}")

        symbol = parts[0]
        timeframe = parts[1]

        # 标准化symbol: 0005HK -> 0005.HK
        if not symbol.endswith(".HK"):
            symbol = (
                f"{symbol[:4]}.{symbol[4:]}" if len(symbol) == 8 else f"{symbol}.HK"
            )

        # 标准化时间框架
        timeframe = self.timeframe_mapping.get(timeframe, timeframe)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "original_symbol": parts[0],
            "original_timeframe": parts[1],
        }

    def migrate_single_file(self, file_path: Path) -> bool:
        """迁移单个Parquet文件到统一schema"""
        try:
            # 读取原始数据
            df = pd.read_parquet(file_path)
            if df.empty:
                logger.warning(f"文件为空: {file_path}")
                return True

            # 解析文件名获取元数据
            file_info = self.parse_filename(file_path.name)

            # 重命名列
            df = df.rename(columns=self.column_mapping)

            # 确保必需列存在
            required_columns = [
                "timestamp",
                "symbol",
                "timeframe",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"缺少必需列: {missing_columns}")

            # 添加symbol和timeframe列
            df["symbol"] = file_info["symbol"]
            df["timeframe"] = file_info["timeframe"]

            # 写回文件
            df.to_parquet(file_path, index=False)
            logger.info(f"迁移完成: {file_path.name}")
            return True

        except Exception as e:
            logger.error(f"迁移失败 {file_path}: {e}")
            return False

    def create_backup(self) -> bool:
        """创建数据备份"""
        try:
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
            self.backup_dir.mkdir(parents=True)
            shutil.copytree(self.raw_dir, self.backup_dir / self.raw_dir.name)
            logger.info(f"备份创建完成: {self.backup_dir}")
            return True
        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            return False

    def validate_migration(self) -> bool:
        """验证迁移结果"""
        try:
            hk_files = list(self.hk_dir.glob("*.parquet"))
            if not hk_files:
                logger.warning("HK目录为空，无需验证")
                return True

            errors: List[str] = []
            for file_path in hk_files:
                try:
                    df = pd.read_parquet(file_path)
                    required_columns = [
                        "timestamp",
                        "symbol",
                        "timeframe",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                    for col in required_columns:
                        if col not in df.columns:
                            errors.append(f"{file_path.name}: 缺少 {col}")

                    # timeframe取值检查
                    valid_timeframes = {
                        "1min",
                        "2min",
                        "3min",
                        "5min",
                        "15min",
                        "30min",
                        "60min",
                        "daily",
                    }
                    if df["timeframe"].iloc[0] not in valid_timeframes:
                        errors.append(f"{file_path.name}: timeframe格式不正确")

                except Exception as e:
                    errors.append(f"{file_path.name}: 验证失败 - {e}")

            if errors:
                logger.error("验证发现错误:")
                for error in errors:
                    logger.error(f"  - {error}")
                return False

            logger.info("验证通过：所有文件符合统一schema")
            return True

        except Exception as e:
            logger.error(f"验证失败: {e}")
            return False

    def migrate_all(self, execute: bool = False) -> bool:
        """执行完整迁移流程"""
        logger.info(f"开始迁移 {'(执行模式)' if execute else '(预览模式)'}")

        # 检查目录
        if not self.hk_dir.exists():
            logger.error(f"HK数据目录不存在: {self.hk_dir}")
            return False

        # 统计文件
        parquet_files = list(self.hk_dir.glob("*.parquet"))
        logger.info(f"发现 {len(parquet_files)} 个Parquet文件")

        if not parquet_files:
            logger.warning("没有找到Parquet文件")
            return True

        if execute:
            # 创建备份
            logger.info("创建数据备份...")
            if not self.create_backup():
                return False

        # 迁移文件
        success_count = 0
        failed_files = []

        for file_path in parquet_files:
            if execute:
                if self.migrate_single_file(file_path):
                    success_count += 1
                else:
                    failed_files.append(file_path.name)
            else:
                # 预览模式：只检查文件格式
                try:
                    self.parse_filename(file_path.name)
                    success_count += 1
                except Exception as e:
                    failed_files.append(f"{file_path.name}: {e}")

        # 报告结果
        if execute:
            logger.info(f"迁移完成: 成功 {success_count}/{len(parquet_files)} 个文件")

            if failed_files:
                logger.error(f"迁移失败的文件:")
                for name in failed_files:
                    logger.error(f"  - {name}")

                # 恢复备份
                logger.warning("恢复备份...")
                if self.hk_dir.exists():
                    shutil.rmtree(self.hk_dir)
                shutil.copytree(self.backup_dir / "HK", self.hk_dir)
                logger.error("已恢复原始数据")
                return False

            # 验证迁移结果
            logger.info("验证迁移结果...")
            if not self.validate_migration():
                logger.error("验证失败，恢复备份...")
                if self.hk_dir.exists():
                    shutil.rmtree(self.hk_dir)
                shutil.copytree(self.backup_dir / "HK", self.hk_dir)
                return False
        else:
            logger.info(f"预览完成: 可迁移 {success_count}/{len(parquet_files)} 个文件")
            if failed_files:
                logger.warning(f"问题文件:")
                for name in failed_files:
                    logger.warning(f"  - {name}")

        return True


def main():
    parser = argparse.ArgumentParser(description="统一Parquet文件Schema")
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("raw"), help="原始数据目录路径"
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path(f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
        help="备份目录路径",
    )
    parser.add_argument(
        "--execute", action="store_true", help="执行迁移（默认为预览模式）"
    )

    args = parser.parse_args()

    migrator = ParquetSchemaMigrator(args.raw_dir, args.backup_dir)

    if not migrator.migrate_all(execute=args.execute):
        exit(1)


if __name__ == "__main__":
    main()
