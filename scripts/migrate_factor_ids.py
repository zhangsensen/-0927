#!/usr/bin/env python3
"""
因子ID迁移脚本 - 标准化因子列名

根本解决方案：
1. 统一因子ID格式：基础因子名（无参数）
2. 删除所有带参数的因子名（如RSI_14, STOCHRSI_14_K等）
3. 强制使用标准因子名 + 参数字典的方式

运行方式：
python migrate_factor_ids.py --dry-run  # 预览模式
python migrate_factor_ids.py --execute  # 执行迁移
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FactorIDMigrator:
    """因子ID标准化迁移工具"""

    def __init__(self, base_dir: Path, backup_dir: Path):
        self.base_dir = base_dir
        self.backup_dir = backup_dir

        # 定义标准因子名映射
        self.factor_mapping = {
            # 移动平均线
            'MA5': 'MA',
            'MA10': 'MA',
            'MA20': 'MA',
            'MA30': 'MA',
            'MA60': 'MA',
            'EMA5': 'EMA',
            'EMA12': 'EMA',
            'EMA26': 'EMA',

            # 技术指标
            'RSI_14': 'RSI',
            'RSI': 'RSI',
            'MACD': 'MACD',
            'STOCH': 'STOCH',
            'STOCHRSI_14_K': 'STOCHRSI',
            'STOCHRSI_14_D': 'STOCHRSI',
            'STOCHRSI': 'STOCHRSI',
            'BB_14': 'BB',
            'BB': 'BB',
            'ATR_14': 'ATR',
            'ATR': 'ATR',
            'ADX_14': 'ADX',
            'ADX': 'ADX',
            'CCI_14': 'CCI',
            'CCI': 'CCI',
            'MFI_14': 'MFI',
            'MFI': 'MFI',
            'WILLR_14': 'WILLR',
            'WILLR': 'WILLR',
            'OBV': 'OBV',
            'VOLUME_SMA': 'VOLUME_SMA',

            # 去除TA_前缀
            'TA_RSI': 'RSI',
            'TA_MACD': 'MACD',
            'TA_STOCH': 'STOCH',
            'TA_STOCHRSI': 'STOCHRSI',
            'TA_BB': 'BB',
            'TA_ATR': 'ATR',
            'TA_ADX': 'ADX',
            'TA_CCI': 'CCI',
            'TA_MFI': 'MFI',
            'TA_WILLR': 'WILLR',
            'TA_OBV': 'OBV',
        }

        # 参数提取规则
        self.param_patterns = {
            'RSI': r'RSI(?:_?(\d+))?',
            'MA': r'MA(?:_?(\d+))?',
            'EMA': r'EMA(?:_?(\d+))?',
            'BB': r'BB(?:_?(\d+))?',
            'ATR': r'ATR(?:_?(\d+))?',
            'ADX': r'ADX(?:_?(\d+))?',
            'CCI': r'CCI(?:_?(\d+))?',
            'MFI': r'MFI(?:_?(\d+))?',
            'WILLR': r'WILLR(?:_?(\d+))?',
            'STOCHRSI': r'STOCHRSI(?:_?(\d+))?',
        }

    def extract_factor_name_and_params(self, column_name: str) -> Tuple[str, Dict]:
        """
        从列名中提取因子名和参数

        Returns:
            (标准因子名, 参数字典)
        """
        # 直接映射
        if column_name in self.factor_mapping:
            return self.factor_mapping[column_name], {}

        # 尝试从列名中解析参数
        for factor_name, pattern in self.param_patterns.items():
            match = re.search(pattern, column_name, re.IGNORECASE)
            if match:
                params = {}
                if match.group(1):
                    # 提取参数值
                    param_value = int(match.group(1))
                    params = self._get_default_params(factor_name, param_value)

                return factor_name, params

        # 去除TA_前缀再试
        if column_name.startswith('TA_'):
            clean_name = column_name[3:]
            return self.extract_factor_name_and_params(clean_name)

        # 移除数字后缀再试
        base_name = re.sub(r'_?\d+$', '', column_name)
        if base_name != column_name and base_name in self.factor_mapping:
            return self.factor_mapping[base_name], {}

        # 无法识别的因子，保持原名
        logger.warning(f"无法识别的因子: {column_name}")
        return column_name, {}

    def _get_default_params(self, factor_name: str, custom_value: int) -> Dict:
        """获取因子参数字典"""
        default_params = {
            'RSI': {'timeperiod': 14},
            'MA': {'timeperiod': 20},
            'EMA': {'timeperiod': 20},
            'BB': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
            'ATR': {'timeperiod': 14},
            'ADX': {'timeperiod': 14},
            'CCI': {'timeperiod': 14},
            'MFI': {'timeperiod': 14},
            'WILLR': {'timeperiod': 14},
            'STOCHRSI': {'timeperiod': 14, 'fastk_period': 5, 'fastd_period': 3},
        }

        params = default_params.get(factor_name, {}).copy()

        # 更新自定义参数值
        if factor_name in ['RSI', 'MA', 'EMA', 'BB', 'ATR', 'ADX', 'CCI', 'MFI', 'WILLR']:
            params['timeperiod'] = custom_value
        elif factor_name == 'STOCHRSI':
            params['timeperiod'] = custom_value

        return params

    def scan_factor_files(self) -> List[Path]:
        """扫描所有因子相关文件"""
        factor_files = []

        # 扫描因子存储目录
        factor_dirs = [
            self.base_dir / "factor_system" / "factor_screening" / "screening_results",
            self.base_dir / "factor_system" / "factor_generation" / "results",
            self.base_dir / "factor_system" / "research" / "results",
        ]

        for dir_path in factor_dirs:
            if dir_path.exists():
                factor_files.extend(dir_path.rglob("*.parquet"))
                factor_files.extend(dir_path.rglob("*.csv"))
                factor_files.extend(dir_path.rglob("*.json"))

        # 扫描缓存目录
        cache_dirs = [
            Path("cache"),
            Path("factor_cache"),
        ]

        for dir_path in cache_dirs:
            if dir_path.exists():
                factor_files.extend(dir_path.rglob("*.pkl"))
                factor_files.extend(dir_path.rglob("*.parquet"))

        return list(set(factor_files))

    def migrate_parquet_file(self, file_path: Path, execute: bool = False) -> Dict:
        """迁移单个Parquet文件的因子列名"""
        result = {
            'file': str(file_path),
            'migrated_columns': [],
            'errors': [],
        }

        try:
            df = pd.read_parquet(file_path)
            if df.empty:
                return result

            # 识别需要迁移的列
            column_mapping = {}
            for col in df.columns:
                if col in ['timestamp', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']:
                    continue  # 跳过OHLCV基础列

                new_name, params = self.extract_factor_name_and_params(col)
                if new_name != col:
                    column_mapping[col] = new_name

            if column_mapping:
                result['migrated_columns'] = list(column_mapping.keys())

                if execute:
                    # 执行列重命名
                    df = df.rename(columns=column_mapping)

                    # 保存文件
                    df.to_parquet(file_path, index=False)
                    logger.info(f"迁移完成: {file_path.name} - {len(column_mapping)} 列")

        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"迁移文件失败 {file_path}: {e}")

        return result

    def migrate_csv_file(self, file_path: Path, execute: bool = False) -> Dict:
        """迁移单个CSV文件的因子列名"""
        result = {
            'file': str(file_path),
            'migrated_columns': [],
            'errors': [],
        }

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return result

            # 识别需要迁移的列
            column_mapping = {}
            for col in df.columns:
                if col in ['timestamp', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume', 'date']:
                    continue  # 跳过基础列

                new_name, params = self.extract_factor_name_and_params(col)
                if new_name != col:
                    column_mapping[col] = new_name

            if column_mapping:
                result['migrated_columns'] = list(column_mapping.keys())

                if execute:
                    # 执行列重命名
                    df = df.rename(columns=column_mapping)

                    # 保存文件
                    df.to_csv(file_path, index=False)
                    logger.info(f"迁移完成: {file_path.name} - {len(column_mapping)} 列")

        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"迁移文件失败 {file_path}: {e}")

        return result

    def migrate_json_file(self, file_path: Path, execute: bool = False) -> Dict:
        """迁移单个JSON文件中的因子引用"""
        result = {
            'file': str(file_path),
            'migrated_references': [],
            'errors': [],
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 递归替换因子名
            changed = False
            def replace_factor_names(obj):
                nonlocal changed
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key in ['factor_id', 'factor_name', 'name'] and isinstance(value, str):
                            new_name, params = self.extract_factor_name_and_params(value)
                            if new_name != value:
                                obj[key] = new_name
                                changed = True
                                result['migrated_references'].append(f"{value} -> {new_name}")
                        else:
                            replace_factor_names(value)
                elif isinstance(obj, list):
                    for item in obj:
                        replace_factor_names(item)

            original_data = json.dumps(data, sort_keys=True)
            replace_factor_names(data)

            if changed and execute:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"迁移完成: {file_path.name} - {len(result['migrated_references'])} 引用")

        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"迁移文件失败 {file_path}: {e}")

        return result

    def create_backup(self, files_to_migrate: List[Path]) -> bool:
        """创建文件备份"""
        try:
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)

            self.backup_dir.mkdir(parents=True)

            for file_path in files_to_migrate:
                relative_path = file_path.relative_to(self.base_dir)
                backup_path = self.backup_dir / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)

            logger.info(f"备份创建完成: {len(files_to_migrate)} 个文件")
            return True

        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            return False

    def migrate_all(self, execute: bool = False) -> bool:
        """执行完整迁移流程"""
        logger.info(f"开始因子ID迁移 {'(执行模式)' if execute else '(预览模式)'}")

        # 扫描文件
        files_to_migrate = self.scan_factor_files()
        logger.info(f"发现 {len(files_to_migrate)} 个因子相关文件")

        if not files_to_migrate:
            logger.warning("没有找到因子相关文件")
            return True

        if execute:
            # 创建备份
            logger.info("创建文件备份...")
            if not self.create_backup(files_to_migrate):
                return False

        # 迁移文件
        migration_results = {
            'parquet': [],
            'csv': [],
            'json': [],
            'total_migrated_columns': 0,
            'total_errors': 0,
        }

        for file_path in files_to_migrate:
            if file_path.suffix == '.parquet':
                result = self.migrate_parquet_file(file_path, execute)
                migration_results['parquet'].append(result)
            elif file_path.suffix == '.csv':
                result = self.migrate_csv_file(file_path, execute)
                migration_results['csv'].append(result)
            elif file_path.suffix == '.json':
                result = self.migrate_json_file(file_path, execute)
                migration_results['json'].append(result)

        # 统计结果
        for file_type in ['parquet', 'csv', 'json']:
            for result in migration_results[file_type]:
                migration_results['total_migrated_columns'] += len(result.get('migrated_columns', [])) + len(result.get('migrated_references', []))
                migration_results['total_errors'] += len(result.get('errors', []))

        # 报告结果
        logger.info(f"迁移完成统计:")
        logger.info(f"  Parquet文件: {len(migration_results['parquet'])}")
        logger.info(f"  CSV文件: {len(migration_results['csv'])}")
        logger.info(f"  JSON文件: {len(migration_results['json'])}")
        logger.info(f"  总迁移列/引用: {migration_results['total_migrated_columns']}")
        logger.info(f"  错误数: {migration_results['total_errors']}")

        if migration_results['total_errors'] > 0:
            logger.error("存在错误，请检查日志")
            if execute:
                logger.error("由于存在错误，建议手动检查迁移结果")
            return False

        return True


def main():
    parser = argparse.ArgumentParser(description="标准化因子ID")
    parser.add_argument('--base-dir', type=Path, default=Path('.'),
                       help='项目根目录路径')
    parser.add_argument('--backup-dir', type=Path,
                       default=Path(f'factor_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                       help='备份目录路径')
    parser.add_argument('--execute', action='store_true',
                       help='执行迁移（默认为预览模式）')

    args = parser.parse_args()

    migrator = FactorIDMigrator(args.base_dir, args.backup_dir)

    if not migrator.migrate_all(execute=args.execute):
        exit(1)


if __name__ == "__main__":
    main()