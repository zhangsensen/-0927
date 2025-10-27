#!/usr/bin/env python3
"""
Step 2: 因子筛选（标准化） - 独立执行脚本

功能：
1. 读取 Step 1 的横截面数据
2. 执行因子标准化（截面标准化，保留NaN）
3. 保存标准化因子到 factor_selection/
4. 详细的统计验证日志

输入：
- cross_section/{date}/{timestamp}/

输出：
- factor_selection/{date}/{timestamp}/standardized/*.parquet
- factor_selection/{date}/{timestamp}/metadata.json
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.precise_factor_library_v2 import PreciseFactorLibrary
from utils.factor_cache import FactorCache


def setup_logging(output_dir: Path):
    """设置详细日志"""
    log_file = output_dir / "step2_factor_selection.log"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def find_latest_cross_section(results_dir: Path):
    """查找最新的横截面数据目录"""
    cross_section_root = results_dir / "cross_section"

    if not cross_section_root.exists():
        return None

    # 查找所有时间戳目录
    all_runs = []
    for date_dir in cross_section_root.iterdir():
        if not date_dir.is_dir():
            continue
        for timestamp_dir in date_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue
            # 验证是否包含必要文件
            if (timestamp_dir / "metadata.json").exists():
                all_runs.append(timestamp_dir)

    if not all_runs:
        return None

    # 按时间戳排序，返回最新
    all_runs.sort(key=lambda p: p.name, reverse=True)
    return all_runs[0]


def load_cross_section_data(cross_section_dir: Path, logger):
    """加载横截面数据"""
    logger.info("-" * 80)
    logger.info("阶段 1/4: 加载横截面数据")
    logger.info("-" * 80)
    logger.info(f"输入目录: {cross_section_dir}")
    logger.info("")

    # 加载元数据
    metadata_path = cross_section_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info("📋 横截面元数据:")
    logger.info(f"  时间戳: {metadata['timestamp']}")
    logger.info(f"  ETF数量: {metadata['etf_count']}")
    logger.info(
        f"  日期范围: {metadata['date_range'][0]} -> {metadata['date_range'][1]}"
    )
    logger.info(f"  总交易日: {metadata['total_dates']}")
    logger.info(f"  因子数量: {metadata['factor_count']}")
    logger.info("")

    # 加载OHLCV
    ohlcv_dir = cross_section_dir / "ohlcv"
    ohlcv_data = {}
    for col_name in ["open", "high", "low", "close", "volume"]:
        parquet_path = ohlcv_dir / f"{col_name}.parquet"
        ohlcv_data[col_name] = pd.read_parquet(parquet_path)
        logger.info(f"  ✅ {col_name}.parquet: {ohlcv_data[col_name].shape}")

    logger.info("")

    # 加载因子
    factors_dir = cross_section_dir / "factors"
    factors_dict = {}
    for factor_name in metadata["factor_names"]:
        parquet_path = factors_dir / f"{factor_name}.parquet"
        factor_df = pd.read_parquet(parquet_path)
        # 因子文件是宽表格式（日期×标的），直接使用整个DataFrame
        factors_dict[factor_name] = factor_df

        nan_ratio = factor_df.isna().sum().sum() / factor_df.size
        logger.info(f"  ✅ {factor_name:25s} NaN率: {nan_ratio*100:.2f}%")

    logger.info("")

    return ohlcv_data, factors_dict, metadata


def standardize_factors(ohlcv_data, factors_dict, cache_dir, logger):
    """标准化因子（使用缓存）"""
    logger.info("-" * 80)
    logger.info("阶段 2/4: 因子标准化（截面标准化，保留NaN）")
    logger.info("-" * 80)

    cache = FactorCache(cache_dir=cache_dir, use_timestamp=True)
    lib = PreciseFactorLibrary()

    # 尝试加载缓存
    cached_standardized = cache.load_factors(
        ohlcv=ohlcv_data, lib_class=lib.__class__, stage="standardized"
    )

    if cached_standardized is not None:
        logger.info("✅ 使用标准化因子缓存")
        standardized_dict = cached_standardized
    else:
        logger.info("🔄 缓存未命中，开始标准化...")
        import time

        start_time = time.time()

        standardized_dict = {}
        for factor_name, factor_df in factors_dict.items():
            # 截面标准化（按行/日期标准化）
            # factor_df是DataFrame（日期×标的），对每一行进行标准化
            standardized = factor_df.apply(
                lambda row: (row - row.mean()) / row.std(), axis=1
            )
            standardized_dict[factor_name] = standardized

        elapsed = time.time() - start_time
        logger.info(f"✅ 标准化完成（耗时 {elapsed:.1f}秒）")

        # 保存缓存
        cache.save_factors(
            factors=standardized_dict,
            ohlcv=ohlcv_data,
            lib_class=lib.__class__,
            stage="standardized",
        )
        logger.info(f"💾 标准化因子缓存已保存")

    logger.info("")

    # 统计验证
    logger.info("📊 标准化验证（每个因子的截面统计）:")
    for factor_name, factor_df in standardized_dict.items():
        # 计算截面均值和标准差（每一行）
        row_means = factor_df.mean(axis=1)  # 每个日期的均值
        row_stds = factor_df.std(axis=1)  # 每个日期的标准差
        cross_sectional_mean = row_means.mean()
        cross_sectional_std = row_stds.mean()
        nan_ratio = factor_df.isna().sum().sum() / factor_df.size

        logger.info(
            f"  {factor_name:25s} "
            f"均值={cross_sectional_mean:7.4f}  "
            f"标准差={cross_sectional_std:7.4f}  "
            f"NaN率={nan_ratio*100:6.2f}%"
        )

    logger.info("")

    return standardized_dict


def save_standardized_factors(standardized_dict, output_dir, metadata, logger):
    """保存标准化因子"""
    logger.info("-" * 80)
    logger.info("阶段 3/4: 保存标准化因子")
    logger.info("-" * 80)

    standardized_dir = output_dir / "standardized"
    standardized_dir.mkdir(parents=True, exist_ok=True)

    for fname, fdata in standardized_dict.items():
        output_path = standardized_dir / f"{fname}.parquet"
        # fdata已经是DataFrame，直接保存
        fdata.to_parquet(output_path)
        logger.info(f"  ✅ {fname}.parquet")

    logger.info("")

    # 保存元数据
    selection_metadata = {
        **metadata,
        "step": "factor_selection",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "standardized_factor_count": len(standardized_dict),
        "standardized_factor_names": list(standardized_dict.keys()),
        "output_dir": str(output_dir),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(selection_metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 元数据已保存: {metadata_path}")
    logger.info("")

    return selection_metadata


def main(cross_section_dir: Path = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_date = timestamp[:8]

    # 输出目录
    output_root = PROJECT_ROOT / "results"
    selection_dir = output_root / "factor_selection" / run_date / timestamp
    selection_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(selection_dir)

    logger.info("=" * 80)
    logger.info("Step 2: 因子筛选（标准化处理）")
    logger.info("=" * 80)
    logger.info(f"输出目录: {selection_dir}")
    logger.info(f"时间戳: {timestamp}")
    logger.info("")

    # 查找输入数据
    if cross_section_dir is None:
        logger.info("🔍 自动查找最新的横截面数据...")
        cross_section_dir = find_latest_cross_section(output_root)

        if cross_section_dir is None:
            logger.error("❌ 未找到横截面数据！请先运行 step1_cross_section.py")
            sys.exit(1)

        logger.info(f"✅ 找到最新数据: {cross_section_dir}")
        logger.info("")

    # 1. 加载数据
    ohlcv_data, factors_dict, metadata = load_cross_section_data(
        cross_section_dir, logger
    )

    # 2. 标准化
    cache_dir = PROJECT_ROOT / "cache" / "factors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    standardized_dict = standardize_factors(ohlcv_data, factors_dict, cache_dir, logger)

    # 3. 保存
    selection_metadata = save_standardized_factors(
        standardized_dict, selection_dir, metadata, logger
    )

    # 完成
    logger.info("-" * 80)
    logger.info("阶段 4/4: 完成摘要")
    logger.info("-" * 80)
    logger.info("=" * 80)
    logger.info("✅ Step 2 完成！因子已标准化")
    logger.info("=" * 80)
    logger.info(f"输出目录: {selection_dir}")
    logger.info(f"  - standardized/: {len(standardized_dict)} 个文件")
    logger.info(f"  - metadata.json")
    logger.info("")
    logger.info("🔜 下一步: 运行 step3_run_wfo.py 进行WFO优化")
    logger.info("")

    return selection_dir


if __name__ == "__main__":
    main()
