#!/usr/bin/env python3
"""
Step 1: 横截面建设 - 独立执行脚本

功能：
1. 加载43只ETF的OHLCV数据
2. 计算10个精确因子
3. 保存原始因子数据到 cross_section/
4. 详细的进度日志和验证

输出：
- cross_section/{date}/{timestamp}/ohlcv/*.parquet
- cross_section/{date}/{timestamp}/factors/*.parquet
- cross_section/{date}/{timestamp}/metadata.json
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

from scripts.standard_real_data_loader import StandardRealDataLoader


def setup_logging(output_dir: Path):
    """设置详细日志"""
    log_file = output_dir / "step1_cross_section.log"

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


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_date = timestamp[:8]

    # 输出目录
    output_root = PROJECT_ROOT / "results"
    cross_section_dir = output_root / "cross_section" / run_date / timestamp
    cross_section_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(cross_section_dir)

    logger.info("=" * 80)
    logger.info("Step 1: 横截面建设（43只ETF，完整因子计算）")
    logger.info("=" * 80)
    logger.info(f"输出目录: {cross_section_dir}")
    logger.info(f"时间戳: {timestamp}")
    logger.info("")

    # ========== 1. 加载数据 ==========
    logger.info("-" * 80)
    logger.info("阶段 1/4: 加载ETF数据")
    logger.info("-" * 80)

    etf_codes = [
        # 深圳ETF (19只)
        "159801",
        "159819",
        "159859",
        "159883",
        "159915",
        "159920",
        "159928",
        "159949",
        "159992",
        "159995",
        "159998",
        # 上海ETF (24只)
        "510050",
        "510300",
        "510500",
        "511010",
        "511260",
        "511380",
        "512010",
        "512100",
        "512400",
        "512480",
        "512660",
        "512690",
        "512720",
        "512800",
        "512880",
        "512980",
        "513050",
        "513100",
        "513130",
        "513500",
        "515030",
        "515180",
        "515210",
        "515650",
        "515790",
        "516090",
        "516160",
        "516520",
        "518850",
        "518880",
        "588000",
        "588200",
    ]

    logger.info(f"ETF代码: {len(etf_codes)}只")
    logger.info(f"代码列表: {etf_codes[:5]}...（共{len(etf_codes)}只）")

    loader = StandardRealDataLoader()
    ohlcv_data = loader.load_ohlcv(
        etf_codes=etf_codes, start_date="2020-01-01", end_date="2025-10-14"
    )

    data_summary = loader.get_summary(ohlcv_data)

    logger.info("✅ 数据加载完成")
    logger.info(f"   日期: {data_summary['total_dates']} 天")
    logger.info(f"   标的: {data_summary['total_symbols']} 只")
    logger.info(
        f"   日期范围: {data_summary['date_range'][0]} -> {data_summary['date_range'][1]}"
    )
    logger.info("")

    # 覆盖率统计
    low_coverage = {
        code: ratio
        for code, ratio in data_summary["coverage_ratio"].items()
        if ratio < 0.95
    }
    if low_coverage:
        logger.warning(f"⚠️  {len(low_coverage)} 只ETF覆盖率 < 95%:")
        for code, ratio in sorted(low_coverage.items(), key=lambda x: x[1])[:10]:
            logger.warning(f"     {code}: {ratio*100:.2f}%")
    logger.info("")

    # ========== 2. 计算因子（带缓存） ==========
    logger.info("-" * 80)
    logger.info("阶段 2/4: 计算精确因子（PreciseFactorLibrary v2）")
    logger.info("-" * 80)

    cache_dir = PROJECT_ROOT / "cache" / "factors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = FactorCache(cache_dir=cache_dir, use_timestamp=True)

    lib = PreciseFactorLibrary()

    # 尝试加载缓存
    cached_factors = cache.load_factors(
        ohlcv=ohlcv_data, lib_class=lib.__class__, stage="raw"
    )

    if cached_factors is not None:
        logger.info("✅ 使用因子缓存（跳过计算）")
        factors_dict = cached_factors
    else:
        logger.info("🔄 缓存未命中，开始计算因子...")
        import time

        start_time = time.time()

        factors_df = lib.compute_all_factors(prices=ohlcv_data)

        elapsed = time.time() - start_time
        logger.info(f"✅ 因子计算完成（耗时 {elapsed:.1f}秒）")

        # 转换为字典
        factors_dict = {}
        for factor_name in lib.list_factors():
            factors_dict[factor_name] = factors_df[factor_name]

        # 保存缓存
        cache.save_factors(
            factors=factors_dict, ohlcv=ohlcv_data, lib_class=lib.__class__, stage="raw"
        )
        logger.info(f"💾 因子缓存已保存")

    logger.info("")
    logger.info(f"因子数量: {len(factors_dict)}")
    for idx, (fname, fdata) in enumerate(factors_dict.items(), start=1):
        nan_ratio = fdata.isna().sum().sum() / fdata.size
        logger.info(f"  {idx:02d}. {fname:25s} NaN率: {nan_ratio*100:.2f}%")
    logger.info("")

    # ========== 3. 保存OHLCV数据 ==========
    logger.info("-" * 80)
    logger.info("阶段 3/4: 保存OHLCV数据")
    logger.info("-" * 80)

    ohlcv_dir = cross_section_dir / "ohlcv"
    ohlcv_dir.mkdir(exist_ok=True)

    for col_name, df in ohlcv_data.items():
        output_path = ohlcv_dir / f"{col_name}.parquet"
        df.to_parquet(output_path)
        logger.info(f"  ✅ {col_name}.parquet ({df.shape[0]} × {df.shape[1]})")

    logger.info("")

    # ========== 4. 保存因子数据 ==========
    logger.info("-" * 80)
    logger.info("阶段 4/4: 保存原始因子数据")
    logger.info("-" * 80)

    factors_dir = cross_section_dir / "factors"
    factors_dir.mkdir(exist_ok=True)

    for fname, fdata in factors_dict.items():
        output_path = factors_dir / f"{fname}.parquet"
        # fdata可能是Series或DataFrame，统一转换为DataFrame
        if isinstance(fdata, pd.Series):
            df_to_save = fdata.to_frame(name=fname)
        else:
            df_to_save = fdata
        df_to_save.to_parquet(output_path)
        total_rows = len(df_to_save)
        logger.info(f"  ✅ {fname}.parquet ({total_rows} 行)")

    logger.info("")

    # ========== 5. 保存元数据 ==========
    metadata = {
        "timestamp": timestamp,
        "step": "cross_section",
        "etf_count": len(etf_codes),
        "etf_codes": etf_codes,
        "date_range": data_summary["date_range"],
        "total_dates": data_summary["total_dates"],
        "factor_count": len(factors_dict),
        "factor_names": list(factors_dict.keys()),
        "coverage_ratio": data_summary["coverage_ratio"],
        "output_dir": str(cross_section_dir),
    }

    metadata_path = cross_section_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 元数据已保存: {metadata_path}")
    logger.info("")

    # ========== 完成 ==========
    logger.info("=" * 80)
    logger.info("✅ Step 1 完成！横截面数据已构建")
    logger.info("=" * 80)
    logger.info(f"输出目录: {cross_section_dir}")
    logger.info(f"  - ohlcv/: {len(ohlcv_data)} 个文件")
    logger.info(f"  - factors/: {len(factors_dict)} 个文件")
    logger.info(f"  - metadata.json")
    logger.info("")
    logger.info("🔜 下一步: 运行 step2_factor_selection.py 进行因子筛选")
    logger.info("")

    # 返回输出目录，供后续步骤使用
    return cross_section_dir


if __name__ == "__main__":
    main()
