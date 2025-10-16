#!/usr/bin/env python3
"""ETF因子面板生产（优化版）

Linus原则:
1. 配置驱动 - 消除硬编码
2. 质量优先 - 自动去重和筛选
3. 性能优化 - 数据类型和压缩优化
4. 完整元数据 - 可追溯可监控
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from etf_factor_engine_production.configs.config_manager import get_config
from etf_factor_engine_production.scripts.factor_optimizer import FactorOptimizer
from etf_factor_engine_production.scripts.metadata_generator import MetadataGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_raw_panel(panel_file: Path) -> pd.DataFrame:
    """加载原始因子面板
    
    Args:
        panel_file: 面板文件路径
        
    Returns:
        因子面板DataFrame
    """
    logger.info(f"加载原始面板: {panel_file}")
    
    if not panel_file.exists():
        raise FileNotFoundError(f"面板文件不存在: {panel_file}")
    
    df = pd.read_parquet(panel_file)
    logger.info(f"  形状: {df.shape}")
    logger.info(f"  内存: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="ETF因子面板优化生产（配置驱动版）"
    )
    parser.add_argument(
        "--input-panel",
        help="输入面板文件路径（默认使用最新的FULL面板）"
    )
    parser.add_argument(
        "--output-suffix",
        default="optimized",
        help="输出文件后缀（默认: optimized）"
    )
    parser.add_argument(
        "--skip-correlation",
        action="store_true",
        help="跳过相关性去重"
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="跳过质量筛选"
    )
    parser.add_argument(
        "--skip-dtype",
        action="store_true",
        help="跳过数据类型优化"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = get_config()
    
    logger.info("=" * 60)
    logger.info("ETF因子面板优化生产")
    logger.info("=" * 60)
    
    # 1. 确定输入文件
    if args.input_panel:
        input_file = Path(args.input_panel)
    else:
        # 使用最新的FULL面板
        output_dir = Path(config.output_dir)
        full_panels = sorted(output_dir.glob("panel_FULL_*.parquet"))
        
        if not full_panels:
            logger.error("未找到FULL面板文件，请先运行produce_full_etf_panel.py")
            return
        
        input_file = full_panels[-1]
    
    logger.info(f"输入文件: {input_file}")
    
    # 2. 加载原始面板
    df = load_raw_panel(input_file)
    
    # 3. 初始化优化器
    optimizer = FactorOptimizer(
        correlation_threshold=config.correlation_threshold,
        null_threshold=config.null_threshold,
        zero_threshold=config.zero_threshold,
        volatility_threshold=config.get('quality.volatility_threshold', 100.0)
    )
    
    # 4. 执行优化
    optimized_df, report = optimizer.optimize_panel(
        df,
        remove_correlation=not args.skip_correlation,
        remove_low_quality=not args.skip_quality,
        optimize_dtype=not args.skip_dtype
    )
    
    # 5. 保存优化后的面板
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取日期范围用于文件名
    if isinstance(optimized_df.index, pd.MultiIndex):
        dates = optimized_df.index.get_level_values('date')
        start_date = dates.min().strftime('%Y%m%d')
        end_date = dates.max().strftime('%Y%m%d')
    else:
        start_date = optimized_df.index.min().strftime('%Y%m%d')
        end_date = optimized_df.index.max().strftime('%Y%m%d')
    
    output_file = output_dir / f"panel_{args.output_suffix}_{start_date}_{end_date}.parquet"
    
    logger.info(f"保存优化面板: {output_file}")
    optimized_df.to_parquet(
        output_file,
        compression=config.compression,
        index=True
    )
    
    file_size_mb = output_file.stat().st_size / 1024 / 1024
    logger.info(f"  文件大小: {file_size_mb:.1f} MB")
    
    # 6. 生成元数据
    run_params = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "engine_version": config.engine_version,
        "skip_correlation": args.skip_correlation,
        "skip_quality": args.skip_quality,
        "skip_dtype": args.skip_dtype,
        "config": {
            "correlation_threshold": config.correlation_threshold,
            "null_threshold": config.null_threshold,
            "zero_threshold": config.zero_threshold,
            "compression": config.compression
        }
    }
    
    metadata = MetadataGenerator.generate_metadata(
        optimized_df,
        output_file,
        run_params,
        report
    )
    
    # 保存元数据
    meta_file = output_dir / f"panel_{args.output_suffix}_{start_date}_{end_date}_meta.json"
    MetadataGenerator.save_metadata(metadata, meta_file)
    
    # 7. 保存移除因子列表
    removed_file = output_dir / f"removed_factors_{args.output_suffix}_{start_date}_{end_date}.json"
    with open(removed_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(optimizer.removed_factors, f, indent=2, ensure_ascii=False)
    
    logger.info(f"移除因子列表已保存: {removed_file}")
    
    # 8. 打印总结
    logger.info("=" * 60)
    logger.info("优化完成总结")
    logger.info("=" * 60)
    logger.info(f"原始因子: {df.shape[1]}个")
    logger.info(f"优化因子: {optimized_df.shape[1]}个")
    logger.info(f"保留率: {optimized_df.shape[1]/df.shape[1]*100:.1f}%")
    logger.info(f"文件大小: {file_size_mb:.1f} MB")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"元数据: {meta_file}")
    logger.info(f"移除列表: {removed_file}")


if __name__ == "__main__":
    main()
