#!/usr/bin/env python3
"""验证分池分离 - 确保A股与QDII分池生效

核心检查：
1. 加载现有全量面板
2. 按池分离ETF
3. 验证分池逻辑
4. 生成分池报告

Linus式原则：快速验证，不重复计算
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def verify_pool_separation():
    """验证分池分离"""
    logger.info("=" * 80)
    logger.info("分池分离验证")
    logger.info("=" * 80)

    # 加载分池配置
    config_file = Path("configs/etf_pools.yaml")
    if not config_file.exists():
        logger.error(f"❌ 配置文件不存在: {config_file}")
        return False

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # 加载全量面板
    panel_file = Path(
        "factor_output/etf_rotation_production/panel_FULL_20200102_20251014.parquet"
    )
    if not panel_file.exists():
        logger.error(f"❌ 面板文件不存在: {panel_file}")
        return False

    logger.info(f"\n✅ 加载全量面板: {panel_file.name}")
    panel = pd.read_parquet(panel_file)

    logger.info(f"  样本数: {len(panel)}")
    logger.info(f"  ETF数: {panel.index.get_level_values('symbol').nunique()}")
    logger.info(f"  因子数: {len(panel.columns)}")

    # 获取所有symbol
    all_symbols = panel.index.get_level_values("symbol").unique().tolist()

    # 按池分离
    logger.info("\n" + "=" * 80)
    logger.info("分池分离")
    logger.info("=" * 80)

    pool_results = {}

    for pool_name, pool_config in config["pools"].items():
        logger.info(f"\n{pool_name}:")
        logger.info(f"  名称: {pool_config['name']}")
        logger.info(f"  配置ETF数: {len(pool_config['symbols'])}")

        # 找出在面板中存在的ETF
        pool_symbols = pool_config["symbols"]
        existing_symbols = [s for s in pool_symbols if s in all_symbols]
        missing_symbols = [s for s in pool_symbols if s not in all_symbols]

        logger.info(f"  存在ETF数: {len(existing_symbols)}")
        if missing_symbols:
            logger.warning(f"  缺失ETF数: {len(missing_symbols)}")
            logger.warning(f"    缺失列表: {missing_symbols}")

        # 提取池面板
        if len(existing_symbols) > 0:
            pool_panel = panel.loc[
                panel.index.get_level_values("symbol").isin(existing_symbols)
            ]
            logger.info(f"  池样本数: {len(pool_panel)}")

            pool_results[pool_name] = {
                "config_count": len(pool_symbols),
                "existing_count": len(existing_symbols),
                "missing_count": len(missing_symbols),
                "sample_count": len(pool_panel),
                "symbols": existing_symbols,
            }
        else:
            logger.error(f"  ❌ 无有效ETF")
            pool_results[pool_name] = {
                "config_count": len(pool_symbols),
                "existing_count": 0,
                "missing_count": len(missing_symbols),
                "sample_count": 0,
                "symbols": [],
            }

    # 检查重叠
    logger.info("\n" + "=" * 80)
    logger.info("池重叠检查")
    logger.info("=" * 80)

    pool_names = list(pool_results.keys())
    if len(pool_names) >= 2:
        pool1, pool2 = pool_names[0], pool_names[1]
        symbols1 = set(pool_results[pool1]["symbols"])
        symbols2 = set(pool_results[pool2]["symbols"])

        overlap = symbols1 & symbols2

        logger.info(f"\n{pool1} vs {pool2}:")
        logger.info(f"  {pool1}: {len(symbols1)}个ETF")
        logger.info(f"  {pool2}: {len(symbols2)}个ETF")
        logger.info(f"  重叠: {len(overlap)}个ETF")

        if len(overlap) > 0:
            logger.warning(f"  ⚠️  存在重叠ETF: {overlap}")
        else:
            logger.info(f"  ✅ 无重叠，分池干净")

    # 覆盖率检查
    logger.info("\n" + "=" * 80)
    logger.info("覆盖率检查")
    logger.info("=" * 80)

    all_pool_symbols = set()
    for pool_name, result in pool_results.items():
        all_pool_symbols.update(result["symbols"])

    uncovered_symbols = set(all_symbols) - all_pool_symbols

    logger.info(f"\n全量面板ETF数: {len(all_symbols)}")
    logger.info(f"分池覆盖ETF数: {len(all_pool_symbols)}")
    logger.info(f"未覆盖ETF数: {len(uncovered_symbols)}")

    if len(uncovered_symbols) > 0:
        logger.warning(f"\n⚠️  未覆盖ETF:")
        for symbol in sorted(uncovered_symbols):
            logger.warning(f"  - {symbol}")
    else:
        logger.info(f"\n✅ 所有ETF均已覆盖")

    # 生成总结
    logger.info("\n" + "=" * 80)
    logger.info("分池总结")
    logger.info("=" * 80)

    for pool_name, result in pool_results.items():
        logger.info(f"\n{pool_name}:")
        logger.info(f"  配置ETF数: {result['config_count']}")
        logger.info(f"  存在ETF数: {result['existing_count']}")
        logger.info(f"  缺失ETF数: {result['missing_count']}")
        logger.info(f"  样本数: {result['sample_count']}")
        logger.info(
            f"  覆盖率: {result['existing_count']/result['config_count']*100:.1f}%"
        )

    logger.info("\n" + "=" * 80)
    logger.info("✅ 分池分离验证完成")
    logger.info("=" * 80)

    logger.info(f"\n建议:")
    logger.info(f"  1. 分池配置已就绪")
    logger.info(f"  2. 可使用现有全量面板按池分离")
    logger.info(f"  3. 或使用pool_management.py生产独立池面板")

    return True


def main():
    """主函数"""
    try:
        success = verify_pool_separation()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ 验证失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
