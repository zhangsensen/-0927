#!/usr/bin/env python3
"""测试One Pass全量面板方案"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_panel_structure():
    """测试面板结构"""
    logger.info("=" * 60)
    logger.info("测试1: 面板结构")
    logger.info("=" * 60)

    # 查找最新的全量面板
    panel_dir = Path("factor_output/etf_rotation")
    panel_files = list(panel_dir.glob("panel_FULL_*.parquet"))

    if not panel_files:
        logger.error("未找到全量面板文件")
        return False

    panel_file = sorted(panel_files)[-1]
    logger.info(f"加载面板: {panel_file}")

    panel = pd.read_parquet(panel_file)
    logger.info(f"面板形状: {panel.shape}")
    logger.info(f"索引类型: {type(panel.index)}")
    logger.info(f"索引名称: {panel.index.names}")

    # 检查MultiIndex
    if not isinstance(panel.index, pd.MultiIndex):
        logger.error("❌ 索引不是MultiIndex")
        return False

    if panel.index.names != ["symbol", "date"]:
        logger.error(f"❌ 索引名称错误: {panel.index.names}")
        return False

    logger.info("✅ 面板结构正确")
    return True


def test_factor_summary():
    """测试因子概要"""
    logger.info("\n" + "=" * 60)
    logger.info("测试2: 因子概要")
    logger.info("=" * 60)

    # 查找最新的因子概要
    summary_dir = Path("factor_output/etf_rotation")
    summary_files = list(summary_dir.glob("factor_summary_*.csv"))

    if not summary_files:
        logger.error("未找到因子概要文件")
        return False

    summary_file = sorted(summary_files)[-1]
    logger.info(f"加载概要: {summary_file}")

    summary = pd.read_csv(summary_file)
    logger.info(f"因子数量: {len(summary)}")

    # 检查必需字段
    required_fields = [
        "factor_id",
        "coverage",
        "zero_variance",
        "min_history",
        "required_fields",
        "reason",
    ]
    for field in required_fields:
        if field not in summary.columns:
            logger.error(f"❌ 缺少字段: {field}")
            return False

    logger.info("✅ 因子概要字段完整")

    # 统计
    logger.info(f"\n覆盖率分布:\n{summary['coverage'].describe()}")
    logger.info(f"\n零方差因子: {summary['zero_variance'].sum()}/{len(summary)}")

    failed = summary[summary["reason"] != "success"]
    if not failed.empty:
        logger.warning(f"\n失败因子: {len(failed)}")
        for _, row in failed.head(5).iterrows():
            logger.warning(f"  {row['factor_id']}: {row['reason']}")

    return True


def test_metadata():
    """测试元数据"""
    logger.info("\n" + "=" * 60)
    logger.info("测试3: 元数据")
    logger.info("=" * 60)

    import json

    meta_file = Path("factor_output/etf_rotation/panel_meta.json")
    if not meta_file.exists():
        logger.error("未找到元数据文件")
        return False

    with open(meta_file) as f:
        meta = json.load(f)

    logger.info(f"元数据: {json.dumps(meta, indent=2)}")

    # 检查必需字段
    required_fields = ["engine_version", "price_field", "run_params", "timestamp"]
    for field in required_fields:
        if field not in meta:
            logger.error(f"❌ 缺少字段: {field}")
            return False

    logger.info("✅ 元数据完整")
    return True


def test_filter_workflow():
    """测试筛选流程"""
    logger.info("\n" + "=" * 60)
    logger.info("测试4: 筛选流程")
    logger.info("=" * 60)

    # 查找筛选后的面板
    panel_dir = Path("factor_output/etf_rotation")

    # 检查生产模式
    prod_panel = panel_dir / "panel_filtered_production.parquet"
    if prod_panel.exists():
        panel = pd.read_parquet(prod_panel)
        logger.info(f"✅ 生产模式面板: {panel.shape}")
    else:
        logger.warning("⚠️  未找到生产模式面板")

    # 检查研究模式
    research_panel = panel_dir / "panel_filtered_research.parquet"
    if research_panel.exists():
        panel = pd.read_parquet(research_panel)
        logger.info(f"✅ 研究模式面板: {panel.shape}")
    else:
        logger.warning("⚠️  未找到研究模式面板")

    # 检查因子清单
    prod_factors = panel_dir / "factors_selected_production.yaml"
    if prod_factors.exists():
        import yaml

        with open(prod_factors) as f:
            factors = yaml.safe_load(f)
        logger.info(f"✅ 生产模式因子清单: {len(factors.get('factors', []))} 个")
    else:
        logger.warning("⚠️  未找到生产模式因子清单")

    return True


def main():
    logger.info("=" * 60)
    logger.info("One Pass全量面板方案测试")
    logger.info("=" * 60)

    results = []

    # 测试1: 面板结构
    results.append(("面板结构", test_panel_structure()))

    # 测试2: 因子概要
    results.append(("因子概要", test_factor_summary()))

    # 测试3: 元数据
    results.append(("元数据", test_metadata()))

    # 测试4: 筛选流程
    results.append(("筛选流程", test_filter_workflow()))

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("测试汇总")
    logger.info("=" * 60)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{name}: {status}")

    all_passed = all(r for _, r in results)
    if all_passed:
        logger.info("\n🎉 所有测试通过！")
    else:
        logger.error("\n❌ 部分测试失败")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
