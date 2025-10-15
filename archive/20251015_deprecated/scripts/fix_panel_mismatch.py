#!/usr/bin/env python3
"""修复面板数据不匹配问题"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fix_panel_mismatch():
    """修复面板数据不匹配问题"""
    logger.info("=== 修复面板数据不匹配问题 ===")

    # 1. 加载完整面板
    full_panel_file = "factor_output/etf_rotation/panel_20240101_20251014.parquet"
    if not Path(full_panel_file).exists():
        logger.error(f"完整面板文件不存在: {full_panel_file}")
        return

    full_panel = pd.read_parquet(full_panel_file)
    logger.info(f"完整面板加载完成: {full_panel.shape}")

    # 2. 加载白名单
    whitelist_file = "factor_output/etf_rotation/whitelist.yaml"
    if not Path(whitelist_file).exists():
        logger.error(f"白名单文件不存在: {whitelist_file}")
        return

    with open(whitelist_file) as f:
        whitelist = yaml.safe_load(f)

    whitelist_factors = [item["factor_id"] for item in whitelist["factors"]]
    logger.info(f"白名单因子: {whitelist_factors}")

    # 3. 创建因子映射
    factor_mapping = {
        # 动量因子映射（短期代理长期）
        "Momentum252": "Momentum20",  # 252日动量 -> 20日动量
        "Momentum126": "Momentum15",  # 126日动量 -> 15日动量
        "Momentum63": "Momentum10",  # 63日动量 -> 10日动量
        # 其他因子映射
        "VOLATILITY_120D": None,  # 暂时跳过
        "MOM_ACCEL": None,  # 暂时跳过
        "DRAWDOWN_63D": None,  # 暂时跳过
        # 直接匹配的因子
        "ATR14": "ATR14",  # 直接匹配
        "TA_ADX_14": "TRENDLB5",  # ADX14 -> 趋势线突破
    }

    # 4. 选择可用的因子
    available_factors = []
    factor_mapping_final = {}

    for whitelist_factor in whitelist_factors:
        mapped_factor = factor_mapping.get(whitelist_factor)

        if mapped_factor is None:
            logger.warning(f"跳过因子 {whitelist_factor}（无可用映射）")
            continue

        if mapped_factor in full_panel.columns:
            available_factors.append(mapped_factor)
            factor_mapping_final[whitelist_factor] = mapped_factor
            logger.info(f"映射: {whitelist_factor} -> {mapped_factor}")
        else:
            logger.warning(f"映射因子 {mapped_factor} 不在面板中")

    logger.info(f"最终可用因子: {len(available_factors)}个")
    logger.info(f"映射关系: {factor_mapping_final}")

    # 5. 创建修正后的面板
    if available_factors:
        # 确保索引结构一致
        if isinstance(full_panel.index, pd.MultiIndex):
            corrected_panel = full_panel[available_factors].copy()

            # 更新列名为白名单因子名
            corrected_panel.columns = [
                factor_mapping_final.get(col, col) for col in corrected_panel.columns
            ]

            # 保存修正后的面板
            output_file = (
                "factor_output/etf_rotation/panel_corrected_20240101_20251014.parquet"
            )
            corrected_panel.to_parquet(output_file)

            logger.info(f"✅ 修正面板已保存: {output_file}")
            logger.info(f"修正面板形状: {corrected_panel.shape}")

            # 6. 验证修正结果
            verify_corrected_panel(output_file, whitelist_factors, factor_mapping_final)
        else:
            logger.error("面板索引结构不是MultiIndex")
    else:
        logger.error("没有可用的因子进行修正")


def verify_corrected_panel(panel_file, whitelist_factors, factor_mapping):
    """验证修正后的面板"""
    logger.info("=== 验证修正后的面板 ===")

    panel = pd.read_parquet(panel_file)
    logger.info(f"加载修正面板: {panel.shape}")

    # 检查白名单因子是否都在修正面板中
    panel_factors = set(panel.columns)
    whitelist_set = set(factor_mapping.values())

    missing_factors = whitelist_set - panel_factors
    extra_factors = panel_factors - whitelist_set

    if missing_factors:
        logger.error(f"缺失因子: {missing_factors}")
    else:
        logger.info("✅ 所有映射因子都在修正面板中")

    if extra_factors:
        logger.warning(f"额外因子: {extra_factors}")

    # 检查覆盖率
    logger.info("=== 覆盖率统计 ===")
    for factor in panel.columns:
        coverage = (1 - panel[factor].isna().sum() / len(panel)) * 100
        status = "✅" if coverage >= 90 else "⚠️" if coverage >= 70 else "❌"
        logger.info(f"{status} {factor}: {coverage:.1f}%")

    # 检查时间范围
    if isinstance(panel.index, pd.MultiIndex):
        dates = panel.index.get_level_values(0).unique()
        logger.info(f"时间跨度: {dates[0]} 到 {dates[-1]}")
        logger.info(f"交易日数: {len(dates)}")

    logger.info("✅ 修正面板验证完成")


def main():
    """主函数"""
    fix_panel_mismatch()

    logger.info("=== 下一步建议 ===")
    logger.info("1. 更新脚本使用修正后的面板文件")
    logger.info("2. 重新运行回测验证效果")
    logger.info("3. 如需要，可考虑添加缺失的因子到etf_momentum.py")


if __name__ == "__main__":
    main()
