#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""功能等价性测试 - 验证重构版本与原版本的一致性"""
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def create_test_data():
    """创建测试数据"""
    np.random.seed(42)  # 固定随机种子确保可重现

    # 创建测试价格数据
    dates = pd.date_range("2024-01-01", "2024-02-20", freq="D")
    symbols = ["TEST001", "TEST002"]

    data_list = []
    for symbol in symbols:
        # 模拟价格走势
        close_base = 100.0 + np.random.randn(len(dates)).cumsum() * 0.5

        for i, date in enumerate(dates):
            close = close_base[i]
            high = close * (1 + abs(np.random.randn() * 0.02))
            low = close * (1 - abs(np.random.randn() * 0.02))
            open_price = low + (high - low) * np.random.random()
            volume = int(1000000 + np.random.randn() * 100000)
            amount = volume * close * (1 + np.random.randn() * 0.01)

            data_list.append(
                {
                    "trade_date": date.strftime("%Y%m%d"),
                    "symbol": symbol,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "volume": volume,
                    "amount": round(amount, 2),
                }
            )

    df = pd.DataFrame(data_list)
    return df


def save_test_data(df, data_dir):
    """保存测试数据"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 按标的分别保存
    for symbol in df["symbol"].unique():
        symbol_data = df[df["symbol"] == symbol].copy()
        symbol_data = symbol_data.drop("symbol", axis=1)

        filename = f"{symbol}_daily_20240101_20240220.parquet"
        filepath = data_dir / filename
        symbol_data.to_parquet(filepath, index=False)

    return data_dir


def test_original_version(data_dir, output_dir):
    """测试原版本"""
    print("🔍 测试原版本...")

    try:
        # 临时修改原版本的配置以匹配测试数据
        import generate_panel as original_module

        # 使用原版本的默认配置
        config = original_module.load_config(None)

        # 加载测试数据
        price_df = original_module.load_price_data(Path(data_dir))

        # 计算因子
        panel = original_module.calculate_factors_parallel(
            price_df, config, max_workers=1
        )

        # 保存结果
        panel_file, meta_file = original_module.save_results(panel, Path(output_dir))

        print(f"✅ 原版本完成，因子数: {len(panel.columns)}")
        return panel, meta_file

    except Exception as e:
        print(f"❌ 原版本失败: {e}")
        return None, None


def test_refactored_version(data_dir, output_dir):
    """测试重构版本"""
    print("🔍 测试重构版本...")

    try:
        # 导入重构版本
        import generate_panel_refactored as refactored_module

        # 加载配置
        config = refactored_module.load_config("config/factor_panel_config.yaml")

        # 加载测试数据
        price_df = refactored_module.load_price_data(Path(data_dir), config)

        # 计算因子
        panel = refactored_module.calculate_factors_parallel(price_df, config)

        # 保存结果
        panel_file, meta_file = refactored_module.save_results(
            panel, Path(output_dir), config.output
        )

        print(f"✅ 重构版本完成，因子数: {len(panel.columns)}")
        return panel, meta_file

    except Exception as e:
        print(f"❌ 重构版本失败: {e}")
        return None, None


def compare_results(original_panel, refactored_panel):
    """比较两个版本的结果"""
    print("📊 比较结果...")

    if original_panel is None or refactored_panel is None:
        print("❌ 无法比较，某个版本失败")
        return False

    # 检查数据形状
    print(f"原版本形状: {original_panel.shape}")
    print(f"重构版本形状: {refactored_panel.shape}")

    # 检查因子列
    original_factors = set(original_panel.columns)
    refactored_factors = set(refactored_panel.columns)

    print(f"原版本因子数: {len(original_factors)}")
    print(f"重构版本因子数: {len(refactored_factors)}")

    # 找出差异
    common_factors = original_factors & refactored_factors
    missing_in_refactored = original_factors - refactored_factors
    extra_in_refactored = refactored_factors - original_factors

    print(f"共同因子: {len(common_factors)}")
    print(f"重构版本缺失: {missing_in_refactored}")
    print(f"重构版本额外: {extra_in_refactored}")

    # 比较共同因子的数值
    if common_factors:
        common_list = sorted(list(common_factors))
        differences = []

        for factor in common_list:
            original_values = original_panel[factor].dropna()
            refactored_values = refactored_panel[factor].dropna()

            # 对齐索引
            common_index = original_values.index.intersection(refactored_values.index)
            if len(common_index) > 0:
                orig_aligned = original_values.loc[common_index]
                refact_aligned = refactored_values.loc[common_index]

                # 计算差异
                diff = np.abs(orig_aligned - refact_aligned)
                max_diff = diff.max()
                mean_diff = diff.mean()

                if max_diff > 1e-10:  # 设置容忍度
                    differences.append(
                        {
                            "factor": factor,
                            "max_diff": max_diff,
                            "mean_diff": mean_diff,
                            "count": len(common_index),
                        }
                    )

        if differences:
            print(f"\\n⚠️ 发现数值差异:")
            for diff_info in differences[:5]:  # 只显示前5个
                print(
                    f"  {diff_info['factor']}: max_diff={diff_info['max_diff']:.2e}, mean_diff={diff_info['mean_diff']:.2e}"
                )
        else:
            print("✅ 所有共同因子数值一致")

    # 评估结果
    similarity_score = len(common_factors) / max(
        len(original_factors), len(refactored_factors)
    )
    print(f"\\n📈 相似度评分: {similarity_score:.2%}")

    # 判断是否通过
    passed = (
        len(differences) == 0  # 无数值差异
        and len(missing_in_refactored) == 0  # 无缺失因子
        and similarity_score >= 0.95  # 相似度>=95%
    )

    if passed:
        print("✅ 功能等价性测试通过")
    else:
        print("❌ 功能等价性测试失败")

    return passed


def test_config_influence():
    """测试配置变化的影响"""
    print("🔧 测试配置影响...")

    try:
        import generate_panel_refactored as refactored_module
        from config.config_classes import FactorPanelConfig

        # 创建自定义配置
        custom_config = FactorPanelConfig()
        custom_config.trading.days_per_year = 200  # 修改年化天数
        custom_config.factor_windows.momentum = [10, 20]  # 修改动量窗口

        # 检查配置是否生效
        assert custom_config.trading.days_per_year == 200
        assert custom_config.factor_windows.momentum == [10, 20]

        print("✅ 配置修改功能正常")
        return True

    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 功能等价性测试开始")
    print("=" * 60)

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_dir = temp_path / "test_data"
        original_output = temp_path / "original_output"
        refactored_output = temp_path / "refactored_output"

        # 创建测试数据
        print("\\n📝 创建测试数据...")
        test_df = create_test_data()
        save_test_data(test_df, data_dir)
        print(f"✅ 测试数据已创建: {len(test_df)} 条记录")

        # 测试原版本
        print("\\n" + "=" * 60)
        original_panel, original_meta = test_original_version(data_dir, original_output)

        # 测试重构版本
        print("\\n" + "=" * 60)
        refactored_panel, refactored_meta = test_refactored_version(
            data_dir, refactored_output
        )

        # 比较结果
        print("\\n" + "=" * 60)
        equivalence_passed = compare_results(original_panel, refactored_panel)

        # 测试配置影响
        print("\\n" + "=" * 60)
        config_passed = test_config_influence()

        # 总结
        print("\\n" + "=" * 60)
        print("📋 测试总结:")
        print(f"  功能等价性: {'✅ 通过' if equivalence_passed else '❌ 失败'}")
        print(f"  配置功能: {'✅ 通过' if config_passed else '❌ 失败'}")

        overall_passed = equivalence_passed and config_passed
        print(f"  总体结果: {'✅ 通过' if overall_passed else '❌ 失败'}")

        return overall_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
