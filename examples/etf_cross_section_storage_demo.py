#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面数据存储演示脚本
展示横截面数据的持久化存储、缓存和查询功能
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from factor_system.factor_engine.factors.etf_cross_section import ETFCrossSectionFactors
from factor_system.factor_engine.providers.etf_cross_section_provider import (
    ETFCrossSectionDataManager,
)
from factor_system.factor_engine.providers.etf_cross_section_storage import (
    ETFCrossSectionStorage,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_storage_basic():
    """演示存储基础功能"""
    print("=" * 60)
    print("🗄️  ETF横截面存储基础功能演示")
    print("=" * 60)

    # 初始化存储管理器
    storage = ETFCrossSectionStorage()

    # 显示存储信息
    info = storage.get_storage_info()
    print(f"📊 存储信息:")
    for key, value in info.items():
        print(f"   {key}: {value}")

    print(f"\n📁 存储目录结构:")
    print(f"   基础目录: {storage.base_dir}")
    print(f"   日度数据: {storage.daily_dir}")
    print(f"   月度数据: {storage.monthly_dir}")
    print(f"   因子数据: {storage.factors_dir}")
    print(f"   缓存数据: {storage.cache_dir}")
    print(f"   元数据: {storage.metadata_dir}")


def demo_factor_calculation_with_storage():
    """演示带存储的因子计算"""
    print("\n" + "=" * 60)
    print("🧮 带存储的因子计算演示")
    print("=" * 60)

    # 初始化因子计算器（启用存储）
    calculator = ETFCrossSectionFactors(enable_storage=True)

    # 计算因子（会自动保存）
    start_date = "2025-09-01"
    end_date = "2025-10-14"
    test_etfs = ["510300.SH", "159915.SZ", "515030.SH", "518880.SH", "513100.SH"]

    print(f"⚙️ 开始计算因子: {start_date} ~ {end_date}")
    print(f"📈 ETF列表: {test_etfs}")

    # 第一次计算（会保存到存储）
    print("\n🔄 第一次计算（保存到存储）...")
    factors_df = calculator.calculate_all_factors(
        start_date=start_date,
        end_date=end_date,
        etf_codes=test_etfs,
        use_cache=False,  # 不使用缓存
        save_to_storage=True,  # 保存到存储
    )

    if not factors_df.empty:
        print(f"✅ 因子计算成功: {len(factors_df)} 条记录")
        print(f"🎯 覆盖ETF: {factors_df['etf_code'].nunique()} 只")
        print(f"📅 时间范围: {factors_df['date'].min()} ~ {factors_df['date'].max()}")

        # 显示存储信息
        storage_info = calculator.get_storage_info()
        print(f"\n💾 存储后信息:")
        print(f"   因子文件数: {storage_info.get('factors_files', 0)}")
        print(f"   元数据文件数: {storage_info.get('metadata_files', 0)}")
        print(f"   总存储大小: {storage_info.get('total_size_mb', 0):.2f} MB")

        # 第二次计算（应该从缓存加载）
        print(f"\n🔄 第二次计算（从缓存加载）...")
        start_time = datetime.now()
        cached_factors = calculator.calculate_all_factors(
            start_date=start_date,
            end_date=end_date,
            etf_codes=test_etfs,
            use_cache=True,  # 使用缓存
            save_to_storage=False,  # 不再保存
        )
        end_time = datetime.now()
        cache_time = (end_time - start_time).total_seconds()

        if not cached_factors.empty:
            print(f"✅ 从缓存加载成功: {len(cached_factors)} 条记录")
            print(f"⚡ 加载时间: {cache_time:.3f} 秒")

            # 验证数据一致性
            if len(factors_df) == len(cached_factors):
                print(f"✅ 数据一致性验证通过")
            else:
                print(f"❌ 数据一致性验证失败")
    else:
        print("❌ 因子计算失败")


def demo_data_loading():
    """演示数据加载功能"""
    print("\n" + "=" * 60)
    print("📂 数据加载功能演示")
    print("=" * 60)

    # 初始化存储管理器
    storage = ETFCrossSectionStorage()

    # 查找可用的数据
    print("🔍 查找可用的存储数据...")

    # 加载综合因子数据
    stored_factors = storage.load_composite_factors("2025-09-01", "2025-10-14")

    if stored_factors is not None:
        print(f"✅ 成功加载综合因子数据:")
        print(f"   记录数: {len(stored_factors)}")
        print(f"   ETF数: {stored_factors['etf_code'].nunique()}")
        print(
            f"   因子列数: {len([col for col in stored_factors.columns if col not in ['etf_code', 'date']])}"
        )

        # 显示最新数据
        latest_date = stored_factors["date"].max()
        latest_data = stored_factors[stored_factors["date"] == latest_date]

        if not latest_data.empty and "composite_score" in latest_data.columns:
            print(f"\n📊 最新因子排名 ({latest_date}):")
            top_etfs = latest_data.nlargest(5, "composite_score")
            for i, (_, row) in enumerate(top_etfs.iterrows()):
                print(f"   {i+1}. {row['etf_code']}: {row['composite_score']:.4f}")
    else:
        print("❌ 未找到存储的综合因子数据")


def demo_cache_management():
    """演示缓存管理功能"""
    print("\n" + "=" * 60)
    print("💾 缓存管理功能演示")
    print("=" * 60)

    # 初始化因子计算器
    calculator = ETFCrossSectionFactors(enable_storage=True)

    # 生成测试数据
    test_data = pd.DataFrame(
        {
            "etf_code": ["510300.SH", "159915.SZ", "515030.SH"],
            "date": ["2025-10-14", "2025-10-14", "2025-10-14"],
            "composite_score": [0.1, 0.2, 0.3],
            "momentum_score": [0.15, 0.25, 0.35],
            "quality_score": [0.12, 0.22, 0.32],
        }
    )

    print(f"🧪 创建测试缓存数据: {len(test_data)} 条记录")

    # 保存到缓存
    cache_key = "test_cache_key"
    success = calculator.storage.save_cache(cache_key, test_data, ttl_hours=1)

    if success:
        print(f"✅ 缓存数据保存成功: {cache_key}")

        # 从缓存加载
        print(f"\n📂 从缓存加载数据...")
        loaded_data = calculator.storage.load_cache(cache_key)

        if loaded_data is not None:
            print(f"✅ 缓存数据加载成功: {len(loaded_data)} 条记录")
            print(f"📋 数据列: {loaded_data.columns.tolist()}")
        else:
            print(f"❌ 缓存数据加载失败")

        # 清理过期缓存
        print(f"\n🧹 清理过期缓存...")
        cleaned_count = calculator.clear_cache()
        print(f"✅ 清理完成，删除 {cleaned_count} 个文件")

    # 显示存储信息
    storage_info = calculator.get_storage_info()
    print(f"\n📊 当前存储信息:")
    for key, value in storage_info.items():
        print(f"   {key}: {value}")


def demo_factor_ranking():
    """演示因子排名功能"""
    print("\n" + "=" * 60)
    print("🏆 因子排名功能演示")
    print("=" * 60)

    # 初始化因子计算器
    calculator = ETFCrossSectionFactors(enable_storage=True)

    # 测试日期
    test_date = "2025-10-14"

    print(f"📅 查询日期: {test_date}")

    # 获取因子排名
    ranking_df = calculator.get_factor_ranking(
        date=test_date, top_n=5, factor_col="composite_score"
    )

    if not ranking_df.empty:
        print(f"✅ 成功获取因子排名:")
        for i, (_, row) in enumerate(ranking_df.iterrows()):
            score = row.get("composite_score", 0)
            print(f"   {i+1}. {row['etf_code']}: {score:.4f}")

        # 尝试其他因子排名
        other_factors = ["momentum_score", "quality_score", "liquidity_score"]
        print(f"\n📊 其他因子排名 ({test_date}):")

        for factor in other_factors:
            if factor in ranking_df.columns:
                factor_ranking = ranking_df.sort_values(factor, ascending=False)
                top_etf = factor_ranking.iloc[0]
                print(f"   {factor}: {top_etf['etf_code']} ({top_etf[factor]:.4f})")
    else:
        print(f"❌ 未获取到因子排名数据")


def demo_data_validation():
    """演示数据验证功能"""
    print("\n" + "=" * 60)
    print("🔍 数据验证功能演示")
    print("=" * 60)

    # 初始化存储管理器
    storage = ETFCrossSectionStorage()

    # 加载存储数据
    factors_df = storage.load_composite_factors("2025-09-01", "2025-10-14")

    if factors_df is not None:
        print(f"📊 数据验证报告:")
        print(f"   总记录数: {len(factors_df)}")
        print(f"   ETF数量: {factors_df['etf_code'].nunique()}")
        print(f"   日期范围: {factors_df['date'].min()} ~ {factors_df['date'].max()}")
        print(
            f"   因子列数: {len([col for col in factors_df.columns if col not in ['etf_code', 'date']])}"
        )

        # 检查数据完整性
        print(f"\n🔍 数据完整性检查:")

        # 检查空值
        null_counts = factors_df.isnull().sum()
        high_null_cols = null_counts[
            null_counts > len(factors_df) * 0.1
        ]  # 超过10%空值的列

        if high_null_cols.empty:
            print(f"   ✅ 无高缺失率列")
        else:
            print(f"   ⚠️  高缺失率列:")
            for col, count in high_null_cols.items():
                null_rate = count / len(factors_df) * 100
                print(f"      {col}: {null_rate:.1f}%")

        # 检查数值范围
        numeric_cols = factors_df.select_dtypes(include=[np.number]).columns
        for col in [
            "composite_score",
            "momentum_score",
            "quality_score",
            "liquidity_score",
        ]:
            if col in numeric_cols:
                col_data = factors_df[col].dropna()
                if not col_data.empty:
                    print(
                        f"   {col}: 范围 [{col_data.min():.4f}, {col_data.max():.4f}], 均值 {col_data.mean():.4f}"
                    )

        # 检查ETF覆盖度
        etf_counts = factors_df.groupby("etf_code").size().sort_values(ascending=False)
        print(f"\n📈 ETF覆盖度:")
        for etf, count in etf_counts.head(10).items():
            coverage = (
                count
                / len(factors_df[factors_df["date"] == factors_df["date"].max()])
                * 100
            )
            print(f"   {etf}: {count} 条记录 ({coverage:.1f}% 覆盖度)")

    else:
        print("❌ 未找到可用于验证的数据")


def main():
    """主演示函数"""
    print("🚀 ETF横截面数据存储完整演示")
    print("=" * 80)

    try:
        # 1. 存储基础功能演示
        demo_storage_basic()

        # 2. 带存储的因子计算演示
        demo_factor_calculation_with_storage()

        # 3. 数据加载功能演示
        demo_data_loading()

        # 4. 缓存管理功能演示
        demo_cache_management()

        # 5. 因子排名功能演示
        demo_factor_ranking()

        # 6. 数据验证功能演示
        demo_data_validation()

        print("\n" + "=" * 80)
        print("🎉 演示完成！")
        print("=" * 80)

    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
