#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面策略演示脚本
展示完整的ETF横截面策略分析流程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime, timedelta
from typing import Dict, List

from factor_system.factor_engine.etf_cross_section_strategy import (
    ETFCrossSectionStrategy, StrategyConfig, run_etf_cross_section_strategy
)
from factor_system.factor_engine.providers.etf_cross_section_provider import ETFCrossSectionDataManager
from factor_system.factor_engine.factors.etf_cross_section import ETFCrossSectionFactors

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_etf_data_overview():
    """演示ETF数据概览"""
    print("=" * 60)
    print("🔍 ETF数据概览")
    print("=" * 60)

    manager = ETFCrossSectionDataManager()
    summary = manager.get_data_summary()

    print(f"📊 数据摘要:")
    print(f"   总ETF数量: {summary['total_etfs']}")
    print(f"   数据目录: {summary['data_directory']}")
    if 'date_range' in summary:
        print(f"   数据时间范围: {summary['date_range']['start']} ~ {summary['date_range']['end']}")

    print(f"\n📈 前10只ETF:")
    for i, etf in enumerate(summary['etf_list'][:10]):
        print(f"   {i+1:2d}. {etf}")

    # 获取最新横截面数据
    latest_date = "2025-10-14"
    cross_section = manager.get_cross_section_data(latest_date)

    if cross_section is not None:
        print(f"\n🎯 最新横截面数据 ({latest_date}):")
        print(f"   可用ETF: {len(cross_section)} 只")
        print(f"   平均收盘价: {cross_section['close'].mean():.3f}")
        print(f"   总成交额: {cross_section['amount'].sum()/1e8:.2f} 亿元")

        # 按成交额排序
        top_by_amount = cross_section.nlargest(5, 'amount')
        print(f"\n💰 成交额前5名:")
        for _, row in top_by_amount.iterrows():
            print(f"   {row['etf_code']}: {row['amount']/1e8:.2f} 亿元")


def demo_factor_calculation():
    """演示因子计算"""
    print("\n" + "=" * 60)
    print("🧮 因子计算演示")
    print("=" * 60)

    # 计算因子
    start_date = "2025-01-01"
    end_date = "2025-10-14"
    test_etfs = ['510300.SH', '159915.SZ', '515030.SH', '518880.SH', '513100.SH']

    calculator = ETFCrossSectionFactors()
    factors_df = calculator.calculate_all_factors(start_date, end_date, test_etfs)

    if factors_df.empty:
        print("❌ 因子计算失败")
        return

    print(f"✅ 因子计算成功: {len(factors_df)} 条记录")
    print(f"📊 因子维度: {factors_df.shape}")
    print(f"🎯 覆盖ETF: {factors_df['etf_code'].nunique()} 只")
    print(f"📅 时间范围: {factors_df['date'].min()} ~ {factors_df['date'].max()}")

    # 显示可用因子
    factor_cols = [col for col in factors_df.columns if col not in ['etf_code', 'date']]
    print(f"\n📋 可用因子 ({len(factor_cols)} 个):")
    for i, col in enumerate(factor_cols):
        print(f"   {i+1:2d}. {col}")

    # 最新因子排名
    latest_date = factors_df['date'].max()
    latest_factors = factors_df[factors_df['date'] == latest_date].copy()

    if 'composite_score' in latest_factors.columns:
        latest_factors = latest_factors.sort_values('composite_score', ascending=False)

        print(f"\n🏆 最新因子排名 ({latest_date}):")
        for i, (_, row) in enumerate(latest_factors.iterrows()):
            score = row['composite_score'] if not pd.isna(row['composite_score']) else 0
            print(f"   {i+1:2d}. {row['etf_code']}: {score:.4f}")


def demo_strategy_backtest():
    """演示策略回测"""
    print("\n" + "=" * 60)
    print("📈 策略回测演示")
    print("=" * 60)

    # 策略配置
    config = StrategyConfig(
        start_date="2024-01-01",
        end_date="2025-10-14",
        top_n=8,
        rebalance_freq="M",
        weight_method="equal",
        max_single_weight=0.20
    )

    print(f"⚙️ 策略配置:")
    print(f"   回测期间: {config.start_date} ~ {config.end_date}")
    print(f"   选择ETF数量: {config.top_n} 只")
    print(f"   调仓频率: {config.rebalance_freq}")
    print(f"   权重方法: {config.weight_method}")
    print(f"   单ETF最大权重: {config.max_single_weight:.0%}")

    # 运行回测
    strategy = ETFCrossSectionStrategy(config)
    result = strategy.run_backtest()

    if not result["success"]:
        print(f"❌ 回测失败: {result.get('error')}")
        return

    print(f"\n✅ 回测成功!")
    performance = result.get("performance", {})

    print(f"📊 回测结果:")
    print(f"   调仓次数: {result['rebalance_count']}")
    print(f"   总收益: {performance.get('total_return', 0):.2%}")
    print(f"   年化收益: {performance.get('annualized_return', 0):.2%}")
    print(f"   夏普比率: {performance.get('sharpe_ratio', 0):.2f}")
    print(f"   最大回撤: {performance.get('max_drawdown', 0):.2%}")
    print(f"   年化波动率: {performance.get('volatility', 0):.2%}")

    # 最新组合
    latest_portfolio = strategy.get_latest_portfolio()
    if latest_portfolio:
        print(f"\n🎯 最新组合:")
        for etf, weight in latest_portfolio:
            print(f"   {etf}: {weight:.2%}")

    # 组合历史分析
    portfolio_history = result.get("portfolio_history", [])
    if portfolio_history:
        print(f"\n📈 组合历史分析:")
        etf_counts = {}
        for record in portfolio_history:
            for etf, _ in record['etfs']:
                etf_counts[etf] = etf_counts.get(etf, 0) + 1

        # 按出现次数排序
        sorted_etfs = sorted(etf_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"   最常选择的ETF:")
        for etf, count in sorted_etfs[:10]:
            freq = count / len(portfolio_history) * 100
            print(f"     {etf}: {count} 次 ({freq:.1f}%)")


def demo_strategy_comparison():
    """演示策略对比"""
    print("\n" + "=" * 60)
    print("⚖️ 策略对比演示")
    print("=" * 60)

    # 不同配置的策略
    strategies = [
        {"name": "保守策略", "top_n": 5, "weight_method": "equal"},
        {"name": "均衡策略", "top_n": 8, "weight_method": "equal"},
        {"name": "激进策略", "top_n": 10, "weight_method": "score"},
        {"name": "风险平价", "top_n": 6, "weight_method": "inverse_vol"},
    ]

    results = []

    for strategy_config in strategies:
        print(f"\n🔄 运行 {strategy_config['name']}...")

        result = run_etf_cross_section_strategy(
            start_date="2024-01-01",
            end_date="2025-10-14",
            top_n=strategy_config["top_n"],
            weight_method=strategy_config["weight_method"]
        )

        if result["success"]:
            performance = result.get("performance", {})
            results.append({
                "策略": strategy_config["name"],
                "ETF数量": strategy_config["top_n"],
                "权重方法": strategy_config["weight_method"],
                "总收益": performance.get("total_return", 0),
                "年化收益": performance.get("annualized_return", 0),
                "夏普比率": performance.get("sharpe_ratio", 0),
                "最大回撤": performance.get("max_drawdown", 0),
                "调仓次数": result.get("rebalance_count", 0)
            })

    # 显示对比结果
    if results:
        comparison_df = pd.DataFrame(results)
        print(f"\n📊 策略对比结果:")
        print(comparison_df.to_string(index=False, float_format='%.2%'))

        # 找出最佳策略
        best_sharpe = comparison_df.loc[comparison_df['夏普比率'].idxmax()]
        best_return = comparison_df.loc[comparison_df['年化收益'].idxmax()]

        print(f"\n🏆 最佳夏普比率策略: {best_sharpe['策略']}")
        print(f"   夏普比率: {best_sharpe['夏普比率']:.2f}")
        print(f"   年化收益: {best_sharpe['年化收益']:.2%}")

        print(f"\n🎯 最高收益策略: {best_return['策略']}")
        print(f"   年化收益: {best_return['年化收益']:.2%}")
        print(f"   夏普比率: {best_return['夏普比率']:.2f}")


def demo_factor_analysis():
    """演示因子分析"""
    print("\n" + "=" * 60)
    print("🔬 因子分析演示")
    print("=" * 60)

    # 获取因子数据
    start_date = "2024-01-01"
    end_date = "2025-10-14"
    etf_list = ['510300.SH', '159915.SZ', '515030.SH', '518880.SH', '513100.SH',
                '512880.SH', '512480.SH', '159995.SZ', '159801.SZ', '512400.SH']

    calculator = ETFCrossSectionFactors()
    factors_df = calculator.calculate_all_factors(start_date, end_date, etf_list)

    if factors_df.empty:
        print("❌ 无法获取因子数据")
        return

    # 因子统计
    factor_cols = [col for col in factors_df.columns if col not in ['etf_code', 'date']]
    print(f"📊 因子统计 ({len(factor_cols)} 个因子):")

    for col in factor_cols[:10]:  # 显示前10个因子
        valid_data = factors_df[col].dropna()
        if len(valid_data) > 0:
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            min_val = valid_data.min()
            max_val = valid_data.max()
            print(f"   {col}:")
            print(f"     均值: {mean_val:.4f}, 标准差: {std_val:.4f}")
            print(f"     范围: [{min_val:.4f}, {max_val:.4f}]")

    # 因子相关性分析
    numeric_factors = factors_df.select_dtypes(include=[np.number])
    if not numeric_factors.empty:
        correlation_matrix = numeric_factors.corr()

        # 找出高相关性因子对
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7 and not pd.isna(corr_val):
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_val
                    ))

        if high_corr_pairs:
            print(f"\n🔗 高相关性因子对 (|相关系数| > 0.7):")
            for factor1, factor2, corr in high_corr_pairs[:5]:
                print(f"   {factor1} ↔ {factor2}: {corr:.3f}")
        else:
            print(f"\n✅ 无高相关性因子对")


def main():
    """主演示函数"""
    print("🚀 ETF横截面策略完整演示")
    print("=" * 80)

    try:
        # 1. ETF数据概览
        demo_etf_data_overview()

        # 2. 因子计算演示
        demo_factor_calculation()

        # 3. 策略回测演示
        demo_strategy_backtest()

        # 4. 策略对比演示
        demo_strategy_comparison()

        # 5. 因子分析演示
        demo_factor_analysis()

        print("\n" + "=" * 80)
        print("🎉 演示完成！")
        print("=" * 80)

    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()