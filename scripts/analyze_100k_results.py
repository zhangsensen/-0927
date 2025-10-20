#!/usr/bin/env python3
"""
100,000组合向量多因子网格优化汇总分析
分析所有10个批次的结果，找出最优策略
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

def load_batch_results():
    """加载所有批次的结果"""
    results_dir = Path("strategies/results")
    batch_files = sorted(glob.glob(str(results_dir / "top35_batch*.csv")))

    all_results = []
    batch_summaries = []

    print(f"🔄 加载 {len(batch_files)} 个批次的结果...")

    for i, file in enumerate(batch_files):
        try:
            df = pd.read_csv(file)
            all_results.append(df)

            # 批次汇总信息
            best_result = df.iloc[0]
            batch_summary = {
                'batch': i,
                'total_results': len(df),
                'best_sharpe': best_result['sharpe'],
                'best_return': best_result['annual_return'],
                'best_calmar': best_result['calmar'],
                'best_top_n': best_result['top_n'],
                'max_drawdown': best_result['max_drawdown']
            }
            batch_summaries.append(batch_summary)

            print(f"  批次 {i}: {len(df)} 个结果, 最佳夏普 {best_result['sharpe']:.4f}")

        except Exception as e:
            print(f"  ❌ 批次 {i} 加载失败: {e}")

    return all_results, batch_summaries

def analyze_global_performance(all_results):
    """分析全局性能"""
    # 合并所有结果
    combined_df = pd.concat(all_results, ignore_index=True)

    print(f"\n📊 全局性能分析 (总共 {len(combined_df)} 个策略):")
    print("="*80)

    # 去重处理（可能存在重复策略）
    original_count = len(combined_df)
    # 根据权重、top_n等参数去重
    feature_cols = [col for col in combined_df.columns if col.startswith('weight_')] + ['top_n']
    combined_df = combined_df.drop_duplicates(subset=feature_cols, keep='first')
    print(f"去重前: {original_count} 个策略")
    print(f"去重后: {len(combined_df)} 个策略")

    # 全局最优策略
    global_best = combined_df.iloc[0]
    print(f"\n🏆 全局最优策略:")
    print(f"  夏普比率: {global_best['sharpe']:.4f}")
    print(f"  年化收益: {global_best['annual_return']:.4f} ({global_best['annual_return']*100:.2f}%)")
    print(f"  最大回撤: {global_best['max_drawdown']:.4f} ({global_best['max_drawdown']*100:.2f}%)")
    print(f"  卡尔玛比率: {global_best['calmar']:.4f}")
    print(f"  胜率: {global_best['win_rate']:.4f} ({global_best['win_rate']*100:.2f}%)")
    print(f"  换手率: {global_best['turnover']:.2f}")
    print(f"  Top-N: {int(global_best['top_n'])}")

    # 统计分析
    print(f"\n📈 性能统计分布:")
    print(f"  夏普比率 - 均值: {combined_df['sharpe'].mean():.4f}, 标准差: {combined_df['sharpe'].std():.4f}")
    print(f"  年化收益 - 均值: {combined_df['annual_return'].mean():.4f}, 标准差: {combined_df['annual_return'].std():.4f}")
    print(f"  最大回撤 - 均值: {combined_df['max_drawdown'].mean():.4f}, 标准差: {combined_df['max_drawdown'].std():.4f}")

    # Top-N分析
    top_n_stats = combined_df.groupby('top_n').agg({
        'sharpe': ['mean', 'std', 'max'],
        'annual_return': ['mean', 'std', 'max'],
        'max_drawdown': ['mean', 'std', 'min']
    }).round(4)

    print(f"\n🎯 Top-N性能分析:")
    print(top_n_stats)

    return combined_df, global_best

def analyze_factor_importance(combined_df):
    """分析因子重要性"""
    print(f"\n🔍 因子重要性分析:")
    print("="*60)

    # 提取权重列
    weight_cols = [col for col in combined_df.columns if col.startswith('weight_')]

    # 计算平均权重
    avg_weights = combined_df[weight_cols].mean()

    # 找出权重最大的因子
    top_factors = avg_weights.nlargest(10)

    print("Top 10 重要因子 (按平均权重):")
    for i, (factor, weight) in enumerate(top_factors.items(), 1):
        factor_name = factor.replace('weight_', '')
        print(f"  {i:2d}. {factor_name:15s}: {weight:.6f}")

    return avg_weights, top_factors

def save_results(combined_df, global_best, batch_summaries, avg_weights):
    """保存分析结果"""

    # 保存完整合并结果
    combined_df.to_csv("strategies/results/combined_100k_results.csv", index=False)
    print(f"\n💾 完整结果已保存: strategies/results/combined_100k_results.csv")

    # 保存Top 1000策略
    top_1000 = combined_df.head(1000)
    top_1000.to_csv("strategies/results/top1000_strategies.csv", index=False)
    print(f"💾 Top 1000策略已保存: strategies/results/top1000_strategies.csv")

    # 保存批次汇总
    batch_summary_df = pd.DataFrame(batch_summaries)
    batch_summary_df.to_csv("strategies/results/batch_summary.csv", index=False)
    print(f"💾 批次汇总已保存: strategies/results/batch_summary.csv")

    # 保存因子重要性
    factor_importance = pd.DataFrame({
        'factor': avg_weights.index,
        'avg_weight': avg_weights.values,
        'factor_name': [col.replace('weight_', '') for col in avg_weights.index]
    })
    factor_importance = factor_importance.sort_values('avg_weight', ascending=False)
    factor_importance.to_csv("strategies/results/factor_importance.csv", index=False)
    print(f"💾 因子重要性已保存: strategies/results/factor_importance.csv")

    # 保存最优策略详情
    best_strategy_info = {
        'sharpe': global_best['sharpe'],
        'annual_return': global_best['annual_return'],
        'max_drawdown': global_best['max_drawdown'],
        'calmar': global_best['calmar'],
        'win_rate': global_best['win_rate'],
        'turnover': global_best['turnover'],
        'top_n': int(global_best['top_n']),
        'total_combinations_tested': 100000,
        'factors_used': 35
    }

    # 添加权重信息
    weight_cols = [col for col in global_best.index if col.startswith('weight_')]
    for col in weight_cols:
        factor_name = col.replace('weight_', '')
        best_strategy_info[f'weight_{factor_name}'] = global_best[col]

    # 保存最优策略
    import json
    with open("strategies/results/best_strategy.json", 'w') as f:
        json.dump(best_strategy_info, f, indent=2)
    print(f"💾 最优策略详情已保存: strategies/results/best_strategy.json")

def main():
    """主函数"""
    print("🚀 100,000组合向量多因子网格优化 - 汇总分析")
    print("="*60)

    # 加载所有批次结果
    all_results, batch_summaries = load_batch_results()

    if not all_results:
        print("❌ 没有找到任何批次结果")
        return

    # 全局性能分析
    combined_df, global_best = analyze_global_performance(all_results)

    # 因子重要性分析
    avg_weights, top_factors = analyze_factor_importance(combined_df)

    # 保存结果
    save_results(combined_df, global_best, batch_summaries, avg_weights)

    print(f"\n✅ 分析完成!")
    print(f"   总共测试了 100,000 个权重组合 × 3个Top-N = 300,000 个策略")
    print(f"   最优策略夏普比率: {global_best['sharpe']:.4f}")
    print(f"   所有结果文件已保存到 strategies/results/ 目录")

if __name__ == "__main__":
    main()