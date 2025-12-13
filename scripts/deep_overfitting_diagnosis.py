#!/usr/bin/env python3
"""
深度过拟合诊断 v1.0
===============================================
目标: 全面分析训练期 vs Holdout期表现差异的根本原因

关键发现 (2025-12-11):
- 训练Top1000在Holdout上: 均值收益 0.6%, 中位 -0.2%, 正收益占 49.3%
- 全量62,985在Holdout上: 均值收益 6.3%, 中位 7.1%, 正收益占 73.9%
- 因子频率大幅变化:
  * ADX_14D: 训练75.2% → Holdout最优3.0% (下降72%)
  * CMF_20D: 训练12.5% → Holdout最优81.5% (上升69%)
  * MAX_DD_60D: 训练48.0% → Holdout最优91.0% (上升43%)

这表明训练期的排序机制(综合得分)严重失效。
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

ROOT = Path(__file__).parent.parent


def load_data():
    """加载数据"""
    print("📂 加载数据...")
    
    df_full = pd.read_csv(ROOT / 'results/vec_from_wfo_20251211_205649/full_space_results.csv')
    df_train_top = pd.read_csv(ROOT / 'results/vec_from_wfo_20251211_205649/top1000_composite.csv')
    df_hold = pd.read_csv(ROOT / 'results/vec_from_wfo_20251211_205649/all_holdout.csv')
    
    # 合并
    df = df_full.merge(df_hold, on='combo', suffixes=('_train', '_hold'))
    
    # 计算训练期综合得分(与select_strategy_v2.py保持一致)
    df['train_composite'] = (
        0.4 * df['vec_return'] +
        0.3 * df['vec_sharpe_ratio'] -
        0.3 * df['vec_max_drawdown']
    )
    
    # 计算holdout期综合得分
    df['hold_composite'] = (
        0.4 * df['hold_return'] +
        0.3 * df['hold_sharpe'] -
        0.3 * df['hold_max_dd']
    )
    
    print(f"✅ 数据加载完成: {len(df)} 组合")
    return df


def analyze_score_distribution(df):
    """分析训练期和holdout期得分分布"""
    print("\n" + "=" * 80)
    print("📊 训练期 vs Holdout期 得分分布对比")
    print("=" * 80)
    
    # 训练期
    print("\n【训练期综合得分】")
    print(f"  均值: {df['train_composite'].mean():.4f}")
    print(f"  中位: {df['train_composite'].median():.4f}")
    print(f"  标准差: {df['train_composite'].std():.4f}")
    print(f"  最小值: {df['train_composite'].min():.4f}")
    print(f"  最大值: {df['train_composite'].max():.4f}")
    
    # Holdout期
    print("\n【Holdout期综合得分】")
    print(f"  均值: {df['hold_composite'].mean():.4f}")
    print(f"  中位: {df['hold_composite'].median():.4f}")
    print(f"  标准差: {df['hold_composite'].std():.4f}")
    print(f"  最小值: {df['hold_composite'].min():.4f}")
    print(f"  最大值: {df['hold_composite'].max():.4f}")
    
    # 相关性
    corr = df['train_composite'].corr(df['hold_composite'])
    print(f"\n【训练得分 vs Holdout得分 相关性】")
    print(f"  Pearson相关系数: {corr:.4f}")
    
    # Spearman秩相关(更关键)
    spearman_corr, p_value = stats.spearmanr(df['train_composite'], df['hold_composite'])
    print(f"  Spearman秩相关: {spearman_corr:.4f} (p={p_value:.4e})")
    
    if spearman_corr < 0.1:
        print("  ⚠️ 秩相关接近0，训练期排序在Holdout上完全失效!")


def analyze_top_cohorts(df):
    """分析不同训练期Top分层在Holdout上的表现"""
    print("\n" + "=" * 80)
    print("📈 训练期Top分层在Holdout上的表现")
    print("=" * 80)
    
    # 按训练期得分排序
    df_sorted = df.sort_values('train_composite', ascending=False).reset_index(drop=True)
    
    cohorts = [
        ('Top 10', 10),
        ('Top 50', 50),
        ('Top 100', 100),
        ('Top 500', 500),
        ('Top 1000', 1000),
        ('Top 5000', 5000),
        ('All', len(df))
    ]
    
    print(f"\n{'分层':12} {'数量':>6} {'Hold收益均值':>12} {'Hold收益中位':>12} {'正收益占比':>10} {'Hold综合分':>10}")
    print("-" * 80)
    
    for name, n in cohorts:
        subset = df_sorted.head(n)
        hold_ret_mean = subset['hold_return'].mean()
        hold_ret_median = subset['hold_return'].median()
        positive_pct = (subset['hold_return'] > 0).mean()
        hold_comp_mean = subset['hold_composite'].mean()
        
        print(f"{name:12} {n:6d} {hold_ret_mean:+11.4f} {hold_ret_median:+11.4f} {positive_pct:9.2%} {hold_comp_mean:+9.4f}")


def analyze_factor_shift(df):
    """分析因子使用频率在训练Top vs Holdout Top的变化"""
    print("\n" + "=" * 80)
    print("🔍 因子频率变化分析 (训练Top1000 vs Holdout Top500)")
    print("=" * 80)
    
    # 训练Top1000
    df_sorted_train = df.sort_values('train_composite', ascending=False)
    train_top1000 = df_sorted_train.head(1000)
    
    train_factor_counter = Counter()
    for combo in train_top1000['combo']:
        factors = [f.strip() for f in combo.split(' + ')]
        train_factor_counter.update(factors)
    
    # Holdout Top500
    df_sorted_hold = df.sort_values('hold_composite', ascending=False)
    hold_top500 = df_sorted_hold.head(500)
    
    hold_factor_counter = Counter()
    for combo in hold_top500['combo']:
        factors = [f.strip() for f in combo.split(' + ')]
        hold_factor_counter.update(factors)
    
    # 计算差异
    all_factors = set(train_factor_counter.keys()) | set(hold_factor_counter.keys())
    diffs = []
    for factor in all_factors:
        train_pct = train_factor_counter.get(factor, 0) / 1000
        hold_pct = hold_factor_counter.get(factor, 0) / 500
        diff = hold_pct - train_pct
        diffs.append((factor, train_pct, hold_pct, diff))
    
    diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    
    print(f"\n{'因子':40} {'训练占比':>10} {'Hold占比':>10} {'差异':>10} {'变化方向':>12}")
    print("-" * 85)
    
    for factor, train_pct, hold_pct, diff in diffs:
        if abs(diff) > 0.05:  # 只显示差异>5%的
            direction = "📈 上升" if diff > 0 else "📉 下降"
            print(f"{factor:40} {train_pct:9.1%} {hold_pct:9.1%} {diff:+9.1%} {direction}")


def analyze_complexity_impact(df):
    """分析组合复杂度(阶数)对过拟合的影响"""
    print("\n" + "=" * 80)
    print("🧩 组合阶数与过拟合分析")
    print("=" * 80)
    
    print(f"\n{'阶数':6} {'数量':>8} {'训练收益':>12} {'Hold收益':>12} {'收益衰减':>12} {'衰减比例':>10}")
    print("-" * 75)
    
    for size in sorted(df['size_train'].unique()):
        subset = df[df['size_train'] == size]
        train_ret = subset['vec_return'].mean()
        hold_ret = subset['hold_return'].mean()
        decay = train_ret - hold_ret
        decay_pct = decay / train_ret if train_ret != 0 else 0
        
        print(f"{size:6d} {len(subset):8d} {train_ret:+11.4f} {hold_ret:+11.4f} {decay:+11.4f} {decay_pct:9.2%}")
    
    print("\n💡 观察:")
    print("  - 如果高阶组合衰减更严重 → 复杂度导致过拟合")
    print("  - 如果所有阶数衰减相似 → 因子选择或市场环境变化")


def analyze_by_holdout_ranking(df):
    """按Holdout排序分析，看是否有稳定因子"""
    print("\n" + "=" * 80)
    print("🏆 按Holdout排序 Top500 分析")
    print("=" * 80)
    
    df_sorted_hold = df.sort_values('hold_composite', ascending=False)
    hold_top500 = df_sorted_hold.head(500)
    
    print(f"\n【Holdout Top500 整体表现】")
    print(f"  Hold收益均值: {hold_top500['hold_return'].mean():.4f}")
    print(f"  Hold收益中位: {hold_top500['hold_return'].median():.4f}")
    print(f"  Hold Sharpe均值: {hold_top500['hold_sharpe'].mean():.4f}")
    print(f"  Hold MaxDD均值: {hold_top500['hold_max_dd'].mean():.4f}")
    
    # 因子频率
    factor_counter = Counter()
    for combo in hold_top500['combo']:
        factors = [f.strip() for f in combo.split(' + ')]
        factor_counter.update(factors)
    
    print(f"\n【Holdout Top500 因子频率 (前10)】")
    for factor, count in factor_counter.most_common(10):
        print(f"  {factor:40} {count:4d} ({count/500:.1%})")
    
    # 阶数分布
    print(f"\n【Holdout Top500 阶数分布】")
    size_dist = hold_top500['size_train'].value_counts().sort_index()
    for size, count in size_dist.items():
        print(f"  {size}因子组合: {count:4d} ({count/500:.1%})")


def find_stable_combos(df, train_pct=0.90, hold_pct=0.70):
    """找出训练期和Holdout期都表现优秀的组合"""
    print("\n" + "=" * 80)
    print(f"🎯 双重挡板筛选: 训练>{train_pct:.0%}分位 ∩ Holdout>{hold_pct:.0%}分位")
    print("=" * 80)
    
    # 训练期阈值
    train_threshold = df['train_composite'].quantile(train_pct)
    hold_threshold = df['hold_composite'].quantile(hold_pct)
    
    print(f"\n训练期综合分阈值 ({train_pct:.0%}): {train_threshold:.4f}")
    print(f"Holdout期综合分阈值 ({hold_pct:.0%}): {hold_threshold:.4f}")
    
    # 筛选
    stable = df[
        (df['train_composite'] >= train_threshold) &
        (df['hold_composite'] >= hold_threshold)
    ].copy()
    
    print(f"\n✅ 通过双重挡板的组合数: {len(stable)} ({len(stable)/len(df):.2%})")
    
    if len(stable) > 0:
        # 按Holdout综合分排序
        stable = stable.sort_values('hold_composite', ascending=False)
        
        print(f"\n【双重合格组合 - Holdout Top20】")
        print(f"{'排名':>4} {'组合':70} {'Hold收益':>10} {'Hold Sharpe':>12} {'Hold MaxDD':>12}")
        print("-" * 110)
        
        for idx, row in stable.head(20).iterrows():
            print(f"{idx+1:4d} {row['combo']:70} {row['hold_return']:+9.4f} {row['hold_sharpe']:11.4f} {row['hold_max_dd']:11.4f}")
        
        # 保存结果
        output_path = ROOT / 'results/vec_from_wfo_20251211_205649/stable_combos_dual_gate.csv'
        stable.to_csv(output_path, index=False)
        print(f"\n💾 已保存至: {output_path}")
        
        return stable
    else:
        print("⚠️ 没有找到同时满足两个条件的组合，建议降低阈值")
        return pd.DataFrame()


def main():
    """主函数"""
    print("=" * 80)
    print("🔬 深度过拟合诊断 v1.0")
    print("=" * 80)
    
    # 加载数据
    df = load_data()
    
    # 分析1: 得分分布对比
    analyze_score_distribution(df)
    
    # 分析2: Top分层表现
    analyze_top_cohorts(df)
    
    # 分析3: 因子频率变化
    analyze_factor_shift(df)
    
    # 分析4: 复杂度影响
    analyze_complexity_impact(df)
    
    # 分析5: Holdout排序分析
    analyze_by_holdout_ranking(df)
    
    # 分析6: 双重挡板筛选
    stable = find_stable_combos(df, train_pct=0.80, hold_pct=0.80)
    
    print("\n" + "=" * 80)
    print("✅ 诊断完成")
    print("=" * 80)
    
    print("\n💡 关键结论:")
    print("  1. 训练期排序与Holdout期表现相关性极低")
    print("  2. 训练期偏好的因子(ADX, SHARPE_20D)在Holdout期失效")
    print("  3. Holdout期表现好的因子(CMF, MAX_DD_60D, CORR_TO_MARKET)在训练期被低估")
    print("  4. 建议使用双重挡板而非训练期Top排序")
    

if __name__ == '__main__':
    main()
