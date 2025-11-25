#!/usr/bin/env python3
"""
Phase 1.1: WFO窗口配置诊断

核心问题：WFO窗口设置是否导致过拟合？
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def analyze_wfo_windows(run_dir: Path):
    """分析WFO窗口配置"""
    
    print("="*60)
    print("🔍 WFO窗口配置诊断")
    print("="*60)
    
    # 读取WFO结果
    all_combos = pd.read_parquet(run_dir / "all_combos.parquet")
    
    # 分析窗口信息
    print("\n📊 窗口统计:")
    print(f"   - 总组合数: {len(all_combos)}")
    
    # 从oos_ic_list推断窗口数
    if 'oos_ic_list' in all_combos.columns:
        sample_oos = all_combos['oos_ic_list'].iloc[0]
        n_windows = len(sample_oos) if isinstance(sample_oos, (list, np.ndarray)) else 0
        print(f"   - WFO窗口数: {n_windows}")
        
        # 分析IC分布
        all_oos_ic = []
        for ic_list in all_combos['oos_ic_list']:
            if isinstance(ic_list, (list, np.ndarray)):
                all_oos_ic.extend(ic_list)
        
        print(f"\n📈 样本外IC统计:")
        print(f"   - 总样本数: {len(all_oos_ic)}")
        print(f"   - 均值: {np.mean(all_oos_ic):.6f}")
        print(f"   - 标准差: {np.std(all_oos_ic):.6f}")
        print(f"   - 中位数: {np.median(all_oos_ic):.6f}")
        print(f"   - 最小值: {np.min(all_oos_ic):.6f}")
        print(f"   - 最大值: {np.max(all_oos_ic):.6f}")
        
        # 检查异常值
        q1 = np.percentile(all_oos_ic, 25)
        q3 = np.percentile(all_oos_ic, 75)
        iqr = q3 - q1
        outliers_low = sum(1 for x in all_oos_ic if x < q1 - 1.5*iqr)
        outliers_high = sum(1 for x in all_oos_ic if x > q3 + 1.5*iqr)
        
        print(f"\n⚠️ 异常值检测:")
        print(f"   - 低端异常值 (<Q1-1.5IQR): {outliers_low} ({outliers_low/len(all_oos_ic)*100:.2f}%)")
        print(f"   - 高端异常值 (>Q3+1.5IQR): {outliers_high} ({outliers_high/len(all_oos_ic)*100:.2f}%)")
    
    # 分析mean_oos_ic分布
    print(f"\n📊 mean_oos_ic分布:")
    ic_mean = all_combos['mean_oos_ic'].mean()
    ic_std = all_combos['mean_oos_ic'].std()
    ic_median = all_combos['mean_oos_ic'].median()
    
    print(f"   - 均值: {ic_mean:.6f}")
    print(f"   - 标准差: {ic_std:.6f}")
    print(f"   - 中位数: {ic_median:.6f}")
    print(f"   - 范围: [{all_combos['mean_oos_ic'].min():.6f}, {all_combos['mean_oos_ic'].max():.6f}]")
    
    # 检查IC为负的比例
    negative_ratio = (all_combos['mean_oos_ic'] < 0).sum() / len(all_combos)
    print(f"   - 负IC比例: {negative_ratio*100:.2f}%")
    
    # 检查IC接近0的比例
    near_zero = ((all_combos['mean_oos_ic'].abs() < 0.01).sum() / len(all_combos))
    print(f"   - 接近0的IC (|IC|<0.01): {near_zero*100:.2f}%")
    
    # 分析stability_score
    if 'stability_score' in all_combos.columns:
        print(f"\n📊 stability_score分布:")
        print(f"   - 均值: {all_combos['stability_score'].mean():.6f}")
        print(f"   - 标准差: {all_combos['stability_score'].std():.6f}")
        print(f"   - 中位数: {all_combos['stability_score'].median():.6f}")
        print(f"   - 范围: [{all_combos['stability_score'].min():.6f}, {all_combos['stability_score'].max():.6f}]")
    
    # 诊断结论
    print("\n" + "="*60)
    print("🔬 诊断结论")
    print("="*60)
    
    issues = []
    
    # 检查1: IC分布过于集中
    if ic_std < 0.01:
        issues.append("⚠️ IC标准差过小 - 组合间区分度不足")
    
    # 检查2: 负IC比例过高
    if negative_ratio > 0.4:
        issues.append(f"⚠️ 负IC比例过高({negative_ratio*100:.1f}%) - 可能存在数据质量问题")
    
    # 检查3: 样本数量过多
    if len(all_combos) > 5000:
        issues.append(f"⚠️ 组合数量过多({len(all_combos)}) - 存在多重检验问题")
    
    # 检查4: 窗口数量
    if n_windows < 10:
        issues.append(f"⚠️ WFO窗口过少({n_windows}) - 验证不充分")
    elif n_windows > 30:
        issues.append(f"⚠️ WFO窗口过多({n_windows}) - 可能过拟合")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ 窗口配置基本合理")
    
    # 建议
    print("\n💡 改进建议:")
    if len(all_combos) > 5000:
        print("   1. 减少组合空间到500-1000个（预筛选）")
    print("   2. 重新评估窗口长度（IS/OOS比例）")
    print("   3. 考虑使用滑动窗口而非扩展窗口")
    print("   4. 增加Bonferroni校正或FDR控制")
    
    return {
        'n_combos': len(all_combos),
        'n_windows': n_windows,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'negative_ratio': negative_ratio,
        'issues': issues
    }


def main():
    run_dir = Path("etf_rotation_experiments/results/run_20251113_145102")
    
    if not run_dir.exists():
        print(f"❌ WFO结果目录不存在: {run_dir}")
        return
    
    results = analyze_wfo_windows(run_dir)
    
    # 保存诊断结果
    output_dir = run_dir / "diagnosis"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "window_diagnosis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ 诊断结果已保存: {output_dir}/window_diagnosis.json")


if __name__ == "__main__":
    main()
