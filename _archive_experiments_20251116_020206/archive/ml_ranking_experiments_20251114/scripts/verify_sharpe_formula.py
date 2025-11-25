#!/usr/bin/env python3
"""验证夏普比率的数学定义是否与数据一致"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("🔍 验证夏普比率数学定义")
print("="*80)

# 从WFO结果中直接验证
wfo_file = "/Users/zhangshenshen/深度量化0927/etf_rotation_experiments/results/run_20251113_194451/ranking_oos_compound_sharpe_top1000.parquet"

if Path(wfo_file).exists():
    df = pd.read_parquet(wfo_file)
    print(f"数据集: {len(df)} rows")
    
    # 检查关键列
    required_cols = ['oos_compound_mean', 'oos_compound_std', 'oos_compound_sharpe']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"❌ 缺少列: {missing}")
        print("可用列:", df.columns.tolist()[:10], "...")
    else:
        print("✅ 找到所有必需列")
        
        # 验证数学关系
        calculated_sharpe = df['oos_compound_mean'] / df['oos_compound_std']
        actual_sharpe = df['oos_compound_sharpe']
        
        # 处理NaN和Inf
        valid_mask = (
            ~calculated_sharpe.isna() & 
            ~actual_sharpe.isna() & 
            np.isfinite(calculated_sharpe) & 
            np.isfinite(actual_sharpe) &
            (df['oos_compound_std'] != 0)
        )
        
        if valid_mask.sum() > 0:
            calc_valid = calculated_sharpe[valid_mask]
            actual_valid = actual_sharpe[valid_mask]
            
            # 计算统计量
            correlation = np.corrcoef(calc_valid, actual_valid)[0,1]
            
            diff = (calc_valid - actual_valid).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            median_diff = diff.median()
            
            print(f"\n📊 统计结果 (有效样本: {valid_mask.sum()})")
            print(f"   相关系数: {correlation:.8f}")
            print(f"   平均差异: {mean_diff:.8f}")
            print(f"   中位差异: {median_diff:.8f}")
            print(f"   最大差异: {max_diff:.8f}")
            
            # 检查是否是完美关系
            if correlation > 0.9999 and mean_diff < 1e-10:
                print("\n✅ 确认: oos_compound_sharpe = oos_compound_mean / oos_compound_std")
                print("   这是标准的夏普比率定义，数学上完全正确")
            elif correlation > 0.99:
                print("\n✅ 接近完美关系，可能有小的数值误差")
            else:
                print(f"\n❌ 关系不完美，correlation = {correlation:.6f}")
            
            # 显示一些例子
            print(f"\n📝 示例对比 (前5行):")
            sample = df[valid_mask].head()
            for i, (idx, row) in enumerate(sample.iterrows()):
                calc = row['oos_compound_mean'] / row['oos_compound_std']
                actual = row['oos_compound_sharpe']
                print(f"   [{i+1}] {calc:.6f} vs {actual:.6f} (diff: {abs(calc-actual):.8f})")
                
        else:
            print("❌ 没有有效样本进行验证")
            print("可能原因: 所有std都是0，或存在大量NaN")
            
else:
    print("❌ WFO结果文件不存在")

print("\n" + "="*80)
print("📚 量化金融中的夏普比率")
print("="*80)
print("""
夏普比率 (Sharpe Ratio) 的标准定义:
   Sharpe = (Portfolio_Return - Risk_Free_Rate) / Portfolio_Volatility

在实践中的常见变形:
   1. 忽略无风险利率: Sharpe ≈ mean_return / std_return
   2. 使用超额收益: Sharpe = mean(excess_return) / std(excess_return)
   3. 年化形式: Annual_Sharpe = Sharpe * sqrt(252)

在我们的case中:
   - oos_compound_mean: 样本外复合收益的均值
   - oos_compound_std: 样本外复合收益的标准差  
   - oos_compound_sharpe: 复合夏普比率 = mean / std

这个公式是量化金融的基础概念，关系完美(correlation=1.0)是合理的！
""")

print("\n🤔 关于'过拟合'的担忧:")
print("""
这里不是过拟合，而是特征工程的问题:
   ❌ 错误: 将计算目标的组成部分作为特征
   ✅ 正确: 使用独立的、有预测能力的特征
   
类比: 如果要预测 BMI = weight/height²，
     不应该把 weight 和 height 直接作为特征，
     而应该用其他健康指标来预测 BMI
""")