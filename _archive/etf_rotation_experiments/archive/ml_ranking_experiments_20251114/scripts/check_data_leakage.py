#!/usr/bin/env python3
"""快速验证数据泄露"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("🔍 数据泄露检查")
print("="*80)

# 加载数据集
dataset_path = Path("/Users/zhangshenshen/深度量化0927/etf_rotation_experiments/ml_ranking/data/training_dataset.parquet")
df = pd.read_parquet(dataset_path)

label = 'oos_compound_sharpe'

print(f"\n数据集大小: {len(df)} rows × {len(df.columns)} columns")
print(f"标签列: {label}")

# 检查1: 标签是否在特征中
feature_cols = [c for c in df.columns if c != label]
print(f"\n[检查1] 标签是否在特征中?")
print(f"   结果: {label in feature_cols}")
if label in feature_cols:
    print("   🚨 严重泄露!")

# 检查2: 数学泄露 - 能否从特征重构标签
print(f"\n[检查2] 能否从 compound_mean/std 重构标签?")
if 'oos_compound_mean' in df.columns and 'oos_compound_std' in df.columns:
    # 重构
    reconstructed = df['oos_compound_mean'] / df['oos_compound_std']
    actual = df[label]
    
    # 计算相关性
    valid_mask = ~(reconstructed.isna() | actual.isna())
    corr = np.corrcoef(reconstructed[valid_mask], actual[valid_mask])[0,1]
    
    # 计算差异
    diff = (reconstructed - actual).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"   相关系数: {corr:.8f}")
    print(f"   平均差异: {mean_diff:.8f}")
    print(f"   最大差异: {max_diff:.8f}")
    
    if corr > 0.9999:
        print("   🚨 确认数学泄露! oos_compound_sharpe = oos_compound_mean / oos_compound_std")
    elif corr > 0.95:
        print("   ⚠️  高度相关，可能存在泄露")
    else:
        print("   ✅ 相关性正常")
else:
    print("   ⚠️  未找到 oos_compound_mean 或 oos_compound_std")

# 检查3: 找出所有高度相关的特征
print(f"\n[检查3] 与标签高度相关的特征 (|corr| > 0.90):")
correlations = {}
for col in feature_cols:
    if col == label:
        continue
    if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
        try:
            corr = df[col].corr(df[label])
            if abs(corr) > 0.90:
                correlations[col] = corr
        except:
            pass

if correlations:
    for col, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        emoji = "🚨" if abs(corr) > 0.95 else "⚠️"
        print(f"   {emoji} {col}: {corr:.6f}")
else:
    print("   ✅ 没有发现高度相关特征")

# 检查4: 特征名称模式检查
print(f"\n[检查4] 可疑特征名称:")
suspicious_patterns = ['compound', 'sharpe', 'oos_']
suspicious_features = []
for col in feature_cols:
    col_lower = col.lower()
    if any(pattern in col_lower for pattern in suspicious_patterns):
        if 'mean' in col_lower or 'std' in col_lower or col_lower == label.lower():
            suspicious_features.append(col)

if suspicious_features:
    print("   可疑特征 (包含 compound/sharpe/oos + mean/std):")
    for feat in suspicious_features[:15]:
        print(f"      - {feat}")
    if len(suspicious_features) > 15:
        print(f"      ... 及其他 {len(suspicious_features)-15} 个")
else:
    print("   ✅ 未发现明显可疑命名")

# 总结
print("\n" + "="*80)
print("📊 总结")
print("="*80)

if 'oos_compound_mean' in df.columns and 'oos_compound_std' in df.columns:
    reconstructed = df['oos_compound_mean'] / df['oos_compound_std']
    actual = df[label]
    valid_mask = ~(reconstructed.isna() | actual.isna())
    corr = np.corrcoef(reconstructed[valid_mask], actual[valid_mask])[0,1]
    
    if corr > 0.9999:
        print("\n🚨 严重数据泄露确认!")
        print(f"\n原因: oos_compound_sharpe = oos_compound_mean / oos_compound_std")
        print(f"证据: 重构相关性 = {corr:.8f} (接近1.0)")
        print(f"\n这解释了为什么:")
        print(f"  - Spearman达到0.993 (模型直接学到了除法关系)")
        print(f"  - Top50重叠率96% (基本完美预测)")
        print(f"  - NDCG@50接近1.0 (排序几乎完全正确)")
        
        print(f"\n✅ 修复方案:")
        print(f"  1. 从特征中移除: oos_compound_mean, oos_compound_std")
        print(f"  2. 或者使用不同的标签 (如 oos_sharpe_true)")
        print(f"  3. 重新训练模型")
        print(f"  4. 预期修复后性能: Spearman 0.65-0.75, Top50 30-45%")
    else:
        print("\n✅ 未检测到明显数学泄露")
        print(f"但Spearman=0.99仍然异常，需进一步调查")

print("\n" + "="*80)
