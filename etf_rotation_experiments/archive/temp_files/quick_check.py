#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_parquet('/Users/zhangshenshen/深度量化0927/etf_rotation_experiments/ml_ranking/data/training_dataset.parquet')

print("="*80)
print("数据集快速检查")
print("="*80)
print(f"\n样本数: {len(df)}")
print(f"特征数: {len(df.columns)-1}")
print(f"列名: {df.columns.tolist()[:10]}...")

label = 'oos_compound_sharpe'
if label in df.columns:
    print(f"\n标签统计:")
    print(df[label].describe())
else:
    print(f"\n标签列 '{label}' 不存在!")
    print(f"可用列: {df.columns.tolist()}")

print(f"\n缺失值检查:")
missing = df.isnull().mean() * 100
print(f"总体缺失率: {missing.mean():.2f}%")
high_missing = missing[missing > 5]
if len(high_missing) > 0:
    print(f"高缺失特征 (>5%):")
    for col, rate in high_missing.head().items():
        print(f"  {col}: {rate:.2f}%")

print(f"\n前3行样本:")
print(df.head(3))
