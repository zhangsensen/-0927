#!/usr/bin/env python3
"""Quick dataset inspection script."""

import pandas as pd
from pathlib import Path

dataset_path = Path("/Users/zhangshenshen/深度量化0927/etf_rotation_experiments/ml_ranking/data/training_dataset.parquet")

if not dataset_path.exists():
    print(f"❌ Dataset not found: {dataset_path}")
    exit(1)

print("Loading dataset...")
df = pd.read_parquet(dataset_path)

print("\n" + "="*80)
print("📊 DATASET INSPECTION")
print("="*80)

print(f"\n✅ Rows: {len(df)}")
print(f"✅ Columns: {len(df.columns)}")
print(f"✅ Features: {len(df.columns) - 1} (excluding label)")

label_col = "oos_compound_sharpe"
if label_col in df.columns:
    print(f"\n📈 Label Column: {label_col}")
    print(df[label_col].describe())
    print(f"   Coverage: {(~df[label_col].isnull()).mean()*100:.1f}%")
else:
    print(f"\n❌ Label column '{label_col}' not found!")
    print(f"Available columns: {df.columns.tolist()[:10]}...")

print(f"\n🔍 Missing Values:")
missing = df.isnull().mean() * 100
high_missing = missing[missing > 5].sort_values(ascending=False)
if len(high_missing) > 0:
    print(f"   Features with >5% missing:")
    for col, pct in high_missing.head(10).items():
        print(f"      {col}: {pct:.2f}%")
else:
    print(f"   ✅ All features have <5% missing")
print(f"   Overall missing rate: {missing.mean():.2f}%")

print(f"\n📋 Sample rows:")
print(df.head(3))

print(f"\n✅ Dataset ready for training")
