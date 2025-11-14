#!/usr/bin/env python3
"""
Ensemble策略回测 - 综合IC和Calibrator排序
"""
import pandas as pd
from pathlib import Path

# 路径配置
exp_root = Path("/Users/zhangshenshen/深度量化0927/etf_rotation_experiments")
ranking_ic = exp_root / "results/run_20251113_145102/ranking_blends/ranking_baseline.parquet"
ranking_cal = exp_root / "results/run_20251113_145102/ranking_blends/ranking_lightgbm.parquet"
output_dir = exp_root / "results/run_20251113_145102/ensemble_rankings"
output_dir.mkdir(exist_ok=True)

print("=" * 100)
print("🔄 生成Ensemble排序文件")
print("=" * 100)

# 读取数据
df_ic = pd.read_parquet(ranking_ic)
df_cal = pd.read_parquet(ranking_cal)

print(f"\n读取数据:")
print(f"  IC排序: {len(df_ic)} 组合")
print(f"  Calibrator排序: {len(df_cal)} 组合")

# 策略1: IC Top1000 ∩ Calibrator Top1000 的交集
print("\n【策略1】IC Top1000 ∩ Calibrator Top1000 交集")
print("-" * 100)

ic_top1000 = set(df_ic.head(1000)['combo'])
cal_top1000 = set(df_cal.head(1000)['combo'])
intersection_1000 = ic_top1000 & cal_top1000

print(f"  IC Top1000: {len(ic_top1000)} 组合")
print(f"  Calibrator Top1000: {len(cal_top1000)} 组合")
print(f"  交集: {len(intersection_1000)} 组合")
print(f"  Overlap率: {len(intersection_1000)/1000:.1%}")

# 生成交集ranking文件 - 按IC和Calibrator分数的平均值排序
df_intersection = df_ic[df_ic['combo'].isin(intersection_1000)].copy()
df_cal_scores = df_cal[df_cal['combo'].isin(intersection_1000)][['combo', 'rank_score']].rename(
    columns={'rank_score': 'cal_rank_score'}
)
df_intersection = df_intersection.merge(df_cal_scores, on='combo')
df_intersection['ensemble_score'] = (df_intersection['rank_score'] + df_intersection['cal_rank_score']) / 2
df_intersection = df_intersection.sort_values('ensemble_score', ascending=False).reset_index(drop=True)

output_path_1 = output_dir / "ranking_intersection_top1000.parquet"
df_intersection.to_parquet(output_path_1, index=False)
print(f"  ✅ 已保存: {output_path_1.name}")

# 策略2: IC Top500 + Calibrator Top500 的并集
print("\n【策略2】IC Top500 + Calibrator Top500 并集(去重)")
print("-" * 100)

ic_top500 = set(df_ic.head(500)['combo'])
cal_top500 = set(df_cal.head(500)['combo'])
union_500 = ic_top500 | cal_top500

print(f"  IC Top500: {len(ic_top500)} 组合")
print(f"  Calibrator Top500: {len(cal_top500)} 组合")
print(f"  并集: {len(union_500)} 组合")
print(f"  Overlap: {len(ic_top500 & cal_top500)} 组合")

# 生成并集ranking文件 - 包含所有在IC或Calibrator Top500中的组合
df_union = df_ic[df_ic['combo'].isin(union_500)].copy()
df_cal_union = df_cal[df_cal['combo'].isin(union_500)][['combo', 'rank_score']].rename(
    columns={'rank_score': 'cal_rank_score'}
)
df_union = df_union.merge(df_cal_union, on='combo', how='left')
df_union['cal_rank_score'] = df_union['cal_rank_score'].fillna(0)  # 只在IC中的组合，cal分数为0
df_union['ensemble_score'] = (df_union['rank_score'] + df_union['cal_rank_score']) / 2
df_union = df_union.sort_values('ensemble_score', ascending=False).reset_index(drop=True)

output_path_2 = output_dir / "ranking_union_top500.parquet"
df_union.to_parquet(output_path_2, index=False)
print(f"  ✅ 已保存: {output_path_2.name}")

# 策略3: 加权ensemble (IC 50% + Calibrator 50%) - 全部组合
print("\n【策略3】全局加权Ensemble (IC 50% + Calibrator 50%)")
print("-" * 100)

df_ensemble = df_ic[['combo', 'rank_score']].copy()
df_cal_all = df_cal[['combo', 'rank_score']].rename(columns={'rank_score': 'cal_rank_score'})
df_ensemble = df_ensemble.merge(df_cal_all, on='combo')

# 归一化分数到0-1
df_ensemble['ic_norm'] = (df_ensemble['rank_score'] - df_ensemble['rank_score'].min()) / \
                          (df_ensemble['rank_score'].max() - df_ensemble['rank_score'].min())
df_ensemble['cal_norm'] = (df_ensemble['cal_rank_score'] - df_ensemble['cal_rank_score'].min()) / \
                           (df_ensemble['cal_rank_score'].max() - df_ensemble['cal_rank_score'].min())
df_ensemble['ensemble_score'] = 0.5 * df_ensemble['ic_norm'] + 0.5 * df_ensemble['cal_norm']
df_ensemble = df_ensemble.sort_values('ensemble_score', ascending=False).reset_index(drop=True)

# 保留原始rank_score供回测使用
df_ensemble['rank_score'] = df_ensemble['ensemble_score']

# 保存Top1000
df_ensemble_top1000 = df_ensemble.head(1000).copy()
# 需要merge回原始特征
df_ensemble_top1000 = df_ensemble_top1000[['combo', 'rank_score']].merge(
    df_ic[['combo', 'mean_oos_ic', 'stability_score', 'best_rebalance_freq']],
    on='combo'
)

output_path_3 = output_dir / "ranking_ensemble_50_50_top1000.parquet"
df_ensemble_top1000.to_parquet(output_path_3, index=False)
print(f"  ✅ 已保存: {output_path_3.name} ({len(df_ensemble_top1000)} 组合)")

print("\n" + "=" * 100)
print("✅ Ensemble排序文件生成完成!")
print("=" * 100)
print(f"\n输出目录: {output_dir}")
print(f"  1. ranking_intersection_top1000.parquet  ({len(df_intersection)} 组合)")
print(f"  2. ranking_union_top500.parquet          ({len(df_union)} 组合)")
print(f"  3. ranking_ensemble_50_50_top1000.parquet ({len(df_ensemble_top1000)} 组合)")
print("\n下一步: 对这3个ensemble策略运行回测")
