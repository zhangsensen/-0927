#!/usr/bin/env python3
"""
Calibrator诊断分析 - 揭示Top1相同但中位数提升的真相
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
exp_root = Path("/Users/zhangshenshen/深度量化0927/etf_rotation_experiments")
calibrator_path = exp_root.parent / "etf_rotation_experiments/results/calibrator_gbdt_full.joblib"
ranking_ic = exp_root / "results/run_20251113_145102/ranking_blends/ranking_baseline.parquet"
ranking_cal = exp_root / "results/run_20251113_145102/ranking_blends/ranking_lightgbm.parquet"
output_dir = exp_root / "results/run_20251113_145102/calibrator_diagnosis"
output_dir.mkdir(exist_ok=True)

print("=" * 100)
print("🔬 Calibrator 诊断分析")
print("=" * 100)

# 1. 特征重要性分析
print("\n【1】特征重要性分析")
print("-" * 100)

calibrator = joblib.load(calibrator_path)

# calibrator是dict, 包含model和metadata
if isinstance(calibrator, dict):
    model = calibrator.get('model')
    feature_names = calibrator.get('feature_names', [])
else:
    model = calibrator
    feature_names = []

if model and hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 重要特征:")
    print(feat_imp.head(10).to_string(index=False))
    
    # 检查mean_oos_ic的重要性
    ic_features = feat_imp[feat_imp['feature'].str.contains('ic', case=False)]
    total_ic_importance = ic_features['importance'].sum()
    print(f"\n⚠️  所有IC相关特征的总重要性: {total_ic_importance:.1%}")
    
    if total_ic_importance > 0.7:
        print("❌ 警告: IC特征占主导地位(>70%),calibrator可能只是IC的变种!")
    elif total_ic_importance > 0.5:
        print("⚠️  注意: IC特征占比较高(>50%),需要增加其他特征权重")
    else:
        print("✅ IC特征权重合理(<50%),calibrator学到了IC之外的模式")
    
    # 保存特征重要性图
    plt.figure(figsize=(10, 6))
    feat_imp.head(15).plot(x='feature', y='importance', kind='barh', figsize=(10, 8))
    plt.xlabel('重要性')
    plt.title('Calibrator特征重要性 Top15')
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    print(f"\n✅ 特征重要性图已保存: {output_dir / 'feature_importance.png'}")

# 2. 排序相关性分析
print("\n【2】排序相关性分析")
print("-" * 100)

df_ic = pd.read_parquet(ranking_ic)
df_cal = pd.read_parquet(ranking_cal)

# 确保combo列对齐
merged = df_ic[['combo', 'rank_score']].rename(columns={'rank_score': 'ic_score'}).merge(
    df_cal[['combo', 'rank_score']].rename(columns={'rank_score': 'cal_score'}),
    on='combo'
)

# 计算相关性
spearman_corr, _ = spearmanr(merged['ic_score'], merged['cal_score'])
kendall_corr, _ = kendalltau(merged['ic_score'], merged['cal_score'])

print(f"\n全局排序相关性:")
print(f"  Spearman相关系数: {spearman_corr:.4f}")
print(f"  Kendall相关系数:  {kendall_corr:.4f}")

if spearman_corr > 0.95:
    print("❌ 警告: 排序高度相关(>0.95),calibrator几乎等同于IC排序!")
elif spearman_corr > 0.85:
    print("⚠️  注意: 排序相关性较高(>0.85),calibrator改变有限")
else:
    print("✅ 排序有显著差异(<0.85),calibrator提供了新视角")

# 不同TopK的overlap分析
topk_list = [10, 50, 100, 500, 1000, 3000]
overlap_results = []

for topk in topk_list:
    ic_top = set(df_ic.head(topk)['combo'])
    cal_top = set(df_cal.head(topk)['combo'])
    overlap = len(ic_top & cal_top)
    overlap_rate = overlap / topk
    
    overlap_results.append({
        'topk': topk,
        'overlap_count': overlap,
        'overlap_rate': overlap_rate,
        'unique_to_ic': len(ic_top - cal_top),
        'unique_to_cal': len(cal_top - ic_top)
    })

overlap_df = pd.DataFrame(overlap_results)
print(f"\n不同TopK的组合overlap:")
print(overlap_df.to_string(index=False))

# 3. Top1详细分析
print("\n【3】Top1组合分析")
print("-" * 100)

top1_ic = df_ic.iloc[0]
top1_cal = df_cal.iloc[0]

print(f"\nIC排序Top1:")
print(f"  组合: {top1_ic['combo']}")
print(f"  mean_oos_ic: {top1_ic.get('mean_oos_ic', 'N/A')}")
print(f"  stability_score: {top1_ic.get('stability_score', 'N/A')}")

print(f"\n校准排序Top1:")
print(f"  组合: {top1_cal['combo']}")
print(f"  calibrated_sharpe_pred: {top1_cal.get('calibrated_sharpe_pred', 'N/A')}")
print(f"  mean_oos_ic: {top1_cal.get('mean_oos_ic', 'N/A')}")

if top1_ic['combo'] == top1_cal['combo']:
    print(f"\n✅ Top1组合完全相同: {top1_ic['combo']}")
    print(f"   这说明两种方法对最优组合达成共识")
else:
    print(f"\n❌ Top1组合不同!")
    print(f"   IC选择: {top1_ic['combo']}")
    print(f"   校准选择: {top1_cal['combo']}")

# 4. 散点图: IC score vs Calibrated score
print("\n【4】生成可视化图表")
print("-" * 100)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 4.1 全局scatter
ax = axes[0, 0]
ax.scatter(merged['ic_score'], merged['cal_score'], alpha=0.3, s=10)
ax.plot([merged['ic_score'].min(), merged['ic_score'].max()], 
        [merged['ic_score'].min(), merged['ic_score'].max()], 
        'r--', label='y=x')
ax.set_xlabel('IC Score')
ax.set_ylabel('Calibrated Score')
ax.set_title(f'全局排序对比 (Spearman={spearman_corr:.3f})')
ax.legend()
ax.grid(alpha=0.3)

# 4.2 Top1000 scatter
ax = axes[0, 1]
top1000_merged = merged.sort_values('ic_score', ascending=False).head(1000)
ax.scatter(top1000_merged['ic_score'], top1000_merged['cal_score'], alpha=0.5, s=20)
ax.set_xlabel('IC Score')
ax.set_ylabel('Calibrated Score')
ax.set_title('Top1000排序对比')
ax.grid(alpha=0.3)

# 4.3 Overlap rate vs TopK
ax = axes[1, 0]
ax.plot(overlap_df['topk'], overlap_df['overlap_rate'], marker='o', linewidth=2)
ax.set_xlabel('TopK')
ax.set_ylabel('Overlap Rate')
ax.set_title('组合Overlap率 vs TopK规模')
ax.axhline(y=0.8, color='r', linestyle='--', label='80%阈值')
ax.legend()
ax.grid(alpha=0.3)

# 4.4 Rank difference histogram
ax = axes[1, 1]
merged['ic_rank'] = merged['ic_score'].rank(ascending=False)
merged['cal_rank'] = merged['cal_score'].rank(ascending=False)
merged['rank_diff'] = merged['cal_rank'] - merged['ic_rank']
ax.hist(merged['rank_diff'], bins=100, alpha=0.7, edgecolor='black')
ax.set_xlabel('排名变化 (Calibrated - IC)')
ax.set_ylabel('组合数')
ax.set_title('排名变化分布')
ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "ranking_comparison.png", dpi=150)
print(f"✅ 排序对比图已保存: {output_dir / 'ranking_comparison.png'}")

# 5. 生成诊断报告
print("\n【5】生成诊断报告")
print("-" * 100)

report_path = output_dir / "diagnosis_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Calibrator诊断报告\n\n")
    f.write(f"**生成时间**: {pd.Timestamp.now()}\n\n")
    
    f.write("## 1. 特征重要性\n\n")
    f.write(feat_imp.head(15).to_markdown(index=False))
    f.write(f"\n\n**IC相关特征总重要性**: {total_ic_importance:.1%}\n\n")
    
    f.write("## 2. 排序相关性\n\n")
    f.write(f"- Spearman相关系数: {spearman_corr:.4f}\n")
    f.write(f"- Kendall相关系数: {kendall_corr:.4f}\n\n")
    
    f.write("## 3. TopK Overlap分析\n\n")
    f.write(overlap_df.to_markdown(index=False))
    f.write("\n\n")
    
    f.write("## 4. 核心结论\n\n")
    
    if total_ic_importance > 0.7 and spearman_corr > 0.95:
        f.write("❌ **Calibrator过度依赖IC特征,排序高度相关**\n\n")
        f.write("**建议行动**:\n")
        f.write("1. 重新训练calibrator,移除mean_oos_ic特征\n")
        f.write("2. 添加新特征: 换手率、最大回撤、持仓集中度等\n")
        f.write("3. 或者直接使用IC排序,不使用calibrator\n")
    elif overlap_df[overlap_df['topk']==3000]['overlap_rate'].values[0] > 0.95:
        f.write("⚠️  **Calibrator与IC排序高度重叠,但可能在细节上有差异**\n\n")
        f.write("**建议行动**:\n")
        f.write("1. 分析Top100-1000之间的排序差异\n")
        f.write("2. 如果中位数提升显著,calibrator仍有价值\n")
        f.write("3. 考虑将calibrator用于组合池筛选而非单一选择\n")
    else:
        f.write("✅ **Calibrator提供了与IC不同的排序视角**\n\n")
        f.write("**建议行动**:\n")
        f.write("1. 继续使用calibrator进行组合筛选\n")
        f.write("2. 定期重新训练以适应市场变化\n")
        f.write("3. 建立ensemble策略综合IC和calibrator排序\n")

print(f"✅ 诊断报告已保存: {report_path}")

print("\n" + "=" * 100)
print("🎯 诊断完成!")
print("=" * 100)
print(f"\n输出目录: {output_dir}")
print(f"  - feature_importance.png")
print(f"  - ranking_comparison.png")
print(f"  - diagnosis_report.md")
print("\n请查看报告并根据建议决定下一步行动。")
