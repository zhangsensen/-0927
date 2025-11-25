#!/usr/bin/env python3
"""
手动生成TopK对比报告 - 修复排序问题
"""
import pandas as pd
import json
from pathlib import Path

# 定义结果路径
base_dir = Path("/Users/zhangshenshen/深度量化0927")
exp_dir = base_dir / "etf_rotation_experiments"
results_dir = exp_dir / "results_combo_wfo"

# 定义回测结果
backtests = {
    'Top100': {
        'ic': results_dir / "20251113_145102_20251113_151619",
        'calibrated': results_dir / "20251113_145102_20251113_151823",
    },
    'Top1000': {
        'ic': results_dir / "20251113_145102_20251113_152903",
        'calibrated': results_dir / "20251113_145102_20251113_152905",
    },
    'Top3000': {
        'ic': results_dir / "20251113_145102_20251113_152907",
        'calibrated': results_dir / "20251113_145102_20251113_152909",
    },
}

results = {}

for topk_name, paths in backtests.items():
    results[topk_name] = {}
    
    for ranking_type, path in paths.items():
        # 读取CSV
        csv_files = list(path.glob("top*.csv"))
        if not csv_files:
            continue
        
        csv_file = csv_files[0]
        df = pd.read_csv(csv_file)
        
        # 按sharpe_net降序排序获取真正的Top1
        df_sorted = df.sort_values('sharpe_net', ascending=False).reset_index(drop=True)
        top1 = df_sorted.iloc[0]
        
        # 读取SUMMARY
        json_files = list(path.glob("SUMMARY*.json"))
        if json_files:
            with open(json_files[0]) as f:
                summary = json.load(f)
        else:
            summary = {}
        
        count = summary.get('count', len(df))
        
        results[topk_name][ranking_type] = {
            'count': count,
            'top1_annual_net': top1['annual_ret_net'],
            'top1_sharpe_net': top1['sharpe_net'],
            'mean_annual_net': summary.get('mean_annual_net', df['annual_ret_net'].mean()),
            'median_annual_net': summary.get('median_annual_net', df['annual_ret_net'].median()),
            'mean_sharpe_net': summary.get('mean_sharpe_net', df['sharpe_net'].mean()),
            'median_sharpe_net': summary.get('median_sharpe_net', df['sharpe_net'].median()),
        }

# 生成报告
print("=" * 100)
print("📊 校准器大规模验证报告 - TopK对比分析")
print("=" * 100)
print(f"WFO Run: 20251113_145102")
print(f"总组合数: 12,597")
print(f"回测滑点: 2.0 bps")
print()

for topk_name in ['Top100', 'Top1000', 'Top3000']:
    if topk_name not in results:
        continue
    
    data = results[topk_name]
    if 'ic' not in data or 'calibrated' not in data:
        continue
    
    ic = data['ic']
    cal = data['calibrated']
    count = ic['count']
    pct = count / 12597 * 100
    
    print(f"\n{'=' * 100}")
    print(f"📈 {topk_name} ({count:,} 组合, {pct:.2f}% 样本)")
    print(f"{'=' * 100}")
    
    # Top1对比
    print(f"\n**Top1组合性能** (按Sharpe降序选择):")
    print(f"  年化收益(净):  IC={ic['top1_annual_net']:.2%}  →  校准={cal['top1_annual_net']:.2%}  "
          f"(+{cal['top1_annual_net'] - ic['top1_annual_net']:.2%}, "
          f"{(cal['top1_annual_net'] / ic['top1_annual_net'] - 1) * 100:+.1f}%)")
    print(f"  Sharpe比率(净): IC={ic['top1_sharpe_net']:.3f}  →  校准={cal['top1_sharpe_net']:.3f}  "
          f"(+{cal['top1_sharpe_net'] - ic['top1_sharpe_net']:.3f}, "
          f"{(cal['top1_sharpe_net'] / ic['top1_sharpe_net'] - 1) * 100:+.1f}%)")
    
    # 整体质量
    print(f"\n**整体质量提升**:")
    print(f"  均值年化(净):     IC={ic['mean_annual_net']:.2%}  →  校准={cal['mean_annual_net']:.2%}  "
          f"(+{cal['mean_annual_net'] - ic['mean_annual_net']:.2%}, "
          f"{(cal['mean_annual_net'] / ic['mean_annual_net'] - 1) * 100:+.1f}%)")
    print(f"  中位数年化(净):   IC={ic['median_annual_net']:.2%}  →  校准={cal['median_annual_net']:.2%}  "
          f"(+{cal['median_annual_net'] - ic['median_annual_net']:.2%}, "
          f"{(cal['median_annual_net'] / ic['median_annual_net'] - 1) * 100:+.1f}%)")
    print(f"  均值Sharpe(净):   IC={ic['mean_sharpe_net']:.3f}  →  校准={cal['mean_sharpe_net']:.3f}  "
          f"(+{cal['mean_sharpe_net'] - ic['mean_sharpe_net']:.3f}, "
          f"{(cal['mean_sharpe_net'] / ic['mean_sharpe_net'] - 1) * 100:+.1f}%)")
    print(f"  中位数Sharpe(净): IC={ic['median_sharpe_net']:.3f}  →  校准={cal['median_sharpe_net']:.3f}  "
          f"(+{cal['median_sharpe_net'] - ic['median_sharpe_net']:.3f}, "
          f"{(cal['median_sharpe_net'] / ic['median_sharpe_net'] - 1) * 100:+.1f}%)")

# 最终结论
print(f"\n\n{'=' * 100}")
print("🎯 最终结论")
print(f"{'=' * 100}")

# 使用Top3000作为最全面的评估
top3k_ic = results['Top3000']['ic']
top3k_cal = results['Top3000']['calibrated']

top1_annual_improve = (top3k_cal['top1_annual_net'] / top3k_ic['top1_annual_net'] - 1) * 100
top1_sharpe_improve = (top3k_cal['top1_sharpe_net'] / top3k_ic['top1_sharpe_net'] - 1) * 100
median_annual_improve = (top3k_cal['median_annual_net'] / top3k_ic['median_annual_net'] - 1) * 100
median_sharpe_improve = (top3k_cal['median_sharpe_net'] / top3k_ic['median_sharpe_net'] - 1) * 100

verdict = "✅ PASS" if (top1_annual_improve > 0 and top1_sharpe_improve > 0) else "❌ FAIL"

print(f"\n校准器验证结论: **{verdict}**")
print(f"\n基于 Top3000 (23.8%样本, 3000组合) 回测结果:")
print(f"\n1. **Top1组合提升** (按Sharpe选择最优组合):")
print(f"   - 年化收益: {top3k_ic['top1_annual_net']:.2%} → {top3k_cal['top1_annual_net']:.2%} ({top1_annual_improve:+.1f}%)")
print(f"   - Sharpe比率: {top3k_ic['top1_sharpe_net']:.3f} → {top3k_cal['top1_sharpe_net']:.3f} ({top1_sharpe_improve:+.1f}%)")
print(f"\n2. **整体质量提升** (中位数指标):")
print(f"   - 年化收益: {top3k_ic['median_annual_net']:.2%} → {top3k_cal['median_annual_net']:.2%} ({median_annual_improve:+.1f}%)")
print(f"   - Sharpe比率: {top3k_ic['median_sharpe_net']:.3f} → {top3k_cal['median_sharpe_net']:.3f} ({median_sharpe_improve:+.1f}%)")
print(f"\n3. **核心价值**:")
if median_annual_improve > 20:
    print(f"   ✅ 校准器显著提升了整体组合池质量 (中位数年化提升{median_annual_improve:.1f}%)")
else:
    print(f"   ⚠️  校准器对整体质量提升有限 (中位数年化仅提升{median_annual_improve:.1f}%)")

if top1_sharpe_improve > 10:
    print(f"   ✅ 校准器成功识别了高夏普组合 (Top1 Sharpe提升{top1_sharpe_improve:.1f}%)")
else:
    print(f"   ⚠️  校准器对Top1组合识别能力有限 (Top1 Sharpe提升{top1_sharpe_improve:.1f}%)")

print(f"\n{'=' * 100}")
print("📝 建议")
print(f"{'=' * 100}")
if verdict == "✅ PASS":
    print("\n✅ 校准器通过验证,建议:")
    print("   1. 使用校准排序进行生产环境组合选择")
    print("   2. 定期重新训练校准器(建议每季度)")
    print("   3. 监控实盘表现与回测偏差")
else:
    print("\n❌ 校准器未通过验证,建议:")
    print("   1. 检查特征工程是否充分")
    print("   2. 调整GBDT超参数")
    print("   3. 增加训练样本窗口数")

print(f"\n{'=' * 100}\n")
