#!/usr/bin/env python3
"""
对比不同TopK规模(Top100/1000/3000)下IC排序vs校准排序的回测效果
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


def find_backtest_results(results_dir: Path, run_ts: str) -> Dict[str, Dict]:
    """
    扫描results_combo_wfo目录,找到所有相关回测结果
    
    Returns:
        {
            'ic_100': {'csv': Path, 'json': Path, 'top1_annual': 0.17, ...},
            'calibrated_100': {...},
            ...
        }
    """
    results = {}
    pattern_mapping = {
        'ic': 'ranking_baseline',
        'calibrated': 'ranking_lightgbm'
    }
    
    for summary_file in results_dir.glob(f"{run_ts}_*/SUMMARY_*.json"):
        with open(summary_file) as f:
            data = json.load(f)
        
        # 解析top_source识别类型
        top_source = data.get('top_source', '')
        ranking_type = None
        if 'ranking_baseline' in top_source or 'ic_top' in top_source.lower():
            ranking_type = 'ic'
        elif 'ranking_lightgbm' in top_source or 'calibrated' in top_source.lower():
            ranking_type = 'calibrated'
        else:
            continue
        
        # 识别TopK规模
        count = data.get('count', 0)
        if count == 0:
            continue
        
        # 读取CSV获取Top1指标
        csv_file = summary_file.parent / f"top{count}_profit_backtest_slip2bps_{summary_file.parent.name}.csv"
        if not csv_file.exists():
            continue
        
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            continue
        
        top1 = df.iloc[0]
        
        key = f"{ranking_type}_{count}"
        results[key] = {
            'csv': csv_file,
            'json': summary_file,
            'count': count,
            'top1_annual_net': top1['annual_ret_net'],
            'top1_sharpe_net': top1['sharpe_net'],
            'top1_max_dd_net': top1['max_dd_net'],
            'mean_annual_net': data['mean_annual_net'],
            'median_annual_net': data['median_annual_net'],
            'mean_sharpe_net': data['mean_sharpe_net'],
            'median_sharpe_net': data['median_sharpe_net'],
        }
    
    return results


def generate_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """生成对比表格"""
    rows = []
    
    # 按TopK分组
    topk_groups = {}
    for key, data in results.items():
        ranking_type, count = key.rsplit('_', 1)
        count = int(count)
        if count not in topk_groups:
            topk_groups[count] = {}
        topk_groups[count][ranking_type] = data
    
    for topk in sorted(topk_groups.keys()):
        group = topk_groups[topk]
        
        ic_data = group.get('ic', {})
        cal_data = group.get('calibrated', {})
        
        if not ic_data or not cal_data:
            continue
        
        # Top1对比
        rows.append({
            'TopK': topk,
            '样本占比': f"{topk/12597*100:.2f}%",
            '指标': 'Top1年化(净)',
            'IC排序': f"{ic_data['top1_annual_net']:.2%}",
            '校准排序': f"{cal_data['top1_annual_net']:.2%}",
            '绝对提升': f"{cal_data['top1_annual_net'] - ic_data['top1_annual_net']:.2%}",
            '相对提升': f"{(cal_data['top1_annual_net'] / ic_data['top1_annual_net'] - 1) * 100:.1f}%",
        })
        
        rows.append({
            'TopK': topk,
            '样本占比': f"{topk/12597*100:.2f}%",
            '指标': 'Top1 Sharpe(净)',
            'IC排序': f"{ic_data['top1_sharpe_net']:.3f}",
            '校准排序': f"{cal_data['top1_sharpe_net']:.3f}",
            '绝对提升': f"{cal_data['top1_sharpe_net'] - ic_data['top1_sharpe_net']:.3f}",
            '相对提升': f"{(cal_data['top1_sharpe_net'] / ic_data['top1_sharpe_net'] - 1) * 100:.1f}%",
        })
        
        # 均值对比
        rows.append({
            'TopK': topk,
            '样本占比': f"{topk/12597*100:.2f}%",
            '指标': '均值年化(净)',
            'IC排序': f"{ic_data['mean_annual_net']:.2%}",
            '校准排序': f"{cal_data['mean_annual_net']:.2%}",
            '绝对提升': f"{cal_data['mean_annual_net'] - ic_data['mean_annual_net']:.2%}",
            '相对提升': f"{(cal_data['mean_annual_net'] / ic_data['mean_annual_net'] - 1) * 100:.1f}%",
        })
        
        rows.append({
            'TopK': topk,
            '样本占比': f"{topk/12597*100:.2f}%",
            '指标': '中位数年化(净)',
            'IC排序': f"{ic_data['median_annual_net']:.2%}",
            '校准排序': f"{cal_data['median_annual_net']:.2%}",
            '绝对提升': f"{cal_data['median_annual_net'] - ic_data['median_annual_net']:.2%}",
            '相对提升': f"{(cal_data['median_annual_net'] / ic_data['median_annual_net'] - 1) * 100:.1f}%",
        })
        
        rows.append({
            'TopK': topk,
            '样本占比': '',
            '指标': '---',
            'IC排序': '---',
            '校准排序': '---',
            '绝对提升': '---',
            '相对提升': '---',
        })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="对比不同TopK的回测结果")
    parser.add_argument("--run-ts", type=str, default="20251113_145102", help="WFO run timestamp")
    parser.add_argument("--results-dir", type=str, default="../results_combo_wfo", help="回测结果目录")
    parser.add_argument("--output", type=str, default=None, help="输出Markdown文件路径")
    args = parser.parse_args()
    
    here = Path(__file__).resolve().parent
    results_dir = (here.parent.parent / args.results_dir).resolve()
    
    print(f"📊 扫描回测结果: {results_dir}")
    results = find_backtest_results(results_dir, args.run_ts)
    
    if not results:
        print("❌ 未找到任何回测结果")
        return
    
    print(f"✅ 找到 {len(results)} 个回测结果:")
    for key in sorted(results.keys()):
        data = results[key]
        print(f"  - {key}: Top1年化={data['top1_annual_net']:.2%}, Sharpe={data['top1_sharpe_net']:.3f}")
    print()
    
    # 生成对比表格
    df = generate_comparison_table(results)
    
    # 输出到终端
    print("=" * 100)
    print("📈 TopK 规模对比分析")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    
    # 保存到文件
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# TopK规模对比分析报告\n\n")
            f.write(f"**WFO Run**: {args.run_ts}\n")
            f.write(f"**总组合数**: 12,597\n")
            f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 对比结果\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            # 添加结论
            f.write("## 核心结论\n\n")
            
            # 找到最大TopK的结果
            max_topk = max(int(k.split('_')[1]) for k in results.keys())
            ic_key = f"ic_{max_topk}"
            cal_key = f"calibrated_{max_topk}"
            
            if ic_key in results and cal_key in results:
                ic_data = results[ic_key]
                cal_data = results[cal_key]
                
                top1_annual_improve = (cal_data['top1_annual_net'] / ic_data['top1_annual_net'] - 1) * 100
                top1_sharpe_improve = (cal_data['top1_sharpe_net'] / ic_data['top1_sharpe_net'] - 1) * 100
                median_annual_improve = (cal_data['median_annual_net'] / ic_data['median_annual_net'] - 1) * 100
                
                verdict = "✅ **PASS**" if (top1_annual_improve > 0 and top1_sharpe_improve > 0) else "❌ **FAIL**"
                
                f.write(f"### 校准器验证结论: {verdict}\n\n")
                f.write(f"基于 **Top{max_topk}** ({max_topk/12597*100:.1f}%样本) 回测结果:\n\n")
                f.write(f"1. **Top1性能提升**:\n")
                f.write(f"   - 年化收益: {ic_data['top1_annual_net']:.2%} → {cal_data['top1_annual_net']:.2%} (+{top1_annual_improve:.1f}%)\n")
                f.write(f"   - Sharpe比率: {ic_data['top1_sharpe_net']:.3f} → {cal_data['top1_sharpe_net']:.3f} (+{top1_sharpe_improve:.1f}%)\n\n")
                f.write(f"2. **整体质量提升**:\n")
                f.write(f"   - 中位数年化: {ic_data['median_annual_net']:.2%} → {cal_data['median_annual_net']:.2%} (+{median_annual_improve:.1f}%)\n\n")
        
        print(f"\n✅ 报告已保存: {output_path}")


if __name__ == "__main__":
    main()
