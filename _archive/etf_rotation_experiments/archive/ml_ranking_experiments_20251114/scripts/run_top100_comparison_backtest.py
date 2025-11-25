#!/usr/bin/env python3
"""
对最新 run 的 IC 排序和校准排序 Top100 运行真实回测

用法:
  cd /Users/zhangshenshen/深度量化0927/etf_rotation_experiments
  python scripts/run_top100_comparison_backtest.py --run-dir results/run_20251112_223854
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


def prepare_ranking_file(combos_csv: Path, output_parquet: Path, sort_column: str, run_dir: Path):
    """将 CSV 格式的策略列表转换为回测脚本需要的 parquet 格式"""
    df = pd.read_csv(combos_csv)
    
    # 确保有 combo 列
    if 'combo' not in df.columns:
        raise ValueError(f"CSV 文件缺少 'combo' 列: {combos_csv}")
    
    # 按排序列降序排列
    if sort_column in df.columns:
        df = df.sort_values(sort_column, ascending=False)
    
    # 添加 rank_score（回测脚本可能需要）
    if 'rank_score' not in df.columns:
        if sort_column in df.columns:
            df['rank_score'] = df[sort_column]
        else:
            df['rank_score'] = range(len(df), 0, -1)
    
    # 合并必要的元信息（如 best_rebalance_freq）
    all_path = run_dir / "all_combos.parquet"
    if all_path.exists():
        try:
            all_df = pd.read_parquet(all_path)[["combo", "best_rebalance_freq"]]
            df = df.merge(all_df, on="combo", how="left")
        except Exception:
            pass

    # 保存为 parquet
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    
    print(f"✅ 已生成排序文件: {output_parquet}")
    print(f"   - 策略数: {len(df)}")
    print(f"   - 排序列: {sort_column}")
    if sort_column in df.columns:
        print(f"   - {sort_column} 范围: {df[sort_column].min():.4f} ~ {df[sort_column].max():.4f}")
    

def run_backtest(
    ranking_file: Path,
    topk: int,
    slippage_bps: float,
    label: str,
    python_bin: str = "python",
):
    """运行真实回测"""
    print("\n" + "="*80)
    print(f"🚀 开始 {label} 回测")
    print("="*80)
    print(f"排序文件: {ranking_file}")
    print(f"TopK: {topk}")
    print(f"滑点: {slippage_bps} bps")
    print()
    
    # 构建回测命令
    cmd = [
        python_bin,
        "real_backtest/run_profit_backtest.py",
        "--topk", str(topk),
        "--ranking-file", str(ranking_file),
        "--slippage-bps", str(slippage_bps),
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    # 运行回测
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✅ {label} 回测完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {label} 回测失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Top100 对照回测")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="WFO run 目录，例如 results/run_20251112_223854",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="回测 TopK，默认 100",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=2.0,
        help="滑点（bps），默认 2.0",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="python",
        help="Python 解释器路径",
    )
    parser.add_argument(
        "--skip-ic",
        action="store_true",
        help="跳过 IC 排序回测",
    )
    parser.add_argument(
        "--skip-calibrated",
        action="store_true",
        help="跳过校准排序回测",
    )
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"❌ Run 目录不存在: {run_dir}")
        sys.exit(1)
    
    print("="*80)
    print("📊 Top100 对照回测")
    print("="*80)
    print(f"Run 目录: {run_dir}")
    print(f"TopK: {args.topk}")
    print(f"滑点: {args.slippage_bps} bps")
    print()
    
    # 检查输入文件
    ic_csv = run_dir / "top100_ic_combos.csv"
    cal_csv = run_dir / "top100_calibrated_combos.csv"
    
    if not ic_csv.exists():
        print(f"❌ IC 排序文件不存在: {ic_csv}")
        sys.exit(1)
    
    if not cal_csv.exists():
        print(f"❌ 校准排序文件不存在: {cal_csv}")
        sys.exit(1)
    
    # 创建回测临时目录
    backtest_dir = run_dir / "backtest_comparison"
    backtest_dir.mkdir(exist_ok=True)
    
    # 准备排序文件
    ic_ranking = backtest_dir / "ranking_ic_top100.parquet"
    cal_ranking = backtest_dir / "ranking_calibrated_top100.parquet"
    
    print("📝 准备排序文件...")
    prepare_ranking_file(ic_csv, ic_ranking, "mean_oos_ic", run_dir)
    prepare_ranking_file(cal_csv, cal_ranking, "calibrated_sharpe_pred", run_dir)
    print()
    
    # 运行回测
    results = {}
    
    if not args.skip_ic:
        results['ic'] = run_backtest(
            ranking_file=ic_ranking,
            topk=args.topk,
            slippage_bps=args.slippage_bps,
            label="IC排序Top100",
            python_bin=args.python_bin,
        )
    
    if not args.skip_calibrated:
        results['calibrated'] = run_backtest(
            ranking_file=cal_ranking,
            topk=args.topk,
            slippage_bps=args.slippage_bps,
            label="校准排序Top100",
            python_bin=args.python_bin,
        )
    
    # 总结
    print("\n" + "="*80)
    print("📊 回测完成总结")
    print("="*80)
    for label, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{label}: {status}")
    print()
    
    if all(results.values()):
        print("✅ 所有回测均成功完成")
        print()
        print("下一步:")
        print("  1. 查看回测结果 CSV 文件（在 results_combo_wfo/ 目录下）")
        print("  2. 运行对照分析脚本:")
        print(f"     python scripts/analyze_historical_backtest.py \\")
        print(f"       --backtest-csv <回测结果路径> \\")
        print(f"       --ic-ranking {ic_csv} \\")
        print(f"       --calibrated-ranking {cal_csv} \\")
        print(f"       --output {run_dir}/latest_backtest_comparison")
    else:
        print("⚠️  部分回测失败，请检查日志")
    
    print("="*80)


if __name__ == "__main__":
    main()
