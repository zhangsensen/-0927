#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断工具：计算回测的窗口平均 Sharpe，对比 WFO 的 mean_oos_sharpe

用途：
1. 从回测结果 CSV 读取每日收益
2. 按 WFO 的窗口划分（is_period + oos_period + step_size）切割
3. 计算每个窗口的 Sharpe，取平均
4. 与 WFO 的 mean_oos_sharpe 计算相关性

如果此相关性仍低，则证明问题不在度量不一致，而在调仓逻辑实现差异。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def parse_args():
    parser = argparse.ArgumentParser(description="诊断窗口 Sharpe 对齐")
    parser.add_argument("--wfo-dir", required=True, help="WFO results 目录 (含 all_combos.parquet)")
    parser.add_argument("--backtest-csv", required=True, help="回测结果 CSV (含每日净值曲线)")
    parser.add_argument("--is-period", type=int, default=180, help="IS 窗口天数")
    parser.add_argument("--oos-period", type=int, default=90, help="OOS 窗口天数")
    parser.add_argument("--step-size", type=int, default=90, help="滚动步长")
    parser.add_argument("--output", help="输出 JSON 文件路径")
    return parser.parse_args()


def load_wfo_results(wfo_dir: Path) -> pd.DataFrame:
    """加载 WFO 全量结果"""
    all_combos = wfo_dir / "all_combos.parquet"
    if not all_combos.exists():
        raise FileNotFoundError(f"未找到 {all_combos}")
    
    df = pd.read_parquet(all_combos)
    print(f"✓ WFO 结果: {len(df)} 组合")
    return df


def load_backtest_results(csv_path: Path) -> pd.DataFrame:
    """加载回测结果（需要包含每日收益）"""
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ 回测结果: {len(df)} 组合")
    
    # 检查是否有 daily_returns 列（需要回测脚本支持）
    if "daily_returns" not in df.columns:
        print("⚠️  回测 CSV 缺少 daily_returns 列，无法计算窗口 Sharpe")
        print("提示：需要修改 run_profit_backtest.py 保存每日收益序列")
        return df
    
    return df


def compute_window_avg_sharpe(
    daily_rets: np.ndarray, 
    window_starts: List[int], 
    oos_period: int
) -> Tuple[float, List[float]]:
    """
    计算窗口平均 Sharpe（模拟 WFO 的 mean_oos_sharpe）
    
    参数:
        daily_rets: 全周期日收益率数组
        window_starts: 各 OOS 窗口起始索引
        oos_period: OOS 窗口长度
    
    返回:
        (mean_sharpe, window_sharpes)
    """
    window_sharpes = []
    total_days = len(daily_rets)
    
    for start_idx in window_starts:
        end_idx = min(start_idx + oos_period, total_days)
        window_rets = daily_rets[start_idx:end_idx]
        
        if len(window_rets) < 20:  # 样本过少
            continue
        
        mean_ret = np.mean(window_rets)
        std_ret = np.std(window_rets, ddof=1)
        
        if std_ret < 1e-8:  # 波动率为零
            window_sharpe = 0.0
        else:
            window_sharpe = mean_ret / std_ret * np.sqrt(252)
        
        window_sharpes.append(window_sharpe)
    
    if len(window_sharpes) == 0:
        return 0.0, []
    
    mean_sharpe = np.mean(window_sharpes)
    return mean_sharpe, window_sharpes


def get_oos_window_starts(
    total_days: int, 
    is_period: int, 
    oos_period: int, 
    step_size: int
) -> List[int]:
    """
    计算 OOS 窗口起始索引（与 WFO 逻辑一致）
    
    返回:
        [start_idx1, start_idx2, ...] (相对全周期的索引)
    """
    window_starts = []
    current_start = 0
    
    while current_start + is_period + oos_period <= total_days:
        oos_start = current_start + is_period
        window_starts.append(oos_start)
        current_start += step_size
    
    return window_starts


def main():
    args = parse_args()
    
    wfo_dir = Path(args.wfo_dir)
    backtest_csv = Path(args.backtest_csv)
    
    print("=" * 100)
    print("诊断窗口 Sharpe 对齐")
    print("=" * 100)
    
    # 1. 加载数据
    wfo_df = load_wfo_results(wfo_dir)
    backtest_df = load_backtest_results(backtest_csv)
    
    # 检查回测是否支持窗口分析
    if "daily_returns" not in backtest_df.columns:
        print("\n❌ 无法继续：回测 CSV 缺少 daily_returns 列")
        print("\n📝 需要修改 real_backtest/run_profit_backtest.py:")
        print("   在保存 CSV 时，将每日收益序列序列化为 JSON 字符串")
        print("   示例: df['daily_returns'] = df['daily_rets_array'].apply(json.dumps)")
        sys.exit(1)
    
    # 2. 计算 OOS 窗口起始索引（假设回测与 WFO 用相同数据）
    # 这里需要知道回测的总天数，先用第一个组合的 daily_returns 长度估计
    sample_rets = json.loads(backtest_df.iloc[0]["daily_returns"])
    total_days = len(sample_rets)
    
    window_starts = get_oos_window_starts(
        total_days, args.is_period, args.oos_period, args.step_size
    )
    print(f"✓ OOS 窗口: {len(window_starts)} 个，起始索引 {window_starts[:3]}...")
    
    # 3. 逐组合计算窗口平均 Sharpe
    print("\n计算回测的窗口平均 Sharpe...")
    backtest_window_sharpes = []
    
    for idx, row in backtest_df.iterrows():
        daily_rets = np.array(json.loads(row["daily_returns"]))
        mean_sharpe, _ = compute_window_avg_sharpe(daily_rets, window_starts, args.oos_period)
        backtest_window_sharpes.append(mean_sharpe)
        
        if (idx + 1) % 200 == 0:
            print(f"  进度: {idx + 1}/{len(backtest_df)}")
    
    backtest_df["backtest_window_avg_sharpe"] = backtest_window_sharpes
    
    # 4. 合并 WFO 的 mean_oos_sharpe
    merged = backtest_df.merge(
        wfo_df[["combo", "mean_oos_sharpe"]], 
        on="combo", 
        how="inner"
    )
    
    print(f"\n✓ 合并后共 {len(merged)} 个组合")
    
    # 5. 计算相关性
    wfo_metric = merged["mean_oos_sharpe"].values
    backtest_metric_window = merged["backtest_window_avg_sharpe"].values
    backtest_metric_full = merged["sharpe_net"].values  # 原全周期 Sharpe
    
    rho_window, p_window = spearmanr(wfo_metric, backtest_metric_window)
    tau_window, p_tau_window = kendalltau(wfo_metric, backtest_metric_window)
    
    rho_full, p_full = spearmanr(wfo_metric, backtest_metric_full)
    tau_full, p_tau_full = kendalltau(wfo_metric, backtest_metric_full)
    
    # 6. 输出结果
    print("\n" + "=" * 100)
    print("📊 相关性分析")
    print("=" * 100)
    
    print("\n【对比1】WFO mean_oos_sharpe vs 回测窗口平均 Sharpe")
    print(f"  Spearman ρ: {rho_window:.4f} (p={p_window:.4e})")
    print(f"  Kendall τ:  {tau_window:.4f} (p={p_tau_window:.4e})")
    
    print("\n【对比2】WFO mean_oos_sharpe vs 回测全周期 Sharpe")
    print(f"  Spearman ρ: {rho_full:.4f} (p={p_full:.4e})")
    print(f"  Kendall τ:  {tau_full:.4f} (p={p_tau_full:.4e})")
    
    # 解读
    print("\n💡 解读:")
    if rho_window > 0.5:
        print("  ✅ 窗口平均 Sharpe 与 WFO 高度相关 → 度量一致性良好")
        print("  ⚠️  但全周期 Sharpe 相关性低 → 建议 WFO 改用复利累积 Sharpe")
    elif rho_window < 0.2:
        print("  ❌ 窗口平均 Sharpe 与 WFO 也不相关 → 问题在实现细节差异")
        print("  🔍 可能原因:")
        print("     1. 回测的信号重构与 WFO 不一致")
        print("     2. 调仓日期对齐偏差")
        print("     3. Top-5 选股逻辑差异")
    else:
        print(f"  ⚠️  窗口平均 Sharpe 弱相关 (ρ={rho_window:.2f})")
        print("  需进一步诊断信号生成与持仓逻辑")
    
    # 7. 保存结果
    result = {
        "wfo_dir": str(wfo_dir),
        "backtest_csv": str(backtest_csv),
        "total_combos": len(merged),
        "oos_windows_count": len(window_starts),
        "correlation": {
            "wfo_vs_backtest_window_avg": {
                "spearman_rho": rho_window,
                "spearman_p": p_window,
                "kendall_tau": tau_window,
                "kendall_p": p_tau_window
            },
            "wfo_vs_backtest_full_period": {
                "spearman_rho": rho_full,
                "spearman_p": p_full,
                "kendall_tau": tau_full,
                "kendall_p": p_tau_full
            }
        },
        "statistics": {
            "wfo_mean_oos_sharpe": {
                "mean": float(np.mean(wfo_metric)),
                "std": float(np.std(wfo_metric))
            },
            "backtest_window_avg_sharpe": {
                "mean": float(np.mean(backtest_metric_window)),
                "std": float(np.std(backtest_metric_window))
            },
            "backtest_full_sharpe": {
                "mean": float(np.mean(backtest_metric_full)),
                "std": float(np.std(backtest_metric_full))
            }
        }
    }
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 结果已保存: {output_path}")
    
    print("=" * 100)


if __name__ == "__main__":
    main()
