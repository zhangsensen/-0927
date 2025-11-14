#!/usr/bin/env python3
"""
8天 WFO 全量 + 全频回测自动化流水线

工作流:
1. 检查 8天 WFO 是否完成
2. 提取通过门槛的组合白名单
3. 启动全频真实回测（30 频率）
4. 生成频率泛化分析报告
"""

import json
import os
import sys
import time
from pathlib import Path
from subprocess import run, PIPE
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_wfo_completion(ts: str) -> bool:
    """检查 WFO 是否完成"""
    pid_file = ROOT / f"results/.wfo_8d_full_{ts}.pid"
    if not pid_file.exists():
        return False
    
    pid = int(pid_file.read_text().strip())
    result = run(["ps", "-p", str(pid)], capture_output=True)
    return result.returncode != 0  # 进程不存在=完成


def resolve_wfo_run_ts(ts: str) -> str | None:
    """根据 PID 派生时间戳尝试定位真实落盘的 WFO 结果 run_<ts> 目录。

    有时 WFO 主脚本未正确创建对应的 run_{ts} 输出目录 (例如出现日志但无 run_2025* 目录)，
    导致后续自动化找不到 `wfo_learned_ranking_{ts}.csv` 而提前退出。

    回退策略:
    1. 若 `results/run_{ts}/wfo_learned_ranking_{ts}.csv` 存在, 直接使用。
    2. 否则在 `results/run_*/wfo_learned_ranking_*.csv` 中选择最近修改的一份作为替代。
    3. 若仍不存在, 返回 None。
    """
    run_dir = ROOT / "results" / f"run_{ts}"
    wfo_file = run_dir / f"wfo_learned_ranking_{ts}.csv"
    if wfo_file.exists():
        return ts
    candidates = sorted(
        (ROOT / "results").glob("run_*/wfo_learned_ranking_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        latest = candidates[0]
        latest_ts = latest.name.removeprefix("wfo_learned_ranking_").removesuffix(".csv")
        print(f"⚠️ 未找到预期输出 {wfo_file}，使用最近的有效 WFO 结果: run_{latest_ts}")
        return latest_ts
    print("❌ 未找到任何 WFO 结果文件，无法继续后续阶段")
    return None


def extract_qualified_whitelist(ts: str, ic_threshold: float = 0.03, min_stability: float = 0.5) -> Path:
    """
    从 WFO 结果提取合格组合白名单
    
    筛选标准:
    - wfo_ic > ic_threshold (默认 0.03)
    - stability > min_stability (默认 0.5)
    - FDR 校验通过（如果启用）
    """
    run_dir = ROOT / "results" / f"run_{ts}"
    wfo_file = run_dir / f"wfo_learned_ranking_{ts}.csv"
    
    if not wfo_file.exists():
        print(f"❌ WFO 结果文件不存在: {wfo_file}")
        return None
    
    df = pd.read_csv(wfo_file)
    print(f"✅ 加载 WFO 结果: {len(df)} 组合")
    
    # 筛选合格组合
    qualified = df[
        (df.get('wfo_ic', df.get('mean_oos_ic', 0)) > ic_threshold) &
        (df.get('stability_score', 1.0) > min_stability)
    ].copy()
    
    print(f"✅ 筛选后合格组合: {len(qualified)} (IC>{ic_threshold}, 稳定性>{min_stability})")
    
    # 保存白名单
    whitelist_file = run_dir / f"whitelist_8d_wfo_qualified_{ts}.txt"
    qualified['combo'].to_csv(whitelist_file, index=False, header=False)
    print(f"✅ 白名单已保存: {whitelist_file}")
    
    return whitelist_file, len(qualified)


def run_all_freq_backtest(whitelist_file: Path, ts: str) -> Path:
    """启动全频真实回测"""
    log_file = ROOT / "results" / "logs" / f"all_freq_scan_8d_wfo_{ts}.log"
    
    env = {
        "RB_WHITELIST_FILE": str(whitelist_file),
        "RB_TEST_ALL_FREQS": "1",
        "RB_SKIP_PREV": "1",
        "RB_OUTPUT_PREFIX": f"8d_wfo_qualified_{ts}",
    }
    
    cmd = [sys.executable, "-u", "-m", "real_backtest.run_production_backtest"]
    
    print(f"✅ 启动全频回测...")
    print(f"   日志: {log_file}")
    
    with open(log_file, "w") as f:
        proc = run(cmd, cwd=ROOT, env={**os.environ, **env}, stdout=f, stderr=f)
    
    if proc.returncode != 0:
        print(f"❌ 全频回测失败，查看日志: {log_file}")
        return None
    
    print(f"✅ 全频回测完成")
    
    # 查找输出文件（兼容两种命名风格）
    output_dir = ROOT / "results_combo_wfo"
    scan_file = list(output_dir.glob(f"*/all_freq_scan_8d_wfo_qualified_{ts}.csv"))
    if not scan_file:
        # 回退：部分回测脚本以 all_freq_scan_<ts>.csv 命名
        scan_file = list(output_dir.glob(f"*/all_freq_scan_{ts}.csv"))
    
    if not scan_file:
        print("❌ 未找到全频回测结果文件")
        return None
    
    return scan_file[0]


def analyze_freq_generalization(scan_file: Path, ts: str) -> dict:
    """分析频率泛化能力"""
    df = pd.read_csv(scan_file)
    df['test_freq'] = df['test_freq'].astype(int)
    
    print(f"✅ 加载全频扫描结果: {len(df)} 行")
    
    report = {
        "run_ts": ts,
        "n_combos": len(df['combo'].unique()),
        "n_freqs": len(df['test_freq'].unique()),
    }
    
    # 1. 各频率的 Sharpe 分布
    freq_stats = {}
    for freq in sorted(df['test_freq'].unique()):
        subset = df[df['test_freq'] == freq]['sharpe']
        freq_stats[int(freq)] = {
            'n': int(len(subset)),
            'median': float(np.median(subset)),
            'p20': float(np.percentile(subset, 20)),
            'p80': float(np.percentile(subset, 80)),
            'iqr': float(np.percentile(subset, 75) - np.percentile(subset, 25)),
            'gt_1.0_share': float(np.mean(subset > 1.0)),
        }
    
    report['freq_stats'] = freq_stats
    
    # 2. 每个组合的最佳频率分布
    best_freq_per_combo = df.loc[df.groupby('combo')['sharpe'].idxmax()]
    best_freq_dist = best_freq_per_combo['test_freq'].value_counts().to_dict()
    report['best_freq_distribution'] = {int(k): int(v) for k, v in best_freq_dist.items()}
    
    # 3. 8天 vs 其他频率的秩相关（泛化检验）
    D8 = df[df['test_freq'] == 8][['combo', 'sharpe']].rename(columns={'sharpe': 'sharpe_8'})
    corr_vs_8d = {}
    
    for freq in [6, 7, 9, 10, 12, 16, 21, 24]:
        Df = df[df['test_freq'] == freq][['combo', 'sharpe']].rename(columns={'sharpe': f'sharpe_{freq}'})
        merged = D8.merge(Df, on='combo', how='inner')
        
        if len(merged) > 10:
            sp = float(spearmanr(merged['sharpe_8'], merged[f'sharpe_{freq}']).correlation)
            corr_vs_8d[int(freq)] = sp
    
    report['spearman_8d_vs_other_freqs'] = corr_vs_8d
    
    # 4. 泛化质量判断
    median_corr = np.median(list(corr_vs_8d.values()))
    report['generalization_quality'] = {
        'median_spearman': float(median_corr),
        'judgment': '优秀' if median_corr > 0.7 else ('良好' if median_corr > 0.5 else ('一般' if median_corr > 0.3 else '差'))
    }
    
    # 保存报告
    report_file = scan_file.parent / f"freq_generalization_report_{ts}.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"✅ 分析报告已保存: {report_file}")
    
    return report


def main():
    # 获取最新的运行时间戳（从 PID 文件）
    pid_files = list((ROOT / "results").glob(".wfo_8d_full_*.pid"))
    if not pid_files:
        print("❌ 未找到 8天 WFO 运行记录")
        return 1
    
    latest_pid_file = max(pid_files, key=lambda p: p.stat().st_mtime)
    ts = latest_pid_file.stem.replace(".wfo_8d_full_", "")
    
    print(f"🔍 检测到运行时间戳: {ts}")
    
    # Step 1: 等待 WFO 完成
    print("⏳ 等待 8天 WFO 完成...")
    while not check_wfo_completion(ts):
        time.sleep(60)
        print("   仍在运行，60秒后重试...")
    
    print("✅ 8天 WFO 已完成")
    
    # 时间戳容错修正：解析真实存在的 run_{ts}
    resolved_ts = resolve_wfo_run_ts(ts)
    if not resolved_ts:
        return 1
    ts = resolved_ts

    # Step 2: 提取白名单
    result = extract_qualified_whitelist(ts, ic_threshold=0.03, min_stability=0.5)
    if result is None:
        return 1
    
    whitelist_file, n_qualified = result
    
    if n_qualified < 100:
        print(f"⚠️  合格组合数量过少（{n_qualified}），建议降低筛选标准")
        return 1
    
    # Step 3: 全频回测
    scan_file = run_all_freq_backtest(whitelist_file, ts)
    if scan_file is None:
        return 1
    
    # Step 4: 分析报告
    report = analyze_freq_generalization(scan_file, ts)
    
    # 打印关键结论
    print("\n" + "="*60)
    print("🎯 频率泛化评估结果")
    print("="*60)
    print(f"合格组合数: {report['n_combos']}")
    print(f"测试频率数: {report['n_freqs']}")
    print(f"\n8天 vs 其他频率秩相关（中位数）: {report['generalization_quality']['median_spearman']:.3f}")
    print(f"泛化质量: {report['generalization_quality']['judgment']}")
    
    print("\n最佳频率分布（前5）:")
    best_dist = sorted(report['best_freq_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
    for freq, count in best_dist:
        print(f"  {freq}天: {count} 组合 ({count/report['n_combos']*100:.1f}%)")
    
    print("\n各频率中位 Sharpe（前5）:")
    freq_by_median = sorted(report['freq_stats'].items(), key=lambda x: x[1]['median'], reverse=True)[:5]
    for freq, stats in freq_by_median:
        print(f"  {freq}天: {stats['median']:.3f} (P20={stats['p20']:.3f}, >1.0占比={stats['gt_1.0_share']:.1%})")
    
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
