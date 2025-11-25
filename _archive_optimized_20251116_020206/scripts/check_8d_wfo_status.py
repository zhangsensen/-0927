#!/usr/bin/env python3
"""
8天 WFO 流水线状态检查工具

用法: python scripts/check_8d_wfo_status.py
"""

import sys
from pathlib import Path
from subprocess import run, PIPE
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def check_process(pid: int) -> bool:
    """检查进程是否运行"""
    result = run(["ps", "-p", str(pid)], capture_output=True)
    return result.returncode == 0


def get_latest_ts() -> str:
    """获取最新运行时间戳"""
    pid_files = list((ROOT / "results").glob(".wfo_8d_full_*.pid"))
    if not pid_files:
        return None
    return max(pid_files, key=lambda p: p.stat().st_mtime).stem.replace(".wfo_8d_full_", "")


def main():
    ts = get_latest_ts()
    if not ts:
        print("❌ 未找到 8天 WFO 运行记录")
        return
    
    print("="*70)
    print(f"🔍 8天 WFO 流水线状态检查 (TS: {ts})")
    print("="*70)
    
    # 检查 WFO 进程
    pid_file = ROOT / f"results/.wfo_8d_full_{ts}.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        wfo_running = check_process(pid)
        
        print(f"\n📍 阶段 1: 8天 WFO 全量运行")
        print(f"   PID: {pid}")
        print(f"   状态: {'✅ 运行中' if wfo_running else '✅ 已完成'}")
        
        log_file = ROOT / f"results/logs/wfo_8d_full_{ts}.log"
        if log_file.exists():
            print(f"   日志: {log_file}")
            
            # 尝试读取进度（最后 10 行）
            try:
                with open(log_file) as f:
                    lines = f.readlines()
                    if lines:
                        print(f"   最后更新: {lines[-1].strip()[:80]}...")
            except:
                pass
    else:
        wfo_running = False
        print(f"\n📍 阶段 1: 8天 WFO 全量运行 - ⚠️  未启动")
    
    # 检查白名单
    run_dir = ROOT / "results" / f"run_{ts}"
    whitelist_file = run_dir / f"whitelist_8d_wfo_qualified_{ts}.txt"
    
    print(f"\n📍 阶段 2: 白名单生成")
    if whitelist_file.exists():
        n_qualified = len(whitelist_file.read_text().strip().split('\n'))
        print(f"   状态: ✅ 已完成")
        print(f"   合格组合数: {n_qualified}")
    else:
        print(f"   状态: ⏳ 待完成")
    
    # 检查全频回测
    scan_files = list((ROOT / "results_combo_wfo").glob(f"*/all_freq_scan_8d_wfo_qualified_{ts}.csv"))
    
    print(f"\n📍 阶段 3: 全频真实回测")
    if scan_files:
        scan_file = scan_files[0]
        df = pd.read_csv(scan_file)
        n_combos = len(df['combo'].unique())
        n_freqs = len(df['test_freq'].unique())
        
        print(f"   状态: ✅ 已完成")
        print(f"   组合数: {n_combos}")
        print(f"   频率数: {n_freqs}")
        print(f"   结果文件: {scan_file}")
    else:
        print(f"   状态: ⏳ 待完成")
    
    # 检查分析报告
    report_files = list((ROOT / "results_combo_wfo").glob(f"*/freq_generalization_report_{ts}.json"))
    
    print(f"\n📍 阶段 4: 频率泛化分析")
    if report_files:
        import json
        report = json.loads(report_files[0].read_text())
        
        print(f"   状态: ✅ 已完成")
        print(f"   泛化质量: {report['generalization_quality']['judgment']} (Spearman中位: {report['generalization_quality']['median_spearman']:.3f})")
        print(f"   报告文件: {report_files[0]}")
    else:
        print(f"   状态: ⏳ 待完成")
    
    print("\n" + "="*70)
    
    # 监控进程检查
    monitor_log = ROOT / f"results/logs/monitor_8d_wfo_{ts}.log"
    if monitor_log.exists():
        print(f"\n📊 监控日志: {monitor_log}")
        print(f"   查看实时进度: tail -f {monitor_log}")
    
    print()


if __name__ == "__main__":
    main()
