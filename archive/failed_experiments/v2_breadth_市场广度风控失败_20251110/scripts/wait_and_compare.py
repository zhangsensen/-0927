#!/usr/bin/env python3
"""
等待回测完成并运行比较分析的脚本
"""
import subprocess
import time
import sys
from pathlib import Path
import pandas as pd
import json
import numpy as np

def wait_for_backtest_completion():
    """等待回测完成"""
    pid_file = Path("results/logs/rb_full_all_combos.pid")

    if not pid_file.exists():
        print("❌ PID文件不存在")
        return False

    pid = int(pid_file.read_text().strip())
    print(f"🔍 监控回测进程 PID: {pid}")

    while True:
        try:
            # 检查进程是否还在运行
            result = subprocess.run(['ps', '-p', str(pid)],
                                      capture_output=True, text=True)
            if str(pid) not in result.stdout:
                print(f"✅ 回测进程 {pid} 已完成")
                return True

            # 显示进度
            log_files = list(Path("results/logs").glob("rb_full_all_combos_8d_*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_log, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if "Done" in last_line:
                                # 提取进度信息
                                import re
                                match = re.search(r'Done (\d+) tasks', last_line)
                                if match:
                                    completed = int(match.group(1))
                                    progress = completed / 12597 * 100
                                    print(f"📊 进度: {progress:.1f}% ({completed}/12597)")
                except:
                    pass

            print(f"⏳ 等待回测完成... (PID {pid} 仍在运行)")
            time.sleep(30)  # 每30秒检查一次

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断监控")
            return False
        except Exception as e:
            print(f"❌ 监控错误: {e}")
            return False

def run_comparison():
    """运行比较分析"""
    print("🔍 开始运行比较分析...")

    root = Path("results")

    # 获取最新运行目录
    link = root / "run_latest"
    if link.exists():
        latest_dir = link.resolve()
    else:
        latest_dir = sorted([p for p in root.glob("run_*") if p.is_dir()],
                           key=lambda p: p.name)[-1]

    # 获取前一运行目录
    cands = sorted([p for p in root.glob("run_*") if p.is_dir()],
                   key=lambda p: p.name)
    prev_dir = None
    for p in reversed(cands):
        if p.name != latest_dir.name:
            prev_dir = p
            break

    print(f"📁 最新运行: {latest_dir.name}")
    if prev_dir:
        print(f"📁 前一运行: {prev_dir.name}")

    # 加载并计算比较
    new_df = pd.read_parquet(latest_dir / "all_combos.parquet")

    # 排序辅助函数
    sort_old = lambda df: df.sort_values(['mean_oos_ic', 'stability_score'],
                                        ascending=[False, False])

    def sort_new(df):
        if 'calibrated_sharpe_pred' in df.columns:
            return df.sort_values(['calibrated_sharpe_pred', 'stability_score'],
                                   ascending=[False, False])
        elif 'calibrated_sharpe_full' in df.columns:
            return df.sort_values(['calibrated_sharpe_full', 'stability_score'],
                                   ascending=[False, False])
        else:
            return sort_old(df)

    new_sorted = sort_new(new_df)
    ks = [100, 500, 1000, 2000]

    res = {
        'latest': latest_dir.name,
        'new_mean_ic': float(new_df['mean_oos_ic'].mean())
    }

    if 'calibrated_sharpe_pred' in new_df.columns:
        res['calibrated_mean'] = float(new_df['calibrated_sharpe_pred'].mean())
    elif 'calibrated_sharpe_full' in new_df.columns:
        res['calibrated_mean'] = float(new_df['calibrated_sharpe_full'].mean())

    thr_new = np.percentile(new_df['mean_oos_ic'], 80)
    res['precision_ic_latest'] = {
        k: float((new_sorted.head(k)['mean_oos_ic'] > thr_new).mean())
        for k in ks
    }

    if prev_dir and (prev_dir / "all_combos.parquet").exists():
        old_df = pd.read_parquet(prev_dir / "all_combos.parquet")
        old_sorted = sort_old(old_df)

        overlaps = {}
        for k in ks:
            old_top = set(old_sorted.head(k)['combo'])
            new_top = set(new_sorted.head(k)['combo'])
            ov = len(old_top & new_top)
            overlaps[k] = {
                'overlap_count': ov,
                'overlap_ratio': ov / max(1, len(old_top))
            }

        res['overlap'] = overlaps
        res['previous'] = prev_dir.name
        res['old_mean_ic'] = float(old_df['mean_oos_ic'].mean())

        thr_old = np.percentile(old_df['mean_oos_ic'], 80)
        res['precision_ic_prev'] = {
            k: float((old_sorted.head(k)['mean_oos_ic'] > thr_old).mean())
            for k in ks
        }

    # 写入输出文件
    cmp_dir = latest_dir / "comparison"
    cmp_dir.mkdir(exist_ok=True)

    (cmp_dir / "comparison_metrics.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False)
    )

    # 生成Markdown报告
    lines = []
    lines.append(f"# 排序对比报告 ({latest_dir.name} vs {res.get('previous', 'N/A')})\n")
    lines.append("## 摘要\n")
    lines.append(f"- 最新 run: {latest_dir.name}\n")
    if 'previous' in res:
        lines.append(f"- 前一 run: {res['previous']}\n")
    lines.append(f"- 最新 mean_oos_ic: {res['new_mean_ic']:.6f}\n")
    if 'calibrated_mean' in res:
        lines.append(f"- 校准分均值: {res['calibrated_mean']:.6f}\n")
    lines.append("\n## Overlap & Precision@K\n")

    if 'overlap' in res:
        for k, v in res['overlap'].items():
            lp = res['precision_ic_latest'].get(k, float('nan'))
            op = res.get('precision_ic_prev', {}).get(k, float('nan'))
            lines.append(f"- K={k}: overlap={v['overlap_count']} ({v['overlap_ratio']*100:.1f}%), "
                         f"prev_P@K={op:.3f}, new_P@K={lp:.3f}\n")
    else:
        for k, lp in res['precision_ic_latest'].items():
            lines.append(f"- K={k}: new_P@K={lp:.3f}\n")

    lines.append("\n## 备注\n")
    lines.append("- 本次生产回测采用校准优先排序逻辑\n")
    lines.append("- 若校准列缺失则回退到IC排序\n")

    (cmp_dir / "FINAL_REPORT.md").write_text(
        ''.join(lines), encoding='utf-8'
    )

    print(f"✅ 比较报告已生成: {cmp_dir}")
    print(f"📄 JSON: {cmp_dir / 'comparison_metrics.json'}")
    print(f"📄 Markdown: {cmp_dir / 'FINAL_REPORT.md'}")

    return True

def main():
    print("🚀 启动回测监控和比较分析...")

    # 等待回测完成
    if wait_for_backtest_completion():
        # 运行比较分析
        if run_comparison():
            print("🎉 全部任务完成！")
        else:
            print("❌ 比较分析失败")
            sys.exit(1)
    else:
        print("❌ 回测监控失败")
        sys.exit(1)

if __name__ == "__main__":
    main()