#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量运行 TopN 敏感性分析：
- 调用现有 scripts/compute_wfo_backtest_metrics.py，多次运行不同的 --top-n，
- 将每次的标准输出保存为独立文本，并汇总为一个 Markdown 报告，便于人工复核。

优点：
- 不依赖 compute_wfo_backtest_metrics.py 的内部函数或数据结构；
- 避免对现有脚本做侵入式修改，降低风险；

用法示例：
  python3 scripts/run_topn_sensitivity.py \
    --wfo-dir results/wfo/20251028_174331 \
    --topn-list 5,7,10,12,15 \
    --tx-cost-bps 5.0 --max-turnover 0.5 --target-vol 0.1 --weight-mode equal

若未提供 --wfo-dir，将自动取 results/wfo 下最近的一个目录。
输出目录：results/analysis/topn_sensitivity_YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_latest_wfo_dir(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    candidates = sorted(
        [p for p in base.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_compute_once(
    wfo_dir: Path,
    top_n: int,
    tx_cost_bps: Optional[float] = None,
    max_turnover: Optional[float] = None,
    target_vol: Optional[float] = None,
    weight_mode: Optional[str] = None,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "compute_wfo_backtest_metrics.py"),
        str(wfo_dir),
        "--top-n",
        str(top_n),
    ]

    if tx_cost_bps is not None:
        cmd += ["--tx-cost-bps", str(tx_cost_bps)]
    if max_turnover is not None:
        cmd += ["--max-turnover", str(max_turnover)]
    if target_vol is not None:
        cmd += ["--target-vol", str(target_vol)]
    if weight_mode is not None:
        cmd += ["--weight-mode", str(weight_mode)]

    # 捕获 stdout/stderr，便于汇总
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run TopN sensitivity using compute_wfo_backtest_metrics.py"
    )
    parser.add_argument(
        "--wfo-dir", type=str, default="", help="WFO结果目录（缺省则自动选择最新）"
    )
    parser.add_argument(
        "--topn-list",
        type=str,
        default="5,7,10,12,15",
        help="逗号分隔列表，如 5,7,10,12,15",
    )
    parser.add_argument(
        "--tx-cost-bps", type=float, default=None, help="单边交易成本（基点）"
    )
    parser.add_argument(
        "--max-turnover", type=float, default=None, help="每期最大换手比例(0-1)"
    )
    parser.add_argument(
        "--target-vol", type=float, default=None, help="目标波动率（如0.1表示10%）"
    )
    parser.add_argument(
        "--weight-mode",
        type=str,
        default=None,
        help="权重模式：equal / ic / riskpar 等（视脚本支持）",
    )
    parser.add_argument(
        "--out-dir", type=str, default="", help="输出目录（默认自动生成）"
    )

    args = parser.parse_args()

    wfo_dir = Path(args.wfo_dir) if args.wfo_dir else None
    if not wfo_dir:
        wfo_dir = find_latest_wfo_dir(PROJECT_ROOT / "results" / "wfo")
        if not wfo_dir:
            print("未找到results/wfo下的任何目录，请先运行WFO。", file=sys.stderr)
            sys.exit(1)

    if not wfo_dir.exists():
        print(f"指定的WFO目录不存在: {wfo_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        topn_list: List[int] = [
            int(x.strip()) for x in args.topn_list.split(",") if x.strip()
        ]
    except Exception as e:
        print(f"解析 --topn-list 失败: {e}", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (PROJECT_ROOT / "results" / "analysis" / f"topn_sensitivity_{timestamp}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    summary_lines.append(f"# TopN敏感性分析汇总\n")
    summary_lines.append(f"- WFO目录: `{wfo_dir}`\n")
    summary_lines.append(f"- 运行时间: {timestamp}\n")
    if args.tx_cost_bps is not None:
        summary_lines.append(f"- 交易成本: {args.tx_cost_bps} bps\n")
    if args.max_turnover is not None:
        summary_lines.append(f"- 最大换手: {args.max_turnover}\n")
    if args.target_vol is not None:
        summary_lines.append(f"- 目标波动率: {args.target_vol}\n")
    if args.weight_mode is not None:
        summary_lines.append(f"- 权重模式: {args.weight_mode}\n")
    summary_lines.append("")

    for n in topn_list:
        print(f"运行 TopN={n} …")
        proc = run_compute_once(
            wfo_dir=wfo_dir,
            top_n=n,
            tx_cost_bps=args.tx_cost_bps,
            max_turnover=args.max_turnover,
            target_vol=args.target_vol,
            weight_mode=args.weight_mode,
        )

        # 保存原始输出
        out_path = out_dir / f"topn_{n}.txt"
        out_path.write_text(
            (proc.stdout or "") + ("\n\n[stderr]\n" + (proc.stderr or "")),
            encoding="utf-8",
        )

        # 汇总Markdown
        summary_lines.append(f"## TopN = {n}\n")
        summary_lines.append("```text")
        # 控制长度，避免过长
        text_preview = proc.stdout.strip() if proc.stdout else "(no stdout)"
        if len(text_preview) > 4000:
            text_preview = text_preview[:4000] + "\n... (truncated)"
        summary_lines.append(text_preview)
        if proc.returncode != 0:
            summary_lines.append("")
            summary_lines.append("[stderr]")
            err_preview = proc.stderr.strip() if proc.stderr else "(no stderr)"
            if len(err_preview) > 2000:
                err_preview = err_preview[:2000] + "\n... (truncated)"
            summary_lines.append(err_preview)
        summary_lines.append("````\n")

    # 写入汇总Markdown
    summary_md = out_dir / "SUMMARY.md"
    summary_md.write_text("\n".join(summary_lines), encoding="utf-8")

    print("—" * 80)
    print(f"已完成，汇总报告: {summary_md}")
    print(f"单项输出位于: {out_dir}")


if __name__ == "__main__":
    main()
