#!/usr/bin/env python3
"""批量运行真实回测，按排名文件分段调用 run_profit_backtest.py"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd


def run_batch(
    subset: pd.DataFrame,
    temp_path: Path,
    python_bin: str,
    slippage_bps: float,
    extra_args: Optional[str] = None,
) -> None:
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(temp_path, index=False)

    cmd = [python_bin, "real_backtest/run_profit_backtest.py", "--topk", str(len(subset)), "--ranking-file", str(temp_path), "--slippage-bps", str(slippage_bps)]
    if extra_args:
        cmd.extend(extra_args.split())
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量运行真实回测")
    parser.add_argument("--run-ts", type=str, required=True, help="目标 run_ts")
    parser.add_argument(
        "--ranking-file",
        type=str,
        help="排名文件（parquet）。默认使用 ranking_two_stage_unlimited.parquet",
    )
    parser.add_argument("--topk-start", type=int, default=0, help="起始位置（包含，默认0）")
    parser.add_argument("--topk-end", type=int, help="结束位置（不包含），默认排名长度")
    parser.add_argument("--batch-size", type=int, default=500, help="每批回测组合数")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="回测滑点 (bps)")
    parser.add_argument(
        "--python-bin",
        type=str,
        default="python",
        help="用于执行 run_profit_backtest.py 的 Python 解释器",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        help="附加传递给 run_profit_backtest.py 的参数字符串，例如 '--cash 1000000'",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="tmp/batch_backtest",
        help="用于存放临时排名文件的目录",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    run_dir = repo_root / "results" / f"run_{args.run_ts}"
    if not run_dir.exists():
        raise FileNotFoundError(f"run 目录不存在: {run_dir}")

    if args.ranking_file:
        ranking_path = Path(args.ranking_file)
        if not ranking_path.is_absolute():
            ranking_path = (repo_root / args.ranking_file).resolve()
    else:
        ranking_path = run_dir / "ranking_blends" / "ranking_two_stage_unlimited.parquet"
    if not ranking_path.exists():
        raise FileNotFoundError(f"排名文件不存在: {ranking_path}")

    print(f"加载排名文件: {ranking_path}")
    ranking = pd.read_parquet(ranking_path)
    if "rank_score" in ranking.columns:
        ranking = ranking.sort_values("rank_score", ascending=False)
    ranking = ranking.reset_index(drop=True)

    start = max(args.topk_start, 0)
    end = min(args.topk_end, len(ranking)) if args.topk_end else len(ranking)
    if start >= end:
        raise ValueError("topk-start 必须小于 topk-end")

    temp_dir = (repo_root / args.temp_dir).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)

    cursor = start
    batch_id = 1
    while cursor < end:
        batch_end = min(cursor + args.batch_size, end)
        subset = ranking.iloc[cursor:batch_end].copy()
        subset_path = temp_dir / f"ranking_{args.run_ts}_{cursor}_{batch_end}.parquet"
        print(f"[Batch {batch_id}] Top {cursor} -> {batch_end}")
        run_batch(
            subset=subset,
            temp_path=subset_path,
            python_bin=args.python_bin,
            slippage_bps=args.slippage_bps,
            extra_args=args.extra_args,
        )
        cursor = batch_end
        batch_id += 1

    print("✅ 批量回测完成")


if __name__ == "__main__":
    main()
