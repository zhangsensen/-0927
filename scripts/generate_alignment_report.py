#!/usr/bin/env python3
"""
生成 VEC vs BT 对齐后的差异报告（TopK）。

- 要求输入包含对齐后的列：
  - VEC: vec_aligned_return, vec_aligned_sharpe
  - BT : bt_aligned_return, bt_aligned_sharpe
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_topk(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",") if x.strip()]


def summarize(diff: pd.Series, top: int, label: str) -> dict:
    abs_diff = diff.abs()
    return {
        "top": top,
        "metric": label,
        "mean": diff.mean(),
        "p95_abs": abs_diff.quantile(0.95),
        "max_abs": abs_diff.max(),
        "count_gt_0.05": int((abs_diff > 0.05).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate aligned VEC/BT diff report.")
    parser.add_argument("--vec", required=True, help="VEC 结果 CSV（需含 vec_aligned_* 列）")
    parser.add_argument("--bt", required=True, help="BT 结果 CSV（需含 bt_aligned_* 列）")
    parser.add_argument(
        "--topk",
        default="50,100,200",
        help="以逗号分隔的 TopK 列表（默认: 50,100,200）",
    )
    parser.add_argument(
        "--sort-by",
        default="vec_aligned_return",
        help="排序依据（默认 vec_aligned_return）",
    )
    parser.add_argument(
        "--output",
        default="results/alignment_diff_report.csv",
        help="输出汇总 CSV 路径",
    )
    args = parser.parse_args()

    vec_path = Path(args.vec)
    bt_path = Path(args.bt)
    if not vec_path.exists() or not bt_path.exists():
        raise FileNotFoundError("请检查 VEC/BT 结果路径是否存在")

    vec_df = pd.read_csv(vec_path)
    bt_df = pd.read_csv(bt_path)

    required_vec = {"combo", "vec_aligned_return", "vec_aligned_sharpe"}
    required_bt = {"combo", "bt_aligned_return", "bt_aligned_sharpe"}
    if not required_vec.issubset(vec_df.columns):
        missing = required_vec - set(vec_df.columns)
        raise ValueError(f"VEC 缺少列: {missing}")
    if not required_bt.issubset(bt_df.columns):
        missing = required_bt - set(bt_df.columns)
        raise ValueError(f"BT 缺少列: {missing}")

    merged = pd.merge(vec_df, bt_df, on="combo", how="inner", suffixes=("_vec", "_bt"))
    merged = merged.sort_values(args.sort_by, ascending=False)

    topk_list = parse_topk(args.topk)
    summary_rows = []
    for k in topk_list:
        top_df = merged.head(k)
        if top_df.empty:
            continue
        summary_rows.append(
            summarize(
                top_df["vec_aligned_return"] - top_df["bt_aligned_return"],
                k,
                "return",
            )
        )
        summary_rows.append(
            summarize(
                top_df["vec_aligned_sharpe"] - top_df["bt_aligned_sharpe"],
                k,
                "sharpe",
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    print("\n📊 对齐后差异统计")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n✅ 报告已保存: {output_path}")


if __name__ == "__main__":
    main()


