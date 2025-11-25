#!/usr/bin/env python3
"""对比多个排名方案在真实回测中的表现"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_results_mapping(text: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"格式错误，需使用 label:path -> {part}")
        label, path_str = part.split(":", 1)
        mapping[label.strip()] = Path(path_str.strip())
    if not mapping:
        raise ValueError("至少提供一个结果文件")
    return mapping


def compute_metrics(df: pd.DataFrame, topk: int) -> Dict[str, float]:
    subset = df.head(topk)
    if subset.empty:
        return {
            "n": 0,
            "mean_annual_ret": float("nan"),
            "median_annual_ret": float("nan"),
            "mean_sharpe": float("nan"),
            "mean_max_dd": float("nan"),
            "top10_mean_annual": float("nan"),
            "positive_ratio": float("nan"),
        }
    return {
        "n": int(len(subset)),
        "mean_annual_ret": float(subset["annual_ret_net"].mean()),
        "median_annual_ret": float(subset["annual_ret_net"].median()),
        "mean_sharpe": float(subset["sharpe_net"].mean()),
        "mean_max_dd": float(subset["max_dd_net"].mean()),
        "top10_mean_annual": float(subset.head(10)["annual_ret_net"].mean()),
        "positive_ratio": float((subset["annual_ret_net"] > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B 测试多个排名方案")
    parser.add_argument("--results", type=str, required=True, help="格式 label:path,label2:path2 ...")
    parser.add_argument("--topks", type=str, default="100,200,500,1000", help="评估用的 TopK 阈值")
    parser.add_argument("--output", type=str, help="输出 JSON 路径，默认 results/calibrator_ab_report.json")
    args = parser.parse_args()

    mapping = parse_results_mapping(args.results)
    topks = [int(x.strip()) for x in args.topks.split(",") if x.strip()]
    if not topks:
        raise ValueError("topks 不能为空")
    topks = sorted(set(topks))

    records: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, path in mapping.items():
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"结果文件不存在: {path}")
        df = pd.read_csv(path)
        df = df.sort_values("annual_ret_net", ascending=False).reset_index(drop=True)
        label_metrics: Dict[str, Dict[str, float]] = {}
        for k in topks:
            label_metrics[f"top{k}"] = compute_metrics(df, k)
        records[label] = label_metrics

    baseline_label = "baseline" if "baseline" in records else next(iter(records))
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, metrics in records.items():
        summary[label] = metrics
        if label != baseline_label:
            deltas: Dict[str, Dict[str, float]] = {}
            for k in topks:
                key = f"top{k}"
                if key not in records[baseline_label]:
                    continue
                base = records[baseline_label][key]
                curr = metrics[key]
                deltas[key] = {
                    "delta_mean_annual": curr["mean_annual_ret"] - base["mean_annual_ret"],
                    "delta_median_annual": curr["median_annual_ret"] - base["median_annual_ret"],
                    "delta_mean_sharpe": curr["mean_sharpe"] - base["mean_sharpe"],
                }
            summary[label]["delta_vs_baseline"] = deltas

    output_path = Path(args.output) if args.output else Path("results/calibrator_ab_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("评估结果:")
    for label, metrics in summary.items():
        print(f"\n== {label} ==")
        for key in [k for k in metrics.keys() if k.startswith("top")]:
            data = metrics[key]
            print(
                f"Top{key[3:]} | mean={data['mean_annual_ret']:.4f} | median={data['median_annual_ret']:.4f} "
                f"| Sharpe={data['mean_sharpe']:.4f} | Pct+={data['positive_ratio']:.2%}"
            )
        if "delta_vs_baseline" in metrics:
            print("  Δ vs baseline:")
            for key, delta in metrics["delta_vs_baseline"].items():
                print(
                    f"    {key}: Δmean={delta['delta_mean_annual']:.4f} | "
                    f"Δmedian={delta['delta_median_annual']:.4f} | ΔSharpe={delta['delta_mean_sharpe']:.4f}"
                )


if __name__ == "__main__":
    main()
