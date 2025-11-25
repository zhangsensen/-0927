#!/usr/bin/env python3
"""
诊断排序校准器关键特征：输出分布、run 间差异与目标相关性。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def parse_float_list(text: str) -> List[float]:
    items = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not items:
        raise ValueError("权重列表不能为空")
    return items


def normalize_weights(weights: Sequence[float]) -> List[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("目标权重之和必须为正")
    return [w / total for w in weights]


def build_weighted_target(df: pd.DataFrame, components: Sequence[str], weights: Sequence[float]) -> pd.Series:
    if "run_ts" not in df.columns:
        raise ValueError("数据缺少 run_ts 列")
    if len(components) != len(weights):
        raise ValueError("目标组件与权重长度不一致")

    comp_series: List[pd.Series] = []
    for col in components:
        if col not in df.columns:
            raise ValueError(f"目标列缺失: {col}")
        series = df[col]
        if series.isna().any():
            series = series.groupby(df["run_ts"]).transform(lambda x: x.fillna(x.median()))
            series = series.fillna(0.0)
        comp_series.append(series.astype(float))

    weighted = sum(w * s for w, s in zip(weights, comp_series))
    return weighted


def describe_feature(series: pd.Series) -> Dict[str, float]:
    series = series.dropna()
    if series.empty:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p05": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "p05": float(series.quantile(0.05)),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def compute_correlations(feature: pd.Series, target: pd.Series) -> Dict[str, float]:
    corr = feature.corr(target, method="pearson")
    spearman = feature.corr(target, method="spearman")
    return {
        "pearson": float(corr) if not np.isnan(corr) else 0.0,
        "spearman": float(spearman) if not np.isnan(spearman) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="诊断排序特征分布与相关性")
    parser.add_argument("--dataset", type=str, default="data/calibrator_dataset.parquet", help="数据文件")
    parser.add_argument(
        "--features",
        type=str,
        default="oos_ic_min,sortino_ratio,oos_ic_std,profit_factor,oos_ic_p90,oos_ic_last,calmar_net",
        help="需要诊断的特征，逗号分隔",
    )
    parser.add_argument(
        "--target-components",
        type=str,
        default="annual_ret_net_z,sharpe_net_z,calmar_net_z",
        help="用于权重目标的列，逗号分隔",
    )
    parser.add_argument(
        "--target-weights",
        type=str,
        default="0.5,0.3,0.2",
        help="目标权重，逗号分隔",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/feature_diagnostics.json",
        help="输出 JSON 文件路径",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = (repo_root / args.dataset).resolve()
    df = pd.read_parquet(dataset_path)

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    components = [c.strip() for c in args.target_components.split(",") if c.strip()]
    weights = normalize_weights(parse_float_list(args.target_weights))

    df = df.copy()
    df["target_weighted"] = build_weighted_target(df, components, weights)
    df = df.dropna(subset=["target_weighted"])

    diagnostics: Dict[str, Dict[str, object]] = {
        "meta": {
            "dataset": str(dataset_path),
            "runs": sorted(df["run_ts"].unique()),
            "target_components": components,
            "target_weights": weights,
            "n_samples": int(len(df)),
        },
        "features": {},
    }

    for feat in features:
        if feat not in df.columns:
            diagnostics["features"][feat] = {"error": "missing"}
            continue
        series = df[feat].astype(float)
        overall = describe_feature(series)
        corr = compute_correlations(series, df["target_weighted"])
        by_run = {
            run: describe_feature(run_series.astype(float))
            for run, run_series in df.groupby("run_ts")[feat]
        }
        diagnostics["features"][feat] = {
            "overall": overall,
            "correlation": corr,
            "by_run": by_run,
        }

    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"诊断完成，输出: {output_path}")


if __name__ == "__main__":
    main()

