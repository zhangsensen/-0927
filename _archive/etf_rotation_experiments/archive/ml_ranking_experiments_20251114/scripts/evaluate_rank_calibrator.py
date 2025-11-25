#!/usr/bin/env python3
"""
评估 LightGBM 排序校准器在不同 blend 权重下的表现，输出 TopK 业绩、交集和组合列表。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from build_rank_dataset import find_backtest_csvs  # type: ignore
from apply_rank_calibrator import find_latest_run  # type: ignore


def load_backtest(run_ts: str, base_dir: Path) -> pd.DataFrame:
    csv_paths = find_backtest_csvs(run_ts, base_dir)
    if not csv_paths:
        raise FileNotFoundError(f"未找到回测CSV (run_ts={run_ts})")
    frames = [pd.read_csv(path) for path in csv_paths]
    return pd.concat(frames, ignore_index=True)


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "mean_annual_ret": float(df["annual_ret_net"].mean()),
        "median_annual_ret": float(df["annual_ret_net"].median()),
        "mean_sharpe": float(df["sharpe_net"].mean()),
        "max_maxdd": float(df["max_dd_net"].min()),
        "top10_mean_annual": float(df.head(10)["annual_ret_net"].mean()) if len(df) >= 10 else float("nan"),
        "positive_ratio": float((df["annual_ret_net"] > 0).mean()),
    }


def evaluate_pair(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    topks: List[int],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    merged_base = baseline_df.merge(backtest_df, on="combo", how="left")
    merged_cand = candidate_df.merge(backtest_df, on="combo", how="left")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for k in topks:
        k_eff = min(k, len(merged_base), len(merged_cand))
        base_slice = merged_base.head(k_eff)
        cand_slice = merged_cand.head(k_eff)
        overlap = len(set(base_slice["combo"]) & set(cand_slice["combo"])) / k_eff if k_eff > 0 else 0.0
        results[f"top{k_eff}"] = {
            "baseline": compute_metrics(base_slice),
            "candidate": compute_metrics(cand_slice),
            "overlap": {"ratio": float(overlap)},
        }
    return results


def load_blend_rankings(blend_dir: Path) -> List[Tuple[float, str, Path]]:
    summary_path = blend_dir / "blend_summary.parquet"
    rankings: List[Tuple[float, str, Path]] = []
    if summary_path.exists():
        summary_df = pd.read_parquet(summary_path)
        for row in summary_df.itertuples(index=False):
            alpha_raw = row.alpha_ml
            try:
                alpha = float(alpha_raw)
            except (TypeError, ValueError):
                continue
            rankings.append((alpha, f"{alpha:.2f}", Path(row.output)))
            limited_output = getattr(row, "limited_output", None)
            if isinstance(limited_output, str) and limited_output:
                rankings.append((alpha, f"{alpha:.2f}_limited", Path(limited_output)))
    else:
        for path in sorted(blend_dir.glob("ranking_*.parquet")):
            name = path.stem
            if name == "blend_summary":
                continue
            if name == "ranking_baseline":
                alpha = 0.0
            elif name == "ranking_lightgbm":
                alpha = 1.0
            elif "ranking_blend_" in name:
                alpha = float(name.replace("ranking_blend_", ""))
            else:
                continue
            rankings.append((alpha, f"{alpha:.2f}", path))
    safe_path = blend_dir / "ranking_two_stage_safe.parquet"
    if safe_path.exists():
        rankings.append((float('nan'), "two_stage_safe", safe_path))
    unlimited_path = blend_dir / "ranking_two_stage_unlimited.parquet"
    if unlimited_path.exists():
        rankings.append((float('nan'), "two_stage_unlimited", unlimited_path))
    rankings.sort(key=lambda x: (x[0], x[1]))
    return rankings


def dump_toplists(
    output_dir: Path,
    alpha: float,
    candidate_df: pd.DataFrame,
    topks: List[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for k in topks:
        k_eff = min(k, len(candidate_df))
        subset = candidate_df.head(k_eff)[["combo", "rank_score"]].copy()
        subset.rename(columns={"rank_score": "score"}, inplace=True)
        path = output_dir / f"top{k_eff}_alpha_{alpha:.2f}.csv"
        subset.to_csv(path, index=False)


def produce_markdown(run_ts: str, overview: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], output_path: Path):
    lines = [f"# 排序校准器评估报告 ({run_ts})", ""]
    for alpha, metrics in overview.items():
        lines.append(f"## alpha={alpha}")
        for bucket, detail in metrics.items():
            base = detail["baseline"]
            cand = detail["candidate"]
            overlap = detail["overlap"]["ratio"]
            lines.append(f"### {bucket}")
            lines.append("| 指标 | 基线 | 候选 | Δ |")
            lines.append("| --- | --- | --- | --- |")
            lines.append(
                f"| 平均年化 | {base['mean_annual_ret']:.4f} | {cand['mean_annual_ret']:.4f} | "
                f"{cand['mean_annual_ret'] - base['mean_annual_ret']:.4f} |"
            )
            lines.append(
                f"| 中位年化 | {base['median_annual_ret']:.4f} | {cand['median_annual_ret']:.4f} | "
                f"{cand['median_annual_ret'] - base['median_annual_ret']:.4f} |"
            )
            lines.append(
                f"| 平均夏普 | {base['mean_sharpe']:.4f} | {cand['mean_sharpe']:.4f} | "
                f"{cand['mean_sharpe'] - base['mean_sharpe']:.4f} |"
            )
            lines.append(
                f"| Top10均值 | {base['top10_mean_annual']:.4f} | {cand['top10_mean_annual']:.4f} | "
                f"{cand['top10_mean_annual'] - base['top10_mean_annual']:.4f} |"
            )
            lines.append(
                f"| 盈利比例 | {base['positive_ratio']:.2%} | {cand['positive_ratio']:.2%} | "
                f"{cand['positive_ratio'] - base['positive_ratio']:.2%} |"
            )
            lines.append(
                f"| 最大回撤 | {base['max_maxdd']:.4f} | {cand['max_maxdd']:.4f} | "
                f"{cand['max_maxdd'] - base['max_maxdd']:.4f} |"
            )
            lines.append(f"| Top 交集 |  | {overlap:.2%} |  |")
            lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="评估排序校准器 Blend 表现")
    parser.add_argument("--run-ts", type=str, help="目标 run_ts（默认最新）")
    parser.add_argument("--blend-dir", type=str, help="Blend 排名所在目录（默认 run_dir/ranking_blends）")
    parser.add_argument("--topk", type=str, default="100,200,500,1000,2000", help="TopK 列表，逗号分隔")
    parser.add_argument("--report", type=str, default="docs/RANKING_CALIBRATOR_REPORT.md", help="Markdown 报告输出路径")
    parser.add_argument("--dump-combos", action="store_true", help="输出各权重 TopK 组合列表 CSV")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results_root = repo_root / "results"
    backtest_root = repo_root / "results_combo_wfo"
    topks = [int(x.strip()) for x in args.topk.split(",") if x.strip()]

    run_dir = results_root / f"run_{args.run_ts}" if args.run_ts else find_latest_run(results_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"run 目录不存在: {run_dir}")
    run_ts = run_dir.name.replace("run_", "")

    blend_dir = Path(args.blend_dir).resolve() if args.blend_dir else run_dir / "ranking_blends"
    if not blend_dir.exists():
        raise FileNotFoundError(f"Blend 目录不存在: {blend_dir}")

    print("=" * 100)
    print("排序校准器评估")
    print("=" * 100)
    print(f"Run目录: {run_dir}")
    print(f"Blend目录: {blend_dir}")

    rankings = load_blend_rankings(blend_dir)
    if not rankings:
        raise RuntimeError(f"{blend_dir} 未找到任何 ranking_*.parquet")

    backtest_df = load_backtest(run_ts, backtest_root)

    baseline_path = None
    for alpha, label, path in rankings:
        if alpha == 0.0 and label == "0.00":
            baseline_path = path
            break
    if baseline_path is None:
        raise RuntimeError("缺少 alpha=0.0 的 baseline 排名文件")

    baseline_df = pd.read_parquet(baseline_path)
    if "combo" not in baseline_df.columns:
        raise KeyError(f"文件 {baseline_path} 缺少 combo 列")
    baseline_df = baseline_df.reset_index(drop=True)

    overview: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    metrics_root = repo_root / "results"
    combos_root = blend_dir / "toplists"
    combos_root.mkdir(parents=True, exist_ok=True)

    for alpha, label, path in rankings:
        candidate_df = pd.read_parquet(path)
        if "combo" not in candidate_df.columns:
            raise KeyError(f"文件 {path} 缺少 combo 列")
        candidate_df = candidate_df.reset_index(drop=True)
        metrics = evaluate_pair(baseline_df, candidate_df, backtest_df, topks)
        overview[label] = metrics
        print(f"alpha={label}:")
        for bucket, detail in metrics.items():
            base = detail["baseline"]
            cand = detail["candidate"]
            overlap = detail["overlap"]["ratio"]
            print(
                f"  {bucket} | meanRet {base['mean_annual_ret']:.4f} → {cand['mean_annual_ret']:.4f} "
                f"| Top10 {base['top10_mean_annual']:.4f} → {cand['top10_mean_annual']:.4f} "
                f"| Sharpe {base['mean_sharpe']:.4f} → {cand['mean_sharpe']:.4f} "
                f"| overlap {overlap:.2%}"
            )
        if args.dump_combos:
            dump_toplists(combos_root, alpha, candidate_df, topks)

    metrics_path = metrics_root / f"calibrator_eval_{run_ts}.json"
    metrics_root.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(overview, f, indent=2)

    report_path = Path(args.report).resolve()
    produce_markdown(run_ts, overview, report_path)

    print(f"评估指标: {metrics_path}")
    print(f"Markdown报告: {report_path}")
    if args.dump_combos:
        print(f"TopK 组合列表输出目录: {combos_root}")
    print("=" * 100)
    print("✅ 评估完成")
    print("=" * 100)


if __name__ == "__main__":
    main()

