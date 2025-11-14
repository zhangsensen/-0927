#!/usr/bin/env python3
"""
将训练好的 LightGBM 排序校准器应用到指定 run_* 的 all_combos.parquet。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from build_rank_dataset import (  # type: ignore
    add_combo_composition_features,
    add_extreme_risk_features,
    add_market_regime_features,
    add_real_derived_features,
    enrich_wfo_features,
)


def list_run_dirs(base: Path) -> List[Path]:
    if not base.exists():
        return []
    runs = [d for d in base.glob("run_*") if d.is_dir()]
    return sorted({d.resolve() for d in runs}, reverse=True)


def find_latest_run(results_root: Path) -> Path:
    runs = list_run_dirs(results_root)
    if not runs:
        raise FileNotFoundError(f"未在 {results_root} 找到 run_* 目录")
    return runs[0]


def prepare_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    feat_df = df.copy()
    for col in feature_names:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    feat_df = feat_df[feature_names]
    if feat_df.isna().any().any():
        feat_df = feat_df.fillna(feat_df.median(numeric_only=True))
    return feat_df


def compute_local_stats(series: pd.Series) -> Dict[str, float]:
    arr = series.astype(float).values
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    p05 = float(np.percentile(arr, 5)) if arr.size else 0.0
    p95 = float(np.percentile(arr, 95)) if arr.size else 0.0
    return {"mean": mean, "std": std, "p05": p05, "p95": p95}


def load_scaler_stats(path: Path) -> Dict[str, float]:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            mean = float(data.get("mean", 0.0))
            std = float(data.get("std", 1.0))
            if not math.isfinite(std) or std <= 0:
                std = 1.0
            p05 = data.get("p05")
            p95 = data.get("p95")
            return {"mean": mean, "std": std, "p05": float(p05) if p05 is not None else None, "p95": float(p95) if p95 is not None else None}
        except Exception:
            pass
    return {"mean": 0.0, "std": 1.0, "p05": None, "p95": None}


def scale_with_stats(values: np.ndarray | pd.Series, stats: Dict[str, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = stats.get("mean", 0.0) or 0.0
    std = stats.get("std", 1.0) or 1.0
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    scaled = (arr - mean) / std
    low = stats.get("p05")
    high = stats.get("p95")
    if low is not None and high is not None and high > low:
        low_scaled = (low - mean) / std
        high_scaled = (high - mean) / std
        scaled = np.clip(scaled, low_scaled, high_scaled)
    return scaled


def _apply_replacement_limit(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    thresholds: Sequence[int],
    allowed_replacements: Dict[int, int],
) -> pd.DataFrame:
    """
    在给定的 TopK 阈值下，限制 ML 排序对基线 TopK 的替换比例。
    当某阈值的替换数超过 allowed_replacements[k] 时，回退为基线组合。
    """
    baseline_order = baseline_df["combo"].tolist()
    candidate_order = candidate_df["combo"].tolist()
    baseline_sets: Dict[int, Set[str]] = {
        k: set(baseline_df.head(k)["combo"].tolist())
        for k in thresholds
    }
    limits_sorted = sorted(thresholds)
    replace_counts = {k: 0 for k in thresholds}
    baseline_ptr = 0

    final: List[str] = []
    used: Set[str] = set()

    for combo in candidate_order:
        if combo in used:
            continue
        final.append(combo)
        used.add(combo)
        exceeded = False
        for k in limits_sorted:
            if len(final) <= k:
                if combo not in baseline_sets[k]:
                    replace_counts[k] += 1
                    allowed = allowed_replacements.get(k, 0)
                    if replace_counts[k] > allowed:
                        exceeded = True
                        replace_counts[k] -= 1
                        break
        if exceeded:
            final.pop()
            used.remove(combo)
            while baseline_ptr < len(baseline_order) and baseline_order[baseline_ptr] in used:
                baseline_ptr += 1
            if baseline_ptr < len(baseline_order):
                base_combo = baseline_order[baseline_ptr]
                final.append(base_combo)
                used.add(base_combo)
                baseline_ptr += 1

    for combo in baseline_order:
        if combo not in used:
            final.append(combo)
            used.add(combo)
    for combo in candidate_order:
        if combo not in used:
            final.append(combo)
            used.add(combo)

    limited_df = candidate_df.set_index("combo").loc[final].reset_index()
    limited_df.rename(columns={"index": "combo"}, inplace=True)
    limited_df["rank_limited"] = np.arange(1, len(limited_df) + 1)
    return limited_df


def _parse_confidence_thresholds(text: Optional[str]) -> Dict[int, float]:
    if not text:
        return {}
    mapping: Dict[int, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"confidence-thresholds 参数格式错误: {part}")
        topk_str, val_str = part.split(":", 1)
        topk = int(topk_str.strip())
        value = float(val_str.strip())
        mapping[topk] = value
    return mapping


def apply_safe_replacement(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    sharpe_scores: pd.Series,
    ml_scores: pd.Series,
    baseline_scores: pd.Series,
    thresholds: Sequence[int],
    confidence_thresholds: Dict[int, float],
    max_replacements: Dict[int, int],
    sharpe_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    baseline_order = baseline_df["combo"].tolist()
    baseline_map = baseline_scores.to_dict()
    sharpe_map = sharpe_scores.to_dict()
    ml_map = ml_scores.to_dict()
    delta_map = {combo: ml_map.get(combo, 0.0) - baseline_map.get(combo, 0.0) for combo in ml_map.keys()}

    baseline_sets: Dict[int, Set[str]] = {k: set(baseline_df.head(k)["combo"].tolist()) for k in thresholds}
    replace_counts = {k: 0 for k in thresholds}
    used: Set[str] = set()
    final: List[str] = []
    baseline_ptr = 0

    limits_sorted = sorted(thresholds)

    for combo in candidate_df["combo"]:
        if combo in used:
            continue

        allow = True
        for k in limits_sorted:
            if k <= 0 or len(final) >= k:
                continue
            base_set = baseline_sets[k]
            if combo not in base_set:
                allowed_max = max_replacements.get(k, 0)
                if allowed_max <= 0 or replace_counts[k] >= allowed_max:
                    allow = False
                    break
                if sharpe_map.get(combo, -math.inf) < sharpe_threshold:
                    allow = False
                    break
                if delta_map.get(combo, 0.0) < confidence_thresholds.get(k, 0.0):
                    allow = False
                    break
        if allow:
            final.append(combo)
            used.add(combo)
            for k in limits_sorted:
                if k > 0 and len(final) <= k and combo not in baseline_sets[k]:
                    replace_counts[k] += 1
        else:
            while baseline_ptr < len(baseline_order) and baseline_order[baseline_ptr] in used:
                baseline_ptr += 1
            if baseline_ptr < len(baseline_order):
                base_combo = baseline_order[baseline_ptr]
                final.append(base_combo)
                used.add(base_combo)
                baseline_ptr += 1

    for combo in baseline_order:
        if combo not in used:
            final.append(combo)
            used.add(combo)
    for combo in candidate_df["combo"]:
        if combo not in used:
            final.append(combo)
            used.add(combo)

    safe_df = candidate_df.set_index("combo").loc[final].reset_index()
    safe_df.rename(columns={"index": "combo"}, inplace=True)
    safe_df["rank_safe"] = np.arange(1, len(safe_df) + 1)

    report: Dict[str, Dict[str, object]] = {}
    for k in limits_sorted:
        if k <= 0:
            continue
        base_top = baseline_df.head(k)
        safe_top = safe_df.head(k)
        replaced = [c for c in safe_top["combo"] if c not in set(base_top["combo"])]
        removed = [c for c in base_top["combo"] if c not in set(safe_top["combo"])]
        report[str(k)] = {
            "max_allowed": int(max_replacements.get(k, 0)),
            "actual_replacements": int(replace_counts.get(k, 0)),
            "confidence_threshold": float(confidence_thresholds.get(k, 0.0)),
            "replaced_combos": replaced,
            "removed_combos": removed,
        }

    meta = {
        "sharpe_threshold": float(sharpe_threshold),
        "results": report,
    }
    return safe_df, meta


def main():
    parser = argparse.ArgumentParser(description="应用排序校准器生成新的组合得分")
    parser.add_argument("--run-ts", type=str, help="目标 run_ts，默认选择最新 run_*")
    parser.add_argument("--model", type=str, default="results/models/calibrator_ranker.txt", help="兼容：单阶段模型文件")
    parser.add_argument("--sharpe-model", type=str, help="Stage1 Sharpe 筛选模型路径")
    parser.add_argument("--profit-model", type=str, help="Stage2 年化收益模型路径")
    parser.add_argument("--sharpe-threshold", type=float, default=0.0, help="Sharpe 预测阈值（z-score）")
    parser.add_argument("--output", type=str, help="兼容参数：单文件输出（仅当不使用blend时）")
    parser.add_argument("--topk", type=int, default=0, help="可选：只导出前K个组合")
    parser.add_argument(
        "--blend-step",
        type=float,
        default=0.1,
        help="baseline 与 ML 权重步长，默认为 0.1（生成 0→1 所有权重）",
    )
    parser.add_argument(
        "--blend-dir",
        type=str,
        help="Blend 结果输出目录（默认写入 run 目录下 ranking_blends/）",
    )
    parser.add_argument(
        "--limit-topk",
        type=str,
        help="限幅控制的 TopK 列表，逗号分隔，例如 100,200,500；为空则不启用",
    )
    parser.add_argument(
        "--limit-frac",
        type=float,
        default=None,
        help="允许替换比例（0~1），例如 0.2 表示 TopK 内最多 20%% 组合可被 ML 排序替换",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="启用安全替换模式（需要提供 --sharpe-model 与 --profit-model）",
    )
    parser.add_argument(
        "--confidence-thresholds",
        type=str,
        help="安全替换显著性阈值，格式如 100:0.03,500:0.02",
    )
    parser.add_argument(
        "--max-replacement-pct",
        type=float,
        default=0.15,
        help="安全替换每个 TopK 的最大替换比例",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results_root = repo_root / "results"

    run_dir = results_root / f"run_{args.run_ts}" if args.run_ts else find_latest_run(results_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"run 目录不存在: {run_dir}")
    run_ts = run_dir.name.replace("run_", "")

    two_stage = bool(args.sharpe_model and args.profit_model)
    if args.safe_mode and not two_stage:
        raise ValueError("安全替换模式需要提供 --sharpe-model 与 --profit-model")
    if args.max_replacement_pct < 0 or args.max_replacement_pct > 1:
        raise ValueError("max-replacement-pct 必须在 [0, 1] 区间")

    if two_stage:
        sharpe_path = (repo_root / args.sharpe_model).resolve()
        profit_path = (repo_root / args.profit_model).resolve()
        if not sharpe_path.exists():
            raise FileNotFoundError(f"Sharpe 模型不存在: {sharpe_path}")
        if not profit_path.exists():
            raise FileNotFoundError(f"收益模型不存在: {profit_path}")
        sharpe_model = lgb.Booster(model_file=str(sharpe_path))
        profit_model = lgb.Booster(model_file=str(profit_path))
        sharpe_features = list(sharpe_model.feature_name())
        profit_features = list(profit_model.feature_name())
        sharpe_scaler = load_scaler_stats(sharpe_path.with_name("calibrator_sharpe_filter_scaler.json"))
        profit_scaler = load_scaler_stats(profit_path.with_name("calibrator_profit_ranker_scaler.json"))
    else:
        model_path = (repo_root / args.model).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        sharpe_model = None
        sharpe_features: List[str] = []
        sharpe_scaler = {"mean": 0.0, "std": 1.0, "p05": None, "p95": None}
        profit_model = lgb.Booster(model_file=str(model_path))
        profit_features = list(profit_model.feature_name())
        profit_scaler = load_scaler_stats(model_path.with_name("calibrator_profit_ranker_scaler.json"))

    all_combos_path = run_dir / "all_combos.parquet"
    if not all_combos_path.exists():
        raise FileNotFoundError(f"缺少 all_combos.parquet: {all_combos_path}")

    print("=" * 100)
    print("应用排序校准器")
    print("=" * 100)
    print(f"Run目录: {run_dir}")
    if two_stage:
        print(f"Stage1 模型: {sharpe_path}")
        print(f"Stage2 模型: {profit_path}")
        print(f"Stage1 特征数: {len(sharpe_features)} | Stage2 特征数: {len(profit_features)}")
    else:
        print(f"模型文件: {model_path}")
        print(f"特征数: {len(profit_features)}")

    enriched = enrich_wfo_features(pd.read_parquet(all_combos_path))
    prices_template = repo_root / "data" / "etf_prices_template.csv"
    if prices_template.exists():
        enriched = add_market_regime_features(enriched, prices_template)
    enriched = add_combo_composition_features(enriched)
    enriched = add_extreme_risk_features(enriched)
    enriched = add_real_derived_features(enriched)
    baseline_col = "calibrated_sharpe_pred" if "calibrated_sharpe_pred" in enriched.columns else "mean_oos_ic"
    if baseline_col not in enriched.columns:
        raise KeyError("baseline 排序缺失 (calibrated_sharpe_pred / mean_oos_ic)")

    baseline_sorted = enriched.sort_values(baseline_col, ascending=False).reset_index(drop=True)
    scored = enriched.copy()
    scored["baseline_score"] = scored[baseline_col].astype(float)
    baseline_stats = compute_local_stats(scored["baseline_score"])
    scored["baseline_score_scaled"] = scale_with_stats(scored["baseline_score"], baseline_stats)

    if two_stage:
        sharpe_feats = prepare_features(enriched, sharpe_features)
        sharpe_raw = sharpe_model.predict(sharpe_feats)
        profit_feats = prepare_features(enriched, profit_features)
        profit_raw = profit_model.predict(profit_feats)
        scored["sharpe_ml_raw"] = sharpe_raw.astype(float)
        scored["ml_score_raw"] = profit_raw.astype(float)
        scored["sharpe_ml_score"] = scale_with_stats(sharpe_raw, sharpe_scaler)
        scored["ml_score"] = scale_with_stats(profit_raw, profit_scaler)
        below_mask = scored["sharpe_ml_score"] < args.sharpe_threshold
        if below_mask.any():
            scored.loc[below_mask, "ml_score"] = scored.loc[below_mask, "baseline_score_scaled"]
            scored.loc[below_mask, "ml_score_raw"] = scored.loc[below_mask, "baseline_score"]
    else:
        feats = prepare_features(enriched, profit_features)
        preds_raw = profit_model.predict(feats)
        scored["ml_score_raw"] = preds_raw.astype(float)
        scored["ml_score"] = scale_with_stats(preds_raw, profit_scaler)

    scored["delta_score"] = scored["ml_score"] - scored["baseline_score_scaled"]
    scored["delta_score_raw"] = scored["ml_score_raw"] - scored["baseline_score"]

    step = max(args.blend_step, 0.0)
    if step <= 0 or step > 1:
        raise ValueError("blend-step 必须在 (0, 1] 范围内")

    blend_dir = Path(args.blend_dir).resolve() if args.blend_dir else run_dir / "ranking_blends"
    blend_dir.mkdir(parents=True, exist_ok=True)

    limit_sets = None
    if args.limit_topk and args.limit_frac is not None and args.limit_frac >= 0:
        limit_values = [int(x.strip()) for x in args.limit_topk.split(",") if x.strip()]
        if limit_values:
            limit_sets = {
                "thresholds": sorted(set(limit_values)),
                "allowed": {k: int(np.ceil(max(0.0, min(1.0, args.limit_frac)) * k)) for k in limit_values},
            }

    confidence_thresholds = _parse_confidence_thresholds(args.confidence_thresholds)
    if not confidence_thresholds:
        confidence_thresholds = {100: 0.03, 500: 0.02, 1000: 0.01}
    safe_thresholds = sorted(set(confidence_thresholds.keys()) | {100, 500, 1000})
    base_thresholds = sorted(confidence_thresholds.keys())
    default_delta = confidence_thresholds[base_thresholds[-1]] if base_thresholds else 0.0
    for k in safe_thresholds:
        confidence_thresholds.setdefault(k, default_delta)
    default_caps = {100: 10, 500: 75, 1000: 200}
    safe_allowed = {}
    for k in safe_thresholds:
        pct_cap = int(math.ceil(k * args.max_replacement_pct)) if args.max_replacement_pct > 0 else 0
        base_cap = default_caps.get(k, pct_cap)
        cap = min(base_cap, pct_cap) if pct_cap else base_cap
        safe_allowed[k] = max(0, cap)

    alphas = np.round(np.arange(0.0, 1.0 + step, step), 4)
    alphas = np.unique(np.clip(alphas, 0.0, 1.0))

    summary = []
    safe_reports: List[Dict[str, object]] = []
    for alpha in alphas:
        blend = scored.copy()
        blend["rank_score"] = (1 - alpha) * blend["baseline_score_scaled"] + alpha * blend["ml_score"]
        blend = blend.sort_values("rank_score", ascending=False).reset_index(drop=True)
        blend["rank_blend"] = np.arange(1, len(blend) + 1)
        if args.topk and args.topk > 0:
            blend = blend.head(args.topk)

        if alpha == 0.0:
            filename = blend_dir / "ranking_baseline.parquet"
        elif alpha == 1.0:
            filename = blend_dir / "ranking_lightgbm.parquet"
        else:
            filename = blend_dir / f"ranking_blend_{alpha:.2f}.parquet"
        blend.to_parquet(filename, index=False)

        top_combo = blend.iloc[0]["combo"] if not blend.empty else None
        record = {"alpha_ml": float(alpha), "output": str(filename), "top_combo": top_combo}

        if limit_sets and alpha > 0.0:
            limited_df = _apply_replacement_limit(
                baseline_sorted,
                blend,
                thresholds=limit_sets["thresholds"],
                allowed_replacements=limit_sets["allowed"],
            )
            limited_path = blend_dir / f"ranking_blend_{alpha:.2f}_limited.parquet"
            limited_df.to_parquet(limited_path, index=False)
            record["limited_output"] = str(limited_path)
            record["limited_top_combo"] = limited_df.iloc[0]["combo"] if not limited_df.empty else None

        if args.safe_mode and two_stage and alpha > 0.0:
            safe_df, safe_meta = apply_safe_replacement(
                baseline_df=baseline_sorted,
                candidate_df=blend,
                sharpe_scores=scored.set_index("combo")["sharpe_ml_score"],
                ml_scores=scored.set_index("combo")["ml_score"],
                baseline_scores=scored.set_index("combo")["baseline_score_scaled"],
                thresholds=safe_thresholds,
                confidence_thresholds=confidence_thresholds,
                max_replacements=safe_allowed,
                sharpe_threshold=args.sharpe_threshold,
            )
            safe_path = blend_dir / f"ranking_blend_{alpha:.2f}_safe.parquet"
            safe_df.to_parquet(safe_path, index=False)
            record["safe_output"] = str(safe_path)
            record["safe_top_combo"] = safe_df.iloc[0]["combo"] if not safe_df.empty else None
            safe_meta = {"alpha": float(alpha), **safe_meta}
            safe_meta["mode"] = f"blend_{alpha:.2f}"
            safe_reports.append(safe_meta)

        print(f"  alpha={alpha:.2f} -> {filename.name} | 样本数={len(blend)}")

        if alpha == 1.0:
            compat_path = run_dir / "ranking_lightgbm.parquet"
            blend.to_parquet(compat_path, index=False)
            print(f"    ↳ 兼容输出: {compat_path}")

        summary.append(record)

    # 额外输出两阶段（不依赖 alpha）的安全/无限制排名，便于直接 A/B
    if two_stage:
        ml_sorted = scored.sort_values("ml_score", ascending=False).reset_index(drop=True)
        unlimited_path = blend_dir / "ranking_two_stage_unlimited.parquet"
        unlimited_df = ml_sorted.copy()
        unlimited_df["rank_score"] = unlimited_df["ml_score"].astype(float)
        unlimited_df.to_parquet(unlimited_path, index=False)
        summary.append({"alpha_ml": "two_stage_unlimited", "output": str(unlimited_path)})
        if args.safe_mode:
            safe_df, safe_meta = apply_safe_replacement(
                baseline_df=baseline_sorted,
                candidate_df=ml_sorted,
                sharpe_scores=scored.set_index("combo")["sharpe_ml_score"] if "sharpe_ml_score" in scored.columns else pd.Series(dtype=float),
                ml_scores=scored.set_index("combo")["ml_score"],
                baseline_scores=scored.set_index("combo")["baseline_score_scaled"],
                thresholds=safe_thresholds,
                confidence_thresholds=confidence_thresholds,
                max_replacements=safe_allowed,
                sharpe_threshold=args.sharpe_threshold,
            )
            safe_path = blend_dir / "ranking_two_stage_safe.parquet"
            # 将安全替换选择结果映射回完整信息（包含回测需要的列，如 best_rebalance_freq）
            safe_combo = safe_df["combo"].tolist() if "combo" in safe_df.columns else []
            merged_safe = scored.set_index("combo").loc[safe_combo].reset_index() if safe_combo else scored.iloc[0:0].copy()
            if "rank_score" not in merged_safe.columns:
                merged_safe["rank_score"] = merged_safe["ml_score"].astype(float)
            merged_safe.to_parquet(safe_path, index=False)
            summary.append({"alpha_ml": "two_stage_safe", "output": str(safe_path)})
            safe_meta = {"alpha": "two_stage_safe", **safe_meta}
            safe_meta["mode"] = "two_stage_safe"
            safe_reports.append(safe_meta)

    summary_path = blend_dir / "blend_summary.parquet"
    summary_df = pd.DataFrame(summary)
    if "alpha_ml" in summary_df.columns:
        summary_df["alpha_ml"] = summary_df["alpha_ml"].astype(str)
    summary_df.to_parquet(summary_path, index=False)
    if safe_reports:
        report_path = blend_dir / "safe_replacement_report.json"
        with open(report_path, "w") as f:
            json.dump(safe_reports, f, indent=2)
        print(f"安全替换报告: {report_path}")
    print(f"Blend 总结: {summary_path}")

    print("=" * 100)
    print("✅ 完成")
    print("=" * 100)


if __name__ == "__main__":
    main()

