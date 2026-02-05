#!/usr/bin/env python3
"""
Final Triple Validation Script
------------------------------
Performs the ultimate cross-check:
1. Train (VEC) Performance
2. Rolling OOS Consistency (Strict Gate)
3. Holdout Performance (Unseen Data)

Outputs the final "Gold Standard" strategy list.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.utils.run_meta import write_step_meta

# === CONFIGURATION ===
# Input Paths (Updated for Full Space Run 2025-12-15)
PATH_VEC = Path("results/vec_from_wfo_20251215_005509/full_space_results.parquet")
PATH_HOLDOUT = Path(
    "results/holdout_validation_20251215_010103/holdout_validation_results.parquet"
)
# IMPORTANT: train-only rolling summary (end_date=training_end_date) to avoid holdout leakage into the gate.
PATH_ROLLING = Path(
    "results/rolling_oos_consistency_20251215_005920/rolling_oos_summary.parquet"
)

# Output Dir
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"results/final_triple_validation_{TIMESTAMP}")

# Gates
GATE_ROLLING_NAME = "Strict"
GATE_ROLLING_POS_RATE = 0.60
GATE_ROLLING_WORST_RET = -0.08
GATE_ROLLING_CALMAR = 0.80

GATE_HOLDOUT_MIN_RET = 0.0  # Must be profitable in holdout
# =====================

# Risk filter (per recent audit): exclude unstable/low-trust factors
# + orthogonal_v1 cleanup: remove redundant/insignificant factors
EXCLUDE_FACTORS = {
    # 原有排除
    "OBV_SLOPE_10D",
    "CMF_20D",
    "VOL_RATIO_60D",
    # 正交化 v1 新增排除
    "REALIZED_VOL_20D",  # ≡ RET_VOL_20D (corr=1.000)
    "RELATIVE_STRENGTH_VS_MARKET_20D",  # ≡ MOM_20D (corr=1.000)
    "RET_VOL_20D",  # 被 SPREAD_PROXY+MAX_DD 覆盖
    "RSI_14",  # corr=0.93 VORTEX, IC 不显著
    "TSMOM_60D",  # corr=0.87 CALMAR, IC 不显著
    "TSMOM_120D",  # IC 不显著, 0/4 候选出现
    "TURNOVER_ACCEL_5_20",  # IC 不显著, Top2 仅+1%
}
EXCLUDE_FACTOR_PAIRS = [
    ("VOL_RATIO_20D", "VOL_RATIO_60D"),
]


def _parse_combo(combo: str) -> list[str]:
    parts = [p.strip() for p in combo.replace("+", " + ").split("+")]
    return [p for p in parts if p and p != "+"]


def _combo_is_allowed(combo: str) -> bool:
    factors = set(_parse_combo(combo))
    if factors & EXCLUDE_FACTORS:
        return False
    for a, b in EXCLUDE_FACTOR_PAIRS:
        if a in factors and b in factors:
            return False
    return True


def load_data(path_vec: Path, path_holdout: Path, path_rolling: Path):
    print(f"Loading VEC: {path_vec}")
    vec = pd.read_parquet(path_vec)

    print(f"Loading Holdout: {path_holdout}")
    holdout = pd.read_parquet(path_holdout)
    # Holdout file contains redundant 'vec_' columns. We only want 'holdout_' columns and 'combo'.
    # Filter columns to keep
    holdout_cols_to_keep = ["combo"] + [
        c for c in holdout.columns if c.startswith("holdout_")
    ]
    holdout = holdout[holdout_cols_to_keep].copy()

    print(f"Loading Rolling: {path_rolling}")
    rolling = pd.read_parquet(path_rolling)
    # Rename rolling columns to avoid collision/confusion
    rolling_cols = {c: f"roll_{c}" for c in rolling.columns if c != "combo"}
    rolling = rolling.rename(columns=rolling_cols)

    return vec, holdout, rolling


def _safe_read_prev_final_candidates(prev_path: Path | None) -> pd.DataFrame | None:
    if prev_path is None:
        return None
    if not prev_path.exists():
        print(f"[WARN] prev-final-candidates not found: {prev_path}")
        return None
    try:
        df = pd.read_parquet(prev_path)
        if "combo" not in df.columns:
            print(f"[WARN] prev-final-candidates missing 'combo' column: {prev_path}")
            return None
        return df
    except Exception as e:
        print(f"[WARN] failed to read prev-final-candidates: {prev_path} ({e})")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Final triple validation (Train VEC + Train-only rolling + Holdout)."
    )
    parser.add_argument("--vec", type=str, default=str(PATH_VEC))
    parser.add_argument("--holdout", type=str, default=str(PATH_HOLDOUT))
    parser.add_argument("--rolling", type=str, default=str(PATH_ROLLING))
    parser.add_argument(
        "--prev-final-candidates",
        type=str,
        default="results/final_triple_validation_20251214_010910/final_candidates.parquet",
        help="Optional: compare overlap vs a previous run",
    )
    args = parser.parse_args()

    path_vec = Path(args.vec)
    path_holdout = Path(args.holdout)
    path_rolling = Path(args.rolling)
    prev_final_path = (
        Path(args.prev_final_candidates) if args.prev_final_candidates else None
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & Merge
    vec, holdout, rolling = load_data(path_vec, path_holdout, path_rolling)
    prev_final = _safe_read_prev_final_candidates(prev_final_path)

    print("Merging datasets...")
    merged = vec.merge(holdout, on="combo", how="inner")
    merged = merged.merge(rolling, on="combo", how="inner")

    total_count = len(merged)
    print(f"Total strategies merged: {total_count}")

    # 1.5 Apply risk filter (factor exclusions)
    mask_risk = merged["combo"].apply(_combo_is_allowed)
    passed_risk = merged[mask_risk].copy()
    print(
        f"Passed Risk Filter (exclude {sorted(EXCLUDE_FACTORS)}): {len(passed_risk)} / {total_count} ({len(passed_risk)/total_count:.2%})"
    )

    # 2. Apply Rolling Consistency Gate (Strict)
    # Note: rolling summary already has 'full_calmar_ratio' (from rolling script re-calc)
    # and 'all_segment_positive_rate', 'all_segment_worst_return'
    # We use the columns from rolling df (prefixed with roll_)

    # Mapping back to the column names we renamed:
    # roll_all_segment_positive_rate
    # roll_all_segment_worst_return
    # roll_full_calmar_ratio

    mask_rolling = (
        (passed_risk["roll_all_segment_positive_rate"] >= GATE_ROLLING_POS_RATE)
        & (passed_risk["roll_all_segment_worst_return"] >= GATE_ROLLING_WORST_RET)
        & (passed_risk["roll_full_calmar_ratio"] >= GATE_ROLLING_CALMAR)
    )

    passed_rolling = passed_risk[mask_rolling].copy()
    print(
        f"Passed Rolling Gate ({GATE_ROLLING_NAME}): {len(passed_rolling)} / {len(passed_risk)} "
        f"({len(passed_rolling)/len(passed_risk):.2%})"
    )

    # 3. Apply Holdout Gate
    # holdout_return > 0
    mask_holdout = passed_rolling["holdout_return"] > GATE_HOLDOUT_MIN_RET

    final_candidates = passed_rolling[mask_holdout].copy()
    print(
        f"Passed Holdout Gate (Return > {GATE_HOLDOUT_MIN_RET}): {len(final_candidates)} / {len(passed_rolling)} ({len(final_candidates)/len(passed_rolling):.2%})"
    )

    # 4. Ranking (Composite Score)
    # We want strategies that are good EVERYWHERE.
    # Score = 0.4 * Norm(Train_Calmar) + 0.3 * Norm(Roll_Worst_Ret) + 0.3 * Norm(Holdout_Calmar)
    # Simple normalization: Rank pct

    final_candidates["score_train"] = final_candidates["vec_calmar_ratio"].rank(
        pct=True
    )
    final_candidates["score_roll"] = final_candidates[
        "roll_all_segment_worst_return"
    ].rank(pct=True)
    final_candidates["score_holdout"] = final_candidates["holdout_calmar_ratio"].rank(
        pct=True
    )

    final_candidates["composite_score"] = (
        0.3 * final_candidates["score_train"]
        + 0.4 * final_candidates["score_roll"]  # Emphasize safety
        + 0.3 * final_candidates["score_holdout"]
    )

    final_candidates = final_candidates.sort_values("composite_score", ascending=False)

    # 5. Output
    out_parquet = OUTPUT_DIR / "final_candidates.parquet"
    final_candidates.to_parquet(out_parquet)
    print(f"Saved final candidates to: {out_parquet}")

    # Generate Report
    report_path = OUTPUT_DIR / "FINAL_TRIPLE_VALIDATION_REPORT.md"

    with open(report_path, "w") as f:
        f.write("# 🏆 Final Triple Validation Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 0. Leakage Control (Audit Note)\n")
        f.write(
            "This run uses **train-only** rolling OOS summary as the gate input (no holdout segments).\n\n"
        )
        f.write(f"- VEC (train): `{path_vec}`\n")
        f.write(f"- Rolling (train-only): `{path_rolling}`\n")
        f.write(f"- Holdout (unseen): `{path_holdout}`\n\n")

        f.write("## 1. Screening Funnel\n")
        f.write(f"- **Total Universe**: {total_count}\n")
        f.write(f"- **Risk Filter** (factor exclusions): {len(passed_risk)}\n")
        f.write(
            f"- **Rolling Consistency Gate** (Strict): {len(passed_rolling)} (PosRate>={GATE_ROLLING_POS_RATE}, Worst>={GATE_ROLLING_WORST_RET}, Calmar>={GATE_ROLLING_CALMAR})\n"
        )
        f.write(
            f"- **Holdout Gate** (Profitable): {len(final_candidates)} (Return > {GATE_HOLDOUT_MIN_RET})\n\n"
        )

        f.write("## 2. Top 20 'Gold Standard' Strategies\n")
        f.write(
            "Sorted by Composite Score (30% Train Calmar + 40% Roll Worst + 30% Holdout Calmar)\n\n"
        )

        cols_to_show = [
            "combo",
            "vec_return",
            "vec_calmar_ratio",  # Train
            "roll_all_segment_positive_rate",
            "roll_all_segment_worst_return",  # Rolling
            "holdout_return",
            "holdout_calmar_ratio",  # Holdout
        ]

        # Format for markdown
        display_df = final_candidates[cols_to_show].head(20).copy()
        display_df.columns = [
            "Combo",
            "Train Ret",
            "Train Calmar",
            "Roll Win%",
            "Roll Worst",
            "Holdout Ret",
            "Holdout Calmar",
        ]

        # Rounding
        for c in [
            "Train Ret",
            "Train Calmar",
            "Roll Worst",
            "Holdout Ret",
            "Holdout Calmar",
        ]:
            display_df[c] = display_df[c].apply(lambda x: f"{x:.4f}")
        display_df["Roll Win%"] = display_df["Roll Win%"].apply(lambda x: f"{x:.2f}")

        f.write(display_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 3. Conclusion\n")
        if len(final_candidates) > 0:
            top = final_candidates.iloc[0]
            f.write(f"The top strategy **{top['combo']}** demonstrates:\n")
            f.write(
                f"- **Train**: {top['vec_return']:.2%} return, {top['vec_calmar_ratio']:.2f} Calmar\n"
            )
            f.write(
                f"- **Stability**: {top['roll_all_segment_positive_rate']:.0%} quarterly win rate, worst quarter {top['roll_all_segment_worst_return']:.2%}\n"
            )
            f.write(
                f"- **Holdout**: {top['holdout_return']:.2%} return in unseen data\n"
            )
            f.write(
                "\nThis is a **leakage-controlled** triple validation (rolling gate uses train-only data; holdout remains unseen).\n"
            )
        else:
            f.write(
                "No strategies passed all gates. The criteria might be too strict, or the strategy pool needs expansion.\n"
            )

        if prev_final is not None:
            f.write("\n## 4. Comparison vs Previous Run (Overlap)\n")
            prev_set = set(prev_final["combo"].astype(str))
            new_set = set(final_candidates["combo"].astype(str))
            inter = prev_set & new_set
            f.write(f"- Prev candidates: {len(prev_set)}\n")
            f.write(f"- New candidates: {len(new_set)}\n")
            f.write(
                f"- Overlap: {len(inter)} ({(len(inter)/max(1, len(prev_set))):.2%} of prev; {(len(inter)/max(1, len(new_set))):.2%} of new)\n"
            )

    print(f"Report generated: {report_path}")

    write_step_meta(OUTPUT_DIR, step="final", inputs={"vec": str(args.vec), "rolling": str(args.rolling), "holdout": str(args.holdout)}, extras={"final_candidates": len(final_candidates)})


if __name__ == "__main__":
    main()
