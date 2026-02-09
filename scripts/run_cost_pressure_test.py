#!/usr/bin/env python3
"""
Exp2: 三档成本压力测试
================================================================================
对 low / med / high 三档成本分别运行完整 pipeline (WFO → VEC → Rolling → Holdout),
输出对比表辅助决策。

用法:
    uv run python scripts/run_cost_pressure_test.py [--top-n 200] [--skip-wfo]
"""

import sys
import os
import argparse
import subprocess
import yaml
import copy
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


def _run(cmd: list[str], env: dict | None = None):
    logger.info(f"🚀 {' '.join(cmd)}")
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    subprocess.run(cmd, env=full_env, check=True, cwd=ROOT)


def _get_latest(pattern: str) -> Path:
    dirs = sorted([d for d in (ROOT / "results").glob(pattern) if d.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"No dir matching {pattern}")
    return dirs[-1]


def run_tier(tier: str, base_config_path: Path, args) -> dict:
    """Run full pipeline for a single cost tier, return summary dict."""
    logger.info("=" * 80)
    logger.info(f"📊 COST PRESSURE TEST: tier={tier}")
    logger.info("=" * 80)

    # Create temporary config with overridden tier
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    config.setdefault("backtest", {}).setdefault("cost_model", {})
    config["backtest"]["cost_model"]["tier"] = tier

    tmp_dir = ROOT / "results" / "cost_pressure_test_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_config = tmp_dir / f"config_tier_{tier}.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    env_cfg = {"WFO_CONFIG_PATH": str(tmp_config)}

    # 1. WFO (shared across tiers since WFO uses cost_arr in OOS return only)
    if not args.skip_wfo:
        _run(
            ["uv", "run", "python", "src/etf_strategy/run_robust_combo_wfo.py"],
            env=env_cfg,
        )

    # 2. VEC
    wfo_dir = _get_latest("run_*")
    combos_file = wfo_dir / "top100_by_ic.parquet"
    if not combos_file.exists():
        combos_file = wfo_dir / "all_combos.parquet"
    _run(
        [
            "uv", "run", "python", "scripts/run_full_space_vec_backtest.py",
            "--combos", str(combos_file),
            "--config", str(tmp_config),
        ],
        env=env_cfg,
    )
    vec_dir = _get_latest("vec_from_wfo_*")
    vec_results = vec_dir / "full_space_results.parquet"

    # 3. Rolling OOS (train-only)
    training_end = config.get("data", {}).get("training_end_date")
    rolling_cmd = [
        "uv", "run", "python", "scripts/run_rolling_oos_consistency.py",
        "--config", str(tmp_config),
        "--input", str(vec_results),
        "--top-n", str(args.top_n),
        "--n-jobs", str(args.n_jobs),
    ]
    if training_end:
        rolling_cmd += ["--end-date", str(training_end)]
    _run(rolling_cmd, env=env_cfg)
    rolling_dir = _get_latest("rolling_oos_consistency_*")
    rolling_results = rolling_dir / "rolling_oos_summary.parquet"

    # 4. Holdout
    _run(
        [
            "uv", "run", "python", "scripts/run_holdout_validation.py",
            "--config", str(tmp_config),
            "--training-results", str(vec_results),
            "--top-n", str(args.top_n),
            "--n-jobs", str(args.n_jobs),
        ],
        env=env_cfg,
    )
    holdout_dir = _get_latest("holdout_validation_*")
    holdout_results = holdout_dir / "holdout_validation_results.parquet"

    # 5. Final triple validation
    _run(
        [
            "uv", "run", "python", "scripts/final_triple_validation.py",
            "--vec", str(vec_results),
            "--rolling", str(rolling_results),
            "--holdout", str(holdout_results),
        ],
    )
    final_dir = _get_latest("final_triple_validation_*")
    final_parquet = final_dir / "final_candidates.parquet"

    # Collect summary
    summary = {"tier": tier, "candidates": 0, "holdout_median": 0.0, "vec_median": 0.0}
    if final_parquet.exists():
        df = pd.read_parquet(final_parquet)
        summary["candidates"] = len(df)
        if len(df) > 0:
            summary["vec_median"] = float(df["vec_return"].median()) if "vec_return" in df.columns else 0.0
            summary["holdout_median"] = float(df["holdout_return"].median()) if "holdout_return" in df.columns else 0.0
            if "vec_turnover_ann" in df.columns:
                summary["turnover_median"] = float(df["vec_turnover_ann"].median())
            if "vec_cost_drag" in df.columns:
                summary["cost_drag_median"] = float(df["vec_cost_drag"].median())

    # Also read VEC-level stats
    if vec_results.exists():
        df_vec = pd.read_parquet(vec_results)
        summary["vec_all_median"] = float(df_vec["vec_return"].median()) if "vec_return" in df_vec.columns else 0.0
        summary["vec_all_count"] = len(df_vec)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Exp2: 三档成本压力测试")
    parser.add_argument("--top-n", type=int, default=200, help="Top-N combos per tier")
    parser.add_argument("--n-jobs", type=int, default=16, help="Parallel workers")
    parser.add_argument("--skip-wfo", action="store_true", help="Skip WFO (reuse latest)")
    parser.add_argument("--config", type=str, default=None, help="Base config path")
    args = parser.parse_args()

    base_config = Path(args.config) if args.config else ROOT / "configs/combo_wfo_config.yaml"

    tiers = ["low", "med", "high"]
    results = []

    for tier in tiers:
        summary = run_tier(tier, base_config, args)
        results.append(summary)
        logger.info(f"✅ tier={tier}: {summary['candidates']} candidates")

    # Print comparison table
    print("\n" + "=" * 100)
    print("📊 EXP2: 三档成本压力测试对比")
    print("=" * 100)
    print(f"{'Tier':<6} | {'Candidates':<12} | {'VEC Median':<12} | {'Holdout Median':<15} | "
          f"{'Turnover':<10} | {'Cost Drag':<10}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['tier']:<6} | {r['candidates']:<12} | "
            f"{r.get('vec_median', 0)*100:>10.2f}% | "
            f"{r.get('holdout_median', 0)*100:>13.2f}% | "
            f"{r.get('turnover_median', 0):>8.2f} | "
            f"{r.get('cost_drag_median', 0)*100:>8.2f}%"
        )

    # Save CSV
    df = pd.DataFrame(results)
    out_path = ROOT / "results" / "cost_pressure_test_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\n💾 Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
