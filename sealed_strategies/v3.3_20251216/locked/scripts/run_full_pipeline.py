#!/usr/bin/env python3
"""
全链路重跑流水线 (Pipeline)
================================================================================
目标：一键执行从 WFO 筛选到最终验证的全流程，确保 Regime Gate 纳入筛选。

流程步骤：
1. WFO (Walk-Forward Optimization): 全空间搜索，纳入 Regime Gate (OOS收益模拟)
2. VEC (Vectorized Backtest): 对 WFO Top-N 进行精细化回测
3. BT (Backtrader Audit): 对 VEC 结果进行事件驱动审计
4. Rolling (Consistency): 滚动一致性验证 (Train-only)
5. Holdout (Validation): 冷数据验证
6. Final Selection: 三重验证筛选 (Train + Rolling + Holdout)

用法：
    uv run python scripts/run_full_pipeline.py --top-n 200 --n-jobs 24
"""

import sys
import os
import argparse
import subprocess
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_training_end_date_from_config(config_path: Path) -> str | None:
    if not config_path.exists():
        return None
    config = _load_config(config_path)
    return (config.get("data") or {}).get("training_end_date")


def _materialize_config(
    base_config_path: Path, *, regime_gate: str, out_dir: Path
) -> Path:
    """Create an effective config file for this pipeline run.

    regime_gate:
      - "auto": use base config as-is
      - "on"  : force enable gate
      - "off" : force disable gate
    """
    if regime_gate == "auto":
        return base_config_path

    cfg = _load_config(base_config_path)
    cfg.setdefault("backtest", {})
    cfg["backtest"].setdefault("regime_gate", {})
    cfg["backtest"]["regime_gate"]["enabled"] = True if regime_gate == "on" else False

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"combo_wfo_config__regime_gate_{regime_gate}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    return out_path


def run_command(cmd: list[str], env: dict = None, check: bool = True):
    """Run a shell command and stream output."""
    cmd_str = " ".join(cmd)
    logger.info(f"🚀 Running: {cmd_str}")

    # Merge env
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    try:
        subprocess.run(cmd, env=full_env, check=check, cwd=ROOT)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Command failed: {cmd_str}")
        sys.exit(e.returncode)


def get_latest_run_dir(pattern: str = "run_*") -> Path:
    """Find the latest run directory in results/."""
    results_dir = ROOT / "results"
    dirs = sorted([d for d in results_dir.glob(pattern) if d.is_dir()])
    if not dirs:
        raise FileNotFoundError(
            f"No directory matching '{pattern}' found in {results_dir}"
        )
    return dirs[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Full Strategy Pipeline (WFO -> VEC -> BT -> Final)"
    )
    parser.add_argument(
        "--top-n", type=int, default=200, help="Number of top combos to keep from WFO"
    )
    parser.add_argument("--n-jobs", type=int, default=16, help="Parallel workers")
    parser.add_argument(
        "--skip-wfo", action="store_true", help="Skip WFO step (use latest run)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (yaml)。默认使用 configs/combo_wfo_config.yaml",
    )
    parser.add_argument(
        "--regime-gate",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Regime gate 开关：auto=按配置文件原样；on/off=为本次运行强制开/关（不改默认配置文件）",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🔥 Starting Full Pipeline at {timestamp}")

    base_config_path = (
        Path(args.config) if args.config else (ROOT / "configs/combo_wfo_config.yaml")
    )
    diagnostics_dir = ROOT / "results" / "diagnostics" / f"pipeline_run_{timestamp}"
    effective_config_path = _materialize_config(
        base_config_path, regime_gate=args.regime_gate, out_dir=diagnostics_dir
    )

    training_end_date = _load_training_end_date_from_config(effective_config_path)
    logger.info(f"🧾 Config: {effective_config_path}")
    logger.info(f"🧯 Regime gate mode: {args.regime_gate}")
    if training_end_date:
        logger.info(f"📌 training_end_date: {training_end_date}")

    env_cfg = {"WFO_CONFIG_PATH": str(effective_config_path)}

    # 1. WFO (Full Space)
    if not args.skip_wfo:
        logger.info("=" * 80)
        logger.info("1️⃣  STEP 1: WFO (Full Space Search)")
        logger.info("=" * 80)
        # Use run_robust_combo_wfo.py for full space search
        # Pass RB_RESULT_TS to coordinate output dir if needed, but WFO generates its own
        run_command(
            ["uv", "run", "python", "src/etf_strategy/run_robust_combo_wfo.py"],
            env=env_cfg,
        )
    else:
        logger.info("⏭️  Skipping WFO (using latest result)")

    # Identify WFO output
    wfo_dir = get_latest_run_dir("run_*")
    logger.info(f"📂 Using WFO results from: {wfo_dir}")

    # Check for top_combos.csv or parquet
    wfo_combos = wfo_dir / "top_combos.csv"
    if not wfo_combos.exists():
        # Try parquet
        wfo_combos = wfo_dir / "top100_by_ic.parquet"

    if not wfo_combos.exists():
        # Fallback to all_combos if top not found
        wfo_combos = wfo_dir / "full_combo_results.csv"

    if not wfo_combos.exists():
        logger.error(f"❌ Could not find combo results in {wfo_dir}")
        sys.exit(1)

    # 2. VEC Backtest (Top-N)
    logger.info("=" * 80)
    logger.info("2️⃣  STEP 2: VEC Backtest (Top-N)")
    logger.info("=" * 80)

    # We use run_full_space_vec_backtest.py but we might need to adapt it to accept a specific file
    # Currently it auto-detects latest run_*. Let's rely on that behavior or pass --combos if supported.
    # The script `scripts/run_full_space_vec_backtest.py` supports `--combos`.

    # Convert CSV to Parquet if needed for VEC script compatibility
    vec_input = wfo_combos
    if wfo_combos.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(wfo_combos)
        vec_input = wfo_dir / "wfo_combos_converted.parquet"
        df.to_parquet(vec_input)
        logger.info(f"🔄 Converted CSV to Parquet: {vec_input}")

    run_command(
        [
            "uv",
            "run",
            "python",
            "scripts/run_full_space_vec_backtest.py",
            "--combos",
            str(vec_input),
            "--config",
            str(effective_config_path),
        ],
        env=env_cfg,
    )

    # Identify VEC output
    vec_dir = get_latest_run_dir("vec_from_wfo_*")
    vec_results = vec_dir / "full_space_results.parquet"
    logger.info(f"📂 VEC Results: {vec_results}")

    # 3. BT Audit (Top-N)
    logger.info("=" * 80)
    logger.info("3️⃣  STEP 3: BT Audit (Backtrader)")
    logger.info("=" * 80)

    run_command(
        [
            "uv",
            "run",
            "python",
            "scripts/batch_bt_backtest.py",
            "--combos",
            str(vec_results),
            "--topk",
            str(args.top_n),
            "--sort-by",
            "vec_calmar_ratio",  # Sort by VEC Calmar
            "--config",
            str(effective_config_path),
        ],
        env=env_cfg,
    )

    bt_dir = get_latest_run_dir("bt_backtest_*")
    bt_results = bt_dir / "bt_results.parquet"
    logger.info(f"📂 BT Results: {bt_results}")

    # 4. Rolling Consistency (Train-only)
    logger.info("=" * 80)
    logger.info("4️⃣  STEP 4: Rolling Consistency (Train-only)")
    logger.info("=" * 80)

    rolling_cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_rolling_oos_consistency.py",
        "--config",
        str(effective_config_path),
        "--input",
        str(vec_results),
        "--top-n",
        str(args.top_n),
        "--n-jobs",
        str(args.n_jobs),
    ]
    # Critical: avoid holdout leakage into rolling gate by truncating to training_end_date
    if training_end_date:
        rolling_cmd += ["--end-date", str(training_end_date)]
    run_command(rolling_cmd, env=env_cfg)

    rolling_dir = get_latest_run_dir("rolling_oos_consistency_*")
    rolling_results = rolling_dir / "rolling_oos_summary.parquet"
    logger.info(f"📂 Rolling Results: {rolling_results}")

    # 5. Holdout Validation
    logger.info("=" * 80)
    logger.info("5️⃣  STEP 5: Holdout Validation")
    logger.info("=" * 80)

    run_command(
        [
            "uv",
            "run",
            "python",
            "scripts/run_holdout_validation.py",
            "--config",
            str(effective_config_path),
            "--training-results",
            str(vec_results),
            "--top-n",
            str(args.top_n),
            "--n-jobs",
            str(args.n_jobs),
        ],
        env=env_cfg,
    )

    holdout_dir = get_latest_run_dir("holdout_validation_*")
    holdout_results = holdout_dir / "holdout_validation_results.parquet"
    logger.info(f"📂 Holdout Results: {holdout_results}")

    # 6. Final Triple Validation
    logger.info("=" * 80)
    logger.info("6️⃣  STEP 6: Final Triple Validation & Selection")
    logger.info("=" * 80)

    run_command(
        [
            "uv",
            "run",
            "python",
            "scripts/final_triple_validation.py",
            "--vec",
            str(vec_results),
            "--rolling",
            str(rolling_results),
            "--holdout",
            str(holdout_results),
        ]
    )

    final_dir = get_latest_run_dir("final_triple_validation_*")
    final_report = final_dir / "FINAL_TRIPLE_VALIDATION_REPORT.md"

    logger.info("=" * 80)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"📄 Final Report: {final_report}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
