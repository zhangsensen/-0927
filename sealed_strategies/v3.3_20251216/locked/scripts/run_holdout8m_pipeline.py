#!/usr/bin/env python3
"""One-click pipeline for the "Holdout ~8 months" protocol.

Protocol (fixed):
- end_date = 2025-12-12
- training_end_date = 2025-04-30
- holdout: (first trading day after training_end_date) .. end_date

Stages:
1) WFO (train only)
2) VEC (train only)
3) Rolling OOS consistency (train-only gate)
4) Holdout validation (unseen)
5) Final triple validation merge + gates
6) BT audit on final candidates (optional)

Run:
  uv run python scripts/run_holdout8m_pipeline.py

Notes:
- Uses configs/combo_wfo_config.yaml by default.
- Assumes you already ran `uv sync` at least once.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    wfo_dir: Path
    combos_parquet: Path
    vec_dir: Path
    vec_results: Path
    rolling_dir: Path
    rolling_summary: Path
    holdout_dir: Path
    holdout_results: Path
    triple_dir: Path
    final_candidates: Path


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _uv_run_python(
    script: str, args: list[str] | None = None, env: dict[str, str] | None = None
) -> None:
    argv = ["uv", "run", "python", script]
    if args:
        argv.extend(args)
    _run(argv, env=env)


def _latest_dir(glob_pat: str) -> Path:
    dirs = sorted(
        [
            p
            for p in (ROOT / "results").glob(glob_pat)
            if p.is_dir() and not p.is_symlink()
        ]
    )
    if not dirs:
        raise FileNotFoundError(f"No dirs matched: results/{glob_pat}")
    return dirs[-1]


def _pick_existing(dir_path: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = dir_path / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of {candidates} exists under {dir_path}")


def _assert_protocol(config_path: Path) -> tuple[str, str, str]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = cfg.get("data", {})
    start = str(data.get("start_date"))
    end = str(data.get("end_date"))
    train_end = str(data.get("training_end_date"))

    if end != "2025-12-12":
        raise ValueError(f"end_date must be 2025-12-12, got {end}")
    if train_end != "2025-04-30":
        raise ValueError(f"training_end_date must be 2025-04-30, got {train_end}")
    if start != "2020-01-01":
        raise ValueError(f"start_date must be 2020-01-01, got {start}")

    return start, train_end, end


def main() -> None:
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    _assert_protocol(config_path)

    umbrella_dir = (
        ROOT / "results" / f"holdout8m_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    umbrella_dir.mkdir(parents=True, exist_ok=True)

    # Ensure WFO uses the same config (it reads WFO_CONFIG_PATH).
    env = os.environ.copy()
    env["WFO_CONFIG_PATH"] = str(config_path)

    # 1) WFO
    _uv_run_python("src/etf_strategy/run_combo_wfo.py", env=env)
    wfo_dir = _latest_dir("run_*")
    combos_parquet = _pick_existing(
        wfo_dir, ["all_combos.parquet", "top100_by_ic.parquet"]
    )

    (umbrella_dir / "wfo").symlink_to(wfo_dir.relative_to(umbrella_dir.parent))
    (umbrella_dir / "wfo_combos.parquet").symlink_to(
        combos_parquet.relative_to(umbrella_dir.parent)
    )

    # 2) VEC (train only)
    _uv_run_python(
        "scripts/run_full_space_vec_backtest.py",
        args=["--combos", str(combos_parquet)],
        env=env,
    )
    vec_dir = _latest_dir("vec_from_wfo_*")
    vec_results = _pick_existing(
        vec_dir, ["full_space_results.parquet", "full_space_results.csv"]
    )

    (umbrella_dir / "vec").symlink_to(vec_dir.relative_to(umbrella_dir.parent))
    (umbrella_dir / "vec_results").symlink_to(
        vec_results.relative_to(umbrella_dir.parent)
    )

    # 3) Rolling (train-only gate) => force end-date=training_end_date
    train_end = "2025-04-30"
    _uv_run_python(
        "scripts/run_rolling_oos_consistency.py",
        args=["--input", str(vec_results), "--segment", "Q", "--end-date", train_end],
        env=env,
    )
    rolling_dir = _latest_dir("rolling_oos_consistency_*")
    rolling_summary = _pick_existing(rolling_dir, ["rolling_oos_summary.parquet"])

    (umbrella_dir / "rolling").symlink_to(rolling_dir.relative_to(umbrella_dir.parent))
    (umbrella_dir / "rolling_oos_summary.parquet").symlink_to(
        rolling_summary.relative_to(umbrella_dir.parent)
    )

    # 4) Holdout validation (full end_date; internally uses training_end_date boundary)
    _uv_run_python(
        "scripts/run_holdout_validation.py",
        args=[
            "--training-results",
            str(vec_results),
            "--n-jobs",
            "-1",
            "--prefer",
            "threads",
        ],
        env=env,
    )
    holdout_dir = _latest_dir("holdout_validation_*")
    holdout_results = _pick_existing(
        holdout_dir, ["holdout_validation_results.parquet"]
    )

    (umbrella_dir / "holdout").symlink_to(holdout_dir.relative_to(umbrella_dir.parent))
    (umbrella_dir / "holdout_validation_results.parquet").symlink_to(
        holdout_results.relative_to(umbrella_dir.parent)
    )

    # 5) Final triple validation
    _uv_run_python(
        "scripts/final_triple_validation.py",
        args=[
            "--vec",
            str(vec_results),
            "--rolling",
            str(rolling_summary),
            "--holdout",
            str(holdout_results),
            "--prev-final-candidates",
            "",
        ],
        env=env,
    )
    triple_dir = _latest_dir("final_triple_validation_*")
    final_candidates = _pick_existing(triple_dir, ["final_candidates.parquet"])

    (umbrella_dir / "triple").symlink_to(triple_dir.relative_to(umbrella_dir.parent))
    (umbrella_dir / "final_candidates.parquet").symlink_to(
        final_candidates.relative_to(umbrella_dir.parent)
    )

    # 6) BT audit (optional but recommended)
    # Default: run on all final candidates (usually not huge). If it's large, rerun manually with --topk.
    _uv_run_python(
        "scripts/batch_bt_backtest.py",
        args=["--combos", str(final_candidates)],
        env=env,
    )

    # Write a tiny manifest for reproducibility
    manifest: dict[str, object] = {
        "protocol": {
            "start_date": "2020-01-01",
            "training_end_date": "2025-04-30",
            "end_date": "2025-12-12",
            "freq": 3,
            "pos_size": 2,
        },
        "config_path": str(config_path),
        "artifacts": {
            "wfo_dir": str(wfo_dir),
            "combos_parquet": str(combos_parquet),
            "vec_dir": str(vec_dir),
            "vec_results": str(vec_results),
            "rolling_dir": str(rolling_dir),
            "rolling_summary": str(rolling_summary),
            "holdout_dir": str(holdout_dir),
            "holdout_results": str(holdout_results),
            "triple_dir": str(triple_dir),
            "final_candidates": str(final_candidates),
        },
    }
    (umbrella_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    # Collect paths
    # Note: bt_backtest_* output dir is timestamped by the BT script; we won't try to parse it here.
    print("\n✅ Pipeline finished.")
    print("Artifacts:")
    print(f"- WFO:     {wfo_dir.relative_to(ROOT)}")
    print(f"- VEC:     {vec_dir.relative_to(ROOT)}")
    print(f"- Rolling: {rolling_dir.relative_to(ROOT)}")
    print(f"- Holdout: {holdout_dir.relative_to(ROOT)}")
    print(f"- Triple:  {triple_dir.relative_to(ROOT)}")
    print(f"- Final candidates: {final_candidates.relative_to(ROOT)}")
    print(f"- Umbrella dir: {umbrella_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
