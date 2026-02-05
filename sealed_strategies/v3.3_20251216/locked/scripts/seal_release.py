#!/usr/bin/env python3
"""scripts/seal_release.py

Seal a strategy release into `sealed_strategies/`.

This tool copies critical artifacts (results/config/scripts/docs) into a versioned
folder, generates a MANIFEST.json and CHECKSUMS.sha256 for auditability.

Design goal: reproducible, immutable, and easy to locate.

Usage example:

uv run python scripts/seal_release.py \
  --version v3.2 \
  --date 20251214 \
  --final-candidates results/final_triple_validation_20251214_011753/final_candidates.parquet \
  --bt-results results/bt_backtest_full_20251214_013635/bt_results.parquet \
  --production-dir results/production_pack_20251214_014022
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Inputs:
    version: str
    date: str
    final_candidates: Path
    bt_results: Path
    production_dir: Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
        return out.decode().strip()
    except Exception:
        return "NO_GIT"


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal a strategy release")
    parser.add_argument("--version", required=True, help="Release version, e.g. v3.2")
    parser.add_argument(
        "--date", required=True, help="Release date yyyymmdd, e.g. 20251214"
    )
    parser.add_argument(
        "--final-candidates", required=True, help="Path to final_candidates.parquet"
    )
    parser.add_argument(
        "--bt-results", required=True, help="Path to bt_results.parquet"
    )
    parser.add_argument(
        "--production-dir", required=True, help="Path to production pack directory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow sealing into an existing version directory (files will be overwritten).",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()

    inputs = Inputs(
        version=args.version,
        date=args.date,
        final_candidates=Path(args.final_candidates),
        bt_results=Path(args.bt_results),
        production_dir=Path(args.production_dir),
    )

    # Validate inputs exist
    for p in [inputs.final_candidates, inputs.bt_results, inputs.production_dir]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    prod_top = inputs.production_dir / "production_candidates.parquet"
    prod_all = inputs.production_dir / "production_all_candidates.parquet"
    prod_report = inputs.production_dir / "PRODUCTION_REPORT.md"

    for p in [prod_top, prod_all, prod_report]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    # Validate schema (audit-critical)
    cand = pd.read_parquet(inputs.final_candidates)
    _require_columns(
        cand,
        ["combo", "vec_return", "holdout_return", "composite_score"],
        "final_candidates",
    )

    bt = pd.read_parquet(inputs.bt_results)
    _require_columns(
        bt,
        [
            "combo",
            "bt_return",
            "bt_train_return",
            "bt_holdout_return",
            "bt_max_drawdown",
            "bt_calmar_ratio",
            "bt_profit_factor",
            "bt_total_trades",
            "bt_margin_failures",
        ],
        "bt_results",
    )

    sealed_root = repo_root / "sealed_strategies"
    sealed_dir = sealed_root / f"{inputs.version}_{inputs.date}"
    if sealed_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Sealed directory already exists: {sealed_dir}. Use --force to overwrite."
            )
        if not sealed_dir.is_dir():
            raise NotADirectoryError(str(sealed_dir))
    else:
        sealed_dir.mkdir(parents=True, exist_ok=False)

    # Copy artifacts
    artifacts: list[tuple[str, Path, Path]] = []

    def add(name: str, src: Path, rel_dst: str) -> None:
        dst = sealed_dir / rel_dst
        _copy(src, dst)
        artifacts.append((name, src, dst))

    add(
        "final_candidates",
        inputs.final_candidates,
        "artifacts/final_candidates.parquet",
    )
    add("bt_results", inputs.bt_results, "artifacts/bt_results.parquet")
    add("production_top", prod_top, "artifacts/production_candidates.parquet")
    add("production_all", prod_all, "artifacts/production_all_candidates.parquet")
    add("production_report", prod_report, "artifacts/PRODUCTION_REPORT.md")

    # Lock configs + scripts (audit surface)
    config = repo_root / "configs" / "combo_wfo_config.yaml"
    if config.exists():
        add("combo_wfo_config", config, "locked/configs/combo_wfo_config.yaml")

    # Lock source code (CRITICAL for reproducibility)
    src_dir = repo_root / "src"
    if src_dir.exists():
        # We use shutil.copytree for directories, but our helper is for files.
        # Let's manually walk and add files to keep the manifest consistent.
        for p in sorted(src_dir.rglob("*")):
            if (
                p.is_file()
                and "__pycache__" not in p.parts
                and not p.name.endswith(".pyc")
            ):
                rel = p.relative_to(repo_root)
                add(f"src_{p.name}", p, f"locked/{rel}")

    # Lock environment definition
    for env_file in ["pyproject.toml", "uv.lock", "Makefile"]:
        p = repo_root / env_file
        if p.exists():
            add(env_file, p, f"locked/{env_file}")

    for script in [
        repo_root / "scripts" / "batch_bt_backtest.py",
        repo_root / "scripts" / "generate_production_pack.py",
        repo_root / "scripts" / "final_triple_validation.py",
    ]:
        if script.exists():
            add(script.name, script, f"locked/scripts/{script.name}")

    for doc in [
        repo_root / "docs" / "PRODUCTION_STRATEGIES_V3_2.md",
        repo_root / "docs" / "PRODUCTION_STRATEGY_LIST_V3_2.md",
        repo_root / "docs" / "RELEASE_NOTES_V3_2.md",
        repo_root / "docs" / "QUICK_REFERENCE.md",
    ]:
        if doc.exists():
            add(doc.name, doc, f"locked/docs/{doc.name}")

    # Manifest
    manifest = {
        "version": inputs.version,
        "date": inputs.date,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_head(repo_root),
        "source_paths": {
            "final_candidates": str(inputs.final_candidates),
            "bt_results": str(inputs.bt_results),
            "production_dir": str(inputs.production_dir),
        },
        "counts": {
            "final_candidates": int(len(cand)),
            "bt_results": int(len(bt)),
        },
        "notes": {
            "ground_truth": "BT (Backtrader) metrics are the production truth; VEC is screening only.",
            "split": "Train/Holdout split at training_end_date from combo_wfo_config.yaml.",
        },
        "files": [
            {
                "name": name,
                "source": str(src),
                "sealed": str(dst.relative_to(sealed_dir)),
            }
            for (name, src, dst) in artifacts
        ],
    }

    (sealed_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Minimal reproduce guide
    reproduce = sealed_dir / "REPRODUCE.md"
    reproduce.write_text(
        "# 复现指南\n\n"
        "本封板版本的产物已被复制到本目录，优先以本目录 artifacts 为准。\n\n"
        "## 1. 校验（防篡改）\n\n"
        "```bash\n"
        "cd sealed_strategies/" + f"{inputs.version}_{inputs.date}" + "\n"
        "sha256sum -c CHECKSUMS.sha256\n"
        "```\n\n"
        "## 2. 环境准备\n\n"
        "本封板包含完整的源码快照 (`locked/src`) 和环境定义 (`locked/pyproject.toml`)。\n\n"
        "```bash\n"
        "# 假设已安装 uv\n"
        "cd locked\n"
        "uv sync --dev\n"
        "```\n\n"
        "## 3. 运行复现\n\n"
        "使用 locked 目录下的脚本与源码进行复现，确保不受主分支变更影响。\n\n"
        "```bash\n"
        "# 运行 BT 审计 (使用 locked 源码)\n"
        "uv run python scripts/batch_bt_backtest.py\n"
        "```\n\n"
        "## 4. 关键产物\n\n"
        "- artifacts/final_candidates.parquet\n"
        "- artifacts/bt_results.parquet\n"
        "- artifacts/production_candidates.parquet\n"
        "- artifacts/PRODUCTION_REPORT.md\n",
        encoding="utf-8",
    )

    # Checksums (for ALL files under sealed_dir)
    # This makes the sealed release audit-grade and tamper-evident.
    checksum_lines: list[str] = []
    all_files = sorted(
        [
            p
            for p in sealed_dir.rglob("*")
            if p.is_file() and p.name != "CHECKSUMS.sha256"
        ]
    )
    for fpath in all_files:
        checksum_lines.append(f"{_sha256_file(fpath)}  {fpath.relative_to(sealed_dir)}")
    (sealed_dir / "CHECKSUMS.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    print(f"✅ Sealed release created: {sealed_dir}")


if __name__ == "__main__":
    main()
