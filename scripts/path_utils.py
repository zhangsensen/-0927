from __future__ import annotations

import os
from pathlib import Path

import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_paths(params_file: str | None = None) -> dict:
    root = get_project_root()
    cfg = _load_yaml(root / (params_file or "configs/etf_pools.yaml"))
    paths = cfg.get("paths") or {}
    output_root = Path(
        os.environ.get(
            "ETF_OUTPUT_DIR",
            paths.get("output_root", root / "factor_output/etf_rotation_production"),
        )
    )
    logs_root = Path(os.environ.get("LOGS_ROOT", paths.get("logs_root", root / "logs")))
    raw_root = Path(os.environ.get("RAW_DATA_DIR", paths.get("raw_root", root / "raw")))
    snapshots_root = Path(
        os.environ.get(
            "SNAPSHOTS_ROOT", paths.get("snapshots_root", root / "snapshots")
        )
    )
    return {
        "project_root": root,
        "output_root": output_root,
        "logs_root": logs_root,
        "raw_root": raw_root,
        "snapshots_root": snapshots_root,
    }


def get_ci_thresholds(params_file: str | None = None) -> dict:
    cfg = _load_yaml(get_project_root() / (params_file or "configs/etf_pools.yaml"))
    return cfg.get(
        "ci_thresholds",
        {
            "min_annual_return": 0.08,
            "max_drawdown": -0.30,
            "min_sharpe": 0.5,
            "min_winrate": 0.45,
            "min_coverage": 0.80,
            "min_factors": 8,
        },
    )


def get_capacity_defaults(params_file: str | None = None) -> dict:
    cfg = _load_yaml(get_project_root() / (params_file or "configs/etf_pools.yaml"))
    return cfg.get(
        "capacity_defaults",
        {
            "target_capital": 7000000,
            "max_single_weight": 0.25,
            "max_adv_pct": 0.05,
        },
    )
