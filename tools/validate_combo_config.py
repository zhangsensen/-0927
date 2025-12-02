#!/usr/bin/env python3
"""Compare combo_wfo configuration against the optimized baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

DEFAULT_TARGET = Path("configs/combo_wfo_config.yaml")
DEFAULT_BASELINE = Path("configs/combo_wfo_config.yaml")
KEY_PATHS: Sequence[Sequence[str]] = (
    ("backtest", "commission_rate"),
    ("backtest", "initial_capital"),
    ("backtest", "lookback_window"),
    ("data", "data_dir"),
    ("data", "cache_dir"),
    ("data", "start_date"),
    ("data", "end_date"),
)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def extract_value(cfg: dict[str, Any], key_path: Iterable[str]) -> Any:
    node: Any = cfg
    for key in key_path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def compare_configs(target: dict[str, Any], baseline: dict[str, Any], *, verbose: bool) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for path in KEY_PATHS:
        tgt_val = extract_value(target, path)
        base_val = extract_value(baseline, path)
        if tgt_val != base_val:
            diffs.append({
                "path": ".".join(path),
                "target": tgt_val,
                "baseline": base_val,
            })
            if verbose:
                print(f"[DIFF] {'.'.join(path)} => target={tgt_val!r}, baseline={base_val!r}")
        elif verbose:
            print(f"[OK]   {'.'.join(path)} => {tgt_val!r}")
    return diffs


def verify_compatibility_section(target: dict[str, Any], baseline_path: Path, target_path: Path) -> list[str]:
    issues: list[str] = []
    compat = target.get("compatibility")
    if not isinstance(compat, dict):
        issues.append("compatibility 节不存在或不是映射")
        return issues
    expected_path = str(baseline_path.resolve())
    actual_path = compat.get("optimized_config_path")
    if actual_path:
        candidate = (target_path.parent / Path(actual_path)).expanduser().resolve()
        if candidate != Path(expected_path):
            issues.append(
                "compatibility.optimized_config_path 不匹配当前 baseline 路径"
            )
    else:
        issues.append(
            "compatibility.optimized_config_path 缺失"
        )
    if "legacy_optimized_mode" not in compat:
        issues.append("compatibility.legacy_optimized_mode 缺失")
    if "strict_validation" not in compat:
        issues.append("compatibility.strict_validation 缺失")
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="待验证的 combo_wfo_config.yaml 路径")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="稳定仓基准配置路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出检查结果")
    parser.add_argument("--verbose", action="store_true", help="打印每个 key 的比对情况")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_cfg = load_yaml(args.target)
    baseline_cfg = load_yaml(args.baseline)

    diffs = compare_configs(target_cfg, baseline_cfg, verbose=args.verbose)
    compat_issues = verify_compatibility_section(target_cfg, args.baseline, args.target)

    ok = not diffs and not compat_issues
    result = {
        "target": str(args.target),
        "baseline": str(args.baseline),
        "differences": diffs,
        "compatibility_issues": compat_issues,
        "status": "ok" if ok else "failed",
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if ok:
            print("✅ 配置验证通过：关键字段与基准一致")
        else:
            print("❌ 配置验证存在差异")
            if diffs:
                print("--- 差异字段 ---")
                for diff in diffs:
                    print(f"  {diff['path']}: target={diff['target']!r} baseline={diff['baseline']!r}")
            if compat_issues:
                print("--- 兼容性配置问题 ---")
                for issue in compat_issues:
                    print(f"  - {issue}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
