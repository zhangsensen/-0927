#!/usr/bin/env python3
"""
Validate live allocation template:
 - For each strategy_combo_key, ensure target_weight sums to 1.0 (+/- tol)
 - Ensure capital_alloc is consistent within a strategy group
 - If live_strategies.csv present, ensure total capital equals the plan
 - Ensure all symbols are in configs/etf_pools.yaml A_SHARE_LIVE (if defined)

Usage:
  python scripts/validate_live_alloc.py \
    --alloc etf_rotation_optimized/results/run_YYYYMMDD_HHMMSS/selection/live/live_alloc_YYYYMMDD.csv \
    --strategies etf_rotation_optimized/results/run_YYYYMMDD_HHMMSS/selection/live/live_strategies.csv \
    --pools configs/etf_pools.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import math
import csv
import json
from collections import defaultdict

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

TOL = 1e-6


def read_csv(path: Path) -> list[dict]:
    rows = []
    with path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows


def parse_float(x: str | None, default: float = float('nan')) -> float:
    if x is None or x == '':
        return default
    try:
        return float(x)
    except Exception:
        return default


def load_live_symbols(pools_yaml: Path) -> set[str]:
    if yaml is None:
        return set()
    data = yaml.safe_load(pools_yaml.read_text(encoding='utf-8'))
    pools = (data or {}).get('pools', {})
    live = pools.get('A_SHARE_LIVE') or {}
    syms = live.get('symbols') or []
    return set(syms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alloc', required=True, help='Path to live_alloc_YYYYMMDD.csv')
    ap.add_argument('--strategies', required=False, help='Path to live_strategies.csv (optional for capital check)')
    ap.add_argument('--pools', required=False, help='Path to etf_pools.yaml for symbol whitelist check')
    ap.add_argument('--tol', type=float, default=0.0005, help='Tolerance for weight sum check')
    args = ap.parse_args()

    alloc_path = Path(args.alloc)
    if not alloc_path.exists():
        print(f"ERROR: alloc file not found: {alloc_path}")
        sys.exit(2)

    rows = read_csv(alloc_path)
    if not rows:
        print('ERROR: alloc file is empty')
        sys.exit(2)

    # group by strategy
    by_strat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = r.get('strategy_combo_key') or ''
        by_strat[key].append(r)

    errors: list[str] = []
    warnings: list[str] = []

    # symbol whitelist
    whitelist: set[str] = set()
    if args.pools:
        pools_yaml = Path(args.pools)
        if pools_yaml.exists():
            whitelist = load_live_symbols(pools_yaml)
        else:
            warnings.append(f"pools yaml not found: {pools_yaml}")

    total_cap_from_alloc = 0.0
    seen_strats: set[str] = set()

    for strat, grp in by_strat.items():
        # weight sum
        wsum = 0.0
        caps: set[float] = set()
        for r in grp:
            wsum += parse_float(r.get('target_weight'), 0.0)
            c = parse_float(r.get('capital_alloc'))
            if not math.isnan(c):
                caps.add(round(c, 6))
            sym = r.get('symbol') or ''
            if whitelist and sym and sym not in whitelist:
                warnings.append(f"{strat}: symbol {sym} not in A_SHARE_LIVE whitelist")

        if abs(wsum - 1.0) > args.tol:
            errors.append(f"{strat}: weight sum={wsum:.6f} deviates from 1 by > tol {args.tol}")

        if len(caps) > 1:
            errors.append(f"{strat}: inconsistent capital_alloc values per row: {sorted(caps)}")

        # Add once per strategy
        if strat not in seen_strats and caps:
            total_cap_from_alloc += list(caps)[0]
            seen_strats.add(strat)

    # compare to strategies plan if provided
    if args.strategies:
        sp = Path(args.strategies)
        if sp.exists():
            srows = read_csv(sp)
            plan_total = 0.0
            for r in srows:
                plan_total += parse_float(r.get('capital_alloc'), 0.0)
            if abs(plan_total - total_cap_from_alloc) > 0.5:
                errors.append(
                    f"Total capital mismatch: alloc={total_cap_from_alloc:.2f} vs plan={plan_total:.2f} (>0.5 diff)"
                )
        else:
            warnings.append(f"strategies plan file not found: {sp}")

    status = 'PASS' if not errors else 'FAIL'
    report = {
        'status': status,
        'alloc_file': str(alloc_path),
        'strategy_count': len(by_strat),
        'total_cap_from_alloc': round(total_cap_from_alloc, 2),
        'errors': errors,
        'warnings': warnings,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(status)
    sys.exit(0 if status == 'PASS' else 1)


if __name__ == '__main__':
    main()
