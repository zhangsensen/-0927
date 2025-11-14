#!/usr/bin/env python3
"""
Generate mock ETF orders for a small capital test.

Reads live allocation CSV and produces order_dryrun.csv with fields:
  strategy_combo_key,symbol,target_weight,strategy_capital,target_notional,approx_price,approx_shares

Approx prices loaded from a JSON/YAML snapshot or provided via --price symbol=price pairs.
If no price found, row flagged with price_missing=1.

Usage examples:
  python scripts/generate_mock_orders.py \
    --alloc path/to/live_alloc_20251109.csv \
    --limit-strategies 2 --fraction 0.05 --price 510300.SH=3.82 159915.SZ=1.87
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import json
import sys
from collections import defaultdict

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def read_alloc(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def parse_price_pairs(pairs: list[str]) -> dict[str, float]:
    out = {}
    for p in pairs:
        if '=' not in p:
            continue
        sym, val = p.split('=', 1)
        try:
            out[sym.strip()] = float(val)
        except Exception:
            pass
    return out


def load_snapshot(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    txt = path.read_text(encoding='utf-8')
    # try json then yaml
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            return {k: float(v) for k, v in data.items() if isinstance(v, (int, float, str))}
    except Exception:
        pass
    if yaml is not None:
        try:
            data = yaml.safe_load(txt)
            if isinstance(data, dict):
                return {k: float(v) for k, v in data.items() if isinstance(v, (int, float, str))}
        except Exception:
            pass
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alloc', required=True, help='live allocation csv path')
    ap.add_argument('--fraction', type=float, default=0.05, help='Fraction of per-strategy capital to trade in mock')
    ap.add_argument('--limit-strategies', type=int, default=2, help='Limit number of strategies to include')
    ap.add_argument('--price-snapshot', help='Optional JSON/YAML file with symbol->price')
    ap.add_argument('--price', nargs='*', default=[], help='Manual symbol=price overrides')
    ap.add_argument('--out', default='order_dryrun.csv')
    args = ap.parse_args()

    alloc_rows = read_alloc(Path(args.alloc))
    if not alloc_rows:
        print('ERROR: empty alloc file', file=sys.stderr)
        sys.exit(2)

    # group by strategy
    grouped = defaultdict(list)
    for r in alloc_rows:
        grouped[r['strategy_combo_key']].append(r)

    strategies = list(grouped.keys())[: args.limit_strategies]

    price_map = {}
    if args.price_snapshot:
        price_map.update(load_snapshot(Path(args.price_snapshot)))
    price_map.update(parse_price_pairs(args.price))

    out_rows = []
    for strat in strategies:
        grp = grouped[strat]
        # capital is same per row, take first
        try:
            strat_capital = float(grp[0]['capital_alloc']) * args.fraction
        except Exception:
            strat_capital = 0.0
        for r in grp:
            try:
                tw = float(r['target_weight'])
            except Exception:
                tw = 0.0
            symbol = r['symbol']
            target_notional = strat_capital * tw
            price = price_map.get(symbol)
            if price is None or price <= 0:
                approx_shares = ''
                price_missing = 1
            else:
                approx_shares = int(target_notional // price)
                price_missing = 0
            out_rows.append({
                'strategy_combo_key': strat,
                'symbol': symbol,
                'target_weight': tw,
                'strategy_capital_mock': f"{strat_capital:.2f}",
                'target_notional': f"{target_notional:.2f}",
                'approx_price': price if price is not None else '',
                'approx_shares': approx_shares,
                'price_missing': price_missing,
            })

    # write output
    out_path = Path(args.out)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = list(out_rows[0].keys()) if out_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)
    print(f"Mock order file written: {out_path}")
    missing = sum(1 for r in out_rows if r['price_missing'])
    if missing:
        print(f"WARNING: {missing} rows missing price; provide --price or --price-snapshot for share calc.")


if __name__ == '__main__':
    main()
