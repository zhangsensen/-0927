#!/usr/bin/env python3
"""
Convert per-ETF parquet/csv daily files into a consolidated CSV with columns:
  date,symbol,close

Supports two ways to select symbols:
  1) --symbols 510300.SH,510500.SH,...
  2) --alloc live_alloc_YYYYMMDD.csv  (extract unique symbols from alloc)

Input root is a directory containing files like:
  <symbol>_daily_*.parquet  (preferred)
or a single CSV with columns [date,symbol,close] which will be copied/filtered.

Usage:
  python scripts/convert_etf_daily_to_csv.py \
    --root /Users/.../raw/ETF/daily \
    --alloc etf_strategy/results/run_YYYYMMDD_HHMMSS/selection/live/live_alloc_YYYYMMDD.csv \
    --start 2020-01-01 --end 2025-11-09 \
    --out etf_strategy/results/run_YYYYMMDD_HHMMSS/selection/live/your_prices.csv
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Iterable


def parse_date(s: str) -> date:
    s = s.strip()
    # Normalize common variants
    # Try ISO date first
    try:
        if 'T' in s:
            s0 = s.split('T', 1)[0]
            return datetime.strptime(s0, '%Y-%m-%d').date()
        if ' ' in s:
            s0 = s.split(' ', 1)[0]
            return datetime.strptime(s0, '%Y-%m-%d').date()
        if '-' in s and len(s) >= 10:
            return datetime.strptime(s[:10], '%Y-%m-%d').date()
        if len(s) == 8 and s.isdigit():
            return datetime.strptime(s, '%Y%m%d').date()
    except Exception:
        pass
    raise ValueError(f'Unrecognized date format: {s}')


def read_alloc_symbols(path: Path) -> list[str]:
    out = []
    import csv as _csv
    with path.open('r', encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        for r in reader:
            sym = (r.get('symbol') or '').strip()
            if sym and sym not in out:
                out.append(sym)
    return out


def iter_parquet_rows(parquet_path: Path):
    # Try pyarrow first, then pandas
    try:
        import pyarrow.parquet as pq  # type: ignore
        table = pq.read_table(parquet_path)
        cols = {c.lower(): i for i, c in enumerate(table.schema.names)}
        # heuristics for column names
        date_col = next((c for c in ('date', 'trade_date', 'datetime') if c in cols), None)
        close_col = next((c for c in ('close', 'adj_close', 'closeprice', 'endprice', 'close_price') if c in cols), None)
        if not date_col or not close_col:
            raise KeyError('Missing date/close columns in parquet')
        date_arr = table.column(cols[date_col]).to_pylist()
        close_arr = table.column(cols[close_col]).to_pylist()
        for d, c in zip(date_arr, close_arr):
            if d is None or c is None:
                continue
            # normalize date to YYYY-MM-DD
            ds = str(d)
            try:
                dd = parse_date(ds)
            except Exception:
                continue
            try:
                cv = float(c)
            except Exception:
                continue
            yield dd, float(cv)
        return
    except Exception:
        pass

    # Fallback to pandas
    try:
        import pandas as pd  # type: ignore
        df = pd.read_parquet(parquet_path)
        df_cols = {c.lower(): c for c in df.columns}
        date_col = next((df_cols[c] for c in ('date', 'trade_date', 'datetime') if c in df_cols), None)
        close_col = next((df_cols[c] for c in ('close', 'adj_close', 'closeprice', 'endprice', 'close_price') if c in df_cols), None)
        if date_col is None or close_col is None:
            raise KeyError('Missing date/close columns in parquet (pandas)')
        for _, row in df[[date_col, close_col]].iterrows():
            ds = str(row[date_col])
            try:
                dd = parse_date(ds)
            except Exception:
                continue
            try:
                cv = float(row[close_col])
            except Exception:
                continue
            yield dd, float(cv)
        return
    except Exception as e:
        raise RuntimeError(f'Failed to read parquet {parquet_path}: {e}')


def write_consolidated(out_path: Path, rows: Iterable[tuple[str, date, float]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['date', 'symbol', 'close'])
        for sym, d, c in rows:
            w.writerow([d.isoformat(), sym, f'{c:.6f}'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Input directory with per-ETF parquet files')
    ap.add_argument('--symbols', help='Comma-separated symbols, e.g., 510300.SH,159915.SZ')
    ap.add_argument('--alloc', help='Path to live_alloc CSV to infer symbols')
    ap.add_argument('--start', help='Start date YYYY-MM-DD (optional)')
    ap.add_argument('--end', help='End date YYYY-MM-DD (optional)')
    ap.add_argument('--out', required=True, help='Output consolidated CSV path')
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f'ERROR: root not found: {root}', file=sys.stderr)
        sys.exit(2)

    symbols: list[str] = []
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    elif args.alloc:
        symbols = read_alloc_symbols(Path(args.alloc))
    else:
        print('ERROR: provide --symbols or --alloc', file=sys.stderr)
        sys.exit(2)

    start_d = parse_date(args.start) if args.start else None
    end_d = parse_date(args.end) if args.end else None

    consolidated = []
    for sym in symbols:
        # locate parquet file
        pattern = f'{sym}_daily_*.parquet'
        matches = list(root.glob(pattern))
        if not matches:
            print(f'WARNING: missing parquet for {sym} under {root}', file=sys.stderr)
            continue
        p = sorted(matches)[-1]  # pick the latest range file
        for d, c in iter_parquet_rows(p):
            if start_d and d < start_d:
                continue
            if end_d and d > end_d:
                continue
            consolidated.append((sym, d, c))

    # sort by date then symbol
    consolidated.sort(key=lambda x: (x[1], x[0]))
    if not consolidated:
        print('ERROR: no data consolidated', file=sys.stderr)
        sys.exit(2)

    write_consolidated(Path(args.out), consolidated)
    print(f'Wrote consolidated CSV with {len(consolidated)} rows to {args.out}')


if __name__ == '__main__':
    main()
