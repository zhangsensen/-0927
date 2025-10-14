#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from factor_system.factor_engine.providers.parquet_provider import ParquetDataProvider


def per_day_counts(index: pd.DatetimeIndex) -> Counter:
    return Counter(index.normalize().date)


def main():
    parser = argparse.ArgumentParser(description="Verify A-share session-aware resample")
    parser.add_argument("symbol", help="e.g., 600036.SH")
    parser.add_argument("raw_dir", help="raw data root, contains SH/SZ etc.")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    provider = ParquetDataProvider(raw_data_dir=Path(args.raw_dir))

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

    # Load 1min base (will fallback to default parquet if pattern matches)
    base = provider._load_timeframe_file(args.symbol, provider.market_dirs[provider._detect_market(args.symbol)], "1min", start, end)
    if base.empty:
        print(f"❌ No 1min data for {args.symbol}")
        return
    print(f"✅ 1min loaded: {len(base)} rows, range: {base.index.min()} ~ {base.index.max()}")

    for tf in ["5min", "15min", "30min", "60min"]:
        df = provider._resample_to_timeframe(base.copy(), tf, args.symbol)
        if df.empty:
            print(f"⚠️ {tf}: empty")
            continue
        # Check for invalid timestamps (lunch break 11:30-13:00 exclusive, after-hours >15:00)
        # Valid bar end times can be 11:30 (last morning bar) and 15:00 (last afternoon bar)
        # Invalid: 11:31-12:59 (lunch), 15:01+ (after hours)
        def is_invalid(ts):
            h, m = ts.hour, ts.minute
            # Lunch break (excluding 11:30 which is valid, excluding 13:00 which is valid)
            if (h == 11 and m > 30) or (h == 12):
                return True
            # After hours
            if h == 15 and m > 0:
                return True
            if h > 15:
                return True
            return False
        bad = [ts for ts in df.index if is_invalid(ts)]
        cnts = per_day_counts(df.index)
        example = list(cnts.items())[:5]
        print(f"\n⏱ {tf}: {len(df)} rows, days={len(cnts)}")
        print(f"   per-day counts (first 5): {example}")
        print(f"   invalid stamps: {len(bad)} {'❌' if bad else '✅'}")
        if not df.empty and 'session_aware' in df.columns:
            print(f"   session_aware flag: {df['session_aware'].iloc[0]}")

    print("\n✅ Verification finished")


if __name__ == "__main__":
    main()

