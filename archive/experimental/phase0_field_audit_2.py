#!/usr/bin/env python3
"""Phase 0 follow-up: verify all 5 QDII NAV coverage + USDHKD + lag pattern."""

import os
import time

import tushare as ts

TOKEN = os.getenv(
    "TUSHARE_TOKEN",
    "4a24bcfff16f7593632e6c46976a83e6a26f8f565daa156cb9ea9c1f",
)
pro = ts.pro_api(TOKEN)

QDII_CODES = ["513100.SH", "513500.SH", "159920.SZ", "513050.SH", "513130.SH"]
QDII_NAMES = ["Nasdaq", "S&P500", "HSI", "ChinaInternet", "HKTech"]

print("=" * 70)
print("  PART 1: fund_nav() coverage for all 5 QDII ETFs")
print("=" * 70)

for code, name in zip(QDII_CODES, QDII_NAMES):
    try:
        df = pro.fund_nav(ts_code=code, start_date="20260101", end_date="20260210")
        if df is not None and len(df) > 0:
            # Compute ann_date - nav_date lag
            df["lag"] = df["ann_date"].astype(int) - df["nav_date"].astype(int)
            print(f"\n{code} ({name}): {len(df)} rows")
            print(f"  Columns: {list(df.columns)}")
            print(f"  ann_date - nav_date lag distribution: {df['lag'].value_counts().sort_index().to_dict()}")
            print(f"  Latest 3 rows:")
            print(df[["ts_code", "ann_date", "nav_date", "unit_nav", "accum_nav"]].head(3).to_string())
        else:
            print(f"\n{code} ({name}): EMPTY — no NAV data")
    except Exception as e:
        print(f"\n{code} ({name}): FAILED — {e}")
    time.sleep(0.3)

print(f"\n{'=' * 70}")
print("  PART 2: USDHKD FX data")
print(f"{'=' * 70}")

try:
    df = pro.fx_daily(ts_code="USDHKD.FXCM", start_date="20250101", end_date="20260210")
    if df is not None and len(df) > 0:
        print(f"Columns: {list(df.columns)}")
        print(f"Rows: {len(df)}")
        print(df.head(5).to_string())
    else:
        print("EMPTY")
except Exception as e:
    print(f"FAILED: {e}")

print(f"\n{'=' * 70}")
print("  PART 3: Compute premium for 513100 (Nasdaq) — recent 10 days")
print(f"{'=' * 70}")

try:
    # Get price
    price_df = pro.fund_daily(ts_code="513100.SH", start_date="20260101", end_date="20260210")
    price_df = price_df.sort_values("trade_date").reset_index(drop=True)

    # Get NAV
    nav_df = pro.fund_nav(ts_code="513100.SH", start_date="20260101", end_date="20260210")
    nav_df = nav_df.sort_values("nav_date").reset_index(drop=True)

    # Merge on trade_date = nav_date
    merged = price_df.merge(
        nav_df[["nav_date", "ann_date", "unit_nav"]],
        left_on="trade_date",
        right_on="nav_date",
        how="inner",
    )
    merged["premium_pct"] = (merged["close"] - merged["unit_nav"]) / merged["unit_nav"] * 100

    print("trade_date  close    unit_nav  ann_date  premium%")
    for _, r in merged.tail(15).iterrows():
        print(
            f"  {r['trade_date']}    {r['close']:.3f}    {r['unit_nav']:.4f}    "
            f"{r['ann_date']}    {r['premium_pct']:+.2f}%"
        )
except Exception as e:
    print(f"FAILED: {e}")

print(f"\n{'=' * 70}")
print("  PART 4: A-share ETF (510300) — does fund_nav also work?")
print(f"{'=' * 70}")

try:
    df = pro.fund_nav(ts_code="510300.SH", start_date="20260101", end_date="20260210")
    if df is not None and len(df) > 0:
        df["lag"] = df["ann_date"].astype(int) - df["nav_date"].astype(int)
        print(f"Rows: {len(df)}")
        print(f"Lag distribution: {df['lag'].value_counts().sort_index().to_dict()}")
        print(df[["ts_code", "ann_date", "nav_date", "unit_nav"]].head(3).to_string())
    else:
        print("EMPTY — no NAV for A-share ETF")
except Exception as e:
    print(f"FAILED: {e}")

print(f"\n{'=' * 70}")
print("  AUDIT COMPLETE")
print(f"{'=' * 70}")
