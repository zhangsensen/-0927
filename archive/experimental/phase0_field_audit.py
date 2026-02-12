#!/usr/bin/env python3
"""Phase 0: Tushare Field Audit for Exp6/Exp7 data availability.

Goal: Print column names + 5 sample rows for each API call.
No strategy logic, pure data discovery.
"""

import os
import sys
import time

import tushare as ts

TOKEN = os.getenv(
    "TUSHARE_TOKEN",
    "4a24bcfff16f7593632e6c46976a83e6a26f8f565daa156cb9ea9c1f",
)
pro = ts.pro_api(TOKEN)

QDII_CODE = "513100.SH"  # Nasdaq ETF as representative
START = "20250101"
END = "20260210"


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Exp7: ETF NAV / IOPV ────────────────────────────────────

# Test 1: fund_daily — what we already use
sep("TEST 1: pro.fund_daily() — baseline (OHLCV)")
try:
    df = pro.fund_daily(ts_code=QDII_CODE, start_date=START, end_date=END)
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"Rows: {len(df)}")
    print(df.head(5).to_string())
except Exception as e:
    print(f"FAILED: {e}")

time.sleep(0.3)

# Test 2: fund_nav — mutual fund NAV, does it cover ETFs?
sep("TEST 2: pro.fund_nav() — NAV data for ETF")
try:
    df = pro.fund_nav(ts_code=QDII_CODE, start_date=START, end_date=END)
    if df is not None and len(df) > 0:
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"Rows: {len(df)}")
        print(df.head(5).to_string())
    else:
        print("EMPTY — fund_nav does not cover ETFs with this ts_code")
except Exception as e:
    print(f"FAILED: {e}")

time.sleep(0.3)

# Test 3: Try fund_nav with OF code (open-end fund code format)
# ETF often has an OF code like 513100 -> needs mapping
sep("TEST 3: pro.fund_nav() — with .OF suffix")
try:
    of_code = QDII_CODE.replace(".SH", ".OF")
    df = pro.fund_nav(ts_code=of_code, start_date=START, end_date=END)
    if df is not None and len(df) > 0:
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"Rows: {len(df)}")
        print(df.head(5).to_string())
    else:
        print(f"EMPTY — fund_nav({of_code}) returned nothing")
except Exception as e:
    print(f"FAILED: {e}")

time.sleep(0.3)

# Test 4: fund_nav without specifying code — get schema
sep("TEST 4: pro.fund_nav() — schema discovery (1 day)")
try:
    df = pro.fund_nav(end_date="20260207")
    if df is not None and len(df) > 0:
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"Total rows returned: {len(df)}")
        # Check if any of our QDII codes appear
        qdii_codes_sh = ["513100.SH", "513500.SH", "159920.SZ", "513050.SH", "513130.SH"]
        qdii_codes_of = [c.replace(".SH", ".OF").replace(".SZ", ".OF") for c in qdii_codes_sh]
        all_codes = qdii_codes_sh + qdii_codes_of
        matched = df[df["ts_code"].isin(all_codes)] if "ts_code" in df.columns else None
        if matched is not None and len(matched) > 0:
            print(f"\nQDII matches found ({len(matched)}):")
            print(matched.to_string())
        else:
            print(f"\nNo QDII matches in {len(df)} rows")
            print("Sample ts_codes:", df["ts_code"].head(10).tolist() if "ts_code" in df.columns else "N/A")
            print("\nFirst 3 rows:")
            print(df.head(3).to_string())
    else:
        print("EMPTY")
except Exception as e:
    print(f"FAILED: {e}")

time.sleep(0.3)

# Test 5: Try cb_daily or other ETF-specific endpoints
sep("TEST 5: pro.fund_adj() — ETF adjustment factor (may contain NAV)")
try:
    df = pro.fund_adj(ts_code=QDII_CODE, start_date=START, end_date=END)
    if df is not None and len(df) > 0:
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"Rows: {len(df)}")
        print(df.head(5).to_string())
    else:
        print("EMPTY")
except Exception as e:
    print(f"FAILED: {e}")

time.sleep(0.3)

# ── Exp6: FX Daily ──────────────────────────────────────────

sep("TEST 6: pro.fx_daily() — USD/CNH FX rates")
try:
    df = pro.fx_daily(ts_code="USDCNH.FXCM", start_date=START, end_date=END)
    if df is not None and len(df) > 0:
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"Rows: {len(df)}")
        print(df.head(5).to_string())
    else:
        print("EMPTY — trying alternative code format")
        # Try other FX code formats
        for code in ["USDCNH", "USDCNY", "USD/CNH", "USDCNH.IB"]:
            try:
                df2 = pro.fx_daily(ts_code=code, start_date=START, end_date=END)
                if df2 is not None and len(df2) > 0:
                    print(f"\n  {code} → {len(df2)} rows")
                    print(f"  Columns: {list(df2.columns)}")
                    print(df2.head(3).to_string())
                    break
            except Exception:
                pass
        else:
            print("All FX code formats returned empty")
except Exception as e:
    print(f"FAILED: {e}")

time.sleep(0.3)

# Test 7: FX schema discovery
sep("TEST 7: pro.fx_daily() — schema discovery (no ts_code)")
try:
    df = pro.fx_daily(start_date="20260201", end_date="20260210")
    if df is not None and len(df) > 0:
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"Total rows: {len(df)}")
        if "ts_code" in df.columns:
            print(f"Available FX pairs: {sorted(df['ts_code'].unique().tolist())}")
        print("\nFirst 5 rows:")
        print(df.head(5).to_string())
    else:
        print("EMPTY")
except Exception as e:
    print(f"FAILED: {e}")

time.sleep(0.3)

# Test 8: Try shibor or exchange rate endpoints as alternatives
sep("TEST 8: pro.fx_obasic() — FX pair listing")
try:
    df = pro.fx_obasic(exchange="FXCM")
    if df is not None and len(df) > 0:
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"Total pairs: {len(df)}")
        # Filter for CNH/CNY pairs
        if "ts_code" in df.columns:
            cnh = df[df["ts_code"].str.contains("CN", case=False, na=False)]
            if len(cnh) > 0:
                print(f"\nCNH/CNY pairs ({len(cnh)}):")
                print(cnh.to_string())
    else:
        print("EMPTY")
except Exception as e:
    print(f"FAILED: {e}")

time.sleep(0.3)

# Test 9: PBOC exchange rate (中国人民银行中间价)
sep("TEST 9: pro.eco_cal() or alternative — PBOC mid-rate")
try:
    # Try direct PBOC FX rate
    df = pro.exchange_rate(start_date=START, end_date=END)
    if df is not None and len(df) > 0:
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"Rows: {len(df)}")
        print(df.head(5).to_string())
    else:
        print("EMPTY — exchange_rate endpoint not available or returned nothing")
except Exception as e:
    print(f"FAILED: {e}")

print(f"\n{'='*60}")
print("  AUDIT COMPLETE")
print(f"{'='*60}")
