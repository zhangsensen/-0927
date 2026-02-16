"""
Download supplementary data for ETF strategy pipeline.

Tasks:
1. fund_daily (raw close) from Tushare → raw/ETF/fund_daily/
2. Recompute premium_rate using raw close / NAV → raw/ETF/factors/
3. FX USDCNH from Tushare → raw/ETF/fx/
4. FX HKDCNY from AkShare BOC → raw/ETF/fx/

Usage:
    uv run python scripts/download_supplementary_data.py
    uv run python scripts/download_supplementary_data.py --only fund_daily
    uv run python scripts/download_supplementary_data.py --only fx
    uv run python scripts/download_supplementary_data.py --only premium
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ETF = PROJECT_ROOT / "raw" / "ETF"

# Load Tushare token from config
with open(PROJECT_ROOT / "configs" / "etf_config.yaml") as f:
    cfg = yaml.safe_load(f)
TUSHARE_TOKEN = cfg.get("tushare_token", "")
pro = ts.pro_api(TUSHARE_TOKEN)

# Strategy pool: all ETFs in daily/
DAILY_DIR = RAW_ETF / "daily"
POOL_CODES = sorted(
    f.split("_daily_")[0]
    for f in os.listdir(DAILY_DIR)
    if f.endswith(".parquet")
)


def download_fund_daily():
    """Download raw close prices from Tushare fund_daily for all pool ETFs."""
    out_dir = RAW_ETF / "fund_daily"
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading fund_daily (raw close) for {len(POOL_CODES)} ETFs")
    print(f"{'='*60}")

    for i, code in enumerate(POOL_CODES):
        out_file = out_dir / f"fund_daily_{code.replace('.', '_')}.parquet"

        # Tushare fund_daily: max 2000 rows per request, need pagination
        all_dfs = []
        end_date = "20261231"
        start_date = "20190101"  # slightly before 2020 for safety

        try:
            df = pro.fund_daily(ts_code=code, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                all_dfs.append(df)

                # If we hit 2000 rows, need to paginate
                while len(df) == 2000:
                    next_end = str(int(df["trade_date"].min()) - 1)
                    df = pro.fund_daily(
                        ts_code=code, start_date=start_date, end_date=next_end
                    )
                    if df is not None and len(df) > 0:
                        all_dfs.append(df)
                    else:
                        break
                    time.sleep(0.15)

            if all_dfs:
                result = pd.concat(all_dfs, ignore_index=True)
                result = result.drop_duplicates(subset=["trade_date"]).sort_values(
                    "trade_date"
                )
                result.to_parquet(out_file, index=False)
                print(
                    f"  [{i+1:2d}/{len(POOL_CODES)}] {code}: {len(result)} rows "
                    f"({result['trade_date'].min()} ~ {result['trade_date'].max()})"
                )
            else:
                print(f"  [{i+1:2d}/{len(POOL_CODES)}] {code}: NO DATA")

        except Exception as e:
            print(f"  [{i+1:2d}/{len(POOL_CODES)}] {code}: ERROR - {e}")

        time.sleep(0.15)  # rate limit

    print(f"\nDone. Files saved to {out_dir}")


def recompute_premium_rate():
    """Recompute premium_rate = (raw_close / unit_nav - 1) * 100."""
    fund_daily_dir = RAW_ETF / "fund_daily"
    fund_nav_dir = RAW_ETF / "fund_nav"
    out_dir = RAW_ETF / "factors"
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print("Recomputing premium_rate using raw close / NAV")
    print(f"{'='*60}")

    for i, code in enumerate(POOL_CODES):
        code_short = code.replace(".", "_")
        fd_file = fund_daily_dir / f"fund_daily_{code_short}.parquet"
        nav_file = fund_nav_dir / f"fund_nav_{code.split('.')[0]}.parquet"
        out_file = out_dir / f"premium_rate_{code.split('.')[0]}.parquet"

        if not fd_file.exists():
            print(f"  [{i+1:2d}] {code}: fund_daily not found, skipping")
            continue
        if not nav_file.exists():
            print(f"  [{i+1:2d}] {code}: fund_nav not found, skipping")
            continue

        fd = pd.read_parquet(fd_file)
        nav = pd.read_parquet(nav_file)

        # Normalize trade_date to string YYYYMMDD
        fd["dt"] = fd["trade_date"].astype(str).str.replace("-", "")
        nav["dt"] = pd.to_datetime(nav["trade_date"]).dt.strftime("%Y%m%d")

        merged = fd[["dt", "close"]].merge(nav[["dt", "unit_nav"]], on="dt", how="inner")
        merged["premium_rate"] = (merged["close"] / merged["unit_nav"] - 1) * 100
        merged = merged.rename(columns={"dt": "trade_date"})
        merged = merged[["trade_date", "premium_rate"]].sort_values("trade_date")

        # Backup old file
        if out_file.exists():
            backup = out_dir / f"premium_rate_{code.split('.')[0]}.bak.parquet"
            if not backup.exists():
                os.rename(out_file, backup)

        merged.to_parquet(out_file, index=False)
        mean_prem = merged["premium_rate"].mean()
        print(
            f"  [{i+1:2d}/{len(POOL_CODES)}] {code}: {len(merged)} rows, "
            f"mean={mean_prem:+.3f}%, "
            f"range: {merged['trade_date'].min()} ~ {merged['trade_date'].max()}"
        )

    print(f"\nDone. Old files backed up as *.bak.parquet")


def download_fx():
    """Download FX data: USDCNH from Tushare, HKDCNY from AkShare."""
    out_dir = RAW_ETF / "fx"
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print("Downloading FX data")
    print(f"{'='*60}")

    # 1. USDCNH from Tushare
    print("\n--- USDCNH (Tushare fx_daily) ---")
    all_dfs = []
    end_date = "20261231"
    start_date = "20190101"

    try:
        df = pro.fx_daily(ts_code="USDCNH.FXCM", start_date=start_date, end_date=end_date)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            while len(df) == 1000:  # fx_daily max 1000 rows
                next_end = str(int(df["trade_date"].min()) - 1)
                df = pro.fx_daily(
                    ts_code="USDCNH.FXCM", start_date=start_date, end_date=next_end
                )
                if df is not None and len(df) > 0:
                    all_dfs.append(df)
                else:
                    break
                time.sleep(0.15)

        if all_dfs:
            result = pd.concat(all_dfs, ignore_index=True)
            result = result.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")
            out_file = out_dir / "usdcnh_daily.parquet"
            result.to_parquet(out_file, index=False)
            print(
                f"  USDCNH: {len(result)} rows "
                f"({result['trade_date'].min()} ~ {result['trade_date'].max()})"
            )
        else:
            print("  USDCNH: NO DATA")
    except Exception as e:
        print(f"  USDCNH ERROR: {e}")

    # 2. HKDCNY from AkShare (BOC midpoint rate)
    print("\n--- HKDCNY (AkShare BOC中间价) ---")
    try:
        import akshare as ak

        boc = ak.currency_boc_safe()
        if boc is not None and len(boc) > 0:
            # Filter to 2019+ and extract HKD column
            boc.columns = boc.columns.str.strip()
            boc["日期"] = pd.to_datetime(boc["日期"])
            boc = boc[boc["日期"] >= "2019-01-01"].copy()
            boc = boc.sort_values("日期")

            # Keep relevant columns
            hkd_col = [c for c in boc.columns if "港" in c]
            usd_col = [c for c in boc.columns if "美元" in c]

            cols_to_keep = ["日期"] + usd_col + hkd_col
            fx_boc = boc[cols_to_keep].copy()
            fx_boc = fx_boc.rename(columns={"日期": "trade_date"})
            fx_boc["trade_date"] = fx_boc["trade_date"].dt.strftime("%Y%m%d")

            # Convert to numeric
            for col in fx_boc.columns:
                if col != "trade_date":
                    fx_boc[col] = pd.to_numeric(fx_boc[col], errors="coerce")

            fx_boc = fx_boc.dropna(subset=fx_boc.columns[1:], how="all")
            out_file = out_dir / "boc_fx_daily.parquet"
            fx_boc.to_parquet(out_file, index=False)
            print(
                f"  BOC FX: {len(fx_boc)} rows, cols: {list(fx_boc.columns)}"
            )
            print(
                f"  Range: {fx_boc['trade_date'].min()} ~ {fx_boc['trade_date'].max()}"
            )
        else:
            print("  BOC FX: NO DATA")
    except Exception as e:
        print(f"  BOC FX ERROR: {e}")

    print(f"\nDone. Files saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download supplementary ETF data")
    parser.add_argument(
        "--only",
        choices=["fund_daily", "premium", "fx", "all"],
        default="all",
        help="Which data to download",
    )
    args = parser.parse_args()

    if args.only in ("fund_daily", "all"):
        download_fund_daily()

    if args.only in ("premium", "all"):
        recompute_premium_rate()

    if args.only in ("fx", "all"):
        download_fx()

    if args.only == "all":
        print(f"\n{'='*60}")
        print("ALL DONE - Summary")
        print(f"{'='*60}")
        for subdir in ["fund_daily", "factors", "fx"]:
            p = RAW_ETF / subdir
            if p.exists():
                files = [f for f in os.listdir(p) if f.endswith(".parquet")]
                print(f"  {subdir}/: {len(files)} parquet files")


if __name__ == "__main__":
    main()
