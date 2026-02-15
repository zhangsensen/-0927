#!/usr/bin/env python3
"""
ETF数据统一每日更新脚本

覆盖全部数据源，增量更新，幂等安全。
建议每个交易日收盘后运行 (15:30+)。

用法:
    uv run python scripts/update_all_data.py              # 全量更新
    uv run python scripts/update_all_data.py --only tushare,fx,premium  # 选择性更新
    uv run python scripts/update_all_data.py --dry-run     # 只检查不更新

数据更新顺序 (有依赖):
    1. daily/       OHLCV前复权 (自动检测: QMT可用时用QMT, 否则Tushare fund_daily+fund_adj)
    2. fund_daily/  Tushare (原始OHLCV)
    3. fund_nav/    Tushare (基金净值, T+1到)
    4. fund_share/  Tushare (基金份额)
    5. margin/      Tushare (融资融券)
    6. fx/          Tushare + AkShare (外汇)
    7. factors/     本地计算 (折溢价, 依赖 fund_daily + fund_nav)
    8. snapshots/   AkShare (全量快照, 盘中IOPV)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ETF = PROJECT_ROOT / "raw" / "ETF"
CONFIG_FILE = PROJECT_ROOT / "configs" / "etf_config.yaml"

# ETF codes: from daily/ directory (strategy pool)
DAILY_DIR = RAW_ETF / "daily"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("update_all")


def get_pool_codes() -> list[str]:
    """Get strategy pool ETF codes from daily/ directory."""
    return sorted(
        f.split("_daily_")[0]
        for f in os.listdir(DAILY_DIR)
        if f.endswith(".parquet")
    )


def get_tushare_pro():
    """Initialize Tushare Pro API."""
    import tushare as ts

    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)
    token = cfg.get("tushare_token", "")
    if not token:
        log.error("tushare_token not found in %s", CONFIG_FILE)
        sys.exit(1)
    return ts.pro_api(token)


def latest_date_in_file(filepath: Path, date_col: str = "trade_date") -> str | None:
    """Read the latest trade_date from a parquet file. Returns YYYYMMDD str."""
    if not filepath.exists():
        return None
    try:
        df = pd.read_parquet(filepath, columns=[date_col])
        if df.empty:
            return None
        val = df[date_col].max()
        # Normalize to YYYYMMDD string
        if isinstance(val, pd.Timestamp):
            return val.strftime("%Y%m%d")
        return str(val).replace("-", "")[:8]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 1: daily/ (QMT or Tushare fallback)
# ---------------------------------------------------------------------------

def _qmt_available() -> bool:
    """Check if QMT bridge is available (Windows only)."""
    try:
        import qmt_bridge  # noqa: F401
        return True
    except ImportError:
        return False


def update_daily(pro, codes: list[str], dry_run: bool = False):
    """Update daily/ OHLCV (前复权). Uses QMT if available, else Tushare."""
    log.info("━━━ [1/8] daily/ (OHLCV 前复权) ━━━")

    if _qmt_available():
        log.info("  QMT bridge detected, using QMT")
        _update_daily_qmt(dry_run)
    else:
        log.info("  QMT not available (Linux), using Tushare fund_daily + fund_adj")
        _update_daily_tushare(pro, codes, dry_run)

    # Invalidate DataLoader cache after daily/ updates
    cache_dir = PROJECT_ROOT / "src" / "etf_strategy" / ".cache"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
        log.info("  DataLoader cache cleared")


def _update_daily_qmt(dry_run: bool = False):
    """Update daily/ from QMT bridge (Windows)."""
    script = PROJECT_ROOT / "scripts" / "update_daily_from_qmt_bridge.py"
    if not script.exists():
        log.warning("QMT update script not found, skipping")
        return

    if dry_run:
        log.info("  DRY RUN: would run update_daily_from_qmt_bridge.py --all")
        return

    import subprocess

    result = subprocess.run(
        ["uv", "run", "python", str(script), "--all"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 0:
        log.info("  QMT update complete")
    else:
        log.warning("  QMT update failed: %s", result.stderr[:200])


def _update_daily_tushare(pro, codes: list[str], dry_run: bool = False):
    """Update daily/ from Tushare fund_daily + fund_adj (Linux fallback).

    Strategy: incremental append for most days. When adj_factor changes
    (distribution event, ~1-2x/year per ETF), recompute the entire file.
    All OHLCV factors are scale-invariant so Tushare adj differs from QMT
    by a constant per segment — no impact on strategy behavior.
    """
    import glob as glob_mod

    today = datetime.now().strftime("%Y%m%d")
    updated, skipped, errors = 0, 0, 0

    for code in codes:
        # Find existing file (naming: CODE_daily_STARTDATE_ENDDATE.parquet)
        pattern = str(DAILY_DIR / f"{code}_daily_*.parquet")
        matches = sorted(glob_mod.glob(pattern))
        existing_file = Path(matches[-1]) if matches else None

        # Read existing data, normalize schema to [trade_date(int), adj_*OHLC, vol]
        df_old = None
        last_date = None
        if existing_file and existing_file.exists():
            df_old = pd.read_parquet(existing_file)
            if not df_old.empty:
                # Normalize trade_date to int (some old files have string dates)
                df_old["trade_date"] = (
                    df_old["trade_date"].astype(str).str.replace("-", "").astype(int)
                )
                # Keep only pipeline-required columns
                expected_cols = ["trade_date", "adj_open", "adj_high", "adj_low", "adj_close", "vol"]
                missing = [c for c in expected_cols if c not in df_old.columns]
                if missing:
                    log.warning("  %s: old file missing columns %s, will full recompute", code, missing)
                    df_old = None
                else:
                    df_old = df_old[expected_cols].copy()
                    last_date = str(int(df_old["trade_date"].max()))
                    if last_date >= today:
                        skipped += 1
                        continue

        if dry_run:
            log.info("  %s: would update %s ~ %s", code, last_date or "full", today)
            updated += 1
            continue

        try:
            # Get adj_factor to detect distribution events
            adj_start = last_date or "20190101"
            adj_df = pro.fund_adj(ts_code=code, start_date=adj_start, end_date=today)
            time.sleep(0.08)

            if adj_df is None or adj_df.empty:
                log.warning("  %s: no adj_factor data", code)
                errors += 1
                continue

            adj_df = adj_df.sort_values("trade_date").reset_index(drop=True)
            current_adj_factor = adj_df["adj_factor"].iloc[-1]

            # Check if adj_factor changed since last update → need full recompute
            need_full_recompute = False
            if df_old is not None and last_date:
                old_factors = adj_df[adj_df["trade_date"] <= last_date]["adj_factor"]
                if not old_factors.empty and abs(old_factors.iloc[-1] - current_adj_factor) > 1e-6:
                    need_full_recompute = True
                    log.info("  %s: adj_factor changed (%.3f → %.3f), full recompute",
                             code, old_factors.iloc[-1], current_adj_factor)

            if need_full_recompute or df_old is None:
                # Full recompute: fetch all history
                fd = pro.fund_daily(ts_code=code, start_date="20190101", end_date=today)
                time.sleep(0.12)
                adj_full = pro.fund_adj(ts_code=code, start_date="20190101", end_date=today)
                time.sleep(0.08)

                if fd is None or fd.empty or adj_full is None or adj_full.empty:
                    log.warning("  %s: missing data for full recompute", code)
                    errors += 1
                    continue

                merged = fd.merge(adj_full[["trade_date", "adj_factor"]], on="trade_date")
                latest_factor = merged["adj_factor"].max()  # Most recent = largest

                # Forward-adjusted prices
                ratio = merged["adj_factor"] / latest_factor
                df_new = pd.DataFrame({
                    "trade_date": merged["trade_date"].astype(int),
                    "adj_open": (merged["open"] * ratio).round(4),
                    "adj_high": (merged["high"] * ratio).round(4),
                    "adj_low": (merged["low"] * ratio).round(4),
                    "adj_close": (merged["close"] * ratio).round(4),
                    "vol": merged["vol"],
                })
                df_new = df_new.sort_values("trade_date").reset_index(drop=True)
            else:
                # Incremental: fetch only new dates, adj_factor unchanged → raw = adj
                start = str(int(last_date) + 1)
                fd = pro.fund_daily(ts_code=code, start_date=start, end_date=today)
                time.sleep(0.12)

                if fd is None or fd.empty:
                    skipped += 1
                    continue

                df_append = pd.DataFrame({
                    "trade_date": fd["trade_date"].astype(int),
                    "adj_open": fd["open"],
                    "adj_high": fd["high"],
                    "adj_low": fd["low"],
                    "adj_close": fd["close"],
                    "vol": fd["vol"],
                })
                df_new = pd.concat([df_old, df_append], ignore_index=True)
                df_new = df_new.drop_duplicates(subset=["trade_date"]).sort_values(
                    "trade_date"
                ).reset_index(drop=True)

            # Write with updated filename
            min_d = int(df_new["trade_date"].min())
            max_d = int(df_new["trade_date"].max())
            new_fname = f"{code}_daily_{min_d}_{max_d}.parquet"
            new_path = DAILY_DIR / new_fname

            df_new.to_parquet(new_path, index=False)
            updated += 1

            # Remove old file if name changed
            if existing_file and existing_file != new_path and existing_file.exists():
                existing_file.unlink()

        except Exception as e:
            log.warning("  %s: %s", code, e)
            errors += 1
            time.sleep(0.5)

    log.info("  daily: %d updated, %d current, %d errors", updated, skipped, errors)


# ---------------------------------------------------------------------------
# Step 2: fund_daily/ (Tushare)
# ---------------------------------------------------------------------------

def update_fund_daily(pro, codes: list[str], dry_run: bool = False):
    """Incremental update of raw close prices from Tushare fund_daily."""
    log.info("━━━ [2/8] fund_daily/ (Tushare) ━━━")
    out_dir = RAW_ETF / "fund_daily"
    out_dir.mkdir(exist_ok=True)

    updated, skipped = 0, 0
    today = datetime.now().strftime("%Y%m%d")

    for code in codes:
        fname = f"fund_daily_{code.replace('.', '_')}.parquet"
        fpath = out_dir / fname

        last = latest_date_in_file(fpath)
        if last and last >= today:
            skipped += 1
            continue

        start = str(int(last) + 1) if last else "20190101"

        if dry_run:
            log.info("  %s: would fetch %s ~ %s", code, start, today)
            updated += 1
            continue

        try:
            df = pro.fund_daily(ts_code=code, start_date=start, end_date=today)
            if df is not None and len(df) > 0:
                if fpath.exists():
                    old = pd.read_parquet(fpath)
                    df = pd.concat([old, df], ignore_index=True)
                    df = df.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")
                df.to_parquet(fpath, index=False)
                updated += 1
            time.sleep(0.12)
        except Exception as e:
            log.warning("  %s: %s", code, e)
            time.sleep(0.5)

    log.info("  fund_daily: %d updated, %d already current", updated, skipped)


# ---------------------------------------------------------------------------
# Step 3: fund_nav/ (Tushare)
# ---------------------------------------------------------------------------

def update_fund_nav(pro, codes: list[str], dry_run: bool = False):
    """Incremental update of NAV from Tushare fund_nav."""
    log.info("━━━ [3/8] fund_nav/ (Tushare) ━━━")
    out_dir = RAW_ETF / "fund_nav"
    out_dir.mkdir(exist_ok=True)

    updated, skipped = 0, 0
    today = datetime.now().strftime("%Y%m%d")

    for code in codes:
        code6 = code.split(".")[0]
        fname = f"fund_nav_{code6}.parquet"
        fpath = out_dir / fname

        # fund_nav trade_date is datetime64, normalize
        last = None
        if fpath.exists():
            try:
                df_old = pd.read_parquet(fpath)
                if not df_old.empty:
                    last = pd.to_datetime(df_old["trade_date"]).max().strftime("%Y%m%d")
            except Exception:
                pass

        if last and last >= today:
            skipped += 1
            continue

        start = str(int(last) + 1) if last else "20190101"

        if dry_run:
            log.info("  %s: would fetch NAV %s ~ %s", code, start, today)
            updated += 1
            continue

        try:
            df = pro.fund_nav(ts_code=code, start_date=start, end_date=today)
            if df is not None and len(df) > 0:
                # Tushare fund_nav returns nav_date, rename to trade_date for consistency
                if "nav_date" in df.columns and "trade_date" not in df.columns:
                    df = df.rename(columns={"nav_date": "trade_date"})

                if fpath.exists():
                    old = pd.read_parquet(fpath)
                    # Normalize both to string for dedup
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
                    old["trade_date"] = pd.to_datetime(old["trade_date"])
                    df = pd.concat([old, df], ignore_index=True)
                    df = df.drop_duplicates(
                        subset=["trade_date"], keep="last"
                    ).sort_values("trade_date")
                else:
                    df["trade_date"] = pd.to_datetime(df["trade_date"])

                df.to_parquet(fpath, index=False)
                updated += 1
            time.sleep(0.12)
        except Exception as e:
            log.warning("  %s: %s", code, e)
            time.sleep(0.5)

    log.info("  fund_nav: %d updated, %d already current", updated, skipped)


# ---------------------------------------------------------------------------
# Step 4: fund_share/ (Tushare)
# ---------------------------------------------------------------------------

def update_fund_share(pro, codes: list[str], dry_run: bool = False):
    """Incremental update of fund shares from Tushare fund_share."""
    log.info("━━━ [4/8] fund_share/ (Tushare) ━━━")
    out_dir = RAW_ETF / "fund_share"
    out_dir.mkdir(exist_ok=True)

    updated, skipped = 0, 0
    today = datetime.now().strftime("%Y%m%d")

    for code in codes:
        code6 = code.split(".")[0]
        fname = f"fund_share_{code6}.parquet"
        fpath = out_dir / fname

        last = None
        if fpath.exists():
            try:
                df_old = pd.read_parquet(fpath)
                if not df_old.empty:
                    last = pd.to_datetime(df_old["trade_date"]).max().strftime("%Y%m%d")
            except Exception:
                pass

        if last and last >= today:
            skipped += 1
            continue

        start = str(int(last) + 1) if last else "20190101"

        if dry_run:
            log.info("  %s: would fetch share %s ~ %s", code, start, today)
            updated += 1
            continue

        try:
            df = pro.fund_share(ts_code=code, start_date=start, end_date=today)
            if df is not None and len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                if fpath.exists():
                    old = pd.read_parquet(fpath)
                    old["trade_date"] = pd.to_datetime(old["trade_date"])
                    df = pd.concat([old, df], ignore_index=True)
                    df = df.drop_duplicates(
                        subset=["trade_date"], keep="last"
                    ).sort_values("trade_date")
                df.to_parquet(fpath, index=False)
                updated += 1
            time.sleep(0.12)
        except Exception as e:
            log.warning("  %s: %s", code, e)
            time.sleep(0.5)

    log.info("  fund_share: %d updated, %d already current", updated, skipped)


# ---------------------------------------------------------------------------
# Step 5: margin/ (Tushare)
# ---------------------------------------------------------------------------

def update_margin(pro, codes: list[str], dry_run: bool = False):
    """Incremental update of margin data (single consolidated file)."""
    log.info("━━━ [5/8] margin/ (Tushare) ━━━")
    fpath = RAW_ETF / "margin" / "margin_pool43_2020_now.parquet"

    if not fpath.exists():
        log.warning("  margin file not found, skipping incremental update")
        return

    df_old = pd.read_parquet(fpath)
    last = df_old["trade_date"].max()
    today = datetime.now().strftime("%Y%m%d")

    if last >= today:
        log.info("  margin: already current (%s)", last)
        return

    start = str(int(last) + 1)

    if dry_run:
        log.info("  would fetch margin %s ~ %s for %d codes", start, today, len(codes))
        return

    new_dfs = []
    for code in codes:
        try:
            df = pro.margin_detail(ts_code=code, start_date=start, end_date=today)
            if df is not None and len(df) > 0:
                new_dfs.append(df)
            time.sleep(0.12)
        except Exception as e:
            log.warning("  margin %s: %s", code, e)
            time.sleep(0.5)

    if new_dfs:
        new_df = pd.concat(new_dfs, ignore_index=True)
        combined = pd.concat([df_old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["trade_date", "ts_code"], keep="last"
        ).sort_values(["trade_date", "ts_code"])
        combined.to_parquet(fpath, index=False)
        log.info("  margin: +%d rows (%s ~ %s)", len(new_df), start, today)
    else:
        log.info("  margin: no new data (holiday?)")


# ---------------------------------------------------------------------------
# Step 6: fx/ (Tushare + AkShare)
# ---------------------------------------------------------------------------

def update_fx(pro, dry_run: bool = False):
    """Update FX data: USDCNH from Tushare, BOC midpoint from AkShare."""
    log.info("━━━ [6/8] fx/ (Tushare + AkShare) ━━━")
    out_dir = RAW_ETF / "fx"
    out_dir.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")

    # --- USDCNH (incremental) ---
    usdcnh_path = out_dir / "usdcnh_daily.parquet"
    last = latest_date_in_file(usdcnh_path)

    if last and last >= today:
        log.info("  USDCNH: already current (%s)", last)
    elif dry_run:
        log.info("  USDCNH: would fetch %s ~ %s", last or "20190101", today)
    else:
        start = str(int(last) + 1) if last else "20190101"
        try:
            df = pro.fx_daily(ts_code="USDCNH.FXCM", start_date=start, end_date=today)
            if df is not None and len(df) > 0:
                if usdcnh_path.exists():
                    old = pd.read_parquet(usdcnh_path)
                    df = pd.concat([old, df], ignore_index=True)
                    df = df.drop_duplicates(subset=["trade_date"]).sort_values(
                        "trade_date"
                    )
                df.to_parquet(usdcnh_path, index=False)
                log.info("  USDCNH: updated to %s (%d rows)", df["trade_date"].max(), len(df))
        except Exception as e:
            log.warning("  USDCNH: %s", e)

    # --- BOC midpoint (full replace — API returns all history) ---
    boc_path = out_dir / "boc_fx_daily.parquet"
    if dry_run:
        log.info("  BOC FX: would fetch full history")
    else:
        try:
            import akshare as ak

            boc = ak.currency_boc_safe()
            if boc is not None and len(boc) > 0:
                boc.columns = boc.columns.str.strip()
                boc["日期"] = pd.to_datetime(boc["日期"])
                boc = boc[boc["日期"] >= "2019-01-01"].sort_values("日期")

                hkd_col = [c for c in boc.columns if "港" in c]
                usd_col = [c for c in boc.columns if "美元" in c]
                cols = ["日期"] + usd_col + hkd_col
                fx_boc = boc[cols].copy()
                fx_boc = fx_boc.rename(columns={"日期": "trade_date"})
                fx_boc["trade_date"] = fx_boc["trade_date"].dt.strftime("%Y%m%d")

                for col in fx_boc.columns:
                    if col != "trade_date":
                        fx_boc[col] = pd.to_numeric(fx_boc[col], errors="coerce")
                fx_boc = fx_boc.dropna(subset=fx_boc.columns[1:], how="all")
                fx_boc.to_parquet(boc_path, index=False)
                log.info(
                    "  BOC FX: %d rows (latest %s)", len(fx_boc), fx_boc["trade_date"].max()
                )
        except Exception as e:
            log.warning("  BOC FX: %s", e)


# ---------------------------------------------------------------------------
# Step 7: factors/premium_rate (local computation)
# ---------------------------------------------------------------------------

def update_premium_rate(codes: list[str], dry_run: bool = False):
    """Recompute premium_rate = (raw_close / unit_nav - 1) * 100."""
    log.info("━━━ [7/8] factors/premium_rate (local) ━━━")
    fund_daily_dir = RAW_ETF / "fund_daily"
    fund_nav_dir = RAW_ETF / "fund_nav"
    out_dir = RAW_ETF / "factors"
    out_dir.mkdir(exist_ok=True)

    updated, skipped, missing = 0, 0, 0

    for code in codes:
        code6 = code.split(".")[0]
        code_under = code.replace(".", "_")

        fd_file = fund_daily_dir / f"fund_daily_{code_under}.parquet"
        nav_file = fund_nav_dir / f"fund_nav_{code6}.parquet"
        out_file = out_dir / f"premium_rate_{code6}.parquet"

        if not fd_file.exists() or not nav_file.exists():
            missing += 1
            continue

        # Check if premium is already up to date
        fd_last = latest_date_in_file(fd_file)
        nav_last = None
        if nav_file.exists():
            try:
                nav_df = pd.read_parquet(nav_file)
                nav_last = pd.to_datetime(nav_df["trade_date"]).max().strftime("%Y%m%d")
            except Exception:
                pass

        prem_last = latest_date_in_file(out_file)
        data_last = min(fd_last or "0", nav_last or "0")
        if prem_last and prem_last >= data_last:
            skipped += 1
            continue

        if dry_run:
            log.info("  %s: would recompute premium", code6)
            updated += 1
            continue

        fd = pd.read_parquet(fd_file)
        nav = pd.read_parquet(nav_file)

        fd["dt"] = fd["trade_date"].astype(str).str.replace("-", "")
        nav["dt"] = pd.to_datetime(nav["trade_date"]).dt.strftime("%Y%m%d")

        merged = fd[["dt", "close"]].merge(nav[["dt", "unit_nav"]], on="dt", how="inner")
        merged["premium_rate"] = (merged["close"] / merged["unit_nav"] - 1) * 100
        merged = merged.rename(columns={"dt": "trade_date"})
        merged = merged[["trade_date", "premium_rate"]].sort_values("trade_date")
        merged.to_parquet(out_file, index=False)
        updated += 1

    log.info(
        "  premium_rate: %d updated, %d current, %d missing source data",
        updated, skipped, missing,
    )


# ---------------------------------------------------------------------------
# Step 8: snapshots/ (AkShare)
# ---------------------------------------------------------------------------

def update_snapshot(dry_run: bool = False):
    """Fetch full ETF snapshot from AkShare (IOPV, fund flow, shares)."""
    log.info("━━━ [8/8] snapshots/ (AkShare) ━━━")
    out_dir = RAW_ETF / "snapshots"
    out_dir.mkdir(exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    out_file = out_dir / f"snapshot_{today}.parquet"

    if out_file.exists():
        log.info("  snapshot: already exists for %s", today)
        return

    if dry_run:
        log.info("  snapshot: would fetch fund_etf_spot_em()")
        return

    try:
        import akshare as ak

        df = ak.fund_etf_spot_em()
        if df is not None and len(df) > 0:
            df["数据日期"] = pd.Timestamp(today)
            df["更新时间"] = pd.Timestamp.now(tz="Asia/Shanghai")
            df.to_parquet(out_file, index=False)
            log.info("  snapshot: %d ETFs saved", len(df))
        else:
            log.warning("  snapshot: empty response")
    except Exception as e:
        log.warning("  snapshot: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_STEPS = ["daily", "tushare", "fx", "premium", "snapshot"]


def main():
    parser = argparse.ArgumentParser(
        description="ETF数据统一每日更新",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        help="只运行指定步骤 (逗号分隔): daily,tushare,fx,premium,snapshot",
    )
    parser.add_argument(
        "--skip",
        help="跳过指定步骤 (逗号分隔)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查不更新",
    )
    args = parser.parse_args()

    # Normalize step names: "qmt" is legacy alias for "daily"
    def _normalize_steps(raw: str) -> set[str]:
        names = set(raw.split(","))
        if "qmt" in names:
            names.discard("qmt")
            names.add("daily")
        return names

    steps = set(ALL_STEPS)
    if args.only:
        steps = _normalize_steps(args.only)
    if args.skip:
        steps -= _normalize_steps(args.skip)

    log.info("=" * 60)
    log.info("ETF 数据统一更新 — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("Steps: %s", ", ".join(sorted(steps)))
    log.info("=" * 60)

    codes = get_pool_codes()
    log.info("Strategy pool: %d ETFs", len(codes))

    # Initialize Tushare if needed (daily/ fallback also needs it on Linux)
    pro = None
    if steps & {"daily", "tushare", "fx"}:
        pro = get_tushare_pro()

    t0 = time.time()

    # 1. daily/ OHLCV (QMT or Tushare fallback)
    if "daily" in steps:
        try:
            update_daily(pro, codes, args.dry_run)
        except Exception as e:
            log.warning("daily update failed: %s", e)

    # 2-5. Tushare batch (fund_daily, fund_nav, fund_share, margin)
    if "tushare" in steps:
        update_fund_daily(pro, codes, args.dry_run)
        update_fund_nav(pro, codes, args.dry_run)
        update_fund_share(pro, codes, args.dry_run)
        update_margin(pro, codes, args.dry_run)

    # 6. FX
    if "fx" in steps:
        update_fx(pro, args.dry_run)

    # 7. Premium rate (depends on fund_daily + fund_nav)
    if "premium" in steps:
        update_premium_rate(codes, args.dry_run)

    # 8. Snapshot
    if "snapshot" in steps:
        update_snapshot(args.dry_run)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("全部完成，耗时 %.1fs", elapsed)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
