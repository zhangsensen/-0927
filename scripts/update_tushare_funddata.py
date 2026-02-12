#!/usr/bin/env python3
"""
Tushare ETF 份额(fund_share) + 净值(fund_nav) 增量更新脚本

存储结构:
  raw/ETF/fund_share/  — 每ETF一个parquet, fund_share_{symbol}.parquet
  raw/ETF/fund_nav/    — 每ETF一个parquet, fund_nav_{symbol}.parquet

增量逻辑: 读取已有文件 → 取最新日期 → 只拉增量 → 合并去重 → 保存

用法:
  uv run python scripts/update_tushare_funddata.py           # 增量更新
  uv run python scripts/update_tushare_funddata.py --full     # 全量重拉 (2020~today)
  uv run python scripts/update_tushare_funddata.py --check    # 仅检查数据状态
"""

import argparse
import sys
import time
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, message=".*concatenation with empty.*")

import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────
TOKEN = "4a24bcfff16f7593632e6c46976a83e6a26f8f565daa156cb9ea9c1f"
FULL_START = "20200101"

# 49 ETFs from combo_wfo_config.yaml (41 A-share + 8 QDII)
SYMBOLS = [
    "159801", "159819", "159859", "159883", "159915", "159920", "159928",
    "159949", "159985", "159992", "159995", "159998", "510050", "510300",
    "510500", "511010", "511260", "511380", "512010", "512100", "512200",
    "512400", "512480", "512660", "512690", "512720", "512800", "512880",
    "512980", "513050", "513100", "513130", "513180", "513400", "513500",
    "513520", "515030", "515180", "515210", "515220", "515650", "515790",
    "516090", "516160", "516520", "518850", "518880", "588000", "588200",
]

SHARE_DIR = ROOT / "raw" / "ETF" / "fund_share"
NAV_DIR = ROOT / "raw" / "ETF" / "fund_nav"


def to_ts_code(symbol: str) -> str:
    if symbol.startswith("15"):
        return f"{symbol}.SZ"
    return f"{symbol}.SH"


def get_latest_date(fpath: Path) -> str | None:
    """读取已有parquet，返回最新trade_date (YYYYMMDD str)，无文件返回None"""
    if not fpath.exists():
        return None
    try:
        df = pd.read_parquet(fpath)
        if df.empty:
            return None
        max_dt = pd.to_datetime(df["trade_date"]).max()
        return max_dt.strftime("%Y%m%d")
    except Exception:
        return None


def save_merged(fpath: Path, old_df: pd.DataFrame | None, new_df: pd.DataFrame) -> int:
    """合并新旧数据, 去重, 保存. 返回新增行数."""
    if new_df.empty:
        return 0

    # Normalize trade_date to datetime
    new_df = new_df.copy()
    if new_df["trade_date"].dtype == object:
        new_df["trade_date"] = pd.to_datetime(new_df["trade_date"], format="%Y%m%d")
    elif not pd.api.types.is_datetime64_any_dtype(new_df["trade_date"]):
        new_df["trade_date"] = pd.to_datetime(new_df["trade_date"])

    if old_df is not None and not old_df.empty:
        old_df = old_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(old_df["trade_date"]):
            old_df["trade_date"] = pd.to_datetime(old_df["trade_date"])
        # Align columns to avoid FutureWarning with empty/NA entries
        common_cols = sorted(set(old_df.columns) & set(new_df.columns))
        old_count = len(old_df)
        combined = pd.concat([old_df[common_cols], new_df[common_cols]], ignore_index=True)
        combined = combined.drop_duplicates(subset=["trade_date"], keep="last")
        combined = combined.sort_values("trade_date").reset_index(drop=True)
        new_count = len(combined) - old_count
    else:
        combined = new_df.sort_values("trade_date").reset_index(drop=True)
        new_count = len(combined)

    fpath.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(fpath, index=False)
    return new_count


class TushareFundDataUpdater:
    def __init__(self):
        self.pro = ts.pro_api(TOKEN)

    def update_fund_share(self, symbol: str, full: bool = False) -> tuple[int, int]:
        """更新单只ETF份额数据. 返回 (总行数, 新增行数)"""
        ts_code = to_ts_code(symbol)
        fpath = SHARE_DIR / f"fund_share_{symbol}.parquet"
        end_date = datetime.now().strftime("%Y%m%d")

        if full:
            start_date = FULL_START
            old_df = None
        else:
            latest = get_latest_date(fpath)
            if latest:
                # 从最新日期前1天开始拉 (确保无遗漏)
                dt = datetime.strptime(latest, "%Y%m%d") - timedelta(days=1)
                start_date = dt.strftime("%Y%m%d")
                old_df = pd.read_parquet(fpath)
            else:
                start_date = FULL_START
                old_df = None

        try:
            df = self.pro.fund_share(ts_code=ts_code, start_date=start_date, end_date=end_date)
        except Exception as e:
            logger.error(f"  {symbol} fund_share API error: {e}")
            existing_count = len(pd.read_parquet(fpath)) if fpath.exists() else 0
            return (existing_count, 0)

        if df is None or df.empty:
            existing_count = len(pd.read_parquet(fpath)) if fpath.exists() else 0
            return (existing_count, 0)

        new_count = save_merged(fpath, old_df if not full else None, df)
        total = len(pd.read_parquet(fpath))
        return (total, new_count)

    def update_fund_nav(self, symbol: str, full: bool = False) -> tuple[int, int]:
        """更新单只ETF净值数据. 返回 (总行数, 新增行数)"""
        ts_code = to_ts_code(symbol)
        fpath = NAV_DIR / f"fund_nav_{symbol}.parquet"
        end_date = datetime.now().strftime("%Y%m%d")

        if full:
            start_date = FULL_START
            old_df = None
        else:
            latest = get_latest_date(fpath)
            if latest:
                dt = datetime.strptime(latest, "%Y%m%d") - timedelta(days=1)
                start_date = dt.strftime("%Y%m%d")
                old_df = pd.read_parquet(fpath)
            else:
                start_date = FULL_START
                old_df = None

        try:
            df = self.pro.fund_nav(ts_code=ts_code, start_date=start_date, end_date=end_date)
        except Exception as e:
            logger.error(f"  {symbol} fund_nav API error: {e}")
            existing_count = len(pd.read_parquet(fpath)) if fpath.exists() else 0
            return (existing_count, 0)

        if df is None or df.empty:
            existing_count = len(pd.read_parquet(fpath)) if fpath.exists() else 0
            return (existing_count, 0)

        # fund_nav uses nav_date as the actual date; may have duplicate nav_dates
        # (same NAV date, different announcement dates) — keep latest announcement
        df = df.rename(columns={"nav_date": "trade_date"})
        df = df.sort_values(["trade_date", "ann_date"]).drop_duplicates(
            subset=["trade_date"], keep="last"
        )

        old_count = len(old_df) if (old_df is not None and not old_df.empty) else 0
        save_merged(fpath, old_df if not full else None, df)
        total = len(pd.read_parquet(fpath))
        new_count = max(0, total - old_count)
        return (total, new_count)

    def run_update(self, full: bool = False):
        """批量更新所有ETF"""
        SHARE_DIR.mkdir(parents=True, exist_ok=True)
        NAV_DIR.mkdir(parents=True, exist_ok=True)

        mode = "全量" if full else "增量"
        logger.info(f"{'=' * 60}")
        logger.info(f"Tushare ETF数据更新 ({mode}) — {len(SYMBOLS)} ETFs")
        logger.info(f"{'=' * 60}")

        share_total_new = 0
        nav_total_new = 0

        for i, sym in enumerate(sorted(SYMBOLS), 1):
            # fund_share
            s_total, s_new = self.update_fund_share(sym, full=full)
            # fund_nav
            n_total, n_new = self.update_fund_nav(sym, full=full)

            share_total_new += s_new
            nav_total_new += n_new

            status = ""
            if s_new > 0 or n_new > 0:
                status = f"share +{s_new}, nav +{n_new}"
            else:
                status = "up to date"
            logger.info(f"  [{i:2d}/{len(SYMBOLS)}] {sym}: {status} (share={s_total}, nav={n_total})")

            time.sleep(0.12)  # Tushare rate limit

        logger.info(f"\n{'=' * 60}")
        logger.info(f"完成! 份额新增 {share_total_new} 行, 净值新增 {nav_total_new} 行")
        logger.info(f"存储: {SHARE_DIR}")
        logger.info(f"存储: {NAV_DIR}")

    def check_status(self):
        """检查数据状态"""
        print(f"{'Symbol':<10} {'Share最新':<14} {'Share行数':>10} {'NAV最新':<14} {'NAV行数':>10}")
        print("-" * 65)

        for sym in sorted(SYMBOLS):
            s_path = SHARE_DIR / f"fund_share_{sym}.parquet"
            n_path = NAV_DIR / f"fund_nav_{sym}.parquet"

            s_latest = get_latest_date(s_path) or "N/A"
            n_latest = get_latest_date(n_path) or "N/A"

            s_count = len(pd.read_parquet(s_path)) if s_path.exists() else 0
            n_count = len(pd.read_parquet(n_path)) if n_path.exists() else 0

            if s_latest != "N/A":
                s_latest = f"{s_latest[:4]}-{s_latest[4:6]}-{s_latest[6:]}"
            if n_latest != "N/A":
                n_latest = f"{n_latest[:4]}-{n_latest[4:6]}-{n_latest[6:]}"

            print(f"{sym:<10} {s_latest:<14} {s_count:>10} {n_latest:<14} {n_count:>10}")


def main():
    parser = argparse.ArgumentParser(description="Tushare ETF fund_share + fund_nav updater")
    parser.add_argument("--full", action="store_true", help="全量重拉 (2020~today)")
    parser.add_argument("--check", action="store_true", help="仅检查数据状态")
    args = parser.parse_args()

    updater = TushareFundDataUpdater()

    if args.check:
        updater.check_status()
    else:
        updater.run_update(full=args.full)


if __name__ == "__main__":
    main()
