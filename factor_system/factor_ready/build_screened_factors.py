#!/usr/bin/env python3
"""Convert CSV factors to parquet format (simple transformer)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
SCREENING_ROOT = PROJECT_ROOT / "factor_screening" / "因子筛选"
OUTPUT_DIR = BASE_DIR


def list_sessions(symbol: str) -> list[Path]:
    prefix = symbol.replace(".", "_")
    pattern = f"{prefix}_*multi_timeframe_*"
    return sorted(SCREENING_ROOT.glob(pattern))


def resolve_session(symbol: str, session_id: str | None) -> Path:
    if session_id:
        matches = [path for path in list_sessions(symbol) if session_id in path.name]
        if not matches:
            raise FileNotFoundError(f"Session {session_id} not found for {symbol}")
        return matches[-1]
    sessions = list_sessions(symbol)
    if not sessions:
        raise FileNotFoundError(f"No screening session found for {symbol}")
    return sessions[-1]


def get_parquet_path(symbol: str) -> Path:
    clean_symbol = symbol.replace(".", "_")
    return OUTPUT_DIR / f"{clean_symbol}_best_factors.parquet"


def process_symbol(symbol: str, session_dir: Path, session_suffix: str) -> None:
    """Convert CSV to parquet - keep all fields and all factors."""
    try:
        # Find best_factors_overall CSV file
        csv_files = list(session_dir.glob("*best_factors_overall_*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No best_factors_overall CSV found in {session_dir}"
            )

        csv_file = csv_files[0]
        parquet_path = get_parquet_path(symbol)

        # Read CSV and convert to parquet directly
        import pandas as pd

        df = pd.read_csv(csv_file)

        # Add session column
        df = df.copy()
        df["session"] = session_suffix

        # Save to parquet (all fields, all factors)
        df.to_parquet(parquet_path, index=False)
        print(f"✅ {symbol}: {len(df)} factors → {parquet_path.name}")

    except Exception as exc:  # noqa: BLE001
        print(f"❌ {symbol}: {exc}")


def run(symbols: Iterable[str], session_id: str | None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        clean = symbol.strip()
        if not clean:
            continue
        print(f"\n🚀 Converting factors for {clean}")
        session = resolve_session(clean, session_id)
        print(f"📁 session: {session.name}")
        session_suffix = session.name.split("_")[-1]
        process_symbol(clean, session, session_suffix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSV factors to parquet")
    parser.add_argument("--symbol", default="0700.HK", help="单个股票代码")
    parser.add_argument("--symbols", help="逗号分隔的多个股票代码")
    parser.add_argument("--session", help="筛选会话ID，如 20251004_002115")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(BASE_DIR)
    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else [args.symbol]
    )
    run(symbols, args.session)


if __name__ == "__main__":
    main()
