from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

_TF_MAP: Dict[str, str] = {
    "daily": "1day",
    "1d": "1day",
    "1day": "1day",
    "60min": "60m",
    "60m": "60m",
    "30min": "30m",
    "30m": "30m",
    "15min": "15m",
    "15m": "15m",
    "5min": "5min",
    "5m": "5min",
}


def _normalize_symbol(symbol: str) -> str:
    """Convert symbol like '0700.HK' to raw filename prefix '0700HK'."""
    return symbol.replace(".", "")


def _normalize_timeframe(tf: str) -> str:
    key = tf.strip().lower()
    return _TF_MAP.get(key, key)


@dataclass(frozen=True)
class PriceDataLoader:
    root: Path = Path("/Users/zhangshenshen/深度量化0927/raw/HK")

    def load_price(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load canonical OHLCV price frame from raw parquet files.

        Requirements:
        - File pattern: <SYMBOL_NO_DOT>_<TF>_*.parquet (e.g., 0700HK_60m_*.parquet)
        - Must contain a 'close' column; otherwise raise.
        """

        symbol_key = _normalize_symbol(symbol)
        tf_key = _normalize_timeframe(timeframe)

        tf_dir = self.root
        if not tf_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {tf_dir}")

        pattern = f"{symbol_key}_{tf_key}_*.parquet"
        files = glob.glob(str(tf_dir / pattern))

        if not files:
            raise FileNotFoundError(
                f"No raw price file found for symbol={symbol} timeframe={timeframe} pattern={pattern}"
            )

        latest_file = max(files, key=os.path.getmtime)
        price_data = pd.read_parquet(latest_file)

        if "datetime" in price_data.columns:
            price_data = price_data.set_index("datetime")
        elif "timestamp" in price_data.columns:
            price_data = price_data.set_index("timestamp")

        price_data.columns = price_data.columns.str.lower()

        if "close" not in price_data.columns:
            raise ValueError(
                f"Raw file {latest_file} lacks 'close' column; cannot proceed without true prices."
            )

        return price_data
