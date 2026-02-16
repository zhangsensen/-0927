"""ETF pool mapping: load pool classification from etf_pools.yaml and build index arrays.

Used by VEC backtest kernel and signal generator for pool diversity constraints.
"""

from pathlib import Path
from typing import Union

import numpy as np
import yaml

# Canonical pool name → integer ID mapping (7 pools, 0-indexed)
POOL_NAME_TO_ID: dict[str, int] = {
    "EQUITY_BROAD": 0,
    "EQUITY_GROWTH": 1,
    "EQUITY_CYCLICAL": 2,
    "EQUITY_DEFENSIVE": 3,
    "BOND": 4,
    "COMMODITY": 5,
    "QDII": 6,
}

POOL_ID_TO_NAME: dict[int, str] = {v: k for k, v in POOL_NAME_TO_ID.items()}


def load_pool_mapping(etf_pools_path: Union[str, Path]) -> dict[str, int]:
    """Read pools section from etf_pools.yaml and return ticker → pool_id dict.

    Only reads the 7 canonical pools (EQUITY_BROAD through QDII).
    Ignores legacy pools like A_SHARE_LIVE.

    Args:
        etf_pools_path: Path to etf_pools.yaml

    Returns:
        Dict mapping ticker string (e.g. "510300") to pool_id integer (0-6).
    """
    with open(etf_pools_path) as f:
        cfg = yaml.safe_load(f)

    pools = cfg.get("pools", {})
    ticker_to_pool: dict[str, int] = {}

    for pool_name, pool_id in POOL_NAME_TO_ID.items():
        pool_cfg = pools.get(pool_name, {})
        symbols = pool_cfg.get("symbols", [])
        for sym in symbols:
            ticker_to_pool[str(sym)] = pool_id

    return ticker_to_pool


def build_pool_array(etf_codes: list[str], pool_mapping: dict[str, int]) -> np.ndarray:
    """Build (N,) int64 array where arr[i] = pool_id of etf_codes[i].

    Unmapped ETFs get -1 (will be treated as unique pool by constraint logic).

    Args:
        etf_codes: Ordered list of ETF tickers matching backtest index order.
        pool_mapping: Output of load_pool_mapping().

    Returns:
        np.ndarray of shape (N,) dtype int64.
    """
    pool_ids = np.full(len(etf_codes), -1, dtype=np.int64)
    for i, code in enumerate(etf_codes):
        # Strip exchange suffix if present (e.g. "510300.SH" → "510300")
        bare = code.split(".")[0]
        if bare in pool_mapping:
            pool_ids[i] = pool_mapping[bare]
    return pool_ids
