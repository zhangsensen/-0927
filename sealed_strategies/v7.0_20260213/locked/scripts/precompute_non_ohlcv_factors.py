#!/usr/bin/env python3
"""
Pre-compute non-OHLCV factors and save as parquet files.

These parquet files are loaded by the WFO via the external factors mechanism
(EXTRA_FACTORS_DIR env var or combo_wfo.extra_factors.factors_dir config).

Usage:
    uv run python scripts/precompute_non_ohlcv_factors.py

Output:
    results/non_ohlcv_factors/{FACTOR_NAME}.parquet
"""

import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.non_ohlcv_factors import (
    NON_OHLCV_FACTOR_NAMES,
    compute_non_ohlcv_factors,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "results" / "non_ohlcv_factors"


def main():
    # Load config
    config_path = BASE_DIR / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load OHLCV data
    data_cfg = config["data"]
    loader = DataLoader(
        data_dir=data_cfg.get("data_dir"),
        cache_dir=data_cfg.get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=data_cfg.get("symbols"),
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
    )
    logger.info(
        f"OHLCV loaded: {len(ohlcv['close'].columns)} ETFs, "
        f"{len(ohlcv['close'])} dates"
    )

    # Compute all non-OHLCV factors
    factors = compute_non_ohlcv_factors(loader, ohlcv)

    # Save each factor as parquet
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in sorted(factors.items()):
        out_path = OUT_DIR / f"{name}.parquet"
        df.to_parquet(out_path)
        valid_pct = df.notna().mean().mean() * 100
        logger.info(f"  Saved {name}: {df.shape}, {valid_pct:.1f}% valid -> {out_path}")

    print(f"\n{len(factors)} factor files saved to {OUT_DIR}/")
    print(f"Set EXTRA_FACTORS_DIR={OUT_DIR} or update config to use these factors.")


if __name__ == "__main__":
    main()
