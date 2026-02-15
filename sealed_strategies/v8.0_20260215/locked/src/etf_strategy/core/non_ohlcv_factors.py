"""
Non-OHLCV Factor Computation Module

Computes factors from fund_share, margin, and other non-price data sources.
These factors are orthogonal to the existing OHLCV-derived factor space.

IC Screening Results (2026-02-12):
  SHARE_CHG_10D:    IC=-0.056, IR=-0.27  (fund_share)
  SHARE_CHG_5D:     IC=-0.050, IR=-0.25  (fund_share)
  SHARE_CHG_20D:    IC=-0.047, IR=-0.21  (fund_share)
  MARGIN_CHG_10D:   IC=-0.047, IR=-0.20  (margin)
  SHARE_ACCEL:      IC=+0.034, IR=+0.16  (fund_share)
  MARGIN_BUY_RATIO: IC=-0.031, IR=-0.13  (margin)

Usage:
    from etf_strategy.core.non_ohlcv_factors import compute_non_ohlcv_factors
    from etf_strategy.core.data_loader import DataLoader

    loader = DataLoader(config)
    ohlcv = loader.load_ohlcv()
    factors = compute_non_ohlcv_factors(loader, ohlcv)
    # factors: Dict[str, pd.DataFrame], each (T, N) aligned to OHLCV dates
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 6 IC-winning non-OHLCV factors
NON_OHLCV_FACTOR_NAMES = [
    "SHARE_CHG_5D",
    "SHARE_CHG_10D",
    "SHARE_CHG_20D",
    "SHARE_ACCEL",
    "MARGIN_CHG_10D",
    "MARGIN_BUY_RATIO",
]


def compute_non_ohlcv_factors(
    loader,
    ohlcv: Dict[str, pd.DataFrame],
    factor_names: Optional[list] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute non-OHLCV factors using data_loader extensions.

    Args:
        loader: DataLoader instance with fund_share/margin methods.
        ohlcv: OHLCV data dict with 'close', 'volume' DataFrames
            (DatetimeIndex, columns=ETF codes).
        factor_names: Subset of factors to compute (None = all 6).

    Returns:
        Dict mapping factor name -> DataFrame (DatetimeIndex, ETF code columns).
        Values are raw (un-standardized); cross-section processing happens in WFO.
    """
    if factor_names is None:
        factor_names = NON_OHLCV_FACTOR_NAMES

    close = ohlcv["close"]
    volume = ohlcv["volume"]
    trading_dates = close.index
    etf_codes = list(close.columns)

    factors: Dict[str, pd.DataFrame] = {}

    # --- fund_share factors ---
    share_factors_needed = [f for f in factor_names if f.startswith("SHARE_")]
    if share_factors_needed:
        logger.info(f"Loading fund_share for {len(share_factors_needed)} factors...")
        share_panel = loader.load_fund_share(
            etf_codes=etf_codes, trading_dates=trading_dates
        )
        # Ensure alignment: reindex to same columns as close
        share_panel = share_panel.reindex(columns=etf_codes)

        share_computed = _compute_share_factors(share_panel, volume, share_factors_needed)
        factors.update(share_computed)

    # --- margin factors ---
    margin_factors_needed = [f for f in factor_names if f.startswith("MARGIN_")]
    if margin_factors_needed:
        logger.info(f"Loading margin data for {len(margin_factors_needed)} factors...")
        rzye = loader.load_margin(field="rzye", etf_codes=etf_codes, trading_dates=trading_dates)
        rzmre = loader.load_margin(field="rzmre", etf_codes=etf_codes, trading_dates=trading_dates)

        margin_computed = _compute_margin_factors(
            rzye, rzmre, close, volume, etf_codes, margin_factors_needed
        )
        factors.update(margin_computed)

    computed = sorted(factors.keys())
    missing = sorted(set(factor_names) - set(computed))
    if missing:
        logger.warning(f"Failed to compute: {missing}")
    logger.info(f"Non-OHLCV factors computed: {computed}")

    return factors


def _compute_share_factors(
    share_panel: pd.DataFrame,
    volume: pd.DataFrame,
    needed: list,
) -> Dict[str, pd.DataFrame]:
    """Compute fund_share-based factors."""
    results: Dict[str, pd.DataFrame] = {}
    eps = 1e-10

    share_shift_5 = share_panel.shift(5)
    share_shift_10 = share_panel.shift(10)
    share_shift_20 = share_panel.shift(20)

    if "SHARE_CHG_5D" in needed or "SHARE_ACCEL" in needed:
        chg_5d = (share_panel - share_shift_5) / (share_shift_5.abs() + eps)
        chg_5d = chg_5d.where(share_shift_5.abs() > eps, np.nan)
        if "SHARE_CHG_5D" in needed:
            results["SHARE_CHG_5D"] = chg_5d

    if "SHARE_CHG_10D" in needed:
        chg_10d = (share_panel - share_shift_10) / (share_shift_10.abs() + eps)
        chg_10d = chg_10d.where(share_shift_10.abs() > eps, np.nan)
        results["SHARE_CHG_10D"] = chg_10d

    if "SHARE_CHG_20D" in needed or "SHARE_ACCEL" in needed:
        chg_20d = (share_panel - share_shift_20) / (share_shift_20.abs() + eps)
        chg_20d = chg_20d.where(share_shift_20.abs() > eps, np.nan)
        if "SHARE_CHG_20D" in needed:
            results["SHARE_CHG_20D"] = chg_20d

    if "SHARE_ACCEL" in needed:
        # SHARE_ACCEL = SHARE_CHG_5D - SHARE_CHG_20D
        results["SHARE_ACCEL"] = chg_5d - chg_20d

    return results


def _compute_margin_factors(
    rzye: pd.DataFrame,
    rzmre: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    etf_codes: list,
    needed: list,
) -> Dict[str, pd.DataFrame]:
    """Compute margin-based factors."""
    results: Dict[str, pd.DataFrame] = {}
    eps = 1e-10

    if "MARGIN_CHG_10D" in needed:
        rzye_shift = rzye.shift(10)
        chg = (rzye - rzye_shift) / (rzye_shift.abs() + eps)
        chg = chg.where(rzye_shift.abs() > eps, np.nan)
        results["MARGIN_CHG_10D"] = chg.reindex(columns=etf_codes)

    if "MARGIN_BUY_RATIO" in needed:
        # daily_amount = close * volume (proxy for turnover in yuan)
        daily_amount = close * volume
        daily_amount_safe = daily_amount.where(daily_amount.abs() > eps, np.nan)
        # rzmre columns may not cover all etf_codes (43/49 have margin data)
        ratio = rzmre / daily_amount_safe
        results["MARGIN_BUY_RATIO"] = ratio.reindex(columns=etf_codes)

    return results
