from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional

import pandas as pd

from .cn_calendar import CNCalendar


def _default_agg_map(columns: Iterable[str]) -> Dict[str, str]:
    cols = set(columns)
    agg: Dict[str, str] = {}
    if {"open", "high", "low", "close"}.issubset(cols):
        agg.update(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }
        )
    if "volume" in cols:
        agg["volume"] = "sum"
    if "amount" in cols:
        agg["amount"] = "sum"
    # any other numeric columns -> sum by default (safe for typical bars)
    for c in cols:
        if c in agg:
            continue
        # pandas will ignore non-numerics when using sum via .agg where possible
        agg[c] = "sum"
    return agg


def resample_ashare_intraday(
    df: pd.DataFrame,
    rule: str,
    *,
    calendar: Optional[CNCalendar] = None,
    agg_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Session-aware resample for A-shares (handles lunch break correctly).

    - Accepts a DataFrame indexed by Timestamp (tz-naive ok), optionally MultiIndex with symbol.
    - Produces bins aligned within each session only; no 11:30/12:30/15:30 artifacts.
    - rule: like '5min','15min','30min','60min'. Must divide 120 minutes (per session) evenly.
    """
    if df.empty:
        return df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must be indexed by DatetimeIndex")

    cal = calendar or CNCalendar()
    # Work per-symbol if MultiIndex
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2:
        pieces = []
        for sym, sdf in df.groupby(level=0, sort=False):
            sdf = sdf.droplevel(0)
            pieces.append(
                _resample_single_symbol(sdf, rule, cal, agg_map)
                .assign(symbol=sym)
                .set_index("symbol", append=True)
                .swaplevel(0, 1)
            )
        out = pd.concat(pieces).sort_index()
        return out
    else:
        return _resample_single_symbol(df, rule, cal, agg_map)


def _resample_single_symbol(
    df: pd.DataFrame,
    rule: str,
    cal: CNCalendar,
    agg_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    df = df.sort_index()
    # normalize date range
    start_d = df.index[0].date()
    end_d = df.index[-1].date()
    out_parts = []
    agg = agg_map or _default_agg_map(df.columns)

    for d in cal.iter_trading_days(start_d, end_d):
        day_slice = (
            df.loc[str(d)]
            if str(d) in df.index.normalize().unique().astype(str)
            else None
        )
        if day_slice is None or day_slice.empty:
            continue
        for s, e in cal.trading_sessions(d):
            # Slice minutes as right-exclusive at session end, but resampled bar labels represent right edge
            ss = pd.Timestamp(s)
            ee = pd.Timestamp(e)
            mask = (day_slice.index >= ss) & (day_slice.index < ee)
            seg = day_slice.loc[mask]
            if seg.empty:
                continue
            # anchor bins at session start for this day
            # Use label='left' (start labels), then shift to right-edge (end) labels
            res = seg.resample(rule, label="left", closed="left", origin=ss).agg(agg)
            # Shift index to represent bar end time (right edge)
            freq_td = pd.Timedelta(rule)
            res.index = res.index + freq_td
            # Keep bars whose right-edge <= ee (include 11:30 and 15:00)
            res = res[res.index <= ee]
            out_parts.append(res)

    if not out_parts:
        return df.iloc[0:0].copy()
    return pd.concat(out_parts).sort_index()
