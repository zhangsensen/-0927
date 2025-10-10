#!/usr/bin/env python3
"""Resample HK 1-minute parquet files into higher timeframes.

This utility replaces the earlier hard-coded script by accepting configurable
data directories and timeframe targets. It uses a lightweight pandas-based
implementation optimized for Hong Kong market data.

Usage examples::

    python batch_resample_hk.py \
        --data-root raw/HK \
        --output-dir raw/HK/resampled \
        --timeframes 15m 30m 60m

Environment variables ``HK_RESAMPLE_DATA_ROOT``, ``HK_RESAMPLE_OUTPUT_DIR`` and
``HK_RESAMPLE_TIMEFRAMES`` (comma-separated) are honoured when CLI arguments are
omitted.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def _normalize_timeframe_label(label: str) -> str:
    mapping = {
        "1h": "60m",
        "60min": "60m",
        "30min": "30m",
        "15min": "15m",
        "5min": "5m",
        "1d": "1day",
        "daily": "1day",
    }
    lower = label.lower()
    return mapping.get(lower, lower)


def _timeframe_to_rule(label: str) -> str:
    normalized = _normalize_timeframe_label(label)
    if normalized.endswith("m"):
        minutes = int(normalized[:-1])
        return f"{minutes}min"
    if normalized.endswith("h"):
        hours = int(normalized[:-1])
        return f"{hours}H"
    if normalized == "1day":
        return "1D"
    raise ValueError(f"Unsupported timeframe label: {label}")


def _ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data.index, pd.DatetimeIndex):
        if "timestamp" in data.columns:
            data = data.set_index(pd.to_datetime(data["timestamp"]))
        else:
            data.index = pd.to_datetime(data.index)
    return data.sort_index()


@dataclass
class _SimpleHKResampler:
    """Minimal pandas-based resampler used when the external helper is absent."""

    def resample(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        rule = _timeframe_to_rule(timeframe)
        normalized = _normalize_timeframe_label(timeframe)

        aggregations = {}
        if "open" in data.columns:
            aggregations["open"] = "first"
        if "high" in data.columns:
            aggregations["high"] = "max"
        if "low" in data.columns:
            aggregations["low"] = "min"
        if "close" in data.columns:
            aggregations["close"] = "last"
        if "volume" in data.columns:
            aggregations["volume"] = "sum"

        if not aggregations:
            raise ValueError(
                "Input data must contain OHLC or volume columns for resampling"
            )

        resampled = data.resample(rule, label="right", closed="right").agg(aggregations)
        resampled.dropna(how="all", inplace=True)
        if "close" in resampled.columns:
            resampled = resampled[resampled["close"].notna()]
        resampled.index.name = data.index.name or "timestamp"
        resampled.attrs["timeframe"] = normalized
        return resampled


def _resolve_resampler() -> callable:
    """Return the pandas-based resampler function."""
    resampler = _SimpleHKResampler()
    return resampler.resample


def _iter_timeframes(timeframes: Sequence[str] | None) -> Iterable[str]:
    if timeframes:
        return tuple(timeframes)
    env_override = os.getenv("HK_RESAMPLE_TIMEFRAMES")
    if env_override:
        return tuple(part.strip() for part in env_override.split(",") if part.strip())
    return ("15m", "30m", "60m")


def batch_resample_all_1m(
    data_root: Path,
    output_dir: Path | None = None,
    timeframes: Sequence[str] | None = None,
) -> Path:
    """Resample all 1-minute parquet files under ``data_root``."""

    data_root = data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    files_1m = sorted(data_root.glob("*1m*.parquet"))
    if not files_1m:
        raise FileNotFoundError(f"No 1-minute parquet files found under {data_root}")

    if output_dir is None:
        output_dir = data_root / "resampled"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resample_func = _resolve_resampler()
    targets = list(_iter_timeframes(timeframes))

    print(f"发现 {len(files_1m)} 个1分钟文件待处理")
    print(f"目标时间框架: {targets}")

    success_count = 0
    for file_path in files_1m:
        try:
            print(f"处理: {file_path.name}")
            data = pd.read_parquet(file_path)
            data = _ensure_datetime_index(data)

            original_rows = len(data)
            stock_code, _, *tail = file_path.stem.split("_")
            date_range = "_".join(tail)

            for tf in targets:
                normalized_tf = _normalize_timeframe_label(tf)
                try:
                    resampled = resample_func(data, normalized_tf)
                except Exception as exc:
                    print(f"  {tf} 失败: {exc}")
                    continue

                output_file = (
                    output_dir / f"{stock_code}_{normalized_tf}_{date_range}.parquet"
                )
                resampled.to_parquet(output_file)
                compression_ratio = original_rows / max(len(resampled), 1)
                print(
                    f"  {normalized_tf}: {len(resampled)} 行 (压缩比 {compression_ratio:.1f}:1)"
                )

            success_count += 1
        except Exception as exc:
            print(f"❌ {file_path.name} 失败: {exc}")

    print("\n批量处理完成:")
    print(f"✅ 成功: {success_count} 个文件")
    print(f"📁 输出目录: {output_dir}")
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch resample HK 1-minute data")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.getenv("HK_RESAMPLE_DATA_ROOT"),
        help="Directory containing raw 1-minute parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=os.getenv("HK_RESAMPLE_OUTPUT_DIR"),
        help="Directory to write resampled parquet files",
    )
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=None,
        help="Target timeframes (e.g. 15m 30m 60m)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = args.data_root or Path("raw/HK")
    output_dir = args.output_dir
    batch_resample_all_1m(
        data_root=data_root, output_dir=output_dir, timeframes=args.timeframes
    )


if __name__ == "__main__":
    main()
