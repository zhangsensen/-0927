import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure `src/` importable when running as a script
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor

DEFAULT_COMBO = (
    "ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARPE_RATIO_20D"
)


def _parse_combo(combo: str) -> list[str]:
    return [part.strip() for part in combo.split("+") if part.strip()]


def load_ohlcv(symbols: list[str], start_date: str | None, end_date: str | None):
    data_dir = PROJECT_ROOT / "raw" / "ETF" / "daily"
    loader = DataLoader(data_dir=str(data_dir))
    return loader.load_ohlcv(
        etf_codes=symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=True,
    )


def compute_factor_panels(ohlcv: dict, factors: list[str]) -> dict[str, pd.DataFrame]:
    lib = PreciseFactorLibrary()
    all_factors = lib.compute_all_factors(ohlcv)
    panels: dict[str, pd.DataFrame] = {}
    for f in factors:
        if f not in all_factors.columns.levels[0]:
            raise ValueError(f"Factor {f} not found in PreciseFactorLibrary output")
        panels[f] = all_factors[f]
    return panels


def score_rank_sum(
    factor_panels: dict[str, pd.DataFrame],
    factors: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """rank(pct=True) 等权求和（接近你原脚本/实盘脚本的融合方式），但按元数据修正 low_is_good。"""
    lib = PreciseFactorLibrary()
    per_factor_scores: dict[str, pd.DataFrame] = {}
    for f in factors:
        direction = lib.factors_metadata[f].direction
        values = factor_panels[f]

        if direction == "low_is_good":
            # 低值好：对值取负再 rank，避免手工 1-pct 的边界问题
            per_factor_scores[f] = (-values).rank(axis=1, pct=True)
        else:
            # high_is_good / neutral：默认越大越好
            per_factor_scores[f] = values.rank(axis=1, pct=True)

    composite = sum(per_factor_scores.values())
    return composite, per_factor_scores


def score_wfo_style(
    factor_panels: dict[str, pd.DataFrame],
    factors: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """WFO 风格：先 CrossSectionProcessor 标准化（无界 zscore+winsor，有界透传），再等权平均。"""
    processor = CrossSectionProcessor(verbose=False)
    processed = processor.process_all_factors({f: factor_panels[f] for f in factors})
    composite = sum(processed.values()) / float(len(factors))
    return composite, processed


def _shift_signal_date(
    index: pd.DatetimeIndex, date: pd.Timestamp, lag: int
) -> pd.Timestamp:
    if date not in index:
        raise ValueError(f"Date {date.date()} not found in trading calendar")
    loc = index.get_loc(date)
    if isinstance(loc, slice) or isinstance(loc, np.ndarray):
        # Should not happen for unique index
        loc = int(np.where(index == date)[0][0])
    signal_loc = int(loc) - lag
    if signal_loc < 0:
        raise ValueError(f"Not enough history to lag {lag} days for {date.date()}")
    return index[signal_loc]


def build_snapshot(
    ohlcv: dict,
    composite: pd.DataFrame,
    per_factor: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    factors: list[str],
) -> pd.DataFrame:
    close = ohlcv["close"].loc[signal_date]
    out = pd.DataFrame({"code": close.index.astype(str), "close": close.values})
    out["SCORE"] = composite.loc[signal_date].reindex(close.index).values
    for f in factors:
        out[f] = per_factor[f].loc[signal_date].reindex(close.index).values
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Rebalance logic diagnosis with aligned factor definitions."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "combo_wfo_config.yaml"),
        help="Path to combo_wfo_config.yaml",
    )
    parser.add_argument(
        "--combo",
        default=DEFAULT_COMBO,
        help="Factor combo string, e.g. 'ADX_14D + MAX_DD_60D + PRICE_POSITION_120D'",
    )
    parser.add_argument(
        "--dates",
        default="2025-10-28,2025-11-10",
        help="Comma separated diagnosis dates (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=1,
        help="Signal lag in trading days (WFO/BT uses t-1 signal)",
    )
    parser.add_argument(
        "--method",
        choices=["rank_sum", "wfo_style", "both"],
        default="both",
        help="Scoring method",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional start date for loading data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional end date for loading data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--targets",
        default="516090,516160,513500,159949",
        help="Comma separated target ETF codes to print",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    symbols = list(config["data"]["symbols"])
    combo_factors = _parse_combo(args.combo)
    dates = [pd.Timestamp(d.strip()) for d in args.dates.split(",") if d.strip()]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    print("⏳ Loading OHLCV via DataLoader...")
    ohlcv = load_ohlcv(
        symbols=symbols, start_date=args.start_date, end_date=args.end_date
    )

    print("⏳ Computing factor panels via PreciseFactorLibrary...")
    factor_panels = compute_factor_panels(ohlcv, combo_factors)

    methods: list[tuple[str, callable]] = []
    if args.method in ("rank_sum", "both"):
        methods.append(("rank_sum", score_rank_sum))
    if args.method in ("wfo_style", "both"):
        methods.append(("wfo_style", score_wfo_style))

    out_dir = PROJECT_ROOT / "results" / "diagnostics" / "rebalance_logic"
    out_dir.mkdir(parents=True, exist_ok=True)

    calendar = ohlcv["close"].index

    for d in dates:
        print("\n" + "=" * 80)
        print(f"📅 Diagnosis Date: {d.date()}  (lag={args.lag})")
        print("=" * 80)

        signal_date = _shift_signal_date(calendar, d, lag=args.lag)
        print(f"🧭 Signal Date Used: {signal_date.date()} (t-{args.lag})")

        for method_name, scorer in methods:
            composite, per_factor = scorer(factor_panels, combo_factors)
            snap = build_snapshot(
                ohlcv=ohlcv,
                composite=composite,
                per_factor=per_factor,
                signal_date=signal_date,
                factors=combo_factors,
            )

            ranked = snap.sort_values("SCORE", ascending=False).reset_index(drop=True)
            ranked["Rank"] = np.arange(1, len(ranked) + 1)

            print(f"\n🏷️  Method: {method_name}")
            print("🏆 Top 5 Selected:")
            print(ranked[["Rank", "code", "SCORE", "close"]].head(5))

            if targets:
                print("\n🔍 Targets:")
                tdf = ranked[ranked["code"].isin(targets)].copy()
                if not tdf.empty:
                    cols = ["Rank", "code", "SCORE", "close"] + combo_factors
                    print(tdf[cols])
                else:
                    print("(no targets found in universe)")

            out_path = (
                out_dir
                / f"rebalance_diag_{d.strftime('%Y%m%d')}_{method_name}_lag{args.lag}.csv"
            )
            ranked.to_csv(out_path, index=False)
            print(f"💾 Saved: {out_path}")


if __name__ == "__main__":
    main()
