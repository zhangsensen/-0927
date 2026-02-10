#!/usr/bin/env python3
"""生成“今日信号/目标持仓”文档（按现有 VEC 逻辑机械输出，不构成投资建议）。

核心对齐原则（与 scripts/batch_vec_backtest.py 保持一致）：
- 因子分数：对选定因子做横截面标准化后，在 t-1 取值求和
- 选股：stable top-k（分数降序；分数相同按 ETF 索引升序）
- 调仓日：generate_rebalance_schedule(total_periods, lookback_window, freq)
- 择时：使用配置的 LightTimingModule，trade_date 使用 asof_date 的择时信号（t-1 信号 -> t 执行）

用法示例：
  uv run python scripts/generate_today_signal.py \
    --candidates results/production_pack_20251215_031920/production_candidates.parquet \
    --asof 2025-12-12 --trade-date 2025-12-15 --capital 50000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable  # noqa: F401

import numpy as np
import pandas as pd
import yaml

from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.frozen_params import get_qdii_tickers, get_universe_mode
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.utils.rebalance import generate_rebalance_schedule
from etf_strategy.regime_gate import compute_regime_gate_arr

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Protocol:
    freq: int
    pos_size: int
    lookback_window: int
    symbols: list[str]
    data_dir: str | None
    cache_dir: str | None
    timing_enabled: bool
    extreme_threshold: float
    extreme_position: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Top 策略候选 parquet（至少包含 combo 列）",
    )
    p.add_argument(
        "--asof", type=str, required=True, help="信号计算日期（使用该日收盘数据）"
    )
    p.add_argument(
        "--trade-date", type=str, required=True, help="执行日（标签用途；按 t-1 信号）"
    )
    p.add_argument(
        "--capital", type=float, default=50_000.0, help="组合资金（用于手数估算）"
    )
    p.add_argument(
        "--lot-size", type=int, default=100, help="最小交易单位（ETF 常见 100 份）"
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="",
        help="输出目录（默认 results/today_signal_<trade-date>/）",
    )
    return p.parse_args()


def _load_protocol(config_path: Path) -> Protocol:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bt = cfg.get("backtest", {})
    data = cfg.get("data", {})
    timing = bt.get("timing") or {}

    freq = int(bt.get("freq"))
    pos_size = int(bt.get("pos_size"))
    lookback_window = int(bt.get("lookback_window") or bt.get("lookback") or 252)

    timing_enabled = bool(timing.get("enabled", True))
    extreme_threshold = float(timing.get("extreme_threshold", -0.4))
    extreme_position = float(timing.get("extreme_position", 0.3))

    symbols = list(map(str, data.get("symbols", [])))
    if not symbols:
        raise ValueError("configs/combo_wfo_config.yaml 缺少 data.symbols")

    return Protocol(
        freq=freq,
        pos_size=pos_size,
        lookback_window=lookback_window,
        symbols=symbols,
        data_dir=data.get("data_dir"),
        cache_dir=data.get("cache_dir"),
        timing_enabled=timing_enabled,
        extreme_threshold=extreme_threshold,
        extreme_position=extreme_position,
    )


def _stable_topk_indices(scores: np.ndarray, k: int) -> list[int]:
    # 与 stable_topk_indices(numba) 同逻辑：分数降序，分数相同索引升序
    # k 很小，直接 O(kN) 选择。
    n = int(scores.shape[0])
    used = np.zeros(n, dtype=bool)
    out: list[int] = []
    for _ in range(k):
        best_idx = -1
        best_score = -np.inf
        for i in range(n):
            if used[i]:
                continue
            s = float(scores[i])
            if s > best_score or (s == best_score and (best_idx < 0 or i < best_idx)):
                best_score = s
                best_idx = i
        if best_idx < 0 or not np.isfinite(best_score):
            break
        used[best_idx] = True
        out.append(best_idx)
    return out


def _iter_combo_factors(combo: str) -> list[str]:
    return [s.strip() for s in combo.split("+") if s.strip()]


def _compute_std_factors(ohlcv: dict) -> dict[str, pd.DataFrame]:
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    raw_factors = {name: raw_factors_df[name] for name in factor_names}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    return std_factors


def _ensure_asof_in_index(df: pd.DataFrame, asof: str) -> pd.Timestamp:
    asof_ts = pd.Timestamp(asof)
    if asof_ts not in df.index:
        # 明确失败，避免隐式最近值带来“错日”
        last = df.index.max()
        raise ValueError(
            f"asof={asof_ts.date()} 不在数据索引中（最新可用 {last.date()}）。"
        )
    return asof_ts


def _shares_from_value(value: float, price: float, lot_size: int) -> int:
    if not np.isfinite(price) or price <= 0:
        return 0
    raw = int(np.floor(value / price))
    return (raw // lot_size) * lot_size


def main() -> int:
    args = _parse_args()

    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    proto = _load_protocol(config_path)

    # Load raw backtest config for regime gate
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    backtest_config = raw_config.get("backtest", {})

    # Universe mode: A_SHARE_ONLY 硬禁止 QDII 交易
    universe_mode = get_universe_mode(raw_config)
    qdii_set = get_qdii_tickers(raw_config)

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        raise FileNotFoundError(str(candidates_path))

    df_candidates = pd.read_parquet(candidates_path)
    if "combo" not in df_candidates.columns:
        raise ValueError("candidates parquet 缺少 combo 列")

    # 输出目录
    outdir = (
        Path(args.outdir)
        if args.outdir
        else (ROOT / "results" / f"today_signal_{args.trade_date.replace('-', '')}")
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 加载 OHLCV 到 asof
    loader = DataLoader(data_dir=proto.data_dir, cache_dir=proto.cache_dir)
    ohlcv = loader.load_ohlcv(
        etf_codes=proto.symbols,
        start_date="2020-01-01",  # 与配置一致即可（仅用于加载范围）
        end_date=args.asof,
    )

    close_df: pd.DataFrame = ohlcv["close"]
    asof_ts = _ensure_asof_in_index(close_df, args.asof)

    # 2) 计算标准化因子（与 VEC 一致）
    std_factors = _compute_std_factors(ohlcv)
    tickers: list[str] = close_df.columns.tolist()

    # 3) 择时（trade_date 使用 asof 的信号，因为 t-1 -> t 执行）
    timing_ratio = 1.0
    if proto.timing_enabled:
        timing_module = LightTimingModule(
            extreme_threshold=proto.extreme_threshold,
            extreme_position=proto.extreme_position,
        )
        timing_series = timing_module.compute_position_ratios(close_df)
        timing_ratio = float(timing_series.loc[asof_ts])

    # 4) Regime gate（与 VEC/BT 保持一致）
    regime_ratio = 1.0
    gate_arr = compute_regime_gate_arr(
        close_df, close_df.index, backtest_config=backtest_config
    )
    if len(gate_arr) > 0:
        idx_pos = close_df.index.get_loc(asof_ts)
        regime_ratio = float(gate_arr[idx_pos])
    timing_ratio = timing_ratio * regime_ratio

    # 5) 判断 trade_date 是否调仓日（用"追加 1 个交易日"的 index 近似）
    #    trade_idx = len(close_df) 表示紧随 asof 的下一个交易日（用户已明确 12/15 是交易日）。
    T = len(close_df)
    trade_idx = T
    schedule = generate_rebalance_schedule(T + 1, proto.lookback_window, proto.freq)
    is_rebalance_day = bool(np.any(schedule == trade_idx))

    # 6) 逐策略选股（使用 asof 日的标准化因子；与 kernel 中 t-1 取值一致）
    price_asof = close_df.loc[asof_ts]

    per_strategy_rows: list[dict] = []
    agg_counts = {t: 0 for t in tickers}

    for _, row in df_candidates.iterrows():
        combo = str(row["combo"]).strip()
        factors = _iter_combo_factors(combo)

        scores = np.full(len(tickers), -np.inf, dtype=float)
        valid = 0
        for i, ticker in enumerate(tickers):
            s = 0.0
            has_value = False
            for f in factors:
                if f not in std_factors:
                    continue
                v = std_factors[f].at[asof_ts, ticker]
                if pd.notna(v):
                    s += float(v)
                    has_value = True
            if has_value and s != 0.0:
                scores[i] = s
                valid += 1

        # ✅ FIX: 过滤涨停/停牌的ETF（避免选中买不进的标的）
        # 涨停判断：当日涨幅 > 9.5%（ETF涨停约10%，留0.5%容差）
        # 停牌判断：成交量 = 0
        tradable_mask = np.ones(len(tickers), dtype=bool)

        if len(close_df) >= 2:  # 需要至少2天数据才能计算涨幅
            prev_close = close_df.iloc[-2]  # t-2日收盘
            curr_close = close_df.iloc[-1]  # t-1日收盘（asof日）
            volume_curr = ohlcv["volume"].iloc[-1]  # t-1日成交量

            for i, ticker in enumerate(tickers):
                # 检查涨停
                if pd.notna(prev_close[ticker]) and pd.notna(curr_close[ticker]):
                    pct_change = (curr_close[ticker] / prev_close[ticker]) - 1.0
                    if pct_change > 0.095:  # 涨幅 > 9.5%
                        tradable_mask[i] = False

                # 检查停牌
                if pd.notna(volume_curr[ticker]) and volume_curr[ticker] <= 0:
                    tradable_mask[i] = False

            # 将不可交易的ETF分数设为-inf
            scores[~tradable_mask] = -np.inf

        # 🚫 QDII 硬禁止: A_SHARE_ONLY 模式下 QDII 不可被选为持仓
        if universe_mode == "A_SHARE_ONLY":
            for i, ticker in enumerate(tickers):
                if ticker in qdii_set:
                    scores[i] = -np.inf

        picks: list[int] = []
        if valid >= proto.pos_size:
            picks = _stable_topk_indices(scores, proto.pos_size)

        picked_tickers = [tickers[i] for i in picks]
        picked_scores = [float(scores[i]) for i in picks]

        for tkr in picked_tickers:
            agg_counts[tkr] += 1

        per_strategy_rows.append(
            {
                "combo": combo,
                "asof_date": str(asof_ts.date()),
                "trade_date": args.trade_date,
                "is_rebalance_day": is_rebalance_day,
                "timing_ratio_trade": timing_ratio,
                "pick1": picked_tickers[0] if len(picked_tickers) > 0 else "",
                "pick2": picked_tickers[1] if len(picked_tickers) > 1 else "",
                "score1": picked_scores[0] if len(picked_scores) > 0 else np.nan,
                "score2": picked_scores[1] if len(picked_scores) > 1 else np.nan,
            }
        )

    df_per = pd.DataFrame(per_strategy_rows)

    # 6b) QDII 排名监控（不交易，仅记录当日 QDII 在全池的排名情况）
    qdii_monitor_rows: list[dict] = []
    if qdii_set:
        # 用第一个策略的因子做一次全池排名（含 QDII）
        first_combo = str(df_candidates.iloc[0]["combo"]).strip() if len(df_candidates) > 0 else ""
        first_factors = _iter_combo_factors(first_combo)
        monitor_scores = np.full(len(tickers), -np.inf, dtype=float)
        for i, ticker in enumerate(tickers):
            s = 0.0
            has_value = False
            for f_name in first_factors:
                if f_name not in std_factors:
                    continue
                v = std_factors[f_name].at[asof_ts, ticker]
                if pd.notna(v):
                    s += float(v)
                    has_value = True
            if has_value and s != 0.0:
                monitor_scores[i] = s

        # 全池排名 (1=最高分)
        valid_mask = np.isfinite(monitor_scores)
        ranks = np.full(len(tickers), len(tickers), dtype=int)
        if valid_mask.any():
            order = np.argsort(-monitor_scores[valid_mask])
            valid_indices = np.where(valid_mask)[0]
            for rank, idx in enumerate(order):
                ranks[valid_indices[idx]] = rank + 1

        for i, ticker in enumerate(tickers):
            if ticker in qdii_set:
                qdii_monitor_rows.append(
                    {
                        "ticker": ticker,
                        "score": float(monitor_scores[i]) if np.isfinite(monitor_scores[i]) else None,
                        "rank": int(ranks[i]),
                        "in_top2": ranks[i] <= 2,
                        "in_top10": ranks[i] <= 10,
                    }
                )
        qdii_monitor_rows.sort(key=lambda r: r["rank"])

    # 7) 汇总权重：等权 Top6 策略 + 每策略等权 2 只 => 每次命中权重 1/(6*2)
    n_strat = len(df_per)
    denom = max(n_strat * proto.pos_size, 1)

    agg_rows: list[dict] = []
    for tkr, cnt in sorted(agg_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if cnt <= 0:
            continue
        w_raw = cnt / denom
        w_expo = w_raw * timing_ratio
        target_value = float(args.capital) * w_expo
        px = float(price_asof.get(tkr, np.nan))
        shares = _shares_from_value(target_value, px, int(args.lot_size))
        est_value = float(shares) * px if np.isfinite(px) else 0.0
        agg_rows.append(
            {
                "ticker": tkr,
                "count": cnt,
                "weight_raw": w_raw,
                "weight_after_timing": w_expo,
                "price_asof": px,
                "target_value": target_value,
                "shares": shares,
                "est_value": est_value,
            }
        )

    df_agg = pd.DataFrame(agg_rows)

    # 8) 输出文件
    df_per.to_csv(outdir / "signals_per_strategy.csv", index=False, encoding="utf-8")
    df_agg.to_csv(outdir / "aggregate_weights.csv", index=False, encoding="utf-8")
    df_per.to_parquet(outdir / "signals_per_strategy.parquet", index=False)
    df_agg.to_parquet(outdir / "aggregate_weights.parquet", index=False)

    # 9) Markdown 简报
    title = f"TODAY_SIGNAL_{args.trade_date.replace('-', '')}"
    md_lines: list[str] = []
    md_lines.append(f"# {title}\n")
    md_lines.append(f"**Universe mode**: `{universe_mode}`")
    md_lines.append("**说明**：以下为策略引擎规则的机械信号输出（非投资建议）。")
    md_lines.append(
        f"- 信号计算日（asof）：{asof_ts.date()}（用于 {args.trade_date} 执行的 t-1 信号）"
    )
    md_lines.append(f"- 执行日（trade_date）：{args.trade_date}")
    md_lines.append(
        f"- 是否调仓日（按 freq={proto.freq}, lookback={proto.lookback_window} 的 index 近似）：{is_rebalance_day}"
    )
    md_lines.append(f"- 择时仓位系数（含 regime gate）：{timing_ratio:.3f} (regime={regime_ratio:.3f})")
    md_lines.append(
        f"- 策略数：{n_strat}，每策略持仓：{proto.pos_size}，资金：{args.capital:,.0f}\n"
    )

    md_lines.append("## 汇总目标持仓（等权 Top6 聚合）")
    if df_agg.empty:
        md_lines.append("（无有效信号）")
    else:
        md_lines.append(df_agg.head(20).to_markdown(index=False))
        if len(df_agg) > 20:
            md_lines.append(
                f"\n（仅展示前 20 行，共 {len(df_agg)} 行；详见 aggregate_weights.csv）"
            )

    md_lines.append("\n## 分策略选股（每策略 Top2）")
    md_lines.append(
        df_per[["combo", "pick1", "pick2", "score1", "score2"]].to_markdown(index=False)
    )

    # QDII 监控（只看不买）
    if qdii_monitor_rows:
        md_lines.append(f"\n## QDII 监控（{universe_mode} — 不交易，仅排名追踪）")
        df_qdii_mon = pd.DataFrame(qdii_monitor_rows)
        md_lines.append(df_qdii_mon.to_markdown(index=False))
        best_qdii = qdii_monitor_rows[0]
        md_lines.append(
            f"\n最佳 QDII: {best_qdii['ticker']} (rank {best_qdii['rank']}/{len(tickers)}, "
            f"in_top10={'Y' if best_qdii['in_top10'] else 'N'})"
        )

    (outdir / "TODAY_SIGNAL.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    print(f"✅ wrote: {outdir}")
    print(f"- {outdir / 'TODAY_SIGNAL.md'}")
    print(f"- {outdir / 'signals_per_strategy.csv'}")
    print(f"- {outdir / 'aggregate_weights.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
