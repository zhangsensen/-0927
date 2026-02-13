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
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable  # noqa: F401

import numpy as np
import pandas as pd
import yaml

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import CURRENT_VERSION, get_qdii_tickers, get_universe_mode
from etf_strategy.core.hysteresis import apply_hysteresis
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import generate_rebalance_schedule
from etf_strategy.regime_gate import compute_regime_gate_arr

ROOT = Path(__file__).resolve().parent.parent


logger = logging.getLogger(__name__)

STATE_DIR = ROOT / "data" / "live"


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
    delta_rank: float
    min_hold_days: int


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
    p.add_argument(
        "--shadow-config",
        type=str,
        default="configs/shadow_strategies.yaml",
        help="Shadow strategy definitions (default: configs/shadow_strategies.yaml)",
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

    hyst = bt.get("hysteresis") or {}
    delta_rank = float(hyst.get("delta_rank", 0.0))
    min_hold_days = int(hyst.get("min_hold_days", 0))

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
        delta_rank=delta_rank,
        min_hold_days=min_hold_days,
    )


def _load_signal_state(path: Path) -> dict | None:
    """Load signal state from JSON file.

    Expected format:
    {
      "last_asof_date": "2025-12-12",
      "strategies": {
        "combo_key": {
          "signal_portfolio": ["510300", "512880"],
          "signal_hold_days": {"510300": 3, "512880": 1}
        }
      }
    }
    """
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_signal_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate_signal_state(
    state: dict,
    freq: int,
    universe_mode: str,
    tradable_symbols: set[str],
) -> list[str]:
    """Validate state file against current environment. Returns list of mismatch reasons.

    If non-empty, caller should cold-start (discard state).
    """
    problems: list[str] = []

    # 1) version compatibility
    sv = state.get("version", "")
    if sv and not sv.startswith("v5.0"):
        problems.append(f"version mismatch: state={sv}, current={CURRENT_VERSION}")

    # 2) freq
    sf = state.get("freq")
    if sf is not None and int(sf) != freq:
        problems.append(f"freq mismatch: state={sf}, config={freq}")

    # 3) universe_mode
    sm = state.get("universe_mode", "")
    if sm and sm != universe_mode:
        problems.append(f"universe_mode mismatch: state={sm}, config={universe_mode}")

    # 4) signal_portfolio tickers in tradable symbols
    for combo, st in state.get("strategies", {}).items():
        for ticker in st.get("signal_portfolio", []):
            if ticker not in tradable_symbols:
                problems.append(
                    f"combo {combo}: ticker {ticker} not in tradable symbols"
                )
                break  # one per combo is enough

    return problems


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


def _pool_diversify_topk(
    scores: np.ndarray, pool_ids: np.ndarray, pos_size: int, extended_k: int
) -> list[int]:
    """Python equivalent of pool_diversify_topk (mirrors @njit version in VEC)."""
    candidates = _stable_topk_indices(scores, extended_k)
    if not candidates:
        return candidates

    result: list[int] = [candidates[0]]
    used_pools: list[int] = [int(pool_ids[candidates[0]])]

    for _ in range(1, pos_size):
        found = False
        # Prefer cross-pool candidate
        for idx in candidates[1:]:
            if not np.isfinite(scores[idx]):
                break
            if idx in result:
                continue
            cand_pool = int(pool_ids[idx])
            if cand_pool != -1 and cand_pool in used_pools:
                continue
            result.append(idx)
            used_pools.append(cand_pool)
            found = True
            break
        # Fallback: best remaining regardless of pool
        if not found:
            for idx in candidates[1:]:
                if not np.isfinite(scores[idx]):
                    break
                if idx not in result:
                    result.append(idx)
                    used_pools.append(int(pool_ids[idx]))
                    found = True
                    break
        if not found:
            break

    return result


def _iter_combo_factors(combo: str) -> list[str]:
    return [s.strip() for s in combo.split("+") if s.strip()]


def _compute_std_factors(
    ohlcv: dict, config: dict, data_dir: Path, loader=None,
) -> dict[str, pd.DataFrame]:
    """Compute standardized factors via FactorCache (includes non-OHLCV factors)."""
    factor_cache = FactorCache(
        cache_dir=Path(config["data"].get("cache_dir") or ".cache")
    )
    cached = factor_cache.get_or_compute(
        ohlcv=ohlcv, config=config, data_dir=data_dir, loader=loader,
    )
    return cached["std_factors"]


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

    # 2) 计算标准化因子（与 VEC 一致，含 non-OHLCV 因子）
    std_factors = _compute_std_factors(ohlcv, raw_config, loader.data_dir, loader=loader)

    # Exp5: temporal EMA smoothing
    ts_cfg = backtest_config.get("temporal_smoothing", {})
    if ts_cfg.get("enabled", False):
        from etf_strategy.core.cross_section_processor import apply_temporal_ema

        ema_span = int(ts_cfg.get("ema_span", 5))
        std_factors = apply_temporal_ema(std_factors, ema_span)

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

    # 6) 加载迟滞状态 (hysteresis state persistence)
    hysteresis_enabled = proto.delta_rank > 0 or proto.min_hold_days > 0
    state_path = STATE_DIR / "signal_state.json"
    prev_state = _load_signal_state(state_path) if hysteresis_enabled else None
    prev_strategies: dict = {}
    elapsed_days = 0  # trading days since last state; surfaced in report

    if prev_state:
        # Validate state against current environment
        tradable_set = set(tickers)
        problems = _validate_signal_state(
            prev_state, proto.freq, universe_mode, tradable_set
        )
        if problems:
            for p in problems:
                logger.warning(f"状态文件校验失败，冷启动: {p}")
            print(f"⚠️  状态文件校验失败 ({len(problems)} 项)，冷启动（不使用旧 state）")
            for p in problems:
                print(f"   - {p}")
            prev_state = None  # cold start

    if prev_state and prev_state.get("last_asof_date"):
        last_ts = pd.Timestamp(prev_state["last_asof_date"])
        # Count trading days elapsed between last_asof and current asof
        mask = (close_df.index > last_ts) & (close_df.index <= asof_ts)
        elapsed_days = int(mask.sum())
        # Increment hold_days for all held positions
        for _combo_key, st in prev_state.get("strategies", {}).items():
            for ticker in st.get("signal_portfolio", []):
                st["signal_hold_days"][ticker] = (
                    st["signal_hold_days"].get(ticker, 0) + elapsed_days
                )
        prev_strategies = prev_state.get("strategies", {})

    # 6a-pre) Pool diversity constraint (mirrors VEC kernel logic)
    pool_constraint_cfg = backtest_config.get("portfolio_constraints", {}).get("pool_diversity", {})
    pool_diversity_enabled = pool_constraint_cfg.get("enabled", False)
    pool_extended_k = pool_constraint_cfg.get("extended_k", 10) if pool_diversity_enabled else 0
    if pool_diversity_enabled:
        from etf_strategy.core.etf_pool_mapper import load_pool_mapping, build_pool_array
        _pool_map = load_pool_mapping(ROOT / "configs" / "etf_pools.yaml")
        pool_ids = build_pool_array(list(tickers), _pool_map)
        logger.info(f"池约束: enabled, extended_k={pool_extended_k}")
    else:
        pool_ids = None
        pool_extended_k = 0

    # 6a) 逐策略选股（使用 asof 日的标准化因子；与 kernel 中 t-1 取值一致）
    price_asof = close_df.loc[asof_ts]

    per_strategy_rows: list[dict] = []
    agg_counts = {t: 0 for t in tickers}
    new_strategies: dict = {}

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

        # 过滤涨停/停牌的ETF（避免选中买不进的标的）
        tradable_mask = np.ones(len(tickers), dtype=bool)

        if len(close_df) >= 2:
            prev_close = close_df.iloc[-2]
            curr_close = close_df.iloc[-1]
            volume_curr = ohlcv["volume"].iloc[-1]

            for i, ticker in enumerate(tickers):
                if pd.notna(prev_close[ticker]) and pd.notna(curr_close[ticker]):
                    pct_change = (curr_close[ticker] / prev_close[ticker]) - 1.0
                    if pct_change > 0.095:
                        tradable_mask[i] = False
                if pd.notna(volume_curr[ticker]) and volume_curr[ticker] <= 0:
                    tradable_mask[i] = False

            scores[~tradable_mask] = -np.inf

        # QDII 硬禁止: A_SHARE_ONLY 模式下 QDII 不可被选为持仓
        if universe_mode == "A_SHARE_ONLY":
            for i, ticker in enumerate(tickers):
                if ticker in qdii_set:
                    scores[i] = -np.inf

        picks: list[int] = []
        hyst_kept: list[str] = []  # tickers kept by hysteresis (for reporting)

        if valid >= proto.pos_size:
            # ✅ Pool diversity: prefer cross-pool candidates
            if pool_ids is not None and pool_extended_k > 0:
                top_indices = _pool_diversify_topk(scores, pool_ids, proto.pos_size, pool_extended_k)
            else:
                top_indices = _stable_topk_indices(scores, proto.pos_size)

            if hysteresis_enabled and is_rebalance_day and len(top_indices) == proto.pos_size:
                # Load per-combo state
                combo_state = prev_strategies.get(combo, {})
                signal_portfolio = set(combo_state.get("signal_portfolio", []))
                signal_hold_days = combo_state.get("signal_hold_days", {})

                # Build hmask/hdays arrays aligned to tickers
                hmask = np.array(
                    [t in signal_portfolio for t in tickers], dtype=np.bool_
                )
                hdays = np.array(
                    [signal_hold_days.get(t, 0) for t in tickers], dtype=np.int64
                )

                top_arr = np.array(top_indices, dtype=np.int64)
                target_mask = apply_hysteresis(
                    scores, hmask, hdays, top_arr,
                    proto.pos_size, proto.delta_rank, proto.min_hold_days,
                )
                picks = [i for i in range(len(tickers)) if target_mask[i]]

                # Track which picks were "kept by hysteresis" vs "newly selected"
                top_set = set(tickers[i] for i in top_indices)
                for idx in picks:
                    t = tickers[idx]
                    if t in signal_portfolio and t not in top_set:
                        hyst_kept.append(t)

                # Update state for this combo
                new_portfolio = [tickers[i] for i in picks]
                new_hold_days = {}
                new_set = set(new_portfolio)
                for t in new_set:
                    if t in signal_portfolio:
                        new_hold_days[t] = signal_hold_days.get(t, 0)  # already incremented
                    else:
                        new_hold_days[t] = 0  # T+1 Open: init to 0, next run increments
                new_strategies[combo] = {
                    "signal_portfolio": new_portfolio,
                    "signal_hold_days": new_hold_days,
                }
            else:
                # Stateless top-k (hysteresis disabled, non-rebalance day, or first run)
                picks = top_indices

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
                "hyst_kept": ",".join(hyst_kept) if hyst_kept else "",
            }
        )

    df_per = pd.DataFrame(per_strategy_rows)

    # 6a-post) Save signal state on rebalance days (hysteresis persistence)
    if hysteresis_enabled and is_rebalance_day:
        # For combos not processed by hysteresis (first run / cold start), backfill from picks
        for _, row in df_per.iterrows():
            combo = str(row["combo"]).strip()
            if combo not in new_strategies:
                picks_list = [row["pick1"], row["pick2"]]
                picks_list = [p for p in picks_list if p]
                new_strategies[combo] = {
                    "signal_portfolio": picks_list,
                    "signal_hold_days": {p: 0 for p in picks_list},
                }

        state_out = {
            "version": CURRENT_VERSION,
            "freq": proto.freq,
            "universe_mode": universe_mode,
            "last_asof_date": str(asof_ts.date()),
            "last_trade_date": args.trade_date,
            "strategies": new_strategies,
        }
        _save_signal_state(state_path, state_out)

    # ──────────────────────────────────────────────────────────────────
    # Shadow strategies: parallel signal generation (no production impact)
    # ──────────────────────────────────────────────────────────────────
    shadow_results: list[dict] = []
    shadow_config_path = ROOT / args.shadow_config

    if shadow_config_path.exists():
        with open(shadow_config_path, "r", encoding="utf-8") as f:
            shadow_cfg = yaml.safe_load(f) or {}

        for shadow_strat in shadow_cfg.get("shadow_strategies", []):
            s_name = shadow_strat["name"]
            s_combo = shadow_strat["combo"]
            s_factors = _iter_combo_factors(s_combo)

            # Verify factors exist
            missing = [f for f in s_factors if f not in std_factors]
            if missing:
                print(f"  Shadow {s_name}: missing factors {missing}, skipping")
                continue

            # Separate state file per shadow strategy
            s_state_path = STATE_DIR / f"signal_state_shadow_{s_name}.json"
            s_prev_state = _load_signal_state(s_state_path) if hysteresis_enabled else None
            s_combo_state: dict = {}
            s_elapsed = 0

            if s_prev_state:
                tradable_set = set(tickers)
                problems = _validate_signal_state(
                    s_prev_state, proto.freq, universe_mode, tradable_set
                )
                if problems:
                    for p in problems:
                        logger.warning(f"Shadow {s_name} state validation failed: {p}")
                    s_prev_state = None

            if s_prev_state and s_prev_state.get("last_asof_date"):
                last_ts = pd.Timestamp(s_prev_state["last_asof_date"])
                mask = (close_df.index > last_ts) & (close_df.index <= asof_ts)
                s_elapsed = int(mask.sum())
                for st in s_prev_state.get("strategies", {}).values():
                    for ticker in st.get("signal_portfolio", []):
                        st["signal_hold_days"][ticker] = (
                            st["signal_hold_days"].get(ticker, 0) + s_elapsed
                        )
                s_combo_state = s_prev_state.get("strategies", {}).get(s_combo, {})

            # Score (same logic as production)
            s_scores = np.full(len(tickers), -np.inf, dtype=float)
            s_valid = 0
            for i, ticker in enumerate(tickers):
                s = 0.0
                has_value = False
                for f_name in s_factors:
                    v = std_factors[f_name].at[asof_ts, ticker]
                    if pd.notna(v):
                        s += float(v)
                        has_value = True
                if has_value and s != 0.0:
                    s_scores[i] = s
                    s_valid += 1

            # Tradable mask (same as production)
            s_tradable = np.ones(len(tickers), dtype=bool)
            if len(close_df) >= 2:
                prev_close = close_df.iloc[-2]
                curr_close = close_df.iloc[-1]
                volume_curr = ohlcv["volume"].iloc[-1]
                for i, ticker in enumerate(tickers):
                    if pd.notna(prev_close[ticker]) and pd.notna(curr_close[ticker]):
                        if (curr_close[ticker] / prev_close[ticker]) - 1.0 > 0.095:
                            s_tradable[i] = False
                    if pd.notna(volume_curr[ticker]) and volume_curr[ticker] <= 0:
                        s_tradable[i] = False
                s_scores[~s_tradable] = -np.inf

            # QDII block
            if universe_mode == "A_SHARE_ONLY":
                for i, ticker in enumerate(tickers):
                    if ticker in qdii_set:
                        s_scores[i] = -np.inf

            # Selection with hysteresis
            s_picks: list[int] = []
            s_hyst_kept: list[str] = []

            if s_valid >= proto.pos_size:
                if pool_ids is not None and pool_extended_k > 0:
                    s_top = _pool_diversify_topk(s_scores, pool_ids, proto.pos_size, pool_extended_k)
                else:
                    s_top = _stable_topk_indices(s_scores, proto.pos_size)

                if hysteresis_enabled and is_rebalance_day and len(s_top) == proto.pos_size:
                    sig_pf = set(s_combo_state.get("signal_portfolio", []))
                    sig_hd = s_combo_state.get("signal_hold_days", {})

                    hmask = np.array([t in sig_pf for t in tickers], dtype=np.bool_)
                    hdays = np.array([sig_hd.get(t, 0) for t in tickers], dtype=np.int64)

                    target_mask = apply_hysteresis(
                        s_scores, hmask, hdays, np.array(s_top, dtype=np.int64),
                        proto.pos_size, proto.delta_rank, proto.min_hold_days,
                    )
                    s_picks = [i for i in range(len(tickers)) if target_mask[i]]

                    top_set = set(tickers[i] for i in s_top)
                    for idx in s_picks:
                        t = tickers[idx]
                        if t in sig_pf and t not in top_set:
                            s_hyst_kept.append(t)
                else:
                    s_picks = s_top

            s_picked_tickers = [tickers[i] for i in s_picks]
            s_picked_scores = [float(s_scores[i]) for i in s_picks]

            # Build new state
            s_new_hold_days: dict[str, int] = {}
            prev_pf = set(s_combo_state.get("signal_portfolio", []))
            prev_hd = s_combo_state.get("signal_hold_days", {})
            for t in s_picked_tickers:
                s_new_hold_days[t] = prev_hd.get(t, 0) if t in prev_pf else 0

            # Save state on rebalance days
            if hysteresis_enabled and is_rebalance_day:
                s_state_out = {
                    "version": CURRENT_VERSION,
                    "freq": proto.freq,
                    "universe_mode": universe_mode,
                    "last_asof_date": str(asof_ts.date()),
                    "last_trade_date": args.trade_date,
                    "strategies": {
                        s_combo: {
                            "signal_portfolio": s_picked_tickers,
                            "signal_hold_days": s_new_hold_days,
                        }
                    },
                }
                _save_signal_state(s_state_path, s_state_out)

            # Append to snapshot log (append-only JSONL for historical tracking)
            snapshot_path = STATE_DIR / "shadow_snapshots.jsonl"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot = {
                "asof_date": str(asof_ts.date()),
                "trade_date": args.trade_date,
                "strategy": s_name,
                "combo": s_combo,
                "is_rebalance": is_rebalance_day,
                "picks": s_picked_tickers,
                "scores": s_picked_scores,
                "timing_ratio": timing_ratio,
                "regime_ratio": regime_ratio,
                "hyst_kept": s_hyst_kept,
                "hold_days": s_new_hold_days,
            }
            with open(snapshot_path, "a", encoding="utf-8") as sf:
                sf.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

            shadow_results.append(snapshot)
            print(f"  Shadow {s_name}: {s_picked_tickers} (scores: {[f'{x:.3f}' for x in s_picked_scores]})")

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
    if hysteresis_enabled:
        prev_date = prev_state.get("last_asof_date", "N/A") if prev_state else "N/A"
        md_lines.append(
            f"- 迟滞参数：delta_rank={proto.delta_rank:.2f}, min_hold_days={proto.min_hold_days}"
        )
        md_lines.append(
            f"- 状态文件：{state_path.relative_to(ROOT)} (上次: {prev_date}, elapsed_td={elapsed_days})"
        )
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
    display_cols = ["combo", "pick1", "pick2", "score1", "score2"]
    if hysteresis_enabled and "hyst_kept" in df_per.columns:
        display_cols.append("hyst_kept")
    md_lines.append(df_per[display_cols].to_markdown(index=False))

    if hysteresis_enabled and not is_rebalance_day and prev_state:
        md_lines.append("\n> 非调仓日：持仓不变，上次调仓状态保持。")

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

    # Shadow strategies markdown section
    if shadow_results:
        md_lines.append(f"\n## Shadow 策略信号（不交易，仅记录）")
        for sr in shadow_results:
            md_lines.append(f"\n### {sr['strategy']} ({sr['combo']})")
            if sr["picks"]:
                for i, (p, s) in enumerate(zip(sr["picks"], sr["scores"])):
                    md_lines.append(f"- Pick {i + 1}: **{p}** (score: {s:.4f})")
            else:
                md_lines.append("- (无有效信号)")
            md_lines.append(f"- 调仓日: {'是' if sr['is_rebalance'] else '否'}")
            md_lines.append(
                f"- 迟滞保留: {', '.join(sr['hyst_kept']) if sr['hyst_kept'] else '无'}"
            )
            if sr["hold_days"]:
                hd_str = ", ".join(f"{k}: {v}d" for k, v in sr["hold_days"].items())
                md_lines.append(f"- Hold days: {hd_str}")

        # Save shadow signals CSV
        shadow_csv_path = outdir / "shadow_signals.csv"
        pd.DataFrame(shadow_results).to_csv(shadow_csv_path, index=False, encoding="utf-8")

    (outdir / "TODAY_SIGNAL.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    print(f"✅ wrote: {outdir}")
    print(f"- {outdir / 'TODAY_SIGNAL.md'}")
    print(f"- {outdir / 'signals_per_strategy.csv'}")
    print(f"- {outdir / 'aggregate_weights.csv'}")
    if shadow_results:
        print(f"- {outdir / 'shadow_signals.csv'}")
        print(f"- {STATE_DIR / 'shadow_snapshots.jsonl'} (appended)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
