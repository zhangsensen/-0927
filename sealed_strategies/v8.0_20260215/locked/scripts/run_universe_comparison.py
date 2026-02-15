#!/usr/bin/env python3
"""GLOBAL vs A_SHARE_ONLY 同口径对比脚本。

一次加载数据，对 WFO Top-N 候选分别在两种 universe mode 下跑 VEC backtest，
输出 6 项关键指标对照表。

用法：
    uv run python scripts/run_universe_comparison.py --top-n 200
    uv run python scripts/run_universe_comparison.py --top-n 500 --cost-tier med
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.cost_model import build_cost_array, load_cost_model
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.frozen_params import FrozenETFPool, get_qdii_tickers
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr

from batch_vec_backtest import run_vec_backtest

warnings.filterwarnings("ignore")


def _load_latest_wfo_combos() -> Path:
    wfo_dirs = sorted(
        d for d in (ROOT / "results").glob("run_*") if d.is_dir()
    )
    if not wfo_dirs:
        raise FileNotFoundError("未找到 WFO 结果目录 run_*")
    latest = wfo_dirs[-1]
    for name in ("top_combos.csv", "top100_by_ic.parquet", "all_combos.parquet"):
        p = latest / name
        if p.exists():
            return p
    raise FileNotFoundError(f"在 {latest} 未找到 combo 文件")


def _parse_args():
    p = argparse.ArgumentParser(description="GLOBAL vs A_SHARE_ONLY 对比")
    p.add_argument("--top-n", type=int, default=200, help="Top N combos to compare")
    p.add_argument("--cost-tier", type=str, default=None, help="Override cost tier (low/med/high)")
    p.add_argument("--combos", type=str, default=None, help="指定 combo 文件")
    return p.parse_args()


def _load_data(config: dict, end_date: str):
    """加载 OHLCV 和计算因子（只做一次）"""
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=end_date,
    )

    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    raw_factors = {name: raw_factors_df[name] for name in factor_names}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    first_df = std_factors[factor_names[0]]
    dates = first_df.index
    etf_codes = first_df.columns.tolist()

    all_factors_stack = np.stack(
        [std_factors[f].values for f in factor_names], axis=-1
    )

    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    return {
        "ohlcv": ohlcv,
        "dates": dates,
        "etf_codes": etf_codes,
        "factor_names": factor_names,
        "all_factors_stack": all_factors_stack,
        "close_prices": close_prices,
        "open_prices": open_prices,
        "high_prices": high_prices,
        "low_prices": low_prices,
    }


def _compute_timing(ohlcv, dates, backtest_config):
    timing_cfg = backtest_config.get("timing", {})
    timing_module = LightTimingModule(
        extreme_threshold=float(timing_cfg.get("extreme_threshold", -0.1)),
        extreme_position=float(timing_cfg.get("extreme_position", 0.1)),
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_raw)

    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64))
    return timing_arr


def _run_vec_for_combos(
    combos: list[str],
    data: dict,
    timing_arr: np.ndarray,
    cost_arr: np.ndarray,
    backtest_config: dict,
    use_t1_open: bool,
    qdii_mask: np.ndarray | None = None,
    label: str = "",
) -> pd.DataFrame:
    """对一组 combo 跑 VEC，返回结果 DataFrame。

    qdii_mask: shape (N,) bool — True 表示该 ETF 是 QDII，需要 mask。
    """
    factor_index_map = {name: idx for idx, name in enumerate(data["factor_names"])}
    factors_stack = data["all_factors_stack"]

    # 如果有 QDII mask，复制一份并将 QDII 因子值设为 NaN
    if qdii_mask is not None and qdii_mask.any():
        factors_stack = factors_stack.copy()
        for i in range(factors_stack.shape[1]):
            if qdii_mask[i]:
                factors_stack[:, i, :] = np.nan

    results = []
    for combo_str in combos:
        factors_in = [f.strip() for f in combo_str.split("+")]
        try:
            combo_indices = [factor_index_map[f] for f in factors_in]
        except KeyError:
            continue

        current_factors = factors_stack[..., combo_indices]
        current_factor_indices = list(range(len(combo_indices)))

        try:
            _, ret, wr, pf, trades, _, risk = run_vec_backtest(
                current_factors,
                data["close_prices"],
                data["open_prices"],
                data["high_prices"],
                data["low_prices"],
                timing_arr,
                current_factor_indices,
                freq=3,
                pos_size=2,
                initial_capital=float(backtest_config.get("initial_capital", 1_000_000)),
                commission_rate=float(backtest_config.get("commission_rate", 0.0002)),
                lookback=int(backtest_config.get("lookback", 252)),
                cost_arr=cost_arr,
                trailing_stop_pct=0.0,
                stop_on_rebalance_only=True,
                use_t1_open=use_t1_open,
            )
            results.append({
                "combo": combo_str,
                "size": len(combo_indices),
                "vec_return": ret,
                "vec_max_drawdown": risk["max_drawdown"],
                "vec_sharpe_ratio": risk["sharpe_ratio"],
                "vec_calmar_ratio": risk.get("calmar_ratio", 0.0),
                "vec_trades": trades,
                "vec_turnover_ann": risk.get("turnover_ann", 0.0),
                "vec_cost_drag": risk.get("cost_drag", 0.0),
                "vec_aligned_return": risk.get("aligned_return", ret),
                "win_rate": wr,
            })
        except Exception as e:
            print(f"  [WARN] {combo_str}: {e}")
            continue

    df = pd.DataFrame(results)
    if label:
        print(f"  {label}: {len(df)} combos completed")
    return df


def _holdout_split(
    combos: list[str],
    config: dict,
    data_full: dict,
    timing_full: np.ndarray,
    cost_arr: np.ndarray,
    backtest_config: dict,
    use_t1_open: bool,
    qdii_mask: np.ndarray | None = None,
    label: str = "",
) -> pd.DataFrame:
    """跑完整期 VEC，然后用训练/holdout 日期切片算分段收益。"""
    # 完整期已在 data_full 中，直接跑
    df = _run_vec_for_combos(
        combos, data_full, timing_full, cost_arr, backtest_config, use_t1_open,
        qdii_mask=qdii_mask, label=label,
    )
    return df


def _compute_qdii_monitoring(data: dict, combos: list[str], qdii_indices: list[int]):
    """统计 QDII 在全池排名中进入 Top2/Top10 的频次。"""
    factor_index_map = {name: idx for idx, name in enumerate(data["factor_names"])}
    factors_stack = data["all_factors_stack"]  # 未 mask 的原始数据
    N = factors_stack.shape[1]
    T = factors_stack.shape[0]

    top2_count = 0
    top10_count = 0
    total_rebalances = 0

    # 用第一个 combo 做监控
    if not combos:
        return {"top2_pct": 0, "top10_pct": 0}
    combo_str = combos[0]
    factors_in = [f.strip() for f in combo_str.split("+")]
    try:
        combo_indices = [factor_index_map[f] for f in factors_in]
    except KeyError:
        return {"top2_pct": 0, "top10_pct": 0}

    lookback = 252
    freq = 3
    for t in range(lookback, T, freq):
        scores = np.zeros(N)
        for i in range(N):
            s = 0.0
            valid = True
            for fi in combo_indices:
                v = factors_stack[t - 1, i, fi]
                if np.isnan(v):
                    valid = False
                    break
                s += v
            scores[i] = s if valid else -np.inf

        order = np.argsort(-scores)
        total_rebalances += 1
        for qi in qdii_indices:
            rank = np.where(order == qi)[0]
            if len(rank) > 0:
                r = rank[0] + 1
                if r <= 2:
                    top2_count += 1
                if r <= 10:
                    top10_count += 1

    n_qdii = len(qdii_indices)
    denom = max(total_rebalances * n_qdii, 1)
    return {
        "top2_pct": top2_count / denom * 100,
        "top10_pct": top10_count / denom * 100,
        "total_rebalances": total_rebalances,
    }


def main():
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("📊 GLOBAL vs A_SHARE_ONLY 同口径对比")
    print("=" * 80)

    # 1. Load config
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest_config = config.get("backtest", {})
    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open

    # Override cost tier if specified
    if args.cost_tier:
        config["backtest"]["cost_model"]["tier"] = args.cost_tier

    cost_model = load_cost_model(config)
    tier = cost_model.active_tier
    print(f"  Execution: {exec_model.mode}")
    print(f"  Cost: tier={cost_model.tier}, A股={tier.a_share*10000:.0f}bp, QDII={tier.qdii*10000:.0f}bp")

    # 2. Load combos
    if args.combos:
        combos_path = Path(args.combos)
    else:
        combos_path = _load_latest_wfo_combos()

    if combos_path.suffix == ".csv":
        combos_df = pd.read_csv(combos_path)
    else:
        combos_df = pd.read_parquet(combos_path)

    combo_list = combos_df["combo"].tolist()[:args.top_n]

    # 确保封板策略 S1/S2 始终在列表中（可能不在 WFO active_factors 搜索空间内）
    sealed_combos = [
        "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
        "ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D",
    ]
    for sc in sealed_combos:
        if sc not in combo_list:
            combo_list.append(sc)

    print(f"  Combos: {len(combo_list)} (from {combos_path.name} + sealed S1/S2)")

    # 3. Load data (full period: train + holdout)
    print("\n📦 Loading data (full period)...")
    data_full = _load_data(config, end_date=config["data"]["end_date"])
    etf_codes = data_full["etf_codes"]
    N = len(etf_codes)

    # Build masks
    qdii_tickers = get_qdii_tickers(config)
    qdii_mask = np.array([code in qdii_tickers for code in etf_codes], dtype=bool)
    qdii_indices = [i for i, m in enumerate(qdii_mask) if m]
    print(f"  ETFs: {N} total, {qdii_mask.sum()} QDII")

    # Build cost array
    qdii_set = set(FrozenETFPool().qdii_codes)
    cost_arr = build_cost_array(cost_model, etf_codes, qdii_set)

    # Timing
    timing_full = _compute_timing(data_full["ohlcv"], data_full["dates"], backtest_config)

    # 4. Run VEC: GLOBAL
    print("\n🌍 Running GLOBAL (49 ETFs, QDII enabled)...")
    df_global = _run_vec_for_combos(
        combo_list, data_full, timing_full, cost_arr, backtest_config, USE_T1_OPEN,
        qdii_mask=None, label="GLOBAL",
    )

    # 5. Run VEC: A_SHARE_ONLY
    print("\n🇨🇳 Running A_SHARE_ONLY (QDII masked)...")
    df_ashare = _run_vec_for_combos(
        combo_list, data_full, timing_full, cost_arr, backtest_config, USE_T1_OPEN,
        qdii_mask=qdii_mask, label="A_SHARE_ONLY",
    )

    # 6. QDII monitoring (on GLOBAL data)
    print("\n🔍 Computing QDII monitoring...")
    qdii_mon = _compute_qdii_monitoring(data_full, combo_list, qdii_indices)

    # 7. Compute comparison metrics
    print("\n" + "=" * 80)
    print("📊 COMPARISON RESULTS")
    print("=" * 80)

    def _stats(df: pd.DataFrame, label: str) -> dict:
        if df.empty:
            return {"label": label}
        return {
            "label": label,
            "n_combos": len(df),
            "n_positive_return": int((df["vec_return"] > 0).sum()),
            "positive_rate": f"{(df['vec_return'] > 0).mean()*100:.1f}%",
            "median_return": f"{df['vec_return'].median()*100:.2f}%",
            "best_return": f"{df['vec_return'].max()*100:.2f}%",
            "worst_return": f"{df['vec_return'].min()*100:.2f}%",
            "median_sharpe": f"{df['vec_sharpe_ratio'].median():.3f}",
            "median_mdd": f"{df['vec_max_drawdown'].median()*100:.2f}%",
            "worst_mdd": f"{df['vec_max_drawdown'].max()*100:.2f}%",
            "median_turnover": f"{df['vec_turnover_ann'].median():.1f}x",
            "median_cost_drag": f"{df['vec_cost_drag'].median()*100:.1f}%",
            "median_trades": f"{df['vec_trades'].median():.0f}",
        }

    g = _stats(df_global, "GLOBAL")
    a = _stats(df_ashare, "A_SHARE_ONLY")

    # Print comparison table
    metrics = [
        "n_combos", "n_positive_return", "positive_rate",
        "median_return", "best_return", "worst_return",
        "median_sharpe", "median_mdd", "worst_mdd",
        "median_turnover", "median_cost_drag", "median_trades",
    ]

    print(f"\n{'Metric':<22} {'GLOBAL':>16} {'A_SHARE_ONLY':>16} {'Delta':>12}")
    print("-" * 70)
    for m in metrics:
        gv = g.get(m, "N/A")
        av = a.get(m, "N/A")
        # Try to compute delta for numeric values
        delta = ""
        try:
            gf = float(str(gv).replace("%", "").replace("x", ""))
            af = float(str(av).replace("%", "").replace("x", ""))
            d = af - gf
            delta = f"{d:+.2f}"
        except (ValueError, TypeError):
            pass
        print(f"  {m:<20} {str(gv):>16} {str(av):>16} {delta:>12}")

    # QDII monitoring
    print(f"\n📡 QDII 监控 (GLOBAL mode, S1 combo):")
    print(f"  Rebalance days: {qdii_mon.get('total_rebalances', 0)}")
    print(f"  Any QDII in Top2: {qdii_mon.get('top2_pct', 0):.1f}% of (rebalance × QDII count)")
    print(f"  Any QDII in Top10: {qdii_mon.get('top10_pct', 0):.1f}%")

    # S1/S2 sealed strategies comparison
    print("\n" + "=" * 80)
    print("🎯 S1/S2 封板策略对比")
    print("=" * 80)
    sealed = [
        "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
        "ADX_14D + OBV_SLOPE_10D + PRICE_POSITION_120D + SHARPE_RATIO_20D + SLOPE_20D",
    ]
    for i, combo in enumerate(sealed, 1):
        g_row = df_global[df_global["combo"] == combo]
        a_row = df_ashare[df_ashare["combo"] == combo]
        name = f"S{i}"
        if g_row.empty and a_row.empty:
            print(f"  {name}: not in top-{args.top_n} combos (try --top-n 5000)")
            continue
        print(f"\n  {name}: {combo}")
        for metric in ["vec_return", "vec_max_drawdown", "vec_sharpe_ratio", "vec_turnover_ann", "vec_cost_drag"]:
            gv = f"{g_row[metric].iloc[0]*100:.2f}%" if not g_row.empty else "N/A"
            av = f"{a_row[metric].iloc[0]*100:.2f}%" if not a_row.empty else "N/A"
            if metric in ("vec_turnover_ann",):
                gv = f"{g_row[metric].iloc[0]:.1f}x" if not g_row.empty else "N/A"
                av = f"{a_row[metric].iloc[0]:.1f}x" if not a_row.empty else "N/A"
            print(f"    {metric:<22} GLOBAL={gv:>10}  A_SHARE={av:>10}")

    # 8. Save results
    out_dir = ROOT / "results" / f"universe_comparison_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_global.to_parquet(out_dir / "global_results.parquet", index=False)
    df_ashare.to_parquet(out_dir / "ashare_only_results.parquet", index=False)

    # Save comparison summary
    summary = {
        "timestamp": timestamp,
        "top_n": args.top_n,
        "cost_tier": cost_model.tier,
        "execution": exec_model.mode,
        "global": g,
        "ashare_only": a,
        "qdii_monitoring": qdii_mon,
    }
    pd.Series(summary).to_json(out_dir / "comparison_summary.json")

    print(f"\n✅ Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
