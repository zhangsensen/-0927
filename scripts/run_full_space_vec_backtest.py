import sys
import os
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_strategy.core.utils.run_meta import write_step_meta
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr, gate_stats
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.cost_model import load_cost_model, build_cost_array
from etf_strategy.core.frozen_params import FrozenETFPool

# Import the backtest engine
from batch_vec_backtest import run_vec_backtest

warnings.filterwarnings("ignore")


def _load_latest_wfo_combos() -> Path:
    """Return path to top100_by_ic.parquet from latest WFO run_* (pipeline enforcement)."""
    wfo_dirs = sorted(
        [
            d
            for d in (ROOT / "results").glob("run_*")
            if d.is_dir() and not d.is_symlink()
        ]
    )
    if not wfo_dirs:
        raise FileNotFoundError("未找到 WFO 结果目录 run_*，请先运行 run_combo_wfo.py")
    latest = wfo_dirs[-1]
    # 优先 top100_by_ic.parquet，其次 all_combos.parquet
    candidates = [
        latest / "top100_by_ic.parquet",
        latest / "all_combos.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"在 {latest} 未找到 top100_by_ic.parquet / all_combos.parquet"
    )


def main():
    parser = argparse.ArgumentParser(
        description="VEC backtest using WFO outputs (pipeline enforced)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (yaml)。默认使用环境变量 WFO_CONFIG_PATH 或 configs/combo_wfo_config.yaml",
    )
    parser.add_argument(
        "--combos",
        type=str,
        default=None,
        help="指定 WFO 导出的组合文件（parquet，需含 combo 列），如 run_xxx/all_combos.parquet",
    )
    args = parser.parse_args()
    print("=" * 80)
    print("🚀 VEC BACKTEST (WFO -> VEC pipeline enforced)")
    print("=" * 80)

    # 1. Load Configuration
    config_path = (
        Path(args.config)
        if args.config
        else Path(
            os.environ.get(
                "WFO_CONFIG_PATH", str(ROOT / "configs/combo_wfo_config.yaml")
            )
        )
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest_config = config.get("backtest", {})

    # Execution model
    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open

    # Read parameters from config (no longer hardcoded)
    FREQ = backtest_config.get("freq", 3)
    POS_SIZE = backtest_config.get("pos_size", 2)
    EXTREME_THRESHOLD = -0.1
    EXTREME_POSITION = 0.1

    # Hysteresis parameters from config
    hyst_config = backtest_config.get("hysteresis", {})
    DELTA_RANK = float(hyst_config.get("delta_rank", 0.0))
    MIN_HOLD_DAYS = int(hyst_config.get("min_hold_days", 0))

    print(f"Configuration:")
    print(f"  Execution Model: {exec_model.mode}")
    print(f"  FREQ: {FREQ}")
    print(f"  POS_SIZE: {POS_SIZE}")
    print(f"  Hysteresis: delta_rank={DELTA_RANK}, min_hold_days={MIN_HOLD_DAYS}")
    print(f"  Timing Threshold: {EXTREME_THRESHOLD}")
    print(f"  Timing Position: {EXTREME_POSITION}")

    # 2. Load Data
    print("\nLoading Data...")

    # 使用 training_end_date 如果设置了（Holdout验证模式）
    data_end_date = (
        config["data"].get("training_end_date") or config["data"]["end_date"]
    )

    if config["data"].get("training_end_date"):
        print("=" * 80)
        print("🔬 HOLDOUT验证模式")
        print("=" * 80)
        print(f"训练集截止日期: {data_end_date}")
        print(f"完整数据截止日期: {config['data']['end_date']}")
        print("⚠️  注意: 当前仅使用训练集数据，Holdout期数据将用于最终验证")
        print("")

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=data_end_date,  # 使用训练集截止日期
    )

    # 3. Load Factors (OHLCV + non-OHLCV via FactorCache)
    print("Loading Factors (cached + external)...")
    from etf_strategy.core.factor_cache import FactorCache

    factor_cache = FactorCache(
        cache_dir=Path(config["data"].get("cache_dir", ".cache"))
    )
    data_dir = Path(config["data"].get("data_dir", "raw/ETF/daily"))
    cached_factors = factor_cache.get_or_compute(ohlcv, config, data_dir)
    factor_names_list = cached_factors["factor_names"]
    dates = cached_factors["dates"]
    etf_codes = cached_factors["etf_codes"]
    all_factors_stack = cached_factors["factors_3d"]
    print(f"  Factors: {len(factor_names_list)}, Shape: {all_factors_stack.shape}")

    # ✅ Exp2: 构建 per-ETF 成本数组
    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)
    cost_arr = build_cost_array(cost_model, etf_codes, qdii_set)
    tier = cost_model.active_tier
    print(f"  Cost Model: mode={cost_model.mode}, tier={cost_model.tier}, "
          f"A股={tier.a_share*10000:.0f}bp, QDII={tier.qdii*10000:.0f}bp")

    # fillna(1.0) instead of bfill() to avoid lookahead bias
    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    # 5. Compute Timing Signal
    timing_module = LightTimingModule(
        extreme_threshold=EXTREME_THRESHOLD,
        extreme_position=EXTREME_POSITION,
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)

    # 5b. Regime gate (optional): multiply exposure into timing
    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64)).astype(
        np.float64
    )
    if bool(backtest_config.get("regime_gate", {}).get("enabled", False)):
        s = gate_stats(gate_arr)
        print(
            f"✅ Regime gate enabled: mean={s['mean']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}"
        )

    # 6. Load combos from latest WFO (pipeline enforcement)
    if args.combos:
        combos_path = Path(args.combos)
        if not combos_path.exists():
            raise FileNotFoundError(f"指定的 combos 文件不存在: {combos_path}")
        if combos_path.suffix == ".csv":
            combos_df = pd.read_csv(combos_path)
        else:
            combos_df = pd.read_parquet(combos_path)
        print(f"✅ 使用指定组合文件: {combos_path} ({len(combos_df)} 个组合)")
    else:
        combos_path = _load_latest_wfo_combos()
        combos_df = pd.read_parquet(combos_path)
        print(f"✅ 使用最新 WFO 结果: {len(combos_df)} 个组合 来自 {combos_path}")

    results = []
    factor_index_map = {name: idx for idx, name in enumerate(factor_names_list)}

    for combo_str in tqdm(combos_df["combo"].tolist(), desc="Running Backtests"):
        factors_in_combo = [f.strip() for f in combo_str.split(" + ")]
        try:
            combo_indices = [factor_index_map[f] for f in factors_in_combo]
        except KeyError as e:
            print(f"[WARN] combo {combo_str} 包含未知因子 {e}, 跳过")
            continue

        current_factors = all_factors_stack[..., combo_indices]
        current_factor_indices = list(range(len(combo_indices)))

        try:
            _, ret, wr, pf, trades, _, risk = run_vec_backtest(
                current_factors,
                close_prices,
                open_prices,
                high_prices,
                low_prices,
                timing_arr,
                current_factor_indices,
                freq=FREQ,
                pos_size=POS_SIZE,
                initial_capital=float(backtest_config["initial_capital"]),
                commission_rate=float(backtest_config["commission_rate"]),
                lookback=backtest_config.get("lookback") or backtest_config.get("lookback_window", 252),
                cost_arr=cost_arr,
                delta_rank=DELTA_RANK,
                min_hold_days=MIN_HOLD_DAYS,
                trailing_stop_pct=0.0,
                stop_on_rebalance_only=True,
                use_t1_open=USE_T1_OPEN,
            )

            results.append(
                {
                    "combo": combo_str,
                    "size": len(combo_indices),
                    "vec_return": ret,
                    "vec_max_drawdown": risk["max_drawdown"],
                    "vec_calmar_ratio": risk["calmar_ratio"],
                    "vec_sharpe_ratio": risk["sharpe_ratio"],
                    "vec_aligned_return": risk.get("aligned_return", ret),
                    "vec_aligned_sharpe": risk.get(
                        "aligned_sharpe", risk.get("sharpe_ratio", 0.0)
                    ),
                    "vec_trades": trades,
                    "vec_turnover_ann": risk.get("turnover_ann", 0.0),
                    "vec_cost_drag": risk.get("cost_drag", 0.0),
                }
            )
        except Exception as e:
            print(f"[WARN] combo {combo_str} failed: {e}")
            continue

    # 7. Save and Report
    df = pd.DataFrame(results)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"vec_from_wfo_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "full_space_results.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    write_step_meta(output_dir, step="vec", inputs={"combos": str(args.combos or "auto")}, config=str(args.config or "default"), extras={"combo_count": len(df)})

    # Top 20 Analysis
    df_sorted = df.sort_values("vec_calmar_ratio", ascending=False)

    print("\n" + "=" * 80)
    print("🏆 TOP 20 STRATEGIES (Sorted by Calmar)")
    print("=" * 80)
    print(
        f"{'Rank':<4} | {'Return':<8} | {'MDD':<8} | {'Calmar':<8} | {'Sharpe':<8} | {'Combo'}"
    )
    print("-" * 80)

    for i, (_, row) in enumerate(df_sorted.head(20).iterrows()):
        print(
            f"{i+1:<4} | {row['vec_return']*100:>7.2f}% | {row['vec_max_drawdown']*100:>7.2f}% | {row['vec_calmar_ratio']:>8.3f} | {row['vec_sharpe_ratio']:>8.3f} | {row['combo'][:50]}"
        )

    # Check specific target
    target_combo = "CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D"
    target_row = df[df["combo"] == target_combo]

    print("\n" + "=" * 80)
    print("🔍 TARGET STRATEGY PERFORMANCE")
    print("=" * 80)
    if not target_row.empty:
        row = target_row.iloc[0]
        rank = df_sorted.index.get_loc(target_row.index[0]) + 1
        print(f"Rank: {rank} / {len(df)}")
        print(f"Return: {row['vec_return']*100:.2f}%")
        print(f"MDD:    {row['vec_max_drawdown']*100:.2f}%")
        print(f"Calmar: {row['vec_calmar_ratio']:.3f}")
    else:
        print("Target strategy not found (check factor names matching).")


if __name__ == "__main__":
    main()
