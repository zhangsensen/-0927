#!/usr/bin/env python3
"""
Exp1 对比脚本: COC vs T1_OPEN 执行模型

对同一组 WFO Top 组合，分别用 COC 和 T1_OPEN 模式跑 VEC 回测，
输出对比表 (收益、MDD、Sharpe、换手率) 和 trade audit。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import load_frozen_config
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr

# Import VEC backtest
from batch_vec_backtest import run_vec_backtest


def main():
    print("=" * 80)
    print("Exp1: COC vs T1_OPEN 执行模型对比")
    print("=" * 80)

    # 1. 加载配置
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    frozen = load_frozen_config(config, config_path=str(config_path))
    backtest_config = config.get("backtest", {})
    FREQ = backtest_config["freq"]
    POS_SIZE = backtest_config["pos_size"]
    LOOKBACK = backtest_config["lookback"]
    INITIAL_CAPITAL = float(backtest_config["initial_capital"])
    COMMISSION_RATE = float(backtest_config["commission_rate"])

    # 2. 加载数据
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # 3. 计算因子
    factor_cache = FactorCache(cache_dir=Path(config["data"].get("cache_dir") or ".cache"))
    cached = factor_cache.get_or_compute(ohlcv=ohlcv, config=config, data_dir=loader.data_dir)
    factor_names = cached["factor_names"]
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    factors_3d = cached["factors_3d"]
    T = len(dates)
    N = len(etf_codes)

    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    # 4. 择时信号
    timing_config = backtest_config.get("timing", {})
    extreme_threshold = timing_config.get("extreme_threshold", -0.4)
    extreme_position = timing_config.get("extreme_position", 0.3)

    timing_module = LightTimingModule(
        extreme_threshold=extreme_threshold, extreme_position=extreme_position
    )
    timing_series_raw = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr = shift_timing_signal(timing_series_raw.reindex(dates).fillna(1.0).values)

    # Regime gate
    if backtest_config.get("regime_gate", {}).get("enabled", False):
        gate_arr = compute_regime_gate_arr(
            close_df=ohlcv["close"],
            dates=dates,
            backtest_config=backtest_config,
        )
        timing_arr = timing_arr * gate_arr

    # vol_regime (disabled)
    vol_regime_arr = np.ones(T, dtype=np.float64)

    # 5. 加载 WFO 结果 (Top 20)
    wfo_dirs = sorted(
        [d for d in (ROOT / "results").glob("run_*") if d.is_dir() and not d.is_symlink()]
    )
    if not wfo_dirs:
        print("未找到 WFO 结果")
        return
    latest_wfo = wfo_dirs[-1]
    combos_path = latest_wfo / "top_combos.csv"
    if not combos_path.exists():
        combos_path = latest_wfo / "full_combo_results.csv"
    df_combos = pd.read_csv(combos_path)
    print(f"加载 {len(df_combos)} 个 WFO 组合 ({latest_wfo.name})")

    # 取 Top 20
    TOP_K = 20
    df_top = df_combos.head(TOP_K)

    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}

    # 6. 对比回测
    results = []
    for _, row in df_top.iterrows():
        combo_str = row["combo"]
        factor_indices = [factor_index_map[f.strip()] for f in combo_str.split(" + ")]

        metrics = {}
        for mode_name, use_t1 in [("COC", False), ("T1_OPEN", True)]:
            _, ret, wr, pf, trades, rounding, risk = run_vec_backtest(
                factors_3d, close_prices, open_prices, high_prices, low_prices,
                timing_arr, factor_indices,
                freq=FREQ, pos_size=POS_SIZE,
                initial_capital=INITIAL_CAPITAL,
                commission_rate=COMMISSION_RATE,
                lookback=LOOKBACK,
                vol_regime_arr=vol_regime_arr,
                use_t1_open=use_t1,
            )
            metrics[mode_name] = {
                "total_return": ret,
                "max_drawdown": risk["max_drawdown"],
                "sharpe": risk["sharpe_ratio"],
                "calmar": risk["calmar_ratio"],
                "trades": trades,
            }

        results.append({
            "combo": combo_str,
            "coc_return": metrics["COC"]["total_return"],
            "t1_return": metrics["T1_OPEN"]["total_return"],
            "coc_mdd": metrics["COC"]["max_drawdown"],
            "t1_mdd": metrics["T1_OPEN"]["max_drawdown"],
            "coc_sharpe": metrics["COC"]["sharpe"],
            "t1_sharpe": metrics["T1_OPEN"]["sharpe"],
            "coc_calmar": metrics["COC"]["calmar"],
            "t1_calmar": metrics["T1_OPEN"]["calmar"],
            "coc_trades": metrics["COC"]["trades"],
            "t1_trades": metrics["T1_OPEN"]["trades"],
            "return_delta_pp": (metrics["T1_OPEN"]["total_return"] - metrics["COC"]["total_return"]) * 100,
            "mdd_delta_pp": (metrics["T1_OPEN"]["max_drawdown"] - metrics["COC"]["max_drawdown"]) * 100,
            "sharpe_delta": metrics["T1_OPEN"]["sharpe"] - metrics["COC"]["sharpe"],
        })

    df_results = pd.DataFrame(results)

    # 7. 输出结果
    print("\n" + "=" * 80)
    print("COC vs T1_OPEN 对比 (Top 20 WFO 组合)")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'COC Mean':>12} {'T1_OPEN Mean':>14} {'Delta':>10}")
    print("-" * 65)
    print(f"{'Total Return':<25} {df_results['coc_return'].mean()*100:>11.1f}% {df_results['t1_return'].mean()*100:>13.1f}% {df_results['return_delta_pp'].mean():>+9.1f}pp")
    print(f"{'Max Drawdown':<25} {df_results['coc_mdd'].mean()*100:>11.1f}% {df_results['t1_mdd'].mean()*100:>13.1f}% {df_results['mdd_delta_pp'].mean():>+9.1f}pp")
    print(f"{'Sharpe Ratio':<25} {df_results['coc_sharpe'].mean():>12.3f} {df_results['t1_sharpe'].mean():>14.3f} {df_results['sharpe_delta'].mean():>+10.3f}")
    print(f"{'Calmar Ratio':<25} {df_results['coc_calmar'].mean():>12.3f} {df_results['t1_calmar'].mean():>14.3f} {(df_results['t1_calmar'].mean()-df_results['coc_calmar'].mean()):>+10.3f}")
    print(f"{'Trades':<25} {df_results['coc_trades'].mean():>12.0f} {df_results['t1_trades'].mean():>14.0f} {(df_results['t1_trades'].mean()-df_results['coc_trades'].mean()):>+10.0f}")

    # Pct of combos where T1_OPEN is worse
    pct_worse_ret = (df_results['return_delta_pp'] < 0).mean() * 100
    pct_worse_mdd = (df_results['mdd_delta_pp'] > 0).mean() * 100
    print(f"\nT1_OPEN worse return: {pct_worse_ret:.0f}% of combos")
    print(f"T1_OPEN worse MDD: {pct_worse_mdd:.0f}% of combos")

    # 8. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / f"exp1_coc_vs_t1open_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_parquet(out_dir / "comparison_results.parquet")
    df_results.to_csv(out_dir / "comparison_results.csv", index=False)
    print(f"\n结果已保存: {out_dir}")

    # 9. 每组合明细
    print(f"\n{'Combo':<60} {'COC Ret':>8} {'T1 Ret':>8} {'Delta':>8} {'COC Sharpe':>10} {'T1 Sharpe':>10}")
    print("-" * 110)
    for _, r in df_results.iterrows():
        combo_short = r['combo'][:57] + "..." if len(r['combo']) > 60 else r['combo']
        print(f"{combo_short:<60} {r['coc_return']*100:>7.1f}% {r['t1_return']*100:>7.1f}% {r['return_delta_pp']:>+7.1f} {r['coc_sharpe']:>10.3f} {r['t1_sharpe']:>10.3f}")


if __name__ == "__main__":
    main()
