#!/usr/bin/env python3
"""
Holdout Validation Script
对所有 WFO 筛选出的策略在冷数据上进行验证，筛选出双稳定策略。
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, delayed

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.utils.run_meta import write_step_meta
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.cost_model import load_cost_model, build_cost_array
from etf_strategy.core.frozen_params import FrozenETFPool

# Import the backtest engine
from batch_vec_backtest import run_vec_backtest

warnings.filterwarnings("ignore")


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    # Fallback: try parquet first, then csv
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.read_csv(p)


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size <= 1:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    denom = np.where(running_max == 0.0, np.nan, running_max)
    dd = (equity - running_max) / denom
    dd = dd[np.isfinite(dd)]
    return float(abs(dd.min())) if dd.size else 0.0


def _sharpe_from_equity(equity: np.ndarray) -> float:
    if equity.size <= 2:
        return 0.0
    rets = np.diff(equity) / equity[:-1]
    rets = rets[np.isfinite(rets)]
    if rets.size <= 1:
        return 0.0
    std = float(np.std(rets, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(rets) / std * np.sqrt(252.0))


def _window_metrics(
    equity_curve: np.ndarray, start_idx: int, end_idx: int | None = None
) -> dict:
    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.size == 0:
        return {"ret": 0.0, "mdd": 0.0, "sharpe": 0.0, "calmar": 0.0}

    start = max(int(start_idx), 0)
    end = equity.size if end_idx is None else min(int(end_idx), equity.size)
    if end - start <= 1:
        return {"ret": 0.0, "mdd": 0.0, "sharpe": 0.0, "calmar": 0.0}

    window = equity[start:end]
    window = window[np.isfinite(window)]
    if window.size <= 1:
        return {"ret": 0.0, "mdd": 0.0, "sharpe": 0.0, "calmar": 0.0}

    ret = float((window[-1] - window[0]) / window[0]) if window[0] != 0 else 0.0
    mdd = _max_drawdown(window)
    sharpe = _sharpe_from_equity(window)
    calmar = float(ret / mdd) if mdd > 1e-12 else 0.0
    return {"ret": ret, "mdd": mdd, "sharpe": sharpe, "calmar": calmar}


def process_combo(
    combo_str,
    factor_index_map,
    all_factors_stack,
    close_prices,
    open_prices,
    high_prices,
    low_prices,
    timing_arr,
    backtest_config,
    holdout_start_idx,
    FREQ,
    POS_SIZE,
    trailing_windows,
    use_t1_open=False,
    cost_arr=None,
):
    factors_in_combo = [f.strip() for f in combo_str.split(" + ")]
    try:
        combo_indices = [factor_index_map[f] for f in factors_in_combo]
    except KeyError:
        return None

    current_factors = all_factors_stack[..., combo_indices]
    current_factor_indices = list(range(len(combo_indices)))

    try:
        (
            equity_curve,
            total_return,
            win_rate,
            profit_factor,
            num_trades,
            rounding_diag,
            risk_metrics,
        ) = run_vec_backtest(
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
            lookback=int(backtest_config["lookback"]),
            cost_arr=cost_arr,
            trailing_stop_pct=0.0,
            stop_on_rebalance_only=True,
            use_t1_open=use_t1_open,
        )

        # Holdout window metrics
        hold = _window_metrics(equity_curve, start_idx=holdout_start_idx)

        out = {
            "combo": combo_str,
            "size": len(combo_indices),
            "holdout_return": hold["ret"],
            "holdout_max_drawdown": hold["mdd"],
            "holdout_sharpe_ratio": hold["sharpe"],
            "holdout_calmar_ratio": hold["calmar"],
            # keep original engine metrics for debugging/consistency
            "full_total_return": float(total_return),
            "full_num_trades": int(num_trades),
        }

        # Trailing windows ending at full_end (recent performance)
        T = len(equity_curve)
        for w in trailing_windows:
            w = int(w)
            start = max(T - w, 0)
            m = _window_metrics(equity_curve, start_idx=start)
            out[f"trail_{w}d_return"] = m["ret"]
            out[f"trail_{w}d_max_drawdown"] = m["mdd"]
            out[f"trail_{w}d_sharpe"] = m["sharpe"]
            out[f"trail_{w}d_calmar"] = m["calmar"]

        # Also track trailing windows within holdout only (if window overlaps training)
        hold_T = max(T - holdout_start_idx, 0)
        for w in trailing_windows:
            w = int(w)
            if hold_T <= 1:
                out[f"holdout_trail_{w}d_return"] = 0.0
                out[f"holdout_trail_{w}d_max_drawdown"] = 0.0
                out[f"holdout_trail_{w}d_sharpe"] = 0.0
                out[f"holdout_trail_{w}d_calmar"] = 0.0
                continue
            start = max(T - w, holdout_start_idx)
            m = _window_metrics(equity_curve, start_idx=start)
            out[f"holdout_trail_{w}d_return"] = m["ret"]
            out[f"holdout_trail_{w}d_max_drawdown"] = m["mdd"]
            out[f"holdout_trail_{w}d_sharpe"] = m["sharpe"]
            out[f"holdout_trail_{w}d_calmar"] = m["calmar"]

        return out
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Holdout验证 - 冷数据回测")
    parser.add_argument(
        "--training-results",
        type=str,
        required=True,
        help="训练集VEC结果文件 (full_space_results.parquet/csv)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="仅验证训练集结果 Top-N（按 vec_calmar_ratio 降序）；默认 None 表示全量",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="并行 worker 数，默认 -1 使用全部核心（threading backend 推荐）",
    )
    parser.add_argument(
        "--prefer",
        type=str,
        default="threads",
        choices=["threads", "processes"],
        help="joblib backend 偏好：threads（共享大数组，避免序列化）或 processes",
    )
    parser.add_argument(
        "--trailing-windows",
        type=str,
        default="21,42,63,60,120,240",
        help="最近窗口（交易日）列表，用逗号分隔；会输出总体与 holdout 内的 trailing 指标",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (yaml)。默认使用环境变量 WFO_CONFIG_PATH 或 configs/combo_wfo_config.yaml",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("🔬 HOLDOUT VALIDATION - 冷数据验证")
    print("=" * 80)
    print(f"训练集结果: {args.training_results}")

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

    # 确认 Holdout 数据范围
    training_end = config["data"].get("training_end_date", "2025-04-30")
    full_end = config["data"]["end_date"]

    print(f"\n数据划分:")
    print(f"  训练集: {config['data']['start_date']} 至 {training_end}")
    print(f"  Holdout: {training_end} 至 {full_end}")
    print(f"\nBacktest参数:")
    print(f"  FREQ: {FREQ}, POS_SIZE: {POS_SIZE}")

    # 2. Load Holdout Data (需要包含 lookback 窗口)
    print("\nLoading Holdout Data...")
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )

    # ⚠️ 关键修复: 从训练集开始日加载，确保因子计算有足够历史数据
    # 但只在 Holdout 期执行交易
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],  # 从头加载，提供 lookback
        end_date=full_end,
    )

    # 3. Compute Factors on Holdout Data
    print("Computing Factors on Holdout Data...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = sorted(
        raw_factors_df.columns.get_level_values(0).unique().tolist()
    )
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    # 4. Prepare Backtest Data
    first_factor = std_factors[factor_names_list[0]]
    all_dates = first_factor.index
    etf_codes = first_factor.columns.tolist()

    print(f"完整数据范围: {all_dates[0]} 至 {all_dates[-1]} ({len(all_dates)} 天)")
    print(f"Holdout 期: {training_end} 至 {full_end}")

    # ⚠️ 使用完整数据计算因子，但回测引擎会自动跳过 lookback 期
    # 这样既有足够历史数据，又不污染 Holdout 验证
    all_factors_stack = np.stack(
        [std_factors[f].values for f in factor_names_list], axis=-1
    )

    # ✅ Exp2: 构建 per-ETF 成本数组
    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)
    cost_arr = build_cost_array(cost_model, etf_codes, qdii_set)

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
    timing_arr_raw = timing_series.reindex(all_dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)

    # Apply Regime Gate if enabled
    regime_gate_cfg = backtest_config.get("regime_gate", {})
    if regime_gate_cfg.get("enabled", False):
        print(
            f"🛡️ Regime Gate ENABLED (mode={regime_gate_cfg.get('mode', 'volatility')})"
        )
        gate_arr = compute_regime_gate_arr(
            close_df=ohlcv["close"], dates=all_dates, backtest_config=backtest_config
        )
        # Apply gate to timing signal
        timing_arr = timing_arr * gate_arr
    else:
        print("🛡️ Regime Gate DISABLED")

    # 6. 计算 Holdout 期的起始索引（用于后续只分析 Holdout 期的收益）
    # ✅ 防泄漏：Holdout 从训练截止日之后的第一个交易日开始（strictly > training_end）
    training_end_ts = pd.Timestamp(training_end)
    holdout_start_candidates = np.where(all_dates > training_end_ts)[0]
    if holdout_start_candidates.size == 0:
        raise ValueError(
            f"Holdout 起始点不存在：training_end_date={training_end} 已到数据末尾"
        )
    holdout_start_idx = int(holdout_start_candidates[0])
    holdout_start_actual = all_dates[holdout_start_idx]
    print(
        f"Holdout 起始索引: {holdout_start_idx} / {len(all_dates)} (起始交易日: {holdout_start_actual})"
    )

    # 确保 lookback 后还有足够数据
    effective_start_idx = backtest_config["lookback"]
    print(
        f"有效交易起始索引: {effective_start_idx} (lookback={backtest_config['lookback']})"
    )

    if holdout_start_idx < effective_start_idx:
        print(
            f"⚠️  警告: Holdout 起始点 ({holdout_start_idx}) 早于 lookback 窗口 ({effective_start_idx})"
        )
        print(f"   将从第 {effective_start_idx} 天开始交易")

    # 6. Load Training Results
    training_df = _read_table(args.training_results)
    if args.top_n is not None:
        if "vec_calmar_ratio" not in training_df.columns:
            raise ValueError(
                "training-results 缺少 vec_calmar_ratio 列，无法按 Top-N 过滤"
            )
        training_df = (
            training_df.sort_values("vec_calmar_ratio", ascending=False)
            .head(int(args.top_n))
            .copy()
        )
    print(f"\n✅ 加载训练集结果: {len(training_df)} 个组合")

    trailing_windows = [
        int(x) for x in str(args.trailing_windows).split(",") if str(x).strip()
    ]
    trailing_windows = [w for w in trailing_windows if w > 1]
    trailing_windows = sorted(set(trailing_windows))
    print(f"Trailing windows (trading days): {trailing_windows}")

    # 7. Run Holdout Backtest (parallel)
    factor_index_map = {name: idx for idx, name in enumerate(factor_names_list)}
    combos = training_df["combo"].tolist()
    print(f"Starting parallel execution on {len(combos)} combos...")

    parallel_results = Parallel(
        n_jobs=int(args.n_jobs), prefer=args.prefer, verbose=5, batch_size=64
    )(
        delayed(process_combo)(
            combo_str,
            factor_index_map,
            all_factors_stack,
            close_prices,
            open_prices,
            high_prices,
            low_prices,
            timing_arr,
            backtest_config,
            holdout_start_idx,
            FREQ,
            POS_SIZE,
            trailing_windows,
            use_t1_open=USE_T1_OPEN,
            cost_arr=cost_arr,
        )
        for combo_str in combos
    )

    results = [r for r in parallel_results if r is not None]

    # 8. Merge with Training Results
    holdout_df = pd.DataFrame(results)

    # Merge on combo
    merged_df = training_df.merge(
        holdout_df, on="combo", how="inner", suffixes=("_train", "_holdout")
    )

    # 9. Calculate Stability Metrics
    # 双稳定性定义: 训练集和Holdout都表现良好
    merged_df["calmar_ratio_avg"] = (
        merged_df["vec_calmar_ratio"] + merged_df["holdout_calmar_ratio"]
    ) / 2
    merged_df["calmar_ratio_stability"] = merged_df[
        ["vec_calmar_ratio", "holdout_calmar_ratio"]
    ].min(axis=1)
    merged_df["return_avg"] = (
        merged_df["vec_return"] + merged_df["holdout_return"]
    ) / 2

    # 计算训练集和Holdout的排名
    merged_df["train_calmar_rank"] = merged_df["vec_calmar_ratio"].rank(ascending=False)
    merged_df["holdout_calmar_rank"] = merged_df["holdout_calmar_ratio"].rank(
        ascending=False
    )
    merged_df["rank_stability"] = (
        merged_df["train_calmar_rank"] - merged_df["holdout_calmar_rank"]
    ).abs()

    # 10. Save Results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "results" / f"holdout_validation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "holdout_validation_results.parquet"
    merged_df.to_parquet(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")

    write_step_meta(output_dir, step="holdout", inputs={"training_results": str(args.training_results)}, config=str(args.config or "default"), extras={"combo_count": len(merged_df)})

    # 11. Analysis Report
    print("\n" + "=" * 80)
    print("📊 HOLDOUT VALIDATION ANALYSIS")
    print("=" * 80)

    # Top 20 by Stability (minimum Calmar between train and holdout)
    df_stable = merged_df.sort_values("calmar_ratio_stability", ascending=False)

    print(
        "\n🏆 TOP 20 STABLE STRATEGIES (双稳定排序: min(Train_Calmar, Holdout_Calmar))"
    )
    print("=" * 80)
    print(
        f"{'Rank':<4} | {'Size':<4} | {'Train_Calmar':<12} | {'Hold_Calmar':<12} | {'Stable':<8} | {'Combo'}"
    )
    print("-" * 80)

    for i, (_, row) in enumerate(df_stable.head(20).iterrows()):
        combo_size = len(row["combo"].split(" + "))
        print(
            f"{i+1:<4} | {combo_size:<4} | {row['vec_calmar_ratio']:>11.3f} | "
            f"{row['holdout_calmar_ratio']:>11.3f} | {row['calmar_ratio_stability']:>7.3f} | "
            f"{row['combo'][:60]}"
        )

    # Check the previously Top 1 strategy
    print("\n" + "=" * 80)
    print("🔍 检查训练集 Top 1 策略在 Holdout 的表现")
    print("=" * 80)

    # Get training Top 1
    train_top1 = training_df.sort_values("vec_calmar_ratio", ascending=False).iloc[0]
    train_top1_combo = train_top1["combo"]

    # Find its Holdout performance
    top1_holdout = merged_df[merged_df["combo"] == train_top1_combo]

    if not top1_holdout.empty:
        row = top1_holdout.iloc[0]
        stable_rank = df_stable.index.get_loc(row.name) + 1

        print(f"训练集 Top 1: {train_top1_combo[:80]}")
        print(f"\n训练集表现:")
        print(f"  Return:  {row['vec_return']*100:.2f}%")
        print(f"  MDD:     {row['vec_max_drawdown']*100:.2f}%")
        print(f"  Calmar:  {row['vec_calmar_ratio']:.3f}")
        print(f"  排名:    1 / {len(training_df)}")

        print(f"\nHoldout表现:")
        print(f"  Return:  {row['holdout_return']*100:.2f}%")
        print(f"  MDD:     {row['holdout_max_drawdown']*100:.2f}%")
        print(f"  Calmar:  {row['holdout_calmar_ratio']:.3f}")
        print(f"  排名:    {int(row['holdout_calmar_rank'])} / {len(merged_df)}")

        print(f"\n稳定性:")
        print(f"  双稳定得分: {row['calmar_ratio_stability']:.3f}")
        print(f"  双稳定排名: {stable_rank} / {len(merged_df)}")

        if row["holdout_calmar_ratio"] < row["vec_calmar_ratio"] * 0.5:
            print(
                f"\n⚠️  警告: 该策略在 Holdout 上表现显著下降 ({row['holdout_calmar_ratio']/row['vec_calmar_ratio']*100:.1f}%)"
            )
            print("   可能存在过拟合！")
        elif stable_rank > 100:
            print(f"\n⚠️  警告: 双稳定排名跌出前100，说明泛化能力不足！")
        else:
            print(f"\n✅ 该策略在 Holdout 上保持良好表现")
    else:
        print("未找到训练集 Top 1 在 Holdout 的结果")

    # Summary Statistics
    print("\n" + "=" * 80)
    print("📈 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"总策略数: {len(merged_df)}")
    print(f"训练集 Calmar 中位数: {merged_df['vec_calmar_ratio'].median():.3f}")
    print(f"Holdout Calmar 中位数: {merged_df['holdout_calmar_ratio'].median():.3f}")
    print(f"双稳定得分中位数: {merged_df['calmar_ratio_stability'].median():.3f}")

    # Overfitting Analysis
    overfit_ratio = (
        merged_df["vec_calmar_ratio"] / merged_df["holdout_calmar_ratio"]
    ).median()
    print(f"\n过拟合指标 (Train/Holdout Calmar比值中位数): {overfit_ratio:.2f}")
    if overfit_ratio > 1.5:
        print("⚠️  显著过拟合！训练集表现远超Holdout")
    elif overfit_ratio > 1.2:
        print("⚠️  轻微过拟合，需要关注")
    else:
        print("✅ 泛化能力良好")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
