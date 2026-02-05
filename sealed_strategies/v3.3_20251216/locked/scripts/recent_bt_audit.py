#!/usr/bin/env python3
"""近期窗口 BT 审计

对最终筛选出来的稳定策略，在最近一段时间内（例如 30/60 自然日）
用 Backtrader 做细粒度回测，输出：
- 最近窗口的收益率 / 最大回撤 / 简易 Sharpe
- 最近窗口的日度权益曲线（便于人工核查买卖节奏）

注意：
- 为保持与生产流程解耦，本脚本只读取已有配置和因子，不改动任何核心逻辑。
- 默认从 stable_top200_analysis/final_ranking_top200.csv 读取排名，选 Top-N。
"""

import sys
from pathlib import Path
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
import backtrader as bt
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
)
from etf_strategy.auditor.core.engine import GenericStrategy, PandasData


def run_bt_with_daily_equity(
    combined_score_df: pd.DataFrame,
    timing_series: pd.Series,
    vol_regime_series: pd.Series,
    etf_codes,
    data_feeds,
    rebalance_schedule,
    freq: int,
    pos_size: int,
    initial_capital: float,
    commission_rate: float,
):
    """单组合 BT 回测，返回全周期风险指标 + 日度权益曲线 Series。

    为了做近期窗口分析，这里保留 TimeReturn 的日期索引。
    """
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=commission_rate, leverage=1.0)
    cerebro.broker.set_coc(True)
    cerebro.broker.set_checksubmit(False)

    for ticker, df in data_feeds.items():
        data = PandasData(dataname=df, name=ticker)
        cerebro.adddata(data)

    cerebro.addstrategy(
        GenericStrategy,
        scores=combined_score_df,
        timing=timing_series,
        vol_regime=vol_regime_series,
        etf_codes=etf_codes,
        freq=freq,
        pos_size=pos_size,
        rebalance_schedule=rebalance_schedule,
    )

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        compression=1,
        riskfreerate=0.0,
        annualize=True,
    )
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="timereturn", timeframe=bt.TimeFrame.Days
    )

    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    end_val = cerebro.broker.getvalue()

    bt_return = (end_val / start_val) - 1.0

    # Drawdown
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd_analysis.get("max", {}).get("drawdown", 0.0) / 100.0

    # Sharpe
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_analysis.get("sharperatio", 0.0) or 0.0

    # 年化收益（与 batch_bt_backtest 中一致）
    trading_days = len(combined_score_df)
    years = trading_days / 252.0 if trading_days > 0 else 1.0
    annual_return = (1.0 + bt_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    if sharpe_ratio != 0.0 and abs(sharpe_ratio) > 1e-4:
        annual_volatility = abs(annual_return / sharpe_ratio)
    else:
        annual_volatility = 0.0

    calmar_ratio = annual_return / max_drawdown if max_drawdown > 1e-4 else 0.0

    # 日度收益（带日期索引）
    tr_analysis = strat.analyzers.timereturn.get_analysis()
    if isinstance(tr_analysis, dict) and len(tr_analysis) > 0:
        daily_ret = pd.Series(tr_analysis)
        daily_ret.index = pd.to_datetime(daily_ret.index)
        daily_ret = daily_ret.sort_index()
    else:
        daily_ret = pd.Series(dtype="float64")

    if not daily_ret.empty:
        equity = initial_capital * (1.0 + daily_ret).cumprod()
    else:
        equity = pd.Series(
            [start_val, end_val],
            index=[pd.Timestamp("1970-01-01"), pd.Timestamp("1970-01-02")],
        )

    risk_metrics = {
        "bt_return": bt_return,
        "bt_max_drawdown": max_drawdown,
        "bt_annual_return": annual_return,
        "bt_annual_volatility": annual_volatility,
        "bt_sharpe_ratio": sharpe_ratio,
        "bt_calmar_ratio": calmar_ratio,
    }

    return risk_metrics, equity


def compute_window_stats(equity: pd.Series, window_days: int):
    """基于日度权益曲线，计算最近 window_days 自然日内的简单统计。"""
    if equity.empty:
        return {
            "window_start": None,
            "window_end": None,
            "window_days": window_days,
            "window_return": np.nan,
            "window_max_drawdown": np.nan,
            "window_sharpe": np.nan,
        }

    end_date = equity.index.max()
    start_cut = end_date - timedelta(days=window_days)
    window_eq = equity[equity.index >= start_cut]

    if len(window_eq) < 5:
        # 交易日太少，不具备统计意义
        return {
            "window_start": window_eq.index.min() if not window_eq.empty else None,
            "window_end": end_date,
            "window_days": window_days,
            "window_return": np.nan,
            "window_max_drawdown": np.nan,
            "window_sharpe": np.nan,
        }

    window_ret = window_eq.iloc[-1] / window_eq.iloc[0] - 1.0

    # 简易最大回撤（基于窗口内部）
    running_max = window_eq.cummax()
    dd = window_eq / running_max - 1.0
    window_mdd = dd.min()

    # 简易 Sharpe（基于日度收益，非严格年化，仅作为相对比较）
    daily_ret = window_eq.pct_change().dropna()
    if daily_ret.std() > 0:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    else:
        sharpe = np.nan

    return {
        "window_start": window_eq.index.min(),
        "window_end": end_date,
        "window_days": window_days,
        "window_return": window_ret,
        "window_max_drawdown": float(window_mdd),
        "window_sharpe": float(sharpe),
    }


def main():
    parser = argparse.ArgumentParser(
        description="近期窗口 BT 审计：验证最近市场阶段表现是否稳健"
    )
    parser.add_argument(
        "--topn", type=int, default=5, help="从最终排名中选取前 N 个组合进行审计"
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[60, 120, 240],
        help="近期窗口长度列表（自然日，多档，例如 60 120 240）",
    )
    args = parser.parse_args()

    windows = sorted(set(args.windows))

    print("=" * 80)
    print(f"📊 近期窗口 BT 审计 (Top-{args.topn}, 窗口 {windows} 天)")
    print("=" * 80)

    # 1. 读取最终排名结果
    ranking_path = (
        ROOT / "results" / "stable_top200_analysis" / "final_ranking_top200.csv"
    )
    if not ranking_path.exists():
        print(f"❌ 未找到最终排名文件: {ranking_path}")
        sys.exit(1)

    ranking_df = pd.read_csv(ranking_path)
    ranking_df = ranking_df.sort_values("final_calmar", ascending=False)
    top_df = ranking_df.head(args.topn).copy()

    print(
        f"✅ 加载最终排名: {len(ranking_df)} 条，选取 Top-{args.topn} 进行近期窗口审计"
    )

    # 2. 加载配置和数据
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()

    backtest_config = config.get("backtest", {})
    freq = backtest_config.get("freq", 3)
    pos_size = backtest_config.get("pos_size", 2)
    initial_capital = float(backtest_config.get("initial_capital", 1_000_000.0))
    commission_rate = float(backtest_config.get("commission_rate", 0.0002))
    lookback = backtest_config.get("lookback", 252)

    print(
        f"✅ 回测参数: FREQ={freq}, POS={pos_size}, Capital={initial_capital}, Comm={commission_rate}, LOOKBACK={lookback}"
    )

    # 择时
    timing_config = backtest_config.get("timing", {})
    extreme_threshold = timing_config.get("extreme_threshold", -0.4)
    extreme_position = timing_config.get("extreme_position", 0.3)

    timing_module = LightTimingModule(
        extreme_threshold=extreme_threshold,
        extreme_position=extreme_position,
    )
    timing_series_raw = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_shifted = shift_timing_signal(
        timing_series_raw.reindex(dates).fillna(1.0).values
    )
    timing_series = pd.Series(timing_arr_shifted, index=dates)

    # 波动率体制 (与主流程保持一致)
    if "510300" in ohlcv["close"].columns:
        hs300 = ohlcv["close"]["510300"]
    else:
        hs300 = ohlcv["close"].iloc[:, 0]
    rets = hs300.pct_change()
    hv = rets.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
    hv_5d = hv.shift(5)
    regime_vol = (hv + hv_5d) / 2
    exposure_s = pd.Series(1.0, index=regime_vol.index)
    exposure_s[regime_vol >= 25] = 0.7
    exposure_s[regime_vol >= 30] = 0.4
    exposure_s[regime_vol >= 40] = 0.1
    vol_regime_series = exposure_s.reindex(dates).fillna(1.0)

    # 统一调仓日程
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=len(dates),
        lookback_window=lookback,
        freq=freq,
    )

    # 准备 data feeds
    data_feeds = {}
    for ticker in etf_codes:
        df = pd.DataFrame(
            {
                "open": ohlcv["open"][ticker],
                "high": ohlcv["high"][ticker],
                "low": ohlcv["low"][ticker],
                "close": ohlcv["close"][ticker],
                "volume": ohlcv["volume"][ticker],
            }
        )
        df = df.reindex(dates)
        df = df.ffill().fillna(0.01)
        data_feeds[ticker] = df

    print(f"✅ 数据加载完成：{len(dates)} 个交易日 × {len(etf_codes)} 只 ETF")

    # 3. 对每个 Top 组合做 BT + 多窗口分析
    output_dir = ROOT / "results" / "recent_bt_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for rank, row in enumerate(top_df.itertuples(index=False), start=1):
        combo = row.combo
        factors = [f.strip() for f in combo.split("+")]
        print("-" * 80)
        print(f"▶️  组合 #{rank}: {combo}")

        combined_score_df = pd.DataFrame(0.0, index=dates, columns=etf_codes)
        for fct in factors:
            combined_score_df = combined_score_df.add(std_factors[fct], fill_value=0.0)

        risk_metrics, equity = run_bt_with_daily_equity(
            combined_score_df,
            timing_series,
            vol_regime_series,
            etf_codes,
            data_feeds,
            rebalance_schedule,
            freq,
            pos_size,
            initial_capital,
            commission_rate,
        )

        print(
            f"  全周期 BT 收益: {risk_metrics['bt_return']*100:6.2f}% | MaxDD: {risk_metrics['bt_max_drawdown']*100:5.1f}% | Calmar: {risk_metrics['bt_calmar_ratio']:5.2f}"
        )

        # 多窗口统计
        window_stats_all = {}
        for w in windows:
            ws = compute_window_stats(equity, w)
            if ws["window_start"] is not None:
                print(
                    f"  近期窗口{w}天 [{ws['window_start'].date()} ~ {ws['window_end'].date()}]"
                    f" 收益: {ws['window_return']*100:6.2f}% | MaxDD: {ws['window_max_drawdown']*100:5.1f}% | Sharpe≈ {ws['window_sharpe']:5.2f}"
                )
            else:
                print(f"  近期窗口{w}天交易日太少，无法评估。")

            prefix = f"window{w}"
            window_stats_all.update(
                {
                    f"{prefix}_start": ws["window_start"],
                    f"{prefix}_end": ws["window_end"],
                    f"{prefix}_days": ws["window_days"],
                    f"{prefix}_return": ws["window_return"],
                    f"{prefix}_max_drawdown": ws["window_max_drawdown"],
                    f"{prefix}_sharpe": ws["window_sharpe"],
                }
            )

        # 保存该组合的全周期日度权益曲线（便于进一步人工核查）
        eq_path = output_dir / f"equity_rank{rank}.csv"
        equity.to_csv(eq_path, header=["equity"])

        summary_row = {
            "rank": rank,
            "combo": combo,
            **risk_metrics,
            **window_stats_all,
        }
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    windows_tag = "-".join(str(w) for w in windows)
    summary_path = (
        output_dir / f"recent_bt_summary_top{args.topn}_windows_{windows_tag}d.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    # 4. 可视化总览（每个窗口一行，每个组合一条柱状）
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(windows), 1, figsize=(10, 4 * len(windows)))
        if len(windows) == 1:
            axes = [axes]

        for ax, w in zip(axes, windows):
            col_ret = f"window{w}_return"
            if col_ret not in summary_df.columns:
                continue
            xs = summary_df["rank"].values
            ys = summary_df[col_ret].values * 100.0
            ax.bar(xs, ys, alpha=0.7)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xlabel("Rank")
            ax.set_ylabel(f"{w}天收益 (%)")
            ax.set_title(f"Top-{args.topn} 组合 {w} 天窗口收益")
            ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        fig_path = output_dir / f"recent_bt_windows_{windows_tag}d.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"📈 多窗口收益总览图已保存: {fig_path}")
    except Exception:
        print("⚠️ 生成多窗口可视化失败（可能未安装 matplotlib），仅输出 CSV。")

    print("=" * 80)
    print(f"✅ 近期窗口 BT 审计完成，结果已保存到: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
