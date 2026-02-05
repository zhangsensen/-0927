#!/usr/bin/env python3
"""
VEC-BT 逐日对账诊断（单组合）

目标：
- 对指定组合（默认：CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D）
  在给定参数下 (FREQ=8, POS_SIZE=2, LOOKBACK=252) 生成 VEC 与 BT 的逐调仓日志，并做基础对比。

输出：
- results/diagnostics/vec_daily_log.csv
- results/diagnostics/bt_daily_log.csv
- results/diagnostics/alignment_diagnosis_report.md
- results/diagnostics/alignment_curve.png
"""
import json
import math
from pathlib import Path
from typing import List, Dict, Any

import backtrader as bt
import numpy as np
import pandas as pd
import yaml

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
)


# ---------- 配置 ----------
COMBO_STR = (
    "CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D"
)
FREQ = 8
POS_SIZE = 2
LOOKBACK = 252
OUTPUT_DIR = Path("results/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMMISSION = 0.0002
INITIAL_CAPITAL = 1_000_000.0
# 仅首个调仓日诊断与简化开关
DIVERGENCE_THRESHOLD_CASH = 1.0
DIVERGENCE_THRESHOLD_TOTAL = 1.0
DIVERGENCE_THRESHOLD_SELECTION = True  # 选股集合不一致立即报警
FIRST_REBALANCE_ONLY = False  # 全程检测；在主流程控制停止点
STOP_AFTER_REBAL_IDX = LOOKBACK + FREQ  # 仅跑到第二个调仓日（t=264）
NEUTRAL_TIMING_VOL = True  # 可选：置为1以排除择时/vol影响


def stable_topk_indices_py(scores: np.ndarray, k: int) -> List[int]:
    """纯 Python 稳定 top-k（score 降序，tie 时索引小优先）"""
    idx = np.arange(len(scores))
    valid = ~np.isnan(scores)
    arr = [(-scores[i], i) for i in idx[valid]]  # 负号实现降序
    arr.sort()
    return [i for _, i in arr[:k]]


def calc_vol_regime(hs300_close: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    rets = hs300_close.pct_change()
    hv = rets.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
    hv_5 = hv.shift(5)
    regime = (hv + hv_5) / 2
    exposure = pd.Series(1.0, index=regime.index)
    exposure[regime >= 25] = 0.7
    exposure[regime >= 30] = 0.4
    exposure[regime >= 40] = 0.1
    return exposure.reindex(dates).fillna(1.0)


# ---------- VEC 纯 Python 模拟器 ----------
def run_vec_diag(
    factors_3d, prices, timing_arr, vol_regime_arr, factor_indices, dates, etf_codes
):
    close, openp, highp, lowp = prices
    T, N = close.shape
    cash = INITIAL_CAPITAL
    holdings = np.zeros(N, dtype=float)
    log_rows = []

    rebalance = generate_rebalance_schedule(
        total_periods=T, lookback_window=LOOKBACK, freq=FREQ
    )
    for t in rebalance:
        # 记录调仓前快照（使用 t-1 收盘价估算持仓市值）
        holdings_value_before = np.nansum(holdings * close[t - 1])
        total_before = cash + holdings_value_before
        cash_before = cash

        if t > STOP_AFTER_REBAL_IDX:
            break

        # 因子得分 (t-1)
        scores = np.zeros(N)
        scores[:] = np.nan
        for n in range(N):
            s = 0.0
            valid = False
            for idx in factor_indices:
                v = factors_3d[t - 1, n, idx]
                if not math.isnan(v):
                    s += v
                    valid = True
            if valid and s != 0.0:
                scores[n] = s

        # timing / vol regime (t)
        timing = 1.0 if NEUTRAL_TIMING_VOL else timing_arr[t]
        vol_r = 1.0 if NEUTRAL_TIMING_VOL else vol_regime_arr[t]
        timing_ratio = timing * vol_r

        print(
            f"[VEC] t={t}, date={dates[t].date()}, cash_before={cash_before:.2f}, total_before={total_before:.2f}, timing={timing:.4f}, vol_regime={vol_r:.4f}"
        )

        # 选股
        selected_idx = stable_topk_indices_py(scores, POS_SIZE)
        selected_codes = [etf_codes[i] for i in selected_idx]
        selected_scores = [scores[i] for i in selected_idx]
        print(
            f"[VEC] selected_idx={selected_idx}, selected_codes={selected_codes}, selected_scores={selected_scores}"
        )

        # 卖出
        sell_list = []
        for n in range(N):
            if holdings[n] > 0 and n not in selected_idx:
                price = close[t, n]
                proceeds = holdings[n] * price * (1 - COMMISSION)
                sell_list.append((etf_codes[n], holdings[n], price, proceeds))
                cash += proceeds
                holdings[n] = 0.0

        # 买入
        kept_value = np.nansum(holdings * close[t])
        target_exposure = (cash + kept_value) * timing_ratio
        available = max(0.0, target_exposure - kept_value)
        buy_list = []
        print(
            f"[VEC] kept_value={kept_value:.2f}, target_exposure={target_exposure:.2f}, available={available:.2f}"
        )
        if available > 0:
            target_pos_value = available / len(selected_idx) / (1 + COMMISSION)
            for n in selected_idx:
                price = close[t, n]
                if price <= 0 or np.isnan(price):
                    continue
                shares = target_pos_value / price
                cost = shares * price * (1 + COMMISSION)
                if cash >= cost - 1e-6:
                    cash -= cost
                    holdings[n] += shares
                    buy_list.append(
                        (etf_codes[n], target_pos_value, price, shares, cost)
                    )
                    print(
                        f"[VEC] buy {etf_codes[n]} price={price:.4f} shares={shares:.4f} cost={cost:.2f}"
                    )

        total_after = cash + np.nansum(holdings * close[t])
        print(f"[VEC] cash_after={cash:.2f}, total_after={total_after:.2f}")

        log_rows.append(
            {
                "date": str(dates[t].date()),
                "t_idx": int(t),
                "cash_before": float(cash_before),
                "total_before": float(total_before),
                "timing_ratio": float(timing),
                "vol_regime": float(vol_r),
                "selected_indices": ",".join(map(str, selected_idx)),
                "selected_etfs": ",".join(selected_codes),
                "selected_scores": ",".join(f"{x:.6f}" for x in selected_scores),
                "sell_list": ";".join(
                    f"{c}:{sh:.4f}:{p:.4f}:{pro:.2f}" for c, sh, p, pro in sell_list
                ),
                "buy_list": ";".join(
                    f"{c}:{tv:.2f}:{p:.4f}:{sh:.4f}:{co:.2f}"
                    for c, tv, p, sh, co in buy_list
                ),
                "cash_after": float(cash),
                "total_after": float(total_after),
            }
        )

    vec_log = pd.DataFrame(log_rows)
    vec_log.to_csv(OUTPUT_DIR / "vec_daily_log.csv", index=False)
    return vec_log


# ---------- BT 诊断策略 ----------
class DiagnosticStrategy(bt.Strategy):
    params = dict(
        scores=None,
        timing=None,
        vol_regime=None,
        freq=FREQ,
        pos_size=POS_SIZE,
        rebalance_schedule=None,
        commission=COMMISSION,
    )

    def __init__(self):
        self.etf_map = {d._name: d for d in self.datas}
        self.rebalance_set = set(self.params.rebalance_schedule.tolist())
        self.cash = INITIAL_CAPITAL
        self.daily_log = []
        self.shadow_holdings = {d._name: 0.0 for d in self.datas}

    def next(self):
        bar_index = len(self) - 1
        if bar_index < LOOKBACK:
            return
        if bar_index not in self.rebalance_set:
            return
        self.rebalance(bar_index)
        if bar_index >= STOP_AFTER_REBAL_IDX:
            self.env.runstop()

    def rebalance(self, bar_index: int):
        dt = self.datas[0].datetime.date(0)
        # holdings before (snapshot)
        holdings = dict(self.shadow_holdings)
        holdings_value_before = sum(
            holdings.get(d._name, 0.0) * d.close[-1] for d in self.datas
        )
        cash = self.broker.getcash()
        cash_before = cash
        total_before = cash + holdings_value_before

        # factor scores from t-1
        prev_ts = self.params.scores.index[bar_index - 1]
        row = self.params.scores.loc[prev_ts]
        scores = row.values

        timing = 1.0 if NEUTRAL_TIMING_VOL else self.params.timing.iloc[bar_index]
        vol_r = 1.0 if NEUTRAL_TIMING_VOL else self.params.vol_regime.iloc[bar_index]
        timing_ratio = timing * vol_r

        selected_idx = stable_topk_indices_py(scores, self.params.pos_size)
        selected_codes = [row.index[i] for i in selected_idx]
        selected_scores = [scores[i] for i in selected_idx]
        print(
            f"[BT] t={bar_index}, date={dt}, cash_before={cash_before:.2f}, total_before={total_before:.2f}, timing={timing:.4f}, vol_regime={vol_r:.4f}"
        )
        print(
            f"[BT] selected_idx={selected_idx}, selected_codes={selected_codes}, selected_scores={selected_scores}"
        )

        # sell
        sell_list = []
        holdings_after_log = dict(holdings)  # 用于日志的持仓快照
        for code, sh in list(holdings.items()):
            if code not in selected_codes and sh > 0:
                data = self.etf_map[code]
                price = data.close[0]
                proceeds = sh * price * (1 - self.params.commission)
                self.close(data=data)
                cash += proceeds
                sell_list.append((code, sh, price, proceeds))
                holdings_after_log[code] = 0.0
                self.shadow_holdings[code] = 0.0

        kept_value = sum(
            self.getposition(self.etf_map[c]).size * self.etf_map[c].close[0]
            for c in selected_codes
            if c in holdings
        )
        target_exposure = (cash + kept_value) * timing_ratio
        available = max(0.0, target_exposure - kept_value)

        buy_list = []
        print(
            f"[BT] kept_value={kept_value:.2f}, target_exposure={target_exposure:.2f}, available={available:.2f}"
        )
        if available > 0:
            target_pos_value = (
                available / self.params.pos_size / (1 + self.params.commission)
            )
            for code in selected_codes:
                if holdings.get(code, 0.0) > 0:
                    continue
                data = self.etf_map[code]
                price = data.close[0]
                if price <= 0 or math.isnan(price):
                    continue
                shares = target_pos_value / price
                cost = shares * price * (1 + self.params.commission)
                if cash >= cost - 1e-6:
                    self.buy(data=data, size=shares)
                    cash -= cost
                    buy_list.append((code, target_pos_value, price, shares, cost))
                    holdings_after_log[code] = (
                        holdings_after_log.get(code, 0.0) + shares
                    )
                    self.shadow_holdings[code] = (
                        self.shadow_holdings.get(code, 0.0) + shares
                    )
                    print(
                        f"[BT] buy {code} price={price:.4f} shares={shares:.4f} cost={cost:.2f}"
                    )

        # 日志中的持仓市值（使用本bar收盘价，包含计划买入的shares；已卖出的设为0）
        holdings_value_after_log = sum(
            (holdings_after_log.get(d._name, 0.0)) * d.close[0] for d in self.datas
        )
        total_after_log = cash + holdings_value_after_log
        print(f"[BT] cash_after={cash:.2f}, total_after_log={total_after_log:.2f}")

        self.daily_log.append(
            {
                "date": str(dt),
                "t_idx": int(bar_index),
                "cash_before": float(cash_before),
                "total_before": float(total_before),
                "timing_ratio": float(timing),
                "vol_regime": float(vol_r),
                "selected_indices": ",".join(map(str, selected_idx)),
                "selected_etfs": ",".join(selected_codes),
                "selected_scores": ",".join(f"{x:.6f}" for x in selected_scores),
                "sell_list": ";".join(
                    f"{c}:{sh:.4f}:{p:.4f}:{pro:.2f}" for c, sh, p, pro in sell_list
                ),
                "buy_list": ";".join(
                    f"{c}:{tv:.2f}:{p:.4f}:{sh:.4f}:{co:.2f}"
                    for c, tv, p, sh, co in buy_list
                ),
                "cash_after": float(cash),
                "total_after": float(total_after_log),
            }
        )

    def stop(self):
        pd.DataFrame(self.daily_log).to_csv(
            OUTPUT_DIR / "bt_daily_log.csv", index=False
        )


def main():
    # 1) 数据与因子
    config = yaml.safe_load(Path("configs/combo_wfo_config.yaml").read_text())
    loader = DataLoader(
        data_dir=config["data"]["data_dir"], cache_dir=config["data"]["cache_dir"]
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = sorted(
        raw_factors_df.columns.get_level_values(0).unique().tolist()
    )
    raw_factors = {f: raw_factors_df[f] for f in factor_names_list}
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    # 因子索引
    factor_index_map = {n: i for i, n in enumerate(factor_names_list)}
    combo_factors = [f.strip() for f in COMBO_STR.split("+")]
    combo_indices = [factor_index_map[f] for f in combo_factors]

    cols = std_factors[factor_names_list[0]].columns
    dates = std_factors[factor_names_list[0]].index

    close = ohlcv["close"][cols].ffill().bfill()
    openp = ohlcv["open"][cols].ffill().bfill()
    highp = ohlcv["high"][cols].ffill().bfill()
    lowp = ohlcv["low"][cols].ffill().bfill()
    prices = (close.values, openp.values, highp.values, lowp.values)
    factors_3d = np.stack([std_factors[f].values for f in factor_names_list], axis=-1)

    # timing
    timing_module = LightTimingModule(
        extreme_threshold=config["backtest"]["timing"]["extreme_threshold"],
        extreme_position=config["backtest"]["timing"]["extreme_position"],
    )
    timing_raw = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr = shift_timing_signal(timing_raw.reindex(dates).fillna(1.0).values)
    timing_series = pd.Series(timing_arr, index=dates)

    # vol regime
    if "510300" in ohlcv["close"].columns:
        hs300 = ohlcv["close"]["510300"]
    else:
        hs300 = ohlcv["close"].iloc[:, 0]
    vol_series = calc_vol_regime(hs300, dates)
    vol_regime_arr = vol_series.values

    # 2) VEC 诊断
    vec_log = run_vec_diag(
        factors_3d,
        prices,
        timing_arr,
        vol_regime_arr,
        combo_indices,
        dates,
        cols.tolist(),
    )

    # 3) BT 诊断
    # 准备 score DF (t-1 使用)
    score_df = pd.DataFrame(0.0, index=dates, columns=cols)
    for f in combo_factors:
        score_df = score_df.add(std_factors[f], fill_value=0)
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=len(dates), lookback_window=LOOKBACK, freq=FREQ
    )

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION, leverage=1.0)
    cerebro.broker.set_coc(True)

    data_feeds = {}
    for c in cols:
        df = (
            pd.DataFrame(
                {
                    "open": ohlcv["open"][c],
                    "high": ohlcv["high"][c],
                    "low": ohlcv["low"][c],
                    "close": ohlcv["close"][c],
                    "volume": ohlcv["volume"][c],
                }
            )
            .reindex(dates)
            .ffill()
            .fillna(0.01)
        )
        data = bt.feeds.PandasData(dataname=df, name=c)
        cerebro.adddata(data)
        data_feeds[c] = df

    cerebro.addstrategy(
        DiagnosticStrategy,
        scores=score_df,
        timing=timing_series,
        vol_regime=vol_series,
        freq=FREQ,
        pos_size=POS_SIZE,
        rebalance_schedule=rebalance_schedule,
        commission=COMMISSION,
    )
    cerebro.run()
    # 日志已在 stop() 写出

    # 4) 对比与报告
    vec_log = pd.read_csv(OUTPUT_DIR / "vec_daily_log.csv")
    bt_log = pd.read_csv(OUTPUT_DIR / "bt_daily_log.csv")
    merged = pd.merge(vec_log, bt_log, on="date", suffixes=("_vec", "_bt"))
    divergences = []
    for _, row in merged.iterrows():
        cash_diff = abs(row["cash_after_vec"] - row["cash_after_bt"])
        total_diff = abs(row["total_after_vec"] - row["total_after_bt"])
        sel_vec = row["selected_etfs_vec"]
        sel_bt = row["selected_etfs_bt"]
        sel_diff = sel_vec != sel_bt if DIVERGENCE_THRESHOLD_SELECTION else False
        if (
            cash_diff > DIVERGENCE_THRESHOLD_CASH
            or total_diff > DIVERGENCE_THRESHOLD_TOTAL
            or sel_diff
        ):
            divergences.append((row, cash_diff, total_diff, sel_diff))
            break

    if divergences:
        first_row = divergences[0][0]
        first_cash = pd.DataFrame([first_row])
        first_total = first_cash
        first_cash["cash_diff"] = divergences[0][1]
        first_cash["total_diff"] = divergences[0][2]
        first_cash["selection_diff"] = divergences[0][3]
    else:
        first_cash = pd.DataFrame()
        first_total = pd.DataFrame()

    # 保存资产曲线图
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(merged["date"], merged["total_after_vec"], label="VEC")
    plt.plot(merged["date"], merged["total_after_bt"], label="BT")
    if len(first_total) > 0:
        plt.axvline(
            first_total.iloc[0]["date"],
            color="r",
            linestyle="--",
            label="first total divergence",
        )
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "alignment_curve.png")

    # Markdown 报告
    report = []
    report.append("# VEC-BT 对齐诊断报告 (单组合)")
    report.append(f"- 组合: {COMBO_STR}")
    report.append(f"- 参数: FREQ={FREQ}, POS={POS_SIZE}, LOOKBACK={LOOKBACK}")
    report.append(f"- 调仓次数: {len(merged)}")
    report.append("")
    report.append("## 结果摘要")
    report.append(
        f"- 现金首次差异 > $1: {first_cash.iloc[0]['date'] if len(first_cash)>0 else '无'}"
    )
    report.append(
        f"- 总资产首次差异 > $1: {first_total.iloc[0]['date'] if len(first_total)>0 else '无'}"
    )
    report.append("")
    report.append("## 差异示例（前1条）")
    if len(first_cash) > 0:
        row = first_cash.iloc[0]
        report.append(
            f"- 日期: {row['date']}, total_diff={row.get('total_diff', 0):.2f}, cash_diff={row.get('cash_diff', 0):.2f}, selection_diff={row.get('selection_diff', False)}"
        )
    else:
        report.append("- 未发现触发阈值的差异")
    report.append("")
    report.append("## 资产曲线")
    report.append(f"![](alignment_curve.png)")

    Path(OUTPUT_DIR / "alignment_diagnosis_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
