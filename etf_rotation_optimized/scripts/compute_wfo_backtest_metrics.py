#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WFO回测指标计算 - Top-N因子加权策略

从 wfo_results.pkl 提取选中因子与窗口时间，基于因子值排序Top-N建仓：
- 合成分数：选中因子等权合成composite_score
- 持仓构建：按composite_score排序取Top-N（默认12只ETF）
- 收益计算：持仓期内等权组合收益
- 指标输出：年化收益、夏普、最大回撤、换手率、胜率

输入：wfo/{timestamp}/wfo_results.pkl + standardized因子 + ohlcv
输出：扩展 wfo_report.txt 与 metadata.json
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 策略参数
# 与 Step4 回测脚本保持一致：默认TopN=5，可通过CLI覆盖
TOP_N_HOLDINGS = 5  # Top-N持仓数量（默认，可通过CLI覆盖）


def load_ohlcv_data():
    """加载OHLCV数据（使用最新cross_section）"""
    cross_section_root = PROJECT_ROOT / "results" / "cross_section"
    all_runs = []
    for date_dir in cross_section_root.iterdir():
        if not date_dir.is_dir():
            continue
        for ts_dir in date_dir.iterdir():
            if (ts_dir / "metadata.json").exists():
                all_runs.append(ts_dir)
    if not all_runs:
        raise FileNotFoundError("无法找到cross_section数据")
    all_runs.sort(key=lambda p: p.name, reverse=True)
    latest = all_runs[0]
    ohlcv_dir = latest / "ohlcv"
    close_df = pd.read_parquet(ohlcv_dir / "close.parquet")
    return close_df


def load_standardized_factors():
    """加载标准化因子数据（使用最新factor_selection）"""
    selection_root = PROJECT_ROOT / "results" / "factor_selection"
    all_runs = []
    for date_dir in selection_root.iterdir():
        if not date_dir.is_dir():
            continue
        for ts_dir in date_dir.iterdir():
            if (ts_dir / "metadata.json").exists():
                all_runs.append(ts_dir)
    if not all_runs:
        raise FileNotFoundError("无法找到factor_selection数据")
    all_runs.sort(key=lambda p: p.name, reverse=True)
    latest = all_runs[0]

    # 读取元数据获取因子列表
    with open(latest / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    factor_names = meta["standardized_factor_names"]
    standardized_dir = latest / "standardized"

    factors_dict = {}
    for fname in factor_names:
        parquet_path = standardized_dir / f"{fname}.parquet"
        if parquet_path.exists():
            factors_dict[fname] = pd.read_parquet(parquet_path)

    return factors_dict


def compute_portfolio_returns_topn(
    close_df: pd.DataFrame,
    window_results: List[dict],
    factors_dict: Dict[str, pd.DataFrame],
    top_n: int = TOP_N_HOLDINGS,
    exclude_factors: Optional[List[str]] = None,
    weight_mode: str = "equal",
    constraint_reports: Optional[List] = None,
    tx_cost_bps: float = 0.0,
    max_turnover: float = 1.0,
):
    """
    计算Top-N因子加权组合收益

    流程：
    1. 每窗口OOS起始，读取选中因子值
    2. 等权合成composite_score
    3. 按score排序取Top-N只ETF等权建仓
    4. 计算持仓期收益与换手
    """
    returns = close_df.pct_change(fill_method=None).iloc[1:]  # 对齐
    gross_rets = []
    net_rets = []
    holdings_history = []
    turnover_list = []
    prev_holdings = set()
    etf_codes = close_df.columns.tolist()

    exclude_factors = set(exclude_factors or [])

    for w in window_results:
        oos_start = int(w["oos_start"])
        oos_end = int(w["oos_end"])
        selected_factors = [
            f for f in w["selected_factors"] if f not in exclude_factors
        ]  # 过滤禁用因子

        if oos_start >= len(returns) or oos_end > len(returns):
            print(f"⚠️  窗口{w['window_id']}: OOS索引超界，跳过")
            continue

        # 构建composite_score（选中因子加权合成）
        # 使用OOS起始日T的因子值（已对齐T-1→T预测）
        composite_scores = pd.Series(0.0, index=etf_codes)

        # 因子权重：equal 或 ic（使用IS IC作为权重，负IC截断为0）
        factor_weights: Dict[str, float] = {}
        if weight_mode == "ic" and constraint_reports is not None:
            try:
                wid = int(w.get("window_id", 0))
                report = (
                    constraint_reports[wid - 1]
                    if wid and len(constraint_reports) >= wid
                    else None
                )
                if report is not None and hasattr(report, "is_ic_stats"):
                    is_ic_stats: Dict[str, float] = report.is_ic_stats or {}
                    for f in selected_factors:
                        icv = float(is_ic_stats.get(f, 0.0))
                        factor_weights[f] = max(icv, 0.0)
                    # 归一化
                    s = sum(factor_weights.values())
                    if s > 0:
                        factor_weights = {k: v / s for k, v in factor_weights.items()}
                    else:
                        factor_weights = {}
            except Exception as e:
                print(f"⚠️  IS IC权重构建失败，回退等权: {e}")
                factor_weights = {}

        # 等权回退
        if not factor_weights:
            if len(selected_factors) > 0:
                eq_w = 1.0 / len(selected_factors)
                factor_weights = {f: eq_w for f in selected_factors}
            else:
                factor_weights = {}

        valid_factor_count = 0
        for factor_name in selected_factors:
            if factor_name not in factors_dict:
                continue
            factor_df = factors_dict[factor_name]
            if oos_start >= len(factor_df):
                continue
            # 取OOS起始日因子值（行索引=oos_start对应日期）
            factor_values = factor_df.iloc[oos_start]
            # 填充NaN为0（因子缺失ETF不参与排序）
            factor_values = factor_values.fillna(0.0)
            wgt = factor_weights.get(factor_name, 0.0)
            if wgt <= 0:
                continue
            composite_scores += factor_values * wgt
            valid_factor_count += 1

        if valid_factor_count == 0:
            print(f"⚠️  窗口{w['window_id']}: 无有效因子（可能被exclude过滤），跳过")
            continue

        # 如果由于权重过滤导致所有权重无效，回退等权一次
        if valid_factor_count == 0 and len(selected_factors) > 0:
            eq_w = 1.0 / len(selected_factors)
            for factor_name in selected_factors:
                if factor_name not in factors_dict:
                    continue
                factor_df = factors_dict[factor_name]
                if oos_start >= len(factor_df):
                    continue
                factor_values = factor_df.iloc[oos_start].fillna(0.0)
                composite_scores += factor_values * eq_w
            valid_factor_count = len(selected_factors)

        # 按score排序取Top-N（目标持仓）
        desired_ranked = (
            composite_scores.sort_values(ascending=False).index[:top_n].tolist()
        )
        desired_set = set(desired_ranked)

        # 应用换手约束：限制每期更换的标的数量（对应总换手 ≤ max_turnover）
        if prev_holdings and max_turnover < 1.0:
            # 允许更换的数量（双向对称差/Top-N = 2*k/top_n ≤ max_turnover）→ k ≤ floor(max_turnover*top_n/2)
            k_allowed = int(np.floor(max_turnover * top_n / 2))
            # 保留前一期且仍在目标list中的标的，按目标得分顺序
            keep = [etf for etf in desired_ranked if etf in prev_holdings]
            # 需要新增的候选（按目标得分顺序）
            adds = [etf for etf in desired_ranked if etf not in prev_holdings]
            adds = adds[:k_allowed]  # 限制新增数量
            # 若保留+新增不足以填满Top-N，用上一期未入选但得分较高的持仓填充
            remain_slots = top_n - (len(keep) + len(adds))
            fillers_candidates = [
                etf for etf in prev_holdings if etf not in desired_set
            ]
            # 用当前分数对上一期候选排序（高分优先）
            fillers_candidates.sort(
                key=lambda x: composite_scores.get(x, -np.inf), reverse=True
            )
            fillers = fillers_candidates[: max(0, remain_slots)]
            current_ranked = keep + adds + fillers
            # 若仍不足（极端情况），补齐为Top-N
            if len(current_ranked) < top_n:
                extra = [
                    etf for etf in desired_ranked if etf not in set(current_ranked)
                ]
                current_ranked += extra[: (top_n - len(current_ranked))]
        else:
            current_ranked = desired_ranked

        curr_holdings = set(current_ranked)

        # 计算换手率（单向，使用集合对称差定义/Top-N）
        if prev_holdings:
            turnover = len(curr_holdings.symmetric_difference(prev_holdings)) / max(
                len(curr_holdings), len(prev_holdings)
            )
        else:
            turnover = 1.0  # 首次全建仓
        turnover_list.append(turnover)

        # 计算持仓期收益（等权Top-N）
        window_rets = returns.iloc[oos_start:oos_end][current_ranked]
        eq_ret = window_rets.mean(axis=1)  # 等权组合日收益（毛收益）
        eq_ret_gross = eq_ret.copy()

        # 交易成本（仅在每个窗口开始日扣除）
        if tx_cost_bps and tx_cost_bps > 0:
            cost = turnover * (tx_cost_bps / 1e4)
            if len(eq_ret) > 0:
                eq_ret.iloc[0] = eq_ret.iloc[0] - cost

        gross_rets.extend(eq_ret_gross.tolist())
        net_rets.extend(eq_ret.tolist())

        prev_holdings = curr_holdings
        holdings_history.append(
            {
                "window": w["window_id"],
                "oos_start": oos_start,
                "oos_end": oos_end,
                "holdings": current_ranked[:5],  # 记录前5个示例
                "avg_score": float(composite_scores[current_ranked].mean()),
            }
        )

    return np.array(gross_rets), np.array(net_rets), turnover_list, holdings_history


def compute_metrics(portfolio_rets):
    """计算组合指标"""
    cum_rets = (1 + portfolio_rets).cumprod()
    total_ret = cum_rets[-1] - 1

    # 最大回撤
    running_max = np.maximum.accumulate(cum_rets)
    drawdowns = (cum_rets - running_max) / running_max
    max_dd = drawdowns.min()

    # 年化收益与波动
    n_days = len(portfolio_rets)
    ann_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 else 0.0
    ann_vol = portfolio_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # 胜率
    win_rate = (
        (portfolio_rets > 0).sum() / len(portfolio_rets)
        if len(portfolio_rets) > 0
        else 0.0
    )

    return {
        "total_return": float(total_ret),
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "trading_days": int(n_days),
    }


def main(
    wfo_dir: Path,
    top_n: int = TOP_N_HOLDINGS,
    exclude_factors=None,
    weight_mode: str = "equal",
    tx_cost_bps: float = 0.0,
    max_turnover: float = 1.0,
    target_vol: float = 0.0,
) -> None:
    """扩展WFO结果补充回测指标（Top-N因子加权策略）"""
    print(f"🔍 处理WFO结果: {wfo_dir}")
    print(
        f"📊 策略: Top-{top_n}因子加权组合 | 排除因子: {','.join(exclude_factors) if exclude_factors else '无'}"
    )

    # 加载结果
    pkl_path = wfo_dir / "wfo_results.pkl"
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    window_results = results.get("window_results", [])
    constraint_reports = results.get("constraint_reports", None)
    if not window_results:
        print("⚠️  无窗口结果，跳过")
        return

    # 加载数据
    close_df = load_ohlcv_data()
    factors_dict = load_standardized_factors()

    # 计算Top-N组合收益（毛/净）
    gross_rets, net_rets, turnover_list, holdings = compute_portfolio_returns_topn(
        close_df,
        window_results,
        factors_dict,
        top_n=top_n,
        exclude_factors=exclude_factors,
        weight_mode=weight_mode,
        constraint_reports=constraint_reports,
        tx_cost_bps=tx_cost_bps,
        max_turnover=max_turnover,
    )

    # 计算指标（毛收益）
    gross_metrics = compute_metrics(gross_rets)

    # 计算指标（净收益：含交易成本）
    net_before_vol_metrics = compute_metrics(net_rets)

    # 目标波动率（在净收益基础上进行缩放）
    final_rets = net_rets.copy()
    if target_vol and target_vol > 0:
        realized_vol = final_rets.std() * np.sqrt(252)
        if realized_vol > 0:
            scale = target_vol / realized_vol
            final_rets = final_rets * scale
    final_metrics = compute_metrics(final_rets)
    avg_turnover = float(np.mean(turnover_list)) if turnover_list else 0.0

    # 扩展metadata
    metadata_path = wfo_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata.update(
        {
            "config_overrides": {
                "tx_cost_bps": float(tx_cost_bps),
                "max_turnover": float(max_turnover),
                "target_vol": float(target_vol),
            },
            "gross_backtest_metrics": {
                **gross_metrics,
                "avg_single_turnover": avg_turnover,
            },
            "net_backtest_metrics": {
                **net_before_vol_metrics,
                "avg_single_turnover": avg_turnover,
            },
            # 最终指标（含成本+目标波动）保持兼容为 backtest_metrics
            "backtest_metrics": {
                **final_metrics,
                "avg_single_turnover": avg_turnover,
                "strategy": f"Top-{top_n}因子加权({weight_mode})",
                "holdings_count": top_n,
                "excluded_factors": list(exclude_factors or []),
            },
        }
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 扩展报告
    report_path = wfo_dir / "wfo_report.txt"
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"回测组合指标（Top-{TOP_N_HOLDINGS}因子加权策略）\n")
        f.write("=" * 80 + "\n\n")
        # 毛收益
        f.write("[毛收益]（不含成本/不含波动率目标）\n")
        f.write(f"  总收益率: {gross_metrics['total_return']*100:.2f}%\n")
        f.write(f"  年化收益: {gross_metrics['annualized_return']*100:.2f}%\n")
        f.write(f"  年化波动: {gross_metrics['annualized_volatility']*100:.2f}%\n")
        f.write(f"  夏普比率: {gross_metrics['sharpe_ratio']:.2f}\n")
        f.write(f"  最大回撤: {gross_metrics['max_drawdown']*100:.2f}%\n")
        f.write(f"  胜率: {gross_metrics['win_rate']*100:.2f}%\n")

        # 净收益（含成本，未应用目标波动）
        f.write("\n[净收益]（含成本/不含波动率目标）\n")
        f.write(f"  年化收益: {net_before_vol_metrics['annualized_return']*100:.2f}%\n")
        f.write(
            f"  年化波动: {net_before_vol_metrics['annualized_volatility']*100:.2f}%\n"
        )
        f.write(f"  夏普比率: {net_before_vol_metrics['sharpe_ratio']:.2f}\n")
        f.write(f"  最大回撤: {net_before_vol_metrics['max_drawdown']*100:.2f}%\n")
        f.write(f"  胜率: {net_before_vol_metrics['win_rate']*100:.2f}%\n")

        # 最终（含成本+目标波动）
        f.write("\n[最终]（含成本/含波动率目标）\n")
        f.write(f"  年化收益: {final_metrics['annualized_return']*100:.2f}%\n")
        f.write(f"  年化波动: {final_metrics['annualized_volatility']*100:.2f}%\n")
        f.write(f"  夏普比率: {final_metrics['sharpe_ratio']:.2f}\n")
        f.write(f"  最大回撤: {final_metrics['max_drawdown']*100:.2f}%\n")
        f.write(f"  胜率: {final_metrics['win_rate']*100:.2f}%\n")

        # 其他信息
        f.write("\n")
        f.write(f"平均换手: {avg_turnover*100:.2f}%\n")
        f.write(f"交易天数: {final_metrics['trading_days']}\n")
        f.write(f"持仓数量: {top_n}\n")
        if exclude_factors:
            f.write(f"排除因子: {','.join(exclude_factors)}\n")
        f.write(
            f"配置: tx_cost_bps={tx_cost_bps}, max_turnover={max_turnover}, target_vol={target_vol}\n"
        )
        f.write("\n")

    print("✅ 指标已补充 (含毛/净/最终)：")
    print(
        f"   毛收益-夏普: {gross_metrics['sharpe_ratio']:.2f} | 年化: {gross_metrics['annualized_return']*100:.2f}%"
    )
    print(
        f"   净收益-夏普: {net_before_vol_metrics['sharpe_ratio']:.2f} | 年化: {net_before_vol_metrics['annualized_return']*100:.2f}%"
    )
    print(
        f"   最终  -夏普: {final_metrics['sharpe_ratio']:.2f} | 年化: {final_metrics['annualized_return']*100:.2f}%"
    )
    print(
        f"   平均换手: {avg_turnover*100:.2f}% | 配置: tx_cost_bps={tx_cost_bps}, max_turnover={max_turnover}, target_vol={target_vol}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算WFO回测指标（Top-N因子加权）")
    parser.add_argument(
        "wfo_timestamp_dir",
        type=str,
        help="WFO结果目录，例如 results/wfo/20251028_151333",
    )
    # 支持 --topn 与 --top-n，两者等价；默认值与Step4一致为5
    parser.add_argument(
        "--topn",
        type=int,
        default=TOP_N_HOLDINGS,
        help="Top-N持仓数量，默认5（与Step4一致）",
    )
    parser.add_argument(
        "--top-n", dest="topn", type=int, help="Top-N持仓数量，与 --topn 等价"
    )
    parser.add_argument(
        "--weight-mode",
        type=str,
        default="equal",
        choices=["equal", "ic"],
        help="因子合成权重：等权或IC权重",
    )
    parser.add_argument(
        "--exclude-factors",
        type=str,
        default="",
        help="以逗号分隔的因子名列表，在组合构建时排除不参与合成，例如 'RSI_14,OBV_SLOPE_10D'",
    )
    parser.add_argument(
        "--tx-cost-bps",
        type=float,
        default=0.0,
        help="每次调仓的单向成本（bp），例如5 表示 5bp；默认0=不计成本",
    )
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=1.0,
        help="单期总换手上限（0~1），默认1.0=不限制；例如0.5 表示总换手≤50%",
    )
    parser.add_argument(
        "--target-vol",
        type=float,
        default=0.0,
        help="目标年化波动率（0=关闭），例如0.10 表示目标10%",
    )

    args = parser.parse_args()
    wfo_dir = Path(args.wfo_timestamp_dir)
    if not wfo_dir.exists():
        print(f"❌ 目录不存在: {wfo_dir}")
        sys.exit(1)

    exclude = (
        [s.strip() for s in args.exclude_factors.split(",") if s.strip()]
        if args.exclude_factors
        else []
    )
    main(
        wfo_dir,
        top_n=args.topn,
        exclude_factors=exclude,
        weight_mode=args.weight_mode,
        tx_cost_bps=args.tx_cost_bps,
        max_turnover=args.max_turnover,
        target_vol=args.target_vol,
    )
