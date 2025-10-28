#!/usr/bin/env python3
"""
Step 4: WFO窗口因子策略回测 - 基于 WFO 因子选择

功能：
1. 加载 WFO 优化结果中选中的因子组合
2. 对每个窗口进行样本外TopN等权多头组合回测
3. 计算收益率、夏普比、最大回撤等性能指标
4. 生成详细的回测报告和日志

输入：
- WFO 结果: wfo/{timestamp}/wfo_results.pkl
- OHLCV 数据: cross_section/{date}/{timestamp}/ohlcv/
- 标准化因子: factor_selection/{date}/{timestamp}/standardized/

输出：
- backtest/{timestamp}/backtest_results.pkl
- backtest/{timestamp}/backtest_report.txt
- backtest/{timestamp}/performance_summary.csv
- backtest/{timestamp}/combination_performance.csv
"""

import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.ic_calculator import ICCalculator


def setup_logging(output_dir: Path):
    """设置详细日志"""
    log_file = output_dir / "step4_backtest.log"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def find_latest_wfo(results_dir: Path):
    """查找最新的 WFO 结果目录"""
    wfo_root = results_dir / "wfo"

    if not wfo_root.exists():
        return None

    # 查找所有时间戳目录
    all_runs = []
    for timestamp_dir in wfo_root.iterdir():
        if not timestamp_dir.is_dir():
            continue
        # 验证是否包含必要文件
        if (timestamp_dir / "wfo_results.pkl").exists():
            all_runs.append(timestamp_dir)

    if not all_runs:
        return None

    # 按时间戳排序，返回最新
    all_runs.sort(key=lambda p: p.name, reverse=True)
    return all_runs[0]


def load_wfo_results(wfo_dir: Path, logger):
    """加载 WFO 结果"""
    logger.info("-" * 80)
    logger.info("阶段 1/4: 加载 WFO 优化结果")
    logger.info("-" * 80)

    # 加载 WFO 结果对象
    wfo_results_path = wfo_dir / "wfo_results.pkl"
    with open(wfo_results_path, "rb") as f:
        wfo_results = pickle.load(f)

    logger.info(f"✅ WFO 结果已加载: {wfo_results_path}")
    logger.info(f"  - 总窗口数: {wfo_results['total_windows']}")
    logger.info(f"  - 有效窗口数: {wfo_results['valid_windows']}")
    logger.info(f"  - 平均OOS IC: {wfo_results['avg_oos_ic']:.4f}")
    logger.info("")

    return wfo_results


def load_backtest_data(wfo_dir: Path, results_dir: Path, logger):
    """加载回测所需的数据"""
    logger.info("-" * 80)
    logger.info("阶段 2/4: 加载回测数据（OHLCV + 标准化因子）")
    logger.info("-" * 80)

    # 加载 WFO metadata 来获取数据路径信息
    wfo_metadata_path = wfo_dir / "metadata.json"
    with open(wfo_metadata_path) as f:
        wfo_meta = json.load(f)

    # 从 WFO 元数据中提取时间戳与日期，用于绑定数据版本
    wfo_ts = str(wfo_meta.get("timestamp", wfo_dir.name))
    wfo_date = wfo_ts.split("_")[0] if "_" in wfo_ts else wfo_ts[:8]
    logger.info(f"绑定数据版本：目标日期={wfo_date}，WFO时间戳={wfo_ts}")

    # 优先按 WFO 时间戳绑定 cross_section 数据（若缺失则回退到最新，并给出告警）
    cross_section_root = results_dir / "cross_section"
    bound_cross_section: Path = None
    try:
        target_date_dir = cross_section_root / wfo_date
        candidates = []
        if target_date_dir.exists():
            for ts_dir in target_date_dir.iterdir():
                if ts_dir.is_dir() and (ts_dir / "metadata.json").exists():
                    # 仅选择时间戳不晚于 WFO 的目录，避免跨版本错配
                    if ts_dir.name <= wfo_ts:
                        candidates.append(ts_dir)
        if candidates:
            candidates.sort(key=lambda p: p.name, reverse=True)
            bound_cross_section = candidates[0]
            logger.info(f"✅ 已按WFO绑定 cross_section 目录: {bound_cross_section}")
        else:
            # 回退：全局最新可用
            all_cross_section = []
            for date_dir in cross_section_root.iterdir():
                if not date_dir.is_dir():
                    continue
                for ts_dir in date_dir.iterdir():
                    if ts_dir.is_dir() and (ts_dir / "metadata.json").exists():
                        all_cross_section.append(ts_dir)
            all_cross_section.sort(key=lambda p: p.name, reverse=True)
            bound_cross_section = all_cross_section[0]
            logger.warning(
                f"⚠️ 未找到与WFO({wfo_ts})同日且不晚于它的 cross_section 版本，已回退至最新: {bound_cross_section}"
            )
    except Exception as e:
        logger.warning(f"⚠️ 绑定 cross_section 版本时出错：{e}，将回退到最新可用版本")
        # 兜底：原有逻辑，选择全局最新
        all_cross_section = []
        for date_dir in cross_section_root.iterdir():
            if not date_dir.is_dir():
                continue
            for ts_dir in date_dir.iterdir():
                if ts_dir.is_dir() and (ts_dir / "metadata.json").exists():
                    all_cross_section.append(ts_dir)
        all_cross_section.sort(key=lambda p: p.name, reverse=True)
        bound_cross_section = all_cross_section[0]

    cross_section_dir = bound_cross_section

    # 加载 OHLCV
    ohlcv_data = {}
    ohlcv_dir = cross_section_dir / "ohlcv"
    for col in ["open", "high", "low", "close", "volume"]:
        ohlcv_data[col] = pd.read_parquet(ohlcv_dir / f"{col}.parquet")

    # 优先按 WFO 时间戳绑定 factor_selection 数据（若缺失则回退到最新，并给出告警）
    factor_sel_root = results_dir / "factor_selection"
    bound_factor_sel: Path = None
    try:
        target_date_dir = factor_sel_root / wfo_date
        candidates = []
        if target_date_dir.exists():
            for ts_dir in target_date_dir.iterdir():
                if ts_dir.is_dir() and (ts_dir / "metadata.json").exists():
                    if ts_dir.name <= wfo_ts:
                        candidates.append(ts_dir)
        if candidates:
            candidates.sort(key=lambda p: p.name, reverse=True)
            bound_factor_sel = candidates[0]
            logger.info(f"✅ 已按WFO绑定 factor_selection 目录: {bound_factor_sel}")
        else:
            # 回退：全局最新可用
            all_factor_sel = []
            for date_dir in factor_sel_root.iterdir():
                if not date_dir.is_dir():
                    continue
                for ts_dir in date_dir.iterdir():
                    if ts_dir.is_dir() and (ts_dir / "metadata.json").exists():
                        all_factor_sel.append(ts_dir)
            all_factor_sel.sort(key=lambda p: p.name, reverse=True)
            bound_factor_sel = all_factor_sel[0]
            logger.warning(
                f"⚠️ 未找到与WFO({wfo_ts})同日且不晚于它的 factor_selection 版本，已回退至最新: {bound_factor_sel}"
            )
    except Exception as e:
        logger.warning(f"⚠️ 绑定 factor_selection 版本时出错：{e}，将回退到最新可用版本")
        # 兜底：原有逻辑，选择全局最新
        all_factor_sel = []
        for date_dir in factor_sel_root.iterdir():
            if not date_dir.is_dir():
                continue
            for ts_dir in date_dir.iterdir():
                if ts_dir.is_dir() and (ts_dir / "metadata.json").exists():
                    all_factor_sel.append(ts_dir)
        all_factor_sel.sort(key=lambda p: p.name, reverse=True)
        bound_factor_sel = all_factor_sel[0]

    factor_sel_dir = bound_factor_sel

    # 加载标准化因子
    std_factors = {}
    std_dir = factor_sel_dir / "standardized"
    for factor_file in std_dir.glob("*.parquet"):
        factor_name = factor_file.stem
        std_factors[factor_name] = pd.read_parquet(factor_file)

    logger.info(f"✅ OHLCV 数据已加载: {ohlcv_data['close'].shape}")
    logger.info(f"✅ 标准化因子已加载: {len(std_factors)} 个因子")
    logger.info(f"  - 因子: {', '.join(list(std_factors.keys())[:5])}...")
    logger.info("")

    return ohlcv_data, std_factors, wfo_meta


def run_backtest_combinations(
    wfo_results: Dict,
    ohlcv_data: Dict,
    std_factors: Dict,
    wfo_meta: Dict,
    logger,
) -> Tuple[pd.DataFrame, Dict]:
    """
    运行所有窗口的组合回测

    对每个 WFO 窗口中选中的因子组合进行回测，计算性能指标
    """
    logger.info("-" * 80)
    logger.info("阶段 3/4: 运行 WFO 窗口因子策略回测（TopN=5 日频多头）")
    logger.info("-" * 80)
    logger.info("")

    # 提取 WFO 结果
    results_df = wfo_results["results_df"]
    constraint_reports = wfo_results["constraint_reports"]
    total_windows = len(constraint_reports)

    # 准备回测结果容器
    backtest_records = []

    # 获取 close 价格用于计算收益
    close_prices = ohlcv_data["close"]
    returns = close_prices.pct_change(fill_method=None)  # (1399, 43) 修复FutureWarning

    # 说明：横截面IC在下方使用scipy直接计算，若需扩展可切换为ICCalculator

    # 全局参数
    TOPN = 5  # 每日选取Top5资产
    COST_BPS = float(os.getenv("TRADING_COST_BPS", "0"))  # 交易成本（单边，bps）
    COST_RATE = COST_BPS / 10000.0

    # 用于拼接非重叠 OOS 权益曲线
    # 估计步长（步进天数），用于拼接每个窗口的前 step_len 天
    try:
        oos_starts = [int(r.oos_start) for r in constraint_reports]
        diffs = [b - a for a, b in zip(oos_starts[:-1], oos_starts[1:]) if (b - a) > 0]
        step_len = int(min(diffs)) if diffs else None
    except Exception:
        step_len = None
    if step_len is None:
        step_len = 20  # 回退：默认 20 天
    logger.info(f"拼接非重叠 OOS 权益的步长(step_len)={step_len}")

    stitched_rows: List[Dict] = []

    # 对每个窗口进行回测
    for window_idx, report in enumerate(constraint_reports, 1):
        is_end = report.is_end
        oos_start = report.oos_start
        oos_end = report.oos_end
        selected_factors = report.selected_factors

        # DEBUG: 打印因子列表
        logger.info(f"[窗口 {window_idx}/{total_windows}] 选中因子: {selected_factors}")

        if not selected_factors:
            logger.info(f"[窗口 {window_idx}/{total_windows}] 无选中因子，跳过")
            continue

        # ========== 核心修复：基于因子信号的TopN选股 ==========

        # 准备OOS期的日收益率
        oos_returns = returns.iloc[oos_start:oos_end]  # (60, 43)
        n_oos_days = len(oos_returns)

        # 准备因子数据：需要在OOS开始前一天到OOS结束前一天（用于T-1预测T日）
        # 因子索引范围：[oos_start-1, oos_end-1)
        factor_start = max(0, oos_start - 1)
        factor_end = max(1, oos_end - 1)

        # 获取选中因子的标准化数据
        factor_signals = []
        for factor_name in selected_factors:
            if factor_name not in std_factors:
                continue
            factor_data = std_factors[factor_name]
            # 取T-1到T-1+N-1的因子值（用于预测T到T+N）
            factor_slice = factor_data.iloc[factor_start:factor_end]
            factor_signals.append(factor_slice.values)

        if not factor_signals:
            logger.info(f"[窗口 {window_idx}/{total_windows}] 无可用因子数据，跳过")
            continue

        # 等权平均多因子信号 (n_days, 43)
        combined_signal = np.nanmean(factor_signals, axis=0)

        # 逐日TopN选股并计算组合收益与换手/净值
        portfolio_daily_returns = []
        net_daily_returns = []
        daily_turnovers = []
        n_assets = oos_returns.shape[1]
        prev_weights = np.zeros(n_assets, dtype=float)

        for day_idx in range(n_oos_days):
            # T日的收益率
            day_returns = oos_returns.iloc[day_idx].values  # (43,)

            # T-1日的因子信号（已经在combined_signal中对齐）
            if day_idx < len(combined_signal):
                day_signal = combined_signal[day_idx]  # (43,)
            else:
                # 边界情况：用最后一个信号（不应该发生）
                day_signal = combined_signal[-1]

            # 找出有效的（非NaN）因子值和收益率
            valid_mask = ~(np.isnan(day_signal) | np.isnan(day_returns))
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                # 当日无有效数据，组合收益为0
                portfolio_daily_returns.append(0.0)
                continue

            # 对有效资产按因子值排序，选TopN
            valid_signals = day_signal[valid_indices]
            valid_rets = day_returns[valid_indices]

            # 降序排列（因子值越大越好）
            sorted_idx = np.argsort(-valid_signals)
            topn_count = min(TOPN, len(sorted_idx))
            topn_idx = sorted_idx[:topn_count]

            # TopN等权组合收益
            topn_returns = valid_rets[topn_idx]
            portfolio_ret = np.mean(topn_returns)
            portfolio_daily_returns.append(portfolio_ret)

            # 计算当日目标权重（全市场维度）
            selected_global_idx = valid_indices[topn_idx]  # 相对全资产的索引
            weights = np.zeros(n_assets, dtype=float)
            if len(selected_global_idx) > 0:
                weights[selected_global_idx] = 1.0 / len(selected_global_idx)

            # 计算当日换手率（单边）：0.5 * L1范数
            turnover = 0.5 * np.sum(np.abs(weights - prev_weights))
            daily_turnovers.append(float(turnover))

            # 扣除成本后的净收益（单边成本）
            net_ret = portfolio_ret - turnover * COST_RATE
            net_daily_returns.append(net_ret)

            # 更新昨日权重
            prev_weights = weights

        portfolio_daily_returns = np.array(portfolio_daily_returns)
        net_daily_returns = np.array(net_daily_returns)
        daily_turnovers = np.array(daily_turnovers) if daily_turnovers else np.array([])

        # ========== 计算横截面IC（每日IC的均值）==========
        daily_ics = []
        for day_idx in range(n_oos_days):
            day_returns = oos_returns.iloc[day_idx].values
            if day_idx < len(combined_signal):
                day_signal = combined_signal[day_idx]
            else:
                day_signal = combined_signal[-1]

            # 计算横截面Spearman相关
            valid_mask = ~(np.isnan(day_signal) | np.isnan(day_returns))
            if valid_mask.sum() < 2:
                continue

            from scipy.stats import spearmanr

            ic, _ = spearmanr(day_signal[valid_mask], day_returns[valid_mask])
            if not np.isnan(ic):
                daily_ics.append(ic)

        avg_oos_ic = np.mean(daily_ics) if daily_ics else 0.0

        # ========== 性能指标计算 ==========
        # 累计收益率（毛/净）
        total_return = (1 + portfolio_daily_returns).prod() - 1
        net_total_return = (1 + net_daily_returns).prod() - 1

        # 样本期天数
        n_days = len(portfolio_daily_returns)

        # 波动率：日标准差（毛/净）
        daily_vol = portfolio_daily_returns.std()
        net_daily_vol = net_daily_returns.std()

        # 统一口径：全部按年化口径报告（并同时保留期间收益字段，避免歧义）
        annual_return = (1 + total_return) ** (252 / max(1, n_days)) - 1
        annual_vol = daily_vol * np.sqrt(252)
        sharpe = (portfolio_daily_returns.mean() / (daily_vol + 1e-6)) * np.sqrt(252)

        net_annual_return = (1 + net_total_return) ** (252 / max(1, n_days)) - 1
        net_annual_vol = net_daily_vol * np.sqrt(252)
        net_sharpe = (net_daily_returns.mean() / (net_daily_vol + 1e-6)) * np.sqrt(252)

        # 最大回撤：基于累计净值
        cumulative_returns = (1 + portfolio_daily_returns).cumprod()
        running_max = cumulative_returns.copy()
        for i in range(1, len(running_max)):
            running_max[i] = max(running_max[i], running_max[i - 1])
        drawdown = (cumulative_returns - running_max) / running_max
        max_dd = drawdown.min()

        # 净值最大回撤
        net_cum = (1 + net_daily_returns).cumprod()
        net_running_max = net_cum.copy()
        for i in range(1, len(net_running_max)):
            net_running_max[i] = max(net_running_max[i], net_running_max[i - 1])
        net_dd = (net_cum - net_running_max) / net_running_max
        net_max_dd = net_dd.min()

        # 记录结果（同时保留期间口径以便二次分析）
        record = {
            "window_idx": window_idx,
            "is_start": report.is_start,
            "is_end": report.is_end,
            "oos_start": report.oos_start,
            "oos_end": report.oos_end,
            "factor_count": len(selected_factors),
            "selected_factors": "|".join(selected_factors),
            "avg_oos_ic": avg_oos_ic,
            "oos_annual_return": annual_return,
            "oos_annual_vol": annual_vol,
            "oos_sharpe": sharpe,
            "oos_max_dd": max_dd,
            "oos_period_return": total_return,  # 期间收益（未年化）
            "oos_total_return": total_return,  # 向后兼容（等同于期间收益）
            # 净值相关
            "oos_net_period_return": net_total_return,
            "oos_net_annual_return": net_annual_return,
            "oos_net_annual_vol": net_annual_vol,
            "oos_net_sharpe": net_sharpe,
            "oos_net_max_dd": net_max_dd,
            # 成本与换手
            "avg_daily_turnover": (
                float(daily_turnovers.mean()) if len(daily_turnovers) else 0.0
            ),
            "cost_bps": COST_BPS,
        }
        backtest_records.append(record)

        if window_idx % 10 == 0 or window_idx == total_windows:
            logger.info(
                f"[窗口 {window_idx}/{total_windows}] "
                f"IC={avg_oos_ic:.4f} Sharpe={sharpe:.4f} AnnRet={annual_return:.4f}"
            )

        # 记录用于拼接非重叠 OOS 权益的前 step_len 天
        take_n = min(step_len, n_oos_days)
        if take_n > 0:
            dates_part = oos_returns.index[:take_n]
            gross_part = portfolio_daily_returns[:take_n]
            net_part = net_daily_returns[:take_n]
            for dt, g, n in zip(dates_part, gross_part, net_part):
                stitched_rows.append(
                    {
                        "date": str(dt),
                        "window_idx": window_idx,
                        "gross_ret": float(g),
                        "net_ret": float(n),
                    }
                )

    logger.info(f"\n✅ 回测完成: {len(backtest_records)} 个窗口")
    logger.info("")

    backtest_df = pd.DataFrame(backtest_records)

    # 生成拼接后的权益曲线
    stitched_df = pd.DataFrame(stitched_rows)
    if not stitched_df.empty:
        stitched_df["cum_gross"] = (1 + stitched_df["gross_ret"]).cumprod()
        stitched_df["cum_net"] = (1 + stitched_df["net_ret"]).cumprod()
    extras = {"stitched_oos": stitched_df}
    return backtest_df, extras


def save_backtest_results(
    backtest_df: pd.DataFrame, output_dir: Path, logger, extras: Dict = None
):
    """保存回测结果"""
    logger.info("-" * 80)
    logger.info("阶段 4/4: 保存回测结果")
    logger.info("-" * 80)

    # 性能摘要（删除不科学的total_return统计）
    performance_summary = {
        "total_windows": len(backtest_df),
        "avg_ic": backtest_df["avg_oos_ic"].mean(),
        "avg_sharpe": backtest_df["oos_sharpe"].mean(),
        "avg_annual_return": backtest_df["oos_annual_return"].mean(),
        "avg_annual_vol": backtest_df["oos_annual_vol"].mean(),
        "avg_max_dd": backtest_df["oos_max_dd"].mean(),
        # 净值摘要
        "avg_net_sharpe": (
            backtest_df.get("oos_net_sharpe", pd.Series(dtype=float)).mean()
            if "oos_net_sharpe" in backtest_df.columns
            else None
        ),
        "avg_net_annual_return": (
            backtest_df.get("oos_net_annual_return", pd.Series(dtype=float)).mean()
            if "oos_net_annual_return" in backtest_df.columns
            else None
        ),
        "avg_net_annual_vol": (
            backtest_df.get("oos_net_annual_vol", pd.Series(dtype=float)).mean()
            if "oos_net_annual_vol" in backtest_df.columns
            else None
        ),
        "avg_net_max_dd": (
            backtest_df.get("oos_net_max_dd", pd.Series(dtype=float)).mean()
            if "oos_net_max_dd" in backtest_df.columns
            else None
        ),
        # 成本/换手
        "avg_daily_turnover": (
            backtest_df.get("avg_daily_turnover", pd.Series(dtype=float)).mean()
            if "avg_daily_turnover" in backtest_df.columns
            else None
        ),
    }

    # 保存组合性能到 CSV
    combo_csv = output_dir / "combination_performance.csv"
    backtest_df.to_csv(combo_csv, index=False)
    logger.info(f"✅ 组合性能已保存: {combo_csv}")

    # 保存性能摘要
    summary_csv = output_dir / "performance_summary.csv"
    pd.DataFrame([performance_summary]).to_csv(summary_csv, index=False)
    logger.info(f"✅ 性能摘要已保存: {summary_csv}")

    # 生成详细报告文本
    report_path = output_dir / "backtest_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        lines = []
        lines.append("=" * 80 + "\n")
        lines.append("WFO窗口因子策略回测详细报告\n")
        lines.append("=" * 80 + "\n\n")

        lines.append("性能摘要\n")
        lines.append("-" * 80 + "\n")
        lines.append(f"总窗口数: {performance_summary['total_windows']}\n")
        lines.append(f"平均 OOS IC (横截面): {performance_summary['avg_ic']:.6f}\n")
        lines.append(f"平均夏普比: {performance_summary['avg_sharpe']:.4f}\n")
        lines.append(f"平均年化收益: {performance_summary['avg_annual_return']:.4f}\n")
        lines.append(f"平均年化波动: {performance_summary['avg_annual_vol']:.4f}\n")
        lines.append(f"平均最大回撤: {performance_summary['avg_max_dd']:.4f}\n")
        if performance_summary.get("avg_net_annual_return") is not None:
            lines.append(
                f"平均净年化收益: {performance_summary['avg_net_annual_return']:.4f}\n"
            )
        if performance_summary.get("avg_net_annual_vol") is not None:
            lines.append(
                f"平均净年化波动: {performance_summary['avg_net_annual_vol']:.4f}\n"
            )
        if performance_summary.get("avg_net_sharpe") is not None:
            lines.append(f"平均净夏普比: {performance_summary['avg_net_sharpe']:.4f}\n")
        if performance_summary.get("avg_net_max_dd") is not None:
            lines.append(
                f"平均净最大回撤: {performance_summary['avg_net_max_dd']:.4f}\n"
            )
        if performance_summary.get("avg_daily_turnover") is not None:
            lines.append(
                f"平均日换手: {performance_summary['avg_daily_turnover']:.4f}\n"
            )
        lines.append("\n")

        # TOP 10 窗口（按 Sharpe）
        lines.append("TOP 10 窗口（按夏普比）\n")
        lines.append("-" * 80 + "\n")
        top10 = backtest_df.nlargest(10, "oos_sharpe")
        for _, row in top10.iterrows():
            lines.append(
                f"窗口 {row['window_idx']}: "
                f"Sharpe={row['oos_sharpe']:.4f} "
                f"AnnReturn={row['oos_annual_return']:.4f} "
                f"AnnVol={row['oos_annual_vol']:.4f} "
                f"IC={row['avg_oos_ic']:.6f}\n"
            )
        lines.append("\n")

        # 统计分布
        lines.append("统计分布\n")
        lines.append("-" * 80 + "\n")
        lines.append(
            f"IC 范围: [{backtest_df['avg_oos_ic'].min():.6f}, "
            f"{backtest_df['avg_oos_ic'].max():.6f}]\n"
        )
        lines.append(
            f"Sharpe 范围: [{backtest_df['oos_sharpe'].min():.4f}, "
            f"{backtest_df['oos_sharpe'].max():.4f}]\n"
        )
        lines.append(
            f"年化收益范围: [{backtest_df['oos_annual_return'].min():.4f}, "
            f"{backtest_df['oos_annual_return'].max():.4f}]\n\n"
        )

        f.writelines(lines)

    logger.info(f"✅ 详细报告已保存: {report_path}")
    logger.info("")

    # 额外产物：拼接的非重叠 OOS 权益曲线
    if extras and isinstance(extras.get("stitched_oos"), pd.DataFrame):
        stitched: pd.DataFrame = extras["stitched_oos"]
        stitched_path = output_dir / "stitched_oos_equity.csv"
        stitched.to_csv(stitched_path, index=False)
        logger.info(f"✅ 非重叠OOS权益已保存: {stitched_path}")


def main(wfo_dir: Path = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 输出目录
    output_root = PROJECT_ROOT / "results"
    backtest_dir = output_root / "backtest" / timestamp
    backtest_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(backtest_dir)

    logger.info("=" * 80)
    logger.info("Step 4: WFO窗口因子策略回测")
    logger.info("=" * 80)
    logger.info(f"输出目录: {backtest_dir}")
    logger.info(f"时间戳: {timestamp}")
    logger.info("")

    # 查找 WFO 结果
    if wfo_dir is None:
        logger.info("🔍 自动查找最新的 WFO 结果...")
        wfo_dir = find_latest_wfo(output_root)

        if wfo_dir is None:
            logger.error("❌ 未找到 WFO 结果！请先运行 step3_run_wfo.py")
            sys.exit(1)

        logger.info(f"✅ 找到最新 WFO 结果: {wfo_dir}")
        logger.info("")

    # 1. 加载 WFO 结果
    wfo_results = load_wfo_results(wfo_dir, logger)

    # 2. 加载回测数据
    ohlcv_data, std_factors, wfo_meta = load_backtest_data(wfo_dir, output_root, logger)

    # 3. 运行回测
    backtest_df, extras = run_backtest_combinations(
        wfo_results, ohlcv_data, std_factors, wfo_meta, logger
    )

    # 4. 保存结果
    save_backtest_results(backtest_df, backtest_dir, logger, extras)

    # 完成
    logger.info("=" * 80)
    logger.info("✅ Step 4 完成！WFO窗口因子策略回测已执行")
    logger.info("=" * 80)
    logger.info(f"输出目录: {backtest_dir}")
    logger.info(f"  - combination_performance.csv: 窗口性能数据")
    logger.info(f"  - performance_summary.csv: 性能摘要")
    logger.info(f"  - backtest_report.txt: 详细报告")
    logger.info(f"  - step4_backtest.log: 执行日志")
    logger.info("")


if __name__ == "__main__":
    main()
