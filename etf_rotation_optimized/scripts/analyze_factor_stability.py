#!/usr/bin/env python3
"""
P1: 因子稳定性分析

读取最新WFO结果，按每个窗口计算每个因子的IS/OOS横截面Spearman IC（T-1→T对齐），
输出两份CSV：
- factor_ic_details.csv: 每窗×因子 明细（is_ic, oos_ic, decay_ratio, selected_flag）
- factor_stability.csv: 按因子聚合（均值/中位数/失败窗口数等）

用法：
  python etf_rotation_optimized/scripts/analyze_factor_stability.py

备注：
- 数据绑定策略与Step4一致：按WFO metadata 中的 timestamp 绑定 cross_section 与 factor_selection 版本；
  找不到同日且不晚于WFO的版本则回退到最新并打印警告。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _setup_logger():
    import logging

    logger = logging.getLogger("factor_stability")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _find_latest_wfo(results_dir: Path) -> Path | None:
    wfo_root = results_dir / "wfo"
    if not wfo_root.exists():
        return None
    runs = [
        p for p in wfo_root.iterdir() if p.is_dir() and (p / "wfo_results.pkl").exists()
    ]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs[0] if runs else None


def _bind_data_version(wfo_dir: Path, results_dir: Path, logger):
    # 读取WFO元数据
    meta_path = wfo_dir / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        wfo_meta = json.load(f)

    wfo_ts = str(wfo_meta.get("timestamp", wfo_dir.name))
    wfo_date = wfo_ts.split("_")[0] if "_" in wfo_ts else wfo_ts[:8]
    logger.info(f"绑定数据版本：目标日期={wfo_date}，WFO时间戳={wfo_ts}")

    # 绑定 cross_section
    cross_root = results_dir / "cross_section"
    bound_cs = None
    target_dir = cross_root / wfo_date
    cands = []
    if target_dir.exists():
        for ts_dir in target_dir.iterdir():
            if (
                ts_dir.is_dir()
                and (ts_dir / "metadata.json").exists()
                and ts_dir.name <= wfo_ts
            ):
                cands.append(ts_dir)
    if cands:
        cands.sort(key=lambda p: p.name, reverse=True)
        bound_cs = cands[0]
        logger.info(f"✅ 已按WFO绑定 cross_section: {bound_cs}")
    else:
        all_cs = []
        for d in cross_root.iterdir():
            if not d.is_dir():
                continue
            for ts_dir in d.iterdir():
                if ts_dir.is_dir() and (ts_dir / "metadata.json").exists():
                    all_cs.append(ts_dir)
        all_cs.sort(key=lambda p: p.name, reverse=True)
        bound_cs = all_cs[0]
        logger.warning(f"⚠️ 未找到同日≤WFO的 cross_section，回退最新: {bound_cs}")

    # 绑定 factor_selection
    sel_root = results_dir / "factor_selection"
    bound_sel = None
    target_dir = sel_root / wfo_date
    cands = []
    if target_dir.exists():
        for ts_dir in target_dir.iterdir():
            if (
                ts_dir.is_dir()
                and (ts_dir / "metadata.json").exists()
                and ts_dir.name <= wfo_ts
            ):
                cands.append(ts_dir)
    if cands:
        cands.sort(key=lambda p: p.name, reverse=True)
        bound_sel = cands[0]
        logger.info(f"✅ 已按WFO绑定 factor_selection: {bound_sel}")
    else:
        all_sel = []
        for d in sel_root.iterdir():
            if not d.is_dir():
                continue
            for ts_dir in d.iterdir():
                if ts_dir.is_dir() and (ts_dir / "metadata.json").exists():
                    all_sel.append(ts_dir)
        all_sel.sort(key=lambda p: p.name, reverse=True)
        bound_sel = all_sel[0]
        logger.warning(f"⚠️ 未找到同日≤WFO的 factor_selection，回退最新: {bound_sel}")

    return bound_cs, bound_sel, wfo_meta


def _load_data(
    bound_cs: Path, bound_sel: Path
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    # OHLCV
    ohlcv_dir = bound_cs / "ohlcv"
    ohlcv = {
        k: pd.read_parquet(ohlcv_dir / f"{k}.parquet")
        for k in ["open", "high", "low", "close", "volume"]
    }
    # 标准化因子
    std_dir = bound_sel / "standardized"
    std_factors: Dict[str, pd.DataFrame] = {}
    for fp in std_dir.glob("*.parquet"):
        std_factors[fp.stem] = pd.read_parquet(fp)
    return ohlcv, std_factors


def _compute_cs_ic(factor_mat: np.ndarray, ret_mat: np.ndarray) -> float:
    """日频横截面Spearman IC的日均值（T-1→T对齐已在调用处保证切片对应）。"""
    from scipy.stats import spearmanr

    n_days, n_assets = ret_mat.shape
    ics: List[float] = []
    for t in range(n_days):
        s = factor_mat[t, :]
        r = ret_mat[t, :]
        valid = ~(np.isnan(s) | np.isnan(r))
        if valid.sum() < 2:
            continue
        ic, _ = spearmanr(s[valid], r[valid])
        if not np.isnan(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else 0.0


def main():
    logger = _setup_logger()
    results_root = PROJECT_ROOT / "results"

    wfo_dir = _find_latest_wfo(results_root)
    if wfo_dir is None:
        logger.error("❌ 未找到WFO结果，请先运行 step3_run_wfo.py")
        sys.exit(1)

    # 读取WFO对象
    import pickle

    with open(wfo_dir / "wfo_results.pkl", "rb") as f:
        wfo_results = pickle.load(f)

    # 绑定版本并加载数据
    bound_cs, bound_sel, wfo_meta = _bind_data_version(wfo_dir, results_root, logger)
    ohlcv, std_factors = _load_data(bound_cs, bound_sel)

    close = ohlcv["close"]
    rets = close.pct_change(fill_method=None)

    reports = wfo_results.get("constraint_reports", [])
    if not reports:
        logger.error("❌ WFO结果缺少 constraint_reports，无法进行稳定性分析")
        sys.exit(1)

    factor_names = sorted(std_factors.keys())

    details_rows: List[Dict[str, object]] = []

    for w_idx, rep in enumerate(reports, 1):
        is_start, is_end = int(rep.is_start), int(rep.is_end)
        oos_start, oos_end = int(rep.oos_start), int(rep.oos_end)

        # T-1 对齐的切片范围（长度等于天数）
        is_len = max(0, is_end - is_start)
        oos_len = max(0, oos_end - oos_start)
        if is_len == 0 or oos_len == 0:
            continue

        is_factor_slice = slice(max(0, is_start - 1), max(1, is_end - 1))
        oos_factor_slice = slice(max(0, oos_start - 1), max(1, oos_end - 1))
        is_ret_slice = slice(is_start, is_end)
        oos_ret_slice = slice(oos_start, oos_end)

        is_rets = rets.iloc[is_ret_slice].values
        oos_rets = rets.iloc[oos_ret_slice].values

        selected_set = (
            set(rep.selected_factors) if hasattr(rep, "selected_factors") else set()
        )

        for fname in factor_names:
            if fname not in std_factors:
                continue
            fdf = std_factors[fname]
            is_sig = fdf.iloc[is_factor_slice].values
            oos_sig = fdf.iloc[oos_factor_slice].values

            # 对齐后的矩阵维度应分别为 (is_len, n_assets) 与 (oos_len, n_assets)
            # 但若最早一天缺失导致长度不等，则截短到一致长度
            is_n = min(len(is_sig), len(is_rets))
            oos_n = min(len(oos_sig), len(oos_rets))
            if is_n <= 0 or oos_n <= 0:
                continue

            is_ic = _compute_cs_ic(is_sig[:is_n], is_rets[:is_n])
            oos_ic = _compute_cs_ic(oos_sig[:oos_n], oos_rets[:oos_n])
            decay = (oos_ic / is_ic) if (is_ic != 0) else np.nan

            details_rows.append(
                {
                    "window_idx": w_idx,
                    "factor": fname,
                    "is_ic": is_ic,
                    "oos_ic": oos_ic,
                    "decay_ratio": decay,
                    "selected": int(fname in selected_set),
                }
            )

    details_df = pd.DataFrame(details_rows)
    if details_df.empty:
        logger.error("❌ 未生成任何IC明细，请检查输入数据与窗口设置")
        sys.exit(2)

    # 聚合统计
    def _safe_mean(x: pd.Series) -> float:
        return (
            float(pd.to_numeric(x, errors="coerce").mean()) if len(x) else float("nan")
        )

    agg = (
        details_df.groupby("factor")
        .agg(
            windows_count=("window_idx", "nunique"),
            mean_is_ic=("is_ic", _safe_mean),
            mean_oos_ic=("oos_ic", _safe_mean),
            median_is_ic=("is_ic", "median"),
            median_oos_ic=("oos_ic", "median"),
            mean_decay_ratio=("decay_ratio", _safe_mean),
            fail_windows=(
                "decay_ratio",
                lambda s: int((pd.to_numeric(s, errors="coerce") < 0.2).sum()),
            ),
            selected_freq=("selected", _safe_mean),
        )
        .reset_index()
        .sort_values(["mean_decay_ratio", "mean_oos_ic"])
        .reset_index(drop=True)
    )

    # 输出
    out_details = wfo_dir / "factor_ic_details.csv"
    out_summary = wfo_dir / "factor_stability.csv"
    details_df.to_csv(out_details, index=False)
    agg.to_csv(out_summary, index=False)

    logger.info(f"✅ 因子IC明细已保存: {out_details}")
    logger.info(f"✅ 因子稳定性汇总已保存: {out_summary}")

    # 打印Top失败因子
    worst = agg.sort_values(["mean_decay_ratio", "mean_oos_ic"]).head(8)
    logger.info("\n🔎 衰减严重的因子（Top 8）:")
    for _, r in worst.iterrows():
        logger.info(
            f"  {r['factor']}: mean(IS)={r['mean_is_ic']:.4f}  mean(OOS)={r['mean_oos_ic']:.4f}  decay={r['mean_decay_ratio']:.3f}  sel_freq={r['selected_freq']:.2f}"
        )


if __name__ == "__main__":
    main()
