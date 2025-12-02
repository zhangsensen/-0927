import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def latest_ts(results_combo_dir: str) -> str:
    subs = [d for d in os.listdir(results_combo_dir) if os.path.isdir(os.path.join(results_combo_dir, d))]
    subs = [s for s in subs if s.isdigit() or '_' in s]
    if not subs:
        raise RuntimeError("No run timestamp directories found")
    subs.sort()
    return subs[-1]


def decile_analysis(df, pred_col, real_col, n_bins=10):
    df = df.copy()
    df["decile"] = pd.qcut(df[pred_col], n_bins, labels=False, duplicates="drop")
    grouped = df.groupby("decile").agg(
        real_mean=(real_col, "mean"),
        real_median=(real_col, "median"),
        real_p20=(real_col, lambda x: np.percentile(x, 20)),
        real_p80=(real_col, lambda x: np.percentile(x, 80)),
        count=(real_col, "size"),
    ).reset_index()
    return grouped


def main():
    base = os.path.dirname(os.path.dirname(__file__))
    results_combo = os.path.join(base, "results_combo_wfo")
    ts = latest_ts(results_combo)
    run_dir = os.path.join(results_combo, ts)

    whitelist = os.path.join(run_dir, f"top5000_wfo8d_whitelist_{ts}.csv")
    ridge_file = os.path.join(run_dir, f"wfo_learned_ranking_8d_refed_ridge_{ts}.csv")

    if not (os.path.exists(whitelist) and os.path.exists(ridge_file)):
        raise FileNotFoundError("Whitelist 或 ridge 排序文件缺失")

    wl = pd.read_csv(whitelist)
    actual_top_n = len(wl)

    # 构造候选回测结果文件名（适配不同topN与命名习惯）
    candidates = [
        os.path.join(run_dir, f"top{actual_top_n}_backtest_8d_{ts}.csv"),
        os.path.join(run_dir, f"top{actual_top_n}_backtest_by_ic_{ts}.csv"),
        os.path.join(run_dir, f"top5000_backtest_8d_{ts}.csv"),
        os.path.join(run_dir, f"top5000_backtest_by_ic_{ts}.csv"),
    ]
    backtest_file = None
    for c in candidates:
        if os.path.exists(c):
            backtest_file = c
            break
    if backtest_file is None:
        # 广泛扫描与实际 topN 数量匹配的文件
        for fn in os.listdir(run_dir):
            if fn.startswith(f"top{actual_top_n}_backtest") and fn.endswith(".csv"):
                backtest_file = os.path.join(run_dir, fn)
                break
    if backtest_file is None:
        raise FileNotFoundError("未找到对应的回测结果CSV文件")

    bt = pd.read_csv(backtest_file)
    ridge = pd.read_csv(ridge_file)[["combo", "pred_8d_sharpe_ridge"]]

    merged = wl.merge(bt, on="combo", how="left").merge(ridge, on="combo", how="left", suffixes=("", "_ridge"))
    merged = merged.dropna(subset=["sharpe", "pred_8d_sharpe_ridge"])

    sp, sp_p = spearmanr(merged["pred_8d_sharpe_ridge"], merged["sharpe"])
    kt, kt_p = kendalltau(merged["pred_8d_sharpe_ridge"], merged["sharpe"])

    # 外推区间 2001-5000（假设原ridge排序列有rank）
    if "rank" in merged.columns:
        oos_part = merged[(merged["rank"] >= 2001) & (merged["rank"] <= 5000)]
    else:
        oos_part = merged.iloc[2000:5000]
    sp_oos, sp_oos_p = spearmanr(oos_part["pred_8d_sharpe_ridge"], oos_part["sharpe"])

    deciles = decile_analysis(merged, "pred_8d_sharpe_ridge", "sharpe")

    summary = {
        "ts": ts,
        "total": int(len(merged)),
        "spearman_all": float(sp),
        "spearman_all_p": float(sp_p),
        "kendall_all": float(kt),
        "kendall_all_p": float(kt_p),
        "spearman_oos_2001_5000": float(sp_oos),
        "spearman_oos_p": float(sp_oos_p),
        "deciles": deciles.to_dict(orient="records"),
    }

    out_json = os.path.join(run_dir, f"top5000_validation_8d_summary_{ts}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary_file": out_json, "spearman_all": sp, "spearman_oos": sp_oos}, ensure_ascii=False))


if __name__ == "__main__":
    main()
