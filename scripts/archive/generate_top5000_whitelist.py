import os
import sys
import json
import pandas as pd


def latest_ts(results_combo_dir: str) -> str:
    subs = [d for d in os.listdir(results_combo_dir) if os.path.isdir(os.path.join(results_combo_dir, d))]
    if not subs:
        raise RuntimeError("No subdirectories in results_combo_wfo")
    subs.sort()
    return subs[-1]


def main():
    base = os.path.dirname(os.path.dirname(__file__))  # etf_strategy
    results_combo = os.path.join(base, "results_combo_wfo")
    ts = latest_ts(results_combo)

    ridge_csv = os.path.join(results_combo, ts, f"wfo_learned_ranking_8d_refed_ridge_{ts}.csv")
    if not os.path.exists(ridge_csv):
        raise FileNotFoundError(ridge_csv)

    df = pd.read_csv(ridge_csv)
    # 必选列: combo, pred_8d_sharpe_ridge
    if "combo" not in df.columns or "pred_8d_sharpe_ridge" not in df.columns:
        raise RuntimeError("Required columns not found in ridge csv")

    topk = 5000
    out = df.sort_values("pred_8d_sharpe_ridge", ascending=False).head(topk)[["combo", "pred_8d_sharpe_ridge"]].reset_index(drop=True)

    out_path = os.path.join(results_combo, ts, f"top5000_wfo8d_whitelist_{ts}.csv")
    out.to_csv(out_path, index=False)
    print(json.dumps({"ts": ts, "whitelist": out_path, "n": len(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
