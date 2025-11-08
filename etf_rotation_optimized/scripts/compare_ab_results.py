import sys
import pandas as pd


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/compare_ab_results.py <baseline_csv> <optimized_csv>")
        sys.exit(1)

    f_base = sys.argv[1]
    f_opt = sys.argv[2]

    a = pd.read_csv(f_base)
    b = pd.read_csv(f_opt)

    key = "combo"
    cols = ["sharpe", "annual_ret", "max_dd", "win_rate", "n_rebalance", "avg_turnover"]

    m = a[[key] + cols].merge(b[[key] + cols], on=key, suffixes=("_base", "_opt"), how="inner")
    if m.empty:
        print("ERROR: no overlap between files")
        sys.exit(2)

    diffs = {}
    for c in ["sharpe", "annual_ret", "max_dd", "win_rate", "avg_turnover"]:
        d = (m[f"{c}_base"] - m[f"{c}_opt"]).abs()
        diffs[c] = {
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
        }
    # integer-like
    d_nr = (m["n_rebalance_base"] - m["n_rebalance_opt"]).abs()
    diffs["n_rebalance"] = {"max_abs": int(d_nr.max())}

    thr = 1e-9
    pass_all = (
        diffs["sharpe"]["max_abs"] < thr
        and diffs["annual_ret"]["max_abs"] < thr
        and diffs["max_dd"]["max_abs"] < thr
        and diffs["win_rate"]["max_abs"] < thr
        and diffs["n_rebalance"]["max_abs"] == 0
    )

    print("A/B diff summary:")
    for k, v in diffs.items():
        print(f"  {k}: {v}")
    print(f"PASS: {pass_all}")


if __name__ == "__main__":
    main()
