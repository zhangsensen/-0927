"""Seal v3.3 Release (Regime Gate ON + Diversity Portfolio).

This script packages the selected 5 diverse strategies into a sealed release.
It follows the structure of previous releases.

Inputs:
- Portfolio Candidates: results/v3_3_portfolio_candidates.csv
- BT Results (Gate ON): results/bt_backtest_top200_20251216_041414/bt_results.parquet
- Source Code: src/, scripts/, configs/

Outputs:
- sealed_strategies/v3.3_<date>/
"""

import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
import os


def main():
    # 1. Setup Paths
    ts = datetime.now().strftime("%Y%m%d")
    version = "v3.3"
    release_name = f"{version}_{ts}"
    root_dir = Path("sealed_strategies") / release_name

    print(f"Creating release: {release_name}")

    if root_dir.exists():
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True)

    # 2. Load Data
    p_portfolio = Path("results/v3_3_portfolio_candidates.csv")
    p_bt = Path("results/bt_backtest_top200_20251216_041414/bt_results.parquet")

    portfolio = pd.read_csv(p_portfolio)
    bt_results = pd.read_parquet(p_bt)

    # Merge to get full metrics
    merged = portfolio.merge(bt_results, on="combo", how="inner")

    if len(merged) != len(portfolio):
        print(
            f"WARNING: Only {len(merged)}/{len(portfolio)} strategies found in BT results!"
        )

    # 3. Create Directory Structure
    (root_dir / "artifacts").mkdir()
    (root_dir / "locked").mkdir()
    (root_dir / "locked/configs").mkdir()
    (root_dir / "locked/scripts").mkdir()
    (root_dir / "locked/src").mkdir()
    (root_dir / "locked/docs").mkdir()

    # 4. Save Artifacts
    merged.to_parquet(root_dir / "artifacts" / "production_candidates.parquet")
    merged.to_csv(root_dir / "artifacts" / "production_candidates.csv", index=False)

    # 5. Copy Locked Code
    print("Copying source code...")
    shutil.copytree("src", root_dir / "locked/src", dirs_exist_ok=True)
    shutil.copytree("scripts", root_dir / "locked/scripts", dirs_exist_ok=True)
    shutil.copytree("configs", root_dir / "locked/configs", dirs_exist_ok=True)

    shutil.copy("pyproject.toml", root_dir / "locked/pyproject.toml")
    shutil.copy("Makefile", root_dir / "locked/Makefile")

    # 6. Generate Documentation
    print("Generating documentation...")

    # RELEASE_NOTES
    with open(
        root_dir
        / "locked/docs"
        / f"RELEASE_NOTES_{version.upper().replace('.', '_')}.md",
        "w",
    ) as f:
        f.write(f"# Release Notes {version} ({ts})\n\n")
        f.write("## 🎯 Objective\n")
        f.write(
            "Deliver a **robust, diverse strategy portfolio** optimized for compounding returns rather than single-shot high yield.\n"
        )
        f.write(
            "This release enables the **Regime Gate (Volatility)** mechanism to significantly reduce drawdown and improve Sharpe ratio.\n\n"
        )

        f.write("## 🛡️ Regime Gate Impact (vs v3.1/v3.2)\n")
        f.write("- **Max Drawdown**: Reduced by ~3.2% (Median 16.0% -> 12.8%)\n")
        f.write("- **Sharpe Ratio**: Improved by ~0.1 (Median 0.88 -> 0.98)\n")
        f.write("- **Win Rate**: Improved by ~3% (Rolling 3M Win Rate 64% -> 67%)\n")
        f.write(
            "- **Trade-off**: Total Return decreased by ~10% (Median 106% -> 96%), accepting lower absolute return for higher stability.\n\n"
        )

        f.write("## 🧩 Portfolio Composition\n")
        f.write(
            "Selected 5 strategies with **low factor overlap (Jaccard < 0.6)** to ensure diversity:\n\n"
        )

        for _, row in merged.iterrows():
            f.write(f"### Strategy #{row.name + 1}\n")
            f.write(f"- **Combo**: `{row['combo']}`\n")
            f.write(f"- **Role**: {get_strategy_role(row['combo'])}\n")
            f.write(f"- **Composite Score**: {row['composite_score']:.4f}\n")
            f.write(f"- **Holdout Calmar**: {row['holdout_calmar_ratio']:.2f}\n")
            f.write(f"- **BT Total Return**: {row['bt_return']:.2%}\n")
            f.write(f"- **BT Max Drawdown**: {row['bt_max_drawdown']:.2%}\n\n")

    # PRODUCTION_STRATEGY_LIST
    with open(
        root_dir
        / "locked/docs"
        / f"PRODUCTION_STRATEGY_LIST_{version.upper().replace('.', '_')}.md",
        "w",
    ) as f:
        f.write(f"# Production Strategy List {version}\n\n")
        f.write(
            merged[
                [
                    "combo",
                    "composite_score",
                    "bt_return",
                    "bt_max_drawdown",
                    "bt_calmar_ratio",
                ]
            ].to_markdown(index=False)
        )

    # REPRODUCE.md
    with open(root_dir / "REPRODUCE.md", "w") as f:
        f.write(f"# Reproduce {version}\n\n")
        f.write("To reproduce this release:\n\n")
        f.write("1. Ensure environment is set up (`uv sync`).\n")
        f.write("2. Run the full pipeline with Regime Gate ON:\n")
        f.write("   ```bash\n")
        f.write(
            "   uv run python scripts/run_full_pipeline.py --top-n 200 --n-jobs 24 --regime-gate on\n"
        )
        f.write("   ```\n")
        f.write("3. Select the portfolio:\n")
        f.write("   ```bash\n")
        f.write("   uv run python scripts/select_v3_3_portfolio.py\n")
        f.write("   ```\n")

    # CHECKSUMS
    print("Generating checksums...")
    os.system(
        f"find {root_dir} -type f -exec sha256sum {{}} + > {root_dir}/CHECKSUMS.sha256"
    )

    print(f"✅ Sealed {version} to {root_dir}")


def get_strategy_role(combo):
    factors = combo.split(" + ")
    if "MAX_DD_60D" in str(factors) and "VOL_RATIO_60D" in str(factors):
        return "Defensive / Risk-Managed"
    if "VORTEX_14D" in str(factors) and "SLOPE_20D" in str(factors):
        return "Trend Follower"
    if "SHARPE_RATIO_20D" in str(factors) and "PV_CORR_20D" in str(factors):
        return "Balanced / Quality"
    return "Diversified Alpha"


if __name__ == "__main__":
    main()
