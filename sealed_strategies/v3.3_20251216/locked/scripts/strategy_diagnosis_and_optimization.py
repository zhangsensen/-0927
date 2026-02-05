import pandas as pd
import numpy as np
from scipy import stats
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RESULTS_PATH = Path("results/full_space_v3_no_lookahead_results.csv")
REPORT_PATH = Path("results/strategy_health_report.md")


def diagnose_vec_issues(df: pd.DataFrame):
    """
    Task 1: VEC Engine Anomaly Diagnosis
    """
    logger.info("Starting VEC Engine Diagnosis...")

    total = len(df)

    # 1.1 Data Quality Statistics
    inf_ann_ret = df["ann_ret"].isin([np.inf, -np.inf]).sum()
    nan_ann_ret = df["ann_ret"].isna().sum()

    inf_max_dd = df["max_dd"].isin([np.inf, -np.inf]).sum()
    nan_max_dd = df["max_dd"].isna().sum()

    inf_sharpe = df["sharpe"].isin([np.inf, -np.inf]).sum()
    nan_sharpe = df["sharpe"].isna().sum()

    # Check for zero trades if column exists
    zero_trades = 0
    low_trades = 0
    if "num_trades" in df.columns:
        zero_trades = (df["num_trades"] == 0).sum()
        low_trades = (df["num_trades"] < 5).sum()

    report = f"""
## 1. VEC Engine Diagnosis Report

**Total Strategies**: {total}

### 1.1 Data Quality Issues
| Metric | Inf Count | NaN Count | % Affected |
|--------|-----------|-----------|------------|
| Ann Ret | {inf_ann_ret} | {nan_ann_ret} | {(inf_ann_ret + nan_ann_ret)/total:.2%} |
| Max DD | {inf_max_dd} | {nan_max_dd} | {(inf_max_dd + nan_max_dd)/total:.2%} |
| Sharpe | {inf_sharpe} | {nan_sharpe} | {(inf_sharpe + nan_sharpe)/total:.2%} |

### 1.2 Trading Activity
- **Zero Trades**: {zero_trades} ({zero_trades/total:.2%})
- **Low Trades (<5)**: {low_trades} ({low_trades/total:.2%})

### 1.3 Root Cause Analysis
- **Inf/NaN in Returns**: Usually caused by division by zero when equity curve is flat (0 trades) or drops to 0 (bankruptcy).
- **Inf in Sharpe**: Caused by zero volatility (std dev = 0) when there are no trades or constant returns.
- **Low Trades**: Strategies with very restrictive entry conditions or incorrect parameter combinations.

### 1.4 Fixes Applied
- **Safe VEC Kernel**: Implemented checks in `batch_vec_backtest.py` to handle division by zero and clamp extreme values.
- **Robust Metrics**: `compute_period_metrics` now handles empty or constant equity curves gracefully.
"""
    return report


def enhanced_filtering(df: pd.DataFrame):
    """
    Task 2: Enhanced Filtering
    """
    logger.info("Running Enhanced Filtering...")

    initial_count = len(df)
    df_clean = df.copy()

    filter_log = []

    # 2.2 Data Quality Thresholds
    # Remove Inf/NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["ann_ret", "max_dd", "sharpe"]
    )
    log_nan = initial_count - len(df_clean)
    filter_log.append(f"Removed {log_nan} strategies with Inf/NaN values.")

    # Min Trades
    if "num_trades" in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean["num_trades"] >= 20]
        filter_log.append(
            f"Removed {before - len(df_clean)} strategies with < 20 trades."
        )

    # 2.1 Existing Logic
    # Ann Ret > 12%
    before = len(df_clean)
    df_clean = df_clean[df_clean["ann_ret"] > 0.12]
    filter_log.append(
        f"Removed {before - len(df_clean)} strategies with Ann Ret <= 12%."
    )

    # Max DD > -30% (i.e., <= 30% loss)
    before = len(df_clean)
    df_clean = df_clean[df_clean["max_dd"] > -0.30]
    filter_log.append(
        f"Removed {before - len(df_clean)} strategies with Max DD <= -30%."
    )

    # Positive Years
    if "ret_2023" in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean["ret_2023"] > 0]
        filter_log.append(
            f"Removed {before - len(df_clean)} strategies with Negative 2023."
        )

    if "ret_2024" in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean["ret_2024"] > 0]
        filter_log.append(
            f"Removed {before - len(df_clean)} strategies with Negative 2024."
        )

    # 2.3 Robustness (Mock implementation if columns missing)
    # Win Rate > 45%
    if "win_rate" in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean["win_rate"] > 0.45]
        filter_log.append(
            f"Removed {before - len(df_clean)} strategies with Win Rate <= 45%."
        )

    # Calmar > 0.3
    # Calmar = Ann Ret / abs(Max DD)
    before = len(df_clean)
    # Avoid division by zero
    calmar = df_clean["ann_ret"] / df_clean["max_dd"].abs().replace(0, 1e-6)
    df_clean = df_clean[calmar > 0.3]
    filter_log.append(
        f"Removed {before - len(df_clean)} strategies with Calmar <= 0.3."
    )

    report = """
## 2. Enhanced Filtering Report

### Filtering Steps
"""
    for log in filter_log:
        report += f"- {log}\n"

    report += f"\n**Final Candidates**: {len(df_clean)} ({(len(df_clean)/initial_count):.2%} of original)\n"

    return df_clean, report


def generate_health_report(df: pd.DataFrame):
    """
    Task 3: Strategy Health Report
    """
    logger.info("Generating Health Report...")

    if len(df) == 0:
        return "No strategies passed filtering. Cannot generate health report."

    # 3.2 Performance Stats
    stats_df = df[["ann_ret", "max_dd", "sharpe"]].describe()

    # 3.4 Factor Ecosystem
    # Parse combos "FactorA + FactorB + FactorC"
    all_factors = []
    for combo in df["combo"]:
        factors = combo.split(" + ")
        all_factors.extend(factors)

    factor_counts = pd.Series(all_factors).value_counts()
    top_10_factors = factor_counts.head(10)

    # Entropy
    probs = factor_counts / factor_counts.sum()
    entropy = -np.sum(probs * np.log2(probs))

    report = f"""
## 3. Strategy Health Report (Top Candidates)

### 3.1 Performance Statistics
| Metric | Mean | Median | Std Dev | Min | Max |
|--------|------|--------|---------|-----|-----|
| Ann Ret | {df['ann_ret'].mean():.2%} | {df['ann_ret'].median():.2%} | {df['ann_ret'].std():.2%} | {df['ann_ret'].min():.2%} | {df['ann_ret'].max():.2%} |
| Max DD | {df['max_dd'].mean():.2%} | {df['max_dd'].median():.2%} | {df['max_dd'].std():.2%} | {df['max_dd'].min():.2%} | {df['max_dd'].max():.2%} |
| Sharpe | {df['sharpe'].mean():.2f} | {df['sharpe'].median():.2f} | {df['sharpe'].std():.2f} | {df['sharpe'].min():.2f} | {df['sharpe'].max():.2f} |

### 3.2 Factor Ecosystem
- **Factor Diversity (Entropy)**: {entropy:.2f} (Higher is better)
- **Top 10 Factors**:
"""
    for factor, count in top_10_factors.items():
        pct = count / len(df)  # Percentage of strategies using this factor
        warning = "⚠️ Over-concentrated" if pct > 0.3 else ""
        report += f"  - **{factor}**: {count} ({pct:.1%}) {warning}\n"

    return report


def parameter_stability_test_code():
    return """
```python
def parameter_stability_test(top_strategies, data, config):
    \"\"\"
    Task 4: Parameter Stability Test (Code Template)
    \"\"\"
    results = []
    perturbations = [0.8, 0.9, 1.1, 1.2] # +/- 10%, 20%
    
    for strategy in top_strategies:
        base_sharpe = strategy['sharpe']
        variations = []
        
        # For each factor in strategy
        for factor in strategy['factors']:
            # Perturb window/period parameters
            base_param = get_param(factor)
            for p in perturbations:
                new_param = int(base_param * p)
                # Run WFO with new param
                # ...
                # new_sharpe = run_backtest(...)
                # variations.append(new_sharpe)
                pass
        
        # Analyze variations
        std_dev = np.std(variations)
        passed = std_dev < 0.15 and all(v > 0 for v in variations)
        results.append({
            'strategy': strategy['name'],
            'stability_score': std_dev,
            'passed': passed
        })
    return results
```
"""


def market_regime_test_code():
    return """
```python
def market_regime_test(strategy, market_data):
    \"\"\"
    Task 5: Market Regime Classification (Code Template)
    \"\"\"
    # 1. Classify Market
    # Trend: ADX > 25
    # Chop: ADX < 20 & Vol < 10%
    # Volatile: Vol > 25%
    
    regimes = classify_market(market_data)
    
    # 2. Calculate Performance per Regime
    perf = {}
    for regime in ['Trend', 'Chop', 'Volatile']:
        mask = regimes == regime
        # Calculate Sharpe on masked returns
        # ...
        pass
        
    return perf
```
"""


def monte_carlo_test_code():
    return """
```python
def monte_carlo_validation(trade_list):
    \"\"\"
    Task 6: Monte Carlo Permutation Test (Code Template)
    \"\"\"
    n_sims = 1000
    original_sharpe = calculate_sharpe(trade_list)
    sim_sharpes = []
    
    for _ in range(n_sims):
        # Shuffle trades
        shuffled_trades = np.random.permutation(trade_list)
        # Reconstruct equity curve
        # ...
        # sim_sharpe = ...
        sim_sharpes.append(sim_sharpe)
        
    p_value = (np.array(sim_sharpes) < original_sharpe).mean()
    return p_value
```
"""


def main():
    if not RESULTS_PATH.exists():
        logger.error(f"Results file not found: {RESULTS_PATH}")
        return

    df = pd.read_csv(RESULTS_PATH)

    # Generate Report Content
    report_content = "# Strategy Diagnosis and Optimization Report\n\n"
    report_content += f"Generated on: {pd.Timestamp.now()}\n\n"

    # Task 1
    report_content += diagnose_vec_issues(df)

    # Task 2
    df_clean, filter_report = enhanced_filtering(df)
    report_content += filter_report

    # Task 3
    report_content += generate_health_report(df_clean)

    # Task 4, 5, 6 (Code Templates)
    report_content += "\n## 4. Advanced Validation (Code Templates)\n"
    report_content += "### 4.1 Parameter Stability Test\n"
    report_content += parameter_stability_test_code()
    report_content += "\n### 4.2 Market Regime Test\n"
    report_content += market_regime_test_code()
    report_content += "\n### 4.3 Monte Carlo Validation\n"
    report_content += monte_carlo_test_code()

    # Save Report
    with open(REPORT_PATH, "w") as f:
        f.write(report_content)

    logger.info(f"Report saved to {REPORT_PATH}")

    # Save Cleaned Candidates
    clean_path = RESULTS_PATH.parent / "v3_enhanced_candidates.csv"
    df_clean.to_csv(clean_path, index=False)
    logger.info(f"Enhanced candidates saved to {clean_path}")


if __name__ == "__main__":
    main()
