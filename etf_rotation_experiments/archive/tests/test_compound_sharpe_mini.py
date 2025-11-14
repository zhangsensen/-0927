import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.combo_wfo_optimizer import (  # noqa: E402
    ComboWFOOptimizer,
    _compute_rebalanced_sharpe_stats,
)


def _compute_metrics_from_returns(returns):
    if not returns:
        return 0.0, 0.0, 0.0, 0
    arr = np.array(returns, dtype=float)
    mean_ret = float(np.mean(arr))
    if arr.size > 1:
        std_ret = float(np.std(arr, ddof=1))
    else:
        std_ret = 0.0
    if std_ret > 1e-12:
        sharpe = float(np.sqrt(252.0) * mean_ret / std_ret)
    else:
        sharpe = 0.0
    return sharpe, mean_ret, std_ret, int(arr.size)


def test_simulate_portfolio_returns_matches_numba():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(24, 5))
    returns = rng.normal(scale=0.01, size=(24, 5))
    nan_mask = rng.random(size=returns.shape) < 0.1
    returns[nan_mask] = np.nan

    rebalance_freq = 3
    top_k = 3

    optimizer = ComboWFOOptimizer(
        combo_sizes=[2],
        is_period=10,
        oos_period=8,
        step_size=5,
        rebalance_frequencies=[rebalance_freq],
        scoring_strategy="ic",
        scoring_position_size=top_k,
        enable_fdr=False,
        verbose=0,
        n_jobs=1,
    )

    simulated_returns = optimizer._simulate_portfolio_returns(
        signal, returns, rebalance_freq, top_k
    )
    sharpe_new, mean_new, std_new, count_new = _compute_metrics_from_returns(simulated_returns)

    sharpe_old, mean_old, std_old, count_old = _compute_rebalanced_sharpe_stats(
        signal, returns, rebalance_freq, top_k
    )

    assert count_new == count_old
    assert np.isclose(mean_new, mean_old, atol=1e-12)
    assert np.isclose(std_new, std_old, atol=1e-12)
    assert np.isclose(sharpe_new, sharpe_old, atol=1e-12)


def test_compound_strategy_prioritises_compound_metric():
    T, N, F = 12, 3, 2
    factor_names = ["F0", "F1"]

    rng = np.random.default_rng(123)
    returns = rng.normal(0.0, 0.002, size=(T, N))
    returns[:, 0] += 0.01  # asset 0: consistently positive drift
    returns[:, 2] -= 0.01  # asset 2: consistently negative drift

    factors_data = np.zeros((T, N, F), dtype=float)
    factors_data[:, 0, 0] = 1.0
    factors_data[:, 1, 0] = 0.2
    factors_data[:, 2, 0] = -1.0

    factors_data[:, 0, 1] = -1.0
    factors_data[:, 1, 1] = 0.1
    factors_data[:, 2, 1] = 1.0

    returns_copy = returns.copy()

    optimizer = ComboWFOOptimizer(
        combo_sizes=[1],
        is_period=4,
        oos_period=4,
        step_size=4,
        rebalance_frequencies=[1],
        scoring_strategy="oos_sharpe_compound",
        scoring_position_size=1,
        enable_fdr=False,
        verbose=0,
        n_jobs=1,
    )

    top_combos, df = optimizer.run_combo_search(
        factors_data,
        returns_copy,
        factor_names,
        top_n=2,
    )

    assert "oos_compound_sharpe" in df.columns
    assert "oos_compound_sample_count" in df.columns

    top_combo_name = top_combos[0]["combo"]
    top_compound_sharpe = df.iloc[0]["oos_compound_sharpe"]

    assert top_combo_name == "F0"
    assert np.isclose(top_compound_sharpe, df["oos_compound_sharpe"].max())
    assert df.iloc[0]["oos_compound_sharpe"] > df.iloc[1]["oos_compound_sharpe"]