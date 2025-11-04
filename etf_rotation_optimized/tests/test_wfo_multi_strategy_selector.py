import os
from pathlib import Path

import numpy as np
import pandas as pd
from core.wfo_multi_strategy_selector import WFOMultiStrategySelector


class _DummyWindow:
    def __init__(self, idx, oos_start, oos_end, selected_factors, factor_weights):
        self.window_index = idx
        self.is_start = 0
        self.is_end = 0
        self.oos_start = oos_start
        self.oos_end = oos_end
        self.selected_factors = selected_factors
        self.factor_weights = factor_weights


def test_multi_strategy_selector_end_to_end(tmp_path: Path):
    # Toy data: T=40 days, N=5 ETFs, K=3 factors
    T, N, K = 40, 5, 3
    rng = np.random.default_rng(42)
    # factors: (T, N, K)
    factors = rng.normal(size=(T, N, K)).astype(float)
    # returns: small random noise with slight signal on factor 0
    base = rng.normal(scale=0.005, size=(T, N)).astype(float)
    returns = base + 0.0005 * factors[:, :, 0]

    factor_names = ["F0", "F1", "F2"]
    dates = pd.date_range("2025-01-01", periods=T, freq="B")

    # Two OOS windows: [10,25), [25,40)
    win0 = _DummyWindow(
        0,
        10,
        25,
        selected_factors=["F0", "F1"],
        factor_weights={"F0": 0.7, "F1": 0.3},
    )
    win1 = _DummyWindow(
        1,
        25,
        40,
        selected_factors=["F0", "F2"],
        factor_weights={"F0": 0.6, "F2": 0.4},
    )
    results_list = [win0, win1]

    selector = WFOMultiStrategySelector(
        min_factor_freq=0.1,
        min_factors=2,
        max_factors=2,
        tau_grid=[0.7, 1.0],
        topn_grid=[3],
        max_strategies=20,
    )

    out_dir = tmp_path
    top5 = selector.select_and_save(
        results_list=results_list,
        factors=factors,
        returns=returns,
        factor_names=factor_names,
        dates=dates,
        out_dir=out_dir,
    )

    # Files created
    assert (out_dir / "strategies_ranked.csv").exists()
    assert (out_dir / "top5_strategies.csv").exists()
    assert (out_dir / "top5_combo_returns.csv").exists()
    assert (out_dir / "top5_combo_equity.csv").exists()
    assert (out_dir / "top5_combo_kpi.csv").exists()

    # Top5 content sanity
    assert len(top5) <= 5
    assert {"factors", "tau", "top_n", "score"}.issubset(set(top5.columns))
