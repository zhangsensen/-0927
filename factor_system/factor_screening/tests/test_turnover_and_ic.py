import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - 防御性检查
        raise ImportError(f"无法加载模块 {module_name} 来自 {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MODULE_ROOT = Path(__file__).resolve().parents[1]
config_module = _load_module("config_manager", MODULE_ROOT / "config_manager.py")
screener_module = _load_module(
    "professional_factor_screener", MODULE_ROOT / "professional_factor_screener.py"
)

ScreeningConfig = config_module.ScreeningConfig
ProfessionalFactorScreener = screener_module.ProfessionalFactorScreener


@pytest.fixture()
def screener(tmp_path: Path) -> ProfessionalFactorScreener:
    cfg = ScreeningConfig(
        data_root=str(tmp_path / "data"),
        raw_data_root=str(tmp_path / "raw"),
        factor_data_root=str(tmp_path / "factor"),
        price_data_root=str(tmp_path / "price"),
        output_root=str(tmp_path / "out"),
        log_root=str(tmp_path / "logs"),
        cache_root=str(tmp_path / "cache"),
    )
    cfg.output_dir = str(tmp_path / "legacy_out")
    cfg.ic_horizons = [1]
    cfg.min_sample_size = 5
    return ProfessionalFactorScreener(config=cfg)


def test_turnover_rate_handles_cumulative_indicators(
    screener: ProfessionalFactorScreener,
) -> None:
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    cumulative = pd.Series(
        np.cumsum(np.full(120, 1000.0)), index=dates, name="OBV_factor"
    )
    oscillator = pd.Series(
        np.sin(np.linspace(0.0, 10.0, 120)), index=dates, name="RSI14"
    )

    turnover_cumulative = screener._calculate_turnover_rate(
        cumulative,
        factor_name="OBV_factor",
        factor_type="volume",
        turnover_profile="cumulative",
    )
    turnover_oscillator = screener._calculate_turnover_rate(
        oscillator,
        factor_name="RSI14",
        factor_type="momentum",
        turnover_profile="oscillator",
    )

    assert math.isfinite(turnover_cumulative)
    assert math.isfinite(turnover_oscillator)
    assert 0.0 <= turnover_cumulative <= 0.1
    assert 0.0 < turnover_oscillator <= 0.2
    assert turnover_cumulative < turnover_oscillator


def test_multi_horizon_ic_uses_historical_alignment(
    screener: ProfessionalFactorScreener,
) -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    factor = pd.DataFrame({"factor_a": np.arange(40, dtype=float)}, index=dates)
    returns = pd.Series(np.arange(40, dtype=float), index=dates)

    ic_results = screener.calculate_multi_horizon_ic(factor, returns)

    assert "factor_a" in ic_results
    factor_metrics = ic_results["factor_a"]

    expected_ic = factor["factor_a"].shift(1).corr(returns)
    assert pytest.approx(factor_metrics["ic_1d"], rel=1e-6) == expected_ic
    assert factor_metrics["sample_size_1d"] == len(dates) - 1
