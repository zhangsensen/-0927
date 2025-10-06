import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config_manager import ScreeningConfig  # noqa: E402
from professional_factor_screener import ProfessionalFactorScreener  # noqa: E402


@pytest.fixture()
def tmp_config(tmp_path: Path) -> ScreeningConfig:
    data_root = tmp_path / "data"
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"

    factor_dir = data_root / "daily"
    factor_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=64, freq="D")

    factors = pd.DataFrame(
        rng.normal(size=(len(index), 16)),
        index=index,
        columns=[f"factor_{i:02d}" for i in range(16)],
    )
    factors.to_parquet(factor_dir / "0005.HK_daily_factors_dummy.parquet")

    price_dir = raw_root / "HK"
    price_dir.mkdir(parents=True, exist_ok=True)

    base_price = 100 + rng.normal(scale=0.5, size=len(index)).cumsum()
    intraday_noise = rng.normal(scale=0.2, size=len(index))
    prices = pd.DataFrame(
        {
            "timestamp": index,
            "open": base_price,
            "close": base_price + rng.normal(scale=0.1, size=len(index)),
            "high": base_price + np.abs(intraday_noise),
            "low": base_price - np.abs(intraday_noise),
            "volume": rng.integers(1_000, 10_000, size=len(index)),
        }
    )
    prices.to_parquet(price_dir / "0005HK_1day_dummy.parquet")

    return ScreeningConfig(
        data_root=str(data_root),
        raw_data_root=str(price_dir),
        output_root=str(output_root),
        symbols=["0005.HK"],
        timeframes=["daily"],
        ic_horizons=[1, 5, 10],
        min_sample_size=30,
        min_data_points=30,
        min_momentum_samples=30,
    )


def test_smoke_pipeline(tmp_config: ScreeningConfig) -> None:
    screener = ProfessionalFactorScreener(config=tmp_config)
    result = screener.screen_factors_comprehensive("0005.HK", "daily")

    assert result
    assert all(isinstance(v.comprehensive_score, float) for v in result.to_numpy()())
    assert Path(screener.screening_results_dir).exists()
