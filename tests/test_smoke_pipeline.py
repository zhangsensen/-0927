from pathlib import Path

import pandas as pd
import pytest

from config_manager import ScreeningConfig
from professional_factor_screener import ProfessionalFactorScreener


@pytest.fixture()
def tmp_config(tmp_path: Path) -> ScreeningConfig:
    data_root = tmp_path / "data"
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"

    for directory in (data_root, raw_root / "HK", output_root):
        directory.mkdir(parents=True, exist_ok=True)

    factor_dir = data_root / "daily"
    factor_dir.mkdir(parents=True, exist_ok=True)
    factors = pd.DataFrame(
        {
            f"factor_{i:02d}": range(10) for i in range(12)
        },
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    factors.to_parquet(factor_dir / "0005.HK_daily_factors_dummy.parquet")

    price_dir = raw_root / "HK"
    price_dir.mkdir(parents=True, exist_ok=True)
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            "open": range(10),
            "high": range(1, 11),
            "low": range(10),
            "close": range(10),
            "volume": [100] * 10,
        }
    )
    prices.to_parquet(price_dir / "0005HK_1day_dummy.parquet")

    return ScreeningConfig(
        data_root=str(data_root),
        raw_data_root=str(price_dir),
        output_root=str(output_root),
        symbols=["0005.HK"],
        timeframes=["daily"],
        ic_horizons=[1],
        min_sample_size=5,
        min_data_points=10,
    )


def test_smoke_pipeline(tmp_config: ScreeningConfig) -> None:
    screener = ProfessionalFactorScreener(config=tmp_config)
    result = screener.screen_factors_comprehensive("0005.HK", "daily")

    assert result
    assert all(isinstance(v.comprehensive_score, float) for v in result.values())
    assert Path(screener.screening_results_dir).exists()
