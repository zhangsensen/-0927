from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batch_resample_hk import batch_resample_all_1m  # noqa: E402


def test_batch_resample_all_1m_produces_expected_outputs(tmp_path) -> None:
    data_root = tmp_path / "hk_raw"
    data_root.mkdir()

    index = pd.date_range("2025-03-05 09:30", periods=120, freq="min")
    base = np.linspace(50, 55, len(index))
    frame = pd.DataFrame(
        {
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.1,
            "volume": np.full(len(index), 1_000_000),
        },
        index=index,
    )
    source = data_root / "0700HK_1m_2025-03-05_2025-03-05.parquet"
    frame.to_parquet(source)

    output_dir = tmp_path / "resampled"
    result_dir = batch_resample_all_1m(
        data_root=data_root,
        output_dir=output_dir,
        timeframes=["15m", "60m"],
    )

    assert result_dir == output_dir.resolve()

    expected_15m = output_dir / "0700HK_15m_2025-03-05_2025-03-05.parquet"
    expected_60m = output_dir / "0700HK_60m_2025-03-05_2025-03-05.parquet"

    assert expected_15m.exists()
    assert expected_60m.exists()

    resampled_60m = pd.read_parquet(expected_60m)
    assert not resampled_60m.empty
    assert "close" in resampled_60m.columns
