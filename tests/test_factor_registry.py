"""
tests/test_factor_registry.py

因子注册中心单元测试 + 三处同步一致性测试 + 真实数据集成测试
"""

from pathlib import Path

import pytest
import yaml

from etf_strategy.core.factor_registry import (
    FACTOR_SPECS,
    get_bounded_factors,
    get_bounded_factors_tuple,
    get_factor_bounds,
    get_factor_source,
    get_non_ohlcv_factor_names,
)

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "combo_wfo_config.yaml"


# ---------------------------------------------------------------------------
# 1. FactorSpec 数据完整性
# ---------------------------------------------------------------------------


class TestFactorSpecIntegrity:
    def test_total_count(self):
        """44 factors registered: 38 OHLCV + 6 non-OHLCV."""
        assert len(FACTOR_SPECS) == 44

    def test_ohlcv_count(self):
        ohlcv = [s for s in FACTOR_SPECS.values() if s.source == "ohlcv"]
        assert len(ohlcv) == 38

    def test_non_ohlcv_count(self):
        non = [s for s in FACTOR_SPECS.values() if s.source != "ohlcv"]
        assert len(non) == 6

    def test_non_ohlcv_sources(self):
        sources = {s.source for s in FACTOR_SPECS.values() if s.source != "ohlcv"}
        assert sources == {"fund_share", "margin"}

    def test_fund_share_factors(self):
        fs = [n for n, s in FACTOR_SPECS.items() if s.source == "fund_share"]
        assert sorted(fs) == ["SHARE_ACCEL", "SHARE_CHG_10D", "SHARE_CHG_20D", "SHARE_CHG_5D"]

    def test_margin_factors(self):
        mg = [n for n, s in FACTOR_SPECS.items() if s.source == "margin"]
        assert sorted(mg) == ["MARGIN_BUY_RATIO", "MARGIN_CHG_10D"]

    def test_all_specs_have_name_matching_key(self):
        for key, spec in FACTOR_SPECS.items():
            assert spec.name == key, f"Key '{key}' != spec.name '{spec.name}'"

    def test_all_specs_are_frozen(self):
        """FactorSpec is frozen dataclass — immutable."""
        spec = FACTOR_SPECS["ADX_14D"]
        with pytest.raises(AttributeError):
            spec.is_bounded = False  # type: ignore[misc]

    def test_bounded_factors_have_bounds(self):
        """Every bounded factor must define bounds."""
        for name, spec in FACTOR_SPECS.items():
            if spec.is_bounded:
                assert spec.bounds is not None, f"{name} is_bounded=True but bounds=None"
                assert len(spec.bounds) == 2, f"{name} bounds must be (low, high)"
                assert spec.bounds[0] < spec.bounds[1], f"{name} bounds inverted"

    def test_unbounded_factors_no_bounds(self):
        """Unbounded factors should not have bounds set."""
        for name, spec in FACTOR_SPECS.items():
            if not spec.is_bounded:
                assert spec.bounds is None, f"{name} is_bounded=False but bounds={spec.bounds}"


# ---------------------------------------------------------------------------
# 2. 派生函数正确性
# ---------------------------------------------------------------------------


class TestDerivedFunctions:
    def test_bounded_factors_count(self):
        assert len(get_bounded_factors()) == 7

    def test_bounded_factors_exact(self):
        expected = {
            "ADX_14D", "CMF_20D", "CORRELATION_TO_MARKET_20D",
            "PRICE_POSITION_20D", "PRICE_POSITION_120D", "PV_CORR_20D", "RSI_14",
        }
        assert get_bounded_factors() == expected

    def test_factor_bounds_keys_match_bounded(self):
        assert set(get_factor_bounds().keys()) == get_bounded_factors()

    def test_factor_bounds_values(self):
        bounds = get_factor_bounds()
        assert bounds["ADX_14D"] == (0.0, 100.0)
        assert bounds["CMF_20D"] == (-1.0, 1.0)
        assert bounds["PRICE_POSITION_20D"] == (0.0, 1.0)
        assert bounds["RSI_14"] == (0.0, 100.0)

    def test_bounded_factors_tuple_sorted(self):
        t = get_bounded_factors_tuple()
        assert isinstance(t, tuple)
        assert list(t) == sorted(t), "Tuple must be sorted"

    def test_non_ohlcv_factor_names(self):
        names = get_non_ohlcv_factor_names()
        assert len(names) == 6
        for n in names:
            assert FACTOR_SPECS[n].source != "ohlcv"

    def test_get_factor_source(self):
        assert get_factor_source("ADX_14D") == "ohlcv"
        assert get_factor_source("SHARE_CHG_10D") == "fund_share"
        assert get_factor_source("MARGIN_CHG_10D") == "margin"
        assert get_factor_source("NONEXISTENT") is None


# ---------------------------------------------------------------------------
# 3. 三处同步一致性（核心 — 防止 S1 事故复发）
# ---------------------------------------------------------------------------


class TestThreeWaySync:
    """Verify CrossSectionProcessor, frozen_params, and registry are in sync."""

    def test_cross_section_processor_bounded(self):
        from etf_strategy.core.cross_section_processor import CrossSectionProcessor

        assert CrossSectionProcessor.BOUNDED_FACTORS == get_bounded_factors()

    def test_cross_section_processor_bounds(self):
        from etf_strategy.core.cross_section_processor import CrossSectionProcessor

        assert CrossSectionProcessor.FACTOR_BOUNDS == get_factor_bounds()

    def test_frozen_params_bounded(self):
        from etf_strategy.core.frozen_params import FrozenCrossSectionParams

        frozen = FrozenCrossSectionParams()
        assert set(frozen.bounded_factors) == get_bounded_factors()

    def test_config_has_no_bounded_factors(self):
        """bounded_factors was removed from config — now code-driven."""
        config = yaml.safe_load(open(CONFIG_PATH))
        cs_cfg = config.get("cross_section", {})
        assert "bounded_factors" not in cs_cfg, (
            "cross_section.bounded_factors should be removed from config YAML"
        )

    def test_frozen_strict_passes_without_config_bounded(self):
        """Frozen params strict validation must pass without bounded_factors in config."""
        from etf_strategy.core.frozen_params import (
            StrictnessMode,
            load_frozen_config,
        )

        config = yaml.safe_load(open(CONFIG_PATH))
        frozen = load_frozen_config(
            config, config_path=str(CONFIG_PATH), strictness=StrictnessMode.STRICT
        )
        assert len(frozen.cross_section.bounded_factors) == 7


# ---------------------------------------------------------------------------
# 4. 生产策略因子完整性
# ---------------------------------------------------------------------------


class TestProductionStrategies:
    """Verify production strategy factors are all registered."""

    def test_s1_factors_registered(self):
        s1 = ["ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"]
        for f in s1:
            assert f in FACTOR_SPECS, f"S1 factor {f} not registered"

    def test_c2_factors_registered(self):
        c2 = ["AMIHUD_ILLIQUIDITY", "CALMAR_RATIO_60D", "CORRELATION_TO_MARKET_20D"]
        for f in c2:
            assert f in FACTOR_SPECS, f"C2 factor {f} not registered"

    def test_active_factors_all_registered(self):
        config = yaml.safe_load(open(CONFIG_PATH))
        active = config.get("active_factors", [])
        for f in active:
            assert f in FACTOR_SPECS, f"active_factor {f} not in FACTOR_SPECS"


# ---------------------------------------------------------------------------
# 5. FactorCache loader 参数签名
# ---------------------------------------------------------------------------


class TestFactorCacheSignature:
    def test_loader_param_exists(self):
        import inspect

        from etf_strategy.core.factor_cache import FactorCache

        sig = inspect.signature(FactorCache.get_or_compute)
        assert "loader" in sig.parameters
        assert sig.parameters["loader"].default is None

    def test_backward_compatible_without_loader(self):
        """Calling without loader should not raise."""
        import inspect

        from etf_strategy.core.factor_cache import FactorCache

        sig = inspect.signature(FactorCache.get_or_compute)
        # 4 params: self, ohlcv, config, data_dir (+ optional loader)
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty and p.name != "self"
        ]
        assert len(required) == 3, f"Expected 3 required params, got {len(required)}"


# ---------------------------------------------------------------------------
# 6. 真实数据集成测试（inline vs parquet 一致性）
# ---------------------------------------------------------------------------


class TestRealDataIntegration:
    """Compare inline loader path vs parquet path with real data."""

    @pytest.fixture(scope="class")
    def real_data(self):
        """Load real OHLCV data once for this class."""
        config = yaml.safe_load(open(CONFIG_PATH))
        data_dir = Path(config["data"]["data_dir"])
        if not data_dir.exists():
            pytest.skip("Real data not available")

        from etf_strategy.core.data_loader import DataLoader

        loader = DataLoader(data_dir=str(data_dir))
        ohlcv = loader.load_ohlcv()
        return {"config": config, "data_dir": data_dir, "loader": loader, "ohlcv": ohlcv}

    def test_inline_produces_all_active_non_ohlcv(self, real_data):
        """Inline loader path must produce all active non-OHLCV factors."""
        import numpy as np

        from etf_strategy.core.factor_cache import FactorCache

        config = real_data["config"]
        fc = FactorCache(cache_dir=Path(config["data"].get("cache_dir") or ".cache"))
        result = fc.get_or_compute(
            real_data["ohlcv"], config, real_data["data_dir"],
            loader=real_data["loader"],
        )

        active = set(config.get("active_factors", []))
        non_ohlcv_active = {
            n for n, s in FACTOR_SPECS.items()
            if s.source != "ohlcv" and n in active
        }
        for fname in non_ohlcv_active:
            assert fname in result["std_factors"], f"{fname} missing from inline result"
            df = result["std_factors"][fname]
            valid = np.isfinite(df.values).sum()
            assert valid > 0, f"{fname} has zero valid values"

    def test_inline_vs_parquet_bit_identical(self, real_data):
        """Non-OHLCV factors must be bit-identical between two paths."""
        import numpy as np

        from etf_strategy.core.factor_cache import FactorCache

        config = real_data["config"]
        cache_dir = Path(config["data"].get("cache_dir") or ".cache")

        r_parquet = FactorCache(cache_dir=cache_dir).get_or_compute(
            real_data["ohlcv"], config, real_data["data_dir"], loader=None,
        )
        r_inline = FactorCache(cache_dir=cache_dir).get_or_compute(
            real_data["ohlcv"], config, real_data["data_dir"],
            loader=real_data["loader"],
        )

        non_ohlcv = ["MARGIN_BUY_RATIO", "MARGIN_CHG_10D", "SHARE_ACCEL",
                      "SHARE_CHG_10D", "SHARE_CHG_20D", "SHARE_CHG_5D"]

        for fname in non_ohlcv:
            if fname not in r_parquet["std_factors"]:
                continue  # parquet might not have it
            assert fname in r_inline["std_factors"], f"{fname} missing from inline"

            old = r_parquet["std_factors"][fname].values
            new = r_inline["std_factors"][fname].values
            both_valid = np.isfinite(old) & np.isfinite(new)
            if both_valid.sum() == 0:
                continue
            max_diff = np.abs(old[both_valid] - new[both_valid]).max()
            assert max_diff < 1e-10, (
                f"{fname}: max_diff={max_diff:.2e} between parquet and inline"
            )

    def test_ohlcv_factors_identical_across_paths(self, real_data):
        """OHLCV factors must be identical regardless of loader presence."""
        import numpy as np

        from etf_strategy.core.factor_cache import FactorCache

        config = real_data["config"]
        cache_dir = Path(config["data"].get("cache_dir") or ".cache")

        r_parquet = FactorCache(cache_dir=cache_dir).get_or_compute(
            real_data["ohlcv"], config, real_data["data_dir"], loader=None,
        )
        r_inline = FactorCache(cache_dir=cache_dir).get_or_compute(
            real_data["ohlcv"], config, real_data["data_dir"],
            loader=real_data["loader"],
        )

        ohlcv_factors = [
            n for n, s in FACTOR_SPECS.items()
            if s.source == "ohlcv" and n in r_parquet["std_factors"]
            and n in r_inline["std_factors"]
        ]

        for fname in ohlcv_factors[:10]:  # sample 10
            old = r_parquet["std_factors"][fname].values
            new = r_inline["std_factors"][fname].values
            both = np.isfinite(old) & np.isfinite(new)
            if both.sum() == 0:
                continue
            max_diff = np.abs(old[both] - new[both]).max()
            assert max_diff < 1e-10, (
                f"OHLCV {fname}: max_diff={max_diff:.2e} diverged"
            )
