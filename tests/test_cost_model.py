"""Tests for CostModel dataclass, loading, and integration.

Validates:
1. CostModel creation & validation (invalid mode/tier should raise)
2. load_cost_model from config (with/without cost_model section)
3. build_cost_array correctly assigns A-share vs QDII costs
4. frozen_params doesn't break with cost_model in config
5. VEC kernel regression: UNIFIED mode with 0.0002 = legacy behavior
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


# ─────────────────────────────────────────────────────────
#  1. CostModel unit tests
# ─────────────────────────────────────────────────────────


class TestCostModelCreation:
    def test_default_values(self):
        from etf_strategy.core.cost_model import CostModel

        m = CostModel()
        assert m.mode == "SPLIT_MARKET"
        assert m.tier == "med"
        assert m.is_split_market is True

    def test_unified_mode(self):
        from etf_strategy.core.cost_model import CostModel

        m = CostModel(mode="UNIFIED", unified_rate=0.0002)
        assert m.is_split_market is False
        t = m.active_tier
        assert t.a_share == 0.0002
        assert t.qdii == 0.0002

    def test_invalid_mode_raises(self):
        from etf_strategy.core.cost_model import CostModel

        with pytest.raises(ValueError, match="Unknown cost model mode"):
            CostModel(mode="INVALID")

    def test_invalid_tier_raises(self):
        from etf_strategy.core.cost_model import CostModel

        with pytest.raises(ValueError, match="Unknown cost tier"):
            CostModel(tier="ultra")

    def test_frozen(self):
        from etf_strategy.core.cost_model import CostModel

        m = CostModel()
        with pytest.raises(Exception):
            m.mode = "UNIFIED"  # type: ignore[misc]

    def test_with_tier(self):
        from etf_strategy.core.cost_model import CostModel

        m = CostModel(
            tiers=(("low", 0.001, 0.003), ("med", 0.002, 0.005), ("high", 0.003, 0.008)),
            tier="med",
        )
        m2 = m.with_tier("high")
        assert m2.tier == "high"
        t = m2.active_tier
        assert t.a_share == 0.003
        assert t.qdii == 0.008


class TestCostTier:
    def test_negative_cost_raises(self):
        from etf_strategy.core.cost_model import CostTier

        with pytest.raises(ValueError, match="non-negative"):
            CostTier(a_share=-0.001, qdii=0.003)

    def test_valid_tier(self):
        from etf_strategy.core.cost_model import CostTier

        t = CostTier(a_share=0.002, qdii=0.005)
        assert t.a_share == 0.002
        assert t.qdii == 0.005


# ─────────────────────────────────────────────────────────
#  2. Config loading tests
# ─────────────────────────────────────────────────────────


class TestLoadCostModel:
    def test_load_with_cost_model_section(self):
        from etf_strategy.core.cost_model import load_cost_model

        config = {
            "backtest": {
                "commission_rate": 0.0002,
                "cost_model": {
                    "mode": "SPLIT_MARKET",
                    "tier": "med",
                    "tiers": {
                        "low": {"a_share": 0.0010, "qdii": 0.0030},
                        "med": {"a_share": 0.0020, "qdii": 0.0050},
                        "high": {"a_share": 0.0030, "qdii": 0.0080},
                    },
                },
            }
        }
        m = load_cost_model(config)
        assert m.mode == "SPLIT_MARKET"
        assert m.tier == "med"
        t = m.active_tier
        assert t.a_share == 0.0020
        assert t.qdii == 0.0050

    def test_load_without_cost_model_fallback(self):
        """When cost_model section is absent, fall back to UNIFIED with commission_rate."""
        from etf_strategy.core.cost_model import load_cost_model

        config = {"backtest": {"commission_rate": 0.0003}}
        m = load_cost_model(config)
        assert m.mode == "UNIFIED"
        assert m.unified_rate == 0.0003
        t = m.active_tier
        assert t.a_share == 0.0003
        assert t.qdii == 0.0003

    def test_load_production_config(self):
        """Production config should load correctly."""
        from etf_strategy.core.cost_model import load_cost_model

        config_path = ROOT / "configs" / "combo_wfo_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        m = load_cost_model(config)
        assert m.mode == "SPLIT_MARKET"
        assert m.tier == "med"
        t = m.active_tier
        assert t.a_share == 0.0020
        assert t.qdii == 0.0050


# ─────────────────────────────────────────────────────────
#  3. build_cost_array tests
# ─────────────────────────────────────────────────────────


class TestBuildCostArray:
    def test_split_market_assigns_correctly(self):
        from etf_strategy.core.cost_model import CostModel, build_cost_array

        m = CostModel(
            mode="SPLIT_MARKET",
            tier="med",
            tiers=(("med", 0.0020, 0.0050),),
        )
        etf_codes = ["510300", "513100", "159920", "512800"]
        qdii_codes = {"513100", "159920"}
        arr = build_cost_array(m, etf_codes, qdii_codes)

        assert arr.shape == (4,)
        assert arr[0] == 0.0020  # A-share: 510300
        assert arr[1] == 0.0050  # QDII: 513100
        assert arr[2] == 0.0050  # QDII: 159920
        assert arr[3] == 0.0020  # A-share: 512800

    def test_unified_mode_all_same(self):
        from etf_strategy.core.cost_model import CostModel, build_cost_array

        m = CostModel(mode="UNIFIED", unified_rate=0.0002)
        etf_codes = ["510300", "513100", "512800"]
        qdii_codes = {"513100"}
        arr = build_cost_array(m, etf_codes, qdii_codes)

        # In UNIFIED mode, all should be 0.0002
        np.testing.assert_allclose(arr, 0.0002)

    def test_production_qdii_set(self):
        """All 5 QDII ETFs get higher cost."""
        from etf_strategy.core.cost_model import load_cost_model, build_cost_array
        from etf_strategy.core.frozen_params import FrozenETFPool

        config_path = ROOT / "configs" / "combo_wfo_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        m = load_cost_model(config)
        etf_codes = config["data"]["symbols"]
        qdii_codes = set(FrozenETFPool().qdii_codes)
        arr = build_cost_array(m, etf_codes, qdii_codes)

        assert arr.shape == (len(etf_codes),)
        # Check all QDII codes get qdii rate
        for i, code in enumerate(etf_codes):
            if code in qdii_codes:
                assert arr[i] == m.active_tier.qdii, f"{code} should have QDII rate"
            else:
                assert arr[i] == m.active_tier.a_share, f"{code} should have A-share rate"

        # Exactly 5 QDII ETFs
        assert np.sum(arr == m.active_tier.qdii) == 5


# ─────────────────────────────────────────────────────────
#  4. frozen_params compatibility
# ─────────────────────────────────────────────────────────


class TestFrozenParamsCompatibility:
    def test_cost_model_does_not_break_frozen_params(self):
        """cost_model key in config should not trigger frozen_params validation error."""
        from etf_strategy.core.frozen_params import load_frozen_config

        config_path = ROOT / "configs" / "combo_wfo_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # This should not raise even though cost_model is present
        frozen = load_frozen_config(config, config_path=str(config_path))
        assert frozen is not None


# ─────────────────────────────────────────────────────────
#  5. VEC kernel regression: UNIFIED mode = legacy behavior
# ─────────────────────────────────────────────────────────


class TestVecKernelCostRegression:
    """UNIFIED cost_arr filled with 0.0002 should match legacy commission_rate=0.0002."""

    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        T, N = 260, 3
        close = np.full((T, N), 100.0)
        open_ = np.full((T, N), 100.0)
        high = np.full((T, N), 101.0)
        low = np.full((T, N), 99.0)
        # Add slight trend so we get nonzero returns
        for t in range(1, T):
            close[t] = close[t - 1] * 1.0003
            open_[t] = close[t]
            high[t] = close[t] * 1.005
            low[t] = close[t] * 0.995

        factors = np.random.randn(T, N, 2)
        timing = np.ones(T)
        return factors, close, open_, high, low, timing

    def test_uniform_cost_arr_matches_legacy(self, synthetic_data):
        from batch_vec_backtest import run_vec_backtest

        factors, close, open_, high, low, timing = synthetic_data
        N = close.shape[1]

        # Legacy: scalar commission_rate, no cost_arr
        _, ret_legacy, *_ = run_vec_backtest(
            factors, close, open_, high, low, timing,
            factor_indices=[0, 1], freq=3, pos_size=1,
            initial_capital=1_000_000.0, commission_rate=0.0002,
            lookback=252, trailing_stop_pct=0.0,
            stop_on_rebalance_only=True, use_t1_open=False,
        )

        # New: cost_arr uniformly filled with 0.0002
        cost_arr = np.full(N, 0.0002, dtype=np.float64)
        _, ret_new, *_ = run_vec_backtest(
            factors, close, open_, high, low, timing,
            factor_indices=[0, 1], freq=3, pos_size=1,
            initial_capital=1_000_000.0, commission_rate=0.0002,
            lookback=252, cost_arr=cost_arr,
            trailing_stop_pct=0.0,
            stop_on_rebalance_only=True, use_t1_open=False,
        )

        # Should produce identical results
        np.testing.assert_allclose(
            ret_new, ret_legacy, atol=1e-10,
            err_msg="Uniform cost_arr should reproduce legacy commission_rate behavior",
        )
