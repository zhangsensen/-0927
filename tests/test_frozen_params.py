"""
tests/test_frozen_params.py

参数冻结模块单元测试
"""

import copy
from pathlib import Path

import pytest
import yaml

from etf_strategy.core.frozen_params import (
    CURRENT_VERSION,
    FrozenBacktestParams,
    FrozenETFPool,
    FrozenParamViolation,
    FrozenProductionConfig,
    FrozenRegimeGateParams,
    FrozenScoringParams,
    FrozenWFOParams,
    StrictnessMode,
    load_frozen_config,
)

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "combo_wfo_config.yaml"


@pytest.fixture
def real_config():
    """Load the real production config."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. 冻结值正确性
# ---------------------------------------------------------------------------


class TestFrozenValues:
    def test_current_version(self):
        assert CURRENT_VERSION == "v4.0"

    def test_backtest_defaults(self):
        p = FrozenBacktestParams()
        assert p.freq == 3
        assert p.pos_size == 2
        assert p.commission_rate == 0.0002
        assert p.initial_capital == 1_000_000
        assert p.lookback_window == 252

    def test_regime_gate_defaults(self):
        p = FrozenRegimeGateParams()
        assert p.enabled is True
        assert p.mode == "volatility"
        assert p.proxy_symbol == "510300"
        assert p.thresholds_pct == (25, 30, 40)
        assert p.exposures == (1.0, 0.7, 0.4, 0.1)

    def test_wfo_defaults(self):
        p = FrozenWFOParams()
        assert p.combo_sizes == (2, 3, 4, 5, 6, 7)
        assert p.enable_fdr is True
        assert p.fdr_alpha == 0.05
        assert p.is_period == 180
        assert p.oos_period == 60
        assert p.step_size == 60
        assert p.top_n == 100_000
        assert p.rebalance_frequencies == (3,)

    def test_etf_pool_counts(self):
        p = FrozenETFPool()
        assert p.total_count == 43
        assert p.qdii_count == 5

    def test_scoring_weights(self):
        p = FrozenScoringParams()
        weights_dict = dict(p.weights)
        assert weights_dict["ann_ret"] == 0.4
        assert weights_dict["sharpe"] == 0.3
        assert weights_dict["max_dd"] == 0.3


# ---------------------------------------------------------------------------
# 2. 不可变性
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_backtest_immutable(self):
        p = FrozenBacktestParams()
        with pytest.raises(AttributeError):
            p.freq = 8  # type: ignore[misc]

    def test_etf_pool_immutable(self):
        p = FrozenETFPool()
        with pytest.raises(AttributeError):
            p.symbols = ("only_one",)  # type: ignore[misc]

    def test_production_config_immutable(self):
        config = yaml.safe_load(open(CONFIG_PATH))
        frozen = load_frozen_config(config, config_path=str(CONFIG_PATH))
        with pytest.raises(AttributeError):
            frozen.version = "v999"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. 真实配置校验
# ---------------------------------------------------------------------------


class TestRealConfigValidation:
    def test_strict_passes(self, real_config):
        frozen = load_frozen_config(
            real_config,
            config_path=str(CONFIG_PATH),
            strictness=StrictnessMode.STRICT,
        )
        assert frozen.version == "v4.0"
        assert frozen.config_sha256 is not None

    def test_returns_frozen_config(self, real_config):
        frozen = load_frozen_config(real_config, config_path=str(CONFIG_PATH))
        assert isinstance(frozen, FrozenProductionConfig)
        assert frozen.backtest.freq == 3
        assert frozen.etf_pool.total_count == 43

    def test_v34_still_accessible(self, real_config):
        """v3.4 config should remain accessible for rollback.

        YAML now reflects v4.0 (7 bounded_factors vs v3.4's 4), so use WARN
        mode — the important thing is that the version entry still exists.
        """
        frozen = load_frozen_config(
            real_config,
            config_path=str(CONFIG_PATH),
            version="v3.4",
            strictness=StrictnessMode.WARN,
        )
        assert frozen.version == "v3.4"


# ---------------------------------------------------------------------------
# 4. 篡改检测
# ---------------------------------------------------------------------------


class TestTamperDetection:
    def test_freq_tampered(self, real_config):
        bad = copy.deepcopy(real_config)
        bad["backtest"]["freq"] = 8
        with pytest.raises(FrozenParamViolation) as exc_info:
            load_frozen_config(bad, strictness=StrictnessMode.STRICT)
        assert "backtest.freq" in str(exc_info.value)

    def test_pos_size_tampered(self, real_config):
        bad = copy.deepcopy(real_config)
        bad["backtest"]["pos_size"] = 5
        with pytest.raises(FrozenParamViolation):
            load_frozen_config(bad, strictness=StrictnessMode.STRICT)

    def test_commission_tampered(self, real_config):
        bad = copy.deepcopy(real_config)
        bad["backtest"]["commission_rate"] = 0.001
        with pytest.raises(FrozenParamViolation):
            load_frozen_config(bad, strictness=StrictnessMode.STRICT)

    def test_regime_gate_threshold_tampered(self, real_config):
        bad = copy.deepcopy(real_config)
        bad["backtest"]["regime_gate"]["volatility"]["thresholds_pct"] = [10, 20, 30]
        with pytest.raises(FrozenParamViolation):
            load_frozen_config(bad, strictness=StrictnessMode.STRICT)

    def test_wfo_combo_sizes_tampered(self, real_config):
        bad = copy.deepcopy(real_config)
        bad["combo_wfo"]["combo_sizes"] = [2, 3]
        with pytest.raises(FrozenParamViolation):
            load_frozen_config(bad, strictness=StrictnessMode.STRICT)

    def test_multiple_violations_collected(self, real_config):
        bad = copy.deepcopy(real_config)
        bad["backtest"]["freq"] = 8
        bad["backtest"]["pos_size"] = 5
        bad["backtest"]["commission_rate"] = 0.001
        with pytest.raises(FrozenParamViolation) as exc_info:
            load_frozen_config(bad, strictness=StrictnessMode.STRICT)
        assert len(exc_info.value.violations) == 3


# ---------------------------------------------------------------------------
# 5. ETF 池篡改检测
# ---------------------------------------------------------------------------


class TestETFPoolTamper:
    def test_qdii_removed(self, real_config):
        bad = copy.deepcopy(real_config)
        # Remove all QDII codes
        qdii = {"159920", "513050", "513100", "513130", "513500"}
        bad["data"]["symbols"] = [s for s in bad["data"]["symbols"] if s not in qdii]
        with pytest.raises(FrozenParamViolation) as exc_info:
            load_frozen_config(bad, strictness=StrictnessMode.STRICT)
        assert "QDII" in str(exc_info.value)

    def test_etf_added(self, real_config):
        bad = copy.deepcopy(real_config)
        bad["data"]["symbols"].append("999999")
        with pytest.raises(FrozenParamViolation):
            load_frozen_config(bad, strictness=StrictnessMode.STRICT)


# ---------------------------------------------------------------------------
# 6. WARN 模式
# ---------------------------------------------------------------------------


class TestWarnMode:
    def test_warn_does_not_raise(self, real_config):
        bad = copy.deepcopy(real_config)
        bad["backtest"]["freq"] = 8
        # Should not raise
        frozen = load_frozen_config(bad, strictness=StrictnessMode.WARN)
        assert frozen.version == "v4.0"

    def test_env_var_warn(self, real_config, monkeypatch):
        monkeypatch.setenv("FROZEN_PARAMS_MODE", "warn")
        bad = copy.deepcopy(real_config)
        bad["backtest"]["freq"] = 8
        # Should not raise due to env var
        frozen = load_frozen_config(bad)
        assert frozen.version == "v4.0"


# ---------------------------------------------------------------------------
# 7. 操作性参数不校验
# ---------------------------------------------------------------------------


class TestOperationalParamsIgnored:
    def test_data_dir_change(self, real_config):
        modified = copy.deepcopy(real_config)
        modified["data"]["data_dir"] = "/some/other/path"
        # Should pass without error
        frozen = load_frozen_config(modified, strictness=StrictnessMode.STRICT)
        assert frozen.version == "v4.0"

    def test_n_jobs_change(self, real_config):
        modified = copy.deepcopy(real_config)
        modified["combo_wfo"]["n_jobs"] = 1
        frozen = load_frozen_config(modified, strictness=StrictnessMode.STRICT)
        assert frozen.version == "v4.0"

    def test_start_end_date_change(self, real_config):
        modified = copy.deepcopy(real_config)
        modified["data"]["start_date"] = "2021-01-01"
        modified["data"]["end_date"] = "2026-01-01"
        frozen = load_frozen_config(modified, strictness=StrictnessMode.STRICT)
        assert frozen.version == "v4.0"


# ---------------------------------------------------------------------------
# 8. 版本注册
# ---------------------------------------------------------------------------


class TestVersionRegistry:
    def test_unknown_version_raises(self, real_config):
        with pytest.raises(KeyError, match="未知的冻结参数版本"):
            load_frozen_config(real_config, version="v99.0")

    def test_v40_registered(self, real_config):
        frozen = load_frozen_config(
            real_config, version="v4.0", strictness=StrictnessMode.STRICT
        )
        assert frozen.version == "v4.0"

    def test_v34_and_v40_share_base_params(self, real_config):
        """v4.0 shares core backtest/pool/wfo params with v3.4."""
        v34 = load_frozen_config(
            real_config, version="v3.4", strictness=StrictnessMode.WARN
        )
        v40 = load_frozen_config(
            real_config, version="v4.0", strictness=StrictnessMode.STRICT
        )
        assert v34.backtest == v40.backtest
        assert v34.etf_pool == v40.etf_pool
        assert v34.wfo == v40.wfo
        # cross_section differs: v4.0 has 7 bounded_factors vs v3.4's 4
        assert v34.cross_section != v40.cross_section
        assert len(v40.cross_section.bounded_factors) == 7
        assert len(v34.cross_section.bounded_factors) == 4
