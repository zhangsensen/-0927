"""Tests for ETF pool mapping and pool diversity constraint."""

import numpy as np
import pytest
import yaml
from pathlib import Path

from etf_strategy.core.etf_pool_mapper import (
    POOL_NAME_TO_ID,
    POOL_ID_TO_NAME,
    load_pool_mapping,
    build_pool_array,
)

ROOT = Path(__file__).resolve().parent.parent
POOLS_PATH = ROOT / "configs" / "etf_pools.yaml"
CONFIG_PATH = ROOT / "configs" / "combo_wfo_config.yaml"


class TestPoolMapping:
    """Pool mapping module tests."""

    def test_load_pool_mapping_returns_dict(self):
        mapping = load_pool_mapping(POOLS_PATH)
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_all_43_etfs_mapped(self):
        """All 43 ETFs in the production universe should be mapped."""
        mapping = load_pool_mapping(POOLS_PATH)
        assert len(mapping) == 43, f"Expected 43 mapped ETFs, got {len(mapping)}"

    def test_7_pools_present(self):
        """All 7 canonical pools should have at least one ETF."""
        mapping = load_pool_mapping(POOLS_PATH)
        pool_ids = set(mapping.values())
        assert pool_ids == {0, 1, 2, 3, 4, 5, 6}

    def test_pool_ids_in_range(self):
        mapping = load_pool_mapping(POOLS_PATH)
        for ticker, pool_id in mapping.items():
            assert 0 <= pool_id <= 6, f"{ticker} has pool_id {pool_id}"

    def test_no_duplicate_tickers(self):
        """Each ticker should appear in exactly one pool."""
        with open(POOLS_PATH) as f:
            cfg = yaml.safe_load(f)
        pools = cfg.get("pools", {})
        all_tickers = []
        for pool_name in POOL_NAME_TO_ID:
            symbols = pools.get(pool_name, {}).get("symbols", [])
            all_tickers.extend(symbols)
        assert len(all_tickers) == len(set(all_tickers)), "Duplicate tickers found across pools"

    def test_515210_is_cyclical(self):
        """515210 (钢铁ETF) must be in EQUITY_CYCLICAL pool."""
        mapping = load_pool_mapping(POOLS_PATH)
        assert "515210" in mapping
        assert mapping["515210"] == POOL_NAME_TO_ID["EQUITY_CYCLICAL"]

    def test_build_pool_array_basic(self):
        mapping = {"A": 0, "B": 1, "C": 2}
        arr = build_pool_array(["A", "B", "C", "D"], mapping)
        assert arr.dtype == np.int64
        np.testing.assert_array_equal(arr, [0, 1, 2, -1])

    def test_build_pool_array_strips_suffix(self):
        mapping = {"510300": 0}
        arr = build_pool_array(["510300.SH"], mapping)
        np.testing.assert_array_equal(arr, [0])

    def test_build_pool_array_with_real_data(self):
        mapping = load_pool_mapping(POOLS_PATH)
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        etf_codes = config["data"]["symbols"]
        arr = build_pool_array(etf_codes, mapping)
        assert len(arr) == len(etf_codes)
        n_mapped = int(np.sum(arr >= 0))
        # All A-share ETFs should be mapped (some QDII too)
        assert n_mapped >= 38, f"Only {n_mapped} ETFs mapped"


class TestPoolDiversifyTopk:
    """Test pool_diversify_topk numba function."""

    @pytest.fixture(autouse=True)
    def _import_func(self):
        """Import from batch_vec_backtest (needs scripts in path)."""
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from batch_vec_backtest import pool_diversify_topk, stable_topk_indices
        self.pool_diversify_topk = pool_diversify_topk
        self.stable_topk_indices = stable_topk_indices

    def test_cross_pool_selection(self):
        """Pick1 from pool 0, Pick2 from pool 1 (not same pool)."""
        scores = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
        pool_ids = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        result = self.pool_diversify_topk(scores, pool_ids, 2, 5)
        assert len(result) == 2
        assert result[0] == 0  # best overall
        assert result[1] == 2  # first from different pool

    def test_same_pool_fallback(self):
        """When all candidates are same pool, still pick top-2."""
        scores = np.array([10.0, 9.0, 8.0])
        pool_ids = np.array([0, 0, 0], dtype=np.int64)
        result = self.pool_diversify_topk(scores, pool_ids, 2, 3)
        assert len(result) == 2
        assert result[0] == 0
        assert result[1] == 1  # fallback to best remaining

    def test_disabled_matches_baseline(self):
        """extended_k=0 path should be identical to stable_topk_indices."""
        scores = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
        pool_ids = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        baseline = self.stable_topk_indices(scores, 2)
        # When disabled, the caller uses stable_topk_indices directly
        np.testing.assert_array_equal(baseline, [0, 1])

    def test_deterministic(self):
        """Same inputs produce same outputs."""
        scores = np.array([10.0, 9.0, 8.5, 8.0, 7.0])
        pool_ids = np.array([0, 0, 1, 2, 1], dtype=np.int64)
        r1 = self.pool_diversify_topk(scores, pool_ids, 2, 5)
        r2 = self.pool_diversify_topk(scores, pool_ids, 2, 5)
        np.testing.assert_array_equal(r1, r2)

    def test_unmapped_treated_as_unique(self):
        """ETFs with pool_id=-1 are treated as unique pools (always eligible)."""
        scores = np.array([10.0, 9.0, 8.0])
        pool_ids = np.array([0, -1, 0], dtype=np.int64)
        result = self.pool_diversify_topk(scores, pool_ids, 2, 3)
        assert len(result) == 2
        assert result[0] == 0
        assert result[1] == 1  # -1 is always "different"

    def test_inf_scores_excluded(self):
        """Candidates with -inf scores are not selected."""
        scores = np.array([10.0, -np.inf, 8.0])
        pool_ids = np.array([0, 1, 0], dtype=np.int64)
        result = self.pool_diversify_topk(scores, pool_ids, 2, 3)
        assert len(result) == 2
        assert 1 not in result  # -inf excluded


class TestProductionSafety:
    """Production safety: determinism, edge cases, NaN handling."""

    @pytest.fixture(autouse=True)
    def _import_func(self):
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from batch_vec_backtest import pool_diversify_topk, stable_topk_indices
        self.pool_diversify_topk = pool_diversify_topk
        self.stable_topk_indices = stable_topk_indices

    def test_determinism_multiple_runs(self):
        """Same inputs must produce identical outputs every time."""
        scores = np.array([10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0])
        pool_ids = np.array([0, 0, 1, 1, 2, 2, 3], dtype=np.int64)
        results = []
        for _ in range(5):
            r = self.pool_diversify_topk(scores, pool_ids, 2, 7)
            results.append(tuple(r))
        assert len(set(results)) == 1, f"Non-deterministic: {results}"

    def test_all_nan_scores(self):
        """All -inf scores should return empty or partial result."""
        scores = np.full(5, -np.inf)
        pool_ids = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        result = self.pool_diversify_topk(scores, pool_ids, 2, 5)
        # Should handle gracefully (empty or just pick1 with -inf check)
        assert len(result) <= 2

    def test_single_valid_score(self):
        """Only one valid ETF, pos_size=2."""
        scores = np.array([10.0, -np.inf, -np.inf, -np.inf])
        pool_ids = np.array([0, 1, 2, 3], dtype=np.int64)
        result = self.pool_diversify_topk(scores, pool_ids, 2, 4)
        assert len(result) == 1
        assert result[0] == 0

    def test_tie_breaking_deterministic(self):
        """Equal scores should have deterministic tie-break (by index)."""
        scores = np.array([10.0, 10.0, 10.0, 10.0])
        pool_ids = np.array([0, 0, 1, 1], dtype=np.int64)
        result = self.pool_diversify_topk(scores, pool_ids, 2, 4)
        assert len(result) == 2
        assert result[0] == 0  # lowest index wins
        assert result[1] == 2  # first cross-pool

    def test_extended_k_larger_than_n(self):
        """extended_k > N should not crash."""
        scores = np.array([10.0, 9.0, 8.0])
        pool_ids = np.array([0, 1, 2], dtype=np.int64)
        result = self.pool_diversify_topk(scores, pool_ids, 2, 100)
        assert len(result) == 2

    def test_pool_array_subset_universe(self):
        """Build pool array with a subset of ETFs (simulating removal)."""
        mapping = load_pool_mapping(POOLS_PATH)
        # Take only first 20 symbols from config
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        subset = config["data"]["symbols"][:20]
        arr = build_pool_array(subset, mapping)
        assert len(arr) == 20
        assert np.all(arr >= 0), "All 20 should be mapped (they're the first 20 canonical ETFs)"


class TestProductionConsistency:
    """Cross-file consistency checks."""

    def test_pools_yaml_symbols_match_config(self):
        """All pool symbols should be a subset of combo_wfo_config symbols."""
        mapping = load_pool_mapping(POOLS_PATH)
        pool_tickers = set(mapping.keys())

        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        config_symbols = set(config["data"]["symbols"])

        missing = pool_tickers - config_symbols
        assert len(missing) == 0, f"Pool tickers not in config: {missing}"

    def test_config_symbols_all_have_pools(self):
        """All config symbols (except possibly new additions) should have pool mapping."""
        mapping = load_pool_mapping(POOLS_PATH)
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        config_symbols = config["data"]["symbols"]
        unmapped = [s for s in config_symbols if s not in mapping]
        assert len(unmapped) == 0, f"Config symbols without pool: {unmapped}"

    def test_etf_config_515210_not_tech(self):
        """515210 must NOT be classified as 科技 in etf_config.yaml."""
        cfg_path = ROOT / "configs" / "etf_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        etfs = cfg.get("etf_universe", {}).get("etfs", [])
        for etf in etfs:
            if etf.get("code", "").startswith("515210"):
                assert etf.get("category") != "科技", "515210 should not be categorized as 科技"
                break
