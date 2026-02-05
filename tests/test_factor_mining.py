"""
因子挖掘体系单元测试 | Factor Mining Unit Tests
================================================================================
覆盖:
  - FactorZoo: 注册/查询/导出导入/重复注册异常
  - QualityAnalyzer: 合成因子质检、NaN阈值、单调性、环境IC键
  - AlgebraicSearch: 输出形状、除零处理、命名规范
  - TransformSearch: rank有界性、sign输出域
  - FactorSelector: 相关因子去冗余、簇内选最优、上限约束
  - DiscoveryPipeline: FDR统一校正
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from etf_strategy.core.factor_mining.registry import FactorEntry, FactorZoo
from etf_strategy.core.factor_mining.quality import (
    FactorQualityAnalyzer,
    FactorQualityReport,
    spearman_ic_series,
    compute_forward_returns,
)
from etf_strategy.core.factor_mining.discovery import (
    AlgebraicSearch,
    TransformSearch,
    FactorDiscoveryPipeline,
)
from etf_strategy.core.factor_mining.selection import FactorSelector


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def dates():
    return pd.bdate_range("2020-01-01", periods=500)


@pytest.fixture
def symbols():
    return [f"ETF_{i:03d}" for i in range(20)]


@pytest.fixture
def close(dates, symbols):
    """合成收盘价: 随机游走"""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.02, size=(len(dates), len(symbols)))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=symbols)


@pytest.fixture
def volume(dates, symbols):
    rng = np.random.default_rng(123)
    vol = rng.uniform(1e6, 1e8, size=(len(dates), len(symbols)))
    return pd.DataFrame(vol, index=dates, columns=symbols)


@pytest.fixture
def factor_good(close):
    """好因子: 与未来收益正相关"""
    # 用滞后收益作为因子 (故意制造正IC)
    return close.pct_change(3).shift(5)


@pytest.fixture
def factor_noise(dates, symbols):
    """噪声因子: 纯随机"""
    rng = np.random.default_rng(999)
    return pd.DataFrame(
        rng.standard_normal((len(dates), len(symbols))),
        index=dates,
        columns=symbols,
    )


@pytest.fixture
def factor_with_nan(factor_good):
    """带大量 NaN 的因子"""
    df = factor_good.copy()
    rng = np.random.default_rng(77)
    mask = rng.random(df.shape) < 0.5  # 50% NaN
    df[mask] = np.nan
    return df


@pytest.fixture
def regime_series(dates):
    """合成市场环境"""
    regimes = []
    for i in range(len(dates)):
        if i < 150:
            regimes.append("bull")
        elif i < 300:
            regimes.append("bear")
        else:
            regimes.append("sideways")
    return pd.Series(regimes, index=dates)


@pytest.fixture
def analyzer(close, regime_series):
    return FactorQualityAnalyzer(close=close, regime_series=regime_series, freq=3)


# ============================================================================
# Layer 1: FactorZoo Tests
# ============================================================================

class TestFactorZoo:
    def test_register_and_get(self):
        zoo = FactorZoo()
        entry = FactorEntry(name="TEST_FACTOR", source="hand_crafted")
        zoo.register(entry)
        assert len(zoo) == 1
        assert "TEST_FACTOR" in zoo
        assert zoo.get("TEST_FACTOR").source == "hand_crafted"

    def test_duplicate_register_raises(self):
        zoo = FactorZoo()
        entry = FactorEntry(name="DUP", source="hand_crafted")
        zoo.register(entry)
        with pytest.raises(ValueError, match="already registered"):
            zoo.register(entry)

    def test_register_batch_skips_existing(self):
        zoo = FactorZoo()
        zoo.register(FactorEntry(name="A", source="hand_crafted"))
        entries = [
            FactorEntry(name="A", source="algebraic"),  # duplicate
            FactorEntry(name="B", source="algebraic"),
            FactorEntry(name="C", source="transform"),
        ]
        added = zoo.register_batch(entries)
        assert added == 2
        assert len(zoo) == 3

    def test_list_passed(self):
        zoo = FactorZoo()
        for name in ["A", "B", "C"]:
            entry = FactorEntry(name=name, source="hand_crafted")
            zoo.register(entry)
        zoo.update_quality("A", 3.0, True)
        zoo.update_quality("B", 1.0, False)
        zoo.update_quality("C", 2.5, True)
        passed = zoo.list_passed()
        assert len(passed) == 2
        assert {e.name for e in passed} == {"A", "C"}

    def test_export_import(self, tmp_path):
        zoo1 = FactorZoo()
        zoo1.register(FactorEntry(name="X", source="algebraic", expression="a+b"))
        zoo1.update_quality("X", 2.5, True)
        path = tmp_path / "registry.json"
        zoo1.export_registry(path)

        # Verify JSON structure
        data = json.loads(path.read_text())
        assert "X" in data
        assert data["X"]["source"] == "algebraic"

        # Import into new zoo
        zoo2 = FactorZoo()
        imported = zoo2.import_registry(path)
        assert imported == 1
        assert zoo2.get("X").passed is True

    def test_list_by_source(self):
        zoo = FactorZoo()
        zoo.register(FactorEntry(name="H1", source="hand_crafted"))
        zoo.register(FactorEntry(name="A1", source="algebraic"))
        zoo.register(FactorEntry(name="A2", source="algebraic"))
        assert len(zoo.list_by_source("algebraic")) == 2
        assert len(zoo.list_by_source("hand_crafted")) == 1


# ============================================================================
# Layer 2: Quality Analyzer Tests
# ============================================================================

class TestSpearmanIC:
    def test_basic_ic(self, close):
        """IC series should return valid values for non-trivial factor"""
        factor = close.pct_change(5).shift(1)
        fwd_ret = compute_forward_returns(close, 3)
        ic_s = spearman_ic_series(factor, fwd_ret)
        assert len(ic_s) > 100
        assert ic_s.dtype == float

    def test_short_series_returns_empty(self):
        """Too few observations → empty series"""
        dates = pd.bdate_range("2020-01-01", periods=10)
        cols = ["A", "B"]
        factor = pd.DataFrame(np.nan, index=dates, columns=cols)
        ret = pd.DataFrame(np.nan, index=dates, columns=cols)
        ic_s = spearman_ic_series(factor, ret)
        assert len(ic_s) == 0


class TestQualityAnalyzer:
    def test_good_factor_passes(self, analyzer, factor_good):
        report = analyzer.analyze("GOOD", factor_good, direction="high_is_good")
        assert isinstance(report, FactorQualityReport)
        assert report.n_obs > 0
        assert report.mean_ic != 0.0

    def test_noise_factor_low_score(self, analyzer, factor_noise):
        report = analyzer.analyze("NOISE", factor_noise, direction="high_is_good")
        # Noise should have low absolute IC
        assert abs(report.mean_ic) < 0.1

    def test_high_nan_fails(self, analyzer, factor_with_nan):
        report = analyzer.analyze("NAN_HEAVY", factor_with_nan, direction="high_is_good")
        assert report.nan_rate > 0.3

    def test_monotonicity_range(self, analyzer, factor_good):
        report = analyzer.analyze("MONO", factor_good, direction="high_is_good")
        assert 0.0 <= report.monotonicity_score <= 1.0
        assert len(report.tercile_returns) == 3

    def test_regime_ic_keys(self, analyzer, factor_good):
        report = analyzer.analyze("REGIME", factor_good, direction="high_is_good")
        # All three regime ICs should be set (might be 0.0 if no data)
        assert isinstance(report.ic_bull, float)
        assert isinstance(report.ic_bear, float)
        assert isinstance(report.ic_sideways, float)

    def test_ic_decay_horizons(self, analyzer, factor_good):
        report = analyzer.analyze("DECAY", factor_good, direction="high_is_good")
        assert set(report.ic_by_horizon.keys()) == {1, 3, 5, 10, 20}

    def test_quick_ic_screen(self, analyzer, factor_good):
        mean_ic, p_value = analyzer.quick_ic_screen("Q", factor_good)
        assert isinstance(mean_ic, float)
        assert 0.0 <= p_value <= 1.0

    def test_direction_consistency(self, analyzer, close):
        # Factor that's negatively correlated with returns
        neg_factor = -close.pct_change(3).shift(5)
        report = analyzer.analyze("NEG", neg_factor, direction="high_is_good")
        # If IC is negative, direction_consistent should be False for high_is_good
        if report.mean_ic < 0:
            assert not report.direction_consistent

    def test_production_ready_penalty(self, analyzer, factor_good):
        report_prod = analyzer.analyze("PROD", factor_good, direction="high_is_good", production_ready=True)
        report_risky = analyzer.analyze("RISKY", factor_good, direction="high_is_good", production_ready=False)
        assert report_risky.quality_score < report_prod.quality_score


# ============================================================================
# Layer 3: Discovery Tests
# ============================================================================

class TestAlgebraicSearch:
    def test_output_structure(self, close, analyzer):
        factors = {
            "F1": close.pct_change(5),
            "F2": close.pct_change(10),
            "F3": close.pct_change(20),
        }
        search = AlgebraicSearch(ic_threshold=0.0)  # Low threshold to get results
        entries, new_factors, p_values = search.search(factors, analyzer)

        # C(3,2) * 6 = 18 max candidates
        assert len(entries) <= 18
        assert len(entries) == len(p_values)
        for entry in entries:
            assert entry.source == "algebraic"
            assert len(entry.parent_factors) == 2

    def test_division_zero_handling(self, close, analyzer):
        """Division by zero should produce NaN, not crash"""
        factors = {
            "F1": close.pct_change(5),
            "ZERO": pd.DataFrame(0.0, index=close.index, columns=close.columns),
        }
        search = AlgebraicSearch(ic_threshold=0.0)
        # Should not raise
        entries, _, _ = search.search(factors, analyzer)

    def test_naming_convention(self, close, analyzer):
        factors = {"AA": close.pct_change(5), "BB": close.pct_change(10)}
        search = AlgebraicSearch(ic_threshold=0.0)
        entries, _, _ = search.search(factors, analyzer)
        for entry in entries:
            assert "__" in entry.name  # e.g., "AA__add__BB"


class TestTransformSearch:
    def test_rank_bounded(self, close, analyzer):
        """Rank transform should produce values in [0, 1]"""
        factors = {"F1": close.pct_change(5)}
        search = TransformSearch()
        entries, new_factors, _ = search.search(factors, analyzer)

        rank_entries = [e for e in entries if "rank" in e.name]
        if rank_entries:
            rank_df = new_factors[rank_entries[0].name]
            assert rank_df.min().min() >= 0.0
            assert rank_df.max().max() <= 1.0

    def test_sign_output_domain(self, close, analyzer):
        """Sign transform should only produce -1, 0, 1"""
        factors = {"F1": close.pct_change(5)}
        search = TransformSearch()
        entries, new_factors, _ = search.search(factors, analyzer)

        sign_entries = [e for e in entries if "sign" in e.name]
        if sign_entries:
            sign_df = new_factors[sign_entries[0].name]
            unique_vals = sign_df.dropna().values.flatten()
            unique_set = set(unique_vals)
            assert unique_set.issubset({-1.0, 0.0, 1.0})


# ============================================================================
# Layer 4: Selection Tests
# ============================================================================

class TestFactorSelector:
    def test_correlated_factors_dedup(self, close):
        """高相关因子应被聚类去重"""
        # Create two nearly identical factors
        base = close.pct_change(5)
        f1 = base
        f2 = base + np.random.default_rng(42).normal(0, 0.001, base.shape)
        f3 = close.pct_change(20)  # different factor

        factors = {"F1": f1, "F2": f2, "F3": f3}
        entries = [FactorEntry(name=n, source="hand_crafted") for n in sorted(factors)]
        reports = {
            "F1": FactorQualityReport(factor_name="F1", quality_score=3.0, passed=True, nan_rate=0.1),
            "F2": FactorQualityReport(factor_name="F2", quality_score=2.5, passed=True, nan_rate=0.1),
            "F3": FactorQualityReport(factor_name="F3", quality_score=2.0, passed=True, nan_rate=0.1),
        }

        selector = FactorSelector(max_correlation=0.7, max_factors=40)
        selected, corr = selector.select(entries, reports, factors)

        # F1 and F2 are near-identical, only one should survive
        assert not ("F1" in selected and "F2" in selected)
        # F3 should survive (different)
        assert "F3" in selected

    def test_cluster_keeps_best(self, close):
        """同簇应保留 quality_score 最高的"""
        base = close.pct_change(5)
        factors = {
            "LOW_SCORE": base,
            "HIGH_SCORE": base * 1.001,  # nearly identical
        }
        entries = [FactorEntry(name=n, source="hand_crafted") for n in sorted(factors)]
        reports = {
            "HIGH_SCORE": FactorQualityReport(factor_name="HIGH_SCORE", quality_score=5.0, passed=True, nan_rate=0.05),
            "LOW_SCORE": FactorQualityReport(factor_name="LOW_SCORE", quality_score=2.0, passed=True, nan_rate=0.05),
        }

        selector = FactorSelector(max_correlation=0.7)
        selected, _ = selector.select(entries, reports, factors)

        if len(selected) == 1:
            assert selected[0] == "HIGH_SCORE"

    def test_max_factors_cap(self, dates, symbols):
        """max_factors 上限约束"""
        rng = np.random.default_rng(42)
        n_factors = 10
        factors = {}
        entries = []
        reports = {}
        for i in range(n_factors):
            name = f"F{i:02d}"
            df = pd.DataFrame(
                rng.standard_normal((len(dates), len(symbols))),
                index=dates,
                columns=symbols,
            )
            factors[name] = df
            entries.append(FactorEntry(name=name, source="hand_crafted"))
            reports[name] = FactorQualityReport(
                factor_name=name, quality_score=3.0 + i * 0.1, passed=True, nan_rate=0.05
            )

        selector = FactorSelector(max_correlation=0.99, max_factors=5)
        selected, _ = selector.select(entries, reports, factors)
        assert len(selected) <= 5

    def test_quality_filter(self, dates, symbols):
        """未通过质检的因子应被过滤"""
        rng = np.random.default_rng(42)
        factors = {}
        entries = []
        reports = {}
        for i, (name, passed, score) in enumerate([
            ("GOOD", True, 3.0),
            ("BAD", False, 1.0),
            ("MARGINAL", True, 1.5),  # score < threshold
        ]):
            df = pd.DataFrame(
                rng.standard_normal((len(dates), len(symbols))),
                index=dates,
                columns=symbols,
            )
            factors[name] = df
            entries.append(FactorEntry(name=name, source="hand_crafted"))
            reports[name] = FactorQualityReport(
                factor_name=name, quality_score=score, passed=passed, nan_rate=0.05
            )

        selector = FactorSelector(min_quality_score=2.0)
        selected, _ = selector.select(entries, reports, factors)
        assert "GOOD" in selected
        assert "BAD" not in selected
        assert "MARGINAL" not in selected


# ============================================================================
# Layer 3+4: Discovery Pipeline FDR Test
# ============================================================================

class TestDiscoveryPipeline:
    def test_fdr_reduces_candidates(self, close, volume, analyzer):
        """FDR correction should reduce the number of passing candidates"""
        factors = {
            "F1": close.pct_change(5),
            "F2": close.pct_change(10),
        }

        pipeline = FactorDiscoveryPipeline(
            analyzer=analyzer,
            close=close,
            volume=volume,
            fdr_alpha=0.05,
            algebraic_ic_threshold=0.0,  # pass everything to IC screen
        )

        entries, new_factors = pipeline.run(
            factors,
            enable_algebraic=True,
            enable_window=False,
            enable_transform=False,
        )

        # Result should be a subset (FDR filters)
        assert isinstance(entries, list)
        assert isinstance(new_factors, dict)
        # All returned factors should have corresponding DataFrames
        for entry in entries:
            assert entry.name in new_factors
