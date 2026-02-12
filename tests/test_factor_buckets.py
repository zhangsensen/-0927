"""Tests for factor bucket constraints."""

import pytest

from etf_strategy.core.factor_buckets import (
    FACTOR_BUCKETS,
    FACTOR_TO_BUCKET,
    check_cross_bucket_constraint,
    generate_cross_bucket_combos,
    get_bucket_coverage,
)

# ─── 固定的 active_factors (与 v5.0 config 一致) ───────────────────
ACTIVE_FACTORS_V50 = [
    "ADX_14D",
    "AMIHUD_ILLIQUIDITY",
    "BREAKOUT_20D",
    "CALMAR_RATIO_60D",
    "CORRELATION_TO_MARKET_20D",
    "GK_VOL_RATIO_20D",
    "MAX_DD_60D",
    "MOM_20D",
    "OBV_SLOPE_10D",
    "PRICE_POSITION_20D",
    "PRICE_POSITION_120D",
    "PV_CORR_20D",
    "SHARPE_RATIO_20D",
    "SLOPE_20D",
    "UP_DOWN_VOL_RATIO_20D",
    "VOL_RATIO_20D",
    "VORTEX_14D",
]


class TestBucketMapping:
    """桶映射完整性."""

    def test_all_active_factors_mapped(self):
        """所有 active_factors 都有桶分配."""
        for f in ACTIVE_FACTORS_V50:
            assert f in FACTOR_TO_BUCKET, f"{f} not mapped to any bucket"

    def test_bucket_count(self):
        """7 个桶: 5 OHLCV + 2 non-OHLCV (FUND_FLOW, LEVERAGE)."""
        assert len(FACTOR_BUCKETS) == 7

    def test_no_duplicate_factors(self):
        """每个因子只属于一个桶."""
        all_factors = []
        for factors in FACTOR_BUCKETS.values():
            all_factors.extend(factors)
        assert len(all_factors) == len(set(all_factors)), "Duplicate factor in buckets"

    def test_reverse_mapping_consistent(self):
        """反向映射与正向映射一致."""
        for bucket_name, factors in FACTOR_BUCKETS.items():
            for f in factors:
                assert FACTOR_TO_BUCKET[f] == bucket_name


class TestCrossBucketConstraint:
    """跨桶约束检查."""

    def test_s1_passes(self):
        """S1 覆盖 3 桶, 应通过 min_buckets=3."""
        s1 = ["ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"]
        ok, _ = check_cross_bucket_constraint(s1, min_buckets=3)
        assert ok

    def test_champion_fails(self):
        """冠军只覆盖 2 桶, 应不通过 min_buckets=3."""
        champ = ["AMIHUD_ILLIQUIDITY", "PRICE_POSITION_20D", "PV_CORR_20D", "SLOPE_20D"]
        ok, _ = check_cross_bucket_constraint(champ, min_buckets=3)
        assert not ok

    def test_max_per_bucket_enforced(self):
        """同桶超过 max_per_bucket 应不通过."""
        # 3 个趋势桶因子 + 1 个其他
        three_trend = ["MOM_20D", "SHARPE_RATIO_20D", "SLOPE_20D", "ADX_14D"]
        ok, reason = check_cross_bucket_constraint(three_trend, max_per_bucket=2)
        assert not ok
        assert "TREND_MOMENTUM" in reason

    def test_coverage_report(self):
        """覆盖报告正确."""
        s1 = ["ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"]
        cov = get_bucket_coverage(s1)
        assert "TREND_STRENGTH_RISK" in cov
        assert "VOLUME_CONFIRMATION" in cov
        assert "TREND_MOMENTUM" in cov
        assert len(cov) == 3


class TestComboGeneration:
    """组合生成正确性."""

    def test_no_duplicates(self):
        """生成的组合无重复."""
        combos = generate_cross_bucket_combos(
            ACTIVE_FACTORS_V50, combo_size=4, min_buckets=3
        )
        combo_set = set(combos)
        assert len(combos) == len(combo_set), "Duplicate combos generated"

    def test_stable_ordering(self):
        """两次生成结果完全一致 (排序稳定性)."""
        combos1 = generate_cross_bucket_combos(
            ACTIVE_FACTORS_V50, combo_size=4, min_buckets=3
        )
        combos2 = generate_cross_bucket_combos(
            ACTIVE_FACTORS_V50, combo_size=4, min_buckets=3
        )
        assert combos1 == combos2

    def test_all_combos_satisfy_constraint(self):
        """生成的所有组合都满足跨桶约束."""
        for size in [3, 4, 5]:
            combos = generate_cross_bucket_combos(
                ACTIVE_FACTORS_V50, combo_size=size, min_buckets=3, max_per_bucket=2
            )
            for combo in combos:
                ok, reason = check_cross_bucket_constraint(
                    list(combo), min_buckets=min(size, 3), max_per_bucket=2
                )
                assert ok, f"Combo {combo} violates constraint: {reason}"

    def test_expected_counts_v50(self):
        """组合数量与预期一致 (防止桶映射变化导致静默回归)."""
        expected = {
            3: 340,
            4: 1876,
            5: 4501,
        }
        for size, expected_count in expected.items():
            combos = generate_cross_bucket_combos(
                ACTIVE_FACTORS_V50, combo_size=size, min_buckets=3, max_per_bucket=2
            )
            assert len(combos) == expected_count, (
                f"Size {size}: expected {expected_count}, got {len(combos)}. "
                f"Bucket mapping may have changed."
            )

    def test_s1_in_generated_combos(self):
        """S1 组合必须出现在生成结果中."""
        combos = generate_cross_bucket_combos(
            ACTIVE_FACTORS_V50, combo_size=4, min_buckets=3, max_per_bucket=2
        )
        s1 = tuple(sorted(["ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"]))
        assert s1 in combos, f"S1 {s1} not in generated combos"

    def test_size2_uses_min_buckets_2(self):
        """size=2 时 min_buckets 应降到 2."""
        combos = generate_cross_bucket_combos(
            ACTIVE_FACTORS_V50, combo_size=2, min_buckets=2, max_per_bucket=2
        )
        assert len(combos) > 0
        # All should cover at least 2 buckets
        for combo in combos:
            cov = get_bucket_coverage(list(combo))
            assert len(cov) >= 2, f"{combo} covers only {len(cov)} bucket(s)"
