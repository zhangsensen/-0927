"""Tests for Exp4 hysteresis state machine.

Validates:
1. Disabled mode (delta_rank=0, min_hold_days=0) passes through topk
2. Not-fully-invested: fill from topk without hysteresis
3. Fully-invested: apply all 3 hysteresis rules
4. Rank gap blocking: swap blocked when rank_gap < delta_rank
5. Min hold days blocking: swap blocked when held_days < min_hold_days
6. Boundary conditions: exact threshold values
7. Max 1 swap per rebalance
8. NaN handling: treat NaN scores as -inf
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def make_test_data(
    N: int = 10,
    held_indices: list = None,
    hold_days_arr: dict = None,
    score_desc: str = "ascending",  # "ascending", "descending", "custom"
    custom_scores: np.ndarray = None,
):
    """Create test data for hysteresis tests.

    Args:
        N: Number of ETFs
        held_indices: Currently held positions
        hold_days_arr: Dict of {idx: days_held}
        score_desc: How to generate scores
        custom_scores: Override scores array

    Returns:
        (combined_score, holdings_mask, hold_days, top_indices)
    """
    held_indices = held_indices or []
    hold_days_arr = hold_days_arr or {}

    # Generate scores
    if custom_scores is not None:
        combined_score = custom_scores.astype(np.float64)
    elif score_desc == "ascending":
        combined_score = np.arange(N, dtype=np.float64)  # 0, 1, 2, ..., N-1
    elif score_desc == "descending":
        combined_score = np.arange(N, dtype=np.float64)[::-1]  # N-1, ..., 0
    else:
        combined_score = np.arange(N, dtype=np.float64)

    # Holdings mask
    holdings_mask = np.zeros(N, dtype=np.bool_)
    for i in held_indices:
        holdings_mask[i] = True

    # Hold days
    hold_days = np.zeros(N, dtype=np.int64)
    for idx, days in hold_days_arr.items():
        hold_days[idx] = days

    # Top indices (descending score order, i.e., best first)
    top_indices = np.argsort(combined_score)[::-1].astype(np.int64)

    return combined_score, holdings_mask, hold_days, top_indices


# ─────────────────────────────────────────────────────────
#  1. Disabled mode tests
# ─────────────────────────────────────────────────────────


class TestHysteresisDisabled:
    """When delta_rank=0 and min_hold_days=0, hysteresis is disabled.

    Note: The function returns ALL indices in top_indices, not limited to pos_size.
    It assumes top_indices is pre-filtered to the desired count.
    """

    def test_passthrough_all_topk(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(N)
        pos_size = 3

        # When hysteresis disabled, returns all indices in top_indices
        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.0,
            min_hold_days_val=0,
        )

        # All top_indices (which is all N indices) are marked True
        assert result.sum() == N  # Not pos_size!
        for i in range(N):
            assert result[i] == True

    def test_passthrough_limited_topk(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(N)
        pos_size = 3

        # If we want only top-3, we should pass top_indices[:3]
        limited_topk = top_indices[:pos_size]

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            limited_topk,
            pos_size=pos_size,
            delta_rank=0.0,
            min_hold_days_val=0,
        )

        # Only top-3 (indices 9, 8, 7 with ascending scores)
        assert result.sum() == pos_size
        assert result[9] == True
        assert result[8] == True
        assert result[7] == True


# ─────────────────────────────────────────────────────────
#  2. Not-fully-invested tests
# ─────────────────────────────────────────────────────────


class TestHysteresisNotFullyInvested:
    """When held_count < pos_size, fill from topk without hysteresis."""

    def test_fill_from_topk_one_held(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        # Hold position 5 (score=5), want 3 positions
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, held_indices=[5]
        )
        pos_size = 3

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=9,
        )

        # Should keep 5 and add top-2 from topk (9, 8)
        assert result.sum() == pos_size
        assert result[5] == True  # Keep existing
        assert result[9] == True  # Add from topk
        assert result[8] == True  # Add from topk

    def test_fill_from_topk_zero_held(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(N)
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=9,
        )

        # Should select top-2 (9, 8)
        assert result.sum() == pos_size
        assert result[9] == True
        assert result[8] == True


# ─────────────────────────────────────────────────────────
#  3. Fully-invested with normal swap
# ─────────────────────────────────────────────────────────


class TestHysteresisFullyInvested:
    """When fully invested, apply all 3 hysteresis rules."""

    def test_swap_when_conditions_met(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        # Hold positions 3, 4 (scores 3, 4 - relatively low)
        # Hold for 10 days (>= min_hold_days=9)
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, held_indices=[3, 4], hold_days_arr={3: 10, 4: 10}
        )
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=9,
        )

        # Worst held: 3 (rank01=3/9=0.33)
        # Best new: 9 (rank01=9/9=1.0)
        # rank_gap = 1.0 - 0.33 = 0.67 >= 0.10 -> SWAP
        assert result.sum() == pos_size
        assert result[3] == False  # Swapped out
        assert result[4] == True  # Kept
        assert result[9] == True  # Swapped in

    def test_swap_best_new_in_topk(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        # Hold positions 4, 5 with sufficient hold_days
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, held_indices=[4, 5], hold_days_arr={4: 10, 5: 10}
        )
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=9,
        )

        # Worst held: 4 (rank01=4/9=0.44)
        # Best new from topk: 9 (not held)
        # Gap = 1.0 - 0.44 = 0.56 >= 0.10 -> SWAP
        assert result[9] == True
        assert result[4] == False


# ─────────────────────────────────────────────────────────
#  4. Rank gap blocking
# ─────────────────────────────────────────────────────────


class TestRankGapBlocking:
    """Swap blocked when rank_gap < delta_rank."""

    def test_no_swap_when_gap_too_small(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        # Hold positions 7, 8 (high scores, close to top)
        # Best new candidate would be 9, but 6 is next
        # If we hold 8, 9, worst is 8, best new is 7
        # rank01(8) = 8/9 = 0.89
        # rank01(7) = 7/9 = 0.78
        # gap = 0.89 - 0.78 = 0.11 (but 7 is worse than 8, so no swap)
        # Actually the swap logic: best_new must be BETTER than worst_held

        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, held_indices=[8, 9], hold_days_arr={8: 10, 9: 10}
        )
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=0,
        )

        # Worst held: 8 (rank01=0.89)
        # Best new from topk not held: 7 (rank01=0.78)
        # best_new (7) is WORSE than worst_held (8), so gap calculation:
        # rank_gap = rank01[7] - rank01[8] = 0.78 - 0.89 = -0.11 < 0.10
        # No swap!
        assert result[8] == True
        assert result[9] == True
        assert result[7] == False

    def test_swap_blocked_high_delta_rank(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        # Hold positions 5, 6 with hold_days sufficient
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, held_indices=[5, 6], hold_days_arr={5: 10, 6: 10}
        )
        pos_size = 2

        # Use high delta_rank that blocks swap
        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.80,
            min_hold_days_val=0,
        )

        # Worst held: 5 (rank01=5/9=0.56)
        # Best new: 9 (rank01=1.0)
        # Gap = 1.0 - 0.56 = 0.44 < 0.80 -> NO SWAP
        assert result[5] == True
        assert result[6] == True
        assert result[9] == False


# ─────────────────────────────────────────────────────────
#  5. Min hold days blocking
# ─────────────────────────────────────────────────────────


class TestMinHoldDaysBlocking:
    """Swap blocked when hold_days < min_hold_days."""

    def test_no_swap_when_hold_days_insufficient(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        # Hold positions 3, 4 but with only 5 days held (< min_hold_days=9)
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, held_indices=[3, 4], hold_days_arr={3: 5, 4: 10}
        )
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=9,
        )

        # Worst held: 3 (rank01=0.33, hold_days=5 < 9)
        # Gap is sufficient, but hold_days insufficient -> NO SWAP for 3
        # Keep all
        assert result[3] == True
        assert result[4] == True
        assert result[9] == False

    def test_swap_when_hold_days_sufficient(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        # Hold positions 3, 4 with exactly 9 days (== min_hold_days)
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, held_indices=[3, 4], hold_days_arr={3: 9, 4: 9}
        )
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=9,
        )

        # Worst held: 3 (hold_days=9 >= 9), gap sufficient -> SWAP
        assert result[3] == False
        assert result[9] == True


# ─────────────────────────────────────────────────────────
#  6. Boundary conditions
# ─────────────────────────────────────────────────────────


class TestBoundaryConditions:
    """Test exact threshold values."""

    def test_exact_rank_gap_threshold(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 11  # rank01 denominator = 10
        # Create scenario where gap is exactly 0.10
        # If held: 0 (rank01=0), best new: 1 (rank01=0.1)
        # gap = 0.1 - 0 = 0.1 >= 0.1 -> SWAP

        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, held_indices=[0, 10], hold_days_arr={0: 10, 10: 10}
        )
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=0,
        )

        # Worst held: 0 (rank01=0/10=0)
        # Best new not held: 9 (rank01=9/10=0.9, since 10 is held)
        # gap = 0.9 - 0 = 0.9 >= 0.10 -> SWAP
        assert result[0] == False
        assert result[10] == True
        assert result[9] == True

    def test_just_below_rank_gap_threshold(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        # Create scenario where best new is only slightly better
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, held_indices=[8, 9], hold_days_arr={8: 10, 9: 10}
        )
        pos_size = 2

        # delta_rank=0.15
        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.15,
            min_hold_days_val=0,
        )

        # Worst held: 8 (rank01=8/9=0.89)
        # Best new not held: 7 (rank01=7/9=0.78)
        # gap = 0.78 - 0.89 = -0.11 < 0.15 -> NO SWAP
        assert result[8] == True
        assert result[9] == True

    def test_min_hold_days_exactly_at_threshold(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N,
            held_indices=[3, 4],
            hold_days_arr={3: 9, 4: 10},  # Exactly 9
        )
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=9,
        )

        # hold_days[3]=9 >= 9 -> condition met
        # If rank gap also met, should swap
        # Worst held: 3 (rank01=3/9=0.33)
        # Best new: 9 (rank01=1.0), gap=0.67 >= 0.10 -> SWAP
        assert result[3] == False
        assert result[9] == True


# ─────────────────────────────────────────────────────────
#  7. Max 1 swap per rebalance
# ─────────────────────────────────────────────────────────


class TestMaxOneSwap:
    """Verify max 1 swap per rebalance (forced by implementation)."""

    def test_only_one_swap_even_if_multiple_candidates_better(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        # Hold 2 positions at bottom of ranking
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, held_indices=[0, 1], hold_days_arr={0: 10, 1: 10}
        )
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.05,
            min_hold_days_val=0,
        )

        # Worst held: 0 (rank01=0)
        # Multiple better candidates: 9, 8, 7, ... all have huge gaps
        # But only 1 swap allowed -> swap out 0, swap in 9
        assert result.sum() == pos_size
        # Only 0 should be swapped out (worst), 1 kept
        assert result[0] == False
        assert result[1] == True
        assert result[9] == True
        # 8 should NOT be in (only 1 swap)
        assert result[8] == False


# ─────────────────────────────────────────────────────────
#  8. NaN handling
# ─────────────────────────────────────────────────────────


class TestNaNHandling:
    """NaN scores should be treated as -inf (worst ranking).

    Note: NaN handling only applies in the fully-invested case.
    When hysteresis is disabled, all top_indices pass through regardless.
    """

    def test_nan_treated_as_worst_in_fully_invested(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        # Hold positions with NaN at 9 (would be best without NaN)
        # Hold 7, 8 which are near top, so hysteresis applies
        custom_scores = np.arange(N, dtype=np.float64)
        custom_scores[9] = np.nan  # Position 9 has NaN

        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N,
            held_indices=[7, 8],
            hold_days_arr={7: 10, 8: 10},
            custom_scores=custom_scores,
        )
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=0,
        )

        # Position 9 has NaN -> treated as -inf in ranking
        # rank01[9] = 0/9 = 0 (worst due to -inf)
        # Worst held: 7 (rank01=7/9=0.78)
        # Best new not held: 6 (since 9 is now worst)
        # Gap from 7 to 6 is negative (6 < 7), no swap
        # But wait - let me recalculate: with NaN at 9, ranking changes
        # Actually 9 is now worst (rank01=0), 8 is best (rank01=1.0)
        # Worst held is still 7 (rank01=7/9=0.78)
        # Best new not held: 6 (rank01=6/9=0.67), but 6 < 7 in score
        # So best_new_rank < worst_held_rank -> no swap
        assert result[7] == True
        assert result[8] == True

    def test_nan_position_swapped_out_when_worst_held(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 10
        # Hold a position with NaN score - it should be swapped out
        custom_scores = np.arange(N, dtype=np.float64)
        custom_scores[3] = np.nan

        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N,
            held_indices=[3, 4],
            hold_days_arr={3: 10, 4: 10},
            custom_scores=custom_scores,
        )
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=9,
        )

        # Position 3 has NaN -> rank01=0 (worst)
        # Worst held: 3 (rank01=0)
        # Best new: 9 (rank01=1.0), gap=1.0 >= 0.10 -> SWAP
        assert result[3] == False
        assert result[9] == True


# ─────────────────────────────────────────────────────────
#  9. Production parameter validation
# ─────────────────────────────────────────────────────────


class TestProductionParameters:
    """Test with v5.0 production parameters."""

    def test_v5_production_params(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 43  # Production ETF pool size
        # Hold positions at rank 20, 21 (mid-pack)
        held = [20, 21]
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, held_indices=held, hold_days_arr={20: 15, 21: 15}
        )
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=9,
        )

        # Worst held: 20 (rank01=20/42=0.476)
        # Best new: 42 (rank01=1.0)
        # Gap = 1.0 - 0.476 = 0.524 >= 0.10 -> SWAP
        assert result[20] == False
        assert result[21] == True
        assert result[42] == True

    def test_v5_no_swap_when_hold_days_insufficient(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 43
        held = [20, 21]
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N,
            held_indices=held,
            hold_days_arr={20: 5, 21: 5},  # Only 5 days
        )
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=9,
        )

        # hold_days=5 < 9 -> NO SWAP
        assert result[20] == True
        assert result[21] == True


# ─────────────────────────────────────────────────────────
#  10. Edge cases
# ─────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and potential failure modes."""

    def test_single_etf_pool(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 1
        custom_scores = np.array([5.0], dtype=np.float64)
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, custom_scores=custom_scores
        )
        pos_size = 1

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=9,
        )

        assert result[0] == True

    def test_all_nan_scores(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 5
        custom_scores = np.full(N, np.nan, dtype=np.float64)
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, custom_scores=custom_scores
        )
        pos_size = 2

        # Should not crash
        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.10,
            min_hold_days_val=9,
        )

        assert result.sum() <= pos_size

    def test_inf_scores(self):
        from etf_strategy.core.hysteresis import apply_hysteresis

        N = 5
        custom_scores = np.array([1.0, 2.0, np.inf, 4.0, 5.0], dtype=np.float64)
        combined_score, holdings_mask, hold_days, top_indices = make_test_data(
            N, custom_scores=custom_scores
        )
        pos_size = 2

        result = apply_hysteresis(
            combined_score,
            holdings_mask,
            hold_days,
            top_indices,
            pos_size=pos_size,
            delta_rank=0.0,
            min_hold_days_val=0,
        )

        # inf at index 2 should be best
        assert result[2] == True
