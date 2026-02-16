"""Exp4: Position change hysteresis + minimum holding period.

Shared @njit logic for VEC kernel and WFO optimizer.
Ensures VEC/WFO alignment by using the same decision function.

Rules:
  1. Max 1 swap per rebalance (forced)
  2. Swap only if rank gap >= delta_rank (hysteresis)
  3. Swap only if held days >= min_hold_days (minimum hold)

rank01 definition:
  - rank_pos = argsort(ascending) index → 0=worst, N-1=best
  - rank01 = rank_pos / (N-1), so best=1.0, worst=0.0
  - delta_rank=0.10 means new candidate must lead worst held by ≥4.2 ranks (N=43)
"""

import numpy as np
from numba import njit


@njit(cache=True)
def apply_hysteresis(
    combined_score,     # (N,) float64 — factor scores (or signal); -inf/NaN = invalid
    holdings_mask,      # (N,) bool — True if position currently held
    hold_days,          # (N,) int64 — trading days held per position
    top_indices,        # (k,) int64 — from stable_topk_indices, best-first
    pos_size,           # int — target number of positions (e.g. 2)
    delta_rank,         # float — minimum rank01 gap for swap (0 = disabled)
    min_hold_days_val,  # int — minimum hold days before allowing sell (0 = disabled)
):
    """Apply hysteresis filter to topk selection.

    Returns:
        target_mask: (N,) bool — which positions to target after filtering
    """
    N = len(combined_score)
    top_count = len(top_indices)
    target_mask = np.zeros(N, dtype=np.bool_)

    # --- Disabled: pass through topk as-is ---
    if delta_rank <= 0.0 and min_hold_days_val <= 0:
        for i in range(top_count):
            target_mask[top_indices[i]] = True
        return target_mask

    # --- Count current holdings ---
    held_count = 0
    for n in range(N):
        if holdings_mask[n]:
            held_count += 1

    # --- Not fully invested: keep current + fill from topk (no hysteresis) ---
    if held_count < pos_size:
        for n in range(N):
            if holdings_mask[n]:
                target_mask[n] = True
        remaining = pos_size - held_count
        for i in range(top_count):
            if remaining <= 0:
                break
            idx = top_indices[i]
            if not target_mask[idx]:
                target_mask[idx] = True
                remaining -= 1
        return target_mask

    # --- Fully invested: apply hysteresis (max 1 swap) ---

    # Compute rank01 from combined_score
    # Treat NaN as -inf for stable ranking
    safe_score = np.empty(N, dtype=np.float64)
    for n in range(N):
        s = combined_score[n]
        if np.isnan(s):
            safe_score[n] = -np.inf
        else:
            safe_score[n] = s

    order = np.argsort(safe_score)  # ascending: worst first
    rank01 = np.zeros(N, dtype=np.float64)
    denom = float(N - 1) if N > 1 else 1.0
    for j in range(N):
        rank01[order[j]] = float(j) / denom

    # Start with keeping all current holdings
    for n in range(N):
        if holdings_mask[n]:
            target_mask[n] = True

    # Find worst held position (lowest rank01)
    worst_idx = -1
    worst_rank = 2.0
    for n in range(N):
        if holdings_mask[n] and rank01[n] < worst_rank:
            worst_rank = rank01[n]
            worst_idx = n

    # Find best new candidate (in topk, not currently held)
    best_new_idx = -1
    for i in range(top_count):
        idx = top_indices[i]
        if not holdings_mask[idx]:
            best_new_idx = idx
            break  # top_indices is sorted best-first

    # Check swap conditions
    if worst_idx >= 0 and best_new_idx >= 0:
        rank_gap = rank01[best_new_idx] - rank01[worst_idx]
        rank_ok = rank_gap >= delta_rank
        days_ok = hold_days[worst_idx] >= min_hold_days_val

        if rank_ok and days_ok:
            # Execute swap: remove worst, add best new
            target_mask[worst_idx] = False
            target_mask[best_new_idx] = True

    return target_mask
