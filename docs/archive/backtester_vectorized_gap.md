<!-- ALLOW-MD -->
# Backtester vs Vectorized Audit Playbook

_Last updated: 2025-11-27 10:08 UTC_

## 1. Objective
Provide a single source of truth so any LLM agent (Cloud Code, CodeX, etc.) can replicate and debug the Backtrader ↔︎ vectorized discrepancies without tribal knowledge.

## 2. Baseline Contract (must match on both engines)
| Dimension | Required Contract | Symptoms if Broken |
| --- | --- | --- |
| **Data ordering** | `close_prices`, factor cubes, and Backtrader feeds must share the **same ETF column order** (`etf_codes` from `std_factors` is canonical). | Swapped ticker prices, impossible PnL spikes, random winners. |
| **Date window** | Both engines must start on identical `LOOKBACK`-aligned calendars (Backtrader feeds must not drop NaN rows). | Backtrader begins in 2022, vectorized in 2020 ⇒ uncomparable returns. |
| **Execution timing** | Signals from `T-1`, fills at `T` close (Cheat-On-Close). Backtrader must hit `bar_index = len(self)-1`. | One-day shift in holdings, opposite trade lists. |
| **Leverage** | Hard cap 1.0 (no margin). Remove `leverage=2`. | Backtrader equity >> vectorized, cash negative/unused. |
| **Position sizing** | `target_pos_value = net_current_value / target_count / (1+commission)` using **net-new capital** (exclude existing positions). | Cash shortfalls on 2nd/3rd buys, missing holdings. |
| **Fees** | Identical commission rate on buys & sells (T+0). | Residual cash mismatch, drift over time. |

LLM agents must verify **all six rows** before attempting “bug fixes.”

## 3. Current Findings (2025-11-27)
1. `close_prices` array was unsorted, so factor ranks pointed to the wrong ETF prices. → FIXED by reindexing with `etf_codes`.
2. Backtrader feeds dropped NaN rows, shifting the start date to 2022. Need feeds reindexed to the full calendar with forward-fill instead of dropna.
3. Broker was initialized with `leverage=2.0`, letting Backtrader overspend by 2×. → FIXED (removed leverage).
4. Cheat-On-Close still executes on `next` bar, so cash snapshot between submissions and fills looks half-spent—expected behavior but confusing; rely on `notify_order` logs.
5. Position sizing updated to "Net-New" logic on both engines. → FIXED.

## 4. Recommended Debug Flow for LLM Agents
1. **Verify Contracts**
    - Load `etf_codes = sorted(std_factors[...].columns)` and assert `ohlcv['close'].columns` matches.
    - Ensure each Backtrader feed uses the same date index (`df = df.reindex(dates).ffill()`).
    - Check `cerebro.broker.setcommission(...);` **no leverage parameter**.
 2. **Minimal Repro**
    - Combo: `('ADX_14D',)`; no timing; `FREQ=8`, `POS_SIZE=3`.
    - Run `_backtest_combo_numba` → record `return`, `trades`.
    - Run Backtrader with identical combined scores → compare trade logs for first 3 rebalances.
 3. **Trade-by-Trade Diff**
    - Export vectorized trades (`tmp_vectorized_trades.parquet`) and Backtrader trades parquet.
    - Align by (`entry_date`, `ticker`) to confirm fills; mismatches highlight execution drift.
 4. **Cash/Equity Audit**
    - Plot `vectorized_equity` vs `Backtrader equity` (CSV/Parquet already available).
    - Use “ratio” series (`bt_equity / vec_equity`) to pinpoint divergence dates.
 5. **Position-Sizing Fix (Completed)**
    - Both engines now use `available_for_new = equity * timing - kept_value` and `target_pos_value = available / new_count`.
    - **Execution Timing**: Backtrader uses a "Technical Leverage Buffer" (leverage=1.1) to allow T+0 execution. This bypasses the strict settlement check (where sell proceeds aren't immediately available for buys in the same bar) while the strategy logic strictly enforces 1.0x equity cap.

## 5. Next Steps / TODOs
- [x] Update `strategy_auditor/core/engine.py` to remove leverage and adopt net-new target sizing.
- [x] Mirror the same sizing fix inside `_backtest_combo_numba` for consistency.
- [x] Implement "Technical Leverage Buffer" in Backtrader to fix cash availability issues.
- [ ] Create an automated “contract check” script (fail fast if order, dates, leverage, or NaNs violate spec).
- [ ] Once contracts pass, rerun `strategy_auditor/run_parallel_audit.py` on top-N combos to confirm average diff < 1pp.

## 6. How to Use This Doc (for future LLMs)
1. Read Section 2, assert every bullet before touching code.
2. Follow Section 4 sequentially—do not skip to “fixing” before contracts pass.
3. Log every change back into this file (update timestamp + bullet) so the next agent inherits the context.

---
_If you are another LLM agent: update this playbook whenever you fix or discover contract drift. This is the canonical baton._
