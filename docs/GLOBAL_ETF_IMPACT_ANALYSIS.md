# Global ETF Impact Analysis

## 1. Executive Summary
Adding Global ETFs (HengSheng Tech, Dow Jones, Nikkei 225) to the A-share ETF rotation strategy **significantly reduces performance** (from 237% to 188%).

- **Baseline (43 A-share ETFs)**: **237.45%** Return, 14.28% MaxDD, Sharpe 1.38
- **+ Nikkei 225**: **210.65%** Return (-27pp)
- **+ Nikkei + Dow**: **198.84%** Return (-12pp)
- **+ Nikkei + Dow + HS Tech**: **188.42%** Return (-10pp)

## 2. Root Cause Analysis

### A. Opportunity Cost (Displacement)
The strategy (`POS=2`) selects only the top 2 assets.
- **A-share Bull Runs**: Sectors like Semiconductor (`512480`) or PV (`515790`) can rally 50-100% in a short period.
- **Global ETFs**: Indices like Dow or Nikkei offer steady, lower-volatility growth (e.g., 10-20% per year).
- **The Problem**: When the strategy picks a "steady" Global ETF (due to high Sharpe/low volatility), it **displaces** an explosive A-share ETF. You trade "explosive alpha" for "steady beta", resulting in lower total returns.

### B. Timing Mismatch
The `LightTimingModule` calculates market sentiment based on the **average of the entire universe**.
- The universe is 93% A-shares (43/46).
- The timing signal is effectively an "A-share Market Timing" signal.
- **Mismatch**: Global markets (US/JP) do not follow A-share timing.
    - If A-shares crash (triggering Bear signal), the strategy reduces positions, potentially selling Global ETFs that are actually doing well.
    - If A-shares rally (triggering Bull signal), the strategy buys, but might pick a Global ETF that is currently flat or correcting.

### C. "Toxic" Assets (HengSheng Tech)
- **HengSheng Tech (`513180`)** has been in a structural bear market with high volatility.
- It frequently triggers "false positive" buy signals during short-term rebounds, only to crash again.
- It contributed a net loss to the portfolio.

## 3. Recommendation
For the current **High-Frequency Rotation Strategy (v3.0)**, which relies on explosive momentum:
1.  **Exclude Global ETFs**: Stick to the 43 A-share sector ETFs to maximize the "Rotation Alpha".
2.  **Separate Strategy**: Run a separate, lower-frequency strategy for Global ETFs (e.g., "Global Asset Allocation") rather than mixing them into the high-volatility A-share rotation pool.

## 4. Verification Script
A reproduction script has been created at `scripts/verify_baseline_237.py`.
Run it to verify the 237% baseline on the original 43 ETFs.
