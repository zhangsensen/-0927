#!/usr/bin/env python3
"""
Statistical Significance Analysis & Stress Test for C2 Shadow Deployment
========================================================================

Computes:
1. Trade-level statistics from VEC/BT equity curves
2. Sample size requirements for various confidence levels
3. Bootstrap confidence intervals (block bootstrap)
4. Live performance consistency check
5. Stress test: drawdown duration, consecutive losses, tail risk
6. Stopping rules for sequential monitoring

Uses VEC equity curve (daily resolution, 1482 days) for bootstrap/stress
and BT summary stats for ground truth parameters.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# VEC equity curves with hysteresis
VEC_DIR = RESULTS_DIR / "vec_full_backtest_20260212_113426"
# BT 6-candidate results
BT_FILE = RESULTS_DIR / "bt_6candidates_20260212_181625" / "bt_results.parquet"

HOLDOUT_START = "2025-05-01"
FREQ = 5  # rebalance every 5 trading days
ANNUAL_TRADING_DAYS = 252

# Live performance
LIVE_RETURN = 0.0637  # +6.37%
LIVE_TRADING_DAYS = 30  # ~6 weeks
LIVE_TRADES = 22

# Bootstrap config
N_BOOTSTRAP = 10000
BLOCK_SIZE = FREQ  # block bootstrap with block = rebalance frequency
RNG_SEED = 42


def load_equity_curves():
    """Load VEC equity curves and extract C2 + S1."""
    npz = np.load(VEC_DIR / "equity_curves.npz", allow_pickle=True)
    curves = npz["curves"]  # (1482, 27)
    dates = pd.to_datetime(npz["dates"])
    combos = list(npz["combos"])

    # Find C2 and S1
    c2_idx = None
    s1_idx = None
    for i, c in enumerate(combos):
        if (
            "AMIHUD" in c
            and "CALMAR" in c
            and "CORRELATION_TO_MARKET" in c
            and c.count("+") == 2
        ):
            c2_idx = i
        if "ADX_14D" in c and "OBV_SLOPE" in c and "SHARPE" in c and "SLOPE_20D" in c:
            s1_idx = i

    if c2_idx is None:
        print("ERROR: C2 not found in equity curves")
        sys.exit(1)
    if s1_idx is None:
        print("WARNING: S1 not found in equity curves")

    return dates, curves[:, c2_idx], curves[:, s1_idx] if s1_idx is not None else None


def load_bt_stats():
    """Load BT summary stats for C2."""
    df = pd.read_parquet(BT_FILE)
    c2 = df[
        df["combo"].str.contains("AMIHUD")
        & df["combo"].str.contains("CALMAR")
        & df["combo"].str.contains("CORR")
    ]
    if len(c2) == 0:
        print("ERROR: C2 not found in BT results")
        sys.exit(1)
    return c2.iloc[0].to_dict()


def extract_trade_returns(equity, dates, holdout_start, freq):
    """Extract rebalance-period returns from equity curve."""
    ho_mask = dates >= pd.Timestamp(holdout_start)
    ho_equity = equity[ho_mask]
    ho_dates = dates[ho_mask]

    # Rebalance-period returns (every FREQ days)
    rebal_indices = list(range(0, len(ho_equity), freq))
    if rebal_indices[-1] != len(ho_equity) - 1:
        rebal_indices.append(len(ho_equity) - 1)

    rebal_equity = ho_equity[rebal_indices]
    rebal_dates = ho_dates[rebal_indices]
    period_returns = np.diff(rebal_equity) / rebal_equity[:-1]

    return period_returns, rebal_dates[1:], ho_equity, ho_dates


def compute_trade_statistics(returns):
    """Compute comprehensive trade-level statistics."""
    n = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)

    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = len(wins) / n if n > 0 else 0
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    profit_factor = (
        abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float("inf")
    )

    # Annualized metrics
    periods_per_year = ANNUAL_TRADING_DAYS / FREQ
    ann_return = (1 + mean_ret) ** periods_per_year - 1
    ann_vol = std_ret * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Max consecutive wins/losses
    signs = np.sign(returns)
    max_consec_loss = 0
    max_consec_win = 0
    curr_loss = 0
    curr_win = 0
    for s in signs:
        if s <= 0:
            curr_loss += 1
            curr_win = 0
            max_consec_loss = max(max_consec_loss, curr_loss)
        else:
            curr_win += 1
            curr_loss = 0
            max_consec_win = max(max_consec_win, curr_win)

    return {
        "n_trades": n,
        "mean_return": mean_ret,
        "std_return": std_ret,
        "skewness": skew,
        "kurtosis": kurt,
        "max_gain": np.max(returns),
        "max_loss": np.min(returns),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_consecutive_losses": max_consec_loss,
        "max_consecutive_wins": max_consec_win,
    }


def compute_sample_sizes(std_ret):
    """
    Compute required sample sizes for different hypotheses and confidence levels.

    For testing H0: mu=0 vs H1: mu=delta, using t-test:
        n = ((z_alpha + z_beta) / (delta / sigma))^2

    We express delta in terms of Sharpe: delta = Sharpe * sigma / sqrt(periods_per_year)
    So n_periods = ((z_alpha + z_beta) * sqrt(periods_per_year) / Sharpe)^2
    """
    periods_per_year = ANNUAL_TRADING_DAYS / FREQ

    results = []
    for alpha, z_alpha_name in [(0.05, "p<0.05"), (0.01, "p<0.01")]:
        z_alpha = stats.norm.ppf(1 - alpha / 2)  # two-sided
        z_beta = stats.norm.ppf(0.80)  # power = 0.80

        for sharpe_target, sharpe_name in [
            (0.0, "Sharpe > 0 (any positive)"),
            (0.5, "Sharpe > 0.5"),
            (1.0, "Sharpe > 1.0"),
            (1.5, "Sharpe > 1.5"),
        ]:
            if sharpe_target == 0:
                # For Sharpe > 0, use observed mean as the effect size
                # This tells us how many trades to confirm the observed Sharpe is real
                sharpe_obs = std_ret  # placeholder - will be replaced
                # Use a minimal positive Sharpe (0.1) as detectable effect
                sharpe_eff = 0.1
            else:
                sharpe_eff = sharpe_target

            # n = ((z_alpha + z_beta)^2 * periods_per_year) / sharpe_eff^2
            n_periods = ((z_alpha + z_beta) ** 2 * periods_per_year) / sharpe_eff**2
            n_trades = int(np.ceil(n_periods))
            # Convert to calendar time
            n_days = n_periods * FREQ
            n_months = n_days / 21  # ~21 trading days/month

            results.append(
                {
                    "confidence": z_alpha_name,
                    "target": sharpe_name if sharpe_target > 0 else "Sharpe > 0 (effect=0.1)",
                    "trades_needed": n_trades,
                    "trading_days": int(np.ceil(n_days)),
                    "approx_months": round(n_months, 1),
                }
            )

    return results


def compute_drawdown_stats(equity, dates):
    """Compute drawdown statistics from equity curve."""
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    # Max drawdown
    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    max_dd_date = dates[max_dd_idx]

    # Find peak before max drawdown
    peak_idx = np.argmax(equity[: max_dd_idx + 1])
    peak_date = dates[peak_idx]

    # Find recovery (first time equity exceeds peak after drawdown)
    recovery_idx = None
    peak_val = equity[peak_idx]
    for i in range(max_dd_idx, len(equity)):
        if equity[i] >= peak_val:
            recovery_idx = i
            break

    recovery_date = dates[recovery_idx] if recovery_idx is not None else None
    dd_duration = max_dd_idx - peak_idx  # days from peak to trough
    recovery_duration = (
        (recovery_idx - max_dd_idx) if recovery_idx is not None else None
    )

    # All drawdown episodes > 5%
    in_drawdown = False
    dd_episodes = []
    dd_start = 0
    for i in range(len(drawdown)):
        if not in_drawdown and drawdown[i] < -0.05:
            in_drawdown = True
            dd_start = i
        elif in_drawdown and drawdown[i] >= 0:
            dd_episodes.append(
                {
                    "start_date": str(dates[dd_start].date()),
                    "trough_date": str(
                        dates[dd_start + np.argmin(drawdown[dd_start:i])].date()
                    ),
                    "end_date": str(dates[i].date()),
                    "depth": float(np.min(drawdown[dd_start:i])),
                    "duration_days": i - dd_start,
                }
            )
            in_drawdown = False

    # If still in drawdown at end
    if in_drawdown:
        dd_episodes.append(
            {
                "start_date": str(dates[dd_start].date()),
                "trough_date": str(
                    dates[dd_start + np.argmin(drawdown[dd_start:])].date()
                ),
                "end_date": "ongoing",
                "depth": float(np.min(drawdown[dd_start:])),
                "duration_days": len(drawdown) - dd_start,
            }
        )

    return {
        "max_drawdown": float(max_dd),
        "max_dd_date": str(max_dd_date.date()),
        "peak_date": str(peak_date.date()),
        "recovery_date": str(recovery_date.date()) if recovery_date is not None else "not recovered",
        "drawdown_duration_days": int(dd_duration),
        "recovery_duration_days": int(recovery_duration) if recovery_duration is not None else None,
        "total_dd_to_recovery_days": (
            int(dd_duration + recovery_duration)
            if recovery_duration is not None
            else None
        ),
        "drawdown_episodes_gt5pct": dd_episodes,
    }


def block_bootstrap(returns, n_boot, block_size, rng):
    """
    Stationary block bootstrap for time series.
    Resamples blocks of returns to preserve serial correlation.
    """
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))
    boot_stats = []

    for _ in range(n_boot):
        # Sample random starting points
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        boot_returns = np.concatenate(
            [returns[s : s + block_size] for s in starts]
        )[:n]

        # Compute equity from returns
        boot_equity = np.cumprod(1 + boot_returns)
        total_ret = boot_equity[-1] - 1

        # Annualized metrics
        periods_per_year = ANNUAL_TRADING_DAYS / FREQ
        n_periods = len(boot_returns)
        ann_factor = periods_per_year / n_periods
        ann_ret = (1 + total_ret) ** ann_factor - 1
        ann_vol = np.std(boot_returns, ddof=1) * np.sqrt(periods_per_year)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        # Max drawdown
        cum_max = np.maximum.accumulate(boot_equity)
        dd = (boot_equity - cum_max) / cum_max
        max_dd = np.min(dd)

        boot_stats.append(
            {
                "total_return": total_ret,
                "annualized_return": ann_ret,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
            }
        )

    return pd.DataFrame(boot_stats)


def daily_bootstrap(daily_returns, n_boot, block_size_days, rng):
    """
    Block bootstrap on daily returns for finer-grained stress testing.
    """
    n = len(daily_returns)
    n_blocks = int(np.ceil(n / block_size_days))
    boot_stats = []

    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block_size_days + 1), size=n_blocks)
        boot_returns = np.concatenate(
            [daily_returns[s : s + block_size_days] for s in starts]
        )[:n]

        boot_equity = np.cumprod(1 + boot_returns)

        # 6-month subset (126 trading days)
        if len(boot_equity) >= 126:
            six_mo = boot_equity[:126]
            six_mo_ret = six_mo[-1] - 1
            cum_max_6 = np.maximum.accumulate(six_mo)
            dd_6 = (six_mo - cum_max_6) / cum_max_6
            six_mo_mdd = np.min(dd_6)
        else:
            six_mo_ret = boot_equity[-1] - 1
            cum_max_6 = np.maximum.accumulate(boot_equity)
            dd_6 = (boot_equity - cum_max_6) / cum_max_6
            six_mo_mdd = np.min(dd_6)

        boot_stats.append(
            {
                "six_month_return": six_mo_ret,
                "six_month_mdd": six_mo_mdd,
            }
        )

    return pd.DataFrame(boot_stats)


def compute_live_consistency(trade_stats, bt_stats):
    """Check if live performance is within expected range."""
    # Annualized live return
    live_ann = (1 + LIVE_RETURN) ** (ANNUAL_TRADING_DAYS / LIVE_TRADING_DAYS) - 1

    # Expected 30-day return from HO Sharpe
    ho_sharpe = bt_stats["ho_sharpe"]
    ho_mdd = bt_stats["ho_mdd"]
    ho_ret = bt_stats["ho_return"]

    # From BT HO: annualized return implied by Sharpe
    # Sharpe = ann_ret / ann_vol  ->  ann_vol = ann_ret / Sharpe
    # For 30 days: expected_ret = ann_ret * (30/252)
    # But more directly: use per-period stats
    periods_per_year = ANNUAL_TRADING_DAYS / FREQ

    # Per-period expected return from trade stats
    mu = trade_stats["mean_return"]
    sigma = trade_stats["std_return"]

    # Expected 30-day return = 6 periods * mu (approx, ignoring compounding)
    n_periods_30d = LIVE_TRADING_DAYS / FREQ
    expected_30d = (1 + mu) ** n_periods_30d - 1
    std_30d = sigma * np.sqrt(n_periods_30d)

    # 95% CI
    ci_low = expected_30d - 1.96 * std_30d
    ci_high = expected_30d + 1.96 * std_30d

    # z-score of live
    z_live = (LIVE_RETURN - expected_30d) / std_30d if std_30d > 0 else 0

    return {
        "live_return": LIVE_RETURN,
        "live_annualized": live_ann,
        "live_trades": LIVE_TRADES,
        "live_trading_days": LIVE_TRADING_DAYS,
        "expected_30d_return": expected_30d,
        "expected_30d_std": std_30d,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "z_score": z_live,
        "within_95_ci": ci_low <= LIVE_RETURN <= ci_high,
        "assessment": (
            "CONSISTENT"
            if ci_low <= LIVE_RETURN <= ci_high
            else ("OUTPERFORMING" if LIVE_RETURN > ci_high else "UNDERPERFORMING")
        ),
    }


def design_stopping_rules(trade_stats, boot_df):
    """Design sequential stopping rules."""
    sigma = trade_stats["std_return"]
    periods_per_year = ANNUAL_TRADING_DAYS / FREQ

    # 1. O'Brien-Fleming spending function boundaries
    # For K=4 interim analyses (at 25%, 50%, 75%, 100% of planned trades)
    # O'Brien-Fleming boundaries are wider early, narrower late
    # z_k = z_final / sqrt(k/K) where z_final = z_alpha/2
    z_final = stats.norm.ppf(1 - 0.05 / 2)  # 1.96
    K = 4
    obf_boundaries = []
    for k in range(1, K + 1):
        info_frac = k / K
        z_k = z_final / np.sqrt(info_frac)
        n_at_look = int(np.ceil(50 * info_frac))  # planned 50 trades
        obf_boundaries.append(
            {
                "look": k,
                "info_fraction": info_frac,
                "trades_at_look": n_at_look,
                "z_boundary": round(z_k, 3),
                "action": f"Stop for futility if z < -{z_k:.2f} or stop for efficacy if z > {z_k:.2f}",
            }
        )

    # 2. MDD threshold from bootstrap
    # Use 99th percentile of 6-month MDD as emergency stop
    # (We'll set this from the daily bootstrap results)

    # 3. Minimum trades before evaluation
    # Practical minimum: first OBF look at 25% of planned (13 trades)
    # Statistical minimum: enough for CLT to hold (~20-30 trades)
    # We use 20 trades as the floor — below this, no statistical test is meaningful
    min_trades = 20

    return {
        "obrien_fleming_boundaries": obf_boundaries,
        "planned_total_trades": 50,
        "min_trades_before_eval": min_trades,
        "mdd_emergency_stop": None,  # filled after bootstrap
        "sharpe_futility_stop": "Sharpe < 0 after 50 trades -> consider stopping",
        "max_consecutive_loss_stop": max(
            5, trade_stats["max_consecutive_losses"] + 2
        ),
    }


def main():
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS & STRESS TEST")
    print("C2: AMIHUD + CALMAR + CORRELATION_TO_MARKET")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("\n[1] Loading data...")
    dates, c2_equity, s1_equity = load_equity_curves()
    bt_stats = load_bt_stats()

    print(f"  VEC equity curve: {len(dates)} days ({dates[0].date()} to {dates[-1].date()})")
    print(f"  BT C2 HO return: {bt_stats['ho_return']:.1%}")
    print(f"  BT C2 HO Sharpe: {bt_stats['ho_sharpe']:.3f}")
    print(f"  BT C2 HO MDD:    {bt_stats['ho_mdd']:.1%}")

    # -----------------------------------------------------------------------
    # 2. Trade-level statistics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[2] TRADE-LEVEL STATISTICS (VEC Holdout)")
    print("=" * 80)

    trade_returns, rebal_dates, ho_equity, ho_dates = extract_trade_returns(
        c2_equity, dates, HOLDOUT_START, FREQ
    )
    trade_stats = compute_trade_statistics(trade_returns)

    print(f"  Number of rebalance periods: {trade_stats['n_trades']}")
    print(f"  Mean period return:    {trade_stats['mean_return']:+.4f} ({trade_stats['mean_return']:+.2%})")
    print(f"  Std period return:     {trade_stats['std_return']:.4f}")
    print(f"  Skewness:              {trade_stats['skewness']:+.3f}")
    print(f"  Excess kurtosis:       {trade_stats['kurtosis']:+.3f}")
    print(f"  Max gain:              {trade_stats['max_gain']:+.2%}")
    print(f"  Max loss:              {trade_stats['max_loss']:+.2%}")
    print(f"  Win rate:              {trade_stats['win_rate']:.1%}")
    print(f"  Avg win:               {trade_stats['avg_win']:+.2%}")
    print(f"  Avg loss:              {trade_stats['avg_loss']:+.2%}")
    print(f"  Profit factor:         {trade_stats['profit_factor']:.3f}")
    print(f"  Annualized return:     {trade_stats['annualized_return']:+.1%}")
    print(f"  Annualized volatility: {trade_stats['annualized_volatility']:.1%}")
    print(f"  Sharpe ratio:          {trade_stats['sharpe_ratio']:.3f}")
    print(f"  Max consecutive losses:{trade_stats['max_consecutive_losses']}")
    print(f"  Max consecutive wins:  {trade_stats['max_consecutive_wins']}")

    # Also compute S1 for comparison
    if s1_equity is not None:
        s1_returns, _, _, _ = extract_trade_returns(s1_equity, dates, HOLDOUT_START, FREQ)
        s1_stats = compute_trade_statistics(s1_returns)
        print(f"\n  [S1 comparison] Mean: {s1_stats['mean_return']:+.4f}, "
              f"Sharpe: {s1_stats['sharpe_ratio']:.3f}, WR: {s1_stats['win_rate']:.1%}")

    # -----------------------------------------------------------------------
    # 3. Sample size requirements
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[3] SAMPLE SIZE REQUIREMENTS")
    print("=" * 80)

    sample_sizes = compute_sample_sizes(trade_stats["std_return"])

    print(f"\n  {'Confidence':<12} {'Target':<32} {'Trades':<10} {'Days':<10} {'Months':<8}")
    print(f"  {'-'*12} {'-'*32} {'-'*10} {'-'*10} {'-'*8}")
    for row in sample_sizes:
        print(
            f"  {row['confidence']:<12} {row['target']:<32} "
            f"{row['trades_needed']:<10} {row['trading_days']:<10} {row['approx_months']:<8}"
        )

    print(f"\n  Note: periods_per_year = {ANNUAL_TRADING_DAYS}/{FREQ} = {ANNUAL_TRADING_DAYS/FREQ:.1f}")
    print(f"  Observed per-period std = {trade_stats['std_return']:.4f}")
    print(f"  Formula: n = ((z_alpha + z_beta)^2 * periods_per_year) / Sharpe_target^2")

    # -----------------------------------------------------------------------
    # 4. Bootstrap confidence intervals
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[4] BOOTSTRAP CONFIDENCE INTERVALS (Block Bootstrap, B=10000)")
    print("=" * 80)

    rng = np.random.default_rng(RNG_SEED)
    boot_df = block_bootstrap(trade_returns, N_BOOTSTRAP, BLOCK_SIZE, rng)

    for metric in ["total_return", "annualized_return", "sharpe", "max_drawdown"]:
        vals = boot_df[metric].values
        ci_5 = np.percentile(vals, 5)
        ci_25 = np.percentile(vals, 25)
        ci_50 = np.percentile(vals, 50)
        ci_75 = np.percentile(vals, 75)
        ci_95 = np.percentile(vals, 95)
        mean = np.mean(vals)

        label = {
            "total_return": "HO Total Return",
            "annualized_return": "Annualized Return",
            "sharpe": "Sharpe Ratio",
            "max_drawdown": "Max Drawdown",
        }[metric]

        print(f"\n  {label}:")
        print(f"    Mean:   {mean:+.4f}")
        print(f"    5th:    {ci_5:+.4f}  (worst case)")
        print(f"    25th:   {ci_25:+.4f}")
        print(f"    Median: {ci_50:+.4f}")
        print(f"    75th:   {ci_75:+.4f}")
        print(f"    95th:   {ci_95:+.4f}")

    # Probability of negative return
    prob_neg = (boot_df["total_return"] < 0).mean()
    print(f"\n  P(HO return < 0):  {prob_neg:.1%}")
    prob_sharpe_lt1 = (boot_df["sharpe"] < 1.0).mean()
    print(f"  P(Sharpe < 1.0):   {prob_sharpe_lt1:.1%}")

    # -----------------------------------------------------------------------
    # 5. Live performance consistency
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[5] LIVE PERFORMANCE CONSISTENCY CHECK")
    print("=" * 80)

    live_check = compute_live_consistency(trade_stats, bt_stats)

    print(f"  Live return:           {live_check['live_return']:+.2%} over {live_check['live_trading_days']} trading days")
    print(f"  Live annualized:       {live_check['live_annualized']:+.1%}")
    print(f"  Live trades:           {live_check['live_trades']}")
    print(f"  Expected 30-day ret:   {live_check['expected_30d_return']:+.2%}")
    print(f"  Expected 30-day std:   {live_check['expected_30d_std']:.2%}")
    print(f"  95% CI:                [{live_check['ci_95_low']:+.2%}, {live_check['ci_95_high']:+.2%}]")
    print(f"  Z-score:               {live_check['z_score']:+.2f}")
    print(f"  Within 95% CI:         {live_check['within_95_ci']}")
    print(f"  Assessment:            {live_check['assessment']}")

    # Note: live is for S1, not C2 yet
    print(f"\n  IMPORTANT: Live data is for S1 (current production).")
    print(f"  C2 has not yet been deployed live. This check validates")
    print(f"  the framework against S1 live data as a sanity check.")

    # -----------------------------------------------------------------------
    # 6. Stress test
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[6] STRESS TEST (VEC Holdout Equity Curve)")
    print("=" * 80)

    dd_stats = compute_drawdown_stats(ho_equity, ho_dates)

    print(f"\n  Max drawdown:         {dd_stats['max_drawdown']:.2%}")
    print(f"  Peak date:            {dd_stats['peak_date']}")
    print(f"  Trough date:          {dd_stats['max_dd_date']}")
    print(f"  Recovery date:        {dd_stats['recovery_date']}")
    print(f"  Drawdown duration:    {dd_stats['drawdown_duration_days']} trading days")
    if dd_stats["recovery_duration_days"] is not None:
        print(f"  Recovery duration:    {dd_stats['recovery_duration_days']} trading days")
        print(f"  Total peak-recovery:  {dd_stats['total_dd_to_recovery_days']} trading days")
    else:
        print(f"  Recovery:             NOT RECOVERED")

    print(f"\n  Drawdown episodes > 5%:")
    for ep in dd_stats["drawdown_episodes_gt5pct"]:
        print(
            f"    {ep['start_date']} to {ep['end_date']}: "
            f"{ep['depth']:.2%} ({ep['duration_days']} days)"
        )

    # Daily bootstrap for forward-looking stress
    print(f"\n  Forward-looking stress (daily block bootstrap, 6-month horizon):")

    # Compute daily returns for holdout
    ho_mask = dates >= pd.Timestamp(HOLDOUT_START)
    ho_daily_equity = c2_equity[ho_mask]
    ho_daily_returns = np.diff(ho_daily_equity) / ho_daily_equity[:-1]

    rng2 = np.random.default_rng(RNG_SEED + 1)
    daily_boot = daily_bootstrap(
        ho_daily_returns, N_BOOTSTRAP, block_size_days=FREQ * 2, rng=rng2
    )

    prob_dd_15 = (daily_boot["six_month_mdd"] < -0.15).mean()
    prob_dd_20 = (daily_boot["six_month_mdd"] < -0.20).mean()
    prob_dd_25 = (daily_boot["six_month_mdd"] < -0.25).mean()
    prob_neg_6m = (daily_boot["six_month_return"] < 0).mean()

    mdd_5th = np.percentile(daily_boot["six_month_mdd"], 5)
    mdd_median = np.percentile(daily_boot["six_month_mdd"], 50)
    ret_5th = np.percentile(daily_boot["six_month_return"], 5)
    ret_median = np.percentile(daily_boot["six_month_return"], 50)

    print(f"    P(6-month MDD > 15%):    {prob_dd_15:.1%}")
    print(f"    P(6-month MDD > 20%):    {prob_dd_20:.1%}")
    print(f"    P(6-month MDD > 25%):    {prob_dd_25:.1%}")
    print(f"    P(6-month return < 0):   {prob_neg_6m:.1%}")
    print(f"    6-month MDD 5th pct:     {mdd_5th:.2%}")
    print(f"    6-month MDD median:      {mdd_median:.2%}")
    print(f"    6-month return 5th pct:  {ret_5th:.2%}")
    print(f"    6-month return median:   {ret_median:.2%}")

    # -----------------------------------------------------------------------
    # 7. Stopping rules
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[7] STOPPING RULES FOR SEQUENTIAL MONITORING")
    print("=" * 80)

    rules = design_stopping_rules(trade_stats, boot_df)

    # Set MDD emergency stop from bootstrap
    rules["mdd_emergency_stop"] = float(abs(mdd_5th)) + 0.05  # 5th pct + 5% buffer
    rules["mdd_emergency_stop"] = round(min(rules["mdd_emergency_stop"], 0.25), 2)  # cap at 25%

    print(f"\n  Planned evaluation: {rules['planned_total_trades']} trades (50 rebalance periods)")
    print(f"  Minimum trades before ANY evaluation: {rules['min_trades_before_eval']}")
    print(f"  MDD emergency stop: {rules['mdd_emergency_stop']:.0%}")
    print(f"  Max consecutive loss stop: {rules['max_consecutive_loss_stop']} periods")
    print(f"  Sharpe futility: {rules['sharpe_futility_stop']}")

    print(f"\n  O'Brien-Fleming Sequential Boundaries (alpha=0.05, K=4 looks):")
    print(f"  {'Look':<6} {'Info%':<8} {'Trades':<10} {'z-boundary':<12} {'Action'}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*12} {'-'*40}")
    for b in rules["obrien_fleming_boundaries"]:
        print(
            f"  {b['look']:<6} {b['info_fraction']:<8.0%} "
            f"{b['trades_at_look']:<10} {b['z_boundary']:<12.3f} {b['action']}"
        )

    print(f"\n  Interpretation:")
    print(f"    - Do NOT evaluate before {rules['min_trades_before_eval']} trades (peeking bias)")
    print(f"    - At each look, compute z = mean_return / (std / sqrt(n))")
    print(f"    - If |z| exceeds boundary: can make a decision")
    print(f"    - If not: continue to next look")
    print(f"    - Emergency stop: MDD > {rules['mdd_emergency_stop']:.0%} at any time (unconditional)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary = {
        "strategy": "C2: AMIHUD + CALMAR + CORR_MKT",
        "bt_ho_return": f"{bt_stats['ho_return']:.1%}",
        "bt_ho_sharpe": f"{bt_stats['ho_sharpe']:.3f}",
        "bt_ho_mdd": f"{bt_stats['ho_mdd']:.1%}",
        "vec_ho_sharpe": f"{trade_stats['sharpe_ratio']:.3f}",
        "vec_ho_trades": trade_stats["n_trades"],
        "vec_ho_win_rate": f"{trade_stats['win_rate']:.1%}",
        "bootstrap_sharpe_ci": f"[{np.percentile(boot_df['sharpe'], 5):.2f}, {np.percentile(boot_df['sharpe'], 95):.2f}]",
        "bootstrap_return_ci": f"[{np.percentile(boot_df['total_return'], 5):.1%}, {np.percentile(boot_df['total_return'], 95):.1%}]",
        "prob_negative_return": f"{prob_neg:.1%}",
        "trades_for_sharpe_gt0_p05": next(
            r["trades_needed"]
            for r in sample_sizes
            if r["confidence"] == "p<0.05" and "effect=0.1" in r["target"]
        ),
        "trades_for_sharpe_gt1_p05": next(
            r["trades_needed"]
            for r in sample_sizes
            if r["confidence"] == "p<0.05" and "1.0" in r["target"]
        ),
        "months_for_sharpe_gt1_p05": next(
            r["approx_months"]
            for r in sample_sizes
            if r["confidence"] == "p<0.05" and "1.0" in r["target"]
        ),
        "mdd_emergency_stop": f"{rules['mdd_emergency_stop']:.0%}",
        "min_trades_before_eval": rules["min_trades_before_eval"],
        "prob_6m_mdd_gt15": f"{prob_dd_15:.1%}",
        "prob_6m_mdd_gt20": f"{prob_dd_20:.1%}",
        "live_consistency": live_check["assessment"],
    }

    print(f"\n  C2 statistical profile:")
    print(f"    BT HO Return:          {summary['bt_ho_return']}")
    print(f"    BT HO Sharpe:          {summary['bt_ho_sharpe']}")
    print(f"    BT HO MDD:             {summary['bt_ho_mdd']}")
    print(f"    VEC HO Sharpe:         {summary['vec_ho_sharpe']}")
    print(f"    Bootstrap Sharpe 90%CI: {summary['bootstrap_sharpe_ci']}")
    print(f"    Bootstrap Return 90%CI: {summary['bootstrap_return_ci']}")
    print(f"    P(negative return):     {summary['prob_negative_return']}")

    print(f"\n  Sample size requirements:")
    print(f"    Detect Sharpe>0 (p<0.05):  {summary['trades_for_sharpe_gt0_p05']} trades")
    print(f"    Detect Sharpe>1 (p<0.05):  {summary['trades_for_sharpe_gt1_p05']} trades (~{summary['months_for_sharpe_gt1_p05']} months)")

    print(f"\n  Risk assessment (6-month horizon):")
    print(f"    P(MDD > 15%):  {summary['prob_6m_mdd_gt15']}")
    print(f"    P(MDD > 20%):  {summary['prob_6m_mdd_gt20']}")

    print(f"\n  Monitoring rules:")
    print(f"    Min trades before eval:  {summary['min_trades_before_eval']}")
    print(f"    MDD emergency stop:      {summary['mdd_emergency_stop']}")
    print(f"    S1 live consistency:      {summary['live_consistency']}")

    # Save results
    output_dir = RESULTS_DIR / "significance_analysis"
    output_dir.mkdir(exist_ok=True)

    # Save summary JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save bootstrap distributions
    boot_df.to_csv(output_dir / "bootstrap_distributions.csv", index=False)

    # Save trade-level stats
    with open(output_dir / "trade_statistics.json", "w") as f:
        json.dump(trade_stats, f, indent=2)

    # Save stopping rules
    with open(output_dir / "stopping_rules.json", "w") as f:
        json.dump(rules, f, indent=2, default=str)

    # Save sample size table
    pd.DataFrame(sample_sizes).to_csv(output_dir / "sample_size_table.csv", index=False)

    # Save drawdown stats
    with open(output_dir / "drawdown_stats.json", "w") as f:
        json.dump(dd_stats, f, indent=2)

    # Save daily bootstrap stress
    with open(output_dir / "stress_test_6m.json", "w") as f:
        json.dump(
            {
                "prob_mdd_gt_15pct": float(prob_dd_15),
                "prob_mdd_gt_20pct": float(prob_dd_20),
                "prob_mdd_gt_25pct": float(prob_dd_25),
                "prob_negative_6m": float(prob_neg_6m),
                "mdd_5th_pct": float(mdd_5th),
                "mdd_median": float(mdd_median),
                "return_5th_pct": float(ret_5th),
                "return_median": float(ret_median),
            },
            f,
            indent=2,
        )

    print(f"\n  Results saved to: {output_dir}/")
    print(f"    summary.json")
    print(f"    bootstrap_distributions.csv")
    print(f"    trade_statistics.json")
    print(f"    stopping_rules.json")
    print(f"    sample_size_table.csv")
    print(f"    drawdown_stats.json")
    print(f"    stress_test_6m.json")


if __name__ == "__main__":
    main()
