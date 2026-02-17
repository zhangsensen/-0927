#!/usr/bin/env python3
"""
Premium/Discount Factor Validation
===================================
Quick validation of ETF premium/discount (折溢价) as a potential new factor.

Tests:
1. Data coverage & quality
2. Factor variants: raw, deviation (20D MA), change (5D/20D), Z-score
3. IC / ICIR (Train vs Holdout)
4. Orthogonality vs existing top factors (from factor library)
5. Cross-sectional spread analysis

Author: Claude
Date: 2026-02-17
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

QDII = {"159920", "513050", "513100", "513130", "513180", "513400", "513500", "513520"}
TRAIN_END = "2025-04-30"
HO_START = "2025-05-01"
DATA_END = "2026-02-11"


def load_premium_panel(etf_codes: list, start: str = "2020-01-01", end: str = DATA_END) -> pd.DataFrame:
    """Load premium_rate data into panel (dates x ETFs)."""
    factors_dir = PROJECT_ROOT / "raw" / "ETF" / "factors"
    series_dict = {}
    for code in etf_codes:
        fp = factors_dir / f"premium_rate_{code}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.drop_duplicates(subset="trade_date", keep="last")
        s = df.set_index("trade_date")["premium_rate"]
        series_dict[code] = s
    panel = pd.DataFrame(series_dict)
    panel = panel.loc[start:end]
    panel = panel.sort_index()
    return panel


def compute_ic_series(factor_panel: pd.DataFrame, return_panel: pd.DataFrame) -> pd.Series:
    """Compute cross-sectional rank IC (Spearman) per date."""
    common_dates = factor_panel.index.intersection(return_panel.index)
    common_cols = factor_panel.columns.intersection(return_panel.columns)
    f = factor_panel.loc[common_dates, common_cols]
    r = return_panel.loc[common_dates, common_cols]

    ics = []
    for dt in common_dates:
        frow = f.loc[dt].dropna()
        rrow = r.loc[dt].reindex(frow.index).dropna()
        common = frow.index.intersection(rrow.index)
        if len(common) < 10:
            continue
        ic, _ = stats.spearmanr(frow[common], rrow[common])
        ics.append((dt, ic))
    return pd.Series(dict(ics))


def main():
    from etf_strategy.core.data_loader import DataLoader

    data_dir = PROJECT_ROOT / "raw" / "ETF" / "daily"
    loader = DataLoader(data_dir=str(data_dir))
    ohlcv = loader.load_ohlcv(start_date="2020-01-01", end_date=DATA_END)
    close = ohlcv["close"]

    all_codes = sorted(close.columns.tolist())
    a_share_codes = sorted([c for c in all_codes if c not in QDII])

    # Forward 5-day return (FREQ=5)
    fwd_ret = close.shift(-5) / close - 1
    fwd_ret_a = fwd_ret[a_share_codes]

    print("=" * 70)
    print("Premium/Discount Factor Validation")
    print("=" * 70)
    print(f"Total ETFs: {len(all_codes)} (A-share: {len(a_share_codes)}, QDII: {len(all_codes)-len(a_share_codes)})")
    print(f"Train: 2020-01 ~ {TRAIN_END} | Holdout: {HO_START} ~ {DATA_END}")
    print()

    # --- Step 1: Load premium data ---
    print("STEP 1: Data Coverage")
    print("-" * 70)
    prem = load_premium_panel(all_codes)
    print(f"Panel shape: {prem.shape}")
    coverage = prem.notna().sum() / len(prem) * 100
    print(f"  Min coverage: {coverage.min():.1f}%, Median: {coverage.median():.1f}%")
    missing = [c for c in all_codes if c not in prem.columns]
    if missing:
        print(f"  MISSING ETFs: {missing}")
    print()

    # A-share vs QDII distribution
    a_vals = prem[[c for c in prem.columns if c not in QDII]].values.ravel()
    a_vals = a_vals[~np.isnan(a_vals)]
    q_vals = prem[[c for c in prem.columns if c in QDII]].values.ravel()
    q_vals = q_vals[~np.isnan(q_vals)]
    print(f"A-share premium: mean={np.mean(a_vals):.4f}%, std={np.std(a_vals):.4f}%")
    print(f"  [P5={np.percentile(a_vals,5):.3f}%, P95={np.percentile(a_vals,95):.3f}%]")
    print(f"QDII premium:    mean={np.mean(q_vals):.4f}%, std={np.std(q_vals):.4f}%")
    print(f"  [P5={np.percentile(q_vals,5):.3f}%, P95={np.percentile(q_vals,95):.3f}%]")
    print()

    # --- Step 2: Factor variants ---
    print("STEP 2: Factor Variants")
    print("-" * 70)
    variants = {}
    variants["PREMIUM_RAW"] = prem.copy()
    prem_ma20 = prem.rolling(20, min_periods=10).mean()
    variants["PREMIUM_DEV_20D"] = prem - prem_ma20
    variants["PREMIUM_CHG_5D"] = prem.diff(5)
    variants["PREMIUM_CHG_20D"] = prem.diff(20)
    prem_std20 = prem.rolling(20, min_periods=10).std()
    variants["PREMIUM_ZSCORE_20D"] = (prem - prem_ma20) / prem_std20.replace(0, np.nan)

    for name, v in variants.items():
        valid = v.notna().sum().sum()
        total = v.shape[0] * v.shape[1]
        print(f"  {name}: {valid/total*100:.1f}% valid")
    print()

    # --- Step 3: IC/ICIR ---
    print("STEP 3: IC / ICIR (A-share only, FREQ=5 forward returns)")
    print("-" * 70)
    print(f"{'Factor':<22s} {'Train IC':>9s} {'Train ICIR':>11s} {'HO IC':>9s} {'HO ICIR':>11s} {'Stab':>5s} {'Dir':>4s}")
    print("-" * 70)

    results = []
    for name, variant in variants.items():
        v_a = variant[[c for c in variant.columns if c in a_share_codes]]
        ic_series = compute_ic_series(v_a, fwd_ret_a)

        ic_train = ic_series[ic_series.index <= TRAIN_END]
        ic_ho = ic_series[ic_series.index > TRAIN_END]

        if len(ic_train) < 20 or len(ic_ho) < 5:
            continue

        t_ic = ic_train.mean()
        t_icir = t_ic / ic_train.std() if ic_train.std() > 0 else 0
        h_ic = ic_ho.mean()
        h_icir = h_ic / ic_ho.std() if ic_ho.std() > 0 else 0
        sign_stab = abs(np.sign(ic_train).mean())
        dir_match = "YES" if np.sign(t_ic) == np.sign(h_ic) else "NO"

        results.append({
            "name": name, "t_ic": t_ic, "t_icir": t_icir,
            "h_ic": h_ic, "h_icir": h_icir,
            "stab": sign_stab, "dir": dir_match,
        })
        print(f"{name:<22s} {t_ic:>+9.4f} {t_icir:>+11.4f} {h_ic:>+9.4f} {h_icir:>+11.4f} {sign_stab:>5.2f} {dir_match:>4s}")

    print()

    # --- Step 4: Orthogonality ---
    print("STEP 4: Orthogonality vs Existing Factors (A-share)")
    print("-" * 70)

    # Compute existing factors directly
    close_a = close[a_share_codes]
    high_a = ohlcv["high"][[c for c in ohlcv["high"].columns if c in a_share_codes]]
    low_a = ohlcv["low"][[c for c in ohlcv["low"].columns if c in a_share_codes]]

    ref_factors = {}

    # SLOPE_20D: linear regression slope (simplified: use rolling return as proxy)
    ref_factors["SLOPE_20D"] = close_a.pct_change(20)

    # PRICE_POSITION_120D: (close - low_120) / (high_120 - low_120)
    hh = high_a.rolling(120, min_periods=60).max()
    ll = low_a.rolling(120, min_periods=60).min()
    ref_factors["PP_120D"] = (close_a - ll) / (hh - ll).replace(0, np.nan)

    # ADX_14D proxy: use absolute return volatility ratio
    ref_factors["VOLATILITY_20D"] = close_a.pct_change().rolling(20).std()

    # Non-OHLCV: load fund_share
    try:
        share_panel = loader.load_fund_share(etf_codes=a_share_codes,
                                              start_date="2020-01-01", end_date=DATA_END)
        if share_panel is not None and not share_panel.empty:
            # 5D change
            ref_factors["SHARE_CHG_5D"] = share_panel.pct_change(5)
    except Exception as e:
        print(f"  Skip SHARE_CHG_5D: {e}")

    best_premium_variants = ["PREMIUM_RAW", "PREMIUM_DEV_20D", "PREMIUM_CHG_5D"]
    for pname in best_premium_variants:
        pv = variants[pname][[c for c in variants[pname].columns if c in a_share_codes]]
        print(f"\n  {pname}:")
        for ename, efactor in ref_factors.items():
            ef = efactor[[c for c in efactor.columns if c in a_share_codes]] if hasattr(efactor, 'columns') else efactor
            common_dates = pv.index.intersection(ef.index)
            common_cols = pv.columns.intersection(ef.columns)
            if len(common_dates) < 100 or len(common_cols) < 10:
                print(f"    vs {ename:25s}: insufficient overlap (dates={len(common_dates)}, cols={len(common_cols)})")
                continue

            # Sample 200 dates for speed
            step = max(1, len(common_dates) // 200)
            sample_dates = common_dates[::step]
            rhos = []
            for dt in sample_dates:
                p_row = pv.loc[dt, common_cols].dropna()
                e_row = ef.loc[dt, common_cols].reindex(p_row.index).dropna()
                shared = p_row.index.intersection(e_row.index)
                if len(shared) < 10:
                    continue
                rho, _ = stats.spearmanr(p_row[shared], e_row[shared])
                rhos.append(rho)
            if rhos:
                mean_rho = np.mean(rhos)
                std_rho = np.std(rhos)
                tag = "ORTHOGONAL" if abs(mean_rho) < 0.3 else ("MODERATE" if abs(mean_rho) < 0.5 else "REDUNDANT")
                print(f"    vs {ename:25s}: rho={mean_rho:+.3f} ± {std_rho:.3f}  [{tag}]")

    print()

    # --- Step 5: Cross-sectional spread ---
    print("STEP 5: Cross-Sectional Spread (A-share only)")
    print("-" * 70)
    a_prem = prem[[c for c in prem.columns if c not in QDII]]
    daily_std = a_prem.std(axis=1)
    daily_range = a_prem.max(axis=1) - a_prem.min(axis=1)
    print(f"  Daily cross-sectional std:   mean={daily_std.mean():.4f}%, median={daily_std.median():.4f}%")
    print(f"  Daily cross-sectional range: mean={daily_range.mean():.4f}%, median={daily_range.median():.4f}%")
    # Compare with typical IC-relevant threshold
    print(f"  For reference: daily premium std is ~{daily_std.mean():.3f}% vs typical factor IC ~0.05")
    print(f"  {'SUFFICIENT' if daily_std.mean() > 0.05 else 'POSSIBLY TOO SMALL'} cross-sectional variation")
    print()

    # --- Summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    viable = []
    for r in results:
        if abs(r["h_icir"]) > 0.2 and r["dir"] == "YES" and r["stab"] >= 0.5:
            viable.append(r)
            print(f"  VIABLE: {r['name']} | HO ICIR={r['h_icir']:+.3f}")

    if not viable:
        print("  NO VIABLE VARIANTS (threshold: |HO ICIR|>0.2, direction consistent, stability>=0.5)")

    print()
    if viable:
        print("VERDICT: Premium factor shows POTENTIAL. Consider Phase 2 (WFO integration test).")
    else:
        print("VERDICT: Premium factor does NOT meet minimum bar.")
        print("  A-share ETF arbitrage likely keeps premium too stable for cross-sectional alpha.")


if __name__ == "__main__":
    main()
