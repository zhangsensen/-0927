
import sys
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.utils.rebalance import generate_rebalance_schedule, shift_timing_signal
from etf_strategy.core.market_timing import LightTimingModule


def compute_metrics(daily_returns: pd.Series) -> dict:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return {}
    ann_return = (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1
    ann_vol = daily_returns.std(ddof=0) * np.sqrt(252)
    sharpe = (daily_returns.mean() * 252 / ann_vol) if ann_vol > 0 else 0.0
    cum = (1 + daily_returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return {'ann_return': ann_return, 'ann_vol': ann_vol, 'sharpe': sharpe, 'max_dd': abs(max_dd)}


def hrp_weights(cov: pd.DataFrame, shrink: float = 0.10) -> pd.Series:
    if cov.shape[0] == 1:
        return pd.Series([1.0], index=cov.index)

    # Handle zero variance
    std = np.sqrt(np.diag(cov))
    if np.any(std == 0):
        print("Warning: Found strategies with zero variance. Removing them.")
        valid_idx = std > 0
        cov = cov.loc[valid_idx, valid_idx]
        std = std[valid_idx]
        if cov.empty:
            return pd.Series()

    # Light shrinkage toward diagonal to stabilize distances
    if shrink > 0:
        diag = np.diag(np.diag(cov))
        cov = (1 - shrink) * cov + shrink * diag

    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1e-9
    outer = np.outer(std, std)
    corr = cov / outer
    np.fill_diagonal(corr.values, 1.0)
    corr = corr.clip(-1.0, 1.0)  # Ensure valid range

    dist = np.sqrt(0.5 * (1 - corr))
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist.values, 0)

    # Ensure symmetry for squareform
    dist_vals = dist.values
    dist_vals = (dist_vals + dist_vals.T) / 2
    np.fill_diagonal(dist_vals, 0)

    link = linkage(squareform(dist_vals), method='average')
    order = leaves_list(link)
    ordered_cov = cov.iloc[order, order]
    weights = pd.Series(1.0, index=ordered_cov.index)

    def split_cluster(cov_mat, w):
        if len(cov_mat) == 1:
            return w
        split = len(cov_mat) // 2
        left = cov_mat.iloc[:split, :split]
        right = cov_mat.iloc[split:, split:]
        w_left = 1.0 / np.diag(left).sum()
        w_right = 1.0 / np.diag(right).sum()
        alpha_left = w_left / (w_left + w_right)
        w[left.index] *= alpha_left
        w[right.index] *= (1 - alpha_left)
        w = split_cluster(left, w)
        w = split_cluster(right, w)
        return w

    weights = split_cluster(ordered_cov, weights)
    return weights.reindex(cov.index).fillna(0) / weights.sum()


def main():
    print("🔍 Selecting Robust Strategies...")
    df = pd.read_csv('results/holdout_validation/passed_strategies_20251209_215429.csv')

    robust_df = df[
        (df['train_sharpe'] > 0.4)
        & (df['holdout_sharpe'] > 1.0)
        & (df['train_max_dd'] < 0.35)
        & (df['holdout_return'] > 0.20)
    ].copy()

    print(f"Total Passed: {len(df)}")
    print(f"Robust Selected: {len(robust_df)}")
    print(robust_df[['combo', 'train_sharpe', 'holdout_sharpe']].head())

    if len(robust_df) == 0:
        print("❌ No robust strategies found. Relaxing criteria...")
        robust_df = df[
            (df['train_sharpe'] > 0.2)
            & (df['holdout_sharpe'] > 0.5)
            & (df['holdout_return'] > 0.0)
        ].copy()
        print(f"Relaxed Robust Selected: {len(robust_df)}")

    target_combos = robust_df['combo'].tolist()

    print("\n🚀 Running VEC Backtest for Robust Strategies...")

    with open(ROOT / 'configs/combo_wfo_config.yaml') as f:
        config = yaml.safe_load(f)
    data_dir = Path(config['data']['data_dir'])
    loader = DataLoader(data_dir=data_dir, cache_dir=ROOT / '.cache')
    etf_files = list(data_dir.glob('*.parquet'))
    etf_codes = [f.stem.split('_')[0].split('.')[0] for f in etf_files]
    ohlcv = loader.load_ohlcv(etf_codes=etf_codes, start_date='2025-01-01', end_date='2025-12-08')

    all_factors = set()
    for combo in target_combos:
        for f in combo.split(' + '):
            all_factors.add(f.strip())

    print(f"Computing {len(all_factors)} unique factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors({f: raw_factors_df[f] for f in all_factors})

    dates = std_factors[list(all_factors)[0]].index
    HOLDOUT_START = '2025-06-01'
    HOLDOUT_END = '2025-12-08'
    start_idx = np.where(dates >= HOLDOUT_START)[0][0]
    end_idx = np.where(dates <= HOLDOUT_END)[0][-1]
    slice_start = max(0, start_idx - 1)
    slice_end = end_idx + 1
    holdout_dates = dates[slice_start:slice_end]

    close_arr = ohlcv['close'].values[slice_start:slice_end]
    timing_module = LightTimingModule(extreme_threshold=-0.1, extreme_position=0.1)
    timing_series = timing_module.compute_position_ratios(ohlcv['close'])
    timing_arr = shift_timing_signal(timing_series.values)[slice_start:slice_end]

    daily_returns_dict = {}

    for combo in tqdm(target_combos):
        factors = [f.strip() for f in combo.split(' + ')]
        factors_3d = np.stack([std_factors[f].values[slice_start:slice_end] for f in factors], axis=-1)
        combined_score = np.sum(factors_3d, axis=2)

        n_dates, n_etfs = close_arr.shape
        schedule = generate_rebalance_schedule(n_dates, 1, 3)
        cash = 1_000_000.0
        positions = np.zeros(n_etfs)
        equity = np.zeros(n_dates)
        equity[0] = cash

        for t in range(1, n_dates):
            equity[t] = cash + np.nansum(positions * close_arr[t])
            if t in schedule:
                for i in range(n_etfs):
                    if positions[i] > 0:
                        px = close_arr[t, i]
                        if not np.isnan(px):
                            cash += positions[i] * px * (1 - 0.0002)
                positions[:] = 0

                scores = combined_score[t-1].copy()
                valid = ~np.isnan(scores) & ~np.isnan(close_arr[t])
                if np.any(valid):
                    scores[~valid] = -np.inf
                    top_idx = np.argsort(scores)[-2:][::-1]
                    pos_ratio = timing_arr[t]
                    if np.isnan(pos_ratio):
                        pos_ratio = 1.0
                    pos_ratio = np.clip(pos_ratio, 0.0, 1.0)

                    invest = cash * pos_ratio
                    per_pos = invest / len(top_idx)
                    for idx in top_idx:
                        px = close_arr[t, idx]
                        if not np.isnan(px) and px > 0:
                            shares = int(per_pos / px)
                            cost = shares * px * (1 + 0.0002)
                            if cost <= cash:
                                positions[idx] = shares
                                cash -= cost

        ret = np.diff(equity) / equity[:-1]
        daily_returns_dict[combo] = ret

    returns_df = pd.DataFrame(daily_returns_dict, index=holdout_dates[1:])
    returns_df.to_csv('results/holdout_validation/robust_daily_returns.csv')
    print("Saved robust daily returns.")

    print("\n⚖️ Running HRP on Robust Set...")

    cov_mat_full = returns_df.cov() * 252
    w_hrp_full = hrp_weights(cov_mat_full, shrink=0.10)
    ret_hrp_full = returns_df @ w_hrp_full
    perf_hrp_full = compute_metrics(ret_hrp_full)

    top40_combos = robust_df.sort_values('holdout_sharpe', ascending=False).head(40)['combo'].tolist()
    top40_returns_df = returns_df[top40_combos]
    cov_mat_top40 = top40_returns_df.cov() * 252
    w_hrp_top40 = hrp_weights(cov_mat_top40, shrink=0.10)
    ret_hrp_top40 = top40_returns_df @ w_hrp_top40
    perf_hrp_top40 = compute_metrics(ret_hrp_top40)

    print(f"\n[Robust HRP Portfolio - FULL]")
    print(f"  Strategies: {len(w_hrp_full)} (Weights > 1%: {(w_hrp_full > 0.01).sum()})")
    print(f"  Sharpe: {perf_hrp_full['sharpe']:.3f}")
    print(f"  Return: {perf_hrp_full['ann_return']:.2%}")
    print(f"  MaxDD:  {perf_hrp_full['max_dd']:.2%}")

    print(f"\n[Robust HRP Portfolio - TOP40]")
    print(f"  Strategies: {len(w_hrp_top40)} (Weights > 1%: {(w_hrp_top40 > 0.01).sum()})")
    print(f"  Sharpe: {perf_hrp_top40['sharpe']:.3f}")
    print(f"  Return: {perf_hrp_top40['ann_return']:.2%}")
    print(f"  MaxDD:  {perf_hrp_top40['max_dd']:.2%}")

    w_df_full = pd.DataFrame({'combo': w_hrp_full.index, 'weight': w_hrp_full.values})
    w_df_full = w_df_full[w_df_full['weight'] > 0.001].sort_values('weight', ascending=False)
    w_df_full.to_csv('results/holdout_validation/portfolio_weights_robust_full_shrink.csv', index=False)

    w_df_top40 = pd.DataFrame({'combo': w_hrp_top40.index, 'weight': w_hrp_top40.values})
    w_df_top40 = w_df_top40[w_df_top40['weight'] > 0.001].sort_values('weight', ascending=False)
    w_df_top40.to_csv('results/holdout_validation/portfolio_weights_robust_top40_shrink.csv', index=False)

    print("Saved robust weights (full + top40) with shrinkage.")


if __name__ == "__main__":
    main()
