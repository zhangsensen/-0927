#!/usr/bin/env python3
"""
评估混合排序参数网格

Grid:
  - alpha ∈ {0.5, 0.6, 0.7}
  - retain ∈ {0.2, 0.3, 0.4}
K: {50, 100, 200, 500, 1000, 2000}

输出:
  results/<latest_run>/blended_grid_eval.csv
  results/<latest_run>/blended_grid_eval.md （简报）
"""
from pathlib import Path
import pandas as pd
import numpy as np


def load_latest_run_dir() -> Path:
    latest = Path('results/.latest_run').read_text().strip()
    return Path('results') / latest


def minmax_norm(s: pd.Series) -> pd.Series:
    mn, mx = float(s.min()), float(s.max())
    if mx - mn < 1e-12:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def percentile_rank_desc(s: pd.Series) -> pd.Series:
    order = s.rank(method='average', ascending=False)
    return 1.0 - (order - 1) / (len(s) - 1)


def get_latest_backtest_csv() -> Path | None:
    cands = list(Path('results_combo_wfo').rglob('*_full.csv'))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def main():
    run_dir = load_latest_run_dir()
    cal_path = run_dir / 'all_combos_calibrated_gbdt_full.parquet'
    if not cal_path.exists():
        raise FileNotFoundError('缺少校准结果 parquet，请先完成 train_calibrator_full.py')

    df = pd.read_parquet(cal_path)
    bt_csv = get_latest_backtest_csv()
    bt = None
    if bt_csv and bt_csv.exists():
        bt = pd.read_csv(bt_csv)[['combo', 'sharpe']]
        df = df.merge(bt, on='combo', how='left')

    df['calib_score'] = minmax_norm(df['calibrated_sharpe_full'])
    df['ic_rank_score'] = percentile_rank_desc(df['mean_oos_ic'])

    alphas = [0.5, 0.6, 0.7]
    retains = [0.2, 0.3, 0.4]
    topk_list = [50, 100, 200, 500, 1000, 2000]

    rows = []

    for alpha in alphas:
        df['blended_score'] = alpha * df['calib_score'] + (1 - alpha) * df['ic_rank_score']
        for retain in retains:
            for topk in topk_list:
                orig_top = df.nlargest(topk, 'mean_oos_ic')
                blended = df.nlargest(topk, 'blended_score')

                # 保底保留
                need_keep = int(topk * retain)
                keep_orig = orig_top.head(need_keep)['combo'].tolist()
                sel = []
                seen = set()
                for c in keep_orig:
                    if c not in seen:
                        sel.append(c); seen.add(c)
                for c in blended['combo']:
                    if c not in seen:
                        sel.append(c); seen.add(c)
                    if len(sel) >= topk:
                        break

                orig_set = set(orig_top['combo'])
                blended_set = set(blended['combo'])
                final_set = set(sel)

                overlap_ob = len(orig_set & blended_set) / topk
                overlap_of = len(orig_set & final_set) / topk
                overlap_bf = len(blended_set & final_set) / topk

                # Precision@K（Sharpe 为真值，若存在）
                if bt is not None and 'sharpe' in df.columns:
                    real_top = set(df.nlargest(topk, 'sharpe')['combo'])
                    prec_orig = len(real_top & orig_set) / topk
                    prec_blended = len(real_top & blended_set) / topk
                    prec_final = len(real_top & final_set) / topk
                else:
                    prec_orig = prec_blended = prec_final = np.nan

                rows.append({
                    'alpha': alpha,
                    'retain': retain,
                    'topk': topk,
                    'overlap_original_blended': overlap_ob,
                    'overlap_original_final': overlap_of,
                    'overlap_blended_final': overlap_bf,
                    'precision_original': prec_orig,
                    'precision_blended': prec_blended,
                    'precision_final': prec_final,
                })

    out_df = pd.DataFrame(rows)
    csv_path = run_dir / 'blended_grid_eval.csv'
    out_df.to_csv(csv_path, index=False)

    # 简报 Markdown
    md_lines = [
        '# Blended Grid Evaluation\n',
        f'Run dir: {run_dir}\n\n',
        'Columns: alpha, retain, topk, overlaps, precision_*\n\n',
        out_df.to_string(index=False, float_format=lambda x: f'{x:.3f}')
    ]
    md_path = run_dir / 'blended_grid_eval.md'
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')
    print('Saved:', csv_path)
    print('Saved:', md_path)


if __name__ == '__main__':
    main()
