#!/usr/bin/env python3
"""
生成风险受控的混合排序白名单

评分: blended = alpha * calibrated_norm + (1-alpha) * ic_rank_score
并确保至少保留原始 TopK 的一定比例（min_retain_ratio）。

输出:
- results/<latest_run>/whitelist_top{K}_blended_alpha{alpha}_retain{retain}.txt
- results/<latest_run>/blended_stats.txt (覆盖写)
"""
import os
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
    # 大值应更靠前，因此用降序百分位
    order = s.rank(method='average', ascending=False)
    return 1.0 - (order - 1) / (len(s) - 1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.6, help='blending 权重 alpha')
    ap.add_argument('--topk', type=int, default=2000, help='白名单 TopK')
    ap.add_argument('--retain', type=float, default=0.3, help='最小保留原始TopK比例 (0~1)')
    args = ap.parse_args()

    run_dir = load_latest_run_dir()
    wfo_path = run_dir / 'all_combos.parquet'
    calib_path = run_dir / 'all_combos_calibrated_gbdt_full.parquet'
    bt_candidates = list(Path('results_combo_wfo').rglob('*_full.csv'))
    bt_path = max(bt_candidates, key=lambda p: p.stat().st_mtime) if bt_candidates else None

    if not wfo_path.exists() or not calib_path.exists():
        raise FileNotFoundError('缺少 all_combos 或校准结果文件，请先完成WFO与校准训练')

    df = pd.read_parquet(calib_path)  # 已包含WFO列与 calibrated_sharpe_full
    # 规范化分数
    df['calib_score'] = minmax_norm(df['calibrated_sharpe_full'])
    df['ic_rank_score'] = percentile_rank_desc(df['mean_oos_ic'])
    df['blended_score'] = args.alpha * df['calib_score'] + (1-args.alpha) * df['ic_rank_score']

    # 原始与混合TopK
    orig_top = df.nlargest(args.topk, 'mean_oos_ic').copy()
    blended = df.nlargest(args.topk, 'blended_score').copy()

    # 保底保留策略
    need_keep = int(args.topk * args.retain)
    keep_orig = orig_top.head(need_keep)['combo'].tolist()
    sel = []
    seen = set()
    for c in keep_orig:
        if c not in seen:
            sel.append(c); seen.add(c)
    for c in blended['combo']:
        if c not in seen:
            sel.append(c); seen.add(c)
        if len(sel) >= args.topk:
            break

    # 导出白名单
    out_whitelist = run_dir / f"whitelist_top{args.topk}_blended_alpha{args.alpha}_retain{int(args.retain*100)}.txt"
    pd.Series(sel).to_csv(out_whitelist, index=False, header=False)

    # 统计
    orig_set = set(orig_top['combo'])
    blended_set = set(blended['combo'])
    final_set = set(sel)
    overlap_ob = len(orig_set & blended_set)
    overlap_of = len(orig_set & final_set)
    overlap_bf = len(blended_set & final_set)

    lines = []
    lines.append(f'alpha={args.alpha}, topk={args.topk}, retain={args.retain}\n')
    lines.append(f'Overlap original vs blended: {overlap_ob}/{args.topk} ({overlap_ob/args.topk*100:.1f}%)\n')
    lines.append(f'Overlap original vs final:   {overlap_of}/{args.topk} ({overlap_of/args.topk*100:.1f}%)\n')
    lines.append(f'Overlap blended  vs final:   {overlap_bf}/{args.topk} ({overlap_bf/args.topk*100:.1f}%)\n')

    # 若有回测CSV，估算 Precision@K（以Sharpe为真值）
    if bt_path and bt_path.exists():
        bt = pd.read_csv(bt_path)[['combo','sharpe']]
        dtest = df.merge(bt, on='combo', how='left')
        # 实际TopK
        real_top = set(dtest.nlargest(args.topk, 'sharpe')['combo'])
        final_prec = len(real_top & final_set) / args.topk
        orig_prec = len(real_top & orig_set) / args.topk
        blended_prec = len(real_top & blended_set) / args.topk
        lines.append(f'Precision@{args.topk} ( Sharpe truth ):\n')
        lines.append(f'  original: {orig_prec*100:.1f}%\n')
        lines.append(f'  blended:  {blended_prec*100:.1f}%\n')
        lines.append(f'  final:    {final_prec*100:.1f}%\n')

    out_stats = run_dir / 'blended_stats.txt'
    out_stats.write_text(''.join(lines), encoding='utf-8')
    print('Saved:', out_whitelist)
    print('Saved:', out_stats)


if __name__ == '__main__':
    main()
