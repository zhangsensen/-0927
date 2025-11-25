#!/usr/bin/env python3
"""分析自检权重与生产权重差异分布的辅助脚本。

步骤:
1. 读取 real_backtest/top200_baseline_partial_summary.json (或指定文件)
2. 可选采样重新计算某组合在若干调仓日的窗口IC权重 (简单重算路径)
3. 比较与生产窗口均值 (稳定 + 预计算) 权重差异，输出最大/均值/分位 diff

用法:
    python scripts/weight_diff_analyzer.py --combo "ADX_14D + PRICE_POSITION_20D" --freq 8 --sample 20

参数:
    --combo    组合字符串 (与 WFO 输出格式一致)
    --freq     换仓频率
    --sample   抽样调仓次数 (默认10)
    --baseline-json  指定基线摘要文件

输出:
    JSON + 人类可读表格，便于调整 RB_NL_CHECK_RTOL / RB_NL_CHECK_ATOL。
"""
import json
import argparse
import numpy as np
from pathlib import Path
import yaml
from etf_rotation_optimized.core.data_loader import DataLoader
from etf_rotation_optimized.real_backtest.run_production_backtest import (
    _compute_or_load_daily_ic_memmap,
    compute_spearman_ic_numba,
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--combo', required=True, help='组合字符串: "A + B + C"')
    ap.add_argument('--freq', type=int, default=8)
    ap.add_argument('--sample', type=int, default=10)
    ap.add_argument('--baseline-json', default='real_backtest/top200_baseline_partial_summary.json')
    return ap.parse_args()

def main():
    args = parse_args()
    combo_parts = [x.strip() for x in args.combo.split('+') if x.strip()]
    # 加载配置与数据
    cfg = yaml.safe_load(open('configs/combo_wfo_config.yaml','r'))
    loader = DataLoader(
        data_dir=cfg['data'].get('data_dir'),
        cache_dir=cfg['data'].get('cache_dir'),
    )
    o = loader.load_ohlcv(
        etf_codes=cfg['data']['symbols'],
        start_date=cfg['data']['start_date'],
        end_date=cfg['data']['end_date'],
        use_cache=True,
    )
    returns = o['close'].pct_change(fill_method=None).values

    # 读取最新 run 的因子矩阵
    results_root = Path('results')
    latest = (results_root/'.latest_run').read_text().strip()
    run_dir = results_root/latest
    factors_dir = run_dir/'factors'
    parquets = sorted(factors_dir.glob('*.parquet'))
    import pyarrow.parquet as pq
    f_arrays=[]; f_names=[]; ref_index=None; ref_cols=None
    for fp in parquets:
        tbl=pq.read_table(fp)
        dfp=tbl.to_pandas()
        if ref_index is None:
            ref_index=dfp.index; ref_cols=list(dfp.columns)
        else:
            dfp=dfp.reindex(index=ref_index, columns=ref_cols)
        f_arrays.append(dfp.values); f_names.append(fp.stem)
    factors_data = np.stack(f_arrays, axis=-1)  # (T,N,F)
    name_to_idx={n:i for i,n in enumerate(f_names)}
    sel_idxs=[name_to_idx[p] for p in combo_parts if p in name_to_idx]
    if not sel_idxs:
        raise SystemExit('组合因子未在当前 run 中找到')

    lookback=252
    start_idx=lookback+1
    T=factors_data.shape[0]
    rebalance_indices=np.arange(start_idx, T, args.freq, dtype=np.int32)
    # 抽样
    stride=max(1, len(rebalance_indices)//max(1,args.sample))
    sampled=rebalance_indices[::stride][:args.sample]

    # 生产路径: 使用稳定秩 + daily IC memmap
    daily_ic_full=_compute_or_load_daily_ic_memmap(factors_data, returns, True)
    diffs=[]
    for day_idx in sampled:
        hist_start=max(0, day_idx-lookback); hist_end=day_idx
        cols=np.array(sel_idxs,dtype=np.int64)
        di_slice=daily_ic_full[hist_start:hist_end][:, cols]
        window_mean=np.nanmean(di_slice, axis=0)
        abs_mean=np.abs(window_mean)
        if np.nansum(abs_mean)>0:
            w_prod=abs_mean/np.nansum(abs_mean)
        else:
            w_prod=np.full(len(cols),1.0/len(cols))

        # 重算路径（简单 rank 即直接 Spearman 窗口日均值）
        returns_hist=returns[hist_start:hist_end]
        w_recalc=np.zeros(len(cols),dtype=np.float64)
        for i,fid in enumerate(cols):
            ic=compute_spearman_ic_numba(factors_data[hist_start:hist_end,:,fid], returns_hist)
            w_recalc[i]=abs(ic)
        if w_recalc.sum()>0:
            w_recalc/=w_recalc.sum()
        else:
            w_recalc[:]=1.0/len(cols)
        diff=float(np.max(np.abs(w_prod - w_recalc)))
        diffs.append(diff)

    diffs_arr=np.array(diffs)
    summary={
        'combo': args.combo,
        'freq': args.freq,
        'n_samples': int(len(diffs_arr)),
        'max_diff': float(diffs_arr.max()) if len(diffs_arr) else None,
        'mean_diff': float(diffs_arr.mean()) if len(diffs_arr) else None,
        'p95_diff': float(np.quantile(diffs_arr,0.95)) if len(diffs_arr) else None,
        'diffs': [float(d) for d in diffs],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(diffs_arr):
        print('\nDiff distribution:')
        print('  max   : %.4e' % diffs_arr.max())
        print('  mean  : %.4e' % diffs_arr.mean())
        print('  p95   : %.4e' % np.quantile(diffs_arr,0.95))
        print('  sample:', ', '.join(f'{d:.4e}' for d in diffs_arr))

if __name__=='__main__':
    main()
