#!/usr/bin/env python3
"""
频率 × 持仓数 轻量网格扫描
=================================
目标：对最新回测结果中 Sharpe 排名前 TopK 的组合，测试一组备选调仓频率与持仓数的笛卡尔积，评估风险收益改善空间。

特点：
- 仅使用已有的因子与回测逻辑（调用 backtest_no_lookahead）
- 不修改核心模块；可安全删除，不影响主流程
- 支持命令行参数控制 TopK, 频率集合, 持仓数集合，并行核数
- 输出：CSV + Markdown 汇总报告

用法示例：
    python scripts/run_freq_pos_grid.py \
        --topk 200 \
        --freqs 5,6,7,8,10 \
        --positions 4,5,6 \
        --jobs 8 \
        --output-dir results/grid_scan

可选：设置环境变量 RB_PROFILE_BACKTEST=1 以记录性能摘要。
"""
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

# 回测核心函数
sys.path.append(str(Path(__file__).resolve().parent.parent / 'etf_rotation_v2_breadth' / 'real_backtest'))
from etf_rotation_v2_breadth.real_backtest.run_production_backtest import backtest_no_lookahead  # type: ignore

# 因子与数据加载依赖
sys.path.append(str(Path(__file__).resolve().parent.parent / 'etf_rotation_v2_breadth'))
from etf_rotation_v2_breadth.core.cross_section_processor import CrossSectionProcessor  # type: ignore
from etf_rotation_v2_breadth.core.data_loader import DataLoader  # type: ignore
from etf_rotation_v2_breadth.core.precise_factor_library_v2 import PreciseFactorLibrary  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='频率×持仓数网格扫描')
    p.add_argument('--topk', type=int, default=200, help='选取 Sharpe TopK 组合进行扫描')
    p.add_argument('--freqs', type=str, default='5,6,7,8,10', help='调仓频率列表，逗号分隔')
    p.add_argument('--positions', type=str, default='4,5,6', help='持仓数量列表，逗号分隔')
    p.add_argument('--jobs', type=int, default=8, help='并行核数')
    p.add_argument('--output-dir', type=str, default='results/grid_scan', help='输出目录根路径')
    p.add_argument('--latest-backtest-dir', type=str, default='etf_rotation_v2_breadth/results_combo_wfo', help='搜索最新回测CSV的目录')
    p.add_argument('--latest-backtest-pattern', type=str, default='top12597_backtest_by_ic_*_*.csv', help='匹配最新结果的文件模式')
    p.add_argument('--random-sample', type=int, default=0, help='如>0则对 TopK 内随机抽样该数量以加速')
    p.add_argument('--seed', type=int, default=42, help='随机抽样种子')
    return p.parse_args()


def find_latest_backtest_file(root_dir: str, pattern: str) -> Path:
    root = Path(root_dir)
    candidates = sorted(root.glob(f'**/{pattern}'))
    if not candidates:
        raise FileNotFoundError(f'未找到匹配文件: {pattern} in {root_dir}')
    # 以修改时间排序（最新）
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_base_results(backtest_file: Path) -> pd.DataFrame:
    df = pd.read_csv(backtest_file)
    # 兼容列名 annual_ret / annual_return 差异
    if 'annual_ret' not in df.columns and 'annual_return' in df.columns:
        df = df.rename(columns={'annual_return': 'annual_ret'})
    return df


def select_topk(df: pd.DataFrame, topk: int, sample: int = 0, seed: int = 42) -> pd.DataFrame:
    df_sorted = df.sort_values('sharpe', ascending=False).head(topk).copy()
    if sample > 0 and sample < len(df_sorted):
        rng = np.random.default_rng(seed)
        idx = rng.choice(df_sorted.index, size=sample, replace=False)
        df_sorted = df_sorted.loc[idx].copy()
    return df_sorted.reset_index(drop=True)


def load_data_and_factors() -> tuple[dict, np.ndarray, np.ndarray, list[str], list[str]]:
    with open('etf_rotation_v2_breadth/configs/combo_wfo_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    loader = DataLoader(
        data_dir=config['data'].get('data_dir'),
        cache_dir=config['data'].get('cache_dir'),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config['data']['symbols'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        use_cache=True,
    )
    factor_lib = PreciseFactorLibrary()
    factors_df = factor_lib.compute_all_factors(prices=ohlcv)
    factors_dict = {name: factors_df[name] for name in factor_lib.list_factors()}
    processor = CrossSectionProcessor(
        lower_percentile=config['cross_section']['winsorize_lower'] * 100,
        upper_percentile=config['cross_section']['winsorize_upper'] * 100,
        verbose=False,
    )
    standardized_factors = processor.process_all_factors(factors_dict)
    factor_names = sorted(standardized_factors.keys())
    factor_arrays = [standardized_factors[name].values for name in factor_names]
    factors_data = np.stack(factor_arrays, axis=-1)
    returns_df = ohlcv['close'].pct_change(fill_method=None)
    returns = returns_df.values
    etf_names = list(ohlcv['close'].columns)
    dates = returns_df.index.strftime('%Y-%m-%d').tolist()
    print(f'✅ 数据加载完成: {dates[0]} ~ {dates[-1]} 共{len(dates)}日, ETF={len(etf_names)}, 因子={len(factor_names)}')
    return config, factors_data, returns, etf_names, factor_names


def run_single(combo_row: pd.Series, factors_data_full, returns_full, etf_names, factor_names, freq: int, pos_size: int):
    combo = combo_row['combo']
    factor_list = [f.strip() for f in str(combo).split('+') if f.strip()]
    missing = [f for f in factor_list if f not in factor_names]
    if missing:
        return None
    idxs = [factor_names.index(f) for f in factor_list]
    factors_selected = factors_data_full[:, :, idxs]
    try:
        res = backtest_no_lookahead(
            factors_data=factors_selected,
            returns=returns_full,
            etf_names=etf_names,
            rebalance_freq=freq,
            lookback_window=252,
            position_size=pos_size,
            commission_rate=0.00005,
            initial_capital=1_000_000.0,
            factors_data_full=factors_data_full,
            factor_indices_for_cache=idxs,
        )
        return {
            'combo': combo,
            'combo_size': combo_row.get('combo_size', len(factor_list)),
            'test_freq': freq,
            'test_position_size': pos_size,
            'annual_ret': res['annual_ret'],
            'sharpe': res['sharpe'],
            'max_dd': res['max_dd'],
            'win_rate': res['win_rate'],
            'avg_turnover': res['avg_turnover'],
            'avg_n_holdings': res['avg_n_holdings'],
            'calmar_ratio': res.get('calmar_ratio'),
            'sortino_ratio': res.get('sortino_ratio'),
        }
    except Exception as e:
        print(f'❌ 回测失败 combo={combo[:60]} freq={freq} pos={pos_size}: {e}')
        return None


def build_tasks(top_df: pd.DataFrame, freqs: list[int], pos_sizes: list[int]):
    tasks = []
    for _, row in top_df.iterrows():
        for f in freqs:
            for p in pos_sizes:
                tasks.append((row, f, p))
    return tasks


def main():
    args = parse_args()
    freqs = [int(x) for x in args.freqs.split(',') if x.strip()]
    pos_sizes = [int(x) for x in args.positions.split(',') if x.strip()]
    print(f'🔧 参数: TopK={args.topk} freqs={freqs} pos_sizes={pos_sizes} jobs={args.jobs}')

    # 找最新回测文件
    latest_file = find_latest_backtest_file(args.latest_backtest_dir, args.latest_backtest_pattern)
    print(f'📄 使用最新回测文件: {latest_file}')
    base_df = load_base_results(latest_file)

    if 'combo' not in base_df.columns or 'sharpe' not in base_df.columns:
        raise RuntimeError('回测结果缺少必要列: combo 或 sharpe')

    top_df = select_topk(base_df, args.topk, sample=args.random_sample, seed=args.seed)
    print(f'✅ 选取 {len(top_df)} 个组合用于扫描 (随机抽样={args.random_sample})')

    # 加载数据与因子
    config, factors_data_full, returns_full, etf_names, factor_names = load_data_and_factors()

    tasks = build_tasks(top_df, freqs, pos_sizes)
    print(f'📋 任务总数: {len(tasks)} (组合{len(top_df)} × 频率{len(freqs)} × 持仓{len(pos_sizes)})')

    # 并行执行
    def _runner(row, f, p):
        return run_single(row, factors_data_full, returns_full, etf_names, factor_names, f, p)

    results = Parallel(n_jobs=args.jobs, verbose=10)(delayed(_runner)(row, f, p) for row, f, p in tasks)
    valid = [r for r in results if r is not None]
    print(f'✅ 完成 {len(valid)}/{len(results)} 个任务')

    if not valid:
        print('❌ 无有效结果，退出')
        return

    df_res = pd.DataFrame(valid)

    # 输出目录
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / f'freq_pos_grid_{ts}.csv'
    df_res.to_csv(csv_path, index=False)
    print(f'💾 已保存结果: {csv_path}')

    # 汇总统计
    summary_freq = df_res.groupby('test_freq').agg({'sharpe':'mean','annual_ret':'mean','max_dd':'mean'}).round(4)
    summary_pos = df_res.groupby('test_position_size').agg({'sharpe':'mean','annual_ret':'mean','max_dd':'mean'}).round(4)
    summary_pair = df_res.groupby(['test_freq','test_position_size']).agg({'sharpe':'mean','annual_ret':'mean','max_dd':'mean'}).round(4).reset_index()

    best_row = summary_pair.sort_values('sharpe', ascending=False).iloc[0]
    best_freq = int(best_row['test_freq'])
    best_pos = int(best_row['test_position_size'])
    best_sharpe = best_row['sharpe']
    best_annual = best_row['annual_ret']

    # 生成报告
    md_path = out_root / f'GRID_FREQ_POS_REPORT_{ts}.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 频率 × 持仓数 网格扫描报告\n\n')
        f.write(f'- 时间戳: {ts}\n')
        f.write(f'- TopK: {args.topk}\n')
        f.write(f'- 频率集合: {freqs}\n')
        f.write(f'- 持仓数集合: {pos_sizes}\n')
        f.write(f'- 总任务数: {len(tasks)}\n')
        f.write(f'- 有效结果数: {len(valid)}\n')
        f.write(f'- 最优参数: freq={best_freq}, position_size={best_pos}, Sharpe={best_sharpe:.4f}, annual_ret={best_annual:.2%}\n\n')
        f.write('## 按频率汇总\n\n')
        f.write(summary_freq.to_markdown() + '\n\n')
        f.write('## 按持仓数汇总\n\n')
        f.write(summary_pos.to_markdown() + '\n\n')
        f.write('## 频率×持仓数组合汇总 (Sharpe/Annual/MaxDD 均值)\n\n')
        f.write(summary_pair.to_markdown(index=False) + '\n')
    print(f'📝 已生成报告: {md_path}')

    print('\n🎯 最优结果摘要:')
    print(f'   freq={best_freq} pos={best_pos} Sharpe={best_sharpe:.4f} annual={best_annual:.2%}')

if __name__ == '__main__':
    main()
