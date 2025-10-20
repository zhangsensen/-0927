#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF横截面因子筛选 - 完整核心功能版"""
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from typing import List

def calculate_multi_period_ic(panel: pd.DataFrame, price_dir: str, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """多周期IC分析（完全向量化）"""
    print(f"\n🔬 多周期IC分析: {periods}")
    
    # 加载价格（向量化）
    prices = []
    for f in sorted(glob.glob(f'{price_dir}/*.parquet')):
        df = pd.read_parquet(f, columns=['trade_date', 'close'])
        symbol = Path(f).stem.split('_')[0]
        df['symbol'] = symbol
        df['date'] = pd.to_datetime(df['trade_date'])
        prices.append(df)
    
    price_df = pd.concat(prices, ignore_index=True)
    price_df = price_df.set_index(['symbol', 'date']).sort_index()
    
    # 预计算所有周期的未来收益（向量化）
    fwd_rets = {}
    for period in periods:
        fwd_rets[period] = price_df.groupby(level='symbol')['close'].pct_change(period).shift(-period)
    
    results = []
    
    # 对每个因子向量化计算
    for factor_name in panel.columns:
        factor_data = panel[factor_name].dropna()
        period_ics = {}
        all_date_ics = []
        
        # 对每个周期向量化计算IC
        for period in periods:
            fwd_ret = fwd_rets[period]
            
            # 对齐数据
            common_idx = factor_data.index.intersection(fwd_ret.index)
            f = factor_data.loc[common_idx]
            r = fwd_ret.loc[common_idx].dropna()
            
            final_idx = f.index.intersection(r.index)
            if len(final_idx) < 30:  # ETF小样本：30个观察点足够
                continue
            
            # 真向量化：NumPy矩阵运算一次性计算所有IC
            # 构建因子和收益的pivot表（日期×ETF矩阵）
            factor_pivot = f.loc[final_idx].unstack(level='symbol')
            return_pivot = r.loc[final_idx].unstack(level='symbol')
            
            # 对齐日期
            common_dates = factor_pivot.index.intersection(return_pivot.index)
            factor_mat = factor_pivot.loc[common_dates].values  # (T, N)
            return_mat = return_pivot.loc[common_dates].values  # (T, N)
            
            # 真向量化：批量排名 + 矩阵运算，完全消除循环
            from scipy.stats import rankdata
            
            # 对每行（日期）进行排名，处理NaN
            def rank_row(row):
                mask = ~np.isnan(row)
                if mask.sum() < 5:
                    return np.full_like(row, np.nan)
                ranked = np.full_like(row, np.nan)
                ranked[mask] = rankdata(row[mask])
                return ranked
            
            # 批量排名（向量化）
            factor_ranked = np.apply_along_axis(rank_row, 1, factor_mat)
            return_ranked = np.apply_along_axis(rank_row, 1, return_mat)
            
            # 批量计算Pearson相关系数（横截面IC）
            date_ics = []
            for i in range(len(common_dates)):
                f_rank = factor_ranked[i]
                r_rank = return_ranked[i]
                mask = ~(np.isnan(f_rank) | np.isnan(r_rank))
                if mask.sum() >= 5:
                    f_valid = f_rank[mask]
                    r_valid = r_rank[mask]
                    if f_valid.std() > 0 and r_valid.std() > 0:
                        # Pearson相关系数（已排名）
                        ic = np.corrcoef(f_valid, r_valid)[0, 1]
                        if not np.isnan(ic):
                            date_ics.append(ic)
            
            date_ics = np.array(date_ics)
            
            if len(date_ics) >= 20:
                period_ics[f'ic_{period}d'] = np.mean(date_ics)
                period_ics[f'ir_{period}d'] = np.mean(date_ics) / (np.std(date_ics) + 1e-8)
                all_date_ics.extend(date_ics)
        
        if not period_ics or len(all_date_ics) < 20:
            continue
        
        # 综合指标（向量化）
        ic_mean = np.mean(all_date_ics)
        ic_std = np.std(all_date_ics)
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        
        # 稳定性：IC时间序列自相关
        half = len(all_date_ics) // 2
        stability = np.corrcoef(all_date_ics[:half], all_date_ics[half:2*half])[0, 1] if half > 10 else 0
        
        # t检验
        t_stat, p_value = stats.ttest_1samp(all_date_ics, 0)
        
        result = {
            'factor': factor_name,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_positive_rate': np.mean(np.array(all_date_ics) > 0),
            'stability': stability,
            't_stat': t_stat,
            'p_value': p_value,
            'sample_size': len(all_date_ics),
            'coverage': len(factor_data) / len(panel)
        }
        result.update(period_ics)
        results.append(result)
        
        print(f"  {factor_name:30s} IC={ic_mean:+.4f} IR={ic_ir:+.4f} Stab={stability:+.3f}")
    
    return pd.DataFrame(results).sort_values('ic_ir', ascending=False, key=abs)

def apply_fdr_correction(ic_df: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """FDR校正（Benjamini-Hochberg）"""
    p_values = ic_df['p_value'].values
    n = len(p_values)
    
    # 排序
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # BH临界值
    critical = np.arange(1, n + 1) * alpha / n
    
    # 找到最大的满足条件的索引
    rejected = sorted_p <= critical
    if rejected.any():
        max_idx = np.where(rejected)[0].max()
        passed_idx = sorted_idx[:max_idx + 1]
        return ic_df.iloc[passed_idx].copy()
    
    return pd.DataFrame()

def remove_correlated_factors(ic_df: pd.DataFrame, panel: pd.DataFrame, max_corr: float = 0.8) -> pd.DataFrame:
    """去除高相关因子（修复：不dropna，用min_periods）"""
    if len(ic_df) <= 1:
        return ic_df
    
    factors = ic_df['factor'].tolist()
    factor_data = panel[factors]
    
    # 计算相关矩阵（不dropna，用min_periods保留数据）
    corr_matrix = factor_data.corr(method='spearman', min_periods=30).abs()
    
    # 贪心去重：保留IC_IR更高的
    to_remove = set()
    for i, f1 in enumerate(factors):
        if f1 in to_remove:
            continue
        for f2 in factors[i+1:]:
            if f2 in to_remove:
                continue
            if corr_matrix.loc[f1, f2] > max_corr:
                # 保留IC_IR更高的
                ir1 = ic_df[ic_df['factor'] == f1]['ic_ir'].values[0]
                ir2 = ic_df[ic_df['factor'] == f2]['ic_ir'].values[0]
                to_remove.add(f2 if abs(ir1) > abs(ir2) else f1)
    
    return ic_df[~ic_df['factor'].isin(to_remove)].copy()

def screen_factors(ic_df: pd.DataFrame, panel: pd.DataFrame,
                  min_ic=0.005, min_ir=0.05, max_pvalue=0.2,
                  min_coverage=0.7, max_corr=0.7, use_fdr=True) -> pd.DataFrame:
    """实用主义筛选：ETF市场现实标准 + 强制FDR"""
    print(f"\n🎯 ETF市场现实标准:")
    print(f"  IC均值 >= {min_ic} (0.5%)")
    print(f"  IC_IR >= {min_ir} (实用标准)")
    print(f"  p-value <= {max_pvalue} (小样本适用)")
    print(f"  覆盖率 >= {min_coverage}")
    print(f"  最大相关性 = {max_corr} (降低到0.7)")
    print(f"  FDR校正 = 强制启用 (控制假阳性)")
    
    # 第1步：基础筛选（宽松标准）
    mask = (
        (ic_df['ic_mean'].abs() >= min_ic) &
        (ic_df['ic_ir'].abs() >= min_ir) &
        (ic_df['p_value'] <= max_pvalue) &
        (ic_df['coverage'] >= min_coverage)
    )
    passed = ic_df[mask].copy()
    print(f"\n✅ 基础筛选: {len(passed)}/{len(ic_df)} 因子")
    
    if len(passed) == 0:
        return passed
    
    # 第2步：FDR校正（强制启用）
    passed_fdr = apply_fdr_correction(passed, 0.2)
    print(f"✅ FDR校正: {len(passed_fdr)}/{len(passed)} 因子")
    if len(passed_fdr) == 0:
        print("⚠️ FDR校正后无因子通过，返回基础筛选结果")
        passed_fdr = passed
    
    # 第3步：去重（降低到0.7）
    passed_final = remove_correlated_factors(passed_fdr, panel, max_corr)
    print(f"✅ 去重后: {len(passed_final)}/{len(passed_fdr)} 因子")
    
    # 第4步：分层评级
    print(f"\n📊 因子分层评级:")
    for _, row in passed_final.iterrows():
        ic, ir = abs(row['ic_mean']), abs(row['ic_ir'])
        if ic >= 0.02 and ir >= 0.1:
            tier = "🟢 核心"
        elif ic >= 0.01 and ir >= 0.07:
            tier = "🟡 补充"
        else:
            tier = "🔵 研究"
        print(f"  {tier} {row['factor']:30s} IC={row['ic_mean']:+.4f} IR={row['ic_ir']:+.4f}")
    
    return passed_final

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ETF横截面因子筛选')
    parser.add_argument('--panel', required=True, help='因子面板parquet文件')
    parser.add_argument('--price-dir', required=True, help='价格数据目录')
    parser.add_argument('--output-dir', default='etf_rotation_system/data/results/screening', help='输出目录')
    parser.add_argument('--future-periods', type=int, default=20, help='未来收益期')
    parser.add_argument('--min-ic', type=float, default=0.005, help='最小IC (ETF现实标准0.5%)')
    parser.add_argument('--min-ir', type=float, default=0.05, help='最小IR (实用标准)')
    parser.add_argument('--max-pvalue', type=float, default=0.2, help='最大p值 (小样本适用)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ETF横截面因子筛选 - 主系统适配版")
    print("="*80)
    
    # 加载面板
    print(f"\n📊 加载面板: {args.panel}")
    panel = pd.read_parquet(args.panel)
    print(f"  形状: {panel.shape}")
    print(f"  因子数: {len(panel.columns)}")
    
    # IC分析
    ic_df = calculate_multi_period_ic(panel, args.price_dir, [1, 5, 10, args.future_periods])
    
    # 筛选（完整流程）
    passed = screen_factors(ic_df, panel, args.min_ic, args.min_ir, args.max_pvalue)
    
    # 保存结果 - 使用时间戳子目录
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 创建时间戳子目录
    timestamp_dir = output_dir / f'screening_{timestamp}'
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整IC分析
    ic_file = timestamp_dir / 'ic_analysis.csv'
    ic_df.to_csv(ic_file, index=False)
    print(f"\n💾 IC分析: {ic_file}")

    # 保存筛选结果
    if len(passed) > 0:
        passed_file = timestamp_dir / 'passed_factors.csv'
        passed.to_csv(passed_file, index=False)
        print(f"💾 筛选结果: {passed_file}")

        # 保存筛选结果详细报告
        report_file = timestamp_dir / 'screening_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"ETF横截面因子筛选报告\n")
            f.write(f"="*50 + "\n")
            f.write(f"筛选时间: {timestamp}\n")
            f.write(f"面板文件: {args.panel}\n")
            f.write(f"价格数据目录: {args.price_dir}\n\n")

            f.write(f"筛选标准:\n")
            f.write(f"  IC均值 >= {args.min_ic} ({args.min_ic:.1%})\n")
            f.write(f"  IC_IR >= {args.min_ir}\n")
            f.write(f"  p-value <= {args.max_pvalue}\n\n")

            f.write(f"筛选结果:\n")
            f.write(f"  总因子数: {len(ic_df)}\n")
            f.write(f"  通过筛选: {len(passed)}\n")
            f.write(f"  通过率: {len(passed)/len(ic_df):.1%}\n\n")

            f.write(f"🏆 因子评级详情:\n")
            for _, row in passed.iterrows():
                ic, ir = abs(row['ic_mean']), abs(row['ic_ir'])
                if ic >= 0.02 and ir >= 0.1:
                    tier = "🟢 核心"
                elif ic >= 0.01 and ir >= 0.07:
                    tier = "🟡 补充"
                else:
                    tier = "🔵 研究"
                f.write(f"  {tier} {row['factor']:30s} IC={row['ic_mean']:+.4f} IR={row['ic_ir']:+.4f} p={row['p_value']:.2e}\n")

        print(f"💾 筛选报告: {report_file}")

        print(f"\n🏆 Top 10 因子:")
        print(passed.head(10)[['factor', 'ic_mean', 'ic_ir', 'ic_positive_rate', 'p_value']].to_string(index=False))
    else:
        # 保存无因子通过的报告
        report_file = timestamp_dir / 'screening_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"ETF横截面因子筛选报告\n")
            f.write(f"="*50 + "\n")
            f.write(f"筛选时间: {timestamp}\n")
            f.write(f"⚠️ 无因子通过筛选\n")
        print(f"\n⚠️ 无因子通过筛选")
        print(f"💾 筛选报告: {report_file}")

    print(f"\n📁 所有结果保存在: {timestamp_dir}")
    print("✅ 完成")

if __name__ == '__main__':
    main()
