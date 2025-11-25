#!/usr/bin/env python3
"""
验证WFO排序的预测能力

核心问题：
1. WFO排序能否预测真实回测收益？(Rank Correlation)
2. Calibrator相比IC baseline提升了多少？
3. 排序是否有经济价值？(Top-K Precision, Decile Analysis)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class RankingValidator:
    def __init__(self, run_dir: Path, backtest_dir: Path):
        self.run_dir = run_dir
        self.backtest_dir = backtest_dir
        self.data = None
        
    def load_data(self):
        """加载WFO排序和真实回测数据"""
        print("📂 加载数据...")
        
        # 1. 加载all_combos（combo_id → combo_str映射）
        all_combos = pd.read_parquet(self.run_dir / "all_combos.parquet")
        print(f"   - all_combos: {len(all_combos)} 组合")
        
        # 2. 加载IC排序
        ranking_ic = pd.read_parquet(self.run_dir / "ranking_blends" / "ranking_baseline.parquet")
        ranking_ic['ic_rank'] = range(1, len(ranking_ic) + 1)
        ranking_ic['combo_str'] = ranking_ic['combo'].astype(str)
        print(f"   - IC排序: {len(ranking_ic)} 组合")
        
        # 3. 加载Calibrator排序
        ranking_cal = pd.read_parquet(self.run_dir / "ranking_blends" / "ranking_lightgbm.parquet")
        ranking_cal['cal_rank'] = range(1, len(ranking_cal) + 1)
        ranking_cal['combo_str'] = ranking_cal['combo'].astype(str)
        print(f"   - Calibrator排序: {len(ranking_cal)} 组合")
        
        # 4. 加载真实回测结果（查找IC baseline的Top1000回测）
        # 真实回测结果在子目录中，格式：{run_ts}_{backtest_ts}/top{K}_profit_backtest_*.csv
        backtest_subdirs = sorted([d for d in self.backtest_dir.iterdir() if d.is_dir()], reverse=True)
        
        backtest_csv = None
        for subdir in backtest_subdirs:
            # 查找包含run_ts的子目录
            if self.run_dir.name.replace('run_', '') in subdir.name:
                csv_files = list(subdir.glob("top*_profit_backtest_*.csv"))
                if csv_files:
                    # 检查run_tag，找IC baseline（ranking_baseline.parquet）
                    for csv in csv_files:
                        df_sample = pd.read_csv(csv, nrows=1)
                        if 'ranking_baseline.parquet' in df_sample['run_tag'].iloc[0]:
                            if 'top1000' in csv.name or 'top3000' in csv.name:
                                backtest_csv = csv
                                break
                    if backtest_csv:
                        break
        
        if not backtest_csv:
            raise FileNotFoundError(f"未找到IC baseline的真实回测CSV文件，请先运行回测")
        
        backtest = pd.read_csv(backtest_csv)
        print(f"   - 真实回测: {len(backtest)} 组合 (from {backtest_csv.name})")
        
        # 5. 合并数据
        # 所有数据都使用combo字符串作为关联键
        
        # 首先，为all_combos添加combo_id（行索引）并标准化combo_str
        all_combos = all_combos.reset_index().rename(columns={'index': 'combo_id'})
        all_combos['combo_str'] = all_combos['combo'].astype(str)
        backtest['combo_str'] = backtest['combo'].astype(str)
        
        # 关联all_combos和ranking（使用combo_str）
        data = all_combos[['combo_id', 'combo_str', 'mean_oos_ic']].copy()
        data = data.merge(
            ranking_ic[['combo_str', 'ic_rank']], 
            on='combo_str', 
            how='left'
        )
        data = data.merge(
            ranking_cal[['combo_str', 'cal_rank']], 
            on='combo_str', 
            how='left'
        )
        
        # 关联真实回测结果
        data = data.merge(
            backtest[['combo_str', 'sharpe_net', 'annual_ret_net', 'max_dd_net', 'win_rate']],
            on='combo_str',
            how='inner'
        )
        
        # 计算真实回测排名（按Sharpe）
        data['backtest_rank'] = data['sharpe_net'].rank(ascending=False, method='min').astype(int)
        
        # 只保留有完整排序信息的组合
        data = data.dropna(subset=['ic_rank', 'cal_rank', 'backtest_rank'])
        
        print(f"   - 合并后: {len(data)} 组合")
        print(f"   - IC排序范围: {data['ic_rank'].min():.0f} - {data['ic_rank'].max():.0f}")
        print(f"   - Calibrator排序范围: {data['cal_rank'].min():.0f} - {data['cal_rank'].max():.0f}")
        
        self.data = data
        return data
    
    def compute_rank_correlation(self):
        """计算排序相关性"""
        print("\n📊 计算排序相关性...")
        
        # IC排序 vs 真实回测排序
        ic_spearman, ic_p = spearmanr(self.data['ic_rank'], self.data['backtest_rank'])
        ic_kendall, ic_kp = kendalltau(self.data['ic_rank'], self.data['backtest_rank'])
        
        # Calibrator排序 vs 真实回测排序
        cal_spearman, cal_p = spearmanr(self.data['cal_rank'], self.data['backtest_rank'])
        cal_kendall, cal_kp = kendalltau(self.data['cal_rank'], self.data['backtest_rank'])
        
        results = {
            'IC_Baseline': {
                'spearman': ic_spearman,
                'spearman_p': ic_p,
                'kendall': ic_kendall,
                'kendall_p': ic_kp
            },
            'Calibrator': {
                'spearman': cal_spearman,
                'spearman_p': cal_p,
                'kendall': cal_kendall,
                'kendall_p': cal_kp
            }
        }
        
        print(f"\n{'排序方法':<15} {'Spearman':<10} {'p值':<10} {'Kendall':<10} {'p值':<10}")
        print("=" * 55)
        print(f"{'IC Baseline':<15} {ic_spearman:>9.4f} {ic_p:>9.4e} {ic_kendall:>9.4f} {ic_kp:>9.4e}")
        print(f"{'Calibrator':<15} {cal_spearman:>9.4f} {cal_p:>9.4e} {cal_kendall:>9.4f} {cal_kp:>9.4e}")
        
        # 判断相关性强度
        def judge_correlation(r):
            if abs(r) > 0.7: return "Excellent"
            if abs(r) > 0.5: return "Good"
            if abs(r) > 0.3: return "Moderate"
            return "Poor"
        
        print(f"\n评价：")
        print(f"  IC Baseline: {judge_correlation(ic_spearman)}")
        print(f"  Calibrator: {judge_correlation(cal_spearman)}")
        
        return results
    
    def compute_topk_precision(self, k_values=[10, 20, 50, 100]):
        """计算Top-K精度"""
        print(f"\n🎯 计算Top-K Precision...")
        
        results = {}
        
        for k in k_values:
            # WFO Top-K vs 真实回测 Top-K的重叠
            ic_topk = set(self.data.nsmallest(k, 'ic_rank')['combo_id'])
            cal_topk = set(self.data.nsmallest(k, 'cal_rank')['combo_id'])
            backtest_topk = set(self.data.nsmallest(k, 'backtest_rank')['combo_id'])
            
            ic_precision = len(ic_topk & backtest_topk) / k
            cal_precision = len(cal_topk & backtest_topk) / k
            
            ic_recall = len(ic_topk & backtest_topk) / k
            cal_recall = len(cal_topk & backtest_topk) / k
            
            results[f'Top{k}'] = {
                'IC_precision': ic_precision,
                'IC_recall': ic_recall,
                'Calibrator_precision': cal_precision,
                'Calibrator_recall': cal_recall
            }
            
            print(f"\nTop-{k}:")
            print(f"  IC Baseline: Precision={ic_precision:.2%}, Recall={ic_recall:.2%}")
            print(f"  Calibrator:  Precision={cal_precision:.2%}, Recall={cal_recall:.2%}")
        
        return results
    
    def decile_analysis(self, n_deciles=10):
        """Decile性能分析"""
        print(f"\n📉 Decile分析 (分成{n_deciles}组)...")
        
        # 按IC排序分组
        self.data['ic_decile'] = pd.qcut(self.data['ic_rank'], n_deciles, labels=False, duplicates='drop') + 1
        # 按Calibrator排序分组
        self.data['cal_decile'] = pd.qcut(self.data['cal_rank'], n_deciles, labels=False, duplicates='drop') + 1
        
        # 计算每个decile的平均Sharpe
        ic_decile_perf = self.data.groupby('ic_decile')['sharpe_net'].agg(['mean', 'median', 'count'])
        cal_decile_perf = self.data.groupby('cal_decile')['sharpe_net'].agg(['mean', 'median', 'count'])
        
        print(f"\nIC Baseline - Decile平均Sharpe:")
        print(ic_decile_perf)
        
        print(f"\nCalibrator - Decile平均Sharpe:")
        print(cal_decile_perf)
        
        # 检查单调性
        ic_monotonic = all(ic_decile_perf['mean'].iloc[i] >= ic_decile_perf['mean'].iloc[i+1] 
                          for i in range(len(ic_decile_perf)-1))
        cal_monotonic = all(cal_decile_perf['mean'].iloc[i] >= cal_decile_perf['mean'].iloc[i+1] 
                           for i in range(len(cal_decile_perf)-1))
        
        print(f"\n单调性检验:")
        print(f"  IC Baseline: {'✓ 单调递减' if ic_monotonic else '✗ 非单调'}")
        print(f"  Calibrator: {'✓ 单调递减' if cal_monotonic else '✗ 非单调'}")
        
        return {
            'IC_decile': ic_decile_perf.to_dict(),
            'Calibrator_decile': cal_decile_perf.to_dict(),
            'IC_monotonic': ic_monotonic,
            'Calibrator_monotonic': cal_monotonic
        }
    
    def statistical_tests(self):
        """统计显著性检验"""
        print("\n📈 统计显著性检验...")
        
        # 提取Top100的Sharpe
        ic_top100 = self.data.nsmallest(100, 'ic_rank')['sharpe_net']
        cal_top100 = self.data.nsmallest(100, 'cal_rank')['sharpe_net']
        
        # Mann-Whitney U检验（非参数检验）
        u_stat, p_value = stats.mannwhitneyu(cal_top100, ic_top100, alternative='greater')
        
        # T检验（参数检验）
        t_stat, t_p = stats.ttest_ind(cal_top100, ic_top100)
        
        print(f"\nTop100 Sharpe对比:")
        print(f"  IC Baseline: 均值={ic_top100.mean():.4f}, 中位数={ic_top100.median():.4f}")
        print(f"  Calibrator:  均值={cal_top100.mean():.4f}, 中位数={cal_top100.median():.4f}")
        print(f"  提升: {(cal_top100.mean() / ic_top100.mean() - 1) * 100:+.2f}%")
        print(f"\nMann-Whitney U检验: U={u_stat:.1f}, p={p_value:.4e}")
        print(f"  结论: {'✓ Calibrator显著优于IC (p<0.05)' if p_value < 0.05 else '✗ 无显著差异'}")
        
        return {
            'IC_mean': ic_top100.mean(),
            'IC_median': ic_top100.median(),
            'Calibrator_mean': cal_top100.mean(),
            'Calibrator_median': cal_top100.median(),
            'improvement_pct': (cal_top100.mean() / ic_top100.mean() - 1) * 100,
            'mann_whitney_u': u_stat,
            'mann_whitney_p': p_value,
            't_stat': t_stat,
            't_p': t_p
        }
    
    def visualize(self, output_dir: Path):
        """生成可视化图表"""
        print("\n🎨 生成可视化图表...")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 图1: Rank Correlation Scatter
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # IC Baseline
        axes[0].scatter(self.data['ic_rank'], self.data['sharpe_net'], alpha=0.3, s=10)
        axes[0].set_xlabel('WFO IC排名', fontsize=12)
        axes[0].set_ylabel('真实回测 Sharpe', fontsize=12)
        axes[0].set_title('IC Baseline: WFO排序 vs 真实表现', fontsize=14)
        rho, _ = spearmanr(self.data['ic_rank'], self.data['sharpe_net'])
        axes[0].text(0.05, 0.95, f'Spearman ρ = {rho:.3f}', 
                    transform=axes[0].transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Calibrator
        axes[1].scatter(self.data['cal_rank'], self.data['sharpe_net'], alpha=0.3, s=10, color='orange')
        axes[1].set_xlabel('WFO Calibrator排名', fontsize=12)
        axes[1].set_ylabel('真实回测 Sharpe', fontsize=12)
        axes[1].set_title('Calibrator: WFO排序 vs 真实表现', fontsize=14)
        rho, _ = spearmanr(self.data['cal_rank'], self.data['sharpe_net'])
        axes[1].text(0.05, 0.95, f'Spearman ρ = {rho:.3f}', 
                    transform=axes[1].transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rank_correlation_scatter.png', dpi=300, bbox_inches='tight')
        print(f"   - 保存: rank_correlation_scatter.png")
        
        # 图2: Decile Performance
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ic_decile_perf = self.data.groupby('ic_decile')['sharpe_net'].mean()
        cal_decile_perf = self.data.groupby('cal_decile')['sharpe_net'].mean()
        
        x = np.arange(len(ic_decile_perf))
        width = 0.35
        
        ax.bar(x - width/2, ic_decile_perf.values, width, label='IC Baseline', alpha=0.8)
        ax.bar(x + width/2, cal_decile_perf.values, width, label='Calibrator', alpha=0.8, color='orange')
        
        ax.set_xlabel('Decile (1=最优, 10=最差)', fontsize=12)
        ax.set_ylabel('平均Sharpe', fontsize=12)
        ax.set_title('Decile性能对比：排名越高是否真的越好？', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'D{i+1}' for i in range(len(ic_decile_perf))])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'decile_performance.png', dpi=300, bbox_inches='tight')
        print(f"   - 保存: decile_performance.png")
        
        # 图3: Cumulative Performance Curve
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 按排名排序后计算累计平均
        ic_sorted = self.data.sort_values('ic_rank')
        cal_sorted = self.data.sort_values('cal_rank')
        
        topk_range = range(10, len(self.data), 10)
        ic_cumulative = [ic_sorted.head(k)['sharpe_net'].mean() for k in topk_range]
        cal_cumulative = [cal_sorted.head(k)['sharpe_net'].mean() for k in topk_range]
        
        ax.plot(topk_range, ic_cumulative, label='IC Baseline', linewidth=2)
        ax.plot(topk_range, cal_cumulative, label='Calibrator', linewidth=2, color='orange')
        
        ax.set_xlabel('Top-K组合数量', fontsize=12)
        ax.set_ylabel('平均Sharpe', fontsize=12)
        ax.set_title('Top-K累计平均性能：选择更多组合如何影响质量', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cumulative_performance.png', dpi=300, bbox_inches='tight')
        print(f"   - 保存: cumulative_performance.png")
        
        # 图4: Correlation Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_data = self.data[['ic_rank', 'cal_rank', 'backtest_rank', 
                               'mean_oos_ic', 'sharpe_net', 'annual_ret_net']].corr()
        
        sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('排序指标相关性矩阵', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"   - 保存: correlation_heatmap.png")
        
        plt.close('all')
    
    def generate_report(self, metrics: dict, output_file: Path):
        """生成Markdown报告"""
        print("\n📝 生成验证报告...")
        
        report = f"""# WFO排序预测能力验证报告

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 执行摘要

本报告验证了WFO排序（IC Baseline和Calibrator）对真实回测收益的**预测能力**。

### 核心发现

1. **排序一致性**：
   - IC Baseline Spearman相关性: **{metrics['rank_correlation']['IC_Baseline']['spearman']:.3f}**
   - Calibrator Spearman相关性: **{metrics['rank_correlation']['Calibrator']['spearman']:.3f}**
   
2. **Top-10精度**：
   - IC Baseline: {metrics['topk_precision']['Top10']['IC_precision']:.1%}
   - Calibrator: {metrics['topk_precision']['Top10']['Calibrator_precision']:.1%}

3. **性能提升**：
   - Calibrator vs IC (Top100平均): **{metrics['statistical_tests']['improvement_pct']:+.2f}%**
   - 统计显著性: p = {metrics['statistical_tests']['mann_whitney_p']:.4e}

---

## 1. 排序相关性分析

### 1.1 Spearman秩相关系数

| 排序方法 | Spearman ρ | p值 | Kendall τ | p值 | 评价 |
|---------|-----------|-----|-----------|-----|------|
| IC Baseline | {metrics['rank_correlation']['IC_Baseline']['spearman']:.4f} | {metrics['rank_correlation']['IC_Baseline']['spearman_p']:.4e} | {metrics['rank_correlation']['IC_Baseline']['kendall']:.4f} | {metrics['rank_correlation']['IC_Baseline']['kendall_p']:.4e} | {self._judge_correlation(metrics['rank_correlation']['IC_Baseline']['spearman'])} |
| Calibrator | {metrics['rank_correlation']['Calibrator']['spearman']:.4f} | {metrics['rank_correlation']['Calibrator']['spearman_p']:.4e} | {metrics['rank_correlation']['Calibrator']['kendall']:.4f} | {metrics['rank_correlation']['Calibrator']['kendall_p']:.4e} | {self._judge_correlation(metrics['rank_correlation']['Calibrator']['spearman'])} |

**解读**：
- Spearman相关系数衡量WFO排序和真实回测排序的一致性
- ρ > 0.7: Excellent, 0.5-0.7: Good, 0.3-0.5: Moderate, <0.3: Poor
- p值 < 0.05表示统计显著

### 1.2 可视化

参见图表：
- `rank_correlation_scatter.png`: WFO排序 vs 真实Sharpe散点图
- `correlation_heatmap.png`: 所有指标的相关性矩阵

---

## 2. Top-K精度分析

### 2.1 Precision@K（预测的Top-K有多少真的在真实Top-K中）

| Top-K | IC Baseline | Calibrator | 提升 |
|-------|------------|-----------|------|
| Top-10 | {metrics['topk_precision']['Top10']['IC_precision']:.1%} | {metrics['topk_precision']['Top10']['Calibrator_precision']:.1%} | {(metrics['topk_precision']['Top10']['Calibrator_precision'] - metrics['topk_precision']['Top10']['IC_precision']) * 100:+.1f}pp |
| Top-20 | {metrics['topk_precision']['Top20']['IC_precision']:.1%} | {metrics['topk_precision']['Top20']['Calibrator_precision']:.1%} | {(metrics['topk_precision']['Top20']['Calibrator_precision'] - metrics['topk_precision']['Top20']['IC_precision']) * 100:+.1f}pp |
| Top-50 | {metrics['topk_precision']['Top50']['IC_precision']:.1%} | {metrics['topk_precision']['Top50']['Calibrator_precision']:.1%} | {(metrics['topk_precision']['Top50']['Calibrator_precision'] - metrics['topk_precision']['Top50']['IC_precision']) * 100:+.1f}pp |
| Top-100 | {metrics['topk_precision']['Top100']['IC_precision']:.1%} | {metrics['topk_precision']['Top100']['Calibrator_precision']:.1%} | {(metrics['topk_precision']['Top100']['Calibrator_precision'] - metrics['topk_precision']['Top100']['IC_precision']) * 100:+.1f}pp |

**解读**：
- Precision > 70%: Excellent, 50-70%: Good, 30-50%: Moderate, <30%: Poor
- 这是排序系统最直接的价值体现

---

## 3. Decile性能分析

### 3.1 单调性检验

- IC Baseline: {'✓ 单调递减' if metrics['decile_analysis']['IC_monotonic'] else '✗ 非单调'}
- Calibrator: {'✓ 单调递减' if metrics['decile_analysis']['Calibrator_monotonic'] else '✗ 非单调'}

### 3.2 可视化

参见图表：
- `decile_performance.png`: 各Decile的平均Sharpe对比
- `cumulative_performance.png`: Top-K累计平均性能曲线

**解读**：
- 单调递减表示排序具有全局有效性（不只是Top几个好）
- Decile 1（最优）应该显著高于Decile 10（最差）

---

## 4. 统计显著性检验

### 4.1 Top100性能对比

| 指标 | IC Baseline | Calibrator | 提升 |
|------|------------|-----------|------|
| 平均Sharpe | {metrics['statistical_tests']['IC_mean']:.4f} | {metrics['statistical_tests']['Calibrator_mean']:.4f} | {metrics['statistical_tests']['improvement_pct']:+.2f}% |
| 中位数Sharpe | {metrics['statistical_tests']['IC_median']:.4f} | {metrics['statistical_tests']['Calibrator_median']:.4f} | {(metrics['statistical_tests']['Calibrator_median'] / metrics['statistical_tests']['IC_median'] - 1) * 100:+.2f}% |

### 4.2 Mann-Whitney U检验

- U统计量: {metrics['statistical_tests']['mann_whitney_u']:.1f}
- p值: {metrics['statistical_tests']['mann_whitney_p']:.4e}
- 结论: **{'Calibrator显著优于IC Baseline (p<0.05)' if metrics['statistical_tests']['mann_whitney_p'] < 0.05 else '无显著差异'}**

---

## 5. 最终结论

### 5.1 排序是否有价值？

基于Spearman相关性：
- IC Baseline: {self._judge_correlation(metrics['rank_correlation']['IC_Baseline']['spearman'])}
- Calibrator: {self._judge_correlation(metrics['rank_correlation']['Calibrator']['spearman'])}

**结论**: {'✅ 排序具有预测价值' if metrics['rank_correlation']['Calibrator']['spearman'] > 0.3 else '❌ 排序预测能力不足，存在过拟合风险'}

### 5.2 对比基准提升了多少？

- Top100平均Sharpe提升: **{metrics['statistical_tests']['improvement_pct']:+.2f}%**
- 统计显著性: **{'显著' if metrics['statistical_tests']['mann_whitney_p'] < 0.05 else '不显著'}** (p={metrics['statistical_tests']['mann_whitney_p']:.4e})

### 5.3 经济价值评估

假设：
- 基准策略（IC Top10）：Sharpe = {metrics['statistical_tests']['IC_mean']:.3f}
- Calibrator策略（Cal Top10）：Sharpe = {metrics['statistical_tests']['Calibrator_mean']:.3f}
- 年化收益提升：约{metrics['statistical_tests']['improvement_pct']:.1f}%

**经济价值**: {'✅ 值得部署' if metrics['statistical_tests']['improvement_pct'] > 10 else '⚠️ 提升有限，需谨慎评估'}

---

## 6. 建议

"""
        
        # 根据结果给出建议
        if metrics['rank_correlation']['Calibrator']['spearman'] > 0.5:
            report += """
### ✅ 强烈推荐：Calibrator排序系统

1. **立即部署**：Calibrator具有良好的预测能力和显著的性能提升
2. **建议配置**：选择Calibrator Top10-50作为实盘组合池
3. **监控指标**：
   - 实盘Sharpe vs 回测Sharpe的偏差
   - Top组合的真实收益排名
   - 定期（每季度）重新训练Calibrator
"""
        elif metrics['rank_correlation']['Calibrator']['spearman'] > 0.3:
            report += """
### ⚠️ 谨慎使用：Calibrator排序系统

1. **小规模试点**：先用少量资金验证
2. **加强监控**：密切跟踪实盘vs回测的偏差
3. **优化方向**：
   - 增加更多稳定性特征
   - 使用Ensemble方法（IC + Calibrator加权）
   - 缩短WFO训练窗口，提高时效性
"""
        else:
            report += """
### ❌ 不建议部署：排序系统预测能力不足

1. **根本问题**：WFO排序和真实回测相关性低，存在严重过拟合
2. **改进方向**：
   - 重新审视WFO窗口设置（可能窗口太长或太短）
   - 增加样本外验证（使用完全独立的时间段）
   - 简化模型，减少过拟合
   - 考虑使用更稳健的排序指标
"""
        
        report += """
---

## 附录：数据统计

- WFO Run: {run_dir}
- 样本数量: {sample_size}
- 排序范围: {rank_range}
- 分析日期: {analysis_date}
""".format(
            run_dir=self.run_dir.name,
            sample_size=len(self.data),
            rank_range=f"1 - {len(self.data)}",
            analysis_date=pd.Timestamp.now().strftime('%Y-%m-%d')
        )
        
        output_file.write_text(report, encoding='utf-8')
        print(f"   - 保存报告: {output_file}")
    
    def _judge_correlation(self, r):
        """判断相关性强度"""
        if abs(r) > 0.7: return "Excellent (优秀)"
        if abs(r) > 0.5: return "Good (良好)"
        if abs(r) > 0.3: return "Moderate (中等)"
        return "Poor (较差)"
    
    def run_full_validation(self):
        """运行完整验证流程"""
        print("="*60)
        print("🚀 WFO排序预测能力验证")
        print("="*60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 计算各项指标
        rank_corr = self.compute_rank_correlation()
        topk_prec = self.compute_topk_precision()
        decile_anal = self.decile_analysis()
        stat_tests = self.statistical_tests()
        
        # 3. 汇总指标
        metrics = {
            'rank_correlation': rank_corr,
            'topk_precision': topk_prec,
            'decile_analysis': decile_anal,
            'statistical_tests': stat_tests
        }
        
        # 4. 生成可视化
        output_dir = self.run_dir / "ranking_validation"
        self.visualize(output_dir)
        
        # 5. 生成报告
        self.generate_report(metrics, output_dir / "RANKING_VALIDATION_REPORT.md")
        
        # 6. 保存数值结果
        with open(output_dir / "validation_metrics.json", 'w') as f:
            # 转换numpy类型为Python原生类型
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(item) for item in obj]
                return obj
            
            json.dump(convert(metrics), f, indent=2)
        
        print("\n" + "="*60)
        print("✅ 验证完成！")
        print(f"📂 结果保存在: {output_dir}")
        print("="*60)
        
        return metrics


def main():
    # 路径配置
    run_dir = Path("etf_rotation_experiments/results/run_20251113_145102")
    backtest_dir = Path("etf_rotation_experiments/results_combo_wfo")
    
    # 运行验证
    validator = RankingValidator(run_dir, backtest_dir)
    metrics = validator.run_full_validation()
    
    # 打印关键结论
    print("\n" + "="*60)
    print("🎯 核心结论")
    print("="*60)
    
    ic_rho = metrics['rank_correlation']['IC_Baseline']['spearman']
    cal_rho = metrics['rank_correlation']['Calibrator']['spearman']
    improvement = metrics['statistical_tests']['improvement_pct']
    p_value = metrics['statistical_tests']['mann_whitney_p']
    
    print(f"\n1. 排序预测能力:")
    print(f"   - IC Baseline: Spearman = {ic_rho:.3f}")
    print(f"   - Calibrator: Spearman = {cal_rho:.3f}")
    print(f"   - 评价: {validator._judge_correlation(cal_rho)}")
    
    print(f"\n2. 性能提升:")
    print(f"   - Top100平均Sharpe提升: {improvement:+.2f}%")
    print(f"   - 统计显著性: {'✓ 显著 (p<0.05)' if p_value < 0.05 else '✗ 不显著'}")
    
    print(f"\n3. 最终建议:")
    if cal_rho > 0.5 and improvement > 10 and p_value < 0.05:
        print("   ✅ 强烈推荐部署Calibrator排序系统")
    elif cal_rho > 0.3:
        print("   ⚠️ 可以试点，但需加强监控")
    else:
        print("   ❌ 不建议部署，需重新优化")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
