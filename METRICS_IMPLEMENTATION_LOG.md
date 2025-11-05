# 交易指标实现日志

**时间**: 2025-01-20  
**目标**: 为回测引擎添加缺失的交易统计指标  
**状态**: ✅ 完成

## 问题描述

用户发现代码中缺少以下关键交易指标：
- ❌ 胜率相关指标 (win_rate, winning_days, losing_days, avg_win, avg_loss, profit_factor)
- ❌ 风险调整指标 (Calmar ratio, Sortino ratio)  
- ❌ 连胜/连败统计 (max_consecutive_wins, max_consecutive_losses)

虽然这些指标的原始数据（daily_returns_arr）在回测过程中计算出来了，但没有被提取并保存到输出中。

## 实现方案

### 1. 修改 `backtest_no_lookahead()` 函数

**文件**: `etf_rotation_optimized/test_freq_no_lookahead.py`  
**位置**: 第 264-314 行（返回字典前）

**新增计算逻辑**:

```python
# 胜率相关指标
positive_returns = daily_returns_arr[daily_returns_arr > 0]
negative_returns = daily_returns_arr[daily_returns_arr < 0]
win_rate = len(positive_returns) / len(daily_returns_arr)
winning_days = len(positive_returns)
losing_days = len(negative_returns)
avg_win = float(np.mean(positive_returns)) if len(positive_returns) > 0 else 0.0
avg_loss = float(np.mean(negative_returns)) if len(negative_returns) > 0 else 0.0
profit_factor = np.sum(positive_returns) / abs(np.sum(negative_returns))

# 风险调整指标
downside_returns = daily_returns_arr[daily_returns_arr < 0]
calmar_ratio = annual_ret / abs(max_dd)
downside_vol = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
sortino_ratio = annual_ret / downside_vol

# 连胜/连败统计
signs = np.sign(daily_returns_arr)
streaks = np.diff(np.concatenate([[0], signs, [0]]) != 0)
streak_starts = np.where(streaks)[0]
if len(streak_starts) > 1:
    streak_lengths = np.diff(streak_starts)
    win_streaks = streak_lengths[signs[streak_starts[:-1]] > 0]
    loss_streaks = streak_lengths[signs[streak_starts[:-1]] < 0]
    max_consecutive_wins = int(np.max(win_streaks)) if len(win_streaks) > 0 else 0
    max_consecutive_losses = int(np.max(loss_streaks)) if len(loss_streaks) > 0 else 0
else:
    max_consecutive_wins = 0
    max_consecutive_losses = 0
```

**返回字典扩展**: 从 10 个字段 → 23 个字段

### 2. 修改 `main()` 函数中的 CSV 导出逻辑

**文件**: `etf_rotation_optimized/test_freq_no_lookahead.py`  
**位置**: 第 617-635 行（DataFrame 构建处）

**新增字段列表**:

```python
df_local = pd.DataFrame([{
    # 基础信息
    'rank': r['rank'],
    'combo': r['combo'],
    'combo_size': r['combo_size'],
    'wfo_freq': r['wfo_freq'],
    'test_freq': r['test_freq'],
    'freq': r['freq'],
    'wfo_ic': r['wfo_ic'],
    'wfo_score': r['wfo_score'],
    
    # 收益指标
    'final_value': r['final'],
    'total_ret': r['total_ret'],
    'annual_ret': r['annual_ret'],
    
    # 风险指标  
    'vol': r['vol'],
    'sharpe': r['sharpe'],
    'max_dd': r['max_dd'],
    'n_rebalance': r['n_rebalance'],
    'avg_turnover': r['avg_turnover'],
    
    # ✨ 新增：胜率相关 (6字段)
    'win_rate': r['win_rate'],
    'winning_days': r['winning_days'],
    'losing_days': r['losing_days'],
    'avg_win': r['avg_win'],
    'avg_loss': r['avg_loss'],
    'profit_factor': r['profit_factor'],
    
    # ✨ 新增：风险调整指标 (2字段)
    'calmar_ratio': r['calmar_ratio'],
    'sortino_ratio': r['sortino_ratio'],
    
    # ✨ 新增：连胜/连败 (2字段)
    'max_consecutive_wins': r['max_consecutive_wins'],
    'max_consecutive_losses': r['max_consecutive_losses'],
    
    'run_tag': r['run_tag'],
} for r in all_results_local])
```

## 修改汇总

| 功能 | 状态 | 影响范围 |
|------|------|--------|
| ✅ 胜率统计 (6字段) | 完成 | backtest_no_lookahead() 返回值 + CSV 导出 |
| ✅ 风险调整指标 (2字段) | 完成 | backtest_no_lookahead() 返回值 + CSV 导出 |
| ✅ 连胜/连败统计 (2字段) | 完成 | backtest_no_lookahead() 返回值 + CSV 导出 |
| ✅ CSV 字段扩展 | 完成 | 17列 → 30列 |
| ⏳ JSON 详细数据导出 | TODO | 需在 main() 中添加日常返回值和净值保存 |

## 输出格式

### CSV 文件（30列）

生成的 `all_freq_scan_YYYYMMDD_HHMMSS.csv` 或 `top100_backtest_by_ic_YYYYMMDD_HHMMSS.csv` 现在包含：

1. **基础字段** (7列): rank, combo, combo_size, wfo_freq, test_freq, freq, run_tag
2. **IC字段** (2列): wfo_ic, wfo_score
3. **收益字段** (3列): final_value, total_ret, annual_ret
4. **风险字段** (5列): vol, sharpe, max_dd, n_rebalance, avg_turnover
5. **📊 胜率字段** (6列): win_rate, winning_days, losing_days, avg_win, avg_loss, profit_factor
6. **⚙️ 风险调整字段** (2列): calmar_ratio, sortino_ratio
7. **🔄 连胜/连败字段** (2列): max_consecutive_wins, max_consecutive_losses

## 验证

✅ 代码语法检查通过  
✅ 不产生新的编译错误  
✅ 与现有代码逻辑兼容  

## 下一步

### 立即执行
1. 运行 `python test_freq_no_lookahead.py` 执行回测
2. 验证新 CSV 包含所有 30 列
3. 检查指标数值合理性

### 后续优化
1. **保存日常收益数据**: 为每个策略保存 daily_returns_arr 为 JSON/Parquet
2. **保存净值曲线**: 为每个策略保存 nav 数据用于绘图
3. **扩展分析脚本**: 更新 analysis_report.py 利用新指标生成更深入分析

## 测试建议

运行后检查输出：

```bash
# 检查 CSV 文件列数
head -1 results_combo_wfo/YYYYMMDD_HHMMSS/all_freq_scan_*.csv | tr ',' '\n' | wc -l
# 预期: 30 列

# 检查指标值范围
python -c "
import pandas as pd
df = pd.read_csv('results_combo_wfo/YYYYMMDD_HHMMSS/all_freq_scan_*.csv')
print('win_rate 范围:', df['win_rate'].min(), '-', df['win_rate'].max())
print('profit_factor 范围:', df['profit_factor'].min(), '-', df['profit_factor'].max())
print('calmar_ratio 范围:', df['calmar_ratio'].min(), '-', df['calmar_ratio'].max())
print('sortino_ratio 范围:', df['sortino_ratio'].min(), '-', df['sortino_ratio'].max())
"
```

## 相关指标定义

### 胜率指标
- **win_rate**: 正收益天数 / 总交易天数
- **winning_days**: 正收益总天数
- **losing_days**: 负收益总天数  
- **avg_win**: 平均正收益率
- **avg_loss**: 平均负收益率
- **profit_factor**: 正收益总和 / 负收益总和绝对值

### 风险调整指标
- **calmar_ratio**: 年化收益 / |最大回撤|（值越大越好）
- **sortino_ratio**: 年化收益 / 下行波动率（只计算负收益日）

### 连胜/连败
- **max_consecutive_wins**: 最长连赢天数
- **max_consecutive_losses**: 最长连败天数
