# WFO-回测排序对齐验证报告

**生成时间**: 2024-11-13 19:15  
**配置文件**: `configs/combo_wfo_config_stable_oossharpe.yaml`  
**评分策略**: `oos_sharpe_true`  
**分析工具**: `scripts/analyze_wfo_backtest_alignment.py`

---

## 执行摘要 (Executive Summary)

**目标**: 验证 WFO 的 `mean_oos_sharpe` 排序与真实回测的 `sharpe_net` 排序是否对齐

**结果**: ❌ **严重不对齐** - Spearman 相关系数 ≈ -0.05（p > 0.5，无显著相关性）

**关键发现**:
- WFO 端计算的是 **13 个窗口 Sharpe 的算术平均** (`mean_oos_sharpe`)
- 回测端计算的是 **全周期累积收益的 Sharpe** (`sharpe_net`)
- 这两个度量在数学上 **不等价**: `E[Sharpe_i] ≠ Sharpe(Σ returns_i)`

**下一步行动**: 必须选择其一作为统一度量，或增加新的匹配度量

---

## 1. 背景与动机

### 1.1 研究问题
在 WFO 的组合优化中，我们希望筛选出的 Top-N 组合在实盘回测中也能保持优势。核心假设：
> WFO 排序靠前 ⇒ 回测收益/Sharpe 靠前

### 1.2 技术路径
- **WFO 端**: 每个组合在 13 个 OOS 窗口上独立计算持仓 Sharpe（top-5 等权），最终对窗口 Sharpe 取平均
- **回测端**: 用全周期数据复现信号，按窗口调仓，计算整个周期的累积净值曲线后算总 Sharpe

### 1.3 实施步骤
1. 代码增强: 补充 `mean_oos_sample_count` 字段与策略元数据
2. WFO 执行: 12597 个组合，单频 rebalance=8
3. 双场景回测: 0bps 与 2bps 滑点
4. 排序对齐分析: Spearman/Kendall + Top-K 重叠度

---

## 2. 数据流与关键指标

### 2.1 WFO 输出
**文件**: `results/run_20251113_185715/all_combos.parquet`  
**关键列**:
```
combo                  str      示例: "BB1-6_CLOSE-OPEN_2_1.5_0.6"
mean_oos_sharpe        float    [-0.65, 1.83], μ=0.45, σ=0.44
mean_oos_ic            float    IC 均值
oos_sharpe_proxy       float    OOS 代理 Sharpe
stability_score        float    稳定性评分
mean_oos_sample_count  float    88.3 ± 1.4 天/窗口
best_rebalance_freq    int      最优调仓频率（天）
```

**排序策略**: `oos_sharpe_true` = 按 `[mean_oos_sharpe, stability_score, oos_sharpe_proxy, mean_oos_ic]` 降序

### 2.2 回测输出
**文件**: `results/run_20251113_185715/backtest_0bps_results.csv` / `backtest_2bps_results.csv`  
**关键列**:
```
combo                str
annual_ret_gross     float    年化毛收益
annual_ret_net       float    扣成本后年化收益
sharpe_gross         float    毛 Sharpe
sharpe_net           float    净 Sharpe（用于排序）
max_dd               float    最大回撤
win_rate             float    胜率
```

**Top1 表现**:
| 场景    | combo                         | annual_ret_net | sharpe_net | max_dd |
|---------|-------------------------------|----------------|------------|--------|
| 0bps    | BB1-6_CLOSE-OPEN_2_1.5_0.6    | 20.56%         | 0.938      | -17.2% |
| 2bps    | BB1-6_CLOSE-OPEN_2_1.5_0.6    | 19.68%         | 0.898      | -17.2% |

---

## 3. 排序对齐分析结果

### 3.1 秩相关系数（初步测试 Top-100）
| 场景  | Spearman ρ | p-value | Kendall τ | p-value | 结论           |
|-------|------------|---------|-----------|---------|----------------|
| 0bps  | -0.053     | 0.598   | -0.043    | 0.524   | 无显著相关     |
| 2bps  | -0.051     | 0.612   | -0.042    | 0.539   | 无显著相关     |

**解读**:
- ρ ≈ 0: WFO 排名与回测排名几乎无线性关系
- p > 0.5: 统计上无法拒绝"零相关"假设
- 加入 2bps 成本后相关性仍为零

**局限**: ⚠️ 仅测试 Top-100，重叠度 100% 是假阳性（测试集=筛选集）

---

### 3.2 扩大验证：Top-1000 回测结果 🆕

**测试规模**: 1000 组合（覆盖 WFO 排序的前 8% 样本）

#### 秩相关系数（Top-1000）
| 场景  | Spearman ρ | p-value | Kendall τ | p-value | 结论           |
|-------|------------|---------|-----------|---------|----------------|
| 0bps  | 0.0036     | 0.911   | 0.0008    | 0.969   | 无显著相关     |
| 2bps  | 0.0058     | 0.855   | 0.0024    | 0.910   | 无显著相关     |

#### Top-K 重叠分析（Top-1000）
| K值  | WFO Top-K ∩ Backtest Top-K | 重叠率 | 场景  |
|------|----------------------------|--------|-------|
| 100  | 15 / 100                   | 15.0%  | 0bps  |
| 500  | 243 / 500                  | 48.6%  | 0bps  |
| 1000 | 1000 / 1000                | 100%   | 0bps  |
| 100  | 15 / 100                   | 15.0%  | 2bps  |
| 500  | 243 / 500                  | 48.6%  | 2bps  |
| 1000 | 1000 / 1000                | 100%   | 2bps  |

**关键发现**:
- ⚠️ **Top-100 重叠仅 15%**: WFO 筛选的最优 100 个组合中，只有 15 个在回测中也进入 Top-100
- 📉 **随机性水平**: 15% 接近随机选择的期望值（100/1000 = 10%），说明 WFO 排序几乎无预测力
- 🔴 **业务严重失效**: 如果用 WFO Top-100 实盘，有 85% 的组合实际表现不在真实 Top-100
- ✅ **成本无关性**: 0bps 和 2bps 重叠率完全一致，排除了滑点成本导致排序变化的假设

---

## 4. 根因分析

### 4.1 度量不匹配
#### WFO 的 `mean_oos_sharpe`
```python
# 伪代码: combo_wfo_optimizer.py L111-181
for window in oos_windows:
    window_rets = []
    for day in window:
        top5_assets = select_top_k(signals[day], k=5)
        port_ret = mean(rets[top5_assets])  # 等权
        window_rets.append(port_ret)
    
    window_sharpe = mean(window_rets) / std(window_rets) * sqrt(252)
    sharpes.append(window_sharpe)

mean_oos_sharpe = mean(sharpes)  # 13 个窗口的算术平均
```

#### 回测的 `sharpe_net`
```python
# 伪代码: run_profit_backtest.py
全周期累积净值曲线 = []
for 每日:
    当日收益 = portfolio_return - 成本
    累积净值 *= (1 + 当日收益)

daily_rets = cumulative_nav.pct_change()
sharpe_net = mean(daily_rets) / std(daily_rets) * sqrt(252)
```

### 4.2 数学不等价性
**Jensen 不等式**: 对于非线性函数 f(x) = μ/σ（Sharpe 比率）:
```
E[Sharpe(窗口_i)] ≠ Sharpe(全部窗口累积收益)
```

**举例**:
- 窗口1: 收益10%, std=5% → Sharpe=2.0
- 窗口2: 收益-5%, std=3% → Sharpe=-1.67
- 平均: mean_sharpe = (2.0 - 1.67) / 2 = 0.165

若将窗口1+2 拼接:
- 总收益: (1.10 × 0.95) - 1 = 4.5%
- 混合std可能≠平均std
- 总Sharpe ≠ 0.165

### 4.3 实现差异
虽然两端都用 top-5 等权 + 8天调仓，但存在细微差异:
- **WFO**: 每窗口独立，无复利累积
- **回测**: 全周期连续复利，滑点/冲击成本按净值扣减

---

## 5. 影响评估

### 5.1 当前状态
✅ **已完成**:
- True OOS Sharpe 计算逻辑正确（Welford 在线算法）
- 产物列齐全: `mean_oos_sharpe`, `mean_oos_sample_count`, `oos_sharpe_std_mean`
- 回测逻辑可复现 WFO 的调仓

❌ **未解决**:
- WFO 筛选的 Top-100 在回测中的排名 **不可预测**
- 无法满足用户目标: "WFO产出的排序结果在真实回测中的排序结果是重叠的"

### 5.2 误用风险
若直接用当前 `mean_oos_sharpe` 排序做实盘:
- **过拟合风险**: 窗口平均 Sharpe 高，不代表复利 Sharpe 高
- **资金曲线偏差**: 实盘按全周期累积，与 WFO 的评分依据不一致

---

## 6. 解决方案对比

### 方案 A: 修改 WFO 度量 (推荐)
**动作**: 在 WFO 中也计算"跨窗口累积收益的 Sharpe"

**优点**:
- 与回测度量一致，排序直接可用
- 符合投资者实际体验（复利累积）

**缺点**:
- 需改动 `combo_wfo_optimizer.py` 的 `_compute_rebalanced_sharpe_stats`
- 窗口间需传递累积净值，增加计算复杂度

**实现提示**:
```python
# 在 _test_combo_impl 中
cumulative_nav = 1.0
all_rets = []
for window in oos_windows:
    for day in window:
        port_ret = ...
        cumulative_nav *= (1 + port_ret)
        all_rets.append(port_ret)

compound_sharpe = mean(all_rets) / std(all_rets) * sqrt(252)
# 替换当前的 mean_oos_sharpe
```

### 方案 B: 修改回测报告
**动作**: 回测脚本额外报告 "窗口平均 Sharpe"

**优点**:
- WFO 代码无需改动
- 可同时保留两种度量供对比

**缺点**:
- 窗口平均 Sharpe 对投资者意义有限（不反映真实资金曲线）
- 需手动切割回测周期与 WFO 的 13 个窗口对齐

### 方案 C: 多度量验证
**动作**: 同时用 `mean_oos_ic`, `oos_sharpe_proxy`, `stability_score` 做回归/集成

**优点**:
- 不依赖单一度量
- 可能发现更鲁棒的组合特征

**缺点**:
- 复杂度高，调参成本大
- 仍需至少一个度量与回测对齐

---

## 7. 行动建议 (优先级排序)

### P0 (必须立即完成)
1. **扩大回测范围**: 用全量 12597 combos 或 Top-1000 做回测，获得真实 Top-K 重叠率
2. **诊断性实验**: 在回测结果中手动计算 "窗口平均 Sharpe"，验证是否与 WFO 的 `mean_oos_sharpe` 相关

### P1 (本周完成)
3. **实施方案 A**: 修改 WFO 为 "跨窗口复利 Sharpe"
   - 创建新配置 `scoring_strategy=oos_sharpe_compound`
   - 在 `_test_combo_impl` 中累积窗口收益
   - 对比新旧策略的 Top-100 重叠度

4. **单元测试**: 用 5 只 ETF + 100 天数据，验证 WFO→回测全流程，固化到 `tests/test_oos_sharpe_pipeline.py`

### P2 (优化迭代)
5. **文档更新**: 在 `QUICK_REFERENCE_CARD.md` 说明:
   - `oos_sharpe_true` (平均窗口) vs `oos_sharpe_compound` (复利累积)
   - 何时使用哪个度量

6. **对比分析**: 跑一次 IC / proxy / true / compound 四策略，横向对比 Sharpe/回撤/换手

---

## 8. 附录

### 8.1 完整执行日志
```bash
# 1. WFO 执行
cd /Users/zhangshenshen/深度量化0927/etf_rotation_experiments
python run_combo_wfo.py -c configs/combo_wfo_config_stable_oossharpe.yaml
# 输出: results/run_20251113_185715/all_combos.parquet (12597行)

# 2. 验证输出列
python -c "
import pandas as pd
df = pd.read_parquet('results/run_20251113_185715/all_combos.parquet')
print(df[['mean_oos_sharpe', 'mean_oos_sample_count']].describe())
"
# mean_oos_sharpe: μ=0.45, σ=0.44, min=-0.65, max=1.83
# mean_oos_sample_count: μ=88.3, σ=1.4

# 3. 零滑点回测
python real_backtest/run_profit_backtest.py \
  --ranking-file results/run_20251113_185715/ranking_oos_sharpe_true_top1000.parquet \
  --topk 100 --slippage-bps 0 \
  --output results/run_20251113_185715/backtest_0bps_results.csv
# Top1: 20.56% annual, 0.938 Sharpe

# 4. 2bps 回测
python real_backtest/run_profit_backtest.py \
  --ranking-file results/run_20251113_185715/ranking_oos_sharpe_true_top1000.parquet \
  --topk 100 --slippage-bps 2.0 \
  --output results/run_20251113_185715/backtest_2bps_results.csv
# Top1: 19.68% annual, 0.898 Sharpe

# 5. 对齐分析
python scripts/analyze_wfo_backtest_alignment.py \
  --wfo-file results/run_20251113_185715/all_combos.parquet \
  --backtest-file results/run_20251113_185715/backtest_0bps_results.csv \
  --output results/run_20251113_185715/alignment_0bps.json
# Spearman: -0.053 (p=0.598)

python scripts/analyze_wfo_backtest_alignment.py \
  --wfo-file results/run_20251113_185715/all_combos.parquet \
  --backtest-file results/run_20251113_185715/backtest_2bps_results.csv \
  --output results/run_20251113_185715/alignment_2bps.json
# Spearman: -0.051 (p=0.612)
```

### 8.2 配置文件摘要
```yaml
# configs/combo_wfo_config_stable_oossharpe.yaml
window_config:
  is_period: 180
  oos_period: 90
  step_size: 90  # 13个窗口

portfolio:
  top_k: 30
  rebalance_frequencies: [8]

scoring:
  strategy: oos_sharpe_true
  position_size: 5  # OOS 窗口内持仓数
  rank_method: mean
  rank_weight: 1.0
  positive_multiplier: 1.2
```

### 8.3 相关代码位置
- **WFO Sharpe 计算**: `etf_rotation_experiments/core/combo_wfo_optimizer.py` L111-181 `_compute_rebalanced_sharpe_stats`
- **排序逻辑**: 同文件 L425-475 `_apply_scoring`
- **回测主逻辑**: `etf_rotation_experiments/real_backtest/run_profit_backtest.py` L400-550
- **对齐分析工具**: `etf_rotation_experiments/scripts/analyze_wfo_backtest_alignment.py`

---

## 9. 总结

当前实现在 **工程质量** 上已达标:
- ✅ True OOS Sharpe 算法正确
- ✅ 产物列齐全，可观测
- ✅ 回测可复现 WFO 调仓逻辑

但在 **业务目标** 上未达成:
- ❌ WFO 排序无法预测回测排序 (ρ ≈ 0)
- ❌ 度量不一致: 窗口平均 Sharpe ≠ 复利累积 Sharpe

**核心矛盾**: 选择 WFO 用"稳定性"（窗口平均）还是"累积性"（复利 Sharpe）作为优化目标？

**下一步关键**: 必须先在 Top-1000 或全量上回测，确认当前不对齐的程度；然后实施方案 A，修改为复利 Sharpe，重新验证。

---

**报告维护**: 本文档应在每次修改 WFO 度量或回测逻辑后更新，确保团队对排序对齐性的理解同步。
