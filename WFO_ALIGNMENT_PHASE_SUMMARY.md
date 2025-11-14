# WFO-回测对齐验证 - 阶段性总结

**更新时间**: 2024-11-13 19:10  
**当前状态**: 已完成 Top-1000 扩大验证，发现严重排序不对齐

---

## 🎯 核心结论

**WFO 的 `oos_sharpe_true` 策略无法预测回测表现，Top-100 重叠度仅 15%（接近随机）**

---

## 📊 验证数据汇总

### 1. WFO 执行结果
- **配置**: `combo_wfo_config_stable_oossharpe.yaml`
- **组合数**: 12597
- **评分策略**: `oos_sharpe_true` (按 mean_oos_sharpe 降序)
- **mean_oos_sharpe 分布**: μ=0.45, σ=0.44, 范围 [-0.65, 1.83]
- **窗口数**: 13 个 OOS 窗口，每窗口平均 88.3 天

### 2. 回测表现

#### Top-100 回测
| 场景  | Top1 年化 | Top1 Sharpe | 测试组合数 |
|-------|-----------|-------------|-----------|
| 0bps  | 20.56%    | 0.938       | 100       |
| 2bps  | 19.68%    | 0.898       | 100       |

#### Top-1000 回测
| 场景  | Top1 年化 | Top1 Sharpe | 测试组合数 |
|-------|-----------|-------------|-----------|
| 0bps  | 23.39%    | 1.033       | 1000      |
| 2bps  | 22.41%    | 0.990       | 1000      |

**观察**: 扩大到 Top-1000 后，Top1 表现**更好**（年化 +2.8%），说明 WFO Top-100 **不是真正的最优集合**。

### 3. 排序对齐分析

#### 秩相关系数（Top-1000 vs Top-1000）
| 场景  | Spearman ρ | p-value | 样本数 | 结论       |
|-------|------------|---------|--------|-----------|
| 0bps  | 0.0036     | 0.911   | 1000   | 无相关性   |
| 2bps  | 0.0058     | 0.855   | 1000   | 无相关性   |

#### Top-K 重叠度（关键指标）
| K值  | WFO Top-K | 回测 Top-K | 重叠数 | 重叠率 | 随机期望 |
|------|-----------|-----------|--------|--------|---------|
| 100  | 100       | 100       | 15     | **15%**| 10%     |
| 500  | 500       | 500       | 243    | 48.6%  | 50%     |

**解读**:
- Top-100 重叠 15%：仅比随机选择（10%）好 5 个百分点
- Top-500 重叠 48.6%：接近随机期望（50%）
- **结论**: WFO 排序对实盘表现几乎无预测力

---

## 🔍 根因分析

### 原因 1: 度量不一致（主要矛盾）

#### WFO 端
```python
# 计算 13 个窗口各自的 Sharpe，然后取算术平均
mean_oos_sharpe = mean([Sharpe_window1, Sharpe_window2, ..., Sharpe_window13])
```

#### 回测端
```python
# 全周期连续复利，计算整体 Sharpe
sharpe_net = Sharpe(全部累积收益)
```

**数学不等价**:
$$
E[\text{Sharpe}_i] \neq \text{Sharpe}\left(\sum \text{returns}_i\right)
$$

### 原因 2: 实现细节差异（次要矛盾）

虽然两端都用：
- Top-5 等权持仓
- 8 天调仓频率
- 相同因子信号

但存在潜在差异：
- **窗口独立 vs 连续复利**: WFO 每窗口独立计算，回测跨窗口累积
- **调仓日期对齐**: 可能存在微小偏差
- **信号重构**: 回测需重新计算因子 IC 权重，可能与 WFO 略有差异

---

## ⚠️ 业务影响评估

### 当前策略的问题

1. **过拟合风险**: WFO 优化的是"窗口平均 Sharpe"，但实盘关心"复利累积 Sharpe"
2. **排序失效**: 用 WFO Top-100 实盘，有 85% 组合实际不在真实 Top-100
3. **资源浪费**: 花费计算资源优化的指标，与实际表现不相关

### 如果直接用于实盘

- ❌ **预期收益**: 无法保证 WFO Top-1 在实盘也是 Top-1（当前 Top-1 年化 20.56%，但 Top-1000 中有组合达 23.39%）
- ❌ **稳定性**: 15% 重叠率意味着换榜频繁，实盘组合与 WFO 筛选结果脱节
- ❌ **可解释性**: 无法向投资者解释为何 WFO 筛选的最优组合实际表现平庸

---

## 🚀 解决方案路径

### 方案 A: 修改 WFO 度量为复利累积 Sharpe（推荐）

**目标**: 让 WFO 优化的指标与回测/实盘一致

**实施步骤**:
1. 在 `combo_wfo_optimizer.py` 创建 `scoring_strategy=oos_sharpe_compound`
2. 修改 `_compute_rebalanced_sharpe_stats` 逻辑:
   ```python
   # 旧逻辑（窗口平均）
   mean_oos_sharpe = mean([sharpe_w1, sharpe_w2, ...])
   
   # 新逻辑（复利累积）
   all_rets = []
   for window in oos_windows:
       all_rets.extend(window_daily_rets)
   oos_sharpe_compound = Sharpe(all_rets)  # 全部 OOS 期收益的 Sharpe
   ```
3. 用新策略重跑 WFO，验证与回测的相关性

**预期效果**:
- Spearman ρ > 0.7（高相关）
- Top-100 重叠率 > 70%

**风险**:
- 计算复杂度略增（需跨窗口累积）
- 失去"窗口稳定性"的评估维度（可用其他指标补充）

### 方案 B: 修改回测报告窗口平均 Sharpe

**目标**: 让回测报告 WFO 使用的度量

**实施步骤**:
1. 修改 `run_profit_backtest.py` 保存 `daily_returns` 列（JSON 序列化）
2. 用 `diagnose_window_sharpe_alignment.py` 工具计算回测的窗口平均 Sharpe
3. 用窗口平均 Sharpe 重新排序，验证与 WFO 的对齐度

**预期效果**:
- 如果相关性高 → 证明问题仅在度量选择
- 如果相关性仍低 → 说明还有实现细节差异

**局限**:
- 窗口平均 Sharpe 对投资者意义有限（不反映复利）
- 治标不治本，实盘仍需关心累积 Sharpe

### 方案 C: 多度量集成优化

**目标**: 用多个度量加权，提升鲁棒性

**实施步骤**:
1. 保留 `mean_oos_sharpe`（稳定性）
2. 增加 `oos_sharpe_compound`（累积性）
3. 综合 `mean_oos_ic`（信号质量）
4. 加权排序: `final_score = 0.4*sharpe_compound + 0.3*mean_sharpe + 0.3*ic`

**优点**: 兼顾多维度
**缺点**: 调参复杂，权重难确定

---

## 📋 下一步行动计划

### P0 - 立即执行（本周内）

1. **实施方案 A**: 修改 WFO 为复利 Sharpe
   - [ ] 在 `combo_wfo_optimizer.py` 添加 `oos_sharpe_compound` 分支
   - [ ] 创建新配置 `combo_wfo_config_compound.yaml`
   - [ ] 跑一次小规模测试（5 ETF, 180 天）验证逻辑
   - [ ] 全量 WFO + 回测，计算新策略的对齐度

2. **诊断窗口差异**（可选，验证假设）
   - [ ] 修改 `run_profit_backtest.py` 保存 `daily_returns`
   - [ ] 用诊断工具计算回测窗口平均 Sharpe
   - [ ] 验证与 WFO 的相关性（预期仍低，因实现差异）

### P1 - 后续优化

3. **创建回归测试**
   - [ ] 用 5 ETF + 100 天数据，跑完整流程
   - [ ] 固化到 `tests/test_oos_sharpe_pipeline.py`
   - [ ] CI 集成，防止未来破坏

4. **文档更新**
   - [ ] 在 `QUICK_REFERENCE_CARD.md` 说明 `oos_sharpe_true` vs `oos_sharpe_compound`
   - [ ] 更新 `WFO_BACKTEST_ALIGNMENT_REPORT.md` 包含最新发现

5. **横向对比**
   - [ ] 对比 IC / proxy / true / compound 四策略
   - [ ] 生成 Sharpe/回撤/换手对比表

---

## 📈 进度跟踪

- ✅ 代码增强（mean_oos_sample_count, wfo_summary 元数据）
- ✅ WFO 执行（12597 组合）
- ✅ Top-100 回测（0bps + 2bps）
- ✅ Top-1000 回测（0bps + 2bps）
- ✅ 排序对齐分析（发现 15% 重叠）
- ✅ 根因分析（度量不一致 + 实现差异）
- ⏳ 实施修复方案（待开始）
- ⏳ 验证新策略对齐度（待开始）

---

## 🔗 相关文件

- **验证报告**: `/Users/zhangshenshen/深度量化0927/WFO_BACKTEST_ALIGNMENT_REPORT.md`
- **对齐分析**:
  - `etf_rotation_experiments/results/run_20251113_185715/alignment_1000_0bps.json`
  - `etf_rotation_experiments/results/run_20251113_185715/alignment_1000_2bps.json`
- **诊断工具**: `etf_rotation_experiments/scripts/diagnose_window_sharpe_alignment.py`
- **待办清单**: 当前会话的 TODO 列表

---

**最后更新**: 2024-11-13 19:10  
**下次审查**: 实施方案 A 后，重新验证对齐度
