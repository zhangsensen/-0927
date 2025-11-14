# WFO重构路线图：找回排序预测能力

**创建时间**: 2025-11-13  
**核心目标**: **让WFO排序和真实回测收益排序高度重叠（Spearman>0.5）**

---

## 🎯 核心问题陈述

**当前状态**：
- WFO排序预测能力 = 0（Spearman=0.015-0.049）
- Top10 Precision = 0%
- Decile非单调（D8最优而非D1）

**目标状态**：
- Spearman > 0.5
- Top10 Precision > 50%
- Decile单调递减
- 可复现、可实盘

---

## 📋 四阶段执行计划

### Phase 1: 根本原因诊断（1-2天）

#### 目标
找出WFO过拟合的根源

#### 任务清单

**Task 1.1: WFO窗口配置审查**
```bash
# 检查项目
- [ ] IS窗口长度（是否过长导致过拟合）
- [ ] OOS窗口长度（是否太短无法验证）
- [ ] 窗口滚动方式（expanding vs sliding）
- [ ] 总窗口数量（19个是否合理）
```

**Task 1.2: 数据质量检查**
```python
# 执行脚本
python scripts/diagnose_wfo_data_quality.py
# 检查内容
- 缺失值分布
- 异常值检测（极端IC值）
- 时间对齐（是否有未来函数）
- 复权一致性
```

**Task 1.3: IC vs 真实收益关系分析**
```python
# 核心问题：为什么IC和Sharpe不相关？
python scripts/analyze_ic_sharpe_divergence.py
# 分析维度
- IC计算方法是否正确
- IC是否忽略了关键成本（换手、极端值）
- 是否存在regime切换
```

**输出**：
- `WFO_DIAGNOSIS_REPORT.md`（诊断报告）
- 明确的过拟合根源
- 3个具体改进方向

---

### Phase 2: 快速验证（2-3天）

#### 目标
用小样本快速测试改进方案

#### 方案设计

**Baseline: 随机选择**
- 从12597个组合中随机选100个
- 计算真实回测的平均Sharpe
- 作为对比基准

**Approach A: IC稳健化**
```python
# 改进IC排序方法
- 使用IR（IC / IC_std）代替mean_oos_ic
- 加入正向率过滤（positive_rate > 60%）
- 多指标综合（IC + stability + turnover）
```

**Approach B: 真实收益模拟**
```python
# 在WFO OOS窗口模拟真实交易
- 计算每个窗口的模拟Sharpe（含成本）
- 使用mean_oos_sharpe代替mean_oos_ic
- 需要修改WFO引擎
```

**Approach C: 组合空间缩减**
```python
# 减少多重检验
- 先用domain knowledge筛选到500个
- 再用WFO排序
- Bonferroni校正
```

**验证流程**：
1. 在100个精选组合上测试3个方案
2. 计算Spearman相关性
3. 选择最优方案进入Phase 3

**输出**：
- 3个方案的对比结果
- 选定的最优方案
- 预期Spearman提升

---

### Phase 3: 框架重构（5-7天）

#### 目标
重新设计WFO排序系统

#### 核心改动

**3.1 优化目标切换**
```python
# 从IC切换到真实收益
class WFOEngine:
    def evaluate_combo(self, combo, oos_data):
        # OLD: 计算IC
        # ic = compute_ic(combo, oos_data)
        
        # NEW: 模拟真实交易
        backtest = simulate_trading(
            combo=combo,
            data=oos_data,
            slippage_bps=2.0,
            cost_model=HKStockCostModel()
        )
        return backtest.sharpe_net  # 直接返回Sharpe
```

**3.2 窗口重新设计**
```yaml
# 新窗口配置
wfo:
  window_type: sliding  # 改为滑动窗口
  is_months: 12  # IS缩短到12个月
  oos_months: 6   # OOS延长到6个月
  step_months: 3  # 每3个月滚动一次
  total_windows: 15  # 减少窗口数
```

**3.3 组合空间缩减**
```python
# 预筛选逻辑
def prefilter_combos(all_combos):
    # 规则1：因子数量 3-5个
    # 规则2：历史IC均值 > 0.01
    # 规则3：正向率 > 55%
    # 规则4：避免高度相关因子组合
    return filtered_combos  # 从12597降到500
```

**3.4 多指标排序**
```python
# 综合排序公式
def compute_rank_score(combo, wfo_results):
    sharpe = wfo_results['mean_oos_sharpe']
    stability = wfo_results['sharpe_std']  # 越低越好
    turnover = wfo_results['avg_turnover']  # 越低越好
    
    # 归一化后加权
    score = (
        0.6 * normalize(sharpe) +
        0.3 * normalize(-stability) +
        0.1 * normalize(-turnover)
    )
    return score
```

**输出**：
- 新的WFO引擎代码
- 500个预筛选组合
- 新的排序文件

---

### Phase 4: 独立验证（3-5天）

#### 目标
确保系统在未见过的数据上有效

#### 验证策略

**4.1 时间切分**
```python
# Holdout验证
train_end = '2024-06-30'  # WFO训练到2024年中
holdout_start = '2024-07-01'  # 留出最近6个月
holdout_end = '2024-12-31'

# 在holdout期间测试
```

**4.2 验证指标**
```python
# 必须通过的检查
checks = [
    ('Spearman > 0.5', check_spearman),
    ('Top10 Precision > 50%', check_top10_precision),
    ('Decile单调性', check_monotonicity),
    ('统计显著 p < 0.05', check_significance),
    ('Top10 vs Random > 30%', check_economic_significance)
]
```

**4.3 稳健性测试**
```python
# Bootstrap验证
for i in range(100):
    sample = bootstrap_sample(holdout_data)
    spearman = compute_spearman(sample)
    spearman_dist.append(spearman)

# 95%置信区间应该 > 0.4
```

**输出**：
- Holdout验证报告
- Bootstrap置信区间
- 通过/失败判定

---

## 🔧 技术实现细节

### 数据流
```
原始数据 → WFO窗口切分 → 每窗口模拟交易 → 计算OOS Sharpe → 
多窗口聚合 → 综合排序 → 验证 → 输出Top100
```

### 核心脚本

**1. 诊断脚本**
```bash
scripts/diagnose_wfo_data_quality.py
scripts/analyze_ic_sharpe_divergence.py
scripts/validate_wfo_windows.py
```

**2. 重构脚本**
```bash
core/wfo_engine_v2.py  # 新WFO引擎
core/combo_prefilter.py  # 组合预筛选
core/multi_metric_ranker.py  # 多指标排序
```

**3. 验证脚本**
```bash
scripts/holdout_validation.py
scripts/bootstrap_test.py
scripts/rank_correlation_monitor.py
```

---

## 📊 成功标准

### 必须达标（Phase 4）

| 指标 | 当前 | 目标 | 方法 |
|------|------|------|------|
| Spearman相关性 | 0.015-0.049 | **> 0.5** | spearmanr() |
| Top-10 Precision | 0% | **> 50%** | overlap计数 |
| Top-100 Precision | 0-22% | **> 30%** | overlap计数 |
| Decile单调性 | ✗ D8最优 | ✓ D1最优 | 人工检查 |
| 统计显著性 | p=0.94 | **p < 0.05** | Mann-Whitney U |
| 经济显著性 | - | **+30% Sharpe** | Top10 vs Random |

### 加分项

- [ ] Precision@5 > 60%
- [ ] Kendall tau > 0.4
- [ ] 不同市场regime下稳定
- [ ] 计算效率 < 10分钟（500组合）

---

## ⚠️ 风险控制

### 主要风险

**Risk 1: 修复后仍然零相关**
- 缓解：Phase 2快速验证失败则暂停
- 备选：完全放弃WFO，改用简单规则

**Risk 2: 过度优化holdout数据**
- 缓解：严格隔离holdout，只用一次
- 备选：使用滚动验证

**Risk 3: 时间超期**
- 缓解：每个Phase设deadlines
- 备选：简化目标（降低Spearman阈值到0.3）

---

## 📅 时间规划

| Phase | 时间 | 里程碑 | 产出 |
|-------|------|--------|------|
| Phase 1 | Day 1-2 | 根因明确 | 诊断报告 |
| Phase 2 | Day 3-5 | 方案选定 | 对比实验 |
| Phase 3 | Day 6-12 | 系统重构 | 新WFO引擎 |
| Phase 4 | Day 13-17 | 验证通过 | 验证报告 |
| **Total** | **17天** | **可部署** | **新排序系统** |

---

## 🚦 决策点

### Checkpoint 1 (Day 2)
**问题**：找到根因了吗？  
**Yes** → 进入Phase 2  
**No** → 延长Phase 1或寻求外部帮助

### Checkpoint 2 (Day 5)
**问题**：有方案Spearman > 0.3吗？  
**Yes** → 进入Phase 3  
**No** → 重新评估可行性，考虑备选方案

### Checkpoint 3 (Day 12)
**问题**：新系统代码完成了吗？  
**Yes** → 进入Phase 4  
**No** → 简化功能或延长时间

### Checkpoint 4 (Day 17)
**问题**：所有指标达标了吗？  
**Yes** → 部署上线  
**No** → 分析差距，决定是否继续或pivot

---

## 💡 Linus准则

1. **简洁优先**：不要复杂的ML，先试简单规则
2. **快速验证**：每个改动必须立即测试相关性
3. **向量化到底**：所有计算必须向量化
4. **日志透明**：每个决策必须可追溯
5. **数据即真理**：不要猜测，用回测证明

---

## 📚 参考资料

- 当前诊断报告：`CRITICAL_FINDING_WFO_OVERFITTING.md`
- 验证脚本：`scripts/validate_ranking_predictive_power.py`
- 可视化结果：`results/run_20251113_145102/ranking_validation/`

---

## 🎬 下一步行动

**立即执行**：
```bash
# 1. 开始Phase 1诊断
cd etf_rotation_experiments
python scripts/diagnose_wfo_data_quality.py

# 2. 分析IC-Sharpe关系
python scripts/analyze_ic_sharpe_divergence.py

# 3. 检查WFO窗口配置
python scripts/validate_wfo_windows.py
```

**预期时间**：今天完成Phase 1，明天开始Phase 2

---

**计划制定者**: Linus AI  
**审核状态**: 待执行  
**下次更新**: 完成Phase 1后
