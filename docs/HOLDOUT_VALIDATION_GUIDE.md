# 🔬 Holdout验证操作指南

> **目的**: 防止过拟合，建立策略真实OOS表现的验证机制  
> **原理**: 预留6个月数据不参与训练，仅用于最终验证  
> **创建日期**: 2025-12-09

---

## 📋 一、问题诊断

### 1.1 当前架构的致命缺陷

```
❌ 错误做法（当前）:
2020-01-01 ──────────────────────────────► 2025-12-08
├───────── 全部用于WFO训练和筛选 ─────────┤

结果: 从12597个组合中选出"在全量数据上表现最好"的
     → 本质是在全量数据上做optimization
     → 2025-10~12 (-24.37%回撤) 是真正的OOS，策略崩溃
```

```
✅ 正确做法（Holdout验证）:
2020-01-01 ────────────► 2025-05-31 ────► 2025-12-08
├──── Training ────┤      ├─ Holdout ─┤
   4.5年训练集            6个月验证集
   
   只用训练集:              Holdout集:
   - 跑WFO                 - 完全不碰
   - 选Top100               - 最后验证
   - 选Top1                 - 决定是否启用
```

### 1.2 WFO的"伪OOS"问题

WFO虽然每个窗口内部有IS/OOS划分，但：
- 19个窗口的OOS期覆盖了2020-2025全部数据
- 最终用19个OOS的**平均表现**来排序选Top1
- **等价于在全量数据上做参数优化**

真正的OOS是：**策略选定后，首次在未见过的数据上运行**。

---

## 🚀 二、操作流程

### Step 1: 启用Holdout模式

编辑 `configs/combo_wfo_config.yaml`:

```yaml
data:
  start_date: '2020-01-01'
  end_date: '2025-12-08'           # 完整数据范围
  training_end_date: '2025-05-31'  # 🔬 训练集截止日期
```

**重要**: `training_end_date` 设置后，WFO/VEC/筛选流程将只使用2020~2025/05的数据。

---

### Step 2: 重新训练（仅用训练集）

```bash
# 清除旧缓存（确保使用新的截止日期）
rm -rf .cache/ohlcv_*.pkl

# Step 1: WFO因子组合挖掘（仅用训练集）
python3 src/etf_strategy/run_combo_wfo.py

# Step 2: 全量VEC回测（仅用训练集）
python3 scripts/run_full_space_vec_backtest.py

# Step 3: 策略筛选（基于训练集表现）
python3 scripts/select_strategy_v2.py
```

**预期输出**:
```
results/run_YYYYMMDD_HHMMSS/  # WFO结果（训练集）
results/vec_full_space_*/      # VEC结果（训练集）
results/selection_v2_*/        # Top100策略（基于训练集）
```

---

### Step 3: Holdout期验证

```bash
# 验证Top 10策略在Holdout期的表现
python3 scripts/validate_holdout.py --top_n 10

# 或指定具体的Top策略文件
python3 scripts/validate_holdout.py \
  --input results/selection_v2_*/top100_by_composite.csv \
  --top_n 20
```

**输出**:
```
results/holdout_validation_YYYYMMDD_HHMMSS/
├── holdout_validation.csv  # 详细验证结果
└── summary.txt             # 汇总报告
```

---

### Step 4: 判断是否启用

验证标准（可调整）:
```python
HOLDOUT_MIN_RETURN = 0.0    # Holdout期收益 > 0%
HOLDOUT_MIN_SHARPE = 0.5    # Sharpe > 0.5
HOLDOUT_MAX_DD = 0.20       # 最大回撤 < 20%
```

**决策树**:
```
IF Top1通过所有标准:
  → ✅ 策略可以启用
  → 记录训练集/Holdout期表现差异
  
ELIF Top1-10中有策略通过:
  → ⚠️ 使用通过验证的策略
  → 训练集Top1失效，需警惕
  
ELSE 无任何策略通过:
  → ❌ 策略库整体失效
  → 需要重新设计因子/参数
```

---

## 📊 三、示例分析

### 3.1 假设结果（封板Top1在Holdout期验证）

```
策略: ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + 
      PRICE_POSITION_20D + SHARPE_RATIO_20D

训练集表现 (2020-01-01 至 2025-05-31):
  收益率: 195.32%
  Sharpe: 1.42
  最大回撤: 13.8%

Holdout集表现 (2025-06-01 至 2025-12-08):
  收益率: -18.67%  ❌
  Sharpe: -0.83    ❌
  最大回撤: 22.3%  ❌

验证标准检查:
  收益率 > 0%: ❌ FAIL
  Sharpe > 0.5: ❌ FAIL
  最大回撤 < 20%: ❌ FAIL

结论: ⚠️ 策略未通过Holdout验证，不建议启用
```

### 3.2 可能的结果场景

| 场景 | 训练集收益 | Holdout收益 | 判断 |
|------|-----------|------------|------|
| **理想** | 200% | 15% | ✅ 策略有效，可上线 |
| **轻微劣化** | 200% | 8% | ⚠️ 有劣化但可接受 |
| **严重劣化** | 200% | -5% | ❌ 过拟合，禁止上线 |
| **崩溃** | 200% | -25% | ❌ 完全失效 |

---

## 🔧 四、参数调整建议

### 4.1 Holdout期长度选择

| 长度 | 优点 | 缺点 | 建议 |
|------|------|------|------|
| 3个月 | 训练集数据多 | 验证期太短，可能偶然通过 | ❌ 不推荐 |
| **6个月** | 平衡 | - | ✅ **推荐** |
| 12个月 | 验证更严格 | 训练集减少1年，性能下降 | ⚠️ 数据充足时可用 |

当前配置：**6个月** (2025-06-01 至 2025-12-08)

### 4.2 验证标准调整

根据策略类型调整：

```python
# 进取型策略（高收益高波动）
HOLDOUT_MIN_RETURN = -5%   # 允许小亏
HOLDOUT_MIN_SHARPE = 0.3
HOLDOUT_MAX_DD = 25%

# 平衡型策略（默认）
HOLDOUT_MIN_RETURN = 0%
HOLDOUT_MIN_SHARPE = 0.5
HOLDOUT_MAX_DD = 20%

# 保守型策略（低回撤）
HOLDOUT_MIN_RETURN = 5%
HOLDOUT_MIN_SHARPE = 0.8
HOLDOUT_MAX_DD = 15%
```

---

## 🎯 五、关键要点

### ✅ DO's
1. **永远预留Holdout期** - 无论数据多少，至少预留10%
2. **Holdout期完全不碰** - 训练/调参/筛选都不能用
3. **只验证一次** - 多次验证就变成调参了
4. **记录劣化程度** - 训练集vs Holdout的差异是关键信号

### ❌ DON'Ts
1. **不要反复调整Holdout期** - 一旦设定就不改
2. **不要peek Holdout数据** - 绝对禁止用于任何决策
3. **不要因为不通过就换标准** - 标准要提前定好
4. **不要忽略负面结果** - 不通过就是不通过

---

## 📈 六、预期改进效果

### 6.1 防止过拟合

```
Before (无Holdout):
  训练集: 237%
  真实上线: -24% (2个月)
  差距: 261pp ❌

After (有Holdout):
  训练集: ~180-200%
  Holdout: 预测在 -5% ~ +15%
  真实上线: 预期与Holdout接近 ✅
```

### 6.2 提升策略可信度

- **量化劣化程度**: 知道策略在未见数据上会损失多少性能
- **筛选真正鲁棒的策略**: 10个策略中可能只有2-3个通过
- **避免灾难性失败**: 提前发现-24%级别的崩溃

---

## 🚨 七、常见问题

### Q1: 为什么不用交叉验证？
**A**: 时间序列数据不能打乱，且我们需要模拟"真实上线"场景，Holdout更合适。

### Q2: WFO的OOS不就是验证吗？
**A**: WFO的19个OOS窗口都参与了"选Top1"，所以不是真正的OOS。

### Q3: Holdout期可以调参吗？
**A**: **绝对不行**！Holdout期只能验证，不能调参，否则又变成训练集了。

### Q4: 如果Top10都不通过怎么办？
**A**: 说明策略库整体失效，需要：
  - 重新设计因子
  - 调整参数范围
  - 增加数据源
  - 改变策略逻辑

### Q5: 训练集要重新跑吗？
**A**: 是的，设置`training_end_date`后，需要清缓存重跑WFO/VEC/筛选。

---

## 📚 八、相关文档

- `scripts/validate_holdout.py` - Holdout验证脚本
- `configs/combo_wfo_config.yaml` - 配置文件（含training_end_date）
- `docs/STRATEGY_SELECTION_METHODOLOGY.md` - 策略筛选方法论
- `AGENTS.md` - 项目开发指南

---

**创建时间**: 2025-12-09  
**维护者**: Linus  
**状态**: ✅ 已实现，待验证
