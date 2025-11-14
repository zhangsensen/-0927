# WFO排序系统重构执行总结

**执行时间**: 2025-11-13  
**执行人**: AI Agent  
**目标**: 解决WFO IC排序与真实回测收益脱节问题

---

## 一、问题发现与诊断

### 1.1 核心问题

通过 `scripts/validate_ranking_predictive_power.py` 验证发现：

```python
# WFO IC排名 vs 真实回测Sharpe的相关性
Spearman ρ = 0.015  # 近乎零相关
Kendall τ = 0.049   # 同样极弱

# Top-10精度
IC Baseline: 0.0%   # 完全无预测能力
```

**结论**: WFO的 `mean_oos_ic` 排序无法预测真实回测表现，系统性失效。

### 1.2 历史验证数据

| 排序方法 | Spearman ρ | Top-10精度 | Top-100平均Sharpe |
|----------|------------|------------|------------------|
| IC Baseline | 0.015 | 0% | 2.341 |
| ML Calibrator | 0.049 | 0% | 2.385 (+1.9%) |

**发现**: ML校准器虽略有提升，但本质仍是在失效的IC基础上修修补补，治标不治本。

---

## 二、根因分析

### 2.1 WFO窗口设计缺陷

```yaml
# 原配置
is_period: 252    # In-Sample 252天
oos_period: 60    # Out-of-Sample 60天
step_size: 60     # 步进60天
```

**问题**:
1. IS窗口过长（1年），捕捉趋势而非稳定信号
2. OOS窗口过短（3个月），不足以验证真实表现
3. 步进过大，导致只有约8-10个验证窗口，样本不足

### 2.2 ML校准器的陷阱

`core/wfo_realbt_calibrator.py` 试图用GBDT拟合 `WFO特征 → 真实Sharpe`：

**致命缺陷**:
- **泄漏风险**: 训练时已知未来真实表现，实盘无法复现
- **过拟合**: 12,597组合仅200-300个有回测标签，样本稀疏
- **错位映射**: WFO时间窗口与真实回测期不同，本质不可比
- **路径依赖**: 校准器学到的是"当前市场环境的最佳IC特征"，换市场失效

---

## 三、重构方案

### 3.1 设计原则

**核心理念**: 回归本质，排序应直接基于"能否在未来赚钱"的直接信号

1. **去ML化**: 移除所有校准器/排序模型，避免过拟合和泄漏
2. **可扩展**: top_n从100扩展到5000，支持更大候选池
3. **可配置**: 支持多种scoring策略（IC, OOS Sharpe等）
4. **减小combo空间**: 从12,597降至合理规模（计划1,000-3,000）

### 3.2 具体改动

#### 代码层面

**1. 移除校准器**

```diff
- from .wfo_realbt_calibrator import WFORealBacktestCalibrator
- calibrator = WFORealBacktestCalibrator.load(...)
- results_df["calibrated_sharpe_pred"] = calibrator.predict(results_df)
```

**2. 新增scoring_strategy参数**

```python
@dataclass
class ComboWFOConfig:
    # ... 其他参数 ...
    scoring_strategy: str = "ic"  # 可选: "ic", "oos_sharpe"
```

**3. 灵活排序逻辑**

```python
def _apply_scoring(self, results_df: pd.DataFrame) -> pd.DataFrame:
    strategy = self.config.scoring_strategy
    if strategy == "oos_sharpe":
        sort_columns = ["oos_sharpe_proxy", "stability_score", "mean_oos_ic"]
    else:
        sort_columns = ["mean_oos_ic", "stability_score"]
    return results_df.sort_values(by=sort_columns, ascending=[False] * len(sort_columns))
```

**4. 扩展输出**

```python
# 原输出
top100_by_ic.parquet

# 新输出
ranking_ic_top5000.parquet  # 主排名文件
top_combos.parquet          # Top5000组合
top100_by_ic.parquet        # 兼容旧流程
```

#### 配置层面

```yaml
combo_wfo:
  top_n: 5000              # 从100 → 5000
  scoring_strategy: ic     # 明确声明排序策略
```

---

## 四、执行步骤

### 4.1 代码修改清单

✅ **完成项**:

1. **删除校准器模块**
   - 移除 `core/wfo_realbt_calibrator.py`
   - 移除 `scripts/patch_calibrator_medians.py`

2. **更新ComboWFOOptimizer**
   - 添加 `scoring_strategy` 参数
   - 删除80行校准器加载/排序逻辑
   - 新增 `_apply_scoring()` 方法
   - 调整 `run_combo_search()` 默认 top_n=5000

3. **更新run_combo_wfo.py**
   - 传递 `scoring_strategy` 参数
   - 生成3个输出文件（ranking_ic_top5000/top_combos/top100_by_ic）
   - 增强日志记录

4. **更新configs/combo_wfo_config.yaml**
   - 设置 `top_n: 5000`
   - 添加 `scoring_strategy: ic`

5. **文档化**
   - 创建 `DEPRECATED_CALIBRATOR_SCRIPTS.md` 列出23个废弃脚本
   - 标注依赖 `ranking_blends/` 的工具不再维护

### 4.2 影响范围评估

**兼容性**:
- ✅ 保留 `top100_by_ic.parquet` 兼容旧回测脚本
- ⚠️  废弃 `ranking_blends/` 目录（ranking_baseline.parquet, ranking_lightgbm.parquet）
- ⚠️  23个依赖校准器的脚本需替换

**性能影响**:
- ⏱️ WFO运行时间不变（优化器逻辑未变）
- 💾 输出文件从0.5MB增至约2.5MB（5000行）
- 🚀 回测启动更快（无需加载校准器模型）

---

## 五、验证计划

### 5.1 单元测试（待执行）

```bash
# 1. 验证新配置可运行
python etf_rotation_experiments/run_combo_wfo.py

# 2. 检查输出文件结构
ls -lh results/run_latest/
# 预期: ranking_ic_top5000.parquet, top_combos.parquet, top100_by_ic.parquet

# 3. 验证排序一致性
python -c "
import pandas as pd
top5k = pd.read_parquet('results/run_latest/ranking_ic_top5000.parquet')
top100 = pd.read_parquet('results/run_latest/top100_by_ic.parquet')
assert all(top100['combo'] == top5k.head(100)['combo']), 'Top100一致性失败'
print('✅ 排序一致性验证通过')
"
```

### 5.2 回测验证（待执行）

```bash
# 使用新排名运行真实回测
python real_backtest/run_profit_backtest.py \
  --topk 1000 \
  --ranking-file results/run_latest/ranking_ic_top5000.parquet

# 计算相关性
python -c "
import pandas as pd
from scipy.stats import spearmanr

wfo = pd.read_parquet('results/run_latest/ranking_ic_top5000.parquet')
backtest = pd.read_csv('real_backtest/latest_backtest.csv')
merged = wfo.merge(backtest, on='combo')
rho, p = spearmanr(merged['mean_oos_ic'], merged['sharpe_net'])
print(f'WFO IC vs 真实Sharpe: Spearman ρ={rho:.4f}, p={p:.4e}')
"
```

---

## 六、后续优化方向

### 6.1 Phase 2: WFO窗口优化（P0 - 高优先级）

**问题**: 当前252/60/60配置产生12,597个combo → 组合爆炸

**解决方案**:

```yaml
# 方案A: 减少combo_sizes
combo_sizes: [2, 3]  # 从[2,3,4,5] → [2,3], 减少80%组合

# 方案B: 增加step_size
step_size: 60 → 120  # 减少一半窗口数

# 方案C: 缩短IS周期
is_period: 252 → 180  # 9个月IS，更敏捷
```

**目标**: 将combo总数降至1,000-3,000个。

### 6.2 Phase 3: 直接优化真实Sharpe（P1 - 中优先级）

**理念**: 为什么不在WFO中直接计算"模拟真实回测的Sharpe"？

**实现**:

```python
def _compute_oos_sharpe_proxy(self, signal_oos, returns_oos, rebalance_freq):
    """在OOS窗口模拟实盘回测，计算Sharpe代理"""
    holdings = []
    for t in range(0, len(signal_oos), rebalance_freq):
        sig = signal_oos[t]
        top_stocks = np.argsort(sig)[-4:]  # Top4持仓
        holdings.append(top_stocks)
    
    # 计算等权组合收益
    portfolio_returns = []
    for h, period_end in zip(holdings, ...):
        ret = returns_oos[period_start:period_end, h].mean(axis=1).mean()
        portfolio_returns.append(ret)
    
    return np.mean(portfolio_returns) / np.std(portfolio_returns)  # Sharpe
```

**配置**:

```yaml
scoring_strategy: oos_sharpe
```

**优势**:
- 排序直接对齐真实回测目标
- 无需ML校准器
- 自然包含持仓数、换仓成本影响

### 6.3 Phase 4: 多时间尺度验证（P2 - 低优先级）

**问题**: 单一窗口配置可能过拟合特定市场环境

**方案**: 在多个IS/OOS配置下评估稳健性

```python
configs = [
    (180, 60),  # 6个月IS + 2个月OOS
    (252, 60),  # 当前配置
    (360, 90),  # 12个月IS + 3个月OOS
]

for is_p, oos_p in configs:
    score = run_wfo(is_period=is_p, oos_period=oos_p)
    stability_score += (score > threshold)

# 只保留在所有配置下都稳定的组合
```

---

## 七、风险与应对

### 7.1 已识别风险

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| IC排序仍失效 | 高 | 高 | Phase 2/3优化方案兜底 |
| Top5000过大导致回测慢 | 中 | 低 | 可调整topk参数 |
| 旧脚本误用 | 低 | 中 | DEPRECATED文档 + 代码注释 |

### 7.2 回滚方案

如果新方案效果更差，可回滚至校准器版本：

```bash
# 1. 恢复校准器模块
git checkout HEAD~1 -- etf_rotation_experiments/core/wfo_realbt_calibrator.py

# 2. 恢复combo_wfo_optimizer.py
git checkout HEAD~1 -- etf_rotation_experiments/core/combo_wfo_optimizer.py

# 3. 恢复配置
git checkout HEAD~1 -- etf_rotation_experiments/configs/combo_wfo_config.yaml
```

---

## 八、总结

### 8.1 核心成果

✅ **移除失效的ML校准器**: 删除800+行过拟合代码  
✅ **回归本质排序**: scoring_strategy可配置，支持未来扩展  
✅ **扩大候选池**: top_n=5000，提供更多优质组合  
✅ **文档完整**: 明确标注23个废弃脚本，避免误用  

### 8.2 关键洞察

1. **ML不是万能药**: 当基础信号失效时，ML只能锦上添花，无法雪中送炭
2. **过拟合陷阱**: 训练集用未来数据（真实回测结果），实盘必然失效
3. **回归本质**: 排序应直接优化目标（Sharpe），而非间接指标（IC）

### 8.3 下一步行动

1. ⏳ **Phase 1验证** (本周): 运行新WFO + 回测，验证IC相关性是否改善
2. 🎯 **Phase 2执行** (下周): 优化窗口配置，减少combo空间
3. 🚀 **Phase 3实验** (两周内): 实现oos_sharpe_proxy排序策略
4. 📊 **持续监控**: 建立WFO→RealBacktest映射质量监控dashboard

---

**附录**: 
- 代码变更: `git diff HEAD~1 HEAD -- etf_rotation_experiments/`
- 废弃脚本清单: `DEPRECATED_CALIBRATOR_SCRIPTS.md`
- Phase 2方案细节: 待补充窗口优化实验结果
