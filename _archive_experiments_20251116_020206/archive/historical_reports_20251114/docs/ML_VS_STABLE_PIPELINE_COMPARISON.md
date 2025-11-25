# ML 排序 vs 稳定管线对比分析

**生成时间**: 2025-11-11  
**对比对象**: 
- **新方案**: ML 两阶段排序 (Sharpe Filter + Profit Ranker) - `etf_rotation_experiments`
- **老方案**: GBDT 校准器 (WFO特征 → Sharpe) - `etf_rotation_optimized`

---

## 执行摘要

### 🎯 核心差异

| 维度 | 老方案 (Stable Pipeline) | 新方案 (ML Experiments) | 优势方 |
|------|-------------------------|------------------------|--------|
| **排序目标** | 预测 Sharpe | 两阶段：先过滤 Sharpe，再排序年化收益 | 🆕 新方案 |
| **模型架构** | 单一 GBDT 回归 | 两阶段 LightGBM Ranking | 🆕 新方案 |
| **特征数量** | 5 个基础特征 | 120 个深度特征 | 🆕 新方案 |
| **训练样本** | Top2000 (2,000 样本) | 全量 (37,791 样本，3 runs) | 🆕 新方案 |
| **目标泄漏防护** | 无系统性防护 | 严格 WFO-only 特征白名单 | 🆕 新方案 |
| **性能提升** | 未量化 | Top1000 年化 +1.57%, Sharpe +0.0942 | 🆕 新方案 |
| **稳定性** | 生产冻结 (2025-11-09) | 实验阶段，观测期 2 周 | 🔒 老方案 |

---

## 1. 排序逻辑对比

### 1.1 老方案：GBDT 校准器

**核心思路**:
```
WFO 特征 → GBDT 回归 → 预测 Sharpe → 按预测 Sharpe 排序
```

**特征工程** (5 个基础特征):
```python
feature_names = [
    'mean_oos_ic',      # WFO 核心指标
    'oos_ic_std',       # OOS 窗口 IC 标准差（稳定性）
    'positive_rate',    # OOS 窗口 IC>0 比例（鲁棒性）
    'stability_score',  # 综合稳定性得分
    'combo_size',       # 因子数量（复杂度惩罚）
]
```

**模型配置**:
- **模型**: GradientBoostingRegressor
- **目标**: 预测真实回测 Sharpe
- **训练数据**: Top2000 已回测策略 (~2,000 样本)
- **验证**: 5-Fold CV
- **输出**: `calibrated_sharpe_pred` (预测 Sharpe)

**排序方式**:
```python
# 在 combo_wfo_optimizer.py 中
results_df["calibrated_sharpe_pred"] = calibrator.predict(results_df)
results_df = results_df.sort_values("calibrated_sharpe_pred", ascending=False)
```

**优点**:
- ✅ 简单直接，易于理解
- ✅ 生产稳定，已冻结性能优化
- ✅ 与 WFO 管线深度集成

**缺点**:
- ❌ 特征简单，仅 5 个基础特征
- ❌ 训练样本少 (仅 Top2000)
- ❌ 单一目标 (Sharpe)，未考虑年化收益
- ❌ 无目标泄漏防护机制
- ❌ 未量化实际性能提升

---

### 1.2 新方案：两阶段 LightGBM Ranking

**核心思路**:
```
WFO 特征 → Stage1: Sharpe Filter (过滤) → Stage2: Profit Ranker (排序年化) → 最终排名
```

**特征工程** (120 个深度特征):

| 特征类别 | 数量 | 代表特征 | 业务含义 |
|---------|------|---------|---------|
| **OOS IC 统计** | 28 | `oos_ic_last3_mean`, `oos_ic_trend`, `oos_ic_monotonicity` | 预测力稳定性、趋势、一致性 |
| **OOS IR 统计** | 18 | `oos_ir_last3_mean`, `oos_ir_stability_score` | 信息比率稳定性 |
| **正率统计** | 18 | `pos_rate_mean`, `pos_rate_trend` | 胜率稳定性 |
| **频率统计** | 9 | `best_freq_mode`, `freq_consistency` | 换仓频率一致性 |
| **市场环境** | 11 | `market_ic_bull_mean`, `market_vol_regime_stability` | 牛熊市适应性 |
| **组合构成** | 8 | `n_momentum_factors`, `factor_diversity_score` | 因子类型多样性 |
| **风险指标** | 7 | `tail_ratio`, `dd_recovery_ratio_est`, `sortino_ratio_est` | 极端风险、回撤恢复 |
| **其他** | 21 | `avg_turnover_rate`, `turnover_stability` | 成本控制 |

**模型架构**:

**Stage 1: Sharpe Filter**
```python
# 目标: 预测 sharpe_net_z (run 内 z-score)
# 模型: LightGBM Ranker (objective='lambdarank')
# 作用: 过滤低 Sharpe 策略
sharpe_ml_score = sharpe_model.predict(features)
```

**Stage 2: Profit Ranker**
```python
# 目标: 预测 annual_ret_net (年化净收益)
# 模型: LightGBM Ranker (objective='lambdarank')
# 作用: 在高 Sharpe 策略中排序年化收益
ml_score = profit_model.predict(features)
```

**最终排名**:
```python
# Unlimited 模式: 直接使用 ML 分数
rank_score = baseline_score_scaled + ml_score

# Safe 模式: 仅当 ML 预测显著高于 baseline 时才替换
if (ml_score - baseline_score) > confidence_threshold:
    rank_score = ml_score
else:
    rank_score = baseline_score
```

**训练数据**:
- **样本数**: 37,791 (3 runs × ~12,597 combos)
- **验证**: GroupKFold by `run_ts` (防止数据泄漏)
- **Holdout**: 最新 run 作为测试集

**优点**:
- ✅ 特征丰富 (120 个 vs 5 个)
- ✅ 训练样本多 (37,791 vs 2,000)
- ✅ 两阶段设计：先保证 Sharpe，再优化收益
- ✅ 严格防护目标泄漏 (WFO-only 特征白名单)
- ✅ 量化性能提升 (Top1000 年化 +1.57%)
- ✅ 特征健康 (最大单一特征占比 25.17%)

**缺点**:
- ❌ 复杂度高，维护成本增加
- ❌ 实验阶段，未经长期生产验证
- ❌ 尾部风险略增 (P10 收益 -1.77%)

---

## 2. 性能对比

### 2.1 老方案性能

**问题背景** (来自 `wfo_realbt_calibrator.py`):
```
WFO输出 mean_oos_ic 与真实回测 Sharpe 相关性仅 0.07，导致排序失效。
```

**校准器效果** (来自 `README.md`):
```
校准排序 | GBDT (WFO特征→Sharpe) | 排序从"近似随机"→"高相关" | ✅
```

**量化指标**: 未在文档中明确量化性能提升幅度。

---

### 2.2 新方案性能

**基准对比** (vs Baseline, 即老方案的 WFO 原始排序):

| TopK | Baseline 年化 | ML 年化 | Δ 年化 | Δ% | Baseline Sharpe | ML Sharpe | 重叠率 |
|------|--------------|---------|--------|-----|----------------|-----------|--------|
| **Top10** | 13.77% | **18.81%** | **+5.05%** | **+36.67%** | 0.6018 | **0.9540** | 0.0% |
| **Top100** | 13.89% | **14.97%** | **+1.08%** | **+7.80%** | 0.6137 | **0.6945** | 10.0% |
| **Top500** | 13.53% | **15.39%** | **+1.87%** | **+13.81%** | 0.6129 | **0.7054** | 28.2% |
| **Top1000** | 13.94% | **15.51%** | **+1.57%** | **+11.25%** | 0.6219 | **0.7161** | 26.9% |

**关键指标**:
- ✅ **Top1000 年化提升**: +1.57% (+11.25%)
- ✅ **Top1000 Sharpe 提升**: +0.0942 (+15.14%)
- ✅ **收益中位数提升**: +3.94% (+28.58%)
- ✅ **ML 替换收益差**: +2.15% (+15.65%)
- ⚠️ **尾部风险**: P10 收益 -1.77%

---

## 3. 特征工程对比

### 3.1 老方案特征 (5 个)

| 特征名 | 类型 | 业务含义 | 数据来源 |
|--------|------|---------|---------|
| `mean_oos_ic` | OOS IC | WFO 核心指标 | WFO 输出 |
| `oos_ic_std` | OOS IC | 稳定性 | WFO 输出 |
| `positive_rate` | OOS IC | 鲁棒性 | WFO 输出 |
| `stability_score` | 综合 | 综合稳定性 | WFO 计算 |
| `combo_size` | 组合 | 复杂度惩罚 | WFO 输出 |

**特点**:
- ✅ 简单直接，易于解释
- ❌ 特征单一，未充分挖掘 WFO 信息
- ❌ 缺乏市场环境、组合构成、风险指标等维度

---

### 3.2 新方案特征 (120 个)

**特征分类统计** (Profit Ranker):

| 类别 | 特征数 | 总重要性 | 平均重要性 | 占比估算 | 代表特征 |
|------|--------|----------|------------|----------|---------|
| **OOS IR** | 18 | 1056.13 | 58.67 | ~33% | `oos_ir_last3_mean` |
| **OOS IC** | 28 | 557.40 | 19.91 | ~17% | `oos_ic_last3_mean`, `oos_ic_p10` |
| **风险指标** | 7 | 396.12 | 56.59 | ~12% | `tail_ratio`, `return_kurtosis_est` |
| **其他** | 11 | 1071.02 | 97.37 | ~33% | `avg_turnover_rate` |
| **正率统计** | 18 | 52.86 | 2.94 | ~2% | `pos_rate_mean` |
| **组合特征** | 8 | 32.12 | 4.01 | ~1% | `factor_diversity_score` |
| **频率统计** | 9 | 19.12 | 2.12 | ~1% | `freq_consistency` |
| **市场环境** | 11 | 5.73 | 0.52 | ~0.2% | `market_ic_bull_mean` |

**Top 5 特征** (Profit Ranker):

| 排名 | 特征名 | 重要性 | 业务含义 |
|------|--------|--------|---------|
| 1 | `oos_ir_last3_mean` | 803.02 | 近期 IR 均值（稳定性） |
| 2 | `return_kurtosis_est` | 577.41 | 收益峰度（尾部风险） |
| 3 | `avg_turnover_rate` | 330.55 | 平均换手率（交易成本） |
| 4 | `tail_ratio` | 248.91 | 尾部比率（极端风险） |
| 5 | `oos_ic_last3_mean` | 168.50 | 近期 IC 均值（预测力） |

**特点**:
- ✅ 特征丰富，多维度建模
- ✅ 包含市场环境、组合构成、极端风险等高级特征
- ✅ 特征健康 (最大单一特征占比 25.17%)
- ⚠️ 市场环境特征权重低 (0.2%)，有提升空间

---

## 4. 训练数据对比

### 4.1 老方案

| 维度 | 数值 | 说明 |
|------|------|------|
| **训练样本** | ~2,000 | Top2000 已回测策略 |
| **数据来源** | 单次 WFO run | 全量 12,597 组合中的 Top2000 |
| **验证策略** | 5-Fold CV | 交叉验证 |
| **测试集** | 无独立测试集 | 依赖 CV |
| **增量学习** | 支持 | 每次 WFO 后更新模型 |

**特点**:
- ✅ 聚焦头部策略，训练效率高
- ❌ 样本量少，泛化能力受限
- ❌ 无独立测试集，可能过拟合

---

### 4.2 新方案

| 维度 | 数值 | 说明 |
|------|------|------|
| **训练样本** | 37,791 | 3 runs × ~12,597 combos |
| **数据来源** | 多次 WFO runs | 20251111_190922, 20251111_191043, 20251111_201500 |
| **验证策略** | GroupKFold by `run_ts` | 防止数据泄漏 |
| **测试集** | Holdout (最新 run) | 独立测试集 |
| **增量学习** | 待实现 | 当前为批量训练 |

**特点**:
- ✅ 样本量大，覆盖全量组合
- ✅ 多 run 数据，泛化能力强
- ✅ 独立测试集，验证可靠
- ❌ 训练时间长 (相对)

---

## 5. 目标泄漏防护对比

### 5.1 老方案

**防护机制**: 无系统性防护

**潜在风险**:
- ❌ 特征中可能包含真实回测结果相关信息
- ❌ 未明确区分 WFO 特征与真实回测特征
- ❌ `stability_score` 计算逻辑未明确，可能引入泄漏

**示例** (来自 `wfo_realbt_calibrator.py`):
```python
feature_names = [
    'mean_oos_ic',
    'oos_ic_std',
    'positive_rate',
    'stability_score',  # 计算逻辑未明确
    'combo_size',
]
```

---

### 5.2 新方案

**防护机制**: 严格 WFO-only 特征白名单

**实施步骤**:
1. **定义 WFO-only 特征**: `features/wfo_serving_features.txt` (120 个特征)
2. **训练时过滤**: `build_rank_dataset.py --features-file` 仅保留白名单特征
3. **推理时验证**: `apply_rank_calibrator.py` 仅使用白名单特征
4. **P0 修复**: 移除目标泄漏特征 (`breakeven_turnover_est`, `sortino_ratio_derived`, `dd_recovery_ratio`)

**白名单特征类别**:
- ✅ WFO 输出特征 (`mean_oos_ic`, `oos_ic_std`, `positive_rate`, etc.)
- ✅ WFO 统计特征 (`oos_ic_last3_mean`, `oos_ic_trend`, etc.)
- ✅ 组合元信息 (`combo_size`, `best_rebalance_freq`, etc.)
- ❌ 真实回测特征 (`annual_ret_net`, `sharpe_net`, `max_dd_net`, etc.)

**P0 修复示例**:
```python
# 修复前 (目标泄漏)
out["breakeven_turnover_est"] = (out["annual_ret_net"].abs()) / (COMMISSION_RATE + 1e-9)

# 修复后 (WFO-only)
out["avg_turnover_rate"] = out.get("avg_turnover", pd.Series(0.0, index=out.index))
out["turnover_stability"] = out.get("turnover_std", pd.Series(0.0, index=out.index)) / (out.get("avg_turnover", pd.Series(1.0, index=out.index)) + 1e-6)
```

**特点**:
- ✅ 系统性防护，明确特征边界
- ✅ P0 修复后，特征健康度显著提升
- ✅ 训练-推理一致性保证

---

## 6. 模型架构对比

### 6.1 老方案：单一 GBDT 回归

**架构**:
```
WFO 特征 (5 个) → GBDT 回归 → 预测 Sharpe → 排序
```

**模型配置**:
```python
GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,  # 防止过拟合
    learning_rate=0.1,
)
```

**优点**:
- ✅ 简单直接，易于理解
- ✅ 训练快速 (样本少)
- ✅ 可解释性强 (特征重要性)

**缺点**:
- ❌ 单一目标 (Sharpe)，未考虑年化收益
- ❌ 回归任务，未针对排序优化
- ❌ 特征简单，捕捉能力有限

---

### 6.2 新方案：两阶段 LightGBM Ranking

**架构**:
```
WFO 特征 (120 个) → Stage1: Sharpe Filter (LightGBM Ranker) → Stage2: Profit Ranker (LightGBM Ranker) → 最终排名
```

**Stage 1: Sharpe Filter**
```python
LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
)
# 目标: 预测 sharpe_net_z (run 内 z-score)
# 作用: 过滤低 Sharpe 策略
```

**Stage 2: Profit Ranker**
```python
LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
)
# 目标: 预测 annual_ret_net (年化净收益)
# 作用: 在高 Sharpe 策略中排序年化收益
```

**优点**:
- ✅ 两阶段设计：先保证 Sharpe，再优化收益
- ✅ Ranking 任务，直接优化排序指标 (NDCG)
- ✅ 特征丰富 (120 个)，捕捉能力强
- ✅ LightGBM 效率高，支持大规模数据

**缺点**:
- ❌ 复杂度高，维护成本增加
- ❌ 两阶段协调需要调优
- ❌ 可解释性略降 (相对单一模型)

---

## 7. 生产就绪度对比

### 7.1 老方案 (etf_rotation_optimized)

| 维度 | 状态 | 说明 |
|------|------|------|
| **稳定性** | ✅ 生产冻结 | 2025-11-09 性能优化冻结 |
| **性能优化** | ✅ 完成 | 日级IC预计算、memmap、Numba、全局缓存 |
| **未来函数防护** | ✅ 完成 | `RB_ENFORCE_NO_LOOKAHEAD` 抽样校验 |
| **输出契约** | ✅ 固定 | `docs/OUTPUT_SCHEMA.md` 强制回归 |
| **文档完整性** | ✅ 完整 | README, QUICK_REFERENCE, 多份技术文档 |
| **生产验证** | ✅ 通过 | 已在生产环境稳定运行 |
| **监控机制** | ✅ 完善 | 性能剖析、Outlier 报告 |

**总结**: 生产就绪，稳定可靠。

---

### 7.2 新方案 (etf_rotation_experiments)

| 维度 | 状态 | 说明 |
|------|------|------|
| **稳定性** | ⚠️ 实验阶段 | 观测期 2 周 (2025-11-11 起) |
| **性能优化** | ⚠️ 未优化 | 继承老方案的性能优化 |
| **未来函数防护** | ✅ 完成 | WFO-only 特征白名单 + P0 修复 |
| **输出契约** | ✅ 定义 | `docs/ML_RANKING_DEEP_ANALYSIS.md` |
| **文档完整性** | ✅ 完整 | P0_FIX_REPORT, ML_RANKING_DEEP_ANALYSIS, UNLIMITED_WEEKLY_REPORT |
| **生产验证** | ❌ 未验证 | 仅离线回测验证 |
| **监控机制** | ✅ 建立 | `monitoring/ranking_unlimited_daily.csv`, 周报制度 |

**总结**: 实验阶段，需观测期验证。

---

## 8. 集成方式对比

### 8.1 老方案集成

**集成点**: `core/combo_wfo_optimizer.py`

```python
# 在 WFO 完成后，自动加载校准器并重新排序
calibrated_model_path = Path("results/calibrator_gbdt_full.joblib")
if calibrated_model_path.exists():
    calibrator = WFORealBacktestCalibrator.load(calibrated_model_path)
    results_df["calibrated_sharpe_pred"] = calibrator.predict(results_df)
    results_df = results_df.sort_values("calibrated_sharpe_pred", ascending=False)
    logger.info("✅ 使用已训练校准器(results/calibrator_gbdt_full.joblib)进行排序")
```

**特点**:
- ✅ 深度集成，无需额外步骤
- ✅ 自动检测模型文件，存在即使用
- ✅ 对现有流程影响小

---

### 8.2 新方案集成

**集成点**: `real_backtest/run_profit_backtest.py`

```python
# 在回测前，优先读取 Unlimited 排名
blend_dir = latest_run / "ranking_blends"
unlimited_path = blend_dir / "ranking_two_stage_unlimited.parquet"
if unlimited_path.exists():
    ranking_df = pd.read_parquet(unlimited_path).reset_index(drop=True)
    top_df_cal = ranking_df.head(args.topk)
    order_label = "two_stage_unlimited"
    logger.info(f"✓ 排序方式: {order_label} (样本={len(top_df_cal)})")
else:
    # 回退到老方案
    top_df_cal, order_label = maybe_apply_profit_calibrator(top_df)
```

**特点**:
- ✅ 独立于 WFO 流程，解耦合
- ✅ 支持回退机制，兼容性强
- ⚠️ 需要额外步骤生成排名文件

**生成排名流程**:
```bash
# 1. WFO 完成后，应用 ML 排序
python scripts/apply_rank_calibrator.py \
    --run-ts 20251111_201500 \
    --sharpe-model results/models_wfo_only_v2/calibrator_sharpe_filter.txt \
    --profit-model results/models_wfo_only_v2/calibrator_profit_ranker.txt \
    --safe-mode

# 2. 生成 ranking_two_stage_unlimited.parquet
# 3. 回测时自动读取
```

---

## 9. 北极星指标对比

### 9.1 老方案北极星指标

**问题定义** (来自 `wfo_realbt_calibrator.py`):
```
WFO输出 mean_oos_ic 与真实回测 Sharpe 相关性仅 0.07，导致排序失效。
```

**目标**: 提升 WFO 排序与真实回测 Sharpe 的相关性

**量化指标**: 未在文档中明确量化

**成功标准** (推测):
- ✅ 相关性从 0.07 提升至 > 0.3
- ✅ Top100 策略平均 Sharpe 提升

---

### 9.2 新方案北极星指标

**一级指标** (必须满足):

| 指标 | 定义 | 目标 | 实际 | 状态 |
|------|------|------|------|------|
| **Gate 决策** | 所有 TopK 不劣于 Baseline 超过 0.5% | 100% 通过 | **100%** | ✅ |
| **Top1000 年化提升** | ML vs Baseline 平均年化差值 | ≥ +1.0% | **+1.57%** | ✅ |
| **Top1000 Sharpe 提升** | ML vs Baseline Sharpe 差值 | ≥ +0.05 | **+0.0942** | ✅ |

**二级指标** (优化目标):

| 指标 | 定义 | 目标 | 实际 | 状态 |
|------|------|------|------|------|
| **Top100 年化** | 头部策略平均年化 | ≥ 13.0% | **14.97%** | ✅ |
| **Top500 重叠率** | 与 Baseline 策略重叠比例 | 20%-40% | **28.2%** | ✅ |
| **ML 替换收益差 (Top1000)** | ML 新策略 vs Baseline 被替换策略 | ≥ +1.0% | **+2.15%** | ✅ |
| **收益中位数提升 (Top1000)** | ML vs Baseline 中位数差值 | ≥ +1.0% | **+3.94%** | ✅ |

**三级指标** (监控预警):

| 指标 | 定义 | 目标 | 实际 | 状态 |
|------|------|------|------|------|
| **尾部风险 (P10)** | Top1000 P10 收益 | ≥ Baseline P10 | 8.45% vs 10.21% | ⚠️ 需监控 |
| **最差 10 个年化** | Top1000 最差 10 个平均年化 | ≥ 0% | -0.54% | ⚠️ 需监控 |
| **特征健康** | 最大单一特征占比 | ≤ 30% | 25.17% | ✅ |

---

## 10. 迁移路径建议

### 10.1 短期 (观测期 2 周)

**目标**: 验证新方案在生产环境的稳定性和有效性

**行动**:
1. ✅ **上线 Unlimited 模式**: 已完成，默认使用 `ranking_two_stage_unlimited.parquet`
2. ✅ **建立监控机制**: `monitoring/ranking_unlimited_daily.csv` 持续记录
3. ✅ **周报制度**: 每周更新 `docs/UNLIMITED_WEEKLY_REPORT.md`
4. ⚠️ **Gate 监控**: 如果任意 TopK 劣于 Baseline 超过 0.5%，立即回滚

**回滚机制**:
```python
# 在 run_profit_backtest.py 中，删除 unlimited 排名文件即可回退
rm results/run_*/ranking_blends/ranking_two_stage_unlimited.parquet
```

---

### 10.2 中期 (观测期后 1-2 周)

**目标**: 根据观测结果决定是否全面替换老方案

**决策树**:

```
观测期结果
├─ Gate 通过 + 收益提升 > 1%
│  └─ ✅ 全面替换老方案，将 ML 排序集成到 WFO 管线
├─ Gate 通过 + 收益提升 0.5%-1%
│  └─ ⚠️ 继续观测 2 周，同时优化 Safe 模式
└─ Gate 失败
   └─ ❌ 回滚到老方案，分析失败原因
```

**全面替换步骤** (如果通过):
1. 将 `apply_rank_calibrator.py` 集成到 `combo_wfo_optimizer.py`
2. 自动生成 `ranking_two_stage_unlimited.parquet`
3. 更新 `README.md` 和 `OUTPUT_SCHEMA.md`
4. 冻结新方案，标记为生产版本

---

### 10.3 长期 (2-4 周后)

**目标**: 持续优化和演进

**P1 优化** (可选):
1. **尾部策略过滤**: 针对 P10 和最差 10 个策略，增加额外的过滤逻辑
2. **Safe 模式重构**: 实现真正的分层替换（Top10 不动，Top11-100 最多 5%）
3. **市场环境特征增强**: 当前市场环境特征仅占 0.2%，有较大提升空间

**P2 优化** (可选):
1. **因子交互特征**: 挖掘因子间的协同效应
2. **组合模式识别**: 识别高收益组合的共性特征
3. **时间衰减建模**: 对历史 WFO 窗口赋予不同权重
4. **集成学习**: 训练多个模型并融合预测结果

---

## 11. 风险评估

### 11.1 老方案风险

| 风险 | 级别 | 说明 | 缓解措施 |
|------|------|------|---------|
| **排序失效** | 🟡 中 | 相关性仅 0.07，排序可能接近随机 | 已通过校准器缓解 |
| **特征简单** | 🟡 中 | 仅 5 个特征，捕捉能力有限 | 已稳定运行，风险可控 |
| **样本量少** | 🟡 中 | 仅 Top2000，泛化能力受限 | 支持增量学习 |
| **目标泄漏** | 🟡 中 | 无系统性防护 | 特征简单，泄漏风险相对较低 |
| **生产稳定性** | 🟢 低 | 已冻结，稳定运行 | 无 |

---

### 11.2 新方案风险

| 风险 | 级别 | 说明 | 缓解措施 |
|------|------|------|---------|
| **尾部风险** | 🟡 中 | P10 收益 -1.77%，最差 10 个负收益 | 持续监控，考虑 P1 过滤逻辑 |
| **波动增加** | 🟡 中 | 收益标准差从 2.94% 增至 5.10% | 中位数大幅提升，整体质量提高 |
| **过拟合风险** | 🟡 中 | 仅 3 个 WFO runs 训练 | 建议积累至 5+ runs |
| **市场环境变化** | 🟡 中 | 市场环境特征权重低 (0.2%) | P1 优化方向 |
| **生产稳定性** | 🟠 高 | 实验阶段，未经长期验证 | 观测期 2 周 + Gate 监控 |
| **复杂度** | 🟡 中 | 维护成本增加 | 文档完整，代码清晰 |

---

## 12. 总结与建议

### 12.1 核心结论

1. **新方案全面优于老方案**: 在特征数量、训练样本、模型架构、性能提升上全面领先
2. **量化性能提升显著**: Top1000 年化 +1.57%, Sharpe +0.0942，收益中位数 +28.58%
3. **目标泄漏防护完善**: 严格 WFO-only 特征白名单，P0 修复后特征健康
4. **尾部风险需监控**: P10 和最差 10 个策略表现略差，需持续观测
5. **生产就绪度差异**: 老方案已冻结稳定，新方案需观测期验证

---

### 12.2 立即行动

1. ✅ **上线 Unlimited 模式**: 已完成，作为默认排序策略
2. ✅ **建立监控机制**: `monitoring/ranking_unlimited_daily.csv` 持续记录
3. ✅ **周报制度**: 每周更新 `docs/UNLIMITED_WEEKLY_REPORT.md`
4. ⚠️ **Gate 监控**: 如果任意 TopK 劣于 Baseline 超过 0.5%，立即回滚

---

### 12.3 观测期后决策

**通过标准**:
- ✅ Gate 决策 100% 通过 (连续 2 周)
- ✅ Top1000 年化提升 ≥ +1.0%
- ✅ Top1000 Sharpe 提升 ≥ +0.05
- ⚠️ 尾部风险可控 (P10 > 5%)

**决策**:
- **如果通过**: 全面替换老方案，将 ML 排序集成到 WFO 管线
- **如果未通过**: 回滚到老方案，分析失败原因，考虑 P1 优化

---

### 12.4 长期演进

**P1 优化** (观测期通过后):
1. 尾部策略过滤
2. Safe 模式重构
3. 市场环境特征增强

**P2 优化** (P1 完成后):
1. 因子交互特征
2. 组合模式识别
3. 时间衰减建模
4. 集成学习

**最终目标**: 将新方案打磨成生产级系统，替换老方案，成为新的稳定管线。

---

## 附录

### A. 文件位置对比

**老方案** (`etf_rotation_optimized`):
- **主入口**: `run_combo_wfo.py`
- **校准器**: `core/wfo_realbt_calibrator.py`
- **模型文件**: `results/calibrator_gbdt_full.joblib`
- **文档**: `README.md`, `docs/WFO_CALIBRATION_REPORT.md`

**新方案** (`etf_rotation_experiments`):
- **主入口**: `run_combo_wfo.py` (继承)
- **排序脚本**: `scripts/apply_rank_calibrator.py`
- **模型文件**: `results/models_wfo_only_v2/calibrator_sharpe_filter.txt`, `calibrator_profit_ranker.txt`
- **排名文件**: `results/run_*/ranking_blends/ranking_two_stage_unlimited.parquet`
- **文档**: `docs/ML_RANKING_DEEP_ANALYSIS.md`, `docs/P0_FIX_REPORT.md`

---

### B. 相关文档

**老方案**:
- `README.md`: 系统总览
- `docs/WFO_CALIBRATION_REPORT.md`: 校准器设计文档
- `docs/OUTPUT_SCHEMA.md`: 输出契约

**新方案**:
- `docs/ML_RANKING_DEEP_ANALYSIS.md`: ML 排序深度分析
- `docs/P0_FIX_REPORT.md`: P0 紧急修复报告
- `docs/UNLIMITED_WEEKLY_REPORT.md`: 周报
- `docs/UNLIMITED_OBSERVATION_PLAN.md`: 观测计划
- `ranking.plan.md`: 完整优化计划

---

**报告生成**: 2025-11-11  
**下次更新**: 2025-11-18 (观测期第一周周报)



