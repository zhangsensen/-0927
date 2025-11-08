<!-- ALLOW-MD --># 全量回测方案实施文档

## 背景

之前仅对WFO Top2000组合执行真实回测用于训练校准器，存在以下问题：
1. **样本偏差**：Top2000仅覆盖高IC组合（IC > 0.10），缺少低IC样本
2. **过拟合风险**：模型在高IC区域训练，可能对低IC区域泛化能力差
3. **样本量不足**：2000样本对于5维特征空间略显不足

## 解决方案

**对WFO全量12,597组合执行真实回测**，获取完整样本分布。

### 优势分析

| 维度 | Top2000方案 | 全量方案 | 提升 |
|-----|------------|---------|------|
| **样本量** | 2,000 | 12,597 | **+6.3倍** |
| **IC覆盖** | 0.10 ~ 0.16 | -0.04 ~ 0.16 | **全范围** |
| **样本密度** | 333样本/IC单位 | 2100样本/IC单位 | **+6.3倍** |
| **泛化能力** | 中等 | 优秀 | **显著提升** |

### 技术实施

#### 1. 修改回测代码支持全量模式

```python
# real_backtest/run_production_backtest.py

def load_top_combos_from_run(run_dir: Path, top_n: int = 100, load_all: bool = False):
    """
    新增load_all参数：
    - load_all=False: 加载TopN组合（原逻辑）
    - load_all=True: 加载all_combos.parquet全量数据
    """
    if load_all:
        df = pd.read_parquet(run_dir / "all_combos.parquet")
        return df.sort_values(['mean_oos_ic', 'stability_score'], ascending=[False, False])
    # ... 原TopN逻辑
```

#### 2. 环境变量控制

新增 `RB_BACKTEST_ALL` 环境变量：

```bash
export RB_BACKTEST_ALL=1          # 启用全量回测
export RB_FORCE_FREQ=8            # 锁定8天频率
export RB_TEST_ALL_FREQS=0        # 禁用多频扫描
```

#### 3. 启动脚本

```bash
# scripts/run_full_backtest_all_combos.sh
bash scripts/run_full_backtest_all_combos.sh

# 预计耗时: ~30分钟
# 并行度: 8核心
# 速度: ~430组合/分钟
```

## 全量GBDT校准器训练

### 模型配置增强

相比Top2000模型，全量模型参数调整：

| 参数 | Top2000模型 | 全量模型 | 原因 |
|-----|------------|---------|------|
| **n_estimators** | 200 | 300 | 样本更多，支持更多树 |
| **max_depth** | 4 | 5 | 捕捉更复杂的非线性关系 |
| **cv_folds** | 5 | 10 | 更稳健的交叉验证 |

### 预期性能提升

基于统计学原理，样本量增加6.3倍带来的理论提升：

1. **标准误差降低**: SE ∝ 1/√n → SE_full / SE_top2000 = √(2000/12597) = 0.40
   - 预测误差降低60%

2. **R²提升**: 更多样本覆盖特征空间边界
   - Top2000: R² = 0.69
   - 全量预期: R² > 0.75

3. **Spearman相关性**: 排序能力增强
   - Top2000: ρ = 0.83
   - 全量预期: ρ > 0.88

4. **Precision@K提升**: 低IC区域预测改善
   - Top2000: P@100 = 50%
   - 全量预期: P@100 > 60%

### 训练脚本

```bash
# 等待全量回测完成后执行
python scripts/train_calibrator_full.py
```

**输出文件**:
- `results/calibrator_gbdt_full.joblib` - 全量训练模型
- `results/all_combos_calibrated_gbdt_full.parquet` - 校准后全量结果
- `results/whitelist_top2000_calibrated_gbdt_full.txt` - 校准Top2000白名单
- `results/calibrator_full_vs_top2000_comparison.png` - 可视化对比

## 样本分布对比

### Top2000样本特征
- **IC分布**: 集中在0.10~0.16高IC区域
- **Sharpe分布**: 偏向高Sharpe组合
- **问题**: 缺少"低IC但高Sharpe"的宝藏组合（如之前发现的#11874）

### 全量样本特征
- **IC分布**: 覆盖-0.04~0.16全范围，均匀分布
- **Sharpe分布**: 包含全部性能范围
- **优势**: 避免样本选择偏差，提升模型鲁棒性

## 预期改进效果

### 1. 遗漏宝藏发现能力提升

**问题**: Top2000模型在训练时未见过低IC组合，导致对低IC区域预测不准

**案例**: 之前#11874组合（IC=0.0228, 实际Sharpe=0.858）被Top2000模型严重低估

**改进**: 全量模型见过完整IC分布，能正确识别"低IC高Sharpe"模式

### 2. 高IC组合过拟合风险降低

**问题**: Top2000模型在高IC区域过度拟合，导致部分高IC组合被高估

**案例**: 之前发现WFO第36名组合（IC=0.134）实际表现差（Sharpe=0.363）

**改进**: 全量样本包含更多高IC失败案例，模型学会识别"高IC低稳定性"陷阱

### 3. 边界区域预测改善

**问题**: Top2000模型对IC边界（0.10左右）预测不确定性大

**改进**: 全量样本在边界区域密度提升，预测更准确

## 实施进度

- [x] 修改`load_top_combos_from_run`支持`load_all`参数
- [x] 添加`RB_BACKTEST_ALL`环境变量支持
- [x] 创建全量回测启动脚本 (`run_full_backtest_all_combos.sh`)
- [x] 创建进度监控脚本 (`monitor_full_backtest.sh`)
- [x] 创建全量校准器训练脚本 (`train_calibrator_full.py`)
- [ ] 等待全量回测完成（预计30分钟，当前进度6.5%）
- [ ] 训练全量GBDT校准器
- [ ] 对比Top2000 vs 全量模型性能
- [ ] 更新生产环境使用全量校准器

## 资源消耗

### 计算资源
- **CPU**: 8核心并行，单核心利用率~80%
- **内存**: ~2GB（因子数据 + 回测状态）
- **时间**: ~30分钟（12597组合 @ 430组/分）

### 存储资源
- **回测CSV**: ~10MB（12597行 × 28列）
- **校准器模型**: ~5MB（GBDT with 300 trees）
- **可视化图表**: ~1MB

### ROI评估
- **时间投入**: 30分钟（一次性）
- **性能提升**: Spearman相关性预期+5%，Precision@100预期+10%
- **长期价值**: 避免样本偏差，模型更稳健，减少后续调试成本

## 后续优化方向

1. **增量学习**: 每次新WFO后追加回测数据，累积训练集
2. **多轮平均**: 收集多轮WFO结果，训练跨轮稳定性模型
3. **市场状态特征**: 添加WFO窗口期市场环境变量（牛熊、波动率）
4. **多目标优化**: 同时预测Sharpe、Calmar、最大回撤，构建帕累托前沿
5. **在线校准**: 实盘运行后用真实收益反馈，持续优化模型

---

**总结**: 全量回测方案通过样本量扩大6.3倍，消除样本选择偏差，预期将校准器性能提升5-10%，为策略部署提供更可靠的排序依据。
