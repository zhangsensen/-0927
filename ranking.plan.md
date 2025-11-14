# 排序校准器盈利优化计划

## 核心目标
让排序模型真正提升Top1000策略的实际盈利能力，不追求胜率，追求盈利覆盖亏损。

## 当前问题诊断
- **模型表现**: Top1000重叠率77.9%，但真实回测平均年化收益下降（baseline 19.17% → blend 0.30 limited 18.14%）
- **数据规模**: 仅3次WFO run（37,791样本），可能不足
- **目标函数**: 当前加权目标（年化50% + Sharpe30% + Calmar20%）未对齐真实盈利
- **特征缺失**: 缺乏市场环境适应性、组合构成、历史一致性、极端风险等关键特征

---

## 阶段1: 深度特征工程

### 1.1 市场环境适应性特征
**文件**: `scripts/build_rank_dataset.py`
**新增函数**: `add_market_regime_features(df, etf_prices_path) -> pd.DataFrame`

实现内容:
- **牛熊市表现差异**:
  - 加载ETF价格数据（`data/etf_prices_template.csv`）
  - 从`oos_ic_list`时间窗口识别上涨/下跌期（基于市场指数MA20/MA60交叉）
  - 计算特征: `ic_bull_mean`, `ic_bear_mean`, `ic_bull_bear_ratio`, `ic_bull_bear_diff`
  
- **波动率regime**:
  - 计算WFO期间市场波动率（20日滚动标准差）的分位数
  - 划分高/中/低波动环境（按33%/67%分位数）
  - 提取特征: `ic_high_vol_mean`, `ic_low_vol_mean`, `ic_vol_regime_ratio`, `vol_regime_stability`
  
- **市场相关性**:
  - 计算策略收益与市场指数的相关性
  - 特征: `market_corr`, `market_beta`, `alpha_vs_market`

### 1.2 组合构成特征
**文件**: `scripts/build_rank_dataset.py`
**新增函数**: `add_combo_composition_features(df) -> pd.DataFrame`

实现内容:
- **因子类型解析**:
  - 从`combo`字符串解析因子名称（如"MAX_DD_60D + RSI_14"）
  - 定义因子类别映射:
    - Momentum: RSI, MACD, MOM, ROC
    - Volatility: ATR, BBANDS, VOL_RATIO, MAX_DD
    - Trend: ADX, VORTEX, AROON, DMI
    - Volume: CMF, OBV, MFI, PV_CORR
  - 统计特征: `n_momentum_factors`, `n_volatility_factors`, `n_trend_factors`, `n_volume_factors`
  - 计算`factor_diversity_score` = 不同类别数 / combo_size
  
- **频率分布特征**:
  - 从`best_freq_list`提取:
    - `freq_consistency`: 1 - (unique_freqs / total_periods)
    - `freq_mode_pct`: 主频率占比
    - `freq_entropy`: 频率分布的信息熵
  
- **因子周期特征**:
  - 从因子名称提取周期参数（如RSI_14中的14）
  - 计算: `avg_factor_period`, `period_std`, `period_range`

### 1.3 历史表现一致性特征
**文件**: `scripts/build_rank_dataset.py`
**增强函数**: `enrich_wfo_features(df) -> pd.DataFrame`

新增特征:
- **IC/IR时序稳定性**:
  - `ic_monotonicity`: IC序列的Spearman自相关（滞后1期）
  - `ic_reversals`: IC符号反转次数 / 总期数
  - `ic_positive_streak`: 最长连续正IC期数
  - `ir_stability_score`: IR变异系数的倒数
  
- **趋势方向一致性**:
  - `ic_trend_strength`: IC序列线性回归的R²
  - `ic_trend_pvalue`: 趋势显著性p-value
  - `last_vs_first_ic_ratio`: 最后3期IC均值 / 最初3期IC均值
  - `ic_acceleration`: IC序列的二阶差分均值

### 1.4 极端风险指标特征
**文件**: `scripts/build_rank_dataset.py`
**新增函数**: `add_extreme_risk_features(df) -> pd.DataFrame`

实现内容:
- **回撤风险**:
  - `max_dd_duration`: 最大回撤持续天数（从真实回测提取，如无则用`max_dd_net`代理）
  - `dd_recovery_ratio`: 回撤恢复速度（总收益 / 最大回撤）
  - `dd_frequency`: 回撤>5%的次数估计
  
- **连续亏损风险**:
  - `max_consecutive_loss_days`: 最大连续亏损天数（已有`max_consecutive_losses`）
  - `avg_consecutive_loss`: 平均连续亏损期长度
  - `loss_clustering_score`: 亏损日的聚集程度（基于`losing_days`分布）
  
- **尾部风险**:
  - `downside_deviation`: 下行标准差（仅负收益日）
  - `sortino_ratio_derived`: 年化收益 / downside_deviation
  - `tail_ratio`: 估计的95分位收益 / 5分位损失（基于`avg_win`/`avg_loss`）
  - `return_skewness`: 收益分布偏度估计
  - `return_kurtosis`: 收益分布峰度估计

### 1.5 成本敏感性特征（增强）
**文件**: `scripts/build_rank_dataset.py`
**增强函数**: `add_real_derived_features(df) -> pd.DataFrame`

新增特征:
- `cost_drag`: (annual_ret - annual_ret_net) / annual_ret（成本拖累比例）
- `turnover_efficiency_rank`: ret_turnover_eff在run内的分位数排名
- `breakeven_turnover_est`: annual_ret_net / commission_rate（盈亏平衡换手率估计）

---

## 阶段2: 多目标分层排序模型

### 2.1 两阶段模型架构
**新增文件**: `scripts/train_two_stage_ranker.py`

实现逻辑:
1. **Stage 1: Sharpe筛选器**
   - 目标: 预测`sharpe_net_z`（标准化Sharpe）
   - 用途: 筛选出Sharpe > 阈值（如中位数或0.8）的候选集
   - 模型配置:
     ```python
     LGBMRanker(
         objective="lambdarank",
         learning_rate=0.03,
         n_estimators=400,
         num_leaves=31,
         min_data_in_leaf=100,
         lambda_l1=0.2,
         lambda_l2=0.2,
     )
     ```
   - 输出: `results/models/calibrator_sharpe_filter.txt`

2. **Stage 2: 年化收益排序器**
   - 目标: 在Sharpe筛选后的样本上预测`annual_ret_net`
   - 用途: 在风险可控前提下最大化收益
   - 模型配置:
     ```python
     LGBMRanker(
         objective="lambdarank",
         learning_rate=0.05,
         n_estimators=600,
         num_leaves=63,
         min_data_in_leaf=50,
         lambda_l1=0.1,
         lambda_l2=0.1,
     )
     ```
   - 输出: `results/models/calibrator_profit_ranker.txt`

### 2.2 训练流程
```bash
# 训练两阶段模型
python scripts/train_two_stage_ranker.py \
  --dataset data/calibrator_dataset.parquet \
  --sharpe-threshold 0.0 \
  --output-dir results/models
```

参数说明:
- `--sharpe-threshold`: Sharpe筛选阈值（默认0.0，即中位数）
- `--holdout-run`: 留作最终测试的run_ts
- `--cv-folds`: 交叉验证折数（默认5）

输出:
- `calibrator_sharpe_filter.txt` + `_metrics.json` + `_importance.json`
- `calibrator_profit_ranker.txt` + `_metrics.json` + `_importance.json`
- `two_stage_training_report.json`: 包含两阶段的联合评估指标

---

## 阶段3: 安全替换策略

### 3.1 安全替换逻辑
**修改文件**: `scripts/apply_rank_calibrator.py`
**新增函数**: `apply_safe_replacement(baseline_df, ml_df, config) -> pd.DataFrame`

实现内容:
- **两阶段推理**:
  1. 加载Sharpe筛选器，对所有组合打分
  2. 筛选出`sharpe_score > threshold`的组合
  3. 加载收益排序器，对筛选后的组合打分
  4. 未通过筛选的组合排在最后（按baseline顺序）

- **显著性检验**:
  - 计算ML预测与baseline预测的差值: `delta = ml_score - baseline_score`
  - 仅当`delta > confidence_threshold`时才考虑替换
  - `confidence_threshold`按TopK动态调整:
    - Top100: 3% (严格)
    - Top500: 2% (中等)
    - Top1000: 1% (宽松)

- **渐进式替换**:
  - 不再使用固定20%限幅
  - 动态计算每个TopK的最大替换数:
    ```python
    max_replacements = {
        100: min(10, int(100 * 0.10)),   # 最多10个
        500: min(75, int(500 * 0.15)),   # 最多75个
        1000: min(200, int(1000 * 0.20)), # 最多200个
    }
    ```
  - 优先替换`delta`最大的组合

### 3.2 新增参数
```bash
python scripts/apply_rank_calibrator.py \
  --run-ts 20251111_145454 \
  --sharpe-model results/models/calibrator_sharpe_filter.txt \
  --profit-model results/models/calibrator_profit_ranker.txt \
  --safe-mode \
  --confidence-thresholds "100:0.03,500:0.02,1000:0.01" \
  --max-replacement-pct 0.15
```

参数说明:
- `--sharpe-model`: Sharpe筛选器模型路径
- `--profit-model`: 收益排序器模型路径
- `--safe-mode`: 启用安全替换（默认False）
- `--confidence-thresholds`: TopK对应的置信度阈值
- `--max-replacement-pct`: 最大替换比例（默认0.15）

输出:
- `ranking_two_stage_safe.parquet`: 安全替换后的排名
- `ranking_two_stage_unlimited.parquet`: 无限制的两阶段排名
- `safe_replacement_report.json`: 替换详情（每个TopK的实际替换数、被替换组合列表）

---

## 阶段4: 数据扩充（并行进行）

### 4.1 批量回测脚本
**新增文件**: `scripts/batch_backtest.py`

实现内容:
```python
def batch_backtest(run_ts, topk_start, topk_end, batch_size=500):
    """
    分批运行真实回测，避免内存溢出
    """
    ranking_file = f"results/run_{run_ts}/all_combos.parquet"
    df = pd.read_parquet(ranking_file)
    
    for start in range(topk_start, topk_end, batch_size):
        end = min(start + batch_size, topk_end)
        print(f"回测 Top{start}-{end}...")
        
        # 提取子集
        subset = df.iloc[start:end]
        subset_file = f"results/run_{run_ts}/subset_{start}_{end}.parquet"
        subset.to_parquet(subset_file)
        
        # 运行回测
        cmd = f"python real_backtest/run_profit_backtest.py --topk {end-start} --ranking-file {subset_file} --slippage-bps 0"
        subprocess.run(cmd, shell=True, check=True)
        
        # 清理临时文件
        os.remove(subset_file)
```

使用示例:
```bash
# 后台运行Top2000回测
nohup python scripts/batch_backtest.py \
  --run-ts 20251111_145454 \
  --topk-start 1000 \
  --topk-end 2000 \
  --batch-size 500 \
  > logs/batch_backtest_2000.log 2>&1 &

# 后台运行Top4000回测
nohup python scripts/batch_backtest.py \
  --run-ts 20251111_145454 \
  --topk-start 2000 \
  --topk-end 4000 \
  --batch-size 500 \
  > logs/batch_backtest_4000.log 2>&1 &
```

### 4.2 增量训练
每完成一批回测后:
```bash
# 重新生成数据集（包含新回测结果）
python scripts/build_rank_dataset.py --run-ts 20251111_145454

# 重新训练两阶段模型
python scripts/train_two_stage_ranker.py --dataset data/calibrator_dataset.parquet

# 评估新模型
python scripts/evaluate_rank_calibrator.py --run-ts 20251111_145454 --topk 1000,2000
```

---

## 阶段5: 模型验证与迭代

### 5.1 验证标准（成功定义）
**必须满足以下至少一项**:
1. Top1000平均年化净收益 > baseline + 1%（即 > 20.17%）
2. Top100平均年化净收益 > baseline Top100 + 2%
3. Top10中至少5个来自ML推荐且收益 > 20%
4. 整体Sharpe提升 + 最大回撤降低（Sharpe > 1.05 且 max_dd < baseline）

### 5.2 A/B测试框架
**新增文件**: `scripts/evaluate_ab_test.py`

实现内容:
```python
def compare_models(run_ts, model_configs, topk_thresholds):
    """
    对比多个模型版本的表现
    
    model_configs = [
        {"name": "baseline", "ranking_file": "ranking_baseline.parquet"},
        {"name": "single_target", "ranking_file": "ranking_blend_0.30.parquet"},
        {"name": "two_stage", "ranking_file": "ranking_two_stage_safe.parquet"},
    ]
    """
    results = {}
    for config in model_configs:
        # 运行真实回测
        backtest_results = run_backtest(run_ts, config["ranking_file"], topk=max(topk_thresholds))
        
        # 计算各TopK的指标
        for k in topk_thresholds:
            topk_df = backtest_results.head(k)
            results[f"{config['name']}_top{k}"] = {
                "mean_annual_ret": topk_df["annual_ret_net"].mean(),
                "median_annual_ret": topk_df["annual_ret_net"].median(),
                "mean_sharpe": topk_df["sharpe_net"].mean(),
                "mean_max_dd": topk_df["max_dd_net"].mean(),
                "top10_mean_annual": topk_df.head(10)["annual_ret_net"].mean(),
                "profitable_ratio": (topk_df["annual_ret_net"] > 0).mean(),
            }
    
    # 生成对比报告
    report = generate_comparison_report(results)
    return report
```

输出报告格式:
```markdown
# A/B测试报告

## Top1000对比
| 模型 | 平均年化 | 中位数年化 | 平均Sharpe | 平均最大回撤 | Top10均值 | 盈利比例 |
|------|---------|-----------|-----------|------------|----------|---------|
| baseline | 19.17% | 19.21% | 0.972 | -20.95% | 22.5% | 95.2% |
| single_target | 18.14% | 19.02% | 0.921 | -21.61% | 21.8% | 94.8% |
| two_stage_safe | 20.35% | 20.12% | 1.025 | -19.87% | 24.2% | 96.1% |

## 结论
两阶段安全替换模型在Top1000上实现+1.18%年化收益提升，满足验证标准1。
```

### 5.3 失败快速止损
如果某个方向迭代3次后仍无改善:
1. 记录失败原因到`docs/FAILED_EXPERIMENTS.md`:
   ```markdown
   ## 实验: 单目标年化收益排序 (2025-11-11)
   - 假设: 直接优化年化收益比加权目标更有效
   - 结果: Top1000平均年化反而下降0.5%
   - 原因分析: 模型过度追求高收益，忽略风险，导致高波动策略排名靠前
   - 结论: 必须保留Sharpe约束
   ```

2. 回退到上一个有效版本:
   ```bash
   git checkout <last_working_commit>
   ```

3. 尝试其他方向（如调整特征选择、模型超参数、目标函数权重）

---

## 实施时间线

### 第1天（立即开始）
- [x] 阶段1.1: 市场环境适应性特征（2小时）
- [x] 阶段1.2: 组合构成特征（1.5小时）
- [x] 阶段1.3: 历史一致性特征（1小时）
- [x] 阶段1.4: 极端风险特征（1.5小时）
- [x] 重新生成数据集，启动Top2000后台回测（0.5小时 + 后台1.5小时）

### 第2天
- [ ] 阶段2: 两阶段排序模型训练（2小时）
- [ ] 阶段3: 安全替换策略实现（2小时）
- [ ] 完整评估（baseline vs 新模型），生成报告（1小时）

### 第3天（如需要）
- [ ] 基于Top2000/4000数据重训模型（1小时）
- [ ] A/B测试，选出最优配置（2小时）
- [ ] 最终验证，输出生产级模型（1小时）

---

## 关键文件清单

### 新增文件
1. `scripts/train_two_stage_ranker.py` - 两阶段排序模型训练
2. `scripts/batch_backtest.py` - 批量回测脚本
3. `scripts/evaluate_ab_test.py` - A/B测试评估框架
4. `docs/FAILED_EXPERIMENTS.md` - 失败实验记录

### 修改文件
1. `scripts/build_rank_dataset.py`:
   - 新增`add_market_regime_features()`
   - 新增`add_combo_composition_features()`
   - 新增`add_extreme_risk_features()`
   - 增强`enrich_wfo_features()`
   
2. `scripts/apply_rank_calibrator.py`:
   - 支持两阶段模型推理
   - 实现`apply_safe_replacement()`逻辑
   - 新增`--safe-mode`、`--confidence-thresholds`参数

3. `scripts/evaluate_rank_calibrator.py`:
   - 扩展评估指标（Top10详情、风险指标）
   - 支持多模型对比

---

## 风险控制

1. **过拟合风险**: 
   - 严格使用GroupKFold + Holdout验证
   - 特征数控制在<100（当前151个，需精简）
   - 正则化参数`lambda_l1=0.2`, `lambda_l2=0.2`

2. **数据泄露风险**:
   - 确保所有新特征仅使用WFO期内数据
   - 市场环境特征必须基于历史窗口，不能用未来信息

3. **计算资源**:
   - Top2000/4000回测预计占用~8GB内存
   - 如内存不足，分批执行（每批500组合）

---

## 预期收益

保守估计，如果优化成功:
- **Top1000平均年化**: 19.17% → 20.5%（+1.3%）
- **Top100平均年化**: ~22% → 24%（+2%）
- **Top10最优策略**: 23.6% → 26%（+2.4%）
- **Sharpe提升**: 0.97 → 1.05（+8%）

---

## To-dos

### 阶段1: 深度特征工程
- [ ] 实现`add_market_regime_features()` - 市场环境适应性
- [ ] 实现`add_combo_composition_features()` - 组合构成特征
- [ ] 增强`enrich_wfo_features()` - 历史一致性特征
- [ ] 实现`add_extreme_risk_features()` - 极端风险指标
- [ ] 重新生成`data/calibrator_dataset.parquet`
- [ ] 运行特征诊断，验证新特征有效性

### 阶段2: 两阶段模型
- [ ] 创建`train_two_stage_ranker.py`
- [ ] 训练Sharpe筛选器模型
- [ ] 训练年化收益排序器模型
- [ ] 评估两阶段模型的CV/Holdout指标

### 阶段3: 安全替换
- [ ] 修改`apply_rank_calibrator.py`支持两阶段推理
- [ ] 实现`apply_safe_replacement()`函数
- [ ] 生成安全替换排名文件
- [ ] 对比安全替换 vs baseline的真实回测表现

### 阶段4: 数据扩充
- [ ] 创建`batch_backtest.py`批量回测脚本
- [ ] 启动Top2000后台回测
- [ ] 启动Top4000后台回测
- [ ] 基于扩充数据重训模型

### 阶段5: 验证迭代
- [ ] 创建`evaluate_ab_test.py` A/B测试框架
- [ ] 运行完整A/B测试
- [ ] 生成最终评估报告
- [ ] 选出最优模型配置并部署

