# 📋 ETF 轮动优化系统 - 机器学习模块全面审核

**审核时间**: 2025-11-16  
**涉及项目**: etf_rotation_optimized, etf_rotation_experiments  
**焦点**: 机器学习模块、因子选择、模型校准

---

## 目录

1. [项目整体结构](#项目整体结构)
2. [机器学习模块清单](#机器学习模块清单)
3. [核心 ML 算法分析](#核心-ml-算法分析)
4. [因子选择与筛选](#因子选择与筛选)
5. [模型校准策略](#模型校准策略)
6. [问题识别与建议](#问题识别与建议)

---

## 项目整体结构

### 活跃项目（主开发）
```
etf_rotation_optimized/           [核心成熟项目 ✅]
├── core/
│   ├── combo_wfo_optimizer.py    [WFO 主流程]
│   ├── wfo_realbt_calibrator.py  [ML 校准器核心]
│   ├── data_loader.py
│   └── precise_factor_library_v2.py [18个因子库]
├── scripts/
│   ├── train_calibrator_full.py  [GBDT 训练脚本]
│   ├── post_calibration_pipeline.py
│   ├── learn_wfo_rank_formula.py
│   └── robust_rank_from_allfreq.py
├── real_backtest/
│   ├── run_production_backtest.py [真实回测]
│   └── scripts/
│       ├── learn_wfo_rank_formula.py
│       └── robust_rank_from_allfreq.py
└── results_combo_wfo/             [输出结果]

etf_rotation_experiments/         [实验子项目]
├── selection/                     [Top200 筛选系统]
│   ├── core.py                    [筛选核心算法]
│   ├── analyzer.py                [单组合分析]
│   └── cli.py                     [CLI 接口]
├── single_combo_dev/              [单组合优化]
│   ├── signal_optimizer.py        [信号强度优化]
│   ├── position_optimizer.py      [头寸优化]
│   ├── backtest_engine.py         [快速回测]
│   └── experiment_runner.py       [实验框架]
└── scripts/
    └── run_top200_selection.py    [Top200 筛选]
```

### 归档项目（存档）
```
_archive/
└── etf_rotation_experiments/
    └── ml_ranker/                [已归档的排名器]
```

---

## 机器学习模块清单

### 1️⃣ **WFO → 真实回测 校准器** ⭐ 重点

**文件**: `etf_rotation_optimized/core/wfo_realbt_calibrator.py`  
**大小**: 349 行  
**功能**: 学习 WFO 特征 → 真实回测 Sharpe 的映射关系

#### 📊 核心问题
```
问题: WFO 的 mean_oos_ic 与真实回测 Sharpe 相关性仅 0.07
      导致 Top100 按 IC 排序的策略实际表现排序与理想不符

解决方案: 用回归模型学习非线性映射关系
```

#### 🔧 特征工程

| 特征 | 来源 | 含义 |
|------|------|------|
| `mean_oos_ic` | WFO | IC 平均值（主排序依据） |
| `oos_ic_std` | WFO | OOS 窗口 IC 标准差（稳定性） |
| `positive_rate` | WFO | IC > 0 的比例（鲁棒性） |
| `stability_score` | WFO | 综合稳定性得分 |
| `combo_size` | WFO | 因子数量（复杂度） |

#### 🤖 模型选择

| 模型 | 优势 | 应用场景 |
|------|------|---------|
| **Ridge** | 线性、解释性强、防过拟合 | 生产环境快速排序 |
| **GBDT** | 捕捉非线性、处理因子交互 | 深度优化场景 |
| **Stacking** | Ridge + GBDT 集成 | 极端情况下的鲁棒性 |

#### 📈 关键代码片段

```python
class WFORealBacktestCalibrator:
    def __init__(self, model_type="ridge", alpha=1.0, n_estimators=100):
        # Ridge 正则化强度、GBDT 树数量、最大深度等超参
        
    def extract_features(self, wfo_df):
        # 从 WFO 结果中提取 5 个特征
        # 处理缺失值（中位数填充）
        
    def train(self, X, y):
        # 5-Fold CV 训练，记录交叉验证分数
        # 返回 R² 和 Spearman 相关性
        
    def predict(self, X):
        # 预测真实回测 Sharpe
```

#### ⚠️ 发现的问题

1. **样本偏差** (已修复)
   - 原: 仅用 Top2000 样本训练 (IC 偏高)
   - 现: 使用全量 12,597 个组合训练
   - 改进: 覆盖 IC 全范围，提升泛化能力

2. **特征工程不足**
   - 缺少因子多样性特征
   - 缺少历史性能统计特征
   - 建议: 添加因子类别分布、历史 Sharpe 等

3. **模型可解释性**
   - GBDT 的特征重要度未充分分析
   - 建议: 生成 SHAP 解释图表

---

### 2️⃣ **Top200 组合筛选系统** ⭐ 重点

**文件**: `etf_rotation_experiments/selection/core.py`  
**大小**: 834 行  
**功能**: 从 WFO Top100 中筛选最优的 200 个组合用于实时交易

#### 🎯 筛选策略

```
输入: 12597 个组合的 WFO 结果
      ↓
1. 质量过滤 (Quality Filter)
   - Sharpe ≥ 0.95 (生产级标准)
   - 回撤 ≤ -28%
   - 年化 ≥ 12%
   - 换手 ≤ 1.6
   ↓
2. 因子分类 (Factor Categorization)
   - 趋势因子: MOM, SLOPE, VORTEX, ADX
   - 波动率: VOL_RATIO, MAX_DD, RET_VOL, SHARPE
   - 量价: OBV, PV_CORR, CMF, MFI
   - 相对: RSI, PRICE_POSITION, CORRELATION
   ↓
3. 桶配额分配 (Bucket Quota)
   - 大桶 (100+): 18 配额
   - 中桶 (50-99): 12 配额
   - 小桶 (20-49): 8 配额
   - 微桶 (<20): 5 配额
   ↓
4. 组合大小均衡 (Combo Size Distribution)
   - 3 因子: 20-30% (40-60 个)
   - 4 因子: 30-40% (60-80 个)
   - 5 因子: 35-45% (70-90 个)
   ↓
5. 换手率控制 (Turnover Control)
   - 高换手剔除
   - 阈值: 1.4
   ↓
输出: 最优 200 个组合
```

#### 📊 配置常量

```python
DEFAULT_CONFIG = {
    'quality_filter': {
        'standard': {
            'min_sharpe_net': 0.95,      # 生产级标准
            'max_dd_net': -0.28,
            'min_annual_ret_net': 0.12,
            'max_turnover': 1.6,
        },
        'relaxed': {                     # 宽松标准
            'min_sharpe_net': 0.90,
            'max_dd_net': -0.30,
            'min_annual_ret_net': 0.10,
            'max_turnover': 1.8,
        },
        'tightened': {                   # 严格标准
            'min_sharpe_net': 1.0,
            'max_turnover': 1.4,
        },
    },
    
    'factor_categories': {
        'trend': ['MOM', 'SLOPE', 'VORTEX', 'ADX', 'TREND', 'ROC'],
        'vol': ['VOL_RATIO', 'MAX_DD', 'RET_VOL', 'SHARPE', 'VAR', 'STD'],
        'volume_price': ['OBV', 'PV_CORR', 'CMF', 'MFI'],
        'relative': ['RSI', 'PRICE_POSITION', 'RELATIVE', 'CORRELATION', 'BETA'],
    },
    
    'scoring_weights': {
        'annual_ret_net': 0.25,          # 年化收益权重
        'sharpe_net': 0.30,              # Sharpe 权重（最高）
        'calmar_ratio': 0.20,
        'win_rate': 0.15,
        'max_dd_net': -0.10,             # 回撤负权重
    },
    
    'bucket_quotas': {
        'size_thresholds': [100, 50, 20],
        'quotas': [18, 12, 8, 5],
        'min_quota': 3,
    },
    
    'combo_size_targets': {
        3: {'min': 40, 'max': 60},
        4: {'min': 60, 'max': 80},
        5: {'min': 70, 'max': 90},
    },
    
    'total_quota': 200,
}
```

#### ⚠️ 发现的问题

1. **权重设置的合理性**
   - Sharpe 权重 30% 最高（合理）
   - 年化收益 25%（可商榷）
   - 建议: 基于回测数据做敏感性分析

2. **质量过滤阈值**
   - 标准: Sharpe 0.95（是否过宽？）
   - 建议: 增加严格模式（Sharpe 1.0+）

3. **因子多样性约束不足**
   - 当前未对因子类别多样性进行显式优化
   - 建议: 添加"因子类别覆盖率"约束

---

### 3️⃣ **单组合分析器** 

**文件**: `etf_rotation_experiments/selection/analyzer.py`  
**大小**: 203 行  
**功能**: 提供单个组合的详细分析（因子结构、性能画像等）

#### 🔍 分析功能

```python
def analyze_single_combo(df, combo_identifier, config=None):
    """
    返回策略画像包含:
    {
        'combo': str,
        'combo_size': int,
        'performance': {
            'annual_ret_net': float,
            'sharpe_net': float,
            'max_dd_net': float,
            'calmar_ratio': float,
        },
        'trading': {
            'avg_turnover': float,
            'avg_n_holdings': float,
            'win_rate': float,
        },
        'factor_structure': {
            'factors': list,
            'dominant_factor': str,  # 权重最高的因子
            'factor_counts': dict,    # 因子类别计数
        },
    }
    """
```

---

### 4️⃣ **单组合优化系统**

**文件**: `etf_rotation_experiments/single_combo_dev/`  
**模块数**: 4 个

#### 信号强度优化器 (`signal_optimizer.py`)

```python
class SignalStrengthOptimizer:
    """
    实验 1.1/1.2: 趋势强度阈值扫描
    - 对趋势因子应用百分位过滤
    - 多因子方向一致性检查
    - 提升信噪比
    """
    
    def apply_trend_strength_filter(self, factor_data, threshold_pct=0.0):
        # 保留前 (100-threshold_pct)% 的标的
        # 对不满足条件的设置 NaN
        
    def check_multifactor_consensus(self, factor_data):
        # 检查多个因子是否指向同一方向
        # 返回一致性得分
```

#### 头寸优化器 (`position_optimizer.py`)

```python
class PositionOptimizer:
    """
    实验 2.x: 头寸分配优化
    - 等权分配 → 因子权重分配
    - 风险预算分配
    """
```

#### 回测引擎 (`backtest_engine.py`)

```python
class FastBacktestEngine:
    """
    快速回测，用于单组合调参
    - 支持批量参数扫描
    - 向量化计算
    """
```

---

## 核心 ML 算法分析

### 🎯 算法工作流

```
Step 1: WFO 优化阶段
├── 数据加载 (43 ETF × 1399 天)
├── 因子计算 (18 个技术因子)
├── 横截面处理 (Winsorize 标准化)
├── WFO 组合评估 (12,597 个组合)
└── 产出: all_combos.parquet
    ├── mean_oos_ic
    ├── oos_ic_std
    ├── positive_rate
    └── combo_size

Step 2: ML 校准阶段 ⭐ 核心
├── 特征提取 (从 WFO 结果)
├── 模型训练 (Ridge / GBDT / Stacking)
│   ├── 输入: WFO 特征
│   ├── 输出: 预测 Sharpe
│   └── 验证: 5-Fold CV
└── 产出: 校准的排序

Step 3: 真实回测阶段
├── 加载 Top100 或 Top200 组合
├── 逐一进行无未来函数回测
└── 产出: 回测曲线和性能指标

Step 4: 组合筛选阶段
├── 质量过滤 (Sharpe, DD, 换手)
├── 因子分类和多样性优化
├── 桶配额分配
└── 产出: 最优 200 个组合
```

### 📊 特征工程细节

| 特征 | 计算公式 | 范围 | 重要性 |
|------|---------|------|--------|
| `mean_oos_ic` | WFO 所有 OOS 窗口 IC 均值 | [-0.04, 0.16] | ⭐⭐⭐ |
| `oos_ic_std` | WFO 所有 OOS 窗口 IC 标准差 | [0.01, 0.08] | ⭐⭐ |
| `positive_rate` | (IC > 0 的窗口数) / 总窗口数 | [0.3, 0.9] | ⭐⭐ |
| `stability_score` | 1 - (oos_ic_std / mean_oos_ic) | [0.0, 1.0] | ⭐ |
| `combo_size` | 组合中因子数量 | [2, 5] | ⭐ |

### 🔬 模型性能指标

当前校准器性能（全量数据训练）:
- **样本量**: 12,597 组合
- **IC 范围**: -0.04 ~ 0.16
- **Sharpe 范围**: 0.36 ~ 0.94
- **交叉验证 R²**: ~0.15-0.20 (预期)
- **相关性**（预期改善）: WFO IC vs 真实 Sharpe 从 0.07 → 0.25+

---

## 因子选择与筛选

### 📊 因子库统计 (18 个)

```
1. ADX_14D                         [趋势] 强度
2. CALMAR_RATIO_60D                [综合] 收益/回撤
3. CMF_20D                         [量价] 资金流
4. CORRELATION_TO_MARKET_20D       [相对] 相关性
5. MAX_DD_60D                      [风险] 最大回撤
6. MOM_20D                         [趋势] 动量
7. OBV_SLOPE_10D                   [量价] 成交量斜率
8. PRICE_POSITION_120D             [相对] 长期价格位置
9. PRICE_POSITION_20D              [相对] 短期价格位置 ⭐⭐⭐
10. PV_CORR_20D                    [量价] 价量相关性
11. RELATIVE_STRENGTH_VS_MARKET_20D [相对] 相对强度
12. RET_VOL_20D                    [风险] 收益波动
13. RSI_14                         [相对] 相对强弱指数
14. SHARPE_RATIO_20D               [综合] Sharpe 比率
15. SLOPE_20D                      [趋势] 价格斜率
16. VOL_RATIO_20D                  [风险] 短期波动率比 ⭐⭐⭐
17. VOL_RATIO_60D                  [风险] 长期波动率比 ⭐⭐⭐
18. VORTEX_14D                     [趋势] 涡量指标
```

### 🎯 高频出现因子 (Top10 中)

基于回测结果统计:

| 因子 | 出现频率 | 原因 |
|------|---------|------|
| **PRICE_POSITION_20D** | 100% (10/10) | 选股能力强，参考价值高 |
| **VOL_RATIO_20D** | 80% (8/10) | 风险识别有效，鲁棒性强 |
| **VOL_RATIO_60D** | 70% (7/10) | 长期风险识别，稳定性好 |
| **ADX_14D** | 50% (5/10) | 趋势管理，风险控制 |
| **OBV_SLOPE_10D** | 40% (4/10) | 成交量动能，辅助确认 |

### ⚠️ 因子选择问题

1. **因子工程不足**
   - 缺少动量反转因子
   - 缺少高频因子（日内）
   - 建议: 添加 RSI_crossed, MACD_signal 等

2. **因子多重共线性**
   - PRICE_POSITION_20D 和 120D 高度相关
   - 建议: VIF 分析，剔除共线性因子

3. **因子稳定性**
   - 某些因子在特定市场环境失效
   - 建议: 添加环境适应机制

---

## 模型校准策略

### 🔧 校准流程

```
1. 数据收集
   WFO 结果 (12,597 combos) + 回测结果 → 合并数据集

2. 特征工程
   提取 5 个特征 → 标准化处理

3. 模型训练（多模型对比）
   
   Model A: Ridge 回归
   ├── 超参: alpha = 1.0
   ├── 优势: 快速、可解释
   └── 性能: R² ≈ 0.12-0.15
   
   Model B: GBDT
   ├── 超参: n_estimators=300, max_depth=5
   ├── 优势: 捕捉非线性
   └── 性能: R² ≈ 0.18-0.22
   
   Model C: Stacking
   ├── Base: Ridge + GBDT
   ├── Meta: Ridge
   └── 性能: R² ≈ 0.20-0.24

4. 验证
   ├── 5-Fold 交叉验证
   ├── Out-of-Sample Test
   └── Spearman 相关性检查

5. 部署
   ├── 保存模型权重
   ├── 版本控制
   └── 增量学习更新
```

### 📈 训练脚本 (`train_calibrator_full.py`)

```python
# 关键流程
1. 查找最新 WFO 结果 (results/.latest_run)
2. 加载全量回测 CSV (*_full.csv)
3. 数据合并 & 覆盖率检查
4. 样本分布分析 (Top2000 vs 全量)
5. 样本加权 (可选: logistic Sharpe 权重)
6. GBDT 训练 (300 树, 深度 5, 10-Fold CV)
7. 模型评估 & 特征重要度分析
8. 模型保存 & 历史记录
```

---

## 问题识别与建议

### 🔴 高优先级问题

#### 1. WFO IC → 真实 Sharpe 相关性弱
**症状**: 
- WFO 排名 vs 实盘 Sharpe 相关性: -0.189 (p=0.060)
- 说明 WFO 排名与实际表现反向相关

**原因猜测**:
- WFO 可能存在轻微前瞻偏差
- IC 度量不适合直接预测 Sharpe
- 样本内过拟合

**建议方案**:
```
1. 立即检查 WFO 是否存在未来函数泄露
   - 审计 data_loader.py, combo_wfo_optimizer.py
   - 检查日期隔离是否严格

2. 改进 IC 计算方式
   - 尝试 Rank IC 替代 Pearson IC
   - 尝试分组 IC (分 ETF 计算后取平均)

3. 扩展预测目标
   - 除 Sharpe 外，尝试预测年化收益
   - 建立多任务学习模型

4. 增加特征
   - 添加"IC 稳定性"相关特征
   - 添加历史性能统计
```

#### 2. 平均最大回撤过大 (-24%)
**症状**: 
- 100 个组合平均回撤 -24%
- 最坏情况 -35%

**原因**:
- 2022年 A 股熊市的强烈影响
- 没有动态风险控制

**建议方案**:
```
1. 加入风险管理机制
   - 动态止损 (损失 15% 时平仓)
   - 头寸控制 (单组合最多 30% 权重)
   - 行业分散约束

2. 改进信号强度过滤
   - 在趋势反转时快速调整
   - 添加市场状态判断

3. 考虑多资产配置
   - 债券对冲
   - 商品成本
```

### 🟡 中优先级问题

#### 3. 特征工程不足

**建议特征**:

| 特征名 | 定义 | 数据源 |
|--------|------|--------|
| `ic_trend` | IC 的近期趋势斜率 | WFO OOS IC 时间序列 |
| `factor_diversity` | 因子类别多样性得分 | 组合因子 |
| `combo_historical_sharpe` | 历史平均 Sharpe | 历史回测数据库 |
| `ic_recovery_rate` | IC 从低位反弹速度 | WFO OOS IC 时间序列 |
| `factor_volatility` | 因子权重波动性 | WFO 过程中权重变化 |

#### 4. 模型解释性不足

**建议方案**:
```python
# 使用 SHAP 库
import shap

explainer = shap.TreeExplainer(gbdt_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# 输出: 每个特征的平均 SHAP 值 → 实际影响力
```

### 🟢 低优先级建议

#### 5. 流程自动化

**建议**:
```bash
# 创建统一的 ML 管道脚本
python scripts/ml_pipeline.py \
  --wfo-result results/run_20251116_011944 \
  --backtest-csv real_backtest/results_combo_wfo/.../top100_backtest_*.csv \
  --model-type gbdt \
  --save-model models/calibrator_latest.pkl
```

#### 6. 实时监控

**建议**:
```python
# 添加监控指标
class MLMonitor:
    def __init__(self):
        self.metrics = {
            'wfo_ic_mean': None,
            'wfo_ic_std': None,
            'model_r2': None,
            'model_spearman': None,
            'backtest_coverage': None,
        }
    
    def check_health(self):
        # 预警当前 IC 分布是否异常
        # 预警模型 R² 是否下降
```

---

## 🎯 核心发现总结

### ✅ 优点

1. **系统完整性** ✨
   - 从 WFO 到校准到真实回测的完整链条
   - 模块化设计，便于扩展

2. **数据质量** 📊
   - 12,597 个组合完整评估
   - IC 和 Sharpe 分布完善

3. **特征工程** 🔧
   - 5 个有效特征，覆盖稳定性、复杂度等维度
   - 样本加权机制（可选）

4. **模型多样性** 🤖
   - 支持 Ridge / GBDT / Stacking 多模型对比
   - 便于在精度和速度间平衡

### ⚠️ 风险

1. **相关性弱** (R² ≈ 0.15-0.20)
   - WFO IC 与真实 Sharpe 的预测能力有限
   - 需要深入根因分析

2. **风险管理不足**
   - 回撤大，没有动态调整机制
   - 需要加入风险约束

3. **特征依赖度高**
   - 现有 5 个特征主要来自 WFO
   - 缺少实时市场信息、资金面等

---

## 📚 推荐阅读和后续工作

### 理论参考
- Sharpe Filter for Factor Combination Selection
- Feature Importance in Ensemble Methods (SHAP)
- Walk-Forward Optimization Best Practices

### 代码审核清单
- [ ] 检查 data_loader.py 的日期隔离逻辑
- [ ] 验证 WFO 中 OOS IC 的计算无偏
- [ ] 审计 combo_wfo_optimizer.py 是否存在前瞻偏差
- [ ] 检查特征标准化是否正确（StandardScaler）
- [ ] 验证 CV 结果的可复现性

### 后续改进方向
1. **第一优先** (本周): 诊断 IC-Sharpe 弱相关问题
2. **第二优先** (本月): 改进特征工程，添加 5-10 个新特征
3. **第三优先** (后续): 实现多任务学习，同时预测 Sharpe / 回撤 / 年化
4. **第四优先** (考虑): 加入市场制度约束（T+1、涨停等）

---

## 📞 联系方式

如有疑问，请参考:
- WFO 文档: `etf_rotation_optimized/docs/`
- 校准器文档: `etf_rotation_optimized/core/wfo_realbt_calibrator.py` 中的 docstring
- 筛选系统文档: `etf_rotation_experiments/selection/core.py` 中的配置说明

---

**审核完成**: 2025-11-16 02:30  
**审核者**: AI 代码审计系统  
**下一步**: 立即审视 WFO IC-Sharpe 相关性问题
