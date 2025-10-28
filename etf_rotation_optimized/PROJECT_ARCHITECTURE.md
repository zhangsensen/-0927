# ETF轮动系统 - 项目架构与核心机制

**版本**: v2.0-optimized  
**最后更新**: 2025-10-27  
**目标受众**: 大模型、开发者、量化研究员

---

## 📋 快速导航

- [项目概览](#项目概览) - 系统定位和规模
- [核心架构](#核心架构) - 整体设计
- [数据流](#数据流) - 数据处理流程
- [核心机制](#核心机制) - 6大核心机制
- [模块详解](#模块详解) - 各模块职责
- [工作流程](#工作流程) - 完整执行流程
- [配置系统](#配置系统) - 参数配置
- [性能优化](#性能优化) - 优化策略

---

## 项目概览

### 🎯 项目定位

**ETF轮动系统优化版** 是一个精简、高效、生产级的量化交易系统。

**核心目标**:
- 通过精确因子计算识别高收益ETF
- 使用前向回测验证策略稳健性
- 生成实时交易信号
- 支持模拟和实盘交易

### 📊 系统规模

| 指标 | 数值 |
|------|------|
| 代码行数 | ~1200 |
| 核心模块 | 7个 |
| 因子库 | 12个精选因子 |
| 支持标的 | 43个港股ETF |
| 执行速度 | 5年数据WFO < 30秒 |
| 内存占用 | < 500MB |

### 🏗️ 设计哲学

遵循 **Linus量化工程哲学**:
- ✅ 无冗余代码 - 每行代码都有明确用途
- ✅ 实战优先 - 为赚钱而设计
- ✅ 数据驱动 - 真实数据，真实信号
- ✅ 风控第一 - 稳健性优于收益率

---

## 核心架构

### 🏛️ 系统架构

```
数据输入 → 数据验证 → 因子计算 → 横截面处理 
  ↓
因子筛选 → 前向回测 → 信号生成 → 交易执行
```

### 📦 模块组织

```
core/                    # 核心计算模块
├── constants.py        # 全局常量
├── data_validator.py   # 数据验证
├── precise_factor_library_v2.py # 12个精选因子
├── cross_section_processor.py   # 横截面处理
├── ic_calculator.py    # IC计算
├── factor_selector.py  # 因子选择
└── walk_forward_optimizer.py    # WFO框架

scripts/                # 工作流脚本
├── step1_cross_section.py      # Step 1
├── step2_factor_selection.py   # Step 2
├── step3_run_wfo.py            # Step 3
└── run_all_steps.py            # 完整流程

configs/                # 配置文件
├── config.yaml         # 主配置
└── wfo_grid.yaml       # WFO参数

tests/                  # 测试模块
docs/                   # 文档
utils/                  # 工具模块
```

---

## 数据流

### 🔄 完整数据流

```
原始OHLCV数据 (43个港股ETF)
  ↓
[数据验证]
  - 满窗NaN检查
  - 覆盖率验证 (97%)
  - 成交量异常检测
  ↓
[因子计算]
  - 12个精选因子
  - 26个全量因子 (可选)
  ↓
[横截面处理]
  - 按日期分组
  - Z-score标准化
  - Winsorize极值截断
  ↓
[因子筛选]
  - IC/IR计算
  - t-test显著性检验
  - FDR多重检验校正
  - 相关性过滤
  ↓
[前向回测]
  - 滑动窗口划分 (IS/OOS)
  - 样本内因子筛选
  - 样本外性能验证
  ↓
[信号生成]
  - 实时因子计算
  - 排序和选择
  - 信号输出
  ↓
[交易执行]
  - 模拟交易
  - 实盘交易 (可选)
```

---

## 核心机制

### 1️⃣ 数据验证机制

**职责**: 确保数据质量满足精确因子计算要求

**验证规则**:

| 检查项 | 规则 | 处理 |
|--------|------|------|
| 满窗NaN | 连续NaN < window_days | 失败则标记无效 |
| 覆盖率 | 覆盖率 ≥ 97% | 不足则过滤标的 |
| 成交量异常 | Z-score > 3.0 | 标记但不删除 |

**关键参数**:
```yaml
coverage_threshold: 0.97      # 97%覆盖率
min_window_days: 20           # 最小窗口
volume_anomaly_z_score: 3.0   # 异常阈值
```

### 2️⃣ 因子计算机制

**12个精选因子**:

| 维度 | 因子 | 窗口 | 有界 |
|------|------|------|------|
| 趋势 | MOM_20D, SLOPE_20D | 20 | ✗ |
| 价格位置 | PRICE_POSITION_20D/120D | 20/120 | ✓ |
| 波动率 | RET_VOL_20D, MAX_DD_60D | 20/60 | ✗ |
| 成交量 | VOL_RATIO_20D/60D | 20/60 | ✗ |
| 价量耦合 | PV_CORR_20D | 20 | ✓ |
| 反转 | RSI_14 | 14 | ✓ |

**缺失值处理**:
- 原始缺失 → 保留NaN
- 满窗不足 → NaN
- 标准化时 → 使用有效数据

**极值处理**:
- 非有界因子: Winsorize 2.5%-97.5%
- 有界因子: 跳过极值截断

### 3️⃣ 横截面处理机制

**目标**: 消除时间序列偏差，实现公平比较

**处理流程**:

```
每日因子数据 (43个ETF)
  ↓
[按日期分组]
  每个交易日单独处理
  ↓
[Z-score标准化]
  Z = (x - mean) / std
  如果std=0，则Z=0
  ↓
[Winsorize极值截断]
  下界: 2.5%分位
  上界: 97.5%分位
  有界因子跳过
  ↓
标准化因子数据
```

### 4️⃣ IC计算机制

**目标**: 衡量因子与未来收益的预测能力

**计算方法**:

```
标准化因子 + 收益率
  ↓
按日期计算相关系数
  - Pearson: 线性关系
  - Spearman: 单调关系
  - Kendall: 秩相关
  ↓
生成IC时间序列
  每个交易日一个IC值
  ↓
计算IC统计量
  - 均值、标准差、IR
  - t-stat、p-value
  - Sharpe比率
  ↓
IC报告
```

**关键指标**:
```yaml
IC_MIN_OBSERVATIONS: 20        # 最小观察数
IC_SIGNIFICANCE_LEVEL: 0.05    # 显著性水平
TRADING_DAYS_PER_YEAR: 252     # 年化系数
```

### 5️⃣ 因子筛选机制

**目标**: 从12个因子中选出最优因子组合

**筛选流程**:

```
12个候选因子
  ↓
[IC/IR筛选]
  规则: IC > 0.01 AND IR > 0.05
  ↓
[显著性检验]
  方法: t-test
  规则: p-value < 0.05
  ↓
[FDR校正]
  方法: Benjamini-Hochberg
  规则: 调整p-value < 0.1
  ↓
[相关性过滤]
  规则: 因子间相关系数 < 0.7
  ↓
最终因子集合
```

**关键参数**:
```yaml
min_ic: 0.01              # 最小IC
min_ir: 0.05              # 最小IR
max_correlation: 0.7      # 最大相关性
use_fdr: true             # FDR校正
fdr_alpha: 0.1            # FDR alpha
```

### 6️⃣ 前向回测机制

**目标**: 验证策略在样本外的稳健性

**WFO流程**:

```
完整历史数据 (2年)
  ↓
[窗口划分]
  IS_WINDOW: 252天 (1年)
  OOS_WINDOW: 60天 (3个月)
  STEP: 20天 (月度调仓)
  ↓
[逐窗口处理]
  Window 1: IS[0:252] → 筛选 | OOS[252:312] → 验证
  Window 2: IS[20:272] → 筛选 | OOS[272:332] → 验证
  ... (重复直到数据结束)
  ↓
[汇总前向性能]
  - 平均OOS Sharpe
  - 平均OOS IC
  - 因子选中频率
  - 过拟合比 = IS Sharpe / OOS Sharpe
  ↓
WFO报告
```

**关键参数**:
```yaml
DEFAULT_IS_WINDOW: 252      # 1年
DEFAULT_OOS_WINDOW: 60      # 3个月
DEFAULT_STEP: 20            # 1个月
DEFAULT_IC_THRESHOLD: 0.05  # IC阈值
```

---

## 模块详解

### 📦 core/constants.py

**职责**: 消除所有魔数，统一常量定义

**关键常量分类**:

| 类别 | 常量 | 值 |
|------|------|-----|
| 数值精度 | EPSILON | 1e-10 |
| 数据质量 | DEFAULT_COVERAGE_THRESHOLD | 0.97 |
| 因子参数 | MOM_PERIODS | [5, 20, 60] |
| 标准化 | WINSORIZE_LOWER_PCT | 2.5% |
| 回测 | INIT_CASH | 100000 |
| 成本 | COMMISSION_RATE | 0.2% |
| 风控 | MAX_POSITION | 0.3 |

### 📦 core/data_validator.py

**职责**: 数据质量验证

**主要方法**:
- `check_full_window()` - 满窗NaN检查
- `check_coverage()` - 覆盖率验证
- `detect_volume_anomalies()` - 成交量异常检测
- `validate_all()` - 完整验证

### 📦 core/precise_factor_library_v2.py

**职责**: 12个精选因子的计算

**主要方法**:
- `compute_all_factors()` - 计算所有因子
- `compute_single_factor()` - 计算单个因子
- `get_factor_metadata()` - 获取因子元数据

### 📦 core/cross_section_processor.py

**职责**: 横截面标准化处理

**主要方法**:
- `standardize()` - Z-score标准化
- `winsorize()` - 极值截断
- `process_all()` - 完整处理流程

### 📦 core/ic_calculator.py

**职责**: IC计算与统计分析

**主要方法**:
- `compute_ic()` - 计算IC时间序列
- `compute_ic_stats()` - 计算IC统计量
- `compute_multi_period_ic()` - 多周期IC

### 📦 core/factor_selector.py

**职责**: 因子筛选与约束检查

**主要方法**:
- `select_by_ic_ir()` - 基于IC/IR筛选
- `apply_fdr_correction()` - FDR校正
- `filter_by_correlation()` - 相关性过滤
- `select_final()` - 最终因子选择

### 📦 core/walk_forward_optimizer.py

**职责**: 前向回测框架

**主要方法**:
- `create_windows()` - 创建WFO窗口
- `run_window()` - 运行单个窗口
- `run_all_windows()` - 运行所有窗口
- `generate_report()` - 生成WFO报告

---

## 工作流程

### 🔄 完整工作流

**Step 1: 横截面建设** (scripts/step1_cross_section.py)

```
原始OHLCV数据
  ↓ [数据验证]
  ↓ [因子计算]
  ↓ [横截面处理]
输出: 标准化因子面板 (panel.parquet)
```

**Step 2: 因子筛选** (scripts/step2_factor_selection.py)

```
标准化因子面板
  ↓ [IC计算]
  ↓ [因子筛选]
  ↓ [约束检查]
输出: 最优因子集合 + IC报告
```

**Step 3: WFO优化** (scripts/step3_run_wfo.py)

```
标准化因子面板 + 最优因子集合
  ↓ [窗口划分]
  ↓ [逐窗口处理]
  ↓ [汇总结果]
输出: WFO结果 + 前向性能指标
```

**完整流程** (scripts/run_all_steps.py)

```python
# 自动按顺序执行3个步骤
cross_section_dir = run_step1()
selection_dir = run_step2(cross_section_dir)
wfo_dir = run_step3(selection_dir)
```

---

## 配置系统

### 📝 config.yaml 结构

```yaml
# 数据路径
data:
  raw_dir: ../raw/ETF/daily
  cache_dir: ./cache
  output_dir: ./output

# 因子配置
factors:
  momentum:
    - name: MOM_20D
      period: 20
      weight: 0.3
  volatility:
    - name: VOL_20D
      period: 20
      weight: -0.2
  volume:
    - name: OBV_RATIO
      period: 20
      weight: 0.15

# 筛选配置
screening:
  min_ic: 0.01
  min_ir: 0.05
  max_correlation: 0.7
  use_fdr: true
  fdr_alpha: 0.1

# 回测配置
backtest:
  init_cash: 100000
  top_n: 5
  rebalance_freq: 5
  commission: 0.002
  slippage: 0.001

# WFO配置
wfo:
  train_months: 12
  test_months: 3
  step_months: 3
  min_sharpe: 0.5

# 性能配置
performance:
  n_jobs: 8
  chunk_size: 100
  use_cache: true
```

---

## 性能优化

### ⚡ 优化策略

| 策略 | 实现 | 效果 |
|------|------|------|
| 向量化计算 | NumPy/Pandas | 10x加速 |
| 智能缓存 | pickle缓存 | 减少重复计算 |
| 并行处理 | 8核并行 | 8x加速 |
| 内存优化 | 数据类型优化 | 50%内存节省 |

### 📊 性能指标

| 操作 | 耗时 | 内存 |
|------|------|------|
| 因子计算 | < 5秒 | 100MB |
| 因子筛选 | < 3秒 | 50MB |
| WFO回测 | < 30秒 | 200MB |
| 总耗时 | < 40秒 | < 500MB |

---

## 质量保证

### ✅ 测试覆盖

- 单元测试: 所有核心模块
- 集成测试: 完整工作流
- 端到端测试: 真实数据验证
- 回归测试: 性能基准

### 🔍 代码规范

- Black格式化 (88字符)
- isort导入排序
- mypy类型检查
- flake8代码检查

### 📈 监控指标

- 因子IC分布
- WFO过拟合比
- 信号生成延迟
- 系统资源占用

---

## 快速参考

### 🚀 常用命令

```bash
# 完整流程
make run-pipeline

# 分步执行
make step1-cross-section
make step2-factor-selection
make step3-wfo

# 测试
make test

# 代码检查
make lint

# 格式化
make format
```

### 📚 关键文件

| 文件 | 用途 |
|------|------|
| core/constants.py | 全局常量 |
| core/precise_factor_library_v2.py | 因子计算 |
| scripts/run_all_steps.py | 完整流程 |
| configs/config.yaml | 主配置 |
| tests/test_end_to_end.py | 端到端测试 |

---

**更新日期**: 2025-10-27  
**维护者**: ETF Rotation System Team  
**许可证**: MIT
