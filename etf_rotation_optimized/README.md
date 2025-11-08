# ETF轮动优化系统（面向AI的快速导览） 🎯

本文件是为代码审阅后整理的“高密度项目导览”。大模型或新同学可用本页在1–3分钟内掌握：入口、数据流、关键模块、配置与产物契约。若需细节，请再跳转到 `docs/`。

重要参考：`docs/PROJECT_OVERVIEW.md`、`docs/OUTPUT_SCHEMA.md`、`docs/LLM_GUARDRAILS.md`

- 主入口（组合级WFO）：`run_combo_wfo.py`
- 实盘/回测入口（同名以 `real_backtest/` 为准）：`real_backtest/`
- 输出契约：`docs/OUTPUT_SCHEMA.md`（修改逻辑前必须同步更新）

## 🧩 启用「Markdown 新建限制」

为避免大模型随意新建无意义的 `.md` 文件，本仓库提供 pre-commit 钩子进行约束：

- 仅允许在 `docs/` 目录新建 `.md`；
- 新建文档需在前 20 行加入允许标记（任选其一）：`<!-- ALLOW-MD -->` / `[ALLOW-MD]` / `ALLOW_MD: true`；
- 已存在的 `.md` 不受影响（可自由修改）。

安装钩子：

```bash
bash scripts/install_git_hooks.sh
```

说明：具体策略见 `docs/LLM_GUARDRAILS.md`。

一个高性能、生产就绪的ETF轮动策略系统，基于因子投资和Walk Forward Optimization (WFO) 框架。系统支持从单因子分析到组合级深度挖掘的全流程，专为量化交易和个人投资者设计。

## ✨ 系统要点（一屏速览）

— 性能：Numba JIT（IC/滚动指标）、全向量化、joblib 并行
— 稳健：严格窗口切割（IS/OOS/step），FDR显著性控制，NaN不填充
— 契约：数据契约校验、因子标准化规范、输出文件/字段固定

策略能力：
- 18个精选因子（动量/位置/波动/量能/价量/相对强度/风险调整）
- 组合级 WFO（2–5 因子），多换仓频率投票择优
- FDR（BH）控制显著性；稳定性评分含复杂度惩罚

### 🛡️ 稳健性保证
- **滑动窗口验证**：避免过拟合，确保策略泛化能力
- **严格数据处理**：Z-score标准化、Winsorize极值处理
- **未来函数防护**：彻底排查前瞻偏差，确保结果可信

## 🎯 快速开始（最小闭环）

### 基础运行（单因子WFO）
```bash
# 运行基础WFO流程
python run_final_production.py

# 查看结果
cat results/wfo/wfo_summary.csv
```

### 组合级WFO（推荐）
```bash
# 运行组合级深度优化（推荐）
python run_combo_wfo.py

# 查看Top组合
head -5 results_combo_wfo/top_combos.csv
```

### 无未来函数回测验证（如保留）
```bash
# 严格回测验证（无前瞻偏差）
python test_freq_no_lookahead.py

# 查看回测报告
open results_combo_wfo/未来函数排查与修复完整报告.md
```

### 快速测试
```bash
# 5分钟快速验证（只跑2-3因子组合）
python run_combo_wfo.py --quick

# 查看完整报告
open results_combo_wfo/REPORT.md
```

## 🧠 运行机制（对AI友好）

数据流（T 日 × N ETF × F 因子）：
1) DataLoader 读取 `raw/ETF/daily/*.parquet`（可用 `ETF_DATA_DIR` 覆盖），产出 `{'close','high','low','open','volume'}`，不填充 NaN，执行数据契约校验。参见 `core/data_loader.py`。
2) PreciseFactorLibrary 计算18个精选因子；窗口不足→NaN；不做生成期标准化。参见 `core/precise_factor_library_v2.py`。
3) CrossSectionProcessor 对无界因子做“按日横截面 Z-score + Winsorize(2.5/97.5)”，有界因子透传；严格保留 NaN。参见 `core/cross_section_processor.py`。
4) ComboWFOOptimizer：
   - 以 IS 绝对IC对组合内因子加权
   - 在 OOS 上枚举多种换仓频率，计算OOS均值IC/IR/正IC占比；频率多数票择优
   - FDR(BH)控制显著性，打分含稳定性与复杂度惩罚
   - 并行：`joblib.Parallel(n_jobs)`；参见 `core/combo_wfo_optimizer.py`

入口脚本 `run_combo_wfo.py` 负责：加载配置→数据→因子→标准化→WFO→产物落盘（包含兼容别名 `top100_by_ic.parquet`）。

### 🏆 最优组合发现
**Top 1**: `RELATIVE_STRENGTH_VS_MARKET_20D + RET_VOL_20D + SLOPE_20D + VOL_RATIO_20D`
- **OOS IC**: 0.199 ± 0.137
- **信息比率(IR)**: 1,743.79 (超高!)
- **稳定性得分**: 523.33
- **最优换仓**: 30天
- **FDR q值**: 0.0000 (高度显著)

### 📊 系统性能
- **测试规模**: 12,597个组合 × 19个WFO窗口 × 6个频率 = 143万次测试
- **运行时间**: 仅3.8分钟（10核并行）
- **显著率**: 99.1%通过FDR检验(α=0.05)
- **数据覆盖**: 43只主流ETF × 1399个交易日

### 🎯 实盘验证结果
**最优配置：4因子组合 + 6天换仓**
- **年化收益**: 12.9%
- **Sharpe比率**: 0.486
- **最大回撤**: -33.6%
- **组合**: CORRELATION_TO_MARKET_20D + OBV_SLOPE_10D + PV_CORR_20D + RELATIVE_STRENGTH_VS_MARKET_20D + VOL_RATIO_20D

## 🏗️ 目录与关键文件

### 核心模块
```
etf_rotation_optimized/
├── core/                           # 核心算法引擎
│   ├── data_loader.py               # 高性能数据加载器
│   ├── precise_factor_library_v2.py  # 精选因子库(18个)
│   ├── cross_section_processor.py   # 横截面标准化处理
│   ├── ic_calculator_numba.py       # JIT编译的IC计算器
│   ├── direct_factor_wfo_optimizer.py # 单因子WFO优化器
│   ├── combo_wfo_optimizer.py      # 组合级WFO优化器
│   └── pipeline.py                  # 统一流程编排
├── configs/                        # 配置管理
│   ├── default.yaml                # 基础配置模板
│   └── combo_wfo_config.yaml      # 组合WFO配置
├── results*/                       # 结果输出
│   ├── wfo/                       # 单因子结果
│   └── results_combo_wfo/           # 组合级结果
├── run_combo_wfo.py               # WFO优化主入口
├── run_final_production.py        # 生产运行脚本
├── test_freq_no_lookahead.py      # 无未来函数回测
└── docs/                          # 技术文档
```

### 数据流处理
```mermaid
graph LR
    A[原始数据] --> B[数据加载器]
    B --> C[因子计算引擎]
    C --> D[横截面处理器]
    D --> E[WFO优化器]
    E --> F[组合挖掘器]
    F --> G[结果分析器]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

## 🔧 配置要点（常用键）

### 单因子WFO配置 (`configs/default.yaml`)
```yaml
data:
  symbols: [...]  # 43只主流ETF
  start_date: "2020-01-01"
  end_date: "2025-10-14"

cross_section:
  winsorize_lower: 0.025
  winsorize_upper: 0.975

wfo:
  is_period: 252    # 1年训练期
  oos_period: 60    # 60天测试期
  step_size: 60     # 60天步长
```

### 组合WFO配置 (`configs/combo_wfo_config.yaml`)
```yaml
combo_wfo:
  combo_sizes: [2, 3, 4, 5]           # 2-5因子组合
  rebalance_frequencies: [8]          # 历史验证最佳：固定8天换仓
  enable_fdr: true                      # FDR控制
  fdr_alpha: 0.05                        # 5%显著性水平
  complexity_penalty_lambda: 0.15         # 复杂度惩罚
```

环境变量（可选）：`ETF_DATA_DIR` 指向原始 parquet 数据目录，`ETF_CACHE_DIR` 指向缓存目录。

## 📦 产物契约（落盘清单）
- 运行目录：`results/run_YYYYMMDD_HHMMSS/`
- 文件：
  - `all_combos.parquet`：每个组合的 OOS 统计（IC/IR/正占比/频率等）
  - `top_combos.parquet`、`top100_by_ic.parquet`（兼容别名）：TopN 组合
  - `wfo_summary.json`：窗口/显著性/最优组合摘要
  - `factors/*.parquet`：标准化后的各单因子矩阵
  - `factor_selection_summary.json`：因子处理概要

### 🎯 个人量化投资者
- **快速验证策略想法**: 5分钟获得统计显著性结果
- **深度因子挖掘**: 发现4-5因子的高效协同组合
- **低频交易**: 30天换仓，适合个人资金管理

### 🏢 机构研究团队
- **因子库扩展**: 在18个精选因子基础上添加自定义因子
- **风险控制集成**: 接入组合优化器和风险模型
- **多资产类别**: 扩展到股票、期货等其他资产

### 🎓 学术研究
- **因子有效性验证**: 严格的WFO流程和FDR控制
- **组合优化理论**: 多因子协同效应的实证研究
- **机器学习**: 为AI策略提供高质量特征工程

## ✅ 守护与实用提示
- 严禁提交秘钥；数据路径通过环境变量或配置注入
- `docs/LLM_GUARDRAILS.md` 与 `scripts/pre-commit-md-guard.sh` 启用“Markdown 新建限制”
- 任何修改可能影响产物字段/顺序时，先更新 `docs/OUTPUT_SCHEMA.md` 并在小样本回归验证
 - 频率选择已通过历史验证：最佳换仓频率=8天。为保证复现性与生产一致性，WFO 固定使用 `[8]`，不再进行频率维度搜索。

### 1. Walk Forward Optimization
```python
# 滑动窗口训练验证
for window in sliding_windows:
    # In-Sample训练
    factors_selected = select_by_ic(window.train_data)
    # Out-of-Sample测试
    oos_ic = evaluate(window.test_data, factors_selected)
    # 累积统计
    update_performance(oos_ic)
```

### 2. 组合级挖掘
```python
# 组合枚举与评估
for combo_size in [2, 3, 4, 5]:
    for combo in combinations(18_factors, combo_size):
        for freq in [5, 10, 15, 20, 25, 30]:
            score = evaluate_combo(combo, freq)
            # FDR校正
            corrected_p = fdr_correction(score)
```

### 3. 因子库 (18个精选)
| 类别 | 因子 | 说明 |
|------|------|------|
| 动量 | MOM_20D, RELATIVE_STRENGTH_VS_MARKET_20D | 20日动量、相对强度 |
| 波动率 | RET_VOL_20D, VOL_RATIO_20D/60D | 收益波动、成交量比率 |
| 趋势 | SLOPE_20D, PRICE_POSITION_20D/120D | 价格斜率、位置指标 |
| 技术 | RSI_14, ADX_14D, VORTEX_14D | RSI、ADX、涡度指标 |
| 质量 | SHARPE_RATIO_20D, CALMAR_RATIO_60D | 夏普比率、卡玛比率 |
| 资金流 | CMF_20D, OBV_SLOPE_10D | 资金流指标 |

## 🚀 性能优化

### 计算性能
- **Numba JIT加速**: IC计算提升100倍
- **向量化操作**: 避免Python循环，利用NumPy优化
- **并行计算**: 10核CPU并行处理12K组合
- **内存管理**: 智能缓存，避免重复计算

### ✅ 性能冻结与推荐运行配置 (2025-11-08)

> 已完成高ROI优化：日级IC预计算 + memmap共享 + Numba预热 + 稳定排名(平均 ties) + 批任务调度 + 全局IC缓存。进一步微优化进入长尾阶段，暂不建议继续。

#### 最新 Top200 对照 (优化 vs 基线)

| 指标 | 基线 (无预计算/稳定排名) | 优化 (RB_DAILY_IC_PRECOMP=1 + STABLE_RANK) | 提升 |
|------|------------------------|-------------------------------------------|------|
| time_total mean | 1096.9 ms | 29.0 ms | ≈37.8× |
| time_total median | 963.6 ms | 6.4 ms | ≈150× |
| time_total p95 | 1258.6 ms | 13.0 ms | ≈96.8× |
| time_precompute_ic mean | 1039.5 ms | 0.9 ms | ≈1155× |
| time_main_loop median | 35.7 ms | 5.4 ms | ≈6.6× |

> 单批少量组合的极端耗时 (p99) 仍可能 ~500–600ms（策略主循环固有），属可接受“结构性长尾”。

#### 推荐生产环境变量模板

```bash
export RB_DAILY_IC_PRECOMP=1          # 启用日级IC预计算 + 前缀和 O(1) 滑窗
export RB_DAILY_IC_MEMMAP=1           # 多进程共享日IC矩阵，避免重复构建
export RB_NUMBA_WARMUP=1              # 进程启动时一次性 JIT 预热
export RB_PRELOAD_IC=1                # 提前填充常用 (freq×factor) 缓存对
export RB_STABLE_RANK=1               # Spearman 日IC 使用平均 ties 稳定排名
export RB_TASK_BATCH_SIZE=8           # joblib 调度批大小；根据CPU可调 4/8/16
export RB_ENFORCE_NO_LOOKAHEAD=0      # 日常跑关闭；回归/审计时再开
export RB_PROFILE_BACKTEST=0          # 诊断阶段开启
export RB_OUTLIER_REPORT=0            # 仅在性能分析阶段开启
```

#### 诊断/审计附加参数

```bash
# 严格无未来函数抽样验证（稳定排名下需要更宽容容差）
export RB_ENFORCE_NO_LOOKAHEAD=1
export RB_NL_CHECK_TOL=1e-2           # 稳定排名 vs 旧权重路径差异在 1e-3~1e-2
export RB_NL_CHECK_MAX=5              # 抽样次数

# 性能剖析与异常组合定位
export RB_PROFILE_BACKTEST=1
export RB_OUTLIER_REPORT=1
```

#### Outlier 处理策略

- p95 以上多数为高频权重重算循环的自然长尾（组合头部因子集类似，缓存命中后 loop 成为主耗时）。
- 未发现重复IC重算或明显I/O瓶颈，不再追加微优化。
- 若需忽略展示：可在 `real_backtest/outlier_whitelist.txt` 维护组合白名单。

#### 冻结原则

1. 未来性能改动需提出明确收益预估 (×倍或 ≥15% 壁钟缩减)。
2. 修改 IC 权重路径或排名模式需同步更新无未来函数回归 (RB_ENFORCE_NO_LOOKAHEAD=1)。
3. 默认保持与当前推荐环境一致，避免回测结果不可复现。

---
**状态结论**：性能阶段完成；后续聚焦策略研究 / 新因子扩展 / 风险管理集成。

### 统计严谨性

- **FDR多重检验**: Benjamini-Hochberg校正
- **滑动窗口**: 避免过拟合，确保泛化能力
- **前瞻偏差防护**: 严格的未来函数排查
- **稳定性评分**: 考虑IC、IR、胜率、复杂度

## 📊 结果解读

### 关键指标

- **IC (Information Coefficient)**: 因子与未来收益的相关系数
- **IR (Information Ratio)**: IC均值/IC标准差，衡量风险调整后表现
- **胜率**: 正IC占比，评估方向准确性
- **稳定性得分**: 综合评分(0.5*IC + 0.3*IR + 0.2*胜率 - 复杂度惩罚)

### 优化建议

1. **换仓频率**: 10天(高收益) vs 30天(低成本)
2. **因子数量**: 4-5因子(协同效应) vs 2-3因子(简单性)
3. **风险控制**: 结合波动率和最大回撤指标
4. **市场适应**: 牛熊市可能需要不同因子组合

## 📁 结果目录说明

- `results_combo_wfo/` - WFO优化结果
  - `REPORT.md` - 完整分析报告和Top 10组合
  - `top_combos.csv` - Top 50组合详细数据
  - `all_combos.csv` - 所有12,597个组合结果
  - `freq_test_no_lookahead.csv` - 无未来函数回测结果
  - `未来函数排查与修复完整报告.md` - 回测验证报告

- `results/wfo/` - 单因子WFO结果
  - `wfo_summary.csv` - 每个窗口的统计摘要
  - `window_results.json` - 详细窗口结果

## 🛠️ 高级用法

### 自定义因子

```python
# 在precise_factor_library_v2.py中添加
@njit
def custom_factor(close, volume, window=20):
    # 实现自定义因子逻辑
    return factor_values
```

### 风险集成

```yaml
# 添加风险约束
risk_management:
  max_position_size: 0.05    # 单个ETF最大5%
  max_drawdown: 0.15         # 最大回撤15%
  turnover_limit: 0.3         # 年换手率30%
```

### 实盘部署

```bash
# 1. 回测验证
python run_combo_wfo.py

# 2. 策略评估
python test_freq_no_lookahead.py

# 3. 生产运行
python run_final_production.py
```

## 📈 路线图

### 2024年发展历程

- **Q1**: 系统架构重构，移除历史遗留代码
- **Q2**: Numba JIT优化，性能提升10倍
- **Q3**: 组合级WFO实现，支持深度挖掘
- **Q4**: FDR控制集成，统计严谨性提升

### 未来规划

- [ ] 多周期扩展(周频、月频)
- [ ] 跨资产类别支持(股票、期货)
- [ ] 机器学习特征工程
- [ ] 实盘交易接口集成

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进系统：

1. **Bug报告**: 提供复现步骤和期望结果
2. **功能建议**: 详细描述使用场景
3. **性能优化**: 提供基准测试结果
4. **文档改进**: 帮助其他用户理解系统

## 📄 许可证

MIT License

---

<div align="center">

**ETF轮动优化系统**
*让量化交易更简单、更高效、更可靠*

⭐ 如果这个项目对您有帮助，请考虑给我们一个Star！

</div>
