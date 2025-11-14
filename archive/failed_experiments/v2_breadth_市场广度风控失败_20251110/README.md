# ETF Rotation V2 - 市场广度风控实验 🔬# ETF轮动优化系统（面向AI的快速导览） 🎯



[![Status](https://img.shields.io/badge/status-experiment-orange)](https://github.com)本文件是为代码审阅后整理的“高密度项目导览”。大模型或新同学可用本页在1–3分钟内掌握：入口、数据流、关键模块、配置与产物契约。若需细节，请再跳转到 `docs/`。

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)

[![License](https://img.shields.io/badge/license-private-red)](https://github.com)重要参考：`docs/PROJECT_OVERVIEW.md`、`docs/OUTPUT_SCHEMA.md`、`docs/LLM_GUARDRAILS.md`



## 🎯 项目目标- 主入口（组合级WFO）：`run_combo_wfo.py`

- 实盘/回测入口（同名以 `real_backtest/` 为准）：`real_backtest/`

在**不破坏动量复利**的前提下，通过**市场广度监控**改善ETF轮动策略的长尾损失。- 输出契约：`docs/OUTPUT_SCHEMA.md`（修改逻辑前必须同步更新）



## 📊 背景## 🧩 启用「Markdown 新建限制」



### 自适应项目的失败教训为避免大模型随意新建无意义的 `.md` 文件，本仓库提供 pre-commit 钩子进行约束：



| 配置 | 年化收益 | Sharpe | 触发次数 | 现金占比 |- 仅允许在 `docs/` 目录新建 `.md`；

|------|---------|--------|---------|---------|- 新建文档需在前 20 行加入允许标记（任选其一）：`<!-- ALLOW-MD -->` / `[ALLOW-MD]` / `ALLOW_MD: true`；

| 部分止盈 | 4.34% | 0.153 | 810次 | 42% |- 已存在的 `.md` 不受影响（可自由修改）。

| 全清止盈 | 2.19% | 0.074 | 810次 | 50% |

| 宽止损 | 5.06% | 0.179 | 264次 | 38% |安装钩子：

| **纯周期调仓** | **11.39%** | **0.502** | 0次 | 5% |

| 基准（老项目） | 12.9% | 0.486 | N/A | 5% |```bash

bash scripts/install_git_hooks.sh

**结论**: 显式止损止盈破坏了动量趋势，频繁交易侵蚀收益。```



### 老项目的隐性风控说明：具体策略见 `docs/LLM_GUARDRAILS.md`。



老项目（etf_rotation_optimized）看似"无风控"，实则有**软性前置风控**：一个高性能、生产就绪的ETF轮动策略系统，基于因子投资和Walk Forward Optimization (WFO) 框架。系统支持从单因子分析到组合级深度挖掘的全流程，专为量化交易和个人投资者设计。



1. **因子清洗**: Winsorize(2.5%/97.5%) 截断极值## 🚀 技术亮点总览（性能冻结后 2025-11-09）

2. **相关去冗**: 相关性>0.8只保留IC高者

3. **家族配额**: 每类因子最多1-2个（MOMENTUM最多4个）| 维度 | 技术实现 | 价值 | 状态 |

4. **IC加权**: 弱因子自动稀释权重|------|----------|------|------|

5. **WFO滚动**: IS 252 + OOS 60，每20天验证一次| 计算性能 | 日级IC预计算 + 前缀和 O(1) 滑窗 | IC权重提取极限加速 | ✅ 冻结 |

| 内存共享 | np.memmap 多进程共享日IC矩阵 | 消除重复构建开销 | ✅ |

这套机制在**信号层而非仓位层**做风控，避免了破坏动量的问题。| 排名稳定性 | Spearman 平均 ties 排名 (Stable Rank) | 降低平局导致的抖动 | ✅ |

| JIT 加速 | Numba 预热 + 向量化 | 主循环成为唯一显著耗时 | ✅ |

## 🆕 第二代风控| 全局缓存 | (freq×factor) 权重/IC 全局缓存 | 提升命中率减少重算 | ✅ |

| 性能剖析 | RB_PROFILE_BACKTEST + Outlier 报告 | 精确识别结构性长尾 | ✅ |

### 核心思路| 无未来函数 | RB_ENFORCE_NO_LOOKAHEAD 抽样重算 | 防止前瞻偏差 | ✅ (路径差异已知) |

| 校准排序 | GBDT (WFO特征→Sharpe) | 排序从“近似随机”→“高相关” | ✅ |

**不在仓位层做止损止盈，而在信号有效性层面做防守。**| Markdown守卫 | pre-commit 钩子限制文档新增 | 控制文档膨胀 | ✅ |

| 输出契约 | 固定字段 + 变更需回归 | 保证版本间可比性 | ✅ |

### 三个模块

> 后续除非能证明 ≥15% 壁钟缩减或显著稳定性提升，否则不再做微优化；重心转向新因子 / 风险集成 / 校准器演进。

#### 🟢 市场广度监控（推荐）

### 稳定排名、自检容差与日级IC预计算

```python

有效信号占比 = (因子得分>0的ETF数量) / 总ETF数量- 稳定排名（Stable Rank）：横截面打分采用“平均并列（average ties）”，并在 memmap 键中标注模式版本，确保并列时的确定性与可复现性。

if 有效信号占比 < 25%:- 日级IC预计算：可通过环境变量开启每日跨截面 Spearman(IC) 的全量矩阵预计算，并使用 memmap 加速重复使用。

    position_scale = 0.5  # 降至50%仓位- 自检容差：当 RB_ENFORCE_NO_LOOKAHEAD=1 时，系统会在调仓日抽样进行权重重算校验。

else:  - 若启用稳定秩 + 日级IC预计算，自检优先采用与生产一致的“日级IC窗口均值”生成 w_chk，避免与简化重算路径的数值差异。

    position_scale = 1.0  # 正常仓位  - 容差通过以下环境变量配置（语义与 numpy.allclose 一致）：

```    - RB_NL_CHECK_RTOL（相对误差，建议 1e-3 ~ 5e-3 起步）

    - RB_NL_CHECK_ATOL（绝对误差，建议 1e-6 ~ 1e-4 起步）

**优势**:  - 在稳定路径与预计算开启的情况下，建议将 rtol 收敛到 1e-3；如遇极少数窗口差异，可临时上调 atol 再定位。

- O(N)复杂度，无额外计算

- 比波动率更早捕捉信号崩溃示例（zsh）：

- 直接检测模型失效而非市场波动

```zsh

#### 🟡 波动率目标（谨慎）export RB_STABLE_RANK=1

export RB_DAILY_IC_PRECOMP=1

```pythonexport RB_DAILY_IC_MEMMAP=1

vol = max(20D波动, 60D波动)export RB_ENFORCE_NO_LOOKAHEAD=1

if vol > 30%:export RB_NL_CHECK_RTOL=1e-3

    position_scale = min_scale  # 降至30%export RB_NL_CHECK_ATOL=1e-6

``````



**风险**: 滞后性，可能踏空反弹（2020年3月）### Daily IC memmap 键模式



#### 🤷 相关性监控（低优先级）为避免跨模式污染，memmap 文件名编码了关键维度与模式版本：



```python```

avg_corr = np.corrcoef(因子矩阵).mean()daily_ic_auto_{v2stable|v1simple}_{T}_{N}_{F}_{digest}_fp64.mmap

if avg_corr > 0.65:```

    penalty = 0.5  # 减权50%

```- v2stable / v1simple：分别表示稳定排名路径与简单排名路径；

- T / N / F：样本天数、标的数量、因子数量；

**为什么低优先级**: 老项目已有静态去冗，动态版改善有限。- digest：基于输入矩阵内容的摘要（确保不同数据或排列下不会复用错误文件）；

- fp64：数据精度标识。

## 🚀 快速开始

当 memmap 被其他进程占用时，会自动回退为内存计算但不写入文件，避免锁冲突。

### 1. 运行集成测试

## ✨ 系统要点（一屏速览）

```bash

cd /Users/zhangshenshen/深度量化0927/etf_rotation_v2_breadth— 性能：日级IC预计算 + memmap + Numba预热 + 向量化 + 并行

python3 test_risk_control.py— 稳健：严格IS/OOS切割、FDR显著性、未来函数抽样校验、NaN不填充

```— 排序：GBDT校准器利用稳定性/波动/正率等多特征

— 契约：数据/输出字段固定 + Markdown新增守卫 + 回归友好

**预期输出**:

策略能力：

```text- 18个精选因子（动量/位置/波动/量能/价量/相对强度/风险调整）

✅ 配置加载成功- 组合级 WFO（2–5 因子），多换仓频率投票择优

   - 市场广度: 已启用- FDR（BH）控制显著性；稳定性评分含复杂度惩罚

✅ 风控日志生成成功: 1379条记录

   - 触发防守天数: 685 (49.7%)### 🛡️ 稳健性保证

   - 平均缩仓比例: 24.2%- **滑动窗口验证**：避免过拟合，确保策略泛化能力

测试总结: ✅ 通过- **严格数据处理**：Z-score标准化、Winsorize极值处理

```- **未来函数防护**：彻底排查前瞻偏差，确保结果可信



### 2. 运行完整对比## 🎯 快速开始（最小闭环）



```bash### 基础运行（单因子WFO）

./quick_start.sh```bash

# 或# 运行基础WFO流程

bash quick_start.shpython run_final_production.py

```

# 查看结果

选择模式:cat results/wfo/wfo_summary.csv

```

1. **仅测试** - 验证集成（已完成）

2. **Baseline** - 无风控基准### 组合级WFO（推荐）

3. **市场广度版** - 仅启用市场广度（推荐）```bash

4. **综合版** - 三模块全开# 运行组合级深度优化（推荐）

5. **全部运行** - 一键对比所有版本（15-30分钟）python run_combo_wfo.py



### 3. 查看结果# 查看Top组合

head -5 results_combo_wfo/top_combos.csv

```bash```

# 风控触发日志

head -100 results/v2_market_breadth/wfo/risk_control_log.csv### 无未来函数回测验证（如保留）

```bash

# WFO统计# 严格回测验证（无前瞻偏差）

cat results/v2_market_breadth/wfo/wfo_summary.csvpython test_freq_no_lookahead.py



# 对比报告（在quick_start.sh末尾自动生成）# 查看回测报告

```open results_combo_wfo/未来函数排查与修复完整报告.md

```

## 📂 项目结构

### 快速测试

```text```bash

etf_rotation_v2_breadth/# 5分钟快速验证（只跑2-3因子组合）

├── core/python run_combo_wfo.py --quick

│   ├── market_breadth.py          # 市场广度监控器（148行）

│   ├── volatility_target.py       # 波动率目标管理器（171行）# 查看完整报告

│   ├── correlation_monitor.py     # 相关性监控器（198行）open results_combo_wfo/REPORT.md

│   └── pipeline.py                # 主流水线（已集成风控层）```

├── configs/

│   ├── risk_control_v2.yaml       # 风控配置模板## 🧠 运行机制（对AI友好）

│   ├── run_baseline.yaml          # Baseline配置

│   ├── run_market_breadth.yaml    # 市场广度配置数据流（T 日 × N ETF × F 因子）：

│   └── run_comprehensive.yaml     # 综合版配置1) DataLoader 读取 `raw/ETF/daily/*.parquet`（可用 `ETF_DATA_DIR` 覆盖），产出 `{'close','high','low','open','volume'}`，不填充 NaN，执行数据契约校验。参见 `core/data_loader.py`。

├── test_risk_control.py           # 集成测试脚本2) PreciseFactorLibrary 计算18个精选因子；窗口不足→NaN；不做生成期标准化。参见 `core/precise_factor_library_v2.py`。

├── quick_start.sh                 # 一键运行脚本3) CrossSectionProcessor 对无界因子做“按日横截面 Z-score + Winsorize(2.5/97.5)”，有界因子透传；严格保留 NaN。参见 `core/cross_section_processor.py`。

├── RISK_CONTROL_V2_GUIDE.md       # 详细指南4) ComboWFOOptimizer：

└── README.md                      # 本文件   - 以 IS 绝对IC对组合内因子加权

```   - 在 OOS 上枚举多种换仓频率，计算OOS均值IC/IR/正IC占比；频率多数票择优

   - FDR(BH)控制显著性，打分含稳定性与复杂度惩罚

## 🔬 实验设计   - 并行：`joblib.Parallel(n_jobs)`；参见 `core/combo_wfo_optimizer.py`



### 为什么要复制项目？入口脚本 `run_combo_wfo.py` 负责：加载配置→数据→因子→标准化→WFO→产物落盘（包含兼容别名 `top100_by_ic.parquet`）。



1. **保护生产基线**: etf_rotation_optimized已验证可行，不能直接修改### 🏆 最优组合发现

2. **隔离实验**: 独立测试新机制，失败不影响原系统**Top 1**: `RELATIVE_STRENGTH_VS_MARKET_20D + RET_VOL_20D + SLOPE_20D + VOL_RATIO_20D`

3. **A/B对比**: 清晰对比有/无风控的差异- **OOS IC**: 0.199 ± 0.137

- **信息比率(IR)**: 1,743.79 (超高!)

### 对比维度- **稳定性得分**: 523.33

- **最优换仓**: 30天

| 指标 | 含义 | 期望 |- **FDR q值**: 0.0000 (高度显著)

|------|------|------|

| 年化收益 | 绝对收益 | 容忍-2~3%下降 |### 📊 系统性能

| Sharpe比率 | 收益/波动 | 应提升 |- **测试规模**: 12,597个组合 × 19个WFO窗口 × 6个频率 = 143万次测试

| 最大回撤 | 最大损失 | 应缩小 |- **运行时间**: 仅3.8分钟（10核并行）

| 触发率 | 防守天数占比 | 30-50%合理 |- **显著率**: 99.1%通过FDR检验(α=0.05)

| 2020危机表现 | 防守vs踏空 | 3月提前防守，但不要全程空仓 |- **数据覆盖**: 43只主流ETF × 1399个交易日



## 📊 关键发现（待验证）### 🎯 实盘验证结果

**最优配置：4因子组合 + 6天换仓**

### 市场广度监控的优势- **年化收益**: 12.9%

- **Sharpe比率**: 0.486

1. **早期预警**: 2020-02-07开始触发（比波动率早1个月）- **最大回撤**: -33.6%

2. **低成本**: O(N)复杂度，无重计算- **组合**: CORRELATION_TO_MARKET_20D + OBV_SLOPE_10D + PV_CORR_20D + RELATIVE_STRENGTH_VS_MARKET_20D + VOL_RATIO_20D

3. **直接**: 检测信号失效而非市场波动

## 🏗️ 目录与关键文件

### 2020年3月案例

### 核心模块

| 日期 | 市场广度 | 仓位scale | 备注 |```

|------|---------|-----------|------|etf_rotation_optimized/

| 2020-02-07 | 0% | 0.5 | 开始防守 |├── core/                           # 核心算法引擎

| 2020-02-20 | 0% | 0.5 | 持续防守 |│   ├── data_loader.py               # 高性能数据加载器

| 2020-03-02 | 0% | 0.5 | 暴跌期 |│   ├── precise_factor_library_v2.py  # 精选因子库(18个)

| 2020-03-23 | 0% | 0.5 | 底部 |│   ├── cross_section_processor.py   # 横截面标准化处理

| 2020-03-24 | 0% | 0.5 | 反弹开始（风险：可能踏空） |│   ├── ic_calculator_numba.py       # JIT编译的IC计算器

| 2020-05-01 | 正常 | 1.0 | 恢复正常 |│   ├── direct_factor_wfo_optimizer.py # 单因子WFO优化器

│   ├── combo_wfo_optimizer.py      # 组合级WFO优化器

**Trade-off**: 提前防守保护回撤，但可能错过3月底反弹。│   └── pipeline.py                  # 统一流程编排

├── configs/                        # 配置管理

## ⚙️ 参数调节│   ├── default.yaml                # 基础配置模板

│   └── combo_wfo_config.yaml      # 组合WFO配置

### 市场广度├── results*/                       # 结果输出

│   ├── wfo/                       # 单因子结果

```yaml│   └── results_combo_wfo/           # 组合级结果

market_breadth:├── run_combo_wfo.py               # WFO优化主入口

  breadth_floor: 0.25       # 25%阈值（降低=更敏感）├── run_final_production.py        # 生产运行脚本

  defensive_scale: 0.5      # 50%防守仓位（降低=更激进）├── test_freq_no_lookahead.py      # 无未来函数回测

  score_threshold: 0.0      # z-score正值为有效（提高=更严格）└── docs/                          # 技术文档

``````



**推荐调节**:### 数据流处理

```mermaid

- 更敏感: `breadth_floor: 0.30`（30%）graph LR

- 更激进: `defensive_scale: 0.3`（30%仓位）    A[原始数据] --> B[数据加载器]

- 更严格: `score_threshold: 0.1`（z-score>0.1才算有效）    B --> C[因子计算引擎]

    C --> D[横截面处理器]

### 综合策略    D --> E[WFO优化器]

    E --> F[组合挖掘器]

```yaml    F --> G[结果分析器]

combine_strategy: "min"       # 多模块取最小值（保守）

# combine_strategy: "multiply" # 相乘（激进）    style A fill:#f9f,stroke:#333

# combine_strategy: "max"      # 取最大值（宽松）    style G fill:#9f9,stroke:#333

``````



## 🛠️ 工程原则## 🔧 配置要点（常用键）



遵循**Linus模式**:### 单因子WFO配置 (`configs/default.yaml`)

```yaml

1. ✅ **向量化优先**: ≥95%向量化，无.apply()data:

2. ✅ **明确复杂度**: 每个方法注明O(N)或O(F²×T)  symbols: [...]  # 43只主流ETF

3. ✅ **函数短小**: <50行，3层缩进限制  start_date: "2020-01-01"

4. ✅ **清晰注释**: 风险警告、参数说明、使用场景  end_date: "2025-10-14"

5. ✅ **可测试**: 独立模块，易于单元测试

cross_section:

## 📚 文档  winsorize_lower: 0.025

  winsorize_upper: 0.975

- **RISK_CONTROL_V2_GUIDE.md**: 详细指南（FAQ、调参、可视化）

- **configs/risk_control_v2.yaml**: 完整配置模板wfo:

- **test_risk_control.py**: 集成测试源码  is_period: 252    # 1年训练期

  oos_period: 60    # 60天测试期

## ❓ FAQ  step_size: 60     # 60天步长

```

### Q: 为什么不直接修改老项目？

### 组合WFO配置 (`configs/combo_wfo_config.yaml`)

**A**: Linus原则 - "如果可能破坏现有系统，先在分支验证"。老项目已验证可行（年化12.9%），不能冒险。```yaml

combo_wfo:

### Q: 市场广度频繁误触发怎么办？  combo_sizes: [2, 3, 4, 5]           # 2-5因子组合

  rebalance_frequencies: [8]          # 历史验证最佳：固定8天换仓

**A**: 调节参数 `breadth_floor: 0.20`（降低敏感度）或 `defensive_scale: 0.7`（减轻防守力度）。  enable_fdr: true                      # FDR控制

  fdr_alpha: 0.05                        # 5%显著性水平

### Q: 如何判断风控是否有效？  complexity_penalty_lambda: 0.15         # 复杂度惩罚

```

**对比指标**:

环境变量（可选）：`ETF_DATA_DIR` 指向原始 parquet 数据目录，`ETF_CACHE_DIR` 指向缓存目录。

1. 最大回撤应缩小

2. Sharpe比率应提升## 📦 产物契约（落盘清单）

3. 年化收益容忍下降2-3%- 运行目录：`results/run_YYYYMMDD_HHMMSS/`

4. 触发率30-50%合理- 文件：

5. 2020年3月提前防守但不全程空仓  - `all_combos.parquet`：每个组合的 OOS 统计（IC/IR/正占比/频率等）

  - `top_combos.parquet`、`top100_by_ic.parquet`（兼容别名）：TopN 组合

### Q: 下一步计划？  - `wfo_summary.json`：窗口/显著性/最优组合摘要

  - `factors/*.parquet`：标准化后的各单因子矩阵

1. **完整回测对比**（baseline vs 市场广度 vs 综合）  - `factor_selection_summary.json`：因子处理概要

2. **参数网格搜索**（breadth_floor: 0.15-0.35）

3. **长尾损失分析**（>20%回撤的次数）### 🎯 个人量化投资者

4. **2020危机期验证**（防守vs踏空trade-off）- **快速验证策略想法**: 5分钟获得统计显著性结果

5. **如果验证成功**: 合并到主项目或建议只启用市场广度- **深度因子挖掘**: 发现4-5因子的高效协同组合

- **低频交易**: 30天换仓，适合个人资金管理

## 🔗 相关项目

### 🏢 机构研究团队

- **etf_rotation_optimized**: 老项目（生产基线）- **因子库扩展**: 在18个精选因子基础上添加自定义因子

- **etf_rotation_adaptive**: 自适应项目（失败教训）- **风险控制集成**: 接入组合优化器和风险模型

- **多资产类别**: 扩展到股票、期货等其他资产

## 📝 日志

### 🎓 学术研究

- **2025-11-10**: 项目创建，集成测试通过- **因子有效性验证**: 严格的WFO流程和FDR控制

- **待定**: 完整回测结果- **组合优化理论**: 多因子协同效应的实证研究

- **待定**: 参数网格搜索- **机器学习**: 为AI策略提供高质量特征工程

- **待定**: 合并决策

## ✅ 守护与实用提示

---- 严禁提交秘钥；数据路径通过环境变量或配置注入

- `docs/LLM_GUARDRAILS.md` 与 `scripts/pre-commit-md-guard.sh` 启用“Markdown 新建限制”

**作者**: GitHub Copilot  - 任何修改可能影响产物字段/顺序时，先更新 `docs/OUTPUT_SCHEMA.md` 并在小样本回归验证

**版本**: V2.0   - 频率选择已通过历史验证：最佳换仓频率=8天。为保证复现性与生产一致性，WFO 固定使用 `[8]`，不再进行频率维度搜索。

**状态**: 实验阶段 🔬  

**最后更新**: 2025-11-10### 1. Walk Forward Optimization

```python

**Talk is cheap, show me the backtest. 📈**# 滑动窗口训练验证

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
    # 性能冻结后生产固定 freq=8；下行枚举仅保留历史示例
    for freq in [8]:  # 旧版本: [5, 10, 15, 20, 25, 30]
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

## 🔄 排序校准质量提升总结

原始 WFO 仅以 mean_oos_ic 排序：Spearman(IC vs 实际 Sharpe) ≈ 0.07，接近随机。引入多特征 + GBDT 后：Spearman ≈ 0.83，Precision@100≈50%。

| 排序方式 | Spearman(真实Sharpe) | Precision@100 | 备注 |
|----------|----------------------|---------------|------|
| mean_oos_ic | ≈0.07 | ≈1% | 单指标，忽略稳定性与波动 |
| Ridge | ≈0.46 | ≈1% | 线性+强正则，区分度不足 |
| GBDT (全量12,597) | 0.915 | 50% | 非线性交互，利用稳定性/波动性/正率 |

特征重要性（GBDT）：`stability_score > oos_ic_std > mean_oos_ic > positive_rate >> combo_size`。

示例：组合 #11874 (IC=0.0228, 原始排名底部6%) → 校准后 Sharpe 预测优秀，提升至 Top0.1%。

生产建议（冻结后策略）：

1. 直接按 `calibrated_sharpe_pred` 单列降序 + 次级关键字 `stability_score` 排序；不再引入 blended/whitelist 过渡方案（已评估后弃用）。
2. 新增因子/特征后需回归再训练，要求 hold-out Spearman ≥ 0.70 且 Pearson ≥ 0.75；否则拒绝上线。
3. 若未来真实 Sharpe vs 预测 Sharpe Pearson 连续两个评估窗口 <0.80，触发“再训练/特征审计”流程。
4. 排序逻辑冻结：除非出现数据结构性漂移（新增资产类别、样本期延长 >30% 或特征失稳），否则不做微调。

再训练脚本：`scripts/train_calibrator_full.py`（全量 12,597 样本）。

### ❌ Blended / Whitelist 方案弃用说明

经过全量 12,597 组合再训练与真实回测对照：

- 纯校准 TopK 与真实 Sharpe 呈高相关（Pearson≈0.93, Spearman≈0.915），无需人为混合降低漂移。

## 🔎 候选池审计与上线前质检（Top2000→Top500）

为防止组合集中度过高或隐含重复，新增一键审计与质检：

- 输入：`results/run_*/selection/candidate_top2000.csv`、`selected_top500.csv`（由 `scripts/select_topk_candidates.py` 生成）
- 输出：`selection/audit/audit_report.md`、`audit_report.json`
- 覆盖指标：
  - 因子覆盖与重合度（Top factors 直方图、Jaccard 重合采样统计）
  - 频率/行业分布（若列存在）
  - 重复组合检测（基于 id/因子集/频率构造去重键）
  - Top500 组合等权 HHI（越低越分散，等权理论基线≈0.002）
  - （可选）真实 Sharpe 质检：若存在全量回测 CSV，则输出 Pearson/Spearman、Precision@K 与 Top500 的 realized Sharpe 分布

使用：

```zsh
# 生成审计报告（需已有 run 目录，如 results/run_YYYYMMDD_HHMMSS）
make audit RUN_DIR=results/run_YYYYMMDD_HHMMSS
```

若需要显式指定全量回测 CSV（未写入 selection_summary.json 时）：

```zsh
python scripts/audit_candidate_pool.py --run-dir results/run_YYYYMMDD_HHMMSS \
  --backtest-csv /path/to/full_backtest.csv
```

## 🚨 排序质量监控（阈值告警）

新增自动监控脚本，防止 WFO 排序/校准器失稳：

- 配置：`config/monitor_thresholds.yaml`（默认：pearson_min=0.85, spearman_min=0.80, precision@500≥0.30, precision@2000≥0.70, top500_hhi≤0.010）
- 数据来源优先级：
  1) `comparison/calibrated_vs_realized_metrics.json`（若存在）
  2) 回退：`all_combos.parquet` + 全量回测 CSV（从 `selection/selection_summary.json` 读取或通过命令行传入）
- 结果：阈值内返回 0；违约返回 2，并打印详细 JSON

使用：

```zsh
make monitor RUN_DIR=results/run_YYYYMMDD_HHMMSS
# 或自定义阈值/CSV：
python scripts/monitor_wfo_rank_quality.py --run-dir results/run_YYYYMMDD_HHMMSS \
  --thresholds config/monitor_thresholds.yaml --backtest-csv /path/to/full_backtest.csv
```

> 建议将 `make monitor` 纳入生产流水线（如 cron/CI）。当 Pearson 连续两个窗口 <0.80 或 Top500 HHI 超阈值时，触发“再训练/特征审计”。

## 🧪 策略研发/拓展（在高质量候选基础上）

围绕 `selection/candidate_top2000.csv` 进行风格/因子覆盖分析：

- 因子覆盖/聚类：依据 `factors`/`factor_names` 列做聚类（Jaccard/MinHash 或因子出现计数向量），挑选“风格互补”的子组合池
- 风格专注：聚焦波动/动量/套利等子类，在 `selected_top500.csv` 中抽样滚动窗口做快速验证
- 轻量验证：直接复用 `scripts/select_topk_candidates.py` 生成的 `spotcheck_*.csv` 作为子集回测样本

> 研发产物一律以 YAML 化配置 + 统一输出契约接入；避免 ad-hoc notebook 结果不可复现。
- Blended 与 whitelist 方案引入额外参数（α, retain, K）与维护复杂度，但不再带来显著 Sharpe 提升。
- 生产一致性与可复现性优先：故正式弃用 `generate_blended_whitelist.py` 等辅助上线脚本，保留仅供历史分析。

迁移策略：

1. 所有上线/回测直接读取 `calibrated_sharpe_pred`；无二次加权。
2. 历史报告中出现的 blended 指标保留归档，不再更新。
3. 若未来模型出现“预测集中度过高 + TopK 收益劣化”双条件，可临时恢复混合评估再决策。触发阈值：HHI Top500 >0.70 且 Top500 实际年化收益下降 ≥25% 相对最近 3 窗平均。

### 🧪 校准后完整验证流水线（冻结后）

一键执行命令（包含：sanity check + hold-out 验证 + 相关性评估）：

```bash
make post-calibration   # （已自动跳过 blended/whitelist 生成）
```

生成的核心文件：

| 文件 | 说明 |
|------|------|
| `all_combos_calibrated_gbdt_full.parquet` | 加入 `calibrated_sharpe_full` 列的全量组合结果 |
| `ranking_sanity_check.csv` / `ranking_sanity_summary.txt` | 校准前后 Top2000 对比与分布差异 |
| `holdout_validation_report.txt` | 80/20 分层 hold-out 验证指标 |
| ~~`whitelist_top2000_blended_alpha*_retain*.txt`~~ | 已弃用（保留历史归档） |
| ~~`blended_grid_eval.*`~~ | 已弃用（仅旧版本分析） |
| `calibrated_vs_realized_metrics.json` | 校准预测 vs 真实 Sharpe 相关性与 Precision@K |
| `calibrator_full_vs_top2000_comparison.png` | 校准器性能可视化对比 |

上线策略建议（新版）：

1. 仅校准排序：无需混合观测期，直接进入监控（窗口 = 最近两个再平衡周期）。
2. 监控指标：Pearson / Spearman，TopK Precision，Top500 HHI，收益漂移（Δ年化 vs 基准窗口）。
3. 再训练触发：重复两个监控窗口 Pearson <0.80 或 Top100 Precision 跌破 40%。
4. 任何改动必须产生 `calibrated_vs_realized_metrics.json` 新版本并回归。

### 🎯 TopK 候选池与分层筛选流程（冻结后主线）

当前主线采用两阶段筛选：

1. 预测排序候选池：取最新 run (`results/run_*/all_combos.parquet`) 中按 `calibrated_sharpe_pred` → `stability_score` → `mean_oos_ic` 降序的前 2000 组，生成 `selection/candidate_top2000.csv`。
2. 分层精简：在候选池内再次按同序列排序截取 Top500 与 Top200，生成 `selected_top500.csv` 与 `selected_top200.csv`。
3. Realized 统计：若存在全量真实回测 CSV（自动探测 `results_combo_wfo/*full.csv`），计算 Sharpe/年化/最大回撤中位，并写入 `selection/selection_summary.json`。
4. Spot-check：随机抽样 Top500 中 20 组输出 `spotcheck_20.csv` 供人工或风险脚本快速评估。

示例（最新运行统计，2025-11-09）：

| 集合 | 数量 | Sharpe均值 | Sharpe中位数 | 年化收益均值 | 最大回撤中位 |
|------|------|-----------|--------------|--------------|--------------|
| Top2000 | 2000 | 0.950 | 0.952 | 0.1882 | -0.2044 |
| Top500  | 500  | 0.969 | 0.968 | 0.1902 | -0.1965 |
| Top200  | 200  | 0.971 | 0.968 | 0.1913 | -0.2006 |

脚本：`scripts/select_topk_candidates.py` （自动解析最新 run；必要参数 `--run-dir` 可覆盖）。

后续策略开发约定：

- 所有新策略实验、风控评估以 Top2000 作为固定候选集合输入，不再回退原始 IC 排序。
- 精简列表（Top500/Top200）仅用于加速人工审查与快速风险回归，不作为硬性白名单。
- 若监控期内预测与真实 Sharpe 相关性显著下降（Pearson <0.80 连续两窗口），需重新生成并审查 Top2000 构成；脚本输出对照供差异分析。

### 🔄 参数网格调优 (示例范围)

默认搜索：α ∈ {0.5, 0.6, 0.7}, retain ∈ {0.2, 0.3, 0.4}，K ∈ {50,100,200,500,1000,2000}。

选择标准（可团队共识）：
1. Precision@2000 ≥ 50%
2. 原始重叠率 ≥ 30%
3. Precision@100 不低于纯校准的 60%

满足标准的参数集合优先作为上线候选；如多个满足，选择重叠更高者以降低风格漂移。

---

## 🧹 清理与维护规范

清理对象：`htmlcov/`、`factor_engine.egg-info/`、历史 *.log、*.pid、`__pycache__/`、`.pytest_cache`、`.mypy_cache`、`.numba_cache`。

策略：
1. 日志与覆盖率目录 → 时间戳归档到 `archive/cleanup_<ts>/`。
2. 缓存与编译产物 → 直接删除（可重建）。
3. 保留最近 N=5 轮 `results/run_*`；其余归档。
4. 任何核心模块 (`core/`, `real_backtest/`) 变更需在 `docs/REFACTOR_NOTES.md` 添加说明。

自动化脚本：`scripts/cleanup_workspace.sh`（若不存在可根据上述策略新增）。

质量 Checklist：
- [x] 输出契约保持稳定
- [x] 性能优化冻结
- [x] 校准器相关性达标
- [x] 文档新增受守卫限制
- [x] 核心脚本入口明确

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
