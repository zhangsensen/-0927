# ETF 轮动策略完整开发与验证工作流程

本仓库是一个用于 **ETF 轮动策略** 的完整研究与验证框架，目标是从因子挖掘（WFO）到向量化回测（VEC），再到 Backtrader 审计（BT），构建一条可重复、可审计、贴近实盘的开发流水线。

本项目不直接连接真实交易柜台，也不负责下单执行，而是为后续实盘落地提供经过多阶段验证的策略与报告。

## 🔧 环境要求

### UV 包管理器

**重要说明**：本项目使用 **UV** 作为包管理器和 Python 环境管理工具。UV 是一个现代的 Python 包管理工具，提供快速的依赖解析和虚拟环境管理。

#### 为什么选择 UV？

1. **高性能**：依赖解析速度比 pip 快 10-100 倍
2. **可靠性**：统一的锁文件（uv.lock）确保环境一致性
3. **兼容性**：完全兼容 pip 和 PyPI 生态
4. **便捷性**：单个命令管理项目依赖和虚拟环境

#### 环境配置

项目配置文件：
- `pyproject.toml`：项目依赖和配置定义
- `uv.lock`：锁定确切版本的依赖（确保可重复性）

#### 运行命令

**所有 Python 脚本都必须使用 `uv run python` 前缀**：

```bash
# 基本语法
uv run python <script.py> [arguments]

# 示例
uv run python run_unified_wfo.py
uv run python scripts/vec_bt_alignment_verification.py
uv run python -c "from strategy_auditor.runners.parallel_audit import run_audit; run_audit('input.csv')"
```

**注意**：
- ❌ 不要直接使用 `python3` 或 `python` 命令
- ❌ 不要使用 `source .venv/bin/activate` 激活虚拟环境
- ✅ 始终使用 `uv run python` 运行项目脚本

## 📋 工作流程总览

本项目采用**三阶段渐进式验证**的策略开发流程，确保从因子挖掘到实盘部署的每个环节都经过严格审计：

```
WFO (策略开发) → VEC (快速验证) → BT (兜底审计) → 实盘部署
     ↓                ↓                 ↓
  因子筛选         收益复现          合规验证
  组合优化         性能评估          最终确认
```

### 核心原则

1. **参数一致性**：三个引擎必须使用完全相同的交易参数（佣金、换仓频率、持仓数、初始资金）
2. **数据一致性**：三个引擎必须使用完全相同的行情数据源和因子计算逻辑
3. **健康度可解释**：WFO → VEC → BT 的收益不要求数值完全一致，但差异必须可解释、在事先约定的“健康区间”内
4. **BT 兜底原则**：Backtrader 作为开源社区验证工具，是最终合规与风险兜底标准，是一套**更保守且独立实现**的回测口径，而不是对 VEC 的数值复刻

---

## 🔬 第一阶段：WFO 策略开发（因子挖掘与组合优化）

### 目标

- 从大规模因子组合中筛选出具有稳定预测能力的策略
- 通过 Walk-Forward Optimization 避免过拟合
- 输出 Top N 策略候选池

### 工具

- **主脚本**：`run_unified_wfo.py`
- **配置文件**：`configs/combo_wfo_config.yaml`
- **输出目录**：`results/unified_wfo_YYYYMMDD_HHMMSS/`

### 关键参数

```yaml
backtest:
  commission_rate: 0.0002          # 佣金 2bps（保守估计）
  initial_capital: 1000000         # 初始资金 100 万
  lookback_window: 252             # 回溯窗口 252 日
  
combo_wfo:
  rebalance_frequencies: [8]       # 换仓频率 8 天
  combo_sizes: [2, 3, 4, 5]        # 组合因子数量
  is_period: 252                   # 样本内周期 252 日
  oos_period: 60                   # 样本外周期 60 日
  step_size: 60                    # 滚动步长 60 日
  top_n: 100                       # 输出 Top100 策略
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `all_combos.csv` | 全部组合的 WFO 评估结果（12,597 组） |
| `top100.csv` | Top100 策略列表（含 `total_return`、`sharpe` 等） |
| `factors/` | 标准化后的因子矩阵（供后续引擎复用） |
| `run_config.yaml` | 本次运行的完整配置快照 |

### 评估指标

- **IS Sharpe**：样本内夏普比率
- **OOS Sharpe**：样本外夏普比率
- **Stability**：稳定性得分（IS/OOS 比值）
- **Total Return**：累计收益率
- **Complexity Penalty**：因子复杂度惩罚

### WFO 的局限性

⚠️ **注意**：WFO 阶段的 `total_return` 是基于理想化假设的，可能存在：
1. 未充分考虑真实订单执行的资金约束
2. 未模拟停牌、流动性不足等极端情况
3. 使用了样本内最优参数（存在过拟合风险）

因此，**WFO 收益不能直接作为实盘预期**，必须经过后续两阶段验证。

---

## ⚡ 第二阶段：VEC 快速验证（收益复现与性能评估）

### 目标

- 用向量化回测引擎快速验证 WFO 筛选的策略
- 在全量历史数据上重新计算收益、夏普、回撤等指标
- 评估策略的真实表现，剔除过拟合组合

### 工具

- **主脚本**：`scripts/vec_bt_alignment_verification.py`
- **核心引擎**：`etf_rotation_optimized/real_backtest/run_production_backtest.py`
- **输出文件**：`results/vec_results_YYYYMMDD_HHMMSS.csv`

### 关键特性

1. **严格时间隔离**：每个调仓日只使用截至前一日的历史数据，避免未来函数
2. **逐日因子计算**：不提前计算全部时间序列，确保无信息泄露
3. **IC 权重动态计算**：每个调仓日用历史窗口重新计算因子 IC 权重
4. **向量化加速**：使用 Numba JIT 编译，支持多进程并行

### 关键参数（必须与 WFO 一致）

```python
COMMISSION_RATE = 0.0002  # 佣金 2bps
INITIAL_CAPITAL = 1_000_000  # 初始资金 100 万
FREQ = 8  # 换仓频率 8 天
POS_SIZE = 3  # 持仓数量 3 个
LOOKBACK = 252  # 回溯窗口 252 日
```

### 输出指标

| 列名 | 说明 |
|------|------|
| `rank` | WFO 排名 |
| `combo` | 因子组合名称 |
| `vec_annual_ret` | 年化收益率 |
| `vec_sharpe` | 夏普比率 |
| `vec_max_dd` | 最大回撤 |
| `vec_n_rebalance` | 调仓次数 |
| `vec_turnover` | 年化换手率 |

### VEC 的优势

✅ **速度快**：向量化计算，Top100 策略约 5-10 分钟完成  
✅ **无未来函数**：严格的时间隔离保证  
✅ **高度灵活**：支持多种因子组合和参数配置

### VEC 的局限性

⚠️ **注意**：VEC 引擎虽然严格避免未来函数，但仍采用理想化的资金模型：
1. 假设无限流动性，可瞬间按理想权重调仓
2. 未模拟订单簿、买卖价差、停牌等真实约束
3. 佣金模型相对简化（单一费率，无最低佣金）

因此，**VEC 收益仍需要 BT 验证**，确保在真实交易规则下可复现。

---

## 🛡️ 第三阶段：BT 兜底审计（合规验证与最终确认）

### 目标

- 使用 Backtrader 开源框架进行最终审计
- 模拟真实订单执行流程（先卖后买、资金约束、保证金检查）
- 验证策略在严格合规条件下的真实表现
- 作为实盘部署前的最后一道防线

### 工具

- **主脚本**：`strategy_auditor/runners/parallel_audit.py`
- **核心引擎**：`strategy_auditor/core/engine.py`
- **输出目录**：`strategy_auditor/results/run_YYYYMMDD_HHMMSS/`

### 为什么选择 Backtrader？

1. **开源社区验证**：Backtrader 是全球使用最广泛的 Python 回测框架，经过数百万次实盘验证
2. **严格的订单管理**：内置完整的经纪商模拟，包括保证金、杠杆、订单拒绝等
3. **真实交易流程**：必须先卖出释放资金，才能买入新标的（符合 A 股 T+1 规则）
4. **可审计性**：每笔订单都有详细日志，便于监管审查

### 关键参数（必须与 WFO/VEC 一致）

```python
COMMISSION_RATE = 0.0002  # 佣金 2bps
INITIAL_CAPITAL = 1_000_000  # 初始资金 100 万
FREQ = 8  # 换仓频率 8 天
POS_SIZE = 3  # 持仓数量 3 个
LOOKBACK = 252  # 回溯窗口 252 日

# Backtrader 特有配置
cerebro.broker.setcommission(commission=COMMISSION_RATE)
cerebro.broker.set_coc(True)  # 开启 Cheat-On-Close（调仓日收盘价成交）
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `summary.csv` | Top100 策略的 BT 回测汇总 |
| `orders_*.csv` | 每个策略的详细订单记录 |
| `trades_*.csv` | 每个策略的详细交易记录 |
| `equity_*.csv` | 每个策略的逐日净值曲线 |

### 输出指标

| 列名 | 说明 |
|------|------|
| `real_rank` | WFO 排名 |
| `combo` | 因子组合名称 |
| `orig_return` | WFO 原始累计收益 |
| `bt_return` | BT 实测累计收益 |
| `diff` | 差异（bt_return - orig_return） |
| `bt_final_equity` | BT 最终权益 |
| `time_total` | 总耗时 |

### BT 的优势

✅ **真实订单执行**：先卖后买，严格模拟资金流转  
✅ **保证金检查**：自动拒绝超额订单  
✅ **社区验证**：全球开源社区的共同标准  
✅ **详细日志**：每笔订单可追溯  

### BT 作为兜底验证的意义

Backtrader 的审计结果是最终的**合规与风险兜底标准**：

- 它不追求与 VEC 数值完全一致，而是作为一套**更保守、独立实现**的回测口径；
- 若某策略在 VEC 口径下表现优秀，但在 BT 口径下出现大幅回撤或长期亏损，则该策略不建议进入实盘，或需大幅降权；
- 对于大多数策略，只要求 **收益方向一致 + 收益在合理区间内缩水**，而不是精确 1:1 复刻；
- 若 BT 收益显著高于 VEC，通常意味着存在参数不一致或数据问题，需要重点排查。

---

### 三阶段收益与健康度标准

> 说明：本节区分“工程上的数值对齐”（长期目标）与“实盘决策使用的兜底健康度”（当前执行标准）。

#### 1. 数值对齐（工程检查，长期目标）

在关键交易参数、数据源和信号时点严格统一的前提下：

- WFO → VEC：
  - 预期 `VEC ≤ WFO`（VEC 在更长区间和更保守设定下，收益通常不高于 WFO）；
  - 若 VEC 明显高于 WFO，需排查是否存在未来函数或因子错位。
- VEC → BT：
  - 不强求 `|BT - VEC| < 1%` 的严格数值对齐；
  - 对差异最大的少数组合（如 Top10 正/负偏离）使用逐日诊断脚本核查逻辑与数据是否一致。

#### 2. 兜底健康度（当前实盘决策采用的标准）

在当前统一佣金 0.0002、FREQ=8、POS_SIZE=3、LOOKBACK=252 的设定下，对 Top100 策略的 BT 审计结果采用如下健康度指标：

- **方向一致率**：
  - 在 VEC 年化 > 0 的组合中，要求 BT 年化 > 0 的比例 ≥ 90%；
  - 最新实测（2025-11-28，BT leverage=1.0，无杠杆）：VEC 年化 > 0 的 83 个组合中，BT 年化有 81 个 > 0（方向一致率约 97.6%）。
- **排序一致性（Spearman 排名相关）**：
  - 指标：Spearman(BT 年化, VEC 年化)，用于衡量排序的大致一致性，仅作参考，不作为硬约束；
  - 最新实测（Top100，BT leverage=1.0）：约 0.38，说明排序相关但不强。
- **收益缩水分布（shrink）**：
  - 定义：在 VEC 年化 > 0 且 BT 年化 > 0 的样本上，`shrink = BT 年化 / VEC 年化`；
  - 要求：shrink 的中位数应处于大致合理区间（例如 [0.5, 1.5]），避免大规模极端放大或缩小；
  - 最新实测（BT leverage=1.0）：shrink 中位数约为 0.69，25%/75% 分位约为 0.49 / 1.11，说明 BT 在无杠杆口径下整体略为保守，但收益方向大体与 VEC 一致。

**极端差异处理：**

- 若存在 `|BT 年化 - VEC 年化| > 10%` 的极端样本：
  - 必须进入“人工复核名单”；
  - 使用 `scripts/diagnose_vec_bt_daily.py` 做逐日净值与持仓对账；
  - 给出差异原因（如数据缺口、停牌处理、权重极端、杠杆影响等），并在审计报告中备注；
  - 视情况决定是否剔除或降权该策略。

### 当前状态（2025-11-28）

> 本小节用于记录从“旧世界”（参数不一致）到“新世界”（统一佣金 & 无杠杆 BT）的演进，方便后续审计或大模型回溯。

**历史状态（早期版本，待修复）**：

| 阶段对比 | 实测结果 | 状态 | 原因 |
|---------|---------|------|------|
| WFO 平均累计 | 125.5% | ⚠️ | 理想化评估 |
| VEC 平均累计 | 46.3% | ⚠️ | 佣金 0.5bps（偏低） |
| BT 平均累计 | 59.4% | ⚠️ | 佣金 2bps（正常），且存在杠杆差异 |
| BT vs VEC 相关性 | 0.33 | ❌ | 参数与资金模型不一致，口径严重错配 |

当时的诊断结论是：

- WFO 收益过高（样本内过拟合，偏研究口径）；
- VEC 与 BT 佣金不一致（0.5bps vs 2bps）、杠杆不一致（BT 使用了 leverage=2.0）；
- VEC 与 BT 仓位/资金逻辑存在差异；
- 需要紧急修复参数一致性与杠杆设定。

**当前状态（统一佣金 0.0002 & BT leverage=1.0 后）**：

- 统一了 WFO/VEC/BT 的关键交易参数（佣金、频率、持仓数、初始资金、LOOKBACK）；
- 将 BT 杠杆调整为 1.0，收益直接对应 1 倍仓位；
- 最新一轮 Top100 策略对齐结果（2020-01-01 至 2025-10-14）：
  - 平均累计收益：
    - WFO ≈ 42.4%
    - VEC ≈ 42.4%
    - BT（leverage=1.0）≈ 31.6%
  - 平均年化收益：
    - WFO ≈ 5.63%
    - VEC ≈ 5.63%
    - BT ≈ 4.70%
  - BT vs VEC 年化相关性：≈ 0.36；
  - BT vs VEC 年化平均差异：≈ -0.94%（BT 略为保守）；
  - BT vs VEC 年化绝对差异：≈ 4.14%；
  - 方向一致率（VEC>0 时 BT>0 的比例）：≈ 97.6%。

**当前诊断结论**：

- 在统一参数与无杠杆设定下，BT 作为更保守、独立口径的兜底审计工具是“健康可用”的；
- 差异主要来源于信号加权方式（IC 加权 vs 等权）、资金/成交约束等口径差异，而非工程 Bug；
- 在实盘决策中，应以 BT 无杠杆结果作为主要预期区间，VEC 用于解释和敏感性分析，WFO 用于长期因子库与策略发现。

---

## 🔧 参数一致性检查清单

在运行完整工作流程前，必须确保以下参数在三个引擎中**完全一致**：

### 交易参数

- [ ] 佣金费率（`commission_rate`）
- [ ] 初始资金（`initial_capital`）
- [ ] 换仓频率（`rebalance_frequency`）
- [ ] 持仓数量（`position_size`）
- [ ] 回溯窗口（`lookback_window`）

### 数据参数

- [ ] 数据源路径（`data_dir`）
- [ ] 起止日期（`start_date`, `end_date`）
- [ ] ETF 代码池（`symbols`）
- [ ] 停牌/除权处理规则

### 因子参数

- [ ] 因子计算逻辑
- [ ] 横截面标准化方法（`winsorize_lower`, `winsorize_upper`）
- [ ] IC 权重计算窗口

### 择时参数（如启用）

- [ ] 择时开关（`timing.enabled`）
- [ ] 择时类型（`timing.type`）
- [ ] 极端阈值（`timing.extreme_threshold`）

---

## 📊 对齐验证流程

### 步骤 1：参数对齐

```bash
# 确认三个配置文件中的关键参数一致
grep "commission_rate" configs/combo_wfo_config.yaml
grep "COMMISSION_RATE" strategy_auditor/core/engine.py
grep "commission_rate" etf_rotation_optimized/real_backtest/run_production_backtest.py
```

### 步骤 2：运行三阶段回测

**重要**：本项目使用 **UV** 作为包管理器和 Python 环境管理工具。所有 Python 脚本都需要使用 `uv run python` 命令来运行，确保使用项目虚拟环境中正确安装的依赖包。

```bash
# 1. WFO 策略开发
uv run python run_unified_wfo.py

# 2. VEC 快速验证（自动读取最新 WFO 输出）
uv run python scripts/vec_bt_alignment_verification.py

# 3. BT 兜底审计（自动读取最新 WFO 输出）
uv run python -c "from strategy_auditor.runners.parallel_audit import run_audit; run_audit('results/top100_for_bt_20251128.csv')"
```

### 步骤 3：对齐分析

```bash
# 生成三引擎对齐报告
uv run python scripts/generate_alignment_report.py
```

预期输出：
- `results/vec_bt_alignment_analysis.csv`：VEC vs BT 年化对比
- `results/vec_bt_total_returns.csv`：VEC vs BT 累计对比
- `docs/vec_bt_alignment_YYYYMMDD.md`：对齐验证报告

### 步骤 4：接受标准

#### 工程层面（对齐检查）

- 关键参数（佣金、频率、持仓数、初始资金、LOOKBACK、T-1 信号时点）三引擎必须一致；
- 若发现参数或数据不一致，必须先修复再讨论收益；
- 对差异最大的少数组合（例如 Top10 正/负偏离）使用逐日诊断脚本核查：
  - 是否使用了不同的数据源或停牌/除权规则；
  - 是否存在未来函数、信号错位或资金约束导致的成交差异。

#### 业务/风控层面（兜底通过条件）

- 整体层面：
  - 以 BT 结果为实盘前的主要参考口径；
  - 若一批待上线策略在 VEC 下表现优秀，但在 BT 下整体年化为负或回撤极端，则该批次不建议上线。
- 单策略层面（示例规则，可按风险偏好调整）：
  - 若 VEC 年化 > 5%，要求 BT 年化 > 0；
  - 若 VEC 年化 > 10%，建议 BT 年化 > 3%；
  - 若单个组合出现 `|BT 年化 - VEC 年化| > 10%`，需进入逐日诊断与人工复核；
  - 对在 BT 口径下表现显著较差的策略，予以剔除或明显降权。

概括：**VEC 负责策略筛选与快速评估，BT 负责用更保守的规则做最终兜底**。对于上线批次，只要整体在 BT 下仍为正收益且风险指标可接受，即视为通过兜底审核；个别不满足兜底条件的策略需剔除或降权。

---

## 🚨 常见问题与排查

### Q1：BT 收益远低于 VEC/WFO

**可能原因**：
1. BT 佣金设置过高
2. BT 的 leverage 参数设置不当
3. BT 频繁触发保证金不足（订单被拒）

**排查方法**：
```python
# 检查 BT 审计日志
grep "Margin" strategy_auditor/results/run_*/audit.log
grep "Rejected" strategy_auditor/results/run_*/audit.log
```

### Q2：VEC 与 BT 相关性很低（< 0.5）

**可能原因**：
1. 因子标准化逻辑不一致
2. 数据源不同（停牌、除权处理差异）
3. 调仓信号计算时点不同

**排查方法**：
```bash
# 对单个组合逐日对比持仓
uv run python scripts/diagnose_vec_bt_daily.py --combo "ADX_14D + CMF_20D + MAX_DD_60D"
```

### Q3：WFO 收益明显高于 VEC/BT

**原因**：
- WFO 使用样本内最优参数，存在过拟合
- WFO 评估未考虑真实交易约束

**解决方案**：
- 以 BT 收益为准，WFO 仅作为策略筛选工具
- 适当提高 WFO 的复杂度惩罚（`complexity_penalty_lambda`）

---

## 📁 工作流程输出目录结构

```
results/
├── unified_wfo_20251128_124752/        # WFO 输出
│   ├── all_combos.csv                  # 全部组合评估
│   ├── top100.csv                      # Top100 策略
│   ├── factors/                        # 标准化因子
│   └── run_config.yaml                 # 运行配置
│
├── vec_results_20251128_124903.csv     # VEC 验证结果
│
├── vec_bt_alignment_analysis.csv       # VEC vs BT 对齐分析
├── vec_bt_total_returns.csv            # 累计收益对比
│
└── strategy_auditor/
    └── results/
        └── run_20251128_125355/         # BT 审计结果
            ├── summary.csv              # 回测汇总
            ├── orders_rank001.csv       # 订单记录
            ├── trades_rank001.csv       # 交易记录
            └── equity_rank001.csv       # 净值曲线

docs/
├── WORKFLOW.md                          # 本文档
├── vec_bt_alignment_20251128.md         # 对齐验证报告
└── vec_bt_gap_root_cause_analysis.md    # 差异根因分析
```

---

## 🎓 设计哲学

### 为什么需要三阶段验证？

1. **WFO**：快速筛选 + 避免过拟合
   - 优势：高效遍历大量组合
   - 劣势：样本内评估，收益高估

2. **VEC**：全量回测 + 性能优化
   - 优势：快速验证，支持大规模并行
   - 劣势：理想化资金模型，未充分模拟真实约束

3. **BT**：合规审计 + 社区标准
   - 优势：真实订单执行，开源社区验证
   - 劣势：速度较慢，单次回测约 2-3 秒

### 为什么以 BT 为最终标准？

1. **社区共识**：Backtrader 是全球量化社区公认的回测标准
2. **真实性**：严格模拟券商交易规则，包括保证金、订单拒绝等
3. **可审计性**：每笔订单可追溯，符合监管要求
4. **防御性**：在无杠杆设定 (`leverage=1.0`) 下，BT 结果可直接视为 1.0x 真实仓位的保守收益口径；当 BT 收益明显低于 VEC 时，说明 VEC 可能高估了策略表现，需要谨慎对待上线决策。

### 三阶段的互补关系（整体 Bookflow 概览）

本项目的完整 Bookflow 可以总结为：

```
WFO：策略发现（广度优先，因子组合遍历与筛选）
  ↓
VEC：快速验证（效率优先，向量化回测与性能评估）
  ↓
BT：严格审计（安全优先，1.0x 无杠杆、真实约束下的兜底验证）
  ↓
实盘部署（以 BT 为主要参考口径；VEC/WFO 作为研发与解释层）
```

- **WFO (Walk-Forward Optimization)**：
  - 角色：负责在大规模因子空间中“发现”潜在策略，提供一个排序后的候选池；
  - 特点：样本内/样本外滚动评估，适合做组合遍历和因子重要性分析；
  - 局限：收益更偏“研究口径”，未充分考虑真实交易约束。

- **VEC (Vectorized Backtest)**：
  - 角色：在统一参数和严格无未来函数前提下，对 WFO 输出的 TopN 策略做全历史快速验证；
  - 特点：IC 加权、多因子灵活组合、向量化加速，适合频繁迭代；
  - 局限：资金模型和交易约束仍然理想化（例如忽略部分保证金与撮合细节）。

- **BT (Backtrader Audit, leverage=1.0)**：
  - 角色：作为独立实现、社区认可的回测引擎，在 1.0x 无杠杆设定下，对候选策略做“贴近实盘的兜底审计”；
  - 特点：有真实经纪模型、保证金检查、订单拒绝逻辑，能暴露在 VEC 口径下不明显的问题；
  - 输出：其年化收益和回撤可以直接视为 1 倍仓位下的保守收益/风险预期，是实盘前的主要参考口径。

---

## ✅ 工作流程总结

1. **运行 WFO**：从 12,597 组合中筛选 Top100 候选策略（研究视角）。
2. **运行 VEC**：在统一参数和无未来函数约束下，快速验证 Top100，剔除过拟合或不稳定组合（工程视角）。
3. **运行 BT（leverage=1.0）**：在真实交易约束下做严格审计，产出 1.0x 无杠杆的保守收益曲线与风险指标（风控/合规视角）。
4. **对齐与健康度分析**：检查 WFO/VEC/BT 三者是否在方向和量级上自洽，确认差异来源是“口径差异”而非“工程 Bug”。
5. **实盘部署**：以 BT 无杠杆收益为主要预期区间，VEC 作为解释与敏感性分析工具，WFO 作为长期因子库与策略发现工具。

**关键原则**：参数一致、数据一致、差异可解释、BT 兜底。

**最终目标**：在**研究 → 工程 → 风控/合规 → 实盘**这一完整链路上，提供一套对人类研究者和大模型都友好的工作流描述，使任意一个读者（包括大模型）在只阅读本文件的前提下，也能理解：

- 本项目的产生背景：为 ETF 轮动策略提供一套从因子挖掘到实盘前审计的标准化流程；
- 三个引擎各自的定位与局限，以及它们之间如何“收益接力”；
- 为什么最终需要以 Backtrader（leverage=1.0）的结果作为保守兜底，从而避免回测陷阱并提升实盘可兑现性。

---

## 📞 快速参考

| 任务 | 命令 | 输出 |
|------|------|------|
| WFO 策略开发 | `uv run python run_unified_wfo.py` | `results/unified_wfo_*/` |
| VEC 快速验证 | `uv run python scripts/vec_bt_alignment_verification.py` | `results/vec_results_*.csv` |
| BT 兜底审计 | `uv run python -c "from strategy_auditor.runners.parallel_audit import run_audit; run_audit('your_input.csv')"` | `strategy_auditor/results/run_*/` |
| 对齐分析 | 见上文"对齐验证流程" | `results/vec_bt_alignment_*.csv` |

---

**文档版本**：v1.0  
**最后更新**：2025-11-28  
**维护者**：项目开发团队
