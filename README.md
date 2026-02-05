# 深度量化0927 - ETF 轮动策略研究平台

> **最新版本**: v3.0 (高频轮动 Alpha)
> **发布日期**: 2025-12-01
> **核心文档**: [STRATEGY_BLUEPRINT_V3.md](STRATEGY_BLUEPRINT_V3.md) (👈 **必读：新一代策略蓝图**)
> **Python**: 3.11+
> **包管理**: [UV](https://docs.astral.sh/uv/) (v0.9+) **🔒 强制使用**

> **🚨 环境要求**：本项目强制使用 UV 包管理器（2025-12-15 更新）
> - ✅ **必须**: `uv run python <script>`
> - ❌ **禁止**: `pip install`, `python -m venv`, `source .venv/bin/activate`
> - 📖 **详见**: [CLAUDE.md](CLAUDE.md)

---

## 🚀 v3.0 重大更新 (2025-12-01)

**我们已从 v1.0 的"防御型"策略全面转向 v3.0 "高频进攻型"策略。**

- **收益率**: 121% (v1.0) -> **237% (v3.0)**
- **Calmar**: 0.66 (v1.0) -> **2.15 (v3.0)**
- **核心逻辑**: 3日轮动 + Top 2 集中持仓 + 无止损
- **详情请见**: [STRATEGY_BLUEPRINT_V3.md](STRATEGY_BLUEPRINT_V3.md)

### ⚠️ 关键发现 (v3.1 审计)

**现有 5 只 QDII 是策略的核心 Alpha 来源！**

| QDII ETF | 名称 | 贡献 | 胜率 |
|----------|------|------|------|
| 513500 | 标普500 | +25.37% | 68.9% |
| 513130 | 恒生科技(港元) | +23.69% | 53.3% |
| 513100 | 纳指100 | +22.03% | 61.3% |
| 159920 | 恒生指数 | +17.13% | 70.0% |
| 513050 | 中概互联 | +2.01% | 44.4% |
| **合计** | | **+90.23%** | **~60%** |

- 移除 QDII → 收益从 237% 降至 177%（损失 60pp）
- **详见**: [docs/ETF_POOL_ARCHITECTURE.md](docs/ETF_POOL_ARCHITECTURE.md)

---

## 🏆 v1.0 历史版本 (Legacy - 仅供参考)

> ⚠️ **注意**: 以下为 v1.0 历史配置，**当前生产环境使用 v3.0 参数**

**本项目 v1.0 版本已于 2025-11-28 封板**，核心策略已验证可复现。

<details>
<summary>📜 点击展开 v1.0 历史配置</summary>

### v1.0 核心策略：43 ETF 统一轮动

| 指标 | 数值 | 说明 |
|------|------|------|
| **策略类型** | 统一池轮动 | 43 只 ETF 统一排名选股 |
| **最佳因子组合** | `CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D` | 4 因子组合 |
| **回测收益率** | **121.02%** | 2020-01-01 至 2025-10-14 |
| **胜率** | 54.59% | 196 笔交易 |
| **盈亏比** | 1.414 | - |
| **FREQ** | 8 | ⚠️ 旧参数 |
| **POS_SIZE** | 3 | ⚠️ 旧参数 |

</details>

---

## 🎯 项目概述

本项目是专业级 **ETF 轮动策略研究平台**，以 WFO 为探索入口，并采用 **VEC + Rolling + Holdout + BT** 的四重验证交付标准（通过后封板归档），从因子挖掘到审计交付全流程覆盖。

| 状态 | 说明 |
|------|------|
| 🔒 v1.0 封板 | 核心策略已验证，可复现 |
| ✅ VEC/BT 对齐 | 差异 < 0.01 个百分点 |
| ✅ 无前视偏差 | `shift_timing_signal` 保证 |
| ✅ 生产就绪 | 43 只 ETF, 18 因子, 12,597 组合 |

---

## ⚡ 快速开始

### 1. 环境安装

```bash
# 安装 UV（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目后
cd /path/to/project
uv sync --dev              # 安装所有依赖
```

### 2. 生产工作流（复现 237% 收益）

```bash
# Step 1: WFO 筛选 - 从 12,597 组合中粗筛 (IC 仅做门槛)
uv run python src/etf_strategy/run_combo_wfo.py

# Step 2: VEC 精算 - 对 WFO 输出候选做向量化复算
uv run python scripts/run_full_space_vec_backtest.py

# Step 3: Rolling + Holdout - 无泄漏与一致性验证（产出 final_candidates）
uv run python scripts/final_triple_validation.py

# Step 4: BT 审计（Ground Truth）- 事件驱动审计，输出 Train/Holdout 分段收益
uv run python scripts/batch_bt_backtest.py

# Step 5: 生产包（交付口径统一用 BT）
uv run python scripts/generate_production_pack.py

# Step 6: 封板归档（冻结产物+脚本+配置+源码快照）
uv run python scripts/seal_release.py --help
```

### 3. 验证结果

最佳策略应为（v3.0）：
```
组合: ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARPE_RATIO_20D
收益率: 237.45%
胜率: 52.9%
盈亏比: 2.16
参数: FREQ=3, POS=2
```

### ⚠️ 运行命令规范

**所有 Python 脚本必须使用 `uv run` 前缀**：

```bash
# ✅ 正确方式
uv run python script.py

# ❌ 错误方式
python script.py
python3 script.py
```

---

## 🏗️ 引擎与交付架构

```
┌─────────────────────────────────────────────────────────────┐
│  WFO（探索入口，粗筛）                                       │
│  ├── 脚本: src/etf_strategy/run_combo_wfo.py                │
│  ├── 功能: 快速探索 + IC 门槛过滤（不做最终排序依据）        │
│  └── 输出: 候选组合（供后续精算/验证）                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  VEC（复算层，精算器）                                       │
│  ├── 脚本: scripts/run_full_space_vec_backtest.py           │
│  ├── 功能: 向量化精算收益/风险，用于高效筛选                 │
│  └── 注意: VEC 属于 Screening，不是对外最终口径              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Rolling + Holdout（稳定性与无泄漏验证）                     │
│  ├── 脚本: scripts/final_triple_validation.py               │
│  ├── 规则: Rolling gate 必须使用 train-only summary          │
│  └── 输出: final_candidates（无泄漏候选）                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  BT（审计层，Ground Truth）                                  │
│  ├── 脚本: scripts/batch_bt_backtest.py                     │
│  ├── 功能: 事件驱动审计 + 资金约束，输出 Train/Holdout 分段收益│
│  └── 输出: bt_results（生产口径统一以 BT 为准）              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  封板（Sealed Release）                                      │
│  ├── 工具: scripts/seal_release.py                          │
│  ├── 冻结: 产物 + 配置 + 关键脚本 + 源码快照 + 依赖锁定       │
│  └── 校验: CHECKSUMS.sha256（防篡改）                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
.
├── README.md                      # 📌 本文件（v1.1 说明）
├── CLAUDE.md                      # LLM 开发指南
├── pyproject.toml                 # 项目配置 + 依赖定义
├── uv.lock                        # 依赖锁文件
│
├── src/                           # ⭐ 源码目录（标准 src 布局）
│   ├── etf_strategy/              # 🎯 主力策略系统
│   │   ├── run_combo_wfo.py       #    WFO 入口脚本
│   │   ├── core/                  #    核心引擎（🔒 禁止修改）
│   │   │   ├── combo_wfo_optimizer.py    # 滚动 WFO 优化器
│   │   │   ├── precise_factor_library_v2.py  # 18 因子库
│   │   │   ├── backtester_vectorized.py  # VEC 回测引擎
│   │   │   └── shared_types.py           # 共享工具函数
│   │   └── auditor/               #    BT 审计模块
│   │       └── core/engine.py     #    Backtrader 策略
│   │
│   └── etf_data/                  # 📊 数据管理模块（独立）
│       ├── core/                  #    下载器核心
│       └── config/                #    配置管理
│
├── scripts/                       # 🔧 操作脚本
│   ├── batch_vec_backtest.py      #    VEC 批量回测
│   ├── batch_bt_backtest.py       #    BT 批量审计
│   └── full_vec_bt_comparison.py  #    VEC/BT 对比验证
│
├── results/                       # 📊 回测结果
│   ├── ARCHIVE_unified_wfo_43etf_best/   # 🏆 最佳 WFO 结果存档
│   └── ARCHIVE_vec_43etf_best/           # 🏆 最佳 VEC 结果存档 (121%)
│
├── docs/                          # 📚 文档
│   ├── BEST_STRATEGY_43ETF_UNIFIED.md    # 🏆 最佳策略详细文档
│   └── INDEX.md                          # 文档索引
│
├── configs/                       # ⚙️ 全局配置
└── raw/                           # 💾 原始数据
```

---

## 📊 核心参数（v3.0 生产配置）

| 参数 | v3.0 值 | v1.0 值 | 说明 |
|------|---------|---------|------|
| `FREQ` | **3** | 8 | 调仓频率（交易日）|
| `POS_SIZE` | **2** | 3 | 持仓数量 |
| `INITIAL_CAPITAL` | 1,000,000 | 同 | 初始资金 |
| `COMMISSION` | 0.0002 | 同 | 手续费率 (2bp) |
| `LOOKBACK` | 252 | 同 | 回看窗口（交易日） |
| **ETF 数量** | **43** | 43 | 38 A股 + 5 QDII ⚠️ |
| **因子数量** | 18 | 18 | PreciseFactorLibrary |
| **最佳因子组合** | ADX + MAX_DD + PP_120D + PP_20D + SHARPE | CORR + MAX_DD + PP_120D + PP_20D | 5因子 vs 4因子 |

> ⚠️ **关键**: 5 只 QDII 贡献 +90% 收益，禁止移除！详见 [docs/ETF_POOL_ARCHITECTURE.md](docs/ETF_POOL_ARCHITECTURE.md)

---

## 🔒 v3.0 封板规则

### ✅ 允许的修改

- Bug 修复（不改变策略逻辑）
- 数据源适配
- 文档完善
- 性能优化（不改变结果）
- 数据更新（新日期数据）

### ❌ 禁止的修改

- 修改核心因子库
- 修改回测引擎逻辑
- **修改参数默认值 (FREQ=3, POS=2)**
- **修改 ETF 池定义（特别是 5 只 QDII）**
- 删除 ARCHIVE 存档

## 🚫 「简化版 / 备份脚本」禁令

- **唯一的 WFO 入口** 是 `src/etf_strategy/run_combo_wfo.py`。任何 `run_unified*`、`direct_factor*`、`*_simple*`、`*_simplified*` 等脚本一律视为违规，不得出现在生产目录。
- `scripts/archive/` 与 `docs/archive/` 保留历史记录，但被视为冷冻区；其中的脚本或文字不得重新引用到生产流程。
- `scripts/ci_checks.py` 已内置「简化版探测」扫描，如发现违规文件会立即失败，严禁通过跳过检查的方式引入非滚动 WFO 实现。
- 若确因研究需要临时恢复旧脚本，必须先将文件移动到 `backup_*/` 类隔离目录，并在任务结束后彻底清除。

---

## 📖 关键文档

| 文档 | 描述 |
|------|------|
| [docs/BEST_STRATEGY_43ETF_UNIFIED.md](docs/BEST_STRATEGY_43ETF_UNIFIED.md) | 🏆 **最佳策略详细文档** |
| [docs/INDEX.md](docs/INDEX.md) | 文档索引 |

---

## ⚠️ 开发注意事项

1. **Set 遍历不确定性**：使用 `sorted()` 确保有序
2. **前视偏差**：使用 `shift_timing_signal` 滞后信号
3. **调仓日程**：使用 `generate_rebalance_schedule` 统一
4. **浮点精度**：比较时使用 0.01% 容差
5. **资金时序**：BT 中使用卖出后现金计算

---

## 📜 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| **v3.1** | 2025-12-01 | 🔬 ETF 池深度审计：确认 5 只 QDII 贡献 90%+ 收益 |
| **v3.0** | 2025-12-01 | 🚀 高频策略升级：FREQ=3, POS=2, 收益 237.45% |
| v1.1 | 2025-11-30 | 架构重构：统一 `src/` 布局 |
| v1.0 | 2025-11-28 | 🔒 统一策略 121.02% 验证通过 |
| v0.9 | 2025-11-16 | VEC/BT 对齐完成 |
| v0.8 | 2025-11-09 | 性能优化冻结 |

---

**维护者**: 深度量化团队 | **License**: MIT | **🔒 v3.1 策略封板 | 237.45% 收益已验证**
