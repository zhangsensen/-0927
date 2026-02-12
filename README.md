# 深度量化0927 - ETF 轮动策略研究平台

> **生产版本**: v5.0-prod1 (FREQ=5 + Exp4 Hysteresis)
> **实盘启动**: 2025-12-18
> **Python**: 3.11+
> **包管理**: [UV](https://docs.astral.sh/uv/) **强制使用**

---

## 实盘成绩 (2025-12-18 ~ 2026-02-09)

| 指标 | 数值 |
|------|------|
| **累计收益** | +6.37% (+49,178 CNY) |
| **交易次数** | 22 笔 |
| **胜率** | 83.3% |
| **盈亏比** | 2.33 |
| **最大回撤** | 可控 |

> 实盘 6 周全部为 A 股交易，零 QDII 交易。经审计确认为市场环境导致（A 股强、美股弱），策略自适应选股正确。

---

## 项目概述

专业级 **ETF 轮动策略研究平台**，覆盖 A 股 + QDII 双市场，采用三层引擎（WFO → VEC → BT）从因子筛选到封板归档全流程。

核心特性：
- **三层引擎架构**：WFO 粗筛 → VEC 精算 → BT 审计
- **Exp4 迟滞控制**：delta_rank + min_hold_days 抑制过度换仓
- **冻结参数系统**：`frozen_params.py` 强制校验，防止配置漂移
- **状态持久化**：信号生成器带 schema 校验和环境不匹配冷启动保护
- **Regime Gate**：基于 510300 波动率的动态仓位调节

---

## 生产参数 (v5.0)

由 `src/etf_strategy/core/frozen_params.py` 强制执行，覆盖 WFO/VEC/BT 三层入口。

| 参数 | 值 | 说明 |
|------|-----|------|
| `FREQ` | 5 | 每 5 个交易日调仓 |
| `POS_SIZE` | 2 | 持有 2 只 ETF |
| `COMMISSION` | 0.0002 | 手续费 2bp |
| `LOOKBACK` | 252 | 回看窗口 1 年 |
| `delta_rank` | 0.10 | 迟滞：rank01 差值门槛 |
| `min_hold_days` | 9 | 迟滞：最小持仓天数 |
| ETF 池 | 43 只 | 38 A 股 + 5 QDII |
| Universe 模式 | `A_SHARE_ONLY` | QDII 硬阻断实盘交易 |
| Regime Gate | ON | 波动率模式，510300 代理 |

### 封板策略

| 策略 | 因子组合 | 全期收益 | Sharpe | MDD |
|------|---------|---------|--------|-----|
| **S1 (4F)** | ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D | 136.52% | 1.03 | 15.47% |
| **S2 (5F)** | S1 + PRICE_POSITION_120D | 129.85% | 1.04 | 13.93% |

---

## 快速开始

### 1. 环境安装

```bash
# 安装 UV（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆并安装
git clone git@github.com:zhangsensen/-0927.git
cd -0927
uv sync --dev
```

### 2. 核心命令

```bash
# Makefile 快捷方式
make wfo                    # WFO 筛选 (~2min)
make vec                    # VEC 回测 (~5min)
make bt                     # BT 审计 (~30-60min)
make pipeline               # 完整流水线 WFO → VEC → BT (~76s)

# 日常操作
uv run python scripts/generate_today_signal.py              # 每日交易信号（带状态持久化）
uv run python scripts/update_daily_from_qmt_bridge.py --all # 从 QMT 更新数据

# 代码质量
make format                 # black + isort
make lint                   # ruff + mypy
make test                   # pytest (157 cases)
make clean-numba            # 清理 Numba JIT 缓存
```

> **所有 Python 脚本必须使用 `uv run python` 前缀，禁止 `pip` / `python` / `python3` 裸执行。**

---

## 三层引擎架构

```
┌──────────────────────────────────────────────────────┐
│  WFO — 因子组合粗筛 (~2min)                           │
│  脚本: src/etf_strategy/run_combo_wfo.py              │
│  方法: 12,597 组合 → IC 门槛 (≥0.05) → 复合评分排名    │
│  评分: Return(40%) + Sharpe(30%) + MaxDD(30%)         │
└──────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────┐
│  VEC — 向量化精算 (~5min)                             │
│  脚本: scripts/batch_vec_backtest.py                  │
│  方法: Numba JIT 内核，float 份额，带迟滞状态机        │
└──────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────┐
│  BT — 事件驱动审计 (~30-60min)                        │
│  脚本: scripts/batch_bt_backtest.py                   │
│  方法: Backtrader 引擎，整手约束 + 资金限制            │
│  地位: Ground Truth，生产口径以 BT 为准                │
└──────────────────────────────────────────────────────┘
```

**VEC/BT 对齐**：基线中位差 ~4.8pp（float 份额 vs 整手的系统差异）。超过 20pp 需停机排查。

---

## 项目结构

```
.
├── src/
│   ├── etf_strategy/               # 主力策略系统
│   │   ├── run_combo_wfo.py        #   WFO 入口
│   │   ├── core/                   #   核心引擎
│   │   │   ├── frozen_params.py    #     冻结参数系统 (v3.4~v5.0)
│   │   │   ├── hysteresis.py       #     Exp4 迟滞内核 (@njit)
│   │   │   ├── cost_model.py       #     分市场成本模型
│   │   │   ├── execution_model.py  #     T+1 Open 执行模型
│   │   │   ├── regime_detector.py  #     Regime Gate 波动率检测
│   │   │   ├── precise_factor_library_v2.py  # 17 因子库
│   │   │   └── utils/rebalance.py  #     调仓工具 (防前视 + 统一日程)
│   │   └── auditor/core/engine.py  #   Backtrader 策略引擎
│   │
│   └── etf_data/                   # 数据管理模块（独立）
│
├── scripts/                        # 操作脚本
│   ├── generate_today_signal.py    #   每日信号生成（状态持久化）
│   ├── batch_vec_backtest.py       #   VEC 批量回测
│   ├── batch_bt_backtest.py        #   BT 批量审计
│   ├── run_full_pipeline.py        #   完整流水线
│   ├── run_exp4_grid.py            #   Exp4 迟滞网格搜索
│   ├── run_sealed_cost_audit.py    #   封板策略成本审计
│   ├── audit_qdii_ranking.py       #   QDII 排名审计
│   └── update_daily_from_qmt_bridge.py  # QMT 数据更新
│
├── configs/
│   └── combo_wfo_config.yaml       # 单一配置源 (43 ETF + 18 因子 + 全参数)
│
├── sealed_strategies/              # 封板归档 (v3.1 ~ v5.0)
├── tests/                          # 测试 (114 cases)
├── docs/                           # 文档
└── Makefile                        # 快捷命令
```

---

## 迟滞状态机 (Exp4)

v5.0 核心创新——通过换仓迟滞控制降低交易频率，提升持仓稳定性。

**规则**：
1. 每次调仓最多换 1 只（强制）
2. 新候选 rank01 差值必须 ≥ delta_rank (0.10) 才能替换
3. 当前持仓天数必须 ≥ min_hold_days (9) 才允许换出

**效果**：换手率从 35x 降至 ~14.6x，持仓存活率提升 3 倍。

**状态持久化**：信号生成器将持仓组合和持仓天数保存在 `data/live/signal_state.json`，每次运行时进行 schema 校验（版本、频率、universe 模式、标的合法性），不匹配时自动冷启动。

---

## Regime Gate

基于 510300 波动率的动态仓位调节，A/B 测试验证 10 万组合中 71.5% Sharpe 提升、86.3% 回撤降低。

| 波动率百分位 | 市场状态 | 仓位暴露 |
|-------------|---------|---------|
| < 25% | 低波 | 100% |
| 25-30% | 中低波 | 70% |
| 30-40% | 中高波 | 40% |
| > 40% | 高波 | 10% |

---

## 已完成实验与研究

| 实验 | 内容 | 结果 |
|------|------|------|
| Exp1 | T+1 Open 执行模型 | 55→46 候选，中位数 -4pp |
| Exp2 | 分市场成本模型 | A 股期健康，QDII 主导期高压 |
| Exp4 | 迟滞网格搜索 | 换手率 35→14.6x |
| Exp4.1 | 信号端状态机 | VEC-BT 链式偏差消除 (-22→-3pp) |
| QDII 审计 | 排名分析 | 平均最佳 QDII 排名 8.2/43 |
| FREQ×Exp4 | 消融分析 | F5_ON 主力 + F20_OFF 防御 |
| 条件因子 | ADX/OBV 条件切换 | **NEGATIVE** — 5个假设全推翻 |
| 行业约束 | 同行业双持限制 | **NEGATIVE** — MDD 反而恶化 |
| 跨桶约束 | 5桶 min_buckets=3 | **POSITIVE** — HO +4.9pp |
| 代数因子 | GP挖掘 78 个组合因子 | 6 个 BT 候选，CMF 家族兼容 Exp4 |
| C2 验证 | AMIHUD+CALMAR+CORR_MKT | **BT +45.9%**，超 S1 +13.4pp |
| 深度审阅 | 6 人团队代码审计 | 6 个 P0 已修复，见 `reports/deep_review_final_report.md` |

---

## 数据分区

```
训练集:  2020-01 ~ 2025-04
Holdout: 2025-05 ~ 2026-02-10
实盘:    2025-12-18 ~
```

数据来源：QMT 交易终端，通过 `qmt-data-bridge` SDK 获取，pickle 缓存 + mtime 失效。

---

## 开发注意事项

| 陷阱 | 正确做法 |
|------|---------|
| 前视偏差 | 使用 `shift_timing_signal()` 滞后 1 日 |
| 调仓日不对齐 | 使用 `generate_rebalance_schedule()` 统一 |
| 有界因子 Winsorize | ADX/RSI/CMF 等天然有界，跳过 |
| Regime Gate 重复 | 仅通过 timing_arr 应用 |
| Numba 缓存过期 | 修改 @njit 签名后 `make clean-numba` |
| Set 遍历不确定 | 使用 `sorted()` |
| 浮点比较 | `abs(a - b) < 1e-6` |

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| **v5.0 sealed** | 2026-02-11 | S1 确认为生产策略，封存至 `sealed_strategies/v5.0_20260211/` |
| v5.0 P0 fixes | 2026-02-12 | 6个P0修复：WFO hysteresis对齐、bounded factors统一(7个)、FREQ=5同步、bfill→fillna |
| v5.0-prod1 | 2026-02-10 | 生产就绪：FREQ=5 + Exp4 迟滞 + 状态校验 + 冷启动保护 |
| v5.0-rc1 | 2026-02-10 | Exp4.1 信号端状态机，链式偏差消除 |
| v4.2 | 2026-02-05 | 16 因子正交集封板 |
| v3.4 | 2025-12-16 | 稳定版封板 (FREQ=3) |
| v3.0 | 2025-12-01 | 高频策略升级：FREQ=3, POS=2, 237% |
| v1.0 | 2025-11-28 | 统一策略 121% 验证通过 |

---

**维护者**: Sensen | **License**: MIT | **v5.0-prod1 实盘运行中**
