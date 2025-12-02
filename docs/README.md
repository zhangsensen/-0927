# ETF 轮动策略研究平台

> **版本**: 3.1  
> **最后更新**: 2025-12-01  
> **状态**: ✅ 生产就绪 | 🔒 v3.0 策略封板

---

## 项目概述

本项目是一个专业级 **ETF 轮动策略** 研究与验证平台，核心目标是：

1. **因子挖掘与组合优化** - 从 18 个精选因子中筛选最优组合
2. **严格回测验证** - 通过三层引擎确保结果可靠
3. **贴近实盘** - 无前视偏差，T+1 执行约束

### 核心特点

| 特性 | v3.0 生产值 | 说明 |
|------|-------------|------|
| **标的池** | 43 只 ETF | 38 A股 + 5 QDII（核心 Alpha 来源） |
| **因子库** | 18 个因子 | 趋势/动量/波动/资金流 |
| **最佳组合** | 5 因子 | ADX_14D + MAX_DD_60D + PRICE_POSITION_* + SHARPE |
| **回测周期** | 2020-01-01 至今 | 4+ 年 |
| **初始资金** | 100 万 | |
| **持仓数** | **2 只** | v3.0 优化（v1.0 为 3） |
| **调仓频率** | **每 3 个交易日** | v3.0 高频（v1.0 为 8） |
| **收益率** | **237.45%** | v3.0 验证结果 |

---

## 三层引擎架构

```
┌─────────────────────────────────────────────────────────────┐
│                    WFO 筛选层                                │
│  run_combo_wfo.py - 真·滚动 WFO，2.5s 覆盖 12,597 组合      │
│  职责：高维空间搜索，产出 Top-N 候选池                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    VEC 复算层                                │
│  batch_vec_backtest.py - 向量化高精度复算                    │
│  职责：统一规则下验证，与 BT 严格对齐 (< 0.01pp)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    BT 审计层                                 │
│  batch_bt_backtest.py - Backtrader 事件驱动                  │
│  职责：资金约束兜底审计，基准真相                            │
└─────────────────────────────────────────────────────────────┘
```

### 关键架构决策

1. **VEC ↔ BT 严格对齐**：差异 < 0.01pp，这是基准一致性
2. **WFO 是粗筛器**：与 VEC/BT 数值可能不同，但排序稳定即可
3. **无前视偏差**：
   - 信号来自 T-1（`shift_timing_signal`）
   - 统一调仓日程（`generate_rebalance_schedule`）
   - 价格校验（`ensure_price_views`）

---

## 目录结构

```
-0927/
├── src/                        # ⭐ 统一源码目录
│   ├── etf_strategy/           # 核心策略模块
│   │   ├── core/               # 核心引擎模块
│   │   ├── auditor/            # BT 审计模块
│   │   └── run_combo_wfo.py    # WFO 入口
│   └── etf_data/               # 数据管理模块
│
├── scripts/                    # 工具脚本
│   ├── batch_vec_backtest.py   # VEC 批量回测
│   ├── batch_bt_backtest.py    # BT 批量回测
│   ├── full_vec_bt_comparison.py # VEC/BT 对比验证
│   └── cache_cleaner.py        # 缓存清理
│
├── configs/                    # 全局配置
│   ├── etf_pools.yaml          # ETF 池配置
│   ├── combo_wfo_config.yaml   # WFO 配置
│   └── default.yaml            # 默认参数
│
├── raw/                        # 原始数据
├── results/                    # 运行结果
├── tests/                      # 测试
└── docs/                       # 文档
```

---

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
uv sync --dev
uv pip install -e .

# 验证环境
./test_environment.sh
```

### 2. 标准工作流

```bash
# 步骤 1: WFO 筛选（12,597 组合 → Top 100）
uv run python src/etf_strategy/run_combo_wfo.py

# 步骤 2: VEC 复算 Top 100
uv run python scripts/batch_vec_backtest.py

# 步骤 3: BT 审计 Top 10
uv run python scripts/batch_bt_backtest.py

# 可选: VEC/BT 对比验证单个组合
uv run python scripts/full_vec_bt_comparison.py --combo "ADX_14D + CMF_20D + ..."
```

### 3. 清理缓存（重要）

```bash
# 每次重大修改后，建议清理缓存
python scripts/cache_cleaner.py
```

---

## 核心参数（v3.0 生产配置）

| 参数 | v3.0 生产值 | v1.0 旧值 | 说明 |
|------|-------------|-----------|------|
| `FREQ` | **3** | 8 | 调仓频率（交易日）|
| `POS_SIZE` | **2** | 3 | 持仓数量 |
| `INITIAL_CAPITAL` | 1,000,000 | 同 | 初始资金 |
| `COMMISSION_RATE` | 0.0002 | 同 | 手续费率 |
| `LOOKBACK` | 252 | 同 | 回看窗口 |

> ⚠️ **重要**: v3.0 的 FREQ=3, POS=2 是经过完整回测验证的最优参数。
> 详见 `docs/BEST_STRATEGY_43ETF_UNIFIED.md`

---

## 相关文档

- [架构与引擎对齐](ARCHITECTURE.md) - 三层引擎详细说明
- [最佳策略 v3.0](BEST_STRATEGY_43ETF_UNIFIED.md) - 237.45% 收益策略详情
- [ETF 池架构](ETF_POOL_ARCHITECTURE.md) - 43 ETF 池设计与 QDII 重要性
- [开发注意事项](DEVELOPMENT_NOTES.md) - 开发规范与陷阱
- [历史修复记录](VEC_BT_ALIGNMENT_HISTORY_AND_FIXES.md) - BUG 修复历史

---

## 联系与维护

- **仓库**: `zhangsensen/-0927`
- **分支**: `refactor/unified-codebase-20251116`
- **状态**: 🔒 v3.1 策略封板 | 237.45% 收益已验证
