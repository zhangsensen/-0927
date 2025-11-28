# 深度量化0927 - ETF 轮动策略研究平台

> **最后更新**: 2025-11-28  
> **Python**: 3.11+  
> **包管理**: [UV](https://docs.astral.sh/uv/) (v0.9+)

---

## 🎯 项目概述

本项目是专业级 **ETF 轮动策略研究平台**，采用三层引擎架构（WFO → VEC → BT），从因子挖掘到回测审计全流程覆盖。

| 状态 | 说明 |
|------|------|
| ✅ VEC/BT 对齐 | 差异 < 0.01 个百分点 |
| ✅ 无前视偏差 | `shift_timing_signal` 保证 |
| ✅ 生产就绪 | 43 只 ETF, 18 因子, 12,597 组合 |

---

## ⚡ 环境配置（UV）

### 什么是 UV？

[UV](https://docs.astral.sh/uv/) 是新一代 Python 包管理器，比 pip 快 10-100 倍，提供锁文件确保环境一致性。

### 安装 UV

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip
pip install uv
```

### 初始化项目环境

```bash
# 克隆项目后，在项目根目录执行：
uv sync              # 安装所有依赖（根据 uv.lock）
uv sync --dev        # 包含开发依赖（pytest, black, etc.）
```

### ⚠️ 重要：运行命令规范

**所有 Python 脚本必须使用 `uv run` 前缀**：

```bash
# ✅ 正确方式
uv run python script.py
uv run python -m pytest

# ❌ 错误方式（不要使用）
python script.py
python3 script.py
source .venv/bin/activate && python script.py
```

---

## 🏗️ 三层引擎架构

```
┌─────────────────────────────────────────────────────────────┐
│  WFO 筛选层                                                  │
│  ├── 脚本: etf_rotation_optimized/run_unified_wfo.py        │
│  ├── 功能: 高维因子组合空间搜索 (12,597 组合)                │
│  ├── 速度: ~2.5 秒完成全量筛选                               │
│  └── 输出: Top-N 候选组合 + 粗排序                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  VEC 复算层                                                  │
│  ├── 脚本: scripts/batch_vec_backtest.py                    │
│  ├── 功能: 共享规则下的高精度矢量化复算                      │
│  ├── 对齐: 严格对齐 BT（< 0.01pp 差异）                      │
│  └── 输出: 精确收益率、夏普比率、最大回撤                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  BT 审计层                                                   │
│  ├── 脚本: scripts/batch_bt_backtest.py                     │
│  ├── 功能: Backtrader 事件驱动 + 资金约束审计               │
│  ├── 角色: 基准真相（Ground Truth）                          │
│  └── 输出: 最终审计报告                                      │
└─────────────────────────────────────────────────────────────┘
```

> **设计哲学**：WFO 是"粗筛器"，数值可能与 VEC/BT 不同（如 234% vs 70%），这是正常的。真正需要严格对齐的是 VEC ↔ BT。

---

## 🚀 快速开始

### 1. 环境安装

```bash
cd /path/to/project
uv sync --dev              # 安装所有依赖
```

### 2. 完整工作流（推荐）

```bash
# Step 1: WFO 筛选 - 从 12,597 组合中筛选 Top-N
uv run python etf_rotation_optimized/run_unified_wfo.py

# Step 2: VEC 复算 - 对 Top-N 进行精确回测
uv run python scripts/batch_vec_backtest.py

# Step 3: BT 审计 - 事件驱动审计验证
uv run python scripts/batch_bt_backtest.py

# Step 4: 对齐验证 - 确认 VEC/BT 差异 < 0.01pp
uv run python scripts/full_vec_bt_comparison.py
```

### 3. 使用 Makefile（简化命令）

```bash
make install       # 安装依赖 + pre-commit 钩子
make test          # 运行测试
make format        # 代码格式化（black + isort）
make lint          # 代码检查（flake8 + mypy）
make clean         # 清理缓存
```

---

## 📁 项目结构

```
.
├── README.md                      # 📌 本文件
├── AGENTS.md                      # AI Agent 指导规则
├── pyproject.toml                 # 项目配置 + 依赖定义
├── uv.lock                        # 依赖锁文件（确保环境一致）
├── Makefile                       # 常用命令快捷方式
│
├── etf_rotation_optimized/        # ⭐ 主力系统：ETF 轮动
│   ├── run_unified_wfo.py         # WFO 入口脚本
│   ├── core/                      # 核心引擎
│   │   ├── backtester_vectorized.py   # VEC 回测引擎
│   │   ├── wfo_engine.py              # WFO 优化引擎
│   │   ├── precise_factor_library_v2.py  # 18 因子库
│   │   └── shared_types.py            # 共享工具函数
│   ├── configs/                   # 配置文件
│   └── results/                   # 运行结果
│
├── scripts/                       # 🔧 操作脚本
│   ├── batch_vec_backtest.py      # VEC 批量回测
│   ├── batch_bt_backtest.py       # BT 批量审计
│   ├── full_vec_bt_comparison.py  # VEC/BT 对比验证
│   ├── cache_cleaner.py           # 缓存清理
│   └── ci_checks.py               # CI 检查
│
├── factor_system/                 # 因子计算框架
│   ├── factor_engine/             # 统一因子引擎
│   └── screening/                 # 因子筛选
│
├── docs/                          # 📚 文档
│   ├── README.md                  # 项目总览
│   ├── ARCHITECTURE.md            # 架构详解
│   ├── DEVELOPMENT_NOTES.md       # 开发注意事项
│   ├── ROADMAP.md                 # 改造计划
│   └── VEC_BT_ALIGNMENT_*.md      # 对齐历史
│
├── configs/                       # 全局配置
│   ├── etf_pools.yaml             # ETF 池定义
│   └── default.yaml               # 默认参数
│
├── tests/                         # 测试套件
├── raw/                           # 原始数据
└── results/                       # 回测结果
```

---

## 🔧 核心脚本速查

| 脚本 | 路径 | 功能 |
|------|------|------|
| **WFO 筛选** | `etf_rotation_optimized/run_unified_wfo.py` | 高速筛选 12,597 组合 |
| **VEC 回测** | `scripts/batch_vec_backtest.py` | 向量化精确回测 |
| **BT 审计** | `scripts/batch_bt_backtest.py` | Backtrader 事件驱动审计 |
| **对齐验证** | `scripts/full_vec_bt_comparison.py` | 验证 VEC/BT 差异 |
| **缓存清理** | `scripts/cache_cleaner.py` | 清理因子/回测缓存 |

### 示例命令

```bash
# 运行 WFO 筛选
uv run python etf_rotation_optimized/run_unified_wfo.py

# 对指定组合运行 VEC 回测
uv run python scripts/batch_vec_backtest.py --input results/wfo_top100.csv

# 对指定组合运行 BT 审计
uv run python scripts/batch_bt_backtest.py --input results/vec_results.csv

# 清理所有缓存
uv run python scripts/cache_cleaner.py --all
```

---

## 📊 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FREQ` | 8 | 调仓频率（交易日） |
| `POS_SIZE` | 3 | 持仓数量 |
| `INITIAL_CAPITAL` | 1,000,000 | 初始资金 |
| `COMMISSION` | 0.0002 | 手续费率 (2bp) |
| `LOOKBACK` | 252 | 回看窗口（交易日） |

---

## 📈 18 因子列表

| 类别 | 因子名 | 说明 |
|------|--------|------|
| **趋势** | `MA_CROSS_20_60` | 均线交叉 |
| | `PRICE_MA_RATIO_20` | 价格/均线比 |
| | `ADX_14` | 趋势强度 |
| **动量** | `MOM_20` | 20 日动量 |
| | `ROC_20` | 变化率 |
| | `RSI_14` | 相对强弱 |
| | `WILLR_14` | 威廉指标 |
| | `STOCH_K_14` | 随机指标 |
| | `CCI_20` | 商品通道 |
| **波动** | `ATR_RATIO_14` | ATR 比率 |
| | `VOLATILITY_20` | 20 日波动率 |
| | `BB_WIDTH_20` | 布林带宽度 |
| **资金流** | `MFI_14` | 资金流指标 |
| | `OBV_SLOPE_20` | OBV 斜率 |
| | `VOLUME_RATIO_5_20` | 量比 |
| **混合** | `MACD_HIST` | MACD 柱 |
| | `PPO_12_26` | 价格振荡器 |
| | `TRIX_14` | 三重平滑 |

---

## 🔄 依赖管理

### 添加新依赖

```bash
uv add pandas           # 添加运行时依赖
uv add --dev pytest     # 添加开发依赖
```

### 更新依赖

```bash
uv sync --upgrade       # 更新所有依赖
uv lock --upgrade       # 更新锁文件
```

### 导出 requirements.txt（兼容）

```bash
uv pip compile pyproject.toml -o requirements.txt
```

---

## 🧪 测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试
uv run pytest tests/test_factor_engine.py -v

# 带覆盖率
uv run pytest --cov=etf_rotation_optimized --cov-report=html
```

---

## 📖 文档导航

| 文档 | 描述 |
|------|------|
| [docs/README.md](docs/README.md) | 项目详细总览 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 三层引擎架构详解 |
| [docs/DEVELOPMENT_NOTES.md](docs/DEVELOPMENT_NOTES.md) | 开发注意事项、5 大陷阱 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 改造计划与路线图 |
| [docs/VEC_BT_ALIGNMENT_HISTORY_AND_FIXES.md](docs/VEC_BT_ALIGNMENT_HISTORY_AND_FIXES.md) | VEC/BT 对齐问题历史 |
| [etf_rotation_optimized/README.md](etf_rotation_optimized/README.md) | ETF 轮动系统详细文档 |

---

## ⚠️ 开发注意事项

1. **Set 遍历不确定性**：使用 `sorted()` 确保有序
2. **前视偏差**：使用 `shift_timing_signal` 滞后信号
3. **调仓日程**：使用 `generate_rebalance_schedule` 统一
4. **浮点精度**：比较时使用 0.01% 容差
5. **资金时序**：BT 中使用卖出后现金计算

详见 [docs/DEVELOPMENT_NOTES.md](docs/DEVELOPMENT_NOTES.md)

---

## 🧊 性能优化冻结声明

自 2025-11-09 起，低 ROI 长尾性能优化已冻结：

- ✅ 排序稳定性（平均排名法）
- ✅ 日度 IC 预计算 + memmap 加速
- ✅ 无前视自检容差已固化

后续仅接受：功能性需求、生产事故修复、数据源适配

---

**维护者**: 深度量化团队 | **License**: MIT
