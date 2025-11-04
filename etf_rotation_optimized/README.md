# ETF轮动系统优化版 v2.0

**精确、简洁、可复现的量化交易系统**

---

## 🎯 核心理念

> **No bullshit. No magic. Just math and code.**

- **横截面加工** → 标准化因子矩阵
- **因子筛选** → IC驱动 + 约束优化
- **WFO验证** → 前向性能测试
- **VBT回测** → 暴力组合测试

---

## 🚀 快速开始

### 安装依赖
```bash
make install
```

### 运行完整流程
```bash
make run
```

### 运行单个步骤
```bash
make run-cross-section    # 横截面加工
make run-factor-selection # 因子筛选
make run-wfo              # WFO验证
make run-backtest         # VBT回测
```

---

## 📁 项目结构

```
etf_rotation_optimized/
├── main.py                    # 统一入口
├── Makefile                   # 简化命令
├── configs/
│   ├── default.yaml           # 默认配置
│   ├── FACTOR_SELECTION_CONSTRAINTS.yaml
│   └── experiments/           # 实验配置
├── core/                      # 核心算法库
│   ├── pipeline.py            # 流程编排
│   ├── data_loader.py         # 数据加载
│   ├── precise_factor_library_v2.py  # 因子计算
│   ├── cross_section_processor.py    # 横截面标准化
│   ├── factor_selector.py     # 因子筛选
│   ├── ic_calculator.py       # IC计算
│   ├── walk_forward_optimizer.py     # WFO框架
│   └── ensemble_wfo_optimizer.py     # 集成WFO
├── vectorbt_backtest/         # VBT回测系统（独立）
├── tests/                     # 测试
├── cache/                     # 缓存
└── results/                   # 输出
```

---

## 🔧 配置文件

所有配置在 `configs/default.yaml` 中：

```yaml
run_id: "ETF_ROTATION_DEFAULT"

data:
  symbols: ["510300", "510500", ...]  # 43只ETF
  start_date: "2020-01-01"
  end_date: "2025-10-14"

cross_section:
  winsorize_lower: 0.025
  winsorize_upper: 0.975

factor_selection:
  min_ic: 0.02
  min_ir: 0.05

wfo:
  is_period: 100
  oos_period: 20
  step_size: 20
  n_samples: 1000
  combo_size: 5
  top_k: 10

backtest:
  init_cash: 100000
  top_n: 5
  commission: 0.0005
```

---

## 📊 工作流

### 1. 横截面加工
- 加载43只ETF的OHLCV数据
- 计算12个精选因子
- 保存原始因子矩阵

### 2. 因子筛选
- 横截面标准化（Z-score）
- Winsorize极值截断
- 保留NaN，不填充

### 3. WFO验证
- 滑动窗口：IS=100天，OOS=20天
- 集成采样：1000组合 × 5因子
- Top10集成加权

#### Phase 2：多策略枚举 + Top-5 组合选择

- 基于窗口结果的“因子子集 × 温度τ × TopN”多策略枚举
- 严格 T+1 拼接全周期 OOS 收益，逐策略计算 KPI
- 产出：`strategies_ranked.csv`、`top5_strategies.csv`、`top5_combo_*.csv`
- 配置（可选，位于 `configs/default.yaml::wfo.phase2`）:
  - `min_factor_freq`：因子最低出现频率（默认 0.3）
  - `min_factors` / `max_factors`：枚举子集大小（默认 3/5）
  - `tau_grid`：温度参数网格，τ<1 更集中、τ>1 更均匀（默认 [0.7,1.0,1.5]）
  - `topn_grid`：TopN 网格（默认 [backtest.top_n]）
  - `max_strategies`：最大枚举策略数（默认 200）

### 4. VBT回测

- 暴力测试所有因子组合
- 计算Sharpe、最大回撤等指标
- 生成性能报告

---

## 🧪 测试

```bash
make test
```

---

## 🧹 清理

```bash
make clean
```

---

## 📝 命令行接口

### 完整流程

```bash
python main.py run --config configs/default.yaml
```

### 指定步骤

```bash
python main.py run-steps \
  --config configs/default.yaml \
  --steps cross_section \
  --steps factor_selection
```

### 查看帮助

```bash
python main.py --help
make help
```

---

## 🔬 因子库

12个精选因子（`core/precise_factor_library_v2.py`）：

| 维度 | 因子 | 说明 |
|------|------|------|
| 趋势/动量 | MOM_20D | 20日动量百分比 |
| 趋势/动量 | SLOPE_20D | 20日线性回归斜率 |
| 价格位置 | PRICE_POSITION_20D | 20日价格位置 |
| 价格位置 | PRICE_POSITION_120D | 120日价格位置 |
| 波动率 | RET_VOL_20D | 20日收益波动率 |
| 波动率 | MAX_DD_60D | 60日最大回撤 |
| 成交量 | VOL_RATIO_20D | 20日成交量比率 |
| 成交量 | VOL_RATIO_60D | 60日成交量比率 |
| 价量耦合 | PV_CORR_20D | 20日价量相关性 |
| 反转 | RSI_14 | 14日相对强度指数 |

---

## 🎨 设计原则

1. **消灭特殊情况** - 用数据结构代替 if/else
2. **Never break userspace** - API 必须稳定
3. **实用主义** - 解决真问题，不造概念
4. **简洁是武器** - 缩进 ≤3 层，函数 <50 行
5. **代码即真理** - 所有假设必须能回测验证

---

## 📈 性能指标

- **向量化率**: ≥95%
- **单因子计算**: <1ms
- **内存效率**: ≥70%
- **并行核数**: 8核（M4 Max）

---

## 🔥 重构历史

- **v1.0** (2024): 初始版本，scripts/手动流程
- **v2.0** (2025-10-28): 统一入口，配置驱动，删除冗余代码

---

## 📄 License

MIT

---

**Built with Linus philosophy: Precise, Concise, Reproducible.**
