# 资金流因子部署指南

## 概述

本文档记录资金流因子系统的完整部署过程，包括数据结构、因子实现、测试验证和监控看板。

## 一、系统架构

### 1.1 目录结构

```
factor_system/
├── factor_engine/
│   ├── providers/
│   │   └── money_flow_provider.py          # 资金流数据加载器
│   └── factors/
│       └── money_flow/
│           ├── __init__.py                 # 模块导出
│           ├── core.py                     # 8个主干因子
│           ├── enhanced.py                 # 4个增强因子
│           ├── constraints.py              # 跳空/午休/尾盘约束
│           └── experimental.md             # 实验因子备案
tests/
├── test_moneyflow_consistency.py           # 口径一致性测试
├── test_moneyflow_ic_stability.py          # IC稳定性测试
└── test_portfolio_t1_execution.py          # T+1执行测试
reports/
└── moneyflow_dashboard.py                  # 资金因子健康看板
```

### 1.2 数据流

```
raw/SH/money_flow/*.parquet
    ↓
MoneyFlowProvider (口径统一、标准化)
    ↓
├─→ core.py (8个主干因子)
├─→ enhanced.py (4个增强因子)
└─→ constraints.py (硬约束)
    ↓
测试验证 (IC/一致性/T+1)
    ↓
监控看板 (健康度报告)
```

## 二、数据结构

### 2.1 原始资金流数据（23字段）

基于TuShare格式：

- **小单（散户）**: `buy_sm_amount`, `sell_sm_amount`, `buy_sm_vol`, `sell_sm_vol`
- **中单**: `buy_md_amount`, `sell_md_amount`, `buy_md_vol`, `sell_md_vol`
- **大单**: `buy_lg_amount`, `sell_lg_amount`, `buy_lg_vol`, `sell_lg_vol`
- **超大单（机构）**: `buy_elg_amount`, `sell_elg_amount`, `buy_elg_vol`, `sell_elg_vol`
- **汇总指标**: `net_mf_vol`, `net_mf_amount`, `main_net_inflow`, `large_net_inflow`, `super_net_inflow`

### 2.2 统一口径规则

1. **分母锁死**: 所有占比因子 ÷ `turnover_amount`（成交额）
2. **净额计算**: `main_net = (buy_lg + buy_elg) - (sell_lg + sell_elg)`
3. **标准化流程**: `winsorize(1%,99%) → zscore → 行业中性 → 市值中性`
4. **信号冻结**: 14:30锁定，T+1开盘执行
5. **可交易性**: `tradability_mask=0` 当涨跌停/停牌/成交额<1%分位

## 三、因子体系

### 3.1 核心8因子（core.py）

| 因子ID | 描述 | 公式 | 用途 |
|--------|------|------|------|
| `MainNetInflow_Rate` | 主力净流入率 | `main_net.rolling(5).mean() / turnover.rolling(5).mean()` | 横截面选股 |
| `LargeOrder_Ratio` | 大单占比 | `(buy_lg+sell_lg).rolling(10).mean() / turnover.rolling(10).mean()` | 横截面选股 |
| `SuperLargeOrder_Ratio` | 超大单占比 | `(buy_elg+sell_elg).rolling(20).mean() / turnover.rolling(20).mean()` | 横截面选股 |
| `OrderConcentration` | 资金集中度 | `main_net / total_net` | 横截面选股 |
| `MoneyFlow_Hierarchy` | 资金层级指数 | `(main_net - retail_net) / total_net` | 横截面选股 |
| `MoneyFlow_Consensus` | 资金共识度 | `(sign(main_net)==sign(total_net)).rolling(5).mean()` | 横截面选股 |
| `MainFlow_Momentum` | 主力资金动量 | `(main_net.rolling(5) - main_net.rolling(10)) / main_net.rolling(10)` | 横截面选股 |
| `Flow_Price_Divergence` | 资金价格背离 | `-main_net.rolling(5).corr(ret)` | 横截面选股 |

### 3.2 增强4因子（enhanced.py）

| 因子ID | 描述 | 公式 | 用途 |
|--------|------|------|------|
| `Institutional_Absorption` | 机构吸筹信号 | `(main_net>0) & (vol(3)<vol(10))` | 择时/风控 |
| `Flow_Tier_Ratio_Delta` | 资金层级变化率 | `((buy_lg+buy_elg)/(buy_sm+buy_md)).pct_change(5)` | 择时/风控 |
| `Flow_Reversal_Ratio` | 资金反转信号 | `sign(main_net) != sign(main_net.shift(1).rolling(3).mean())` | 择时/风控 |
| `Northbound_NetInflow_Rate` | 北向资金净流入率 | 需要额外数据源 | 组合层tilt |

### 3.3 硬约束因子（constraints.py）

- **gap_sig**: 跳空信号，三值{-1,0,1}，阈值1.8σ
- **tail30_ratio**: 尾盘抢筹比率（需分钟数据）
- **tradability_mask**: 可交易性掩码（0/1）

## 四、验收标准

### 4.1 单因子门槛

- `IC_t+5 ≥ 0.03`
- `|σ(IC)| ≤ 0.035`
- `Q5-Q1年化超额 > 6%`
- `DSR > 0.35`

### 4.2 组合门槛

- 8主干组：行业中性年化超额 ≥ 10%，MDD < 18%，年化换手 ≤ 10
- 与现有因子合成：收益/回撤 ≥ 1.5，IR ≥ 0.8
- 资金流因子总权重 ≤ 50%，单因子 ≤ 30%

### 4.3 约束验证

- `tradability_mask`覆盖率 > 99.5%
- 四段小时K生成率 = 100%
- gap/tail30触发后次日回撤分布显著左移
- 极端窗口回放（2020/02、2022/04、2023/08、2024/04）穿越=0

## 五、使用示例

### 5.1 加载资金流数据

```python
from pathlib import Path
from factor_system.factor_engine.providers.money_flow_provider import MoneyFlowProvider

# 创建provider
provider = MoneyFlowProvider(data_dir=Path("raw/SH/money_flow"))

# 加载数据
df = provider.load_money_flow("600036.SH", "2024-01-01", "2024-12-31")

# 冻结信号
df = provider.freeze_signal_at_1430(df)
```

### 5.2 计算因子

```python
from factor_system.factor_engine.factors.money_flow import (
    MainNetInflow_Rate,
    LargeOrder_Ratio,
    MoneyFlow_Hierarchy,
)

# 计算主力净流入率
factor1 = MainNetInflow_Rate(window=5)
values1 = factor1.calculate(df)

# 计算大单占比
factor2 = LargeOrder_Ratio(window=10)
values2 = factor2.calculate(df)

# 计算资金层级指数
factor3 = MoneyFlow_Hierarchy()
values3 = factor3.calculate(df)
```

### 5.3 生成健康报告

```python
from pathlib import Path
from reports.moneyflow_dashboard import MoneyFlowDashboard

# 创建看板
dashboard = MoneyFlowDashboard(data_dir=Path("raw/SH/money_flow"))

# 生成完整报告
report = dashboard.generate_full_report(
    symbol="600036.SH",
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# 绘制相关性热力图
corr_matrix = dashboard.generate_correlation_matrix(
    symbol="600036.SH",
    start_date="2024-01-01",
    end_date="2024-12-31"
)
dashboard.plot_correlation_heatmap(corr_matrix, output_path=Path("reports/heatmap.png"))
```

## 六、测试验证

### 6.1 运行测试

```bash
# 口径一致性测试
pytest tests/test_moneyflow_consistency.py -v

# IC稳定性测试
pytest tests/test_moneyflow_ic_stability.py -v

# T+1执行测试
pytest tests/test_portfolio_t1_execution.py -v

# 运行所有资金流测试
pytest tests/test_moneyflow*.py -v
```

### 6.2 预期输出

```
✅ 字段映射测试通过
✅ tradability_mask覆盖率: 98.76%
✅ 信号冻结测试通过，共7个信号列
✅ 衍生字段计算测试通过
✅ 标准化测试通过，winsorized列7个，zscore列7个
✅ MainNetInflow_Rate计算通过，有效值235个
✅ 所有核心因子计算结果通过
```

## 七、监控看板

### 7.1 看板指标

- **IC健康度**: IC_30D, IC_60D, IC_120D, DSR, IC衰退报警
- **半衰期**: t+1, t+5, t+10, 持仓期建议
- **相关矩阵**: 与动量/换手/波动相关性热力图
- **交易所时钟**: 四段生成率、缺段分布
- **约束触发**: Gap触发计数、Tail30触发计数、次日回撤分布
- **可交易性**: mask覆盖率、被屏蔽样本绩效差异

### 7.2 运行看板

```bash
python reports/moneyflow_dashboard.py
```

## 八、注意事项

### 8.1 数据要求

- 资金流数据必须包含23个标准字段
- 数据格式为Parquet
- 时间索引为trade_date
- 单位：金额（万元），量（手）

### 8.2 性能优化

- 所有计算采用向量化，禁止循环
- 使用rolling/shift/corr等Pandas向量化操作
- 避免apply和iterrows
- 大数据集建议分批处理

### 8.3 风险控制

- 资金流因子半衰期短（7-10日），需高频调仓
- 小盘股流动性差，需设置流动性门槛
- 市场风格依赖性强，需分市场环境测试
- 资金流因子总权重不超过50%

## 九、更新日志

- **2025-10-13**: 初始版本，完成8+4因子体系开发
- **待定**: 增加分钟级数据支持（四段小时K、尾盘抢筹）
- **待定**: 接入北向资金数据
- **待定**: 增加行业共振因子

---

*最后更新: 2025-10-13*
*版本: v1.0*
*维护者: 量化系统团队*
