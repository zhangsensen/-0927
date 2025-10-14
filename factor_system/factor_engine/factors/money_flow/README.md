# 资金流因子模块

## 概述

资金流因子模块提供基于A股资金流数据的量化因子计算能力，包含12个生产级因子和多个实验性因子。

## 特性

- ✅ **向量化计算**: 所有因子使用NumPy/Pandas向量化操作，禁止循环
- ✅ **统一口径**: 分母锁死为turnover_amount，标准化流程一致
- ✅ **T+1执行**: 14:30信号冻结，T+1开盘执行，无前视偏差
- ✅ **可交易性约束**: tradability_mask覆盖率>99.5%
- ✅ **完整测试**: 口径一致性、IC稳定性、T+1执行测试全覆盖

## 因子列表

### 核心因子（8个）

| 因子 | 描述 | 窗口 | 用途 |
|------|------|------|------|
| `MainNetInflow_Rate` | 主力净流入率 | 5日 | 横截面选股 |
| `LargeOrder_Ratio` | 大单占比 | 10日 | 横截面选股 |
| `SuperLargeOrder_Ratio` | 超大单占比 | 20日 | 横截面选股 |
| `OrderConcentration` | 资金集中度 | - | 横截面选股 |
| `MoneyFlow_Hierarchy` | 资金层级指数 | - | 横截面选股 |
| `MoneyFlow_Consensus` | 资金共识度 | 5日 | 横截面选股 |
| `MainFlow_Momentum` | 主力资金动量 | 5-10日 | 横截面选股 |
| `Flow_Price_Divergence` | 资金价格背离 | 5日 | 横截面选股 |

### 增强因子（4个）

| 因子 | 描述 | 用途 |
|------|------|------|
| `Institutional_Absorption` | 机构吸筹信号 | 择时/风控 |
| `Flow_Tier_Ratio_Delta` | 资金层级变化率 | 择时/风控 |
| `Flow_Reversal_Ratio` | 资金反转信号 | 择时/风控 |
| `Northbound_NetInflow_Rate` | 北向资金净流入率 | 组合层tilt |

## 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy scipy
```

### 2. 加载数据

```python
from pathlib import Path
from factor_system.factor_engine.providers.money_flow_provider import MoneyFlowProvider

provider = MoneyFlowProvider(data_dir=Path("raw/SH/money_flow"))
df = provider.load_money_flow("600036.SH", "2024-01-01", "2024-12-31")
```

### 3. 计算因子

```python
from factor_system.factor_engine.factors.money_flow import MainNetInflow_Rate

factor = MainNetInflow_Rate(window=5)
values = factor.calculate(df)
```

### 4. 运行示例

```bash
python examples/moneyflow_quickstart.py
```

## 数据要求

### 输入数据格式

- **格式**: Parquet
- **索引**: trade_date (datetime)
- **必需字段**（23个）:
  - `buy_sm_amount`, `sell_sm_amount` (小单)
  - `buy_md_amount`, `sell_md_amount` (中单)
  - `buy_lg_amount`, `sell_lg_amount` (大单)
  - `buy_elg_amount`, `sell_elg_amount` (超大单)
  - `net_mf_amount`, `main_net_inflow` (汇总)

### 数据来源

- TuShare Pro API
- 东方财富资金流数据
- 同花顺Level-2数据

## 验收标准

### 单因子

- IC_t+5 ≥ 0.03
- |σ(IC)| ≤ 0.035
- Q5-Q1年化超额 > 6%
- DSR > 0.35

### 组合

- 行业中性年化超额 ≥ 10%
- MDD < 18%
- 年化换手 ≤ 10
- 资金流因子总权重 ≤ 50%

## 测试

```bash
# 运行所有测试
pytest tests/test_moneyflow*.py -v

# 口径一致性测试
pytest tests/test_moneyflow_consistency.py -v

# IC稳定性测试
pytest tests/test_moneyflow_ic_stability.py -v

# T+1执行测试
pytest tests/test_portfolio_t1_execution.py -v
```

## 监控看板

```bash
python reports/moneyflow_dashboard.py
```

看板包含：
- IC健康度（30D/60D/120D）
- 半衰期分析
- 因子相关性矩阵
- 可交易性统计
- 约束触发统计

## 注意事项

### 性能优化

- ✅ 使用向量化操作（rolling, shift, corr）
- ✅ 避免循环和apply
- ✅ 大数据集分批处理
- ✅ 缓存中间结果

### 风险控制

- ⚠️ 资金流因子半衰期短（7-10日），需高频调仓
- ⚠️ 小盘股流动性差，需设置流动性门槛
- ⚠️ 市场风格依赖性强，需分环境测试
- ⚠️ 资金流因子总权重不超过50%

### 数据质量

- ✅ tradability_mask覆盖率 > 99.5%
- ✅ 无前视偏差（14:30冻结，T+1执行）
- ✅ 统一口径（分母为turnover_amount）
- ✅ 标准化流程（winsorize → zscore）

## 文档

- [部署指南](../../../../docs/MONEYFLOW_FACTOR_DEPLOYMENT.md)
- [实验因子](experimental.md)
- [因子注册表](../../../FACTOR_REGISTRY.md)

## 更新日志

- **2025-10-13**: v1.0 初始版本
  - 完成8个核心因子
  - 完成4个增强因子
  - 完成数据提供者
  - 完成测试套件
  - 完成监控看板

## 许可证

内部使用，保密

---

*维护者: 量化系统团队*
*最后更新: 2025-10-13*
