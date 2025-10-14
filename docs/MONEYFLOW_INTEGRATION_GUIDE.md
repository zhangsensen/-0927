# 资金流因子集成使用指南

## 📋 概述

资金流因子已无缝集成到FactorEngine中，实现了**166个统一管理的因子库**（154技术+12资金流）。

### 核心特性

✅ **自动数据合并**：日线数据自动合并资金流字段  
✅ **T+1时序安全**：资金流数据自动执行shift(1)处理  
✅ **统一API**：技术和资金流因子使用相同接口  
✅ **因子集支持**：一键调用预定义因子组合  
✅ **优雅降级**：缺失资金流时自动回退到纯价格数据

---

## 🚀 快速开始

### 1. 数据准备

将资金流数据放置在指定目录：

```
raw/SH/money_flow/
├── 600036.SH_money_flow.parquet
├── 000600.SZ_moneyflow.parquet
└── ...
```

**支持的文件名格式**：
- `{symbol}_money_flow.parquet`
- `{symbol}_moneyflow.parquet`
- `{symbol}.parquet`

### 2. 使用因子集（推荐）

```python
from factor_system.factor_engine import api
from datetime import datetime

# 使用A股资金流核心因子集
result = api.calculate_factor_set(
    set_id="a_share_moneyflow_core",
    symbols=["600036.SH", "000600.SZ"],
    timeframe="daily",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
)

print(result.tail())
```

### 3. 单独计算资金流因子

```python
# 计算特定资金流因子
factors = [
    "MainNetInflow_Rate",
    "OrderConcentration",
    "MoneyFlow_Hierarchy",
]

result = api.calculate_factors(
    factor_ids=factors,
    symbols=["600036.SH"],
    timeframe="daily",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
)
```

### 4. 混合计算（技术+资金流）

```python
# 同时计算技术和资金流因子
mixed_factors = [
    "RSI",  # 技术因子
    "MACD",  # 技术因子
    "MainNetInflow_Rate",  # 资金流因子
    "Flow_Price_Divergence",  # 资金流因子
]

result = api.calculate_factors(
    factor_ids=mixed_factors,
    symbols=["600036.SH"],
    timeframe="daily",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
)
```

---

## 📦 可用因子集

### a_share_moneyflow_core
**A股资金流核心因子集**（8个因子）

适用场景：主力资金流动分析

因子列表：
- `MainNetInflow_Rate` - 主力净流入率
- `LargeOrder_Ratio` - 大单占比
- `SuperLargeOrder_Ratio` - 超大单占比
- `OrderConcentration` - 资金集中度
- `MoneyFlow_Hierarchy` - 资金层级指数
- `MoneyFlow_Consensus` - 资金共识度
- `MainFlow_Momentum` - 主力资金动量
- `Flow_Price_Divergence` - 资金价格背离

### a_share_moneyflow_enhanced
**A股资金流增强因子集**（4个因子）

适用场景：资金流与价格行为结合分析

因子列表：
- `Institutional_Absorption` - 机构吸筹信号
- `Flow_Tier_Ratio_Delta` - 资金层级变化率
- `Flow_Reversal_Ratio` - 资金反转信号
- `Northbound_NetInflow_Rate` - 北向资金净流入率

### a_share_moneyflow_all
**A股资金流完整因子集**（12个因子）

包含上述所有资金流因子

---

## 🔧 高级用法

### 列出所有可用因子

```python
# 列出所有因子
all_factors = api.list_available_factors()
print(f"总因子数: {len(all_factors)}")

# 按类别列出
categories = api.list_factor_categories()
for category, factors in categories.items():
    print(f"{category}: {len(factors)}个因子")
```

### 列出所有因子集

```python
engine = api.get_engine()
factor_sets = engine.registry.list_factor_sets()

for set_id in factor_sets:
    factor_set = engine.registry.get_factor_set(set_id)
    print(f"{set_id}: {factor_set['name']}")
```

### 查看因子元数据

```python
metadata = api.get_factor_metadata("MainNetInflow_Rate")
print(metadata)
```

---

## ⚙️ 配置说明

### 环境变量

```bash
# 自定义数据根目录
export FACTOR_ENGINE_RAW_DATA_DIR=/path/to/your/data
```

### 目录结构

```
project_root/
├── raw/
│   └── SH/
│       ├── money_flow/          # 资金流数据目录
│       │   ├── 600036.SH_money_flow.parquet
│       │   └── ...
│       ├── 600036.SH.parquet    # 价格数据
│       └── ...
└── ...
```

---

## 🛡️ 时序安全保证

### T+1滞后机制

资金流数据在T日收盘后才发布，系统自动执行T+1处理：

```python
# 内部自动处理
money_flow_data = money_flow_data.shift(1)
```

### 价格数据时序安全

收益率和波动率计算也执行shift(1)：

```python
returns = close.pct_change().shift(1)
```

---

## 📊 数据质量验证

### 检查数据覆盖率

```python
result = api.calculate_factor_set(...)

# 计算有效率
valid_ratio = (1 - result.isnull().sum().sum() / result.size) * 100
print(f"数据有效率: {valid_ratio:.2f}%")
```

### 按股票统计

```python
for symbol in symbols:
    symbol_data = result.xs(symbol, level='symbol')
    valid_ratio = (1 - symbol_data.isnull().sum().sum() / symbol_data.size) * 100
    print(f"{symbol}: 有效率{valid_ratio:.2f}%")
```

---

## ❓ 常见问题

### Q1: 资金流数据缺失怎么办？

**A**: 系统会优雅降级，自动回退到纯价格数据计算技术因子。

### Q2: 支持哪些时间框架？

**A**: 资金流因子仅支持日线（daily），技术因子支持1min到monthly。

### Q3: 如何确认资金流数据已加载？

**A**: 查看日志输出：
```
INFO - 使用 CombinedMoneyFlowProvider (OHLCV + MoneyFlow)
INFO - 启用资金流合并，目录: raw/SH/money_flow
```

### Q4: 资金流因子计算很慢？

**A**: 
1. 启用缓存：`use_cache=True`
2. 减少股票数量
3. 缩短时间范围

### Q5: 如何验证T+1时序安全？

**A**: 检查数据中的`temporal_safe`标记：
```python
print(data['temporal_safe'].iloc[0])  # 应为True
```

---

## 🧪 测试脚本

运行综合测试：

```bash
python scripts/test_moneyflow_integration_comprehensive.py
```

测试覆盖：
- ✅ 多股票因子计算
- ✅ 边缘情况处理
- ✅ 因子集调用
- ✅ 数据有效性验证
- ✅ 混合因子计算

---

## 📚 参考资料

- [FactorEngine API文档](./FACTOR_ENGINE_DEPLOYMENT_GUIDE.md)
- [资金流因子定义](../factor_system/factor_engine/factors/money_flow/)
- [配置文件](../factor_system/config/enhanced_engine_config.yaml)

---

## 🎯 最佳实践

1. **优先使用因子集**：避免手动列举因子ID
2. **启用缓存**：提升重复计算性能
3. **批量计算**：一次计算多个因子，减少I/O
4. **验证数据质量**：计算前检查数据覆盖率
5. **监控日志**：关注警告和错误信息

---

## 📞 支持

遇到问题？

1. 查看日志输出
2. 运行测试脚本诊断
3. 检查数据文件格式和路径
4. 参考本文档FAQ部分

---

**更新时间**: 2025-01-13  
**版本**: v1.0  
**状态**: ✅ 生产就绪
