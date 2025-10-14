# 🎯 因子集管理系统 - 快速开始

**版本**: v1.0 | **状态**: ✅ 生产就绪

---

## 🚀 5分钟快速上手

### 1. 使用预定义因子集

```bash
# 日常研究（10个因子）
python scripts/production_run.py --set daily_default_research

# 快速回测（4个因子）
python scripts/production_run.py --set backtest_fast

# 所有因子（16个）
python scripts/production_run.py --set all
```

### 2. 在代码中使用

```python
from factor_system.factor_engine.core.registry import get_global_registry
from factor_system.factor_engine.core.engine import FactorEngine
from datetime import datetime

# 获取注册表
registry = get_global_registry(include_money_flow=True)

# 解析因子集
factor_ids = registry.get_factor_ids_by_set("daily_default_research")

# 计算因子
engine = FactorEngine(data_provider=provider, registry=registry)
result = engine.calculate_factors(
    factor_ids=factor_ids,
    symbols=["600036.SH"],
    timeframe="daily",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)
```

### 3. 自定义因子集

编辑 `factor_system/config/factor_sets.yaml`:

```yaml
my_strategy:
  - RSI
  - MACD
  - MainNetInflow_Rate
  - LargeOrder_Ratio
```

使用:
```bash
python scripts/production_run.py --set my_strategy
```

---

## 📋 可用因子集

| 因子集 | 因子数 | 用途 |
|--------|--------|------|
| `tech_mini` | 3 | 基础技术指标 |
| `money_flow_core` | 7 | 核心资金流 |
| `daily_default_research` | 10 | 日常研究 |
| `backtest_fast` | 4 | 快速回测 |
| `production_standard` | 16 | 生产标准 |
| `all` | 16 | 所有因子 |

---

## 📚 完整文档

- **核心指南**: `CLAUDE.md` - 完整项目指导
- **因子注册表**: `factor_system/FACTOR_REGISTRY.md` - 所有因子列表
- **配置文件**: `factor_system/config/factor_sets.yaml` - 因子集配置
- **测试**: `pytest tests/test_factor_sets_yaml.py -v`

---

## ✅ 核心特性

- ✅ **配置驱动** - YAML配置，无需修改代码
- ✅ **组合复用** - 支持因子集嵌套引用
- ✅ **动态扩展** - `all_factors`自动扩展
- ✅ **自动过滤** - 过滤未注册因子
- ✅ **命令行支持** - `--set`参数

---

**维护**: Linus-Style Quant Engineer | **更新**: 2025-10-13
