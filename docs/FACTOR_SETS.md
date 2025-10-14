# 因子集管理系统

**版本**: v1.0  
**状态**: ✅ 生产就绪  
**创建时间**: 2025-10-13

---

## 🎯 概述

因子集管理系统通过YAML配置文件定义可复用的因子集合，支持组合、嵌套和动态扩展，彻底解决硬编码因子列表的问题。

### 核心特性

1. **配置驱动** - 因子集定义在YAML文件中，无需修改代码
2. **组合复用** - 使用`set:`引用其他因子集
3. **动态扩展** - `all_factors`关键字自动扩展到所有已注册因子
4. **自动去重** - 递归展开后自动去重和排序
5. **安全过滤** - 自动过滤未注册的因子，避免运行时错误

---

## 📁 配置文件

### 位置
```
factor_system/config/factor_sets.yaml
```

### 结构示例

```yaml
factor_sets:
  # 基础集合
  tech_mini:
    - RSI
    - MACD
    - STOCH

  # 资金流因子
  money_flow_core:
    - MainNetInflow_Rate
    - LargeOrder_Ratio
    - SuperLargeOrder_Ratio

  # 组合集合（引用其他集合）
  daily_default_research:
    - set: tech_mini
    - set: money_flow_core

  # 动态扩展（所有已注册因子）
  all:
    - all_factors
```

---

## 🔧 使用方法

### 1. 在生产脚本中使用

```python
from factor_system.factor_engine.core.registry import get_global_registry

# 获取注册表
registry = get_global_registry(include_money_flow=True)

# 解析因子集
factor_ids = registry.get_factor_ids_by_set("daily_default_research")

# 使用因子集计算
result = engine.calculate_factors(
    factor_ids=factor_ids,
    symbols=["600036.SH"],
    timeframe="daily",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)
```

### 2. 命令行参数

```bash
# 使用默认集合
python scripts/production_run.py

# 指定因子集
python scripts/production_run.py --set daily_default_research

# 使用所有因子
python scripts/production_run.py --set all

# 快速回测集
python scripts/production_run.py --set backtest_fast
```

---

## 📋 预定义因子集

### tech_mini
**描述**: 基础技术指标  
**因子数**: 3个  
**包含**: RSI, MACD, STOCH

### macd_family
**描述**: MACD家族  
**因子数**: 3个  
**包含**: MACD, MACD_SIGNAL, MACD_HIST

### money_flow_core
**描述**: 核心资金流因子  
**因子数**: 7个  
**包含**:
- MainNetInflow_Rate
- LargeOrder_Ratio
- SuperLargeOrder_Ratio
- MainFlow_Momentum
- Institutional_Absorption
- Flow_Tier_Ratio_Delta
- Flow_Reversal_Ratio

### money_flow_enhanced
**描述**: 增强资金流因子  
**因子数**: 4个  
**包含**:
- MoneyFlow_Consensus
- MoneyFlow_Hierarchy
- OrderConcentration
- Flow_Price_Divergence

### money_flow_all
**描述**: 完整资金流因子（核心+增强）  
**因子数**: 11个  
**组合**: money_flow_core + money_flow_enhanced

### daily_default_research
**描述**: 日线研究默认集  
**因子数**: 10个  
**组合**: tech_mini + money_flow_core

### production_standard
**描述**: 生产环境标准集  
**因子数**: 16个  
**组合**: macd_family + money_flow_all + RSI + STOCH

### backtest_fast
**描述**: 快速回测集（轻量）  
**因子数**: 4个  
**包含**: RSI, MACD, MainNetInflow_Rate, LargeOrder_Ratio

### all
**描述**: 所有已注册因子（动态）  
**因子数**: 16个（当前）  
**特性**: 自动扩展到所有已注册因子

---

## 🔍 API参考

### Registry.get_factor_ids_by_set()

```python
def get_factor_ids_by_set(
    self, 
    set_name: str, 
    filter_registered: bool = True
) -> List[str]:
    """
    根据因子集名称获取因子ID列表
    
    Args:
        set_name: 因子集名称
        filter_registered: 是否只返回已注册的因子（默认True）
        
    Returns:
        去重排序的因子ID列表
        
    Raises:
        ValueError: 如果因子集不存在
    """
```

**示例**:
```python
# 获取因子集
factor_ids = registry.get_factor_ids_by_set("daily_default_research")
# 返回: ['Flow_Reversal_Ratio', 'Flow_Tier_Ratio_Delta', ...]

# 不过滤未注册因子（用于调试）
all_ids = registry.get_factor_ids_by_set("tech_mini", filter_registered=False)
```

### Registry.list_defined_sets()

```python
def list_defined_sets(self) -> List[str]:
    """
    列出所有已定义的因子集
    
    Returns:
        因子集名称列表（YAML + JSON）
    """
```

**示例**:
```python
sets = registry.list_defined_sets()
# 返回: ['all', 'backtest_fast', 'daily_default_research', ...]
```

---

## ⚙️ 配置语法

### 1. 直接列出因子

```yaml
tech_mini:
  - RSI
  - MACD
  - STOCH
```

### 2. 引用其他因子集

```yaml
daily_default_research:
  - set: tech_mini
  - set: money_flow_core
```

### 3. 混合使用

```yaml
production_standard:
  - set: macd_family
  - set: money_flow_all
  - RSI
  - STOCH
```

### 4. 动态扩展

```yaml
all:
  - all_factors  # 扩展到所有已注册因子
```

---

## 🛡️ 安全机制

### 1. 自动过滤未注册因子

如果YAML中定义的因子未在`factor_engine`中注册，会自动过滤并记录警告：

```
⚠️ 因子集 'tech_mini' 过滤了 2 个未注册因子 (5 -> 3)
```

### 2. 循环引用检测

递归解析时自动检测循环引用：

```
⚠️ 检测到循环引用: tech_mini
```

### 3. 未知因子集错误

```python
try:
    factor_ids = registry.get_factor_ids_by_set("unknown_set")
except ValueError as e:
    print(e)
    # 输出: 未定义的因子集: 'unknown_set'
    #      可用因子集: ['all', 'backtest_fast', ...]
```

---

## 📊 性能考虑

### 因子集大小建议

| 因子数 | 适用场景 | 内存占用 | 计算时间 |
|--------|----------|----------|----------|
| 1-10 | 快速回测 | <100MB | <1秒 |
| 10-50 | 日常研究 | <500MB | 1-5秒 |
| 50-100 | 完整分析 | <2GB | 5-30秒 |
| 100+ | 全量计算 | >2GB | >30秒 |

### 优化建议

1. **批量计算** - 对于大因子集，考虑分批计算
2. **启用缓存** - 设置`use_cache=True`
3. **限制标的数** - 先在少量标的上验证
4. **选择性计算** - 使用特定因子集而非`all`

---

## 🧪 测试

### 运行测试

```bash
python -m pytest tests/test_factor_sets_yaml.py -v
```

### 测试覆盖

- ✅ YAML配置加载
- ✅ 因子集列表
- ✅ 基础因子集解析
- ✅ 组合因子集解析
- ✅ 动态扩展（all_factors）
- ✅ 嵌套引用解析
- ✅ 未知因子集错误
- ✅ 去重验证
- ✅ 排序验证

---

## 🔄 扩展到factor_generation

### 当前状态

- `factor_engine`: 16个因子（3个技术指标 + 11个资金流因子）
- `factor_generation`: 154个VectorBT技术指标

### 集成步骤（未来）

1. 将`factor_generation`的指标注册到`factor_engine`
2. 更新`factor_sets.yaml`包含完整指标列表
3. 验证计算一致性
4. 更新文档

---

## 📝 最佳实践

### 1. 命名规范

- 使用小写+下划线：`tech_mini`, `money_flow_core`
- 描述性命名：`daily_default_research` 而非 `set1`
- 避免过长：不超过30个字符

### 2. 组织结构

```yaml
# 基础集合（原子）
tech_mini: [...]
money_flow_core: [...]

# 家族集合（同类因子）
macd_family: [...]
rsi_family: [...]

# 组合集合（多个基础集合）
daily_default_research:
  - set: tech_mini
  - set: money_flow_core

# 特殊集合
all: [all_factors]
```

### 3. 版本控制

- 在YAML中添加注释记录变更
- 重大变更时备份旧配置
- 使用Git跟踪配置文件

### 4. 文档同步

- 更新因子集时同步更新本文档
- 记录每个因子集的用途和适用场景
- 提供使用示例

---

## 🐛 故障排查

### 问题1: 因子集为空

**症状**: `get_factor_ids_by_set()` 返回空列表

**原因**: 所有因子都未注册

**解决**:
```python
# 检查已注册因子
all_factors = set(registry.factors.keys()) | set(registry.metadata.keys())
print(f"已注册因子: {all_factors}")

# 不过滤未注册因子查看原始定义
raw_ids = registry.get_factor_ids_by_set("tech_mini", filter_registered=False)
print(f"YAML定义: {raw_ids}")
```

### 问题2: 计算失败

**症状**: `ValueError: 未注册的因子: 'XXX'`

**原因**: 因子集中包含未注册因子，且`filter_registered=False`

**解决**: 使用默认的`filter_registered=True`或先注册因子

### 问题3: 性能慢

**症状**: 计算时间过长

**解决**:
- 使用较小的因子集
- 启用缓存
- 减少标的数量
- 考虑批量计算

---

## 📚 参考资料

- [FactorEngine文档](FACTOR_ENGINE_DEPLOYMENT_GUIDE.md)
- [Registry API](../factor_system/factor_engine/core/registry.py)
- [生产脚本](../scripts/production_run.py)
- [测试用例](../tests/test_factor_sets_yaml.py)

---

**维护者**: Linus-Style Quant Engineer  
**最后更新**: 2025-10-13
