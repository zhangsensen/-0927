# 因子引擎快速上手指南

## 🚀 快速开始

### 1. 基本使用

```python
from factor_system.factor_engine.api import calculate_factors
from datetime import datetime

# 计算RSI和MACD因子
factors = calculate_factors(
    factor_ids=["RSI14", "MACD_12_26_9"],
    symbols=["0700.HK", "0005.HK"],
    timeframe="15min",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30),
    use_cache=True
)

print(factors.head())
```

### 2. 查看可用因子

```python
from factor_system.factor_engine.api import list_available_factors

# 列出所有246个因子
all_factors = list_available_factors()
print(f"可用因子数量: {len(all_factors)}")
print(f"前10个因子: {all_factors[:10]}")
```

### 3. 按类别查看因子

```python
from factor_system.factor_engine.api import list_factor_categories

# 查看因子分类
categories = list_factor_categories()
for category, factors in categories.items():
    print(f"{category}: {len(factors)}个因子")
```

### 4. 计算单个因子

```python
from factor_system.factor_engine.api import calculate_single_factor

# 计算单只股票的RSI
rsi = calculate_single_factor(
    factor_id="RSI14",
    symbol="0700.HK",
    timeframe="15min",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)

print(f"RSI平均值: {rsi.mean():.2f}")
```

### 5. 计算核心因子集

```python
from factor_system.factor_engine.api import calculate_core_factors

# 计算常用技术指标
core_factors = calculate_core_factors(
    symbols=["0700.HK"],
    timeframe="15min",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)

print(f"核心因子: {core_factors.columns.tolist()}")
```

## 📊 支持的因子类别

### 技术指标 (78个)
- 动量: RSI, WILLR, CCI, STOCH
- 趋势: ADX, AROON, DX
- 波动率: ATR, MSTD
- K线形态: 60+ TA-Lib模式

### 重叠研究 (67个)
- 移动平均: MA, EMA, SMA, WMA
- 布林带: BB_20_2.0_Upper/Middle/Lower
- 其他: DEMA, TEMA, TRIMA, KAMA

### 统计因子 (85个)
- 动量: Momentum1-20
- 位置: Position5-20
- 趋势: Trend指标
- 随机: RAND, RPROB系列

### 成交量指标 (16个)
- OBV, VWAP
- Volume_Ratio, Volume_Momentum

## 🔧 高级用法

### 缓存管理

```python
from factor_system.factor_engine.api import (
    prewarm_cache,
    clear_cache,
    get_cache_stats
)

# 预热缓存
prewarm_cache(
    factor_ids=["RSI14", "MACD_12_26_9"],
    symbols=["0700.HK"],
    timeframe="15min",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)

# 查看缓存统计
stats = get_cache_stats()
print(f"缓存命中率: {stats.get('hit_rate', 0):.1%}")

# 清空缓存
clear_cache()
```

### 自定义配置

```python
from factor_system.factor_engine.api import get_engine
from factor_system.factor_engine.core.cache import CacheConfig
from pathlib import Path

# 自定义引擎配置
engine = get_engine(
    raw_data_dir=Path("raw"),
    cache_config=CacheConfig(
        memory_size_mb=1024,  # 1GB内存缓存
        enable_disk=True,
        ttl_hours=24
    ),
    force_reinit=True
)
```

## ⚠️ 重要提示

### 1. 计算一致性保证
所有因子使用 `shared/factor_calculators.py` 确保：
- ✅ factor_engine 计算结果
- ✅ factor_generation 批量生成
- ✅ factor_screening 因子筛选
- ✅ 回测系统

**完全一致！**

### 2. 参数命名
因子支持两种参数命名：
```python
# 方式1: 下划线命名
calculate_factors(["RSI14"])  # 自动解析为 RSI(period=14)

# 方式2: 参数化命名
calculate_factors(["MACD_12_26_9"])  # 自动解析参数
```

### 3. 数据格式
输入数据必须包含OHLCV列：
```python
required_columns = ['open', 'high', 'low', 'close', 'volume']
```

## 🧪 测试验证

```bash
# 运行一致性测试
pytest tests/test_factor_consistency_final.py -v

# 运行完整测试
pytest tests/ -v
```

## 📚 更多资源

- 完整因子列表: `factor_system/FACTOR_REGISTRY.md`
- API文档: `factor_system/factor_engine/api.py`
- 修复报告: `FACTOR_ENGINE_FIX_REPORT.md`
- 项目文档: `factor_system/factor_screening/PROJECT_DOCUMENTATION.md`

## 🎯 最佳实践

1. **使用缓存**: 开启缓存可显著提升性能
2. **批量计算**: 一次计算多个因子比分别计算更高效
3. **合理并行**: 使用 `n_jobs` 参数控制并行度
4. **监控内存**: 大量标的时注意内存使用

---

**更新时间**: 2025-10-09  
**版本**: v2.0 (修复后)
