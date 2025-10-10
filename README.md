# FactorEngine

专业级因子计算引擎 - 研究与回测的统一计算核心

## 概述

FactorEngine 是为量化交易系统设计的专业级因子计算引擎，提供统一的技术指标计算能力，确保研究、回测、组合管理等环境使用完全相同的计算逻辑。

**基于 VectorBT + TA-Lib 成熟实现**，不重复造轮子，确保与 factor_generation 计算逻辑完全一致。

## 核心特性

- ✅ **统一计算逻辑**：消除研究和回测的计算偏差
- ✅ **VectorBT驱动**：基于成熟的 VectorBT + TA-Lib，计算逻辑与 factor_generation 完全一致
- ✅ **100+技术指标**：覆盖技术分析、动量、趋势、波动率等类别
- ✅ **双层缓存系统**：内存+磁盘缓存，大幅提升性能
- ✅ **配置化部署**：支持环境变量和配置文件
- ✅ **向后兼容**：平滑迁移现有代码

## 快速开始

### 安装

```bash
# 本地开发安装
pip install -e .

# 从Git仓库安装
pip install git+ssh://git@github.com/yourorg/factor-engine.git
```

### 基本使用

```python
from factor_system.factor_engine import api
from datetime import datetime

# 计算核心技术指标
factors = api.calculate_factors(
    factor_ids=["RSI", "MACD", "STOCH"],
    symbols=["0700.HK", "0005.HK"],
    timeframe="15min",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30),
)

print(f"因子数据形状: {factors.shape}")
```

## 文档

- [部署指南](FACTOR_ENGINE_DEPLOYMENT_GUIDE.md)
- [API文档](https://factor-engine.readthedocs.io)
- [示例代码](examples/)

## 版本

当前版本：0.2.0

## 许可证

MIT License