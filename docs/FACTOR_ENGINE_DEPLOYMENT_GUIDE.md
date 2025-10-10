# FactorEngine 部署指南

## 概述

FactorEngine 是专业级因子计算引擎，为研究、回测、组合管理等环境提供统一的因子计算核心。本指南详细说明如何在不同环境中部署和集成 FactorEngine。

## 核心特性

- ✅ **统一计算逻辑**：消除研究和回测的计算偏差
- ✅ **100+技术指标**：覆盖技术分析、动量、趋势、波动率等类别
- ✅ **双层缓存系统**：内存+磁盘缓存，大幅提升性能
- ✅ **配置化部署**：支持环境变量和配置文件
- ✅ **向后兼容**：平滑迁移现有代码

## 快速开始

### 1. 安装方式

#### 本地开发安装（推荐）
```bash
# 在项目根目录
pip install -e .
```

#### 从Git仓库安装
```bash
pip install git+ssh://git@github.com/yourorg/factor-engine.git
```

#### 从私有PyPI安装
```bash
pip install factor-engine --index-url https://pypi.yourcompany.com/simple/
```

### 2. 基本使用

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
print(f"可用因子: {api.list_available_factors()}")
```

## 环境配置

### 环境变量配置

FactorEngine 支持通过环境变量进行配置，推荐在不同环境中使用不同的配置：

```bash
# 数据路径配置
export FACTOR_ENGINE_RAW_DATA_DIR="/data/market/raw"
export FACTOR_ENGINE_CACHE_DIR="/cache/factors"
export FACTOR_ENGINE_REGISTRY_FILE="/config/factor_registry.json"

# 缓存配置
export FACTOR_ENGINE_MEMORY_MB="1024"          # 内存缓存大小
export FACTOR_ENGINE_TTL_HOURS="168"           # 缓存生存时间(小时)
export FACTOR_ENGINE_ENABLE_MEMORY="true"      # 启用内存缓存
export FACTOR_ENGINE_ENABLE_DISK="true"        # 启用磁盘缓存
export FACTOR_ENGINE_COPY_MODE="view"          # 数据复制模式

# 引擎配置
export FACTOR_ENGINE_N_JOBS="-1"               # 并行计算核心数
export FACTOR_ENGINE_VALIDATE_DATA="false"     # 数据验证
export FACTOR_ENGINE_LOG_LEVEL="WARNING"       # 日志级别
```

### 预定义配置模板

#### 开发环境
```python
from factor_system.factor_engine.settings import get_development_config

settings = get_development_config()
# 内存缓存: 200MB
# 缓存时间: 2小时
# 单线程计算
# 详细日志输出
```

#### 研究环境
```python
from factor_system.factor_engine.settings import get_research_config

settings = get_research_config()
# 内存缓存: 512MB
# 缓存时间: 24小时
# 4核心并行
# 信息级别日志
```

#### 生产环境
```python
from factor_system.factor_engine.settings import get_production_config

settings = get_production_config()
# 内存缓存: 1GB
# 缓存时间: 7天
# 全核心并行
# 仅警告和错误日志
```

## 环境集成指南

### 1. 研究环境集成

研究环境通常需要灵活的配置和详细的日志。

#### 项目结构
```
research_project/
├── pyproject.toml
├── notebooks/
├── scripts/
└── config/
    └── factor_config.yaml
```

#### pyproject.toml 配置
```toml
[project]
dependencies = [
    "factor-engine>=0.2.0",
    "pandas>=2.3.2",
    "numpy>=2.3.3",
    "matplotlib>=3.10.6",
]

[project.optional-dependencies]
research = [
    "jupyterlab>=4.4.9",
    "seaborn>=0.13.2",
    "plotly>=5.24.0",
]
```

#### 研究脚本示例
```python
#!/usr/bin/env python3
"""因子研究示例"""

from factor_system.factor_engine import api
from factor_system.factor_engine.settings import get_research_config
from datetime import datetime, timedelta

def research_factor_analysis():
    """因子研究分析"""

    # 使用研究环境配置
    settings = get_research_config()

    # 计算多只股票的动量因子
    symbols = ["0700.HK", "0005.HK", "0941.HK", "1398.HK"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # 批量计算动量因子
    momentum_factors = api.calculate_momentum_factors(
        symbols=symbols,
        timeframe="15min",
        start_date=start_date,
        end_date=end_date,
    )

    # 因子分析逻辑...
    print(f"计算完成，因子数据形状: {momentum_factors.shape}")

    # 列出所有可用因子类别
    categories = api.list_factor_categories()
    print("可用因子类别:", list(categories.keys()))

if __name__ == "__main__":
    research_factor_analysis()
```

### 2. 回测环境集成

回测环境需要高性能和稳定性，确保与因子生成阶段的一致性。

#### 项目结构
```
backtest_project/
├── pyproject.toml
├── src/
│   └── backtest/
│       ├── strategy.py
│       └── factor_provider.py
└── tests/
```

#### 回测适配器使用
```python
"""回测因子提供者"""

from factor_system.factor_engine import api
from datetime import datetime

class BacktestFactorProvider:
    """回测因子提供者"""

    def __init__(self):
        # 使用生产环境配置
        from factor_system.factor_engine.settings import get_production_config
        settings = get_production_config()

        # 预热缓存
        api.prewarm_cache(
            factor_ids=["RSI", "MACD", "STOCH", "WILLR"],
            symbols=["0700.HK"],
            timeframe="15min",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 12, 31),
        )

    def get_factor(self, factor_id: str, symbol: str, date: datetime):
        """获取指定日期的因子值"""
        factor_series = api.calculate_single_factor(
            factor_id=factor_id,
            symbol=symbol,
            timeframe="15min",
            start_date=date,
            end_date=date + timedelta(hours=1),
        )

        return factor_series.iloc[-1] if not factor_series.empty else None

# 在策略中使用
provider = BacktestFactorProvider()
rsi_value = provider.get_factor("RSI", "0700.HK", datetime(2025, 9, 15))
```

### 3. 组合管理环境集成

组合管理需要稳定的因子计算和批量处理能力。

#### 组合因子计算示例
```python
"""组合因子计算"""

from factor_system.factor_engine import api
from factor_system.factor_engine.settings import get_production_config
import pandas as pd

class PortfolioFactorCalculator:
    """组合因子计算器"""

    def __init__(self, universe: list):
        self.universe = universe
        self.settings = get_production_config()

    def calculate_universe_factors(self, date_range):
        """计算全市场因子"""
        all_factors = []

        for symbol in self.universe:
            try:
                # 计算核心因子集
                factors = api.calculate_core_factors(
                    symbols=[symbol],
                    timeframe="daily",
                    start_date=date_range[0],
                    end_date=date_range[1],
                )
                all_factors.append(factors)

            except Exception as e:
                print(f"计算{symbol}因子失败: {e}")
                continue

        # 合并所有因子
        if all_factors:
            universe_factors = pd.concat(all_factors)
            return universe_factors
        else:
            return pd.DataFrame()

    def factor_ranking(self, factors_df, factor_id: str):
        """因子排名"""
        if factor_id not in factors_df.columns:
            raise ValueError(f"因子{factor_id}不存在")

        # 按因子值排名
        ranking = factors_df[factor_id].groupby('timestamp').rank(ascending=False)
        return ranking

# 使用示例
universe = ["0700.HK", "0005.HK", "0941.HK", "1398.HK", "2318.HK"]
calculator = PortfolioFactorCalculator(universe)

# 计算月度因子
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

universe_factors = calculator.calculate_universe_factors([start_date, end_date])
rsi_ranking = calculator.factor_ranking(universe_factors, "RSI")
```

## 性能优化

### 1. 缓存策略

#### 内存缓存优化
```bash
# 根据系统内存调整
export FACTOR_ENGINE_MEMORY_MB="2048"  # 2GB内存缓存
export FACTOR_ENGINE_COPY_MODE="view"  # 使用view模式减少内存复制
```

#### 磁盘缓存优化
```bash
# 使用SSD存储缓存
export FACTOR_ENGINE_CACHE_DIR="/ssd/cache/factors"
export FACTOR_ENGINE_TTL_HOURS="720"   # 30天缓存
```

### 2. 并行计算

```bash
# 根据CPU核心数调整
export FACTOR_ENGINE_N_JOBS="-1"       # 使用所有核心
export FACTOR_ENGINE_N_JOBS="8"        # 使用8个核心
```

### 3. 数据预处理

```python
# 预热常用因子缓存
api.prewarm_cache(
    factor_ids=["RSI", "MACD", "STOCH", "WILLR", "CCI"],
    symbols=["0700.HK", "0005.HK", "0941.HK"],
    timeframe="15min",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31),
)
```

## 监控和日志

### 1. 日志配置

```python
import logging

# 设置日志级别
logging.getLogger("factor_system.factor_engine").setLevel(logging.INFO)

# 添加文件处理器
file_handler = logging.FileHandler("factor_engine.log")
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logging.getLogger("factor_system.factor_engine").addHandler(file_handler)
```

### 2. 性能监控

```python
# 获取缓存统计
cache_stats = api.get_cache_stats()
print(f"内存命中率: {cache_stats['memory_hit_rate']:.2%}")
print(f"磁盘命中率: {cache_stats['disk_hit_rate']:.2%}")
print(f"缓存大小: {cache_stats['cache_size_mb']}MB")

# 列出可用因子
factors = api.list_available_factors()
print(f"注册因子数量: {len(factors)}")
```

## 故障排除

### 1. 常见问题

#### 因子计算失败
```python
# 检查因子是否可用
try:
    api.calculate_single_factor("UNKNOWN_FACTOR", "0700.HK", "15min",
                               datetime(2025,9,1), datetime(2025,9,2))
except api.UnknownFactorError as e:
    print(e)  # 显示可用因子列表
```

#### 数据路径问题
```python
# 检查数据路径
settings = get_settings()
print(f"原始数据目录: {settings.data_paths.raw_data_dir}")
print(f"缓存目录: {settings.data_paths.cache_dir}")

# 确保目录存在
settings.ensure_directories()
```

#### 内存不足
```python
# 调整缓存大小
from factor_system.factor_engine.settings import set_memory_size_mb
set_memory_size_mb(256)  # 减少到256MB

# 清除缓存
api.clear_cache()
```

### 2. 调试模式

```python
# 启用详细日志
import logging
logging.getLogger("factor_system.factor_engine").setLevel(logging.DEBUG)

# 强制重新初始化引擎
engine = api.get_engine(force_reinit=True)
```

## 版本管理

### 1. 版本号说明

FactorEngine 使用语义化版本号：`主版本.次版本.修订版本`

- **主版本**：不兼容的API变更
- **次版本**：向后兼容的功能性新增
- **修订版本**：向后兼容的问题修正

### 2. 版本升级

#### 检查版本
```python
from factor_system.factor_engine import __version__
print(f"FactorEngine版本: {__version__}")
```

#### 升级指南
```bash
# 升级到最新版本
pip install --upgrade factor-engine

# 升级到指定版本
pip install factor-engine==0.2.1

# 升级预发布版本
pip install --pre factor-engine
```

## 最佳实践

### 1. 项目组织

```
project/
├── pyproject.toml          # 项目配置
├── config/
│   ├── factor_config.yaml  # 因子配置
│   └── data_config.yaml    # 数据配置
├── src/
│   └── factor_analysis.py  # 因子分析代码
├── notebooks/              # Jupyter notebooks
├── tests/                  # 测试代码
└── data/                   # 本地数据
```

### 2. 配置管理

```python
# config/factor_config.yaml
factor_engine:
  cache:
    memory_mb: 512
    ttl_hours: 24
  engine:
    n_jobs: 4
    validate_data: true
    log_level: "INFO"

# 代码中加载配置
import yaml
from factor_system.factor_engine.settings import FactorEngineSettings

with open("config/factor_config.yaml", "r") as f:
    config = yaml.safe_load(f)

settings = FactorEngineSettings(**config["factor_engine"])
```

### 3. 错误处理

```python
from factor_system.factor_engine import api, UnknownFactorError

def safe_calculate_factors(factor_ids, symbols, **kwargs):
    """安全计算因子"""
    try:
        return api.calculate_factors(factor_ids, symbols, **kwargs)
    except UnknownFactorError as e:
        # 记录错误并使用替代因子
        available_factors = api.list_available_factors()
        valid_factors = [f for f in factor_ids if f in available_factors]

        if valid_factors:
            print(f"部分因子不可用，使用替代因子: {valid_factors}")
            return api.calculate_factors(valid_factors, symbols, **kwargs)
        else:
            raise e
    except Exception as e:
        print(f"因子计算失败: {e}")
        return pd.DataFrame()
```

### 4. 测试

```python
import pytest
from factor_system.factor_engine import api
from datetime import datetime

def test_factor_consistency():
    """测试因子一致性"""
    # 相同输入应该产生相同输出
    result1 = api.calculate_single_factor(
        "RSI", "0700.HK", "15min",
        datetime(2025,9,1), datetime(2025,9,2)
    )

    result2 = api.calculate_single_factor(
        "RSI", "0700.HK", "15min",
        datetime(2025,9,1), datetime(2025,9,2)
    )

    pd.testing.assert_series_equal(result1, result2)
```

## 技术支持

- **文档**: [FactorEngine Documentation](https://factor-engine.readthedocs.io)
- **问题反馈**: [GitHub Issues](https://github.com/yourorg/factor-engine/issues)
- **技术支持**: factor-engine@yourcompany.com

---

**版本**: 0.2.0
**更新日期**: 2025-10-07
**维护团队**: 量化工程团队