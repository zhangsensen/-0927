# 开发工作流程和最佳实践

## 核心开发命令

### 环境管理
```bash
# 安装依赖（使用uv现代包管理器）
uv sync

# 激活虚拟环境
source .venv/bin/activate

# 安装FactorEngine（开发模式）
pip install -e .

# 验证安装
python -c "from factor_system.factor_engine import api; print('✅ FactorEngine ready')"
```

### 代码质量控制
```bash
# 运行所有测试
pytest --cov=factor_system

# 类型检查
mypy factor_system/

# 代码格式化
black factor_system/ data-resampling/

# 导入排序
isort factor_system/ data-resampling/

# 运行特定测试类别
pytest -m unit          # 单元测试
pytest -m integration   # 集成测试
pytest -m slow          # 慢速测试
```

### 性能基准测试
```bash
# FactorEngine一致性测试
python tests/test_factor_engine_consistency.py

# 性能基准测试
python factor_system/factor_screening/performance_benchmark.py

# 基本功能测试
python factor_system/factor_screening/tests/test_basic_functionality.py
```

## A股分析工作流程

### 数据获取
```bash
# 使用yfinance下载A股数据
python a股/data_download/simple_download.py

# 批量存储分析
python a股/batch_storage_analysis.py
```

### 技术分析
```bash
# 运行技术分析（154个指标）
python a股/stock_analysis/sz_technical_analysis.py <STOCK_CODE>

# 筛选优质A股
python a股/screen_top_stocks.py
```

### 数据格式
```
A股数据: {SYMBOL_CODE}_1d_YYYY-MM-DD.csv
示例: 000001_1d_2025-10-07.csv
```

## 因子系统工作流程

### 快速开始
```bash
# 多时间框架因子分析
python factor_system/factor_generation/quick_start.py <STOCK_CODE>

# 增强因子计算器
python factor_system/factor_generation/enhanced_factor_calculator.py

# 专业因子筛选CLI
python factor_system/factor_screening/cli.py screen <STOCK_CODE> <TIMEFRAME>

# 批量处理多只股票
python factor_system/factor_screening/batch_screener.py
```

### 专业筛选系统
```bash
# 单股票筛选
python factor_system/factor_screening/cli.py screen 0700.HK 60min

# 批量筛选
python factor_system/factor_screening/cli.py batch --symbols 0700.HK,0005.HK --timeframe 60min

# 生成筛选报告
python factor_system/factor_screening/cli.py report 0700.HK --output screening_report.pdf

# 配置管理
python factor_system/factor_screening/cli.py config list
python factor_system/factor_screening/cli.py config create custom_config.yaml
```

## 港股数据处理

### 批量重采样
```bash
# 将1分钟数据重采样到更高时间框架
python batch_resample_hk.py
```

### 港股中频策略
```bash
# 策略核心运行
python hk_midfreq/strategy_core.py

# 组合回测
python hk_midfreq/combination_backtest.py
```

### 港股数据格式
```
原始数据: {SYMBOL}_1min_YYYY-MM-DD_YYYY-MM-DD.parquet
因子输出: {SYMBOL}_{TIMEFRAME}_factors_YYYY-MM-DD_HH-MM-SS.parquet
示例: 0700.HK_1min_2025-09-01_2025-09-30.parquet
```

## FactorEngine API使用模式

### 推荐使用方式
```python
# ✅ 正确：使用统一API
from factor_system.factor_engine import api
from datetime import datetime

# 单因子计算
rsi = api.calculate_single_factor(
    factor_id="RSI",
    symbol="0700.HK",
    timeframe="15min",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)

# 多因子计算
factors = api.calculate_factors(
    factor_ids=["RSI", "MACD", "STOCH"],
    symbols=["0700.HK", "0005.HK"],
    timeframe="15min",
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 9, 30)
)

# 预热缓存
api.prewarm_cache(
    factor_ids=["RSI", "MACD", "STOCH"],
    symbols=["0700.HK", "0005.HK"],
    timeframe="15min",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31)
)

# 获取缓存统计
stats = api.get_cache_stats()
print(f"缓存命中率: {stats['memory_hit_rate']:.2%}")
```

### 避免的使用方式
```python
# ❌ 避免：直接实例化引擎
from factor_system.factor_engine.core.engine import FactorEngine
```

## 配置管理

### 环境配置
```python
# 研究环境
from factor_system.factor_engine.settings import get_research_config
settings = get_research_config()

# 生产环境
from factor_system.factor_engine.settings import get_production_config
settings = get_production_config()
```

### 环境变量
```bash
export FACTOR_ENGINE_RAW_DATA_DIR="/data/market/raw"
export FACTOR_ENGINE_MEMORY_MB="1024"
export FACTOR_ENGINE_N_JOBS="-1"
export FACTOR_ENGINE_CACHE_DIR="/cache/factors"
```

### 配置文件位置
- FactorEngine配置: `factor_engine/configs/engine_config.yaml`
- 因子生成配置: `factor_generation/config.yaml`
- 策略模板: `factor_system/config/`

## 预提交钩子

### Linus风格钩子
```bash
# 安装预提交钩子
pre-commit install

# 手动运行
pre-commit run --all-files
```

### 钩子功能
- **未来函数检测**: 防止量化策略中的前瞻偏差
- **Python语法检查**: 确保代码可执行
- **代码格式化**: Black和isort统一风格

## 性能优化最佳实践

### 缓存策略
- 预热常用因子缓存
- 监控缓存命中率
- 合理设置TTL（开发2小时，研究24小时，生产7天）

### 并行计算
- 符号级并行化
- 配置合适的job数量
- 避免内存过度使用

### 向量化操作
- 所有向量化操作使用VectorBT
- 避免DataFrame.apply
- 使用NumPy/VectorBT内置函数

## 调试和监控

### 日志配置
- 开发环境: 详细日志
- 研究环境: 信息日志
- 生产环境: 警告日志

### 性能监控
- 内存使用跟踪
- 计算时间基准测试
- 数据质量指标
- 系统健康监控

### 错误处理
- 数据完整性检查
- 缺失数据处理
- 异常值检测
- 跨时间周期数据对齐

## 部署注意事项

### 环境要求
- Python 3.11+
- 充足的内存（推荐8GB+）
- SSD存储（缓存和数据）
- 多核CPU（并行计算）

### 数据管理
- 定期清理过期缓存
- 备份重要因子数据
- 监控存储空间
- 数据版本控制

### 安全考虑
- API密钥管理
- 数据访问控制
- 系统监控告警
- 灾难恢复方案