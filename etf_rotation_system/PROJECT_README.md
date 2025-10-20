# ETF轮动量化交易系统 - 项目架构文档

> **专业级ETF量化投资平台，配置驱动架构v2.0**

---

## 📋 项目概述

ETF轮动系统是一个完整的量化投资解决方案，采用现代软件工程理念和配置驱动架构。系统实现了从原始ETF价格数据到可执行投资策略的全流程自动化，包括因子计算、因子筛选、策略回测等核心功能。

### 核心设计理念
- **配置驱动**: 所有参数通过YAML配置，无硬编码
- **模块化设计**: 清晰的模块边界，便于维护和扩展
- **科学筛选**: 基于统计学的因子筛选框架
- **高性能**: 向量化计算，优化的内存使用
- **可复现**: 严格的时间对齐，无未来函数

---

## 🏗️ 系统架构详解

### 整体架构图
```
┌─────────────────────────────────────────────────────────────────┐
│                        ETF轮动系统架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │   数据源层      │    │   计算引擎层    │    │   结果输出层    │  │
│  │                │    │                │    │                │  │
│  │ ETF价格数据     │───▶│ 因子面板生成     │───▶│ 筛选结果报告     │  │
│  │ 43个ETF        │    │ 36个技术因子     │    │ IC分析统计      │  │
│  │ 5.5年历史      │    │ 向量化计算      │    │ 分层评级结果    │  │
│  │ Parquet格式    │    │ 多线程处理      │    │ 时间戳归档      │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    配置管理层                                  │  │
│  │                                                             │  │
│  │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │  │
│  │ │ YAML配置文件    │ │ │ 配置验证器     │ │ │ 参数控制器     │ │  │
│  │ │                 │ │ │                 │ │ │                 │ │  │
│  │ │ 数据源路径      │ │ │ 类型检查        │ │ │ 动态参数调整    │ │  │
│  │ │ 筛选阈值        │ │ │ 路径验证        │ │ │ 环境变量覆盖    │ │  │
│  │ │ 分析参数        │ │ │ 参数范围检查    │ │ │ 命令行参数      │ │  │
│  │ │ 输出格式        │ │ │ 异常处理        │ │ │ 默认值设置      │ │  │
│  │ └─────────────────┘ └─────────────────┘ └─────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    核心业务层                                    │  │
│  │                                                             │  │
│  │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │  │
│  │ │ IC分析引擎      │ │ │ 因子筛选器      │ │ │ 统学校正器     │ │  │
│  │ │                 │ │ │                 │ │ │                 │ │  │
│  │ │ 多周期IC计算    │ │ │ 基础筛选        │ │ │ FDR校正        │ │  │
│  │ │ 向量化相关计算  │ │ │ 相关性去重      │ │ │ t检验          │ │  │
│  │ │ 稳定性分析      │ │ │ 分层评级        │ │ │ 效应量计算      │ │  │
│  │ │ 覆盖率统计      │ │ │ 动态阈值调整    │ │ │ 多重检验控制    │ │  │
│  │ └─────────────────┘ └─────────────────┘ └─────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 模块详细说明

#### 1. 横截面建设模块 (01_横截面建设/)
**功能**: ETF因子面板生成
- **输入**: 原始ETF价格数据 (raw/ETF/daily/*.parquet)
- **输出**: 因子面板数据 (data/results/panels/)
- **核心文件**: `generate_panel.py`
- **因子数量**: 36个技术指标
- **处理方式**: 向量化计算，多线程处理

**因子类别**:
- **动量因子**: MOMENTUM_20D, MOMENTUM_63D, MOMENTUM_126D, MOMENTUM_252D
- **波动率因子**: VOLATILITY_20D, VOLATILITY_60D, VOLATILITY_120D
- **价格位置因子**: PRICE_POSITION_20D, PRICE_POSITION_60D, PRICE_POSITION_120D
- **技术指标因子**: RSI_6, RSI_14, RSI_24, ATR_14
- **成交量因子**: VOLUME_RATIO_5D, VOLUME_RATIO_20D, VOLUME_RATIO_60D
- **形态因子**: DOJI_PATTERN, BULLISH_ENGULFING, HAMMER_PATTERN
- **复合因子**: VOL_MA_RATIO_5, VOL_VOLATILITY_20, PRICE_VOLUME_DIV

#### 2. 因子筛选模块 (02_因子筛选/) ⭐ 核心模块
**功能**: 科学化因子筛选和分析
- **输入**: 因子面板数据
- **输出**: 筛选结果和详细报告
- **架构**: 配置驱动设计
- **筛选方法**: 5维度专业筛选框架

**核心组件**:
```python
run_etf_cross_section_configurable.py    # 主筛选脚本 (推荐)
etf_cross_section_config.py             # 配置类定义
sample_etf_config.yaml                  # 示例配置文件
```

**筛选流程**:
1. **IC分析**: 多周期信息系数计算 (1, 5, 10, 20日)
2. **基础筛选**: IC阈值、IR阈值、p值、覆盖率
3. **FDR校正**: Benjamini-Hochberg假发现率校正
4. **相关性去重**: Spearman相关性分析，贪心去重算法
5. **分层评级**: 核心/补充/研究因子三级分类

#### 3. VBT回测模块 (03_vbt回测/)
**功能**: 大规模策略回测和优化
- **引擎**: VectorBT专业回测框架
- **规模**: 支持数千种策略组合测试
- **指标**: Sharpe比率、最大回撤、年化收益等

---

## 🎛️ 配置系统架构

### 配置层次结构
```
ETFCrossSectionConfig (根配置)
├── DataSourceConfig (数据源配置)
│   ├── price_dir: Path              # 价格数据目录
│   ├── panel_file: Path            # 面板文件路径
│   ├── price_columns: List[str]     # 读取列名
│   ├── file_pattern: str           # 文件匹配模式
│   └── symbol_extract_method: str   # Symbol提取方法
├── AnalysisConfig (分析配置)
│   ├── ic_periods: List[int]       # IC分析周期
│   ├── min_observations: int       # 最小观测值
│   ├── correlation_method: str     # 相关性计算方法
│   └── epsilon_small: float        # 小值防除零
├── ScreeningConfig (筛选配置)
│   ├── min_ic: float               # 最小IC阈值
│   ├── min_ir: float               # 最小IR阈值
│   ├── max_pvalue: float           # 最大p值
│   ├── use_fdr: bool               # FDR校正开关
│   ├── max_correlation: float      # 最大相关性
│   └── tier_thresholds: Dict        # 分层阈值配置
└── OutputConfig (输出配置)
    ├── output_dir: Path            # 输出目录
    ├── use_timestamp_subdir: bool  # 时间戳子目录
    ├── files: Dict                 # 文件命名配置
    └── include_factor_details: bool # 详情包含开关
```

### 预设配置模板
```python
# 标准配置 (日常使用)
ETF_STANDARD_CONFIG = ETFCrossSectionConfig(
    screening=ScreeningConfig(
        min_ic=0.005,      # 0.5% IC阈值
        min_ir=0.05,       # 0.05 IR阈值
        use_fdr=True       # 启用FDR校正
    )
)

# 严格配置 (生产环境)
ETF_STRICT_CONFIG = ETFCrossSectionConfig(
    screening=ScreeningConfig(
        min_ic=0.008,      # 0.8% IC阈值
        min_ir=0.08,       # 0.08 IR阈值
        fdr_alpha=0.1      # 更严格FDR
    )
)

# 宽松配置 (研究探索)
ETF_RELAXED_CONFIG = ETFCrossSectionConfig(
    screening=ScreeningConfig(
        min_ic=0.003,      # 0.3% IC阈值
        min_ir=0.03,       # 0.03 IR阈值
        use_fdr=False      # 禁用FDR
    )
)
```

---

## 📊 数据流程详解

### 数据输入规范
```python
# 价格数据格式
price_data/
├── 510300.SH_daily_20200102_20251014.parquet
├── 159919.SZ_daily_20200102_20251014.parquet
└── ... (43个ETF文件)

# 数据结构
DataFrame(columns=['trade_date', 'close'])
- trade_date: 交易日期 (datetime64[ns])
- close: 收盘价 (float64)
- 索引: 日期时间序列
```

### 因子面板结构
```python
# 因子面板格式
panel_data/
└── panel_20251020_162504.parquet

# 数据结构
DataFrame(index=['symbol', 'date'])
- symbol: ETF代码 (str)
- date: 交易日期 (datetime64[ns])
- columns: 36个因子值 (float64)
- shape: (56575, 36) - 56,575个观测值
```

### 筛选结果输出
```python
screening_results/
└── screening_20251020_184913/
    ├── ic_analysis.csv          # 完整IC分析 (36行 × 多列)
    ├── passed_factors.csv       # 通过筛选因子 (8行 × 12列)
    └── screening_report.txt     # 详细文字报告
```

---

## 🔍 因子筛选科学框架

### IC (Information Coefficient) 分析
```python
# IC计算公式
IC_t = rank_corr(factor_t, return_{t+1})

# 多周期IC分析
periods = [1, 5, 10, 20]  # 预测周期
for period in periods:
    future_return = price.pct_change(period).shift(-period)
    ic = spearman_corr(factor, future_return)
```

### FDR (False Discovery Rate) 校正
```python
# Benjamini-Hochberg算法
def fdr_correction(p_values, alpha=0.2):
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    critical = np.arange(1, n + 1) * alpha / n

    rejected = sorted_p <= critical
    return sorted_idx[rejected.any()][:rejected[rejected].max() + 1]
```

### 相关性去重算法
```python
# 贪心去重策略
def remove_correlated_factors(ic_df, panel, max_corr=0.7):
    factors = ic_df['factor'].tolist()
    corr_matrix = panel[factors].corr(method='spearman').abs()

    to_remove = set()
    for i, f1 in enumerate(factors):
        for f2 in factors[i+1:]:
            if corr_matrix.loc[f1, f2] > max_corr:
                # 保留IC_IR更高的因子
                ir1 = ic_df[ic_df['factor'] == f1]['ic_ir'].values[0]
                ir2 = ic_df[ic_df['factor'] == f2]['ic_ir'].values[0]
                to_remove.add(f2 if abs(ir1) > abs(ir2) else f1)

    return ic_df[~ic_df['factor'].isin(to_remove)]
```

### 分层评级体系
```python
# 三级分类体系
def classify_factor(ic_mean, ic_ir):
    if abs(ic_mean) >= 0.02 and abs(ic_ir) >= 0.1:
        return "🟢 核心"    # 高预测力 + 高稳定性
    elif abs(ic_mean) >= 0.01 and abs(ic_ir) >= 0.07:
        return "🟡 补充"    # 中等预测力 + 中等稳定性
    else:
        return "🔵 研究"    # 基础预测力，需要进一步验证
```

---

## ⚡ 性能优化策略

### 计算性能优化
```python
# 向量化计算 (避免循环)
def calculate_ic_vectorized(factor_matrix, return_matrix):
    """批量计算IC，完全向量化"""
    factor_ranked = np.apply_along_axis(rank_row, 1, factor_matrix)
    return_ranked = np.apply_along_axis(rank_row, 1, return_matrix)

    # 批量相关系数计算
    ics = []
    for i in range(factor_matrix.shape[0]):
        ic = np.corrcoef(factor_ranked[i], return_ranked[i])[0, 1]
        ics.append(ic)
    return np.array(ics)
```

### 内存管理优化
```python
# 分块处理大数据集
def process_large_dataset(panel, chunk_size=1000):
    """分块处理，控制内存使用"""
    results = []
    symbols = panel.index.get_level_values('symbol').unique()

    for i in range(0, len(symbols), chunk_size):
        chunk_symbols = symbols[i:i+chunk_size]
        chunk_data = panel.loc[chunk_symbols]
        chunk_result = analyze_chunk(chunk_data)
        results.append(chunk_result)

        # 清理内存
        del chunk_data
        gc.collect()

    return pd.concat(results, ignore_index=True)
```

### 缓存策略
```python
# IC计算结果缓存
@lru_cache(maxsize=128)
def calculate_ic_cached(panel_hash, factor_name, period):
    """缓存IC计算结果，避免重复计算"""
    return calculate_ic_internal(panel_hash, factor_name, period)
```

---

## 📈 系统监控和质量保证

### 数据质量监控
```python
# 数据完整性检查
def validate_data_quality(panel):
    """检查数据完整性和质量"""
    checks = {
        'total_observations': len(panel),
        'unique_symbols': panel.index.get_level_values('symbol').nunique(),
        'date_range': (panel.index.get_level_values('date').min(),
                       panel.index.get_level_values('date').max()),
        'missing_rate': panel.isna().sum().sum() / panel.size,
        'factor_coverage': (1 - panel.isna().all(axis=1).mean())
    }
    return checks
```

### 性能监控
```python
# 执行时间监控
import time
import psutil

def monitor_performance(func):
    """性能监控装饰器"""
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = process.memory_info().rss

        metrics = {
            'execution_time': end_time - start_time,
            'memory_used': (end_memory - start_memory) / 1024 / 1024,  # MB
            'throughput': len(result) / (end_time - start_time)
        }
        return result, metrics
    return wrapper
```

### 异常处理机制
```python
# 全面的异常处理
class ETFSystemError(Exception):
    """系统基础异常类"""
    pass

class ConfigurationError(ETFSystemError):
    """配置相关异常"""
    pass

class DataValidationError(ETFSystemError):
    """数据验证异常"""
    pass

class CalculationError(ETFSystemError):
    """计算过程异常"""
    pass
```

---

## 🔧 开发和部署指南

### 开发环境设置
```bash
# 克隆项目
git clone <repository_url>
cd etf_rotation_system

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 代码质量标准
```python
# 代码风格 (Black, isort)
black etf_rotation_system/
isort etf_rotation_system/

# 类型检查 (mypy)
mypy etf_rotation_system/02_因子筛选/

# 代码质量 (pylint)
pylint etf_rotation_system/02_因子筛选/run_etf_cross_section_configurable.py
```

### 测试策略
```python
# 单元测试
pytest tests/unit/

# 集成测试
pytest tests/integration/

# 性能测试
pytest tests/performance/

# 覆盖率报告
pytest --cov=etf_rotation_system --cov-report=html
```

### 部署配置
```yaml
# 生产环境配置
data_source:
  price_dir: "/data/etf/production/daily"
  panel_file: "/data/etf/production/latest_panel.parquet"

screening:
  min_ic: 0.008          # 更严格的标准
  min_ir: 0.08
  use_fdr: true
  fdr_alpha: 0.1

output:
  output_dir: "/data/etf/results/production"
  use_timestamp_subdir: true
```

---

## 📚 API参考

### 核心类和函数
```python
# 主要筛选器类
class ETFCrossSectionScreener:
    def __init__(self, config: ETFCrossSectionConfig)
    def calculate_multi_period_ic(self, panel: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame
    def screen_factors(self, ic_df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame
    def run(self) -> Dict[str, Any]

# 配置类
class ETFCrossSectionConfig:
    @classmethod
    def from_yaml(cls, yaml_file: Path) -> 'ETFCrossSectionConfig'
    def to_yaml(self, yaml_file: Path) -> None
    def validate(self) -> List[str]
```

### 使用示例
```python
# 基础使用
from etf_cross_section_config import ETFCrossSectionConfig
from run_etf_cross_section_configurable import ETFCrossSectionScreener

# 加载配置
config = ETFCrossSectionConfig.from_yaml('production_config.yaml')

# 创建筛选器并运行
screener = ETFCrossSectionScreener(config)
results = screener.run()

print(f"通过因子: {results['passed_factors']}/{results['total_factors']}")
print(f"结果目录: {results['output_dir']}")
```

---

## 🎯 最佳实践和建议

### 配置管理
1. **版本控制**: 所有配置文件纳入版本控制
2. **环境分离**: 开发/测试/生产环境使用不同配置
3. **参数验证**: 使用内置验证功能检查配置合理性
4. **文档更新**: 配置变更时同步更新文档

### 数据管理
1. **数据质量**: 定期检查价格数据完整性和准确性
2. **备份策略**: 重要数据定期备份
3. **更新频率**: 根据市场情况确定数据更新频率
4. **存储优化**: 使用parquet格式，考虑数据压缩

### 性能优化
1. **合理配置**: 根据硬件资源调整IC分析周期
2. **缓存利用**: 充分利用计算结果缓存
3. **并行处理**: 在可能的情况下使用多进程计算
4. **内存监控**: 监控内存使用，避免内存溢出

### 结果验证
1. **交叉验证**: 使用不同时间段验证因子稳定性
2. **样本外测试**: 定期进行样本外测试
3. **结果记录**: 详细记录每次运行配置和结果
4. **异常分析**: 分析异常结果的原因和改进方案

---

## 📞 技术支持

### 问题诊断
```bash
# 检查系统状态
python -c "
from etf_cross_section_config import ETFCrossSectionConfig
import pandas as pd
from pathlib import Path

# 检查配置
config = ETFCrossSectionConfig.from_yaml('sample_etf_config.yaml')
print('配置加载: ✅')

# 检查数据
panel_file = Path('/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels/panel_20251020_162504/panel.parquet')
if panel_file.exists():
    panel = pd.read_parquet(panel_file)
    print(f'数据检查: ✅ (形状: {panel.shape})')
else:
    print('数据检查: ❌ (文件不存在)')
"
```

### 常见问题解决
1. **内存不足**: 减少IC分析周期或使用分块处理
2. **配置错误**: 使用配置验证功能检查参数
3. **路径问题**: 确保所有路径使用绝对路径
4. **依赖缺失**: 检查Python环境和包安装状态

---

**文档版本**: v2.0
**最后更新**: 2025-10-20
**维护团队**: ETF轮动系统开发团队
**技术支持**: 通过GitHub Issues或邮件联系