# 因子筛选系统 API 参考文档

> **版本**: 2.0.0  
> **更新日期**: 2025-10-03  
> **作者**: 量化首席工程师

---

## 📚 目录

- [核心类](#核心类)
  - [ProfessionalFactorScreener](#professionalfactorscreener)
  - [EnhancedResultManager](#enhancedresultmanager)
  - [ScreeningConfig](#screeningconfig)
- [数据类](#数据类)
  - [FactorMetrics](#factormetrics)
  - [ScreeningSession](#screeningsession)
- [工具函数](#工具函数)
- [使用示例](#使用示例)

---

## 核心类

### ProfessionalFactorScreener

专业级因子筛选器，实现5维度筛选框架。

#### 初始化

```python
def __init__(
    self,
    data_root: Optional[str] = None,
    config: Optional[ScreeningConfig] = None
) -> None
```

**参数**:
- `data_root`: 数据根目录路径，默认为`"./data"`
- `config`: 筛选配置对象，默认加载`screening_config.yaml`

**示例**:
```python
from professional_factor_screener import ProfessionalFactorScreener
from config_manager import ScreeningConfig

# 使用默认配置
screener = ProfessionalFactorScreener()

# 使用自定义配置
config = ScreeningConfig(
    ic_horizons=[1, 3, 5],
    alpha_level=0.05,
    min_sample_size=200
)
screener = ProfessionalFactorScreener(
    data_root="/path/to/data",
    config=config
)
```

---

#### 核心方法

##### 1. screen_factors_comprehensive

综合5维度因子筛选。

```python
def screen_factors_comprehensive(
    self,
    symbol: str,
    timeframe: str = "60min",
    price_data: Optional[pd.DataFrame] = None,
    factor_data: Optional[pd.DataFrame] = None
) -> Dict[str, FactorMetrics]
```

**参数**:
- `symbol`: 股票代码，如`"0700.HK"`
- `timeframe`: 时间框架，支持`"5min"`, `"15min"`, `"30min"`, `"60min"`, `"daily"`
- `price_data`: 可选的价格数据DataFrame
- `factor_data`: 可选的因子数据DataFrame

**返回**:
- `Dict[str, FactorMetrics]`: 因子名称到指标的映射

**示例**:
```python
results = screener.screen_factors_comprehensive(
    symbol="0700.HK",
    timeframe="60min"
)

# 访问结果
for factor_name, metrics in results.items():
    print(f"{factor_name}: {metrics.comprehensive_score:.3f}")
```

---

##### 2. calculate_multi_horizon_ic

计算多周期IC值（信息系数）。

```python
def calculate_multi_horizon_ic(
    self,
    factors: pd.DataFrame,
    returns: pd.Series
) -> Dict[str, Dict[str, float]]
```

**参数**:
- `factors`: 因子数据DataFrame，索引为时间戳
- `returns`: 收益率序列

**返回**:
- `Dict[str, Dict[str, float]]`: 嵌套字典，格式为`{factor_name: {horizon: ic_value}}`

**算法说明**:
- 使用Spearman等级相关系数计算IC
- 支持多个预测周期（1日、3日、5日、10日、20日）
- 自动处理时间对齐，严格防止未来函数

**示例**:
```python
ic_results = screener.calculate_multi_horizon_ic(factors, returns)
print(ic_results["sma_20"]["1d"])  # 1日IC值
```

---

##### 3. calculate_rolling_ic

计算滚动IC（稳定性评估）。

```python
def calculate_rolling_ic(
    self,
    factors: pd.DataFrame,
    returns: pd.Series,
    window: int = None
) -> Dict[str, Dict[str, float]]
```

**参数**:
- `factors`: 因子数据DataFrame
- `returns`: 收益率序列
- `window`: 滚动窗口大小，默认使用配置中的`rolling_window`

**返回**:
- `Dict[str, Dict[str, float]]`: 滚动IC统计指标
  - `rolling_ic_mean`: 滚动IC均值
  - `rolling_ic_std`: 滚动IC标准差
  - `rolling_ic_stability`: 稳定性指标（均值/标准差）

**性能优化**:
- **向量化实现**：避免Python循环，使用`pandas.DataFrame.rolling`
- **内存优化**：仅保留统计量，不保存完整时间序列
- **并行计算**：支持多核心并行处理

**示例**:
```python
rolling_ic = screener.calculate_rolling_ic(factors, returns, window=60)
stability = rolling_ic["sma_20"]["rolling_ic_stability"]
```

---

##### 4. calculate_vif_scores

计算方差膨胀因子（VIF），评估因子独立性。

```python
def calculate_vif_scores(
    self,
    factors: pd.DataFrame,
    vif_threshold: float = 5.0,
    max_iterations: int = 10
) -> Dict[str, float]
```

**参数**:
- `factors`: 因子数据DataFrame
- `vif_threshold`: VIF阈值，默认5.0
- `max_iterations`: 最大迭代次数

**返回**:
- `Dict[str, float]`: 因子名称到VIF值的映射

**算法说明**:
- **迭代法**：递归移除高VIF因子直到所有VIF < threshold
- **多重共线性检测**：VIF > 10表示严重共线性
- **计算公式**: `VIF_i = 1 / (1 - R²_i)`

**示例**:
```python
vif_scores = screener.calculate_vif_scores(factors)
for factor, vif in vif_scores.items():
    if vif > 10:
        print(f"警告: {factor} VIF过高 ({vif:.2f})")
```

---

##### 5. benjamini_hochberg_correction

Benjamini-Hochberg FDR多重假设检验校正。

```python
def benjamini_hochberg_correction(
    self,
    p_values: Dict[str, float],
    alpha: float = None,
    sample_size: int = None
) -> Tuple[Dict[str, float], float]
```

**参数**:
- `p_values`: 因子名称到p值的映射
- `alpha`: 显著性水平，默认使用配置中的`alpha_level`
- `sample_size`: 样本量，用于自适应阈值调整

**返回**:
- `Tuple[Dict[str, float], float]`: (校正后p值字典, 有效alpha阈值)

**统计原理**:
- **FDR控制**: 控制假发现率（False Discovery Rate）
- **自适应阈值**: 小样本时更严格（α/2），大样本时放宽（α×1.2）
- **排序算法**: 按p值升序排序，逐个判断是否 `p_i ≤ (i/m) × α`

**示例**:
```python
p_values = {"sma_20": 0.001, "ema_10": 0.05, "rsi_14": 0.2}
corrected_p, alpha_threshold = screener.benjamini_hochberg_correction(
    p_values, alpha=0.05
)
```

---

##### 6. calculate_turnover_rate

计算因子换手率（交易成本评估）。

```python
def calculate_turnover_rate(
    self,
    factor_series: pd.Series,
    factor_name: str = "",
    factor_type: Optional[str] = None,
    turnover_profile: Optional[str] = None
) -> float
```

**参数**:
- `factor_series`: 因子时间序列
- `factor_name`: 因子名称（用于分类）
- `factor_type`: 因子类型（trend/volatility/volume等）
- `turnover_profile`: 换手率计算策略（`"cumulative"`或`"differential"`）

**返回**:
- `float`: 标准化换手率（0~2.0）

**算法说明**:
- **累积型因子**（如MA、EMA）：使用百分比变化 `pct_change()`
- **差分型因子**（如MACD、RSI）：使用绝对变化 `diff()`
- **异常值处理**：99%分位数裁剪
- **标准化**：除以因子中位数尺度

**示例**:
```python
turnover = screener.calculate_turnover_rate(
    factors["sma_20"],
    factor_name="sma_20",
    factor_type="trend"
)
```

---

##### 7. generate_screening_report

生成筛选报告并保存。

```python
def generate_screening_report(
    self,
    results: Dict[str, FactorMetrics],
    output_path: Optional[str] = None
) -> str
```

**参数**:
- `results`: 筛选结果字典
- `output_path`: 自定义输出路径（可选）

**返回**:
- `str`: 报告文件路径

**报告内容**:
1. 因子综合得分排序
2. 5维度评分详情
3. 统计显著性标记
4. 因子分层统计
5. 性能指标摘要

---

##### 8. get_top_factors

获取顶级因子列表。

```python
def get_top_factors(
    self,
    results: Dict[str, FactorMetrics],
    top_n: int = 10,
    min_score: float = 0.6,
    require_significant: bool = True
) -> List[FactorMetrics]
```

**参数**:
- `results`: 筛选结果字典
- `top_n`: 返回因子数量
- `min_score`: 最低综合得分阈值
- `require_significant`: 是否要求统计显著性

**返回**:
- `List[FactorMetrics]`: 排序后的顶级因子列表

**筛选标准**:
1. 综合得分 > `min_score`
2. 统计显著性（可选）
3. 按综合得分降序排序

---

### EnhancedResultManager

增强版结果管理器，基于时间戳文件夹的完整存储系统。

#### 初始化

```python
def __init__(self, base_output_dir: str = "./因子筛选") -> None
```

**参数**:
- `base_output_dir`: 输出根目录

---

#### 核心方法

##### create_screening_session

创建完整的筛选会话存储。

```python
def create_screening_session(
    self,
    symbol: str,
    timeframe: str,
    results: Dict[str, Any],
    screening_stats: Dict[str, Any],
    config: Any,
    data_quality_info: Optional[Dict[str, Any]] = None,
    existing_session_dir: Optional[Path] = None
) -> str
```

**功能**:
- 创建时间戳文件夹
- 保存核心筛选数据（CSV、JSON）
- 保存配置和元数据
- 生成分析报告（TXT、Markdown）
- 生成可视化图表（PNG）
- 保存因子相关性分析
- 更新会话索引

**返回**:
- `str`: 会话ID（文件夹名称）

---

##### get_session_history

获取会话历史记录。

```python
def get_session_history(
    self,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: int = 10
) -> List[ScreeningSession]
```

**参数**:
- `symbol`: 过滤股票代码（可选）
- `timeframe`: 过滤时间框架（可选）
- `limit`: 返回数量限制

**返回**:
- `List[ScreeningSession]`: 会话列表（按时间倒序）

---

### ScreeningConfig

筛选配置类。

#### 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ic_horizons` | `List[int]` | `[1, 3, 5, 10, 20]` | IC计算周期 |
| `alpha_level` | `float` | `0.05` | 显著性水平 |
| `min_sample_size` | `int` | `200` | 最小样本量 |
| `rolling_window` | `int` | `60` | 滚动窗口大小 |
| `fdr_method` | `str` | `"bh"` | FDR校正方法 |
| `vif_threshold` | `float` | `5.0` | VIF阈值 |
| `high_rank_threshold` | `float` | `0.8` | 高分位阈值 |

**加载配置**:
```python
from config_manager import load_config

config = load_config("path/to/config.yaml")
```

---

## 数据类

### FactorMetrics

因子综合指标数据类。

#### 字段说明

**预测能力指标**:
- `ic_1d`, `ic_3d`, `ic_5d`, `ic_10d`, `ic_20d`: 多周期IC值
- `ic_mean`: IC均值
- `ic_std`: IC标准差
- `ic_ir`: 信息比率（IC均值/IC标准差）
- `ic_decay_rate`: IC衰减率
- `predictive_score`: 预测能力综合得分

**稳定性指标**:
- `rolling_ic_mean`: 滚动IC均值
- `rolling_ic_std`: 滚动IC标准差
- `rolling_ic_stability`: 稳定性指标
- `stability_score`: 稳定性综合得分

**独立性指标**:
- `vif_score`: 方差膨胀因子
- `correlation_max`: 最大相关系数
- `information_increment`: 信息增量
- `independence_score`: 独立性综合得分

**实用性指标**:
- `turnover_rate`: 换手率
- `transaction_cost`: 交易成本
- `cost_efficiency`: 成本效率
- `practicality_score`: 实用性综合得分

**短周期适应性指标**:
- `reversal_effect`: 反转效应
- `momentum_persistence`: 动量持续性
- `volatility_sensitivity`: 波动敏感度
- `adaptability_score`: 适应性综合得分

**统计显著性**:
- `p_value`: 原始p值
- `corrected_p_value`: FDR校正后p值
- `is_significant`: 是否显著

**综合评分**:
- `comprehensive_score`: 综合得分（0~1）

---

### ScreeningSession

筛选会话信息。

#### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | `str` | 会话唯一ID |
| `timestamp` | `str` | 执行时间戳 |
| `symbol` | `str` | 股票代码 |
| `timeframe` | `str` | 时间框架 |
| `total_factors` | `int` | 总因子数 |
| `significant_factors` | `int` | 显著因子数 |
| `high_score_factors` | `int` | 高分因子数 |
| `total_time_seconds` | `float` | 执行耗时（秒） |
| `memory_used_mb` | `float` | 内存使用（MB） |
| `top_factor_name` | `str` | 顶级因子名称 |
| `top_factor_score` | `float` | 顶级因子得分 |

---

## 工具函数

### find_aligned_factor_files

查找对齐的因子文件。

```python
def find_aligned_factor_files(
    data_root: Path,
    symbol: str,
    timeframe: str
) -> List[Path]
```

---

### validate_factor_alignment

验证因子时间对齐。

```python
def validate_factor_alignment(
    factors: pd.DataFrame,
    returns: pd.Series
) -> Tuple[bool, str]
```

---

## 使用示例

### 基础示例

```python
from professional_factor_screener import ProfessionalFactorScreener

# 初始化
screener = ProfessionalFactorScreener()

# 执行筛选
results = screener.screen_factors_comprehensive(
    symbol="0700.HK",
    timeframe="60min"
)

# 生成报告
report_path = screener.generate_screening_report(results)

# 获取顶级因子
top_factors = screener.get_top_factors(results, top_n=10)
```

---

### 高级示例

```python
from config_manager import ScreeningConfig
from professional_factor_screener import ProfessionalFactorScreener
from enhanced_result_manager import EnhancedResultManager

# 自定义配置
config = ScreeningConfig(
    ic_horizons=[1, 3, 5],
    alpha_level=0.01,  # 更严格
    min_sample_size=300,
    vif_threshold=3.0  # 更独立
)

# 初始化
screener = ProfessionalFactorScreener(config=config)
result_manager = EnhancedResultManager(base_output_dir="./results")

# 执行筛选
results = screener.screen_factors_comprehensive(
    symbol="0700.HK",
    timeframe="60min"
)

# 保存完整会话
session_id = result_manager.create_screening_session(
    symbol="0700.HK",
    timeframe="60min",
    results=results,
    screening_stats=screener.screening_stats,
    config=config
)

print(f"会话已保存: {session_id}")
```

---

### 批量筛选示例

```python
symbols = ["0700.HK", "9988.HK", "0941.HK"]
timeframes = ["15min", "30min", "60min"]

all_results = {}

for symbol in symbols:
    for timeframe in timeframes:
        key = f"{symbol}_{timeframe}"
        try:
            results = screener.screen_factors_comprehensive(
                symbol=symbol,
                timeframe=timeframe
            )
            all_results[key] = results
            print(f"✅ {key}: {len(results)} factors")
        except Exception as e:
            print(f"❌ {key}: {e}")
```

---

## 性能指标

| 操作 | 数据量 | 性能 | 内存 |
|------|--------|------|------|
| IC计算 | 217因子 | 1.32秒 | <200MB |
| 滚动IC | 217因子 | 0.76秒 | <300MB |
| VIF计算 | 50因子 | <1秒 | <100MB |
| 完整筛选 | 217因子 | <5秒 | <500MB |

---

## 错误处理

### 常见异常

| 异常类型 | 原因 | 解决方案 |
|---------|------|----------|
| `FileNotFoundError` | 数据文件不存在 | 检查`data_root`路径 |
| `ValueError` | 样本量不足 | 降低`min_sample_size` |
| `TemporalValidationError` | 时间对齐失败 | 检查数据时间戳 |
| `MemoryError` | 内存不足 | 减少并行度或增加内存 |

---

## 版本历史

- **v2.0.0** (2025-10-03): 完整API文档
- **v1.0.0** (2025-09-29): 初始版本

---

## 相关文档

- [CONTRACT.md](CONTRACT.md) - 系统契约文档
- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md) - 依赖图谱
- [future_function_prevention_guide.md](future_function_prevention_guide.md) - 未来函数防护指南

---

**文档维护**: 量化首席工程师  
**最后更新**: 2025-10-03

