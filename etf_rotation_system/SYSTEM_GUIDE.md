# ETF轮动系统 - 完整使用指南

> **配置驱动架构v2.0 - 从数据到策略的全方位指南**

---

## 🎯 系统概述

ETF轮动系统是一个专业级量化投资平台，采用配置驱动架构设计，实现了从原始ETF价格数据到可执行投资策略的全流程自动化。系统遵循严格的工程标准，确保代码质量、性能表现和结果可靠性。

### 核心价值主张
- **科学筛选**: 基于统计学的因子筛选框架
- **配置驱动**: 完全消除硬编码，所有参数可配置
- **高性能**: 向量化计算，优化的内存管理
- **可复现**: 严格时间对齐，无未来函数偏差
- **易维护**: 清晰的模块化设计，完善的异常处理

---

## 🏗️ 系统架构详解

### 核心流程图
```
┌─────────────────────────────────────────────────────────────────┐
│                    ETF轮动系统完整工作流                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  原始ETF价格数据 (43个ETF, 5.5年历史)                                │
│  ├─ 位置: /Users/zhangshenshen/深度量化0927/raw/ETF/daily/        │
│  └─ 格式: parquet, 包含trade_date, close列                      │
│                             ↓                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              模块1: 因子面板生成                          │    │
│  │                                                             │    │
│  │ • 输入: ETF价格数据                                        │    │
│  │ • 处理: 36个技术因子计算                                  │    │
│  │ • 方法: 向量化计算, 多线程处理                              │    │
│  │ • 输出: 因子面板数据 (56,575×36)                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                             ↓                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              模块2: 配置驱动因子筛选 ⭐                    │    │
│  │                                                             │    │
│  │ • 输入: 因子面板数据                                        │    │
│  │ • 配置: YAML文件驱动                                      │    │
│  │ • 分析: 多周期IC分析 (1,5,10,20日)                         │    │
│  │ • 筛选: 5维度专业框架                                    │    │
│  │ │   - 基础筛选 (IC, IR, p值, 覆盖率)                          │    │
│  │ │   - FDR校正 (Benjamini-Hochberg)                          │    │
│  │ │   - 相关性去重 (Spearman)                             │    │
│  │ │   - 分层评级 (核心/补充/研究)                            │    │
│  │ • 输出: 筛选结果和详细报告                                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                             ↓                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              模块3: 策略回测优化                          │    │
│  │                                                             │    │
│  │ • 引擎: VectorBT专业回测框架                              │    │
│  │ • 规模: 支持数千策略组合测试                            │    │
│  │ • 指标: Sharpe比率, 最大回撤, 年化收益等                        │    │
│  │ • 输出: 最优策略配置                                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速入门指南

### 环境准备
```bash
# 1. 确认Python环境
python3 --version  # 需要 Python 3.11+

# 2. 进入系统目录
cd /Users/zhangshenshen/深度量化0927/etf_rotation_system

# 3. 检查依赖包
pip list | grep -E "(pandas|numpy|scipy|yaml)"
```

### 依赖安装
```bash
# 安装核心依赖
pip install pandas>=2.0 numpy>=1.24 scipy>=1.10 PyYAML>=6.0

# 或者使用requirements.txt
pip install -r requirements.txt
```

### 数据准备检查
```bash
# 检查ETF数据目录
ls -la /Users/zhangshenshen/深度量化0927/raw/ETF/daily/ | head -5

# 验证数据文件
python -c "
import pandas as pd
from pathlib import Path

data_dir = Path('/Users/zhangshenshen/深度量化0927/raw/ETF/daily')
files = list(data_dir.glob('*.parquet'))
print(f'找到 {len(files)} 个ETF数据文件')
if files:
    sample = pd.read_parquet(files[0])
    print(f'示例数据: {files[0].name}')
    print(f'数据形状: {sample.shape}')
    print(f'日期范围: {sample.trade_date.min()} 到 {sample.trade_date.max()}')
"
```

---

## 📝 配置系统深度指南

### 配置文件创建
```bash
cd 02_因子筛选

# 生成默认配置模板
python run_etf_cross_section_configurable.py --create-config

# 查看生成的配置文件
ls -la etf_cross_section_config.yaml
```

### 配置文件结构详解
```yaml
# ===== 数据源配置 =====
data_source:
  price_dir: "/Users/zhangshenshen/深度量化0927/raw/ETF/daily"
  panel_file: "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels/panel_20251020_162504/panel.parquet"
  price_columns: ["trade_date", "close"]     # 读取的列名
  file_pattern: "*.parquet"                    # 文件匹配模式
  symbol_extract_method: "stem_split"        # Symbol提取方法

# ===== 分析参数配置 =====
analysis:
  ic_periods: [1, 5, 10, 20]                # IC分析周期（天）
  min_observations: 30                         # 最小观测值数量
  min_ranking_samples: 5                       # 横截面排名最小样本
  min_ic_observations: 20                        # IC计算最小观测值
  correlation_method: "spearman"                # 相关性计算方法
  correlation_min_periods: 30                 # 相关性计算最小周期
  epsilon_small: 1e-8                           # 小值防止除零
  stability_split_ratio: 0.5                   # 稳定性分析分割比例

# ===== 筛选标准配置 =====
screening:
  # 基础筛选阈值
  min_ic: 0.005                                 # 最小IC阈值 (0.5%)
  min_ir: 0.05                                  # 最小IR阈值
  max_pvalue: 0.2                               # 最大p值
  min_coverage: 0.7                              # 最小覆盖率

  # 去重标准
  max_correlation: 0.7                           # 最大因子间相关性

  # FDR校正设置
  use_fdr: true                                 # 是否启用FDR校正
  fdr_alpha: 0.2                                # FDR显著性水平

  # 分层评级阈值
  tier_thresholds:
    core:                                       # 🟢 核心因子
      ic: 0.02                                 # IC阈值 2%
      ir: 0.1                                  # IR阈值 0.1
    supplement:                                  # 🟡 补充因子
      ic: 0.01                                 # IC阈值 1%
      ir: 0.07                                 # IR阈值 0.07
    research:                                    # 🔵 研究因子
      ic: 0.0                                  # IC阈值 0%
      ir: 0.0                                  # IR阈值 0%

  # 分层评级标签
  tier_labels:
    core: "🟢 核心"
    supplement: "🟡 补充"
    research: "🔵 研究"

# ===== 输出配置 =====
output:
  output_dir: "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/screening"
  use_timestamp_subdir: true                      # 使用时间戳子目录
  timestamp_format: "%Y%m%d_%H%M%S"           # 时间戳格式
  subdir_prefix: "screening_"                   # 子目录前缀

  # 输出文件命名
  files:
    ic_analysis: "ic_analysis.csv"            # IC分析结果
    passed_factors: "passed_factors.csv"         # 通过筛选的因子
    screening_report: "screening_report.txt"     # 筛选报告

  # 报告内容选项
  include_factor_details: true                  # 包含因子详情
  include_summary_statistics: true             # 包含汇总统计
  encoding: "utf-8"                              # 文件编码

# ===== 系统级控制 =====
debug_mode: false                                 # 调试模式开关
progress_reporting: true                           # 进度报告开关
```

### 配置验证和使用
```bash
# 验证配置文件语法
python -c "
import yaml
with open('sample_etf_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
print('✅ YAML语法正确')
print(f'配置项数量: {len(config)}')
"

# 验证配置路径
python -c "
from etf_cross_section_config import ETFCrossSectionConfig
from pathlib import Path

try:
    config = ETFCrossSectionConfig.from_yaml('sample_etf_config.yaml')
    print('✅ 配置加载成功')
    print(f'价格目录: {config.data_source.price_dir}')
    print(f'面板文件: {config.data_source.panel_file}')
except Exception as e:
    print(f'❌ 配置错误: {e}')
"
```

---

## 🔍 因子筛选科学原理

### Information Coefficient (IC) 分析
IC是衡量因子预测能力的核心指标，计算因子值与未来收益的相关性。

#### IC计算公式
```python
# 单周期IC计算
IC_t = corr(factor_t, return_{t+1})

# 多周期IC分析
def calculate_multi_period_ic(factor_data, price_data, periods=[1, 5, 10, 20]):
    """
    计算多周期IC，完全向量化实现

    参数:
    - factor_data: 因子面板数据
    - price_data: 价格数据
    - periods: 分析周期列表

    返回:
    - IC分析结果DataFrame
    """
    # 预计算各周期未来收益
    fwd_returns = {}
    for period in periods:
        fwd_returns[period] = price_data.groupby('symbol')['close'].pct_change(period).shift(-period)

    results = []
    # 对每个因子进行IC计算
    for factor_name in factor_data.columns:
        factor_values = factor_data[factor_name].dropna()
        period_ics = {}
        all_ics = []

        for period in periods:
            fwd_ret = fwd_returns[period]

            # 对齐数据
            common_idx = factor_values.index.intersection(fwd_ret.index)
            f = factor_values.loc[common_idx]
            r = fwd_ret.loc[common_idx].dropna()

            final_idx = f.index.intersection(r.index)
            if len(final_idx) < min_observations:
                continue

            # 向量化IC计算
            factor_pivot = f.loc[final_idx].unstack(level='symbol')
            return_pivot = r.loc[final_idx].unstack(level='symbol')

            # 批量排名计算
            from scipy.stats import rankdata

            factor_ranked = np.apply_along_axis(rank_row, 1, factor_pivot.values)
            return_ranked = np.apply_along_axis(rank_row, 1, return_pivot.values)

            # 计算IC
            ics = []
            for i in range(len(factor_pivot)):
                f_valid = factor_ranked[i]
                r_valid = return_ranked[i]
                mask = ~(np.isnan(f_valid) | np.isnan(r_valid))
                if mask.sum() >= min_ranking_samples:
                    f_clean = f_valid[mask]
                    r_clean = r_valid[mask]
                    if f_clean.std() > 0 and r_clean.std() > 0:
                        ic = np.corrcoef(f_clean, r_clean)[0, 1]
                        if not np.isnan(ic):
                            ics.append(ic)

            if len(ics) >= min_ic_observations:
                period_ics[f'ic_{period}d'] = np.mean(ics)
                period_ics[f'ir_{period}d'] = np.mean(ics) / (np.std(ics) + epsilon_small)
                all_ics.extend(ics)

        # 综合指标计算
        if period_ics and all_ics:
            ic_mean = np.mean(all_ics)
            ic_std = np.std(all_ics)
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0

            # 稳定性分析
            half = len(all_ics) // 2
            stability = np.corrcoef(
                all_ics[:half],
                all_ices[half:2*half]
            )[0, 1] if half > 10 else 0

            # t检验
            t_stat, p_value = stats.ttest_1samp(all_ics, 0)

            result = {
                'factor': factor_name,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ic_ir': ic_ir,
                'ic_positive_rate': np.mean(np.array(all_ics) > 0),
                'stability': stability,
                't_stat': t_stat,
                'p_value': p_value,
                'sample_size': len(all_ics),
                'coverage': len(factor_values) / len(factor_data)
            }
            result.update(period_ics)
            results.append(result)

    return pd.DataFrame(results)
```

### FDR (False Discovery Rate) 校正
FDR校正用于控制多重检验中的假阳性率，提高统计结果的可靠性。

#### Benjamini-Hochberg算法
```python
def apply_fdr_correction(ic_df, alpha=0.2):
    """
    Benjamini-Hochberg FDR校正

    参数:
    - ic_df: IC分析结果DataFrame
    - alpha: 显著性水平

    返回:
    - 通过FDR校正的结果DataFrame
    """
    p_values = ic_df['p_value'].values
    n = len(p_values)

    # 排序p值
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # 计算BH临界值
    critical = np.arange(1, n + 1) * alpha / n

    # 找到最大的满足条件的索引
    rejected = sorted_p <= critical
    if rejected.any():
        max_idx = np.where(rejected)[0].max()
        passed_idx = sorted_idx[:max_idx + 1]
        return ic_df.iloc[passed_idx].copy()

    return pd.DataFrame()  # 无因子通过校正
```

### 相关性去重算法
```python
def remove_correlated_factors(ic_df, panel, max_corr=0.7):
    """
    去除高相关因子，保留IC_IR更高的因子

    参数:
    - ic_df: IC分析结果
    - panel: 因子面板数据
    - max_corr: 最大相关性阈值

    返回:
    - 去重后的IC分析结果
    """
    if len(ic_df) <= 1:
        return ic_df

    factors = ic_df['factor'].tolist()
    factor_data = panel[factors]

    # 计算相关性矩阵
    corr_matrix = factor_data.corr(
        method='spearman',
        min_periods=30
    ).abs()

    # 贪心去重算法
    to_remove = set()
    for i, f1 in enumerate(factors):
        if f1 in to_remove:
            continue
        for f2 in factors[i+1:]:
            if f2 in to_remove:
                continue
            if corr_matrix.loc[f1, f2] > max_corr:
                # 保留IC_IR更高的因子
                ir1 = ic_df[ic_df['factor'] == f1]['ic_ir'].values[0]
                ir2 = ic_df[ic_df['factor'] == f2]['ic_ir'].values[0]
                to_remove.add(f2 if abs(ir1) > abs(ir2) else f1)

    return ic_df[~ic_df['factor'].isin(to_remove)].copy()
```

### 分层评级体系
```python
def classify_factor(ic_mean, ic_ir, tier_thresholds):
    """
    因子分层评级

    参数:
    - ic_mean: IC均值
    - ic_ir: IC_IR
    - tier_thresholds: 分层阈值配置

    返回:
    - 因子评级标签
    """
    thresholds = tier_thresholds

    if abs(ic_mean) >= thresholds["core"]["ic"] and abs(ic_ir) >= thresholds["core"]["ir"]:
        return "🟢 核心"
    elif abs(ic_mean) >= thresholds["supplement"]["ic"] and abs(ic_ir) >= thresholds["supplement"]["ir"]:
        return "🟡 补充"
    else:
        return "🔵 研究"
```

---

## 🚀 实用操作指南

### 基础筛选操作
```bash
# 1. 使用示例配置文件
cd /Users/zhangshenshen/深度量化0927/etf_rotation_system/02_因子筛选
python run_etf_cross_section_configurable.py --config sample_etf_config.yaml

# 2. 使用预设配置模式
python run_etf_cross_section_configurable.py --standard   # 标准模式
python run_etf_cross_section_configurable.py --strict     # 严格模式
python run_etf_cross_section_configurable.py --relaxed    # 宽松模式

# 3. 创建自定义配置文件
python run_etf_cross_section_configurable.py --create-config
```

### 高级配置操作
```yaml
# 自定义配置示例
# custom_config.yaml

# 扩展IC分析周期
analysis:
  ic_periods: [1, 3, 5, 10, 20, 40, 60]    # 添加长周期分析
  min_observations: 60                        # 提高最小样本要求

# 严格筛选标准
screening:
  min_ic: 0.008                              # 提高IC要求
  min_ir: 0.08                               # 提高IR要求
  max_pvalue: 0.1                            # 更严格的显著性
  max_correlation: 0.6                         # 更严格去重

# 自定义分层阈值
tier_thresholds:
  core: {ic: 0.025, ir: 0.12}              # 更严格的核心标准
  supplement: {ic: 0.015, ir: 0.08}           # 更严格的补充标准

# 定制输出格式
output:
  use_timestamp_subdir: false                  # 不使用时间戳子目录
  subdir_prefix: "factor_analysis_"            # 自定义前缀
  files:
    ic_analysis: "comprehensive_ic_analysis.csv"
    passed_factors: "selected_factors.csv"
    screening_report: "detailed_factor_report.md"
```

### 命令行参数覆盖
```bash
# 临时覆盖配置文件中的路径
python run_etf_cross_section_configurable.py \
  --config sample_etf_config.yaml \
  --panel /path/to/new_panel.parquet \
  --price-dir /path/to/new_price_dir \
  --output-dir /path/to/custom_output

# 组合使用预设配置和命令行参数
python run_etf_cross_section_configurable.py \
  --strict \
  --output-dir /path/to/production_results
```

---

## 📊 结果解读和分析

### 输出文件结构
```
data/results/screening/screening_YYYYMMDD_HHMMSS/
├── ic_analysis.csv              # 完整IC分析结果
├── passed_factors.csv           # 通过筛选的因子列表
└── screening_report.txt         # 详细文字报告
```

### IC分析结果解读
```csv
factor,ic_mean,ic_std,ic_ir,ic_positive_rate,stability,t_stat,p_value,sample_size,coverage
PRICE_POSITION_60D,0.042041,0.323614,0.129941,0.567446,0.153,5.049e-22,5624,1.0
MOM_ACCEL,-0.044429,0.349357,-0.127210,0.436731,0.184,1.264e-17,2868,0.81
VOLATILITY_120D,-0.037438,0.402612,-0.092874,0.471182,0.251,5.040e-12,2855,1.0
```

**关键指标说明**:
- `ic_mean`: 因子预测能力，正值表示正向预测
- `ic_ir`: IC稳定性，数值越大越稳定
- `ic_positive_rate`: IC正值比例，反映因子方向一致性
- `stability`: 时间序列稳定性，>0.2为稳定
- `p_value`: 统计显著性，<0.05为显著
- `coverage`: 数据覆盖率，>0.7为良好

### 筛选报告解读
```
ETF横截面因子筛选报告
==================================================
筛选时间: 2025-10-20 18:49:13

筛选标准:
  IC均值 >= 0.005 (0.5%)
  IC_IR >= 0.05
  p-value <= 0.2
  覆盖率 >= 0.7
  最大相关性 = 0.7
  FDR校正 = 启用

筛选结果:
  总因子数: 36
  通过筛选: 8
  通过率: 22.2%

🏆 因子评级详情:
  🟢 核心 PRICE_POSITION_60D   IC=+0.0420 IR=+0.1299 p=5.05e-22
  🟢 核心 MOM_ACCEL          IC=-0.0444 IR=-0.1272 p=1.26e-17
  🟡 补充 VOLATILITY_120D   IC=-0.0374 IR=-0.0929 p=5.04e-12
  🟡 补充 VOL_VOLATILITY_20   IC=+0.0166 IR=+0.0831 p=6.17e-10
```

### 性能指标分析
```python
# 系统性能监控
import time
import psutil

start_time = time.time()
process = psutil.Process()
start_memory = process.memory_info().rss

# 运行筛选
screener = ETFCrossSectionScreener(config)
results = screener.run()

end_time = time.time()
end_memory = process.memory_info().rss

performance_metrics = {
    'execution_time': end_time - start_time,
    'memory_used': (end_memory - start_memory) / 1024 / 1024,  # MB
    'throughput': results['total_factors'] / (end_time - start_time),
    'pass_rate': results['passed_factors'] / results['total_factors']
}

print(f"执行时间: {performance_metrics['execution_time']:.2f}秒")
print(f"内存使用: {performance_metrics['memory_used']:.1f}MB")
print(f"处理速度: {performance_metrics['throughput']:.1f}因子/秒")
print(f"通过率: {performance_metrics['pass_rate']:.1%}")
```

---

## 🛠️ 故障排除指南

### 常见问题诊断

#### 1. 配置文件错误
```bash
# 检查YAML语法
python -c "import yaml; yaml.safe_load(open('sample_etf_config.yaml'))"

# 错误示例和解决方案
错误: yaml.scanner.ScannerError
原因: 缩进错误或特殊字符问题
解决: 检查缩进，确保使用空格而非Tab
```

#### 2. 路径不存在
```bash
# 检查数据路径
ls -la /Users/zhangshenshen/深度量化0927/raw/ETF/daily/
ls -la /Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels/

# 错误示例和解决方案
错误: FileNotFoundError: 价格数据目录不存在
原因: 配置文件中路径错误
解决: 更新配置文件中的路径为正确路径
```

#### 3. 内存不足
```bash
# 监控内存使用
python -c "
import psutil
process = psutil.Process()
print(f'当前内存使用: {process.memory_info().rss / 1024 / 1024:.1f}MB')

# 解决方案:
# 1. 减少IC分析周期
# 2. 使用分块处理大数据集
# 3. 增加系统内存或使用更强大的机器
```

#### 4. 筛选结果为空
```bash
# 检查筛选标准设置
python -c "
config = ETFCrossSectionConfig.from_yaml('sample_etf_config.yaml')
print(f'IC阈值: {config.screening.min_ic}')
print(f'IR阈值: {config.screening.min_ir}')
print(f'FDR启用: {config.screening.use_fdr}')
"

# 解决方案:
# 1. 降低筛选标准
# 2. 禁用FDR校正
# 3. 检查数据质量和覆盖范围
```

### 性能优化建议

#### 1. 数据处理优化
```python
# 使用分块处理大数据集
def optimize_data_processing(panel, chunk_size=1000):
    """分块处理大数据集，控制内存使用"""
    symbols = panel.index.get_level_values('symbol').unique()
    results = []

    for i in range(0, len(symbols), chunk_size):
        chunk_symbols = symbols[i:i+chunk_size]
        chunk_data = panel.loc[chunk_symbols]
        # 处理数据块
        chunk_result = analyze_chunk(chunk_data)
        results.append(chunk_result)
        del chunk_data  # 及时释放内存

    return pd.concat(results, ignore_index=True)
```

#### 2. 计算效率优化
```python
# 缓存IC计算结果
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_ic_calculation(panel_hash, factor_name, period):
    """缓存IC计算结果，避免重复计算"""
    return calculate_ic_internal(panel_hash, factor_name, period)
```

#### 3. 并行处理
```python
# 多进程处理（实验性功能）
import multiprocessing as mp

def parallel_factor_analysis(factors, n_processes=4):
    """并行因子分析"""
    with mp.Pool(n_processes) as pool:
        results = pool.map(analyze_single_factor, factors)
    return results
```

---

## 📚 扩展和定制

### 添加新因子
```python
# 在因子面板生成模块中添加新因子
def calculate_custom_factor(close, volume, high_period=60, low_period=20):
    """
    自定义因子计算函数
    注意：必须使用shift()避免未来函数
    """
    high = close.rolling(high_period).max()
    low = close.rolling(low_period).min()
    return (close - low) / (high - low + 1e-8)

# 在generate_panel.py中集成
for symbol, group in data.groupby('symbol'):
    # 现有因子计算...

    # 添加自定义因子
    factors['CUSTOM_POSITION'] = calculate_custom_factor(close, volume)
```

### 扩展筛选标准
```python
# 在配置类中添加新的筛选维度
@dataclass
class EnhancedScreeningConfig(ScreeningConfig):
    # 新增筛选维度
    min_effect_size: float = 0.1      # 最小效应量
    max_turnover: float = 0.5          # 最大换手率
    stability_threshold: float = 0.3    # 稳定性阈值

    # 高级筛选选项
    enable_bootstrap_test: bool = False   # 启用bootstrap测试
    enable_cross_validation: bool = False # 启用交叉验证
    sector_neutral: bool = False           # 行业中性化
```

### 自定义输出格式
```python
# 扩展输出配置类
@dataclass
class EnhancedOutputConfig(OutputConfig):
    # 新增输出选项
    export_json: bool = True           # 导出JSON格式结果
    export_excel: bool = True          # 导出Excel格式报告
    generate_charts: bool = True        # 生成可视化图表
    create_dashboard: bool = True      # 创建交互式面板
```

---

## 🎯 最佳实践建议

### 配置管理最佳实践
1. **版本控制**: 所有配置文件纳入Git版本控制
2. **环境分离**: 为不同环境创建专门的配置文件
3. **参数验证**: 使用内置验证功能检查参数合理性
4. **文档同步**: 配置变更时同步更新相关文档

### 数据管理最佳实践
1. **质量检查**: 定期验证数据完整性和准确性
2. **备份策略**: 建立数据备份和恢复机制
3. **更新监控**: 监控数据更新频率和质量
4. **存储优化**: 使用高效的数据格式和压缩

### 运行维护最佳实践
1. **定期监控**: 监控系统性能和资源使用
2. **结果验证**: 验证筛选结果的合理性和稳定性
3. **性能调优**: 根据硬件资源优化配置参数
4. **日志记录**: 详细记录运行日志和错误信息

### 研究和开发最佳实践
1. **参数调优**: 使用系统化方法测试不同参数组合
2. **结果分析**: 深度分析因子表现和市场适应性
3. **回测验证**: 进行样本外测试验证策略有效性
4. **文档完善**: 详细记录实验过程和发现

---

## 📞 技术支持

### 系统状态检查
```bash
# 完整的系统状态检查
cd /Users/zhangshenshen/深度量化0927/etf_rotation_system/02_因子筛选

python -c "
# 1. 配置检查
from etf_cross_section_config import ETFCrossSectionConfig
try:
    config = ETFCrossSectionConfig.from_yaml('sample_etf_config.yaml')
    print('✅ 配置加载成功')
except Exception as e:
    print(f'❌ 配置错误: {e}')

# 2. 数据检查
import pandas as pd
from pathlib import Path
try:
    panel_file = Path('/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels/panel_20251020_162504/panel.parquet')
    if panel_file.exists():
        panel = pd.read_parquet(panel_file)
        print(f'✅ 面板数据: {panel.shape}')
        print(f'ETF数量: {panel.index.get_level_values(\"symbol\").nunique()}')
        print(f'因子数量: {len(panel.columns)}')
    else:
        print('❌ 面板文件不存在')
except Exception as e:
    print(f'❌ 数据检查错误: {e}')

# 3. 依赖检查
import import pandas, numpy, scipy, yaml
    print('✅ 核心依赖完整')
    print(f'pandas: {pd.__version__}')
    print(f'numpy: {np.__version__}')
    print(f'scipy: {scipy.__version__}')
"
```

### 问题报告模板
```markdown
## 问题报告

**系统信息**:
- Python版本:
- 系统版本: v2.0
- 运行时间:
- 错误类型:

**问题描述**:
- 详细描述遇到的问题

**重现步骤**:
1.
2.
3.

**错误信息**:
- 完整错误堆栈信息
- 相关日志输出

**已尝试的解决方案**:
1.
2.
3.

**环境信息**:
- 操作系统:
- 内存使用:
- 磁盘空间:
- 相关配置:

**期望结果**:
- 描述期望的正确行为

**附件**:
- 相关日志文件
- 配置文件
- 错误截图
```

---

**文档版本**: v2.0
**最后更新**: 2025-10-20
**维护团队**: ETF轮动系统开发团队
**技术支持**: 通过GitHub Issues提交问题报告