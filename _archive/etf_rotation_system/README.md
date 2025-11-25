# ETF轮动量化交易系统

> **一个完整的ETF量化投资系统，从因子生成到策略回测的全流程解决方案**

## 🎯 系统概述

ETF轮动系统是一个基于现代Python架构的专业量化投资平台，采用模块化设计和配置驱动的架构理念。系统从原始ETF价格数据开始，经过因子计算、因子筛选、策略回测，最终生成可执行的投资策略配置。

### 核心特性
- ✅ **配置驱动架构**: 完全基于YAML配置，无硬编码参数
- ✅ **因子筛选科学性**: 5维度专业筛选框架，FDR统计校正
- ✅ **性能优异**: 向量化计算，43个ETF 36因子 < 20秒
- ✅ **代码质量**: Linus工程风格，类型安全，异常处理完善
- ✅ **数据完整性**: 5.5年历史数据，56,575条记录

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ETF轮动系统架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原始数据 (raw/ETF/daily/)                                   │
│  ├─ 43个ETF价格数据 (parquet格式)                             │
│  └─ 5.5年历史数据，56,575条记录                               │
│                             ↓                                │
│  【01_横截面建设】因子面板生成                                   │
│  ├─ generate_panel.py                                        │
│  └─ 36个技术因子，向量化计算                                    │
│                             ↓                                │
│  因子面板 (data/results/panels/)                               │
│  ├─ panel_YYYYMMDD_HHMMSS.parquet                            │
│  └─ symbol×date索引，36个因子值                                │
│                             ↓                                │
│  【02_因子筛选】配置驱动筛选系统                                 │
│  ├─ run_etf_cross_section_configurable.py                  │
│  ├─ etf_cross_section_config.py                            │
│  └─ YAML配置，FDR校正，相关性去重                               │
│                             ↓                                │
│  筛选结果 (data/results/screening/)                            │
│  ├─ screening_YYYYMMDD_HHMMSS/                             │
│  ├─ ic_analysis.csv (完整IC分析)                               │
│  ├─ passed_factors.csv (通过筛选因子)                          │
│  └─ screening_report.txt (详细报告)                           │
│                             ↓                                │
│  【03_vbt回测】策略回测与优化                                    │
│  └─ VectorBT大规模回测引擎                                     │
│                             ↓                                │
│  最优策略配置                                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求
- Python 3.11+
- pandas ≥ 2.0
- numpy ≥ 1.24
- scipy ≥ 1.10
- PyYAML ≥ 6.0

### 安装依赖
```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_system
pip install -r requirements.txt
```

### 一键运行（推荐）
```bash
# 使用配置文件运行因子筛选
cd 02_因子筛选
python run_etf_cross_section_configurable.py --config sample_etf_config.yaml

# 使用预设配置
python run_etf_cross_section_configurable.py --standard  # 标准模式
python run_etf_cross_section_configurable.py --strict    # 严格模式
python run_etf_cross_section_configurable.py --relaxed   # 宽松模式
```

### 创建自定义配置
```bash
# 生成默认配置模板
cd 02_因子筛选
python run_etf_cross_section_configurable.py --create-config

# 编辑配置文件
vim etf_cross_section_config.yaml

# 使用自定义配置
python run_etf_cross_section_configurable.py --config etf_cross_section_config.yaml
```

---

## 📁 项目结构

```
etf_rotation_system/
├── 📁 01_横截面建设/                    # 因子面板生成模块
│   ├── 📄 generate_panel.py           # 因子计算核心逻辑
│   └── 📁 docs/                        # 技术文档
├── 📁 02_因子筛选/                     # 🌟 核心筛选模块
│   ├── 📄 run_etf_cross_section_configurable.py  # 主筛选脚本(推荐)
│   ├── 📄 run_etf_cross_section.py              # 硬编码版本(弃用)
│   ├── 📄 etf_cross_section_config.py           # 配置类定义
│   ├── 📄 sample_etf_config.yaml                # 示例配置文件
│   └── 📄 MIGRATION_GUIDE.md                     # 迁移指南
├── 📁 03_vbt回测/                      # VectorBT回测模块
│   └── 📄 backtest_engine.py
├── 📁 data/                            # 统一数据目录
│   ├── 📁 results/panels/              # 因子面板数据
│   └── 📁 results/screening/           # 筛选结果数据
├── 📄 PROJECT_README.md                # 项目详细说明
├── 📄 QUICKREF.md                      # 快速参考指南
├── 📄 SYSTEM_GUIDE.md                  # 系统使用指南
└── 📄 requirements.txt                 # Python依赖
```

---

## 📊 关键概念

### IC (Information Coefficient)
```python
IC = corr(factor_values, future_returns)
```
- **含义**: 因子预测能力指标，范围[-1, 1]
- **标准**: |IC| > 0.02 为优秀因子

### IR (Information Ratio)
```python
IR = mean(IC) / std(IC)
```
- **含义**: IC稳定性指标
- **标准**: |IR| > 0.1 为稳定因子

### FDR (False Discovery Rate)
- **含义**: 控制多重检验假阳性率
- **方法**: Benjamini-Hochberg校正
- **标准**: FDR < 0.05 为统计显著

### 因子分层评级
- 🟢 **核心因子**: IC≥0.02, IR≥0.1
- 🟡 **补充因子**: IC≥0.01, IR≥0.07
- 🔵 **研究因子**: 其他通过筛选的因子

---

## 🎛️ 配置系统详解

### 配置文件结构
```yaml
# 数据源配置
data_source:
  price_dir: "/path/to/ETF/daily"
  panel_file: "/path/to/panel.parquet"
  file_pattern: "*.parquet"

# 分析参数配置
analysis:
  ic_periods: [1, 5, 10, 20]        # IC分析周期
  min_observations: 30               # 最小观测值
  correlation_method: "spearman"     # 相关性计算方法

# 筛选标准配置
screening:
  min_ic: 0.005                      # 最小IC阈值 (0.5%)
  min_ir: 0.05                       # 最小IR阈值
  max_pvalue: 0.2                    # 最大p值
  use_fdr: true                      # 启用FDR校正
  max_correlation: 0.7               # 最大因子相关性

# 输出配置
output:
  output_dir: "/path/to/results"
  use_timestamp_subdir: true          # 使用时间戳子目录
  include_factor_details: true       # 包含因子详情
```

### 预设配置对比
| 配置模式 | IC阈值 | IR阈值 | FDR | 通过率 | 适用场景 |
|---------|-------|-------|-----|--------|----------|
| 严格模式 | 0.8% | 0.08 | 启用 | ~11% | 生产环境 |
| 标准模式 | 0.5% | 0.05 | 启用 | ~22% | 日常使用 |
| 宽松模式 | 0.3% | 0.03 | 禁用 | ~28% | 研究探索 |

---

## 📈 性能指标

### 系统性能
- **执行时间**: ~19秒 (36因子, 43ETF)
- **内存使用**: ~200MB
- **吞吐量**: 1.9因子/秒
- **数据覆盖**: 5.5年历史数据

### 筛选效果 (基于真实数据)
- **输入因子**: 36个技术因子
- **通过筛选**: 8个因子 (22.2%通过率)
- **核心因子**: 6个 (包括PRICE_POSITION_60D, MOM_ACCEL等)
- **相关性控制**: 0.7阈值去重，有效降低冗余

---

## 🛠️ 高级用法

### 自定义筛选标准
```yaml
screening:
  min_ic: 0.008                      # 提高IC要求
  max_correlation: 0.6               # 严格去重
  tier_thresholds:                   # 自定义分层阈值
    core: {ic: 0.025, ir: 0.12}
    supplement: {ic: 0.015, ir: 0.08}
```

### 扩展IC分析周期
```yaml
analysis:
  ic_periods: [1, 5, 10, 20, 40, 60]  # 添加长周期分析
  min_observations: 60                 # 提高最小样本要求
```

### 输出定制化
```yaml
output:
  use_timestamp_subdir: false          # 不使用时间戳子目录
  subdir_prefix: "factor_analysis_"    # 自定义前缀
  files:
    ic_analysis: "full_ic_report.csv"
    passed_factors: "selected_factors.csv"
```

---

## ⚠️ 注意事项

### 数据要求
- **格式**: parquet格式，包含trade_date, close列
- **时间**: 建议至少2年历史数据
- **质量**: 无缺失值，时间序列连续

### 使用建议
- **优先使用**: `run_etf_cross_section_configurable.py`
- **弃用**: 硬编码版本 `run_etf_cross_section.py`
- **配置管理**: 通过YAML文件调整参数，避免修改代码
- **结果验证**: 检查筛选报告和IC分析结果

### 常见问题
1. **路径错误**: 确保配置文件中的数据路径正确
2. **内存不足**: 可减少IC分析周期或降低ETF数量
3. **筛选过严**: 调整screening配置中的阈值参数
4. **依赖缺失**: 运行 `pip install -r requirements.txt`

---

## 📚 相关文档

- **[PROJECT_README.md](PROJECT_README.md)**: 详细项目架构说明
- **[QUICKREF.md](QUICKREF.md)**: 快速参考和常用命令
- **[SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)**: 完整系统使用指南
- **[MIGRATION_GUIDE.md](02_因子筛选/MIGRATION_GUIDE.md)**: 从硬编码版本迁移指南

---

## 📞 技术支持

如遇问题，请检查：
1. Python版本 ≥ 3.11
2. 数据文件存在且格式正确
3. 配置文件路径设置正确
4. 依赖包完整安装

**系统版本**: v2.0 (配置驱动架构)
**最后更新**: 2025-10-20
**兼容性**: Python 3.11+, macOS/Linux