# ETF横截面因子筛选系统迁移指南

## 📋 概述

本指南将帮助您从硬编码的`run_etf_cross_section.py`迁移到基于配置的新系统`run_etf_cross_section_configurable.py`。

## 🎯 迁移优势

### 解决的问题
- **消除硬编码参数** - 所有魔法数字都可通过配置调整
- **路径管理统一** - 支持相对和绝对路径，避免硬编码
- **分析参数灵活** - IC周期、筛选标准、FDR控制等完全可配置
- **输出格式定制** - 文件命名、目录结构、报告内容可自定义
- **分层评级可调** - 核心/补充/研究因子阈值可灵活设置

### Linus工程原则
- **无特殊案例** - 单一配置结构，无if/else链
- **数据驱动** - 所有行为由配置数据控制
- **简洁实用** - 纯Python类，无魔法，无复杂度

## 🚀 快速迁移

### 1. 创建配置文件
```bash
# 生成默认配置文件
python run_etf_cross_section_configurable.py --create-config
```

### 2. 使用配置文件运行
```bash
# 使用YAML配置文件
python run_etf_cross_section_configurable.py --config sample_etf_config.yaml

# 使用预定义配置
python run_etf_cross_section_configurable.py --strict    # 严格标准
python run_etf_cross_section_configurable.py --relaxed   # 宽松标准
```

### 3. 命令行覆盖
```bash
# 临时覆盖配置文件中的路径
python run_etf_cross_section_configurable.py \
  --config sample_etf_config.yaml \
  --panel /path/to/new_panel.parquet \
  --price-dir /path/to/new_price_dir
```

## 📝 配置文件详解

### 关键配置项迁移映射

| 旧脚本硬编码 | 新配置项 | 默认值 | 说明 |
|-------------|---------|--------|------|
| `[1, 5, 10, 20]` | `analysis.ic_periods` | `[1, 5, 10, 20]` | IC分析周期 |
| `30` | `analysis.min_observations` | `30` | 最小观测值 |
| `5` | `analysis.min_ranking_samples` | `5` | 排名最小样本 |
| `20` | `analysis.min_ic_observations` | `20` | IC最小观测值 |
| `"spearman"` | `analysis.correlation_method` | `"spearman"` | 相关性方法 |
| `0.8` | `screening.max_correlation` | `0.7` | 最大相关性 |
| `0.1` | `screening.fdr_alpha` | `0.2` | FDR显著性 |
| 固定评级阈值 | `screening.tier_thresholds` | 可配置 | 分层评级 |

### 配置文件结构
```yaml
# 数据源配置
data_source:
  price_dir: "raw/ETF/daily"
  panel_file: "path/to/panel.parquet"
  file_pattern: "*.parquet"

# 分析参数
analysis:
  ic_periods: [1, 5, 10, 20]
  min_observations: 30
  correlation_method: "spearman"

# 筛选标准
screening:
  min_ic: 0.005
  min_ir: 0.05
  max_correlation: 0.7
  use_fdr: true
  fdr_alpha: 0.2
  tier_thresholds:
    core: {ic: 0.02, ir: 0.1}
    supplement: {ic: 0.01, ir: 0.07}

# 输出控制
output:
  output_dir: "results/screening"
  use_timestamp_subdir: true
  files:
    ic_analysis: "ic_analysis.csv"
    passed_factors: "passed_factors.csv"
```

## 🔧 高级配置示例

### 严格筛选配置
```yaml
screening:
  min_ic: 0.008          # 更严格的IC要求
  min_ir: 0.08           # 更严格的IR要求
  max_pvalue: 0.1        # 更严格的显著性
  fdr_alpha: 0.1         # 更严格的FDR
  max_correlation: 0.6   # 更严格的去重
```

### 宽松筛选配置
```yaml
screening:
  min_ic: 0.003          # 更宽松的IC要求
  min_ir: 0.03           # 更宽松的IR要求
  max_pvalue: 0.3        # 更宽松的显著性
  use_fdr: false         # 可选关闭FDR
  max_correlation: 0.8   # 更宽松的去重
```

### 长周期分析配置
```yaml
analysis:
  ic_periods: [1, 5, 10, 20, 40, 60]  # 添加长周期
  min_observations: 60                  # 提高最小样本要求
  stability_split_ratio: 0.4            # 调整稳定性分析
```

## 📊 输出文件对比

### 旧系统输出
```
etf_rotation_system/data/results/screening/screening_20251020_143022/
├── ic_analysis.csv
├── passed_factors.csv
└── screening_report.txt
```

### 新系统输出（可配置）
```yaml
output:
  use_timestamp_subdir: false           # 无时间戳子目录
  subdir_prefix: "factor_screening_"    # 自定义前缀
  files:
    ic_analysis: "full_ic_analysis.csv"
    passed_factors: "selected_factors.csv"
    screening_report: "factor_report.md"
```

## ⚙️ 编程接口使用

### 直接使用配置类
```python
from etf_cross_section_config import ETFCrossSectionConfig, ETF_STRICT_CONFIG
from run_etf_cross_section_configurable import ETFCrossSectionScreener

# 使用预定义配置
config = ETF_STRICT_CONFIG
screener = ETFCrossSectionScreener(config)
results = screener.run()

# 从文件加载配置
config = ETFCrossSectionConfig.from_yaml("my_config.yaml")
screener = ETFCrossSectionScreener(config)
results = screener.run()

# 程序化配置
from etf_cross_section_config import ETFCrossSectionConfig, DataSourceConfig
config = ETFCrossSectionConfig(
    data_source=DataSourceConfig(
        price_dir=Path("custom/price/dir"),
        panel_file=Path("custom/panel.parquet")
    ),
    screening=ScreeningConfig(min_ic=0.01)  # 自定义筛选标准
)
```

## 🛠️ 迁移检查清单

- [ ] 创建并验证配置文件
- [ ] 测试基础功能是否正常运行
- [ ] 验证输出文件格式和内容
- [ ] 检查筛选结果是否一致
- [ ] 调整配置参数以适应需求
- [ ] 更新相关脚本和文档
- [ ] 备份原始脚本

## 🐛 故障排除

### 常见问题

**Q: 配置文件加载失败**
```bash
# 检查YAML语法
python -c "import yaml; yaml.safe_load(open('sample_etf_config.yaml'))"
```

**Q: 路径不存在**
```python
# 在配置类中会自动验证
FileNotFoundError: 价格数据目录不存在: /path/to/price/dir
```

**Q: 筛选结果为空**
```yaml
# 降低筛选标准
screening:
  min_ic: 0.003
  min_ir: 0.03
  use_fdr: false
```

## 📈 性能优势

### 代码复用
- 配置类可在多个脚本间复用
- 预定义配置模板减少重复代码
- 单一配置文件管理所有参数

### 维护性提升
- 配置变更无需修改代码
- 参数集中管理，便于版本控制
- 配置验证确保参数有效性

### 扩展性
- 新增配置项无需修改核心逻辑
- 支持多种配置加载方式
- 配置类支持序列化和反序列化

## 🎯 迁移收益

| 方面 | 旧系统 | 新系统 |
|------|--------|--------|
| 参数灵活性 | ❌ 硬编码 | ✅ 完全可配置 |
| 路径管理 | ❌ 固定格式 | ✅ 灵活路径 |
| 筛选标准 | ❌ 固定阈值 | ✅ 可调标准 |
| 输出控制 | ❌ 固定格式 | ✅ 自定义输出 |
| 维护成本 | ❌ 代码修改 | ✅ 配置调整 |
| 扩展性 | ❌ 需要改代码 | ✅ 配置驱动 |