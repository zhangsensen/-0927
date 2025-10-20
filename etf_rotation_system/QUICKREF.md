# ETF轮动系统 - 快速参考指南

> **配置驱动架构 v2.0 - 从数据到策略的完整解决方案**

---

## 🚀 一键启动

### 基础使用
```bash
# 进入系统目录
cd /Users/zhangshenshen/深度量化0927/etf_rotation_system/02_因子筛选

# 使用示例配置文件
python run_etf_cross_section_configurable.py --config sample_etf_config.yaml

# 使用预设配置模式
python run_etf_cross_section_configurable.py --standard   # 标准模式 (推荐)
python run_etf_cross_section_configurable.py --strict     # 严格模式
python run_etf_cross_section_configurable.py --relaxed    # 宽松模式
```

### 创建自定义配置
```bash
# 生成默认配置模板
python run_etf_cross_section_configurable.py --create-config

# 编辑配置文件
vim etf_cross_section_config.yaml

# 运行自定义配置
python run_etf_cross_section_configurable.py --config etf_cross_section_config.yaml
```

---

## 📁 文件路径速查

### 输入文件
```
raw/ETF/daily/
├── 510300.SH_daily_20200102_20251014.parquet
├── 159919.SZ_daily_20200102_20251014.parquet
└── ... (43个ETF价格文件)
```

### 输出文件
```
data/results/screening/screening_YYYYMMDD_HHMMSS/
├── ic_analysis.csv          # 完整IC分析结果
├── passed_factors.csv       # 通过筛选的因子列表
└── screening_report.txt     # 详细筛选报告
```

### 配置文件
```
02_因子筛选/
├── sample_etf_config.yaml           # 示例配置文件
├── etf_cross_section_config.yaml   # 默认配置模板 (需生成)
└── etf_cross_section_config.py     # 配置类定义
```

---

## 🎛️ 核心配置参数

### 数据源配置
```yaml
data_source:
  price_dir: "/Users/zhangshenshen/深度量化0927/raw/ETF/daily"
  panel_file: "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels/panel_20251020_162504/panel.parquet"
  price_columns: ["trade_date", "close"]
  file_pattern: "*.parquet"
```

### 筛选标准配置
```yaml
screening:
  min_ic: 0.005              # 最小IC阈值 (0.5%)
  min_ir: 0.05               # 最小IR阈值
  max_pvalue: 0.2            # 最大p值
  min_coverage: 0.7          # 最小覆盖率
  max_correlation: 0.7       # 最大因子间相关性
  use_fdr: true              # 启用FDR校正
  fdr_alpha: 0.2             # FDR显著性水平
```

### 分析参数配置
```yaml
analysis:
  ic_periods: [1, 5, 10, 20]           # IC分析周期
  min_observations: 30                  # 最小观测值数量
  min_ranking_samples: 5                # 横截面排名最小样本
  correlation_method: "spearman"         # 相关性计算方法
  epsilon_small: 1e-8                   # 小值防止除零
```

---

## 📊 关键指标含义

| 指标 | 含义 | 计算公式 | 优秀标准 |
|------|------|----------|----------|
| **IC均值** | 因子预测能力 | `mean(corr(factor, return))` | \|IC\| > 0.02 |
| **IC_IR** | IC稳定性 | `mean(IC) / std(IC)` | \|IR\| > 0.1 |
| **IC正率** | IC正值比例 | `mean(IC > 0)` | > 0.55 |
| **稳定性** | 时间序列稳定性 | `corr(IC[:half], IC[half:])` | > 0.2 |
| **p值** | 统计显著性 | t检验p值 | < 0.05 |
| **覆盖率** | 数据完整度 | `valid_values / total_values` | > 0.7 |

---

## 🏆 当前最优因子 (基于真实数据)

### 核心因子 (🟢)
| 因子名 | IC均值 | IC_IR | IC正率 | 特点 |
|--------|--------|-------|---------|------|
| **PRICE_POSITION_60D** | +0.0420 | +0.1299 | 56.7% | 60日价格位置，强预测力 |
| **MOM_ACCEL** | -0.0444 | -0.1272 | 43.7% | 动量加速因子，反向指标 |

### 补充因子 (🟡)
| 因子名 | IC均值 | IC_IR | 特点 |
|--------|--------|-------|------|
| VOLATILITY_120D | -0.0374 | -0.0929 | 120日波动率因子 |
| VOL_VOLATILITY_20 | +0.0166 | +0.0831 | 成交量波动率因子 |
| VOLUME_PRICE_TREND | -0.0162 | -0.0783 | 量价趋势因子 |
| RSI_6 | +0.0240 | +0.0770 | 短期RSI因子 |

---

## ⚙️ 配置模式对比

| 模式 | IC阈值 | IR阈值 | FDR | 通过率 | 执行时间 | 适用场景 |
|------|--------|-------|-----|--------|----------|----------|
| **严格** | 0.8% | 0.08 | 启用 | 11.1% (4/36) | 18.6s | 生产环境 |
| **标准** | 0.5% | 0.05 | 启用 | 22.2% (8/36) | 18.8s | 日常使用 |
| **宽松** | 0.3% | 0.03 | 禁用 | 27.8% (10/36) | 19.7s | 研究探索 |

---

## 🛠️ 常用配置调整

### 提高筛选标准 (保守策略)
```yaml
screening:
  min_ic: 0.008              # 提高到0.8%
  min_ir: 0.08               # 提高到0.08
  max_pvalue: 0.1            # 更严格的显著性
  max_correlation: 0.6       # 更严格去重
```

### 降低筛选标准 (激进策略)
```yaml
screening:
  min_ic: 0.003              # 降低到0.3%
  min_ir: 0.03               # 降低到0.03
  use_fdr: false             # 禁用FDR校正
  max_correlation: 0.8       # 放宽相关性要求
```

### 扩展分析周期
```yaml
analysis:
  ic_periods: [1, 3, 5, 10, 20, 40, 60]    # 添加更多周期
  min_observations: 60                        # 提高最小样本要求
  stability_split_ratio: 0.4                  # 调整稳定性分析
```

---

## 📈 性能基准

### 系统性能
```
数据规模: 43个ETF × 36个因子 × 5.5年数据
执行时间: ~19秒
内存使用: ~200MB
吞吐量: 1.9因子/秒
```

### 筛选效果
```
输入因子: 36个技术因子
通过筛选: 8个因子 (22.2%通过率)
分层结果: 6个核心因子 + 2个补充因子
相关性去重: 0.7阈值，有效降低冗余
```

---

## 🐛 故障排除

### 常见错误及解决方案

#### 1. 路径错误
```bash
错误: FileNotFoundError: 价格数据目录不存在
解决: 检查配置文件中的路径是否正确
命令: ls -la /Users/zhangshenshen/深度量化0927/raw/ETF/daily/
```

#### 2. 配置文件错误
```bash
错误: yaml.scanner.ScannerError
解决: 检查YAML语法，特别注意缩进
命令: python -c "import yaml; yaml.safe_load(open('sample_etf_config.yaml'))"
```

#### 3. 筛选结果为空
```bash
错误: 无因子通过筛选
解决: 降低筛选标准
修改: screening.min_ic = 0.003  # 从0.005降低
```

#### 4. 内存不足
```bash
错误: MemoryError
解决: 减少IC分析周期
修改: analysis.ic_periods = [1, 5, 10]  # 去掉20周期
```

#### 5. 依赖缺失
```bash
错误: ModuleNotFoundError
解决: 安装依赖包
命令: pip install pandas numpy scipy pyyaml
```

---

## 🔍 验证命令

### 验证数据完整性
```bash
python -c "
import pandas as pd
from pathlib import Path

panel_file = Path('/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels/panel_20251020_162504/panel.parquet')
panel = pd.read_parquet(panel_file)

print(f'ETF数量: {panel.index.get_level_values(\"symbol\").nunique()}')
print(f'日期范围: {panel.index.get_level_values(\"date\").min()} 到 {panel.index.get_level_values(\"date\").max()}')
print(f'因子数量: {len(panel.columns)}')
print(f'数据完整性: {(1 - panel.isna().sum().sum() / panel.size) * 100:.1f}%')
"
```

### 验证配置正确性
```bash
cd /Users/zhangshenshen/深度量化0927/etf_rotation_system/02_因子筛选
python -c "
from etf_cross_section_config import ETFCrossSectionConfig
config = ETFCrossSectionConfig.from_yaml('sample_etf_config.yaml')
print(f'配置加载成功: IC阈值={config.screening.min_ic}, 周期={config.analysis.ic_periods}')
"
```

### 验证输出结果
```bash
# 检查最新筛选结果
ls -la /Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/screening/screening_*/

# 查看通过筛选的因子
head -5 /Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/screening/screening_*/passed_factors.csv

# 查看筛选报告
tail -20 /Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/screening/screening_*/screening_report.txt
```

---

## 🎯 最佳实践

### 日常使用流程
1. **数据更新**: 确保ETF价格数据最新
2. **配置选择**: 根据需求选择预设配置
3. **运行筛选**: 执行配置驱动筛选
4. **结果验证**: 检查筛选报告和IC分析
5. **策略应用**: 使用筛选结果构建策略

### 参数调优建议
1. **从标准模式开始**，逐步调整参数
2. **观察IC变化**，确保因子质量稳定
3. **控制通过率**，避免因子过多或过少
4. **记录配置变更**，便于结果复现

### 性能优化
1. **合理设置IC周期**，避免过长周期影响性能
2. **使用FDR校正**，提高统计可靠性
3. **定期清理结果目录**，避免占用过多磁盘空间

---

## 📚 相关文档

- **[README.md](README.md)**: 系统概述和完整介绍
- **[PROJECT_README.md](PROJECT_README.md)**: 详细项目架构说明
- **[SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)**: 系统深度使用指南
- **[MIGRATION_GUIDE.md](02_因子筛选/MIGRATION_GUIDE.md)**: 从硬编码版本迁移

---

**系统版本**: v2.0 (配置驱动架构)
**最后更新**: 2025-10-20
**兼容性**: Python 3.11+, macOS/Linux