# 因子筛选模块

**状态**: 📋 因子筛选和评估核心模块

## 📌 模块职责

该模块负责：
1. **多周期 IC/IR 计算** - 评估因子预测能力
2. **相关性分析** - 因子之间的相关性检查  
3. **统计显著性检验** - Newey-West 标准误、FDR 校验
4. **因子排名和筛选** - 基于 IC/IR 选择优质因子
5. **结果可视化** - 生成筛选报告

## 🎯 关键脚本

### 主脚本

- **`run_etf_cross_section_configurable.py`** (638 行)
  - 完整的因子筛选流程
  - 配置驱动的参数设置
  - 支持多个ETF的并行计算
  - 输出评估报告

- **`run_etf_cross_section.py`** (377 行)
  - 简化的筛选脚本（备选）
  - 用于快速测试

### 配置文件

- **`optimized_screening_config.yaml`** - 筛选参数配置
  - IC 计算周期
  - 显著性阈值
  - 并行计算参数

### 工具模块

- **`etf_cross_section_config.py`** - 配置管理（旧版，待迁移）

## 🚀 使用方法

### 快速开始

```bash
# 运行完整筛选流程
python3 run_etf_cross_section_configurable.py

# 指定配置文件
python3 run_etf_cross_section_configurable.py --config optimized_screening_config.yaml

# 指定输出目录
python3 run_etf_cross_section_configurable.py --output /path/to/output
```

### 配置参数

编辑 `optimized_screening_config.yaml`：

```yaml
# IC 计算
forward_periods: [5, 10, 20]     # 5天、10天、20天的前向收益
min_ic_threshold: 0.05           # 最小 IC 阈值
min_ir_threshold: 0.5            # 最小 IR 阈值

# 统计检验
use_fdr_correction: true         # FDR 多重检验校验
fdr_alpha: 0.05                  # FDR 显著性水平

# Newey-West 标准误
use_newey_west: true             # 校验自相关性
nw_lags: 5                       # Newey-West 滞后数

# 并行计算
n_jobs: -1                       # -1 表示使用所有 CPU 核心
```

### 输出结果

筛选完成后生成：
- `ic_statistics.csv` - IC/IR 统计
- `factor_correlation.csv` - 因子相关性矩阵
- `significant_factors.csv` - 显著因子列表
- `screening_report.html` - 可视化报告

## 📊 核心算法

### IC 计算

对每个因子和前向周期：
```
IC = Pearson相关系数(因子值, 前向收益)
```

### IR 计算

```
IR = mean(IC) / std(IC)
```

### Newey-West 标准误

校验时间序列自相关性的影响

### FDR 校验

控制多重假设检验的假发现率

## 🔗 与其他模块的关系

- **依赖**: 面板生成模块 (`01_横截面建设`)
  - 需要先生成因子面板
  
- **被依赖**: 回测模块 (`03_vbt回测`)
  - 筛选出的因子用于回测
  
- **配置**: ConfigManager (`config/config_manager.py`)
  - 统一配置管理（待集成）

## 🔧 技术栈

- **计算**: NumPy, Pandas
- **统计**: SciPy, StatsModels (Newey-West)
- **并行**: Joblib
- **可视化**: Matplotlib, Seaborn

## 📋 数据流

```
因子面板
    ↓
IC/IR 计算
    ↓
统计检验 (Newey-West, FDR)
    ↓
因子排名
    ↓
结果输出和报告
```

## ⚙️ 配置迁移 (进行中)

**状态**: ConfigManager 已创建，待迁移完成

迁移完成后，所有配置将统一由 `etf_rotation_system/config/config_manager.py` 管理。

### 迁移步骤

1. ✅ ConfigManager 已创建
2. ⏳ 迁移 `run_etf_cross_section_configurable.py` 导入
3. ⏳ 删除旧配置文件
4. ⏳ 更新文档

## 🐛 已知问题

- [ ] 文件过大 (638 行) - 计划拆分为 `ic_calculator.py` 和 `screener.py`
- [ ] 配置分散 - 正在迁移到 ConfigManager
- [ ] 文档不完整 - 正在补充

## 📞 支持

遇到问题？

1. 检查配置文件是否正确
2. 查看输出日志信息
3. 参考 `examples/` 目录的示例
4. 查阅 CLAUDE.md 或其他 README

---

**最后更新**: 2025-10-22  
**维护者**: Linus 量化工程师  
**状态**: 🟡 正在优化中
