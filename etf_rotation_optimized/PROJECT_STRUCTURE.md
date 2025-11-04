# ETF轮动系统 - 项目结构

**版本**: 生产稳定版  
**日期**: 2025-10-29  
**状态**: ✅ 生产就绪

---

## 核心目录

### `core/` - 核心模块
```
core/
├── data_manager.py                    # 数据加载与管理
├── factor_calculator.py               # 因子计算引擎
├── cross_section_processor.py         # 横截面处理
├── factor_screener.py                 # 因子筛选
├── direct_factor_wfo_optimizer.py     # WFO优化器（IC加权）
├── ic_calculator_numba.py             # IC计算（Numba加速）
├── constrained_walk_forward_optimizer.py  # 约束WFO
├── pipeline.py                        # 流程编排
└── ...
```

### `configs/` - 配置文件
```
configs/
├── default.yaml                       # 默认配置（生产）
├── FACTOR_SELECTION_CONSTRAINTS.yaml  # 因子筛选约束
├── etf_pools.yaml                     # ETF池配置
└── experiments/                       # 实验配置
    ├── exp_baseline.yaml
    ├── exp_new_factors.yaml
    └── ...
```

### `research/` - 研究代码
```
research/
└── prior_weighting_experiment/        # 先验加权实验
    ├── README.md                      # 实验说明
    ├── scripts/                       # 验证脚本
    ├── configs/                       # 先验配置
    └── reports/                       # 验证报告
```

### `results/` - 运行结果
```
results/
├── cross_section/20251029/            # 横截面数据
├── factor_selection/20251029/         # 因子筛选结果
├── wfo/20251029/                      # WFO结果
│   └── 20251029_201318/               # 最新运行
│       └── wfo_summary.csv
└── logs/                              # 日志文件
```

---

## 核心文件

### 配置
- `pyproject.toml` - 项目配置
- `uv.lock` - 依赖锁定
- `Makefile` - 构建工具

### 文档
- `README.md` - 项目说明
- `PROJECT_STRUCTURE.md` - 本文档
- `PRODUCTION_CLEANUP_SUMMARY.md` - 清理总结
- `FINAL_EXECUTION_REPORT.md` - 执行报告

### 入口
- `main.py` - CLI入口

---

## 运行流程

### 1. 完整流程
```bash
python main.py run --config configs/default.yaml
```

### 2. 单步运行
```bash
# 横截面处理
python main.py cross-section --config configs/default.yaml

# 因子筛选
python main.py factor-selection --config configs/default.yaml

# WFO验证
python main.py wfo --config configs/default.yaml
```

---

## 配置说明

### 加权方案
- `ic_weighted` - IC加权（生产默认）✅
- `equal` - 等权
- `contribution_weighted` - 贡献加权（实验性）

### 关键参数
```yaml
wfo:
  factor_weighting: "ic_weighted"  # 锁定IC加权
  min_factor_ic: 0.012             # 最小IC门槛
  is_period: 252                   # IS窗口（交易日）
  oos_period: 60                   # OOS窗口（交易日）
  step_size: 20                    # 滑动步长（交易日）
```

---

## 性能指标

### 当前生产性能
```
平均OOS IC:    0.0160
OOS IC胜率:    75.0%
基准IC:        0.0085
超额IC:        +0.0075 (+88.0% vs基准)
总窗口数:      36
```

---

## 开发规范

### 代码质量
- 遵循Linus哲学（无冗余代码）
- Black格式化（88字符）
- 向量化率≥95%
- 函数<50行，缩进≤3层

### 测试
```bash
make test    # 运行测试
make lint    # 代码检查
make format  # 代码格式化
```

---

## 依赖

### 核心依赖
- Python 3.11+
- NumPy 2.3+
- Pandas 2.3+
- VectorBT 0.28+
- TA-Lib 0.6.7+
- SciPy 1.16+
- scikit-learn 1.7+

### 安装
```bash
make install  # 或 uv sync
```

---

## 注意事项

### 生产环境
- ✅ 使用 `ic_weighted` 加权方案
- ✅ 定期更新因子池
- ✅ 监控OOS IC和胜率
- ✅ 保持代码简洁

### 研究环境
- 🔬 实验代码隔离在 `research/`
- 🔬 不影响生产代码
- 🔬 独立验证和测试

---

**维护**: AI Agent (Linus Mode)  
**更新**: 2025-10-29
