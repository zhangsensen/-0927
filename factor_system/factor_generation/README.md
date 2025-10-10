# 因子生成系统 (Factor Generation System)

本目录包含因子生成的核心组件，用于从原始价格数据计算技术指标和统计因子。

## 核心组件

### 主要脚本
- `run_single_stock.py` - 单支股票因子生成入口
- `run_batch_processing.py` - 批量处理多支股票的入口
- `run_complete_pipeline.py` - 完整因子生成流水线
- `batch_factor_processor.py` - 批处理核心逻辑
- `integrated_resampler.py` - 时间框架重采样模块
- `data_validator.py` - 数据验证工具
- `verify_consistency.py` - 因子一致性验证工具

### 配置管理
- `config.py` - 配置加载器
- `config_loader.py` - 高级配置管理
- `config.yaml` - 主配置文件
- `configs/` - 配置文件目录
  - `config_us.yaml` - 美股市场配置（备用）

### 文档
- `FACTOR_GENERATION.md` - 详细开发文档
- `IMPLEMENTATION_DECISIONS.md` - 实现决策记录
- `FACTOR_GENERATION_TECHNICAL_MEMORY.md` - 技术备忘录

## 目录结构

```
factor_generation/
├── configs/                    # 配置文件
│   └── config_us.yaml          # 美股市场配置（备用）
├── scripts/                    # 辅助脚本
│   ├── debug/                  # 调试工具
│   │   ├── check_factors.py
│   │   └── debug_timeframes.py
│   └── legacy/                 # 旧版本/遗留代码
│       ├── enhanced_factor_calculator.py
│       └── multi_tf_vbt_detector.py
├── batch_factor_processor.py   # 批处理核心
├── config.py                   # 配置加载
├── config_loader.py
├── config.yaml
├── data_validator.py           # 数据验证
├── integrated_resampler.py     # 时间框架处理
├── run_batch_processing.py     # 批量处理入口
├── run_complete_pipeline.py    # 完整流水线
├── run_single_stock.py         # 单股票处理
├── verify_consistency.py       # 一致性验证
└── README.md                  # 本文档
```

## 快速开始

### 单支股票因子生成
```bash
python run_single_stock.py <股票代码> [--timeframe 1min] [--start 2023-01-01] [--end 2023-12-31]

# 示例：生成0700.HK的1分钟因子
python run_single_stock.py 0700.HK --timeframe 1min
```

### 批量生成
```bash
# 处理股票列表中的所有股票
python run_batch_processing.py --config config.yaml
```

### 验证因子一致性
```bash
# 验证生成的因子与引擎注册列表是否一致
python verify_consistency.py
```

## 相关文档

- [FACTOR_GENERATION.md](FACTOR_GENERATION.md) - 详细开发文档
- [IMPLEMENTATION_DECISIONS.md](IMPLEMENTATION_DECISIONS.md) - 实现决策记录
- [FACTOR_GENERATION_TECHNICAL_MEMORY.md](FACTOR_GENERATION_TECHNICAL_MEMORY.md) - 技术备忘录

## 依赖管理

所有依赖通过 uv 管理（参见 ../uv.lock）
