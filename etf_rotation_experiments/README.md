# ETF 轮动实验配置

> ⚠️ **注意**: 此目录仅保留配置文件作为参考。核心代码已移至 `etf_rotation_optimized/`。

## 目录结构

```
etf_rotation_experiments/
├── configs/              # 配置文件（含 ML 排序等高级配置）
│   ├── combo_wfo_config.yaml         # 完整 WFO 配置（含 ML 排序说明）
│   ├── combo_wfo_config_no_ml.yaml   # 无 ML 配置
│   ├── combo_wfo_config_compound.yaml
│   ├── combo_wfo_lagtest.yaml
│   ├── ranking_datasets.yaml
│   └── archive/                      # 历史配置存档
└── README.md
```

## 使用说明

### 主开发目录
所有策略开发和回测请使用：
```bash
cd etf_rotation_optimized
python run_unified_wfo.py
```

### 配置参考
如需参考 ML 排序等高级配置，可查看：
- `configs/combo_wfo_config.yaml` - 包含 ML 排序模式详细说明

### 独立验证
Backtrader 验证请使用：
```bash
python run_audit.py --input results/top5000_summary.csv
```

---

## 项目架构

```
-0927/
├── etf_rotation_optimized/   # 🎯 主开发目录（向量化回测）
├── strategy_auditor/         # 🔍 独立验证（Backtrader 事件驱动）
├── etf_rotation_experiments/ # 📁 配置参考（本目录）
├── configs/                  # 生产配置
└── scripts/                  # 工具脚本
```
