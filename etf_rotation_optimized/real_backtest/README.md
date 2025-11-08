# 真实回测系统

## 📋 目录结构

```
real_backtest/
├── configs/                    # 配置文件
│   ├── default.yaml           # 默认配置
│   ├── combo_wfo_config.yaml  # Combo优化配置
│   └── FACTOR_SELECTION_CONSTRAINTS.yaml
├── scripts/                    # 脚本
│   └── cleanup.sh
├── results/                    # 输出结果
│   ├── logs/
│   └── run_*/
├── output/                     # 临时输出
├── run_production_backtest.py  # 生产级回测脚本（主）
├── run_position_grid_search.py # 持仓数网格搜索
└── README.md                    # 本文档
```

## 🚀 快速开始

### 1. 基础频率扫描
```bash
python test_freq_no_lookahead.py
```

### 2. Top 500 仓位优化
```bash
python top500_pos_grid_search.py
```

### 3. Combo因子优化
```bash
python -m core.combo_wfo_optimizer
```

## 📊 关键文件说明

### run_production_backtest.py

- **功能**: 无前向偏差的生产级回测框架
- **优化**: 向量化计算与并行化IC权重预计算
- **特性**: 支持多频率/持仓数扫描、成本明细追踪

### run_position_grid_search.py

- **功能**: 持仓数网格优化
- **特性**: 基于Top 500参数进行网格搜索
- **性能**: 针对性优化，大幅降低计算量

## 🔧 配置说明

### default.yaml

基础配置，包含：

- 数据源路径
- 回测参数
- 因子参数

### combo_wfo_config.yaml
Combo优化配置，包含：
- 优化周期
- 窗口设置
- 因子选择

## 📈 输出结果

### results/logs/
- pipeline.log: 主日志

### results/run_*/
- run_config.json: 运行配置
- wfo_full.log: 详细日志
- factors/: 因子数据 (parquet格式)
- wfo_summary.json: 优化总结

## ✨ 数据一致性

✅ 所有代码已验证：
- 9/9 功能测试通过
- 100% 数据一致性
- 浮点精度 < 1e-10
- 无前向偏差

## 📝 性能指标

| 操作 | 时间 | 加速 |
|------|------|------|
| 单次回测 | 0.0141ms | 9.41x ⚡ |
| Top 500优化 | 5分48秒 | 42秒节省 |

## 🔍 故障排除

### 数据加载错误
检查 configs/ 中的数据路径

### 内存不足
减少回测周期或使用分批处理

### 导入错误
确保 core/ 目录完整

## 📚 相关文档

- 更详细的配置: 查看各配置文件注释
- 因子库: core/precise_factor_library_v2.py
- IC计算: core/ic_calculator_numba.py

---

**版本**: 1.0  
**最后更新**: 2024年11月  
**状态**: 生产就绪 ✅
