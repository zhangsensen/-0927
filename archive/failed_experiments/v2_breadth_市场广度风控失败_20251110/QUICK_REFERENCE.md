# 🎯 项目整理后快速查阅指南

## 📍 关键位置

| 内容 | 位置 | 说明 |
|------|------|------|
| **生产系统** | `real_backtest/` | ⭐ 核心：所有生产代码 |
| **回测框架** | `real_backtest/test_freq_no_lookahead.py` | 主回测脚本 |
| **Top500优化** | `real_backtest/top500_pos_grid_search.py` | 位置优化 |
| **核心引擎** | `real_backtest/core/` | 8个Python模块 |
| **配置文件** | `real_backtest/configs/` | 3个YAML配置 |
| **结果输出** | `results/` 和 `results_combo_wfo/` | 历史优化结果 |
| **项目说明** | `real_backtest/README.md` | 使用文档 |
| **整理报告** | `CLEANUP_REPORT.md` | 整理详情 |

---

## 🚀 常用命令

```bash
# 进入生产目录
cd real_backtest

# 基础回测
python test_freq_no_lookahead.py

# Top500优化
python top500_pos_grid_search.py

# 查看配置
cat configs/default.yaml
cat configs/combo_wfo_config.yaml

# 查看文档
cat README.md
```

---

## 📚 核心文件说明

### test_freq_no_lookahead.py
- **功能**: 无前向偏差的回测框架
- **优化**: 向量化streak计算 (9.41x加速)
- **输出**: 回测结果和性能指标

### top500_pos_grid_search.py
- **功能**: ETF位置优化
- **方法**: Grid Search
- **时间**: ~5分48秒 (42秒节省)

### core/combo_wfo_optimizer.py
- **功能**: 因子组合优化
- **方法**: Walk Forward
- **输出**: 最佳组合

### core/ic_calculator_numba.py
- **功能**: IC计算
- **优化**: Numba JIT编译
- **性能**: 高效计算

### core/precise_factor_library_v2.py
- **功能**: 因子库
- **包含**: 所有因子定义
- **更新**: 可扩展

---

## 🗂️ 配置文件

### default.yaml
基础配置：数据源、回测参数、因子参数

### combo_wfo_config.yaml
WFO配置：优化周期、窗口设置、选择约束

### FACTOR_SELECTION_CONSTRAINTS.yaml
因子约束：因子选择规则

---

## 📊 结果数据位置

```
results/
├── run_20251106_004018/    WFO优化结果1
├── run_20251106_004333/    WFO优化结果2
├── run_20251106_013228/    WFO优化结果3
└── run_20251106_021606/    WFO优化结果4

results_combo_wfo/
├── all_combos.csv          所有组合结果
├── top_combos.csv          Top组合
├── freq_test_no_lookahead.csv  频率测试
└── 20251106_*/             详细结果

```

每个run包含：
- `run_config.json`: 运行配置
- `wfo_summary.json`: 优化总结
- `factors/`: 因子数据 (parquet)
- `wfo_full.log`: 详细日志

---

## ✨ 性能指标

| 指标 | 数值 |
|------|------|
| 单操作加速 | 9.41x ⚡ |
| Top500时间 | 5分48秒 |
| 数据一致性 | 100% ✅ |
| 测试覆盖 | 9/9通过 ✅ |

---

## 🔄 工作流程

### 第1步: 数据加载
```python
from real_backtest.core.data_loader import DataLoader
loader = DataLoader(config)
data = loader.load_data()
```

### 第2步: IC计算
```python
from real_backtest.core.ic_calculator_numba import compute_spearman_ic_numba
ic = compute_spearman_ic_numba(factor_returns, benchmark_returns)
```

### 第3步: 回测
```python
python test_freq_no_lookahead.py
```

### 第4步: 优化
```python
python top500_pos_grid_search.py
```

---

## 🛠️ 常见操作

### 修改回测参数
编辑 `real_backtest/configs/default.yaml`

### 添加新因子
编辑 `real_backtest/core/precise_factor_library_v2.py`

### 修改约束条件
编辑 `real_backtest/configs/FACTOR_SELECTION_CONSTRAINTS.yaml`

### 查看历史结果
```bash
cd results
ls -la run_*/
cat run_20251106_021606/wfo_summary.json
```

---

## ❌ 已删除内容

为避免混淆，以下文件已删除：

- ❌ `test_all_freq_quick.py` (临时)
- ❌ `vectorization_demo.py` (演示)
- ❌ `vectorization_validation.py` (验证)
- ❌ `analysis_report.py` (分析)
- ❌ `*.md` 验证报告 (过期)
- ❌ `.regression_test.py` (测试)
- ❌ `quickstart.py` (启动)

所有功能现在统一通过 `real_backtest/` 目录访问。

---

## 🔍 故障排除

### 导入错误
```bash
cd real_backtest
python -c "from core.data_loader import DataLoader"
```

### 路径错误
确保在 `real_backtest/` 目录运行脚本

### 配置错误
检查 `configs/` 中的YAML文件

### 性能慢
- 检查数据量大小
- 查看CPU使用率
- 考虑使用并行处理

---

## 📞 技术支持

### 查看文档
```bash
cd real_backtest
cat README.md
```

### 查看日志
```bash
tail -f ../results_combo_wfo/top100_backtest.log
```

### 检查配置
```bash
cat configs/default.yaml | grep -E "^[a-z]"
```

---

## 🎯 下一步建议

1. ✅ 验证 `real_backtest/` 功能
2. ✅ 确认结果输出路径
3. ✅ 更新任何外部脚本引用
4. ✅ （可选）删除根目录旧文件

---

**版本**: 1.0  
**更新**: 2024年11月6日  
**状态**: ✅ 完成
