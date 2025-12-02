<!-- ALLOW-MD -->
# 模块导览（Module Map）

本文件列出核心模块与其职责，帮助快速定位代码位置与公共接口。

## 顶层脚本

- `run_combo_wfo.py`
  - 作用：WFO 启动脚本，串联配置加载、数据准备、因子标准化、组合搜索、结果持久化。
  - 依赖：`core/*`、`configs/combo_wfo_config.yaml`
  - 产出：`results/run_*` 下的全套文件（见 OUTPUT_SCHEMA）。

## 核心引擎（core/）

- `core/combo_wfo_optimizer.py`
  - 类：`ComboWFOOptimizer`
  - 关键接口：`run_combo_search(factors_data, returns, factor_names, top_n=100) -> (top_combos_list, all_combos_df)`
  - 说明：实现窗口生成、组合枚举、IC 计算、频率选择、FDR 校正与排序。

- `core/precise_factor_library_v2.py`
  - 类：`PreciseFactorLibrary`
  - 说明：计算全部精确因子，输出多列 DataFrame。

- `core/cross_section_processor.py`
  - 类：`CrossSectionProcessor`
  - 说明：横截面截尾（winsorize）与标准化（z-score），逐日处理。

- `core/ic_calculator_numba.py`
  - 函数：`compute_spearman_ic_numba(factors, returns)`
  - 说明：Numba 加速的 Spearman IC 计算（向量化/并行）。

- `core/data_loader.py`
  - 类：`DataLoader`
  - 说明：读取/缓存 `OHLCV` 数据，提供 `close.pct_change()` 作为收益。

- 其他：`data_contract.py`、`market_timing.py`、`wfo_realbt_calibrator.py`（如需深入可分别阅读源码注释）。

⚠️ **注意**：本项目仅使用真正的滚动 WFO 实现 (`combo_wfo_optimizer.py`)，已删除所有简化版本。

## 实盘风格与批处理（real_backtest/）

- `real_backtest/test_freq_no_lookahead.py`
  - 函数：`backtest_no_lookahead(...) -> dict`
  - 说明：严格无未来函数的单组合回测；包含预计算 IC 权重、向量化胜率与连胜/连败分析。

- `real_backtest/top500_pos_grid_search.py`
  - 说明：从 `all_freq_scan` 结果中提取 Top 500 收益参数，扫描持仓数 1–10，输出统计与报告。

- `real_backtest/configs/*.yaml`
  - 说明：与根 `configs/` 同步的回测侧配置（统一推荐使用根目录 `configs/`）。

## 结果与报告

- `results/`：每次 WFO 运行的产出目录（`run_*`）
- `results_combo_wfo/`：批量回测/扫描产出与对比报告

