<!-- ALLOW-MD -->
# 输出文件结构与字段（Output Schema）

本文档定义了 `results/run_*` 与 `results_combo_wfo/` 的产出结构与字段，属于跨模块契约，不得随意更改。

## results/run_YYYYMMDD_HHMMSS/

- `run_config.json`
  - 说明：记录本次运行使用的配置（含参数快照）

- `all_combos.parquet`
  - 含义：全部（组合 × 频率）的统计结果，按 `mean_oos_ic` 降序排列
  - 必含字段：
    - `combo`: str，形如 `ADX_14D + PRICE_POSITION_20D`
    - `combo_size`: int，因子个数
    - `best_rebalance_freq`: int，最优频率（天）
    - `mean_oos_ic`: float，OOS 平均 IC
    - `stability_score`: float，稳健性评分
    - `rank`: int，排序名次（1 为最佳）

- `top_combos.parquet`
  - 含义：Top N 组合（通常为 100），与 `all_combos` 字段一致

- `top100_by_ic.parquet`
  - 含义：按 IC/稳定性排序的 Top100，用于下游回测脚本的默认读取
  - 字段：同 `top_combos.parquet`

- `wfo_summary.json`
  - 示例字段（可能随实现演进略有不同，但需稳定）：
    - `total_combos`: int，总测试组合数
    - `significant_combos`: int，通过 FDR 的组合数
    - `mean_ic`: float，所有组合 OOS IC 的均值
    - `best_combo`: { `combo`: str, `ic`: float, `best_rebalance_freq`: int }

- `factor_selection_summary.json`
  - 说明：因子覆盖率、入选频率等统计，用于分析与报告

- `factors/` 目录
  - 说明：标准化后的单因子截面序列（逐因子 `.parquet`）

## results_combo_wfo/

- `top100_backtest_by_ic_*.csv`
  - 含义：对 Top100（按 IC 排序）的实盘风格回测结果（单频率）
  - 关键字段：
    - `combo`, `wfo_freq`, `test_freq`, `position_size`
    - 绩效：`annual_ret`, `sharpe`, `max_dd`, `win_rate`, `profit_factor`, `calmar_ratio`, `sortino_ratio`

- `all_freq_scan_*.csv`（若存在）
  - 含义：对多个频率/持仓数的扫描结果汇总

- `top500_pos_scan_*.csv`（由 `top500_pos_grid_search.py` 生成）
  - 含义：Top500 收益参数 × 持仓数的网格搜索结果

## 兼容性注意事项

- 上述字段名、文件名是读取脚本默认假设；若需调整，必须：
  1) 先更新本文件；
  2) 同步修改所有读取相关脚本（例如 `real_backtest/test_freq_no_lookahead.py`）；
  3) 进行回归验证与结果对比；
  4) 在 PR 描述中明确说明兼容性变化。
