<!-- ALLOW-MD -->
# ETF Rotation Optimized — 项目总览

本项目实现了基于横截面因子组合的 Walk-Forward Optimization（WFO）与严格无未来函数的实盘风格回测。文档旨在帮助后续开发者（尤其是基于大模型的自动化代理）安全、准确地理解与修改项目，避免误操作。

## 目录结构

```text
etf_rotation_optimized/
├── run_combo_wfo.py                    # WFO 启动脚本（根目录唯一）
├── configs/                            # 配置文件（统一 YAML）
├── core/                               # 引擎核心（算法实现）
├── real_backtest/                      # 实盘风格回测与实验脚本（规范化入口）
├── results/                            # WFO 结果（run_* 目录）
├── results_combo_wfo/                  # 实盘回测批处理/扫描结果
├── scripts/                            # 运行批处理与清理脚本
└── docs/                               # 项目文档（你正在阅读）
```

重要约定：

- `real_backtest/` 下的脚本是“规范化入口”，与根目录可能存在同名文件。以 `real_backtest/` 为准；根目录同名脚本仅作兼容/包装（建议后续清理成 wrapper）。
- `run_combo_wfo.py` 仅在根目录维护一个副本（WFO 主入口）。

## 功能概览

- 因子库：`core/precise_factor_library_v2.py`
- 横截面处理：`core/cross_section_processor.py`（Winsorize + Z-score）
- 相关性计算：`core/ic_calculator_numba.py`（Numba 加速的 Spearman IC）
- 组合 WFO：`core/combo_wfo_optimizer.py`（2–5 因子组合，FDR 校正）
- 启动脚本：`run_combo_wfo.py`（生成 `results/run_*` 全套产物）
- 实盘回测：`real_backtest/test_freq_no_lookahead.py`（严格无未来函数）
- 参数扫描：`real_backtest/top500_pos_grid_search.py`（Top500 收益参数持仓数网格）

## 数据流与流程

1. 数据加载

- 从 `configs/combo_wfo_config.yaml` 读取数据源目录、标的列表、起止时间
- `core/data_loader.py` 加载 `OHLCV` 数据，得到 `close` 价格矩阵

1. 因子计算与标准化

- `precise_factor_library_v2.PreciseFactorLibrary.compute_all_factors()` 生成多因子列
- `cross_section_processor.CrossSectionProcessor.process_all_factors()` 进行截尾与标准化（逐日横截面）
- 拼接为 `factors_data: (T, N, F)`；收益 `returns: (T, N)`

1. 组合 WFO（`run_combo_wfo.py` → `core/combo_wfo_optimizer.ComboWFOOptimizer`）

- 组合规模：2–5 因子；频率候选：1–30 天（可配）
- 滚动窗口：IS=252 天、OOS=60 天、步长=60 天（可配）
- 目标：在 OOS 上选出 `mean_oos_ic` 最大且稳定性高的组合与频率
- 统计显著性：FDR（Benjamini-Hochberg）校正
- 产物：`results/run_YYYYMMDD_HHMMSS/`（详见 docs/OUTPUT_SCHEMA.md）

1. 实盘无未来函数回测（`real_backtest/test_freq_no_lookahead.py`）

- 每个调仓日仅使用“之前的历史”估计 IC 权重（向量化+并行预计算）
- 基于当日因子值计算信号，按 `position_size` 选取持仓，计算收益
- 输出多维绩效指标（年化、Sharpe、最大回撤、胜率、连胜/连败、Calmar、Sortino 等）

## 关键接口（契约）

- `ComboWFOOptimizer.run_combo_search(factors_data, returns, factor_names, top_n=100) -> (top_combos_list, all_combos_df)`
  - 输入：`factors_data(T,N,F)`、`returns(T,N)`、因子名列表、TopN
  - 输出：
    - `top_combos_list: List[dict]`（包含 combo、best_rebalance_freq、mean_oos_ic、stability_score、rank 等）
    - `all_combos_df: pd.DataFrame`（全部组合/频率统计，按 IC 已排序）

- `backtest_no_lookahead(factors_data, returns, etf_names, rebalance_freq, lookback_window=252, position_size=4, transaction_cost=0.0003, initial_capital=1_000_000.0) -> dict`
  - 仅使用历史数据估计权重；返回包含 `annual_ret / sharpe / max_dd / nav / daily_returns` 等指标

接口稳定性承诺：以上返回结构是下游脚本与结果可复现的基础，严禁在未同步更新文档与下游的情况下更改字段名与含义。

## 配置（YAML）

统一使用 `configs/combo_wfo_config.yaml`，关键段落：

- `data`: `data_dir`、`cache_dir`、`symbols`、`start_date`、`end_date`
- `cross_section`: `winsorize_lower`、`winsorize_upper`
- `combo_wfo`: `top_n`、IS/OOS/step、频率候选、FDR 参数
- `backtest`: `lookback_window`、`position_size`、`transaction_cost`、并行度等

## 如何运行

- WFO 优化（生成 run_* 目录）：

  ```bash
  python run_combo_wfo.py
  ```

- 使用 Top100 组合做无未来函数回测：

  ```bash
  python real_backtest/test_freq_no_lookahead.py
  ```

- Top500 收益参数的持仓数网格搜索：

  ```bash
  python real_backtest/top500_pos_grid_search.py
  ```

详见 `docs/RUNBOOK.md` 获取更细的命令、输出路径与常见问题。

## 重要约束与不变量

- 结果目录结构与字段见 `docs/OUTPUT_SCHEMA.md`，属兼容性契约，不能随意更改
- `real_backtest/` 为脚本入口的“权威位置”，根目录同名脚本如存在，仅作为包装与过渡
- 禁止在未确认的情况下移动/删除：`run_combo_wfo.py`、`core/` 下模块、`real_backtest/core/` 下模块、`results/run_*/` 产物

## 常见坑与建议

- 避免在不同目录保留同名脚本的可运行副本；若为兼容，请在根目录放最薄包装并注明来源
- 修改公共接口前，先更新 `docs/OUTPUT_SCHEMA.md` 与 `docs/MODULE_MAP.md`，并在 PR 描述中强调影响范围
- 回测务必保证“无未来函数”原则：任何指标/权重均应只使用调仓日前的历史

## 变更管理

- 新功能：先补充/更新文档，再提交实现
- 破坏性变更：需给出迁移指南，确保旧结果的可重现性
- 建议添加结果对比脚本以验证一致性（样例见 `results_combo_wfo/*.csv`）

