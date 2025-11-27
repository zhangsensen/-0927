# ETF 轮动系统总览

> 目标：在组合空间上做稳健的 WFO + ML 排序，再用统一回测引擎验证真实盈亏。

---

## 1. 系统定位

本工程 `etf_rotation_experiments/` 是 ETF 轮动策略的统一实现：

- 因子层：对 43 只 ETF 计算 18 个价格/量价/风险因子
- 组合层：在 2–5 因子组合空间上做 Walk-Forward Optimization（WFO）
- 排序层：用 Learning-to-Rank (LTR) 模型在组合层重新排序
- 回测层：基于排序结果跑真实盈亏回测（含成本与滑点）

所有入口都在 `applications/` 和 `real_backtest/`，配置集中在 `configs/`。

---

## 2. 顶层结构

- `applications/`
  - `run_combo_wfo.py`：完整 WFO + ML 排序主入口
  - `run_ranking_pipeline.py`：纯 ML 排序训练与评估
  - `train_ranker.py`：训练 LTR 模型
  - `apply_ranker.py`：加载已训练模型，对新 WFO 结果打分
- `core/`
  - `data_loader.py`：加载 OHLCV 数据 & ETF 基础信息
  - `precise_factor_library_v2.py`：18 个精确因子计算
  - `cross_section_processor.py`：横截面因子标准化与去极值
  - `pipeline.py`：将行情 → 因子 → 标准化 → WFO 输入的一体化流水线
- `strategies/`
  - `wfo/`：组合级 WFO 优化 (`combo_wfo_optimizer.py` 等)
  - `ml_ranker/`：特征工程、LTR 模型、稳健性评估
  - `backtest/`：信号到持仓/交易的回测逻辑（如使用）
- `real_backtest/`
  - `run_profit_backtest.py`：给定排序文件 + TopK + 滑点，回测多组组合
  - `run_production_backtest.py`：生产环境统一回测入口
- `configs/`
  - `combo_wfo_config.yaml`：WFO + 排序主配置
  - `combo_wfo_config_compound.yaml`：组合/复合策略配置
  - `ranking_datasets.yaml`：ML 排序训练数据配置
- `results/`
  - `run_YYYYMMDD_HHMMSS/`：单次 WFO+排序的全量输出
  - `run_latest`：指向最近一次 run 目录的软链接
- `results_combo_wfo/`
  - `TIMESTAMP1_TIMESTAMP2/`：某次回测的结果目录（CSV + SUMMARY JSON）
- `docs/`
  - 核心文档（实现说明、部署说明、本文件）
- `archive/`
  - 历史实验、旧报告、旧版本工程（只读，不再改动）

---

## 3. 数据契约

### 3.1 原始数据

- 位置：`data/etf_prices_template.csv`
- 典型字段：`trade_date`, `etf_code`, `open`, `high`, `low`, `close`, `volume`, `amount` 等
- 约束：
  - 交易日按升序排列
  - 所有 ETF 使用同一时区与复权方式

### 3.2 WFO 输出：`all_combos.parquet`

- 位置：`results/run_*/all_combos.parquet`
- 每一行代表一个因子组合（例如 `ADX_14D + CMF_20D + RET_VOL_20D`）
- 关键字段：
  - `combo`：字符串，因子名以 ` + ` 拼接
  - `combo_size`：组合中的因子数量
  - `mean_oos_ic`：OOS IC 的均值
  - `oos_ic_std` / `oos_ic_ir`：IC 波动与 IR
  - `positive_rate`：OOS IC 为正的比例
  - `best_rebalance_freq`：最优换仓频率（天）
  - `stability_score`：窗口间稳定性评分
  - `oos_ic_list`：每个 WFO 窗口的 IC 序列
  - 其它复合指标：`mean_oos_sharpe`, `oos_compound_sharpe`, `oos_compound_mean` 等

### 3.3 ML 排序输出：`ranking_ml_top2000.parquet`

- 位置：`results/run_*/ranking_ml_top2000.parquet`
- 在 `all_combos.parquet` 的基础上，增加 ML 排序相关字段：
  - `ltr_score`：LTR 模型输出的排序得分
  - 可能的特征列：
    - 标量特征：`mean_oos_ic`, `oos_ic_std`, `positive_rate`, `mean_oos_sharpe` 等
    - 序列聚合特征：IC 序列统计量（均值/方差/偏度等）
- 文件按 `ltr_score` 降序排序

### 3.4 回测结果

- 位置：`results_combo_wfo/T1_T2/`
- 文件：
  - `topK_profit_backtest_slipXBPS_*.csv`
  - `SUMMARY_profit_backtest_slipXBPS_*.json`
- CSV 核心字段：
  - `combo`, `freq`, `wfo_ic`, `wfo_score`
  - `final_value`, `total_ret`, `annual_ret`, `vol`, `sharpe`, `max_dd`
  - `final_value_net`, `annual_ret_net`, `sharpe_net`, `max_dd_net`
- JSON Summary：
  - `latest_run`：对应的 `results/run_*` 目录
  - `config_file`：使用的 YAML 配置
  - `top_source`：排序来源（例如 `ranking_file:ranking_ml_top2000.parquet`）
  - `slippage_bps`：滑点基点数
  - `count`：本次回测的组合个数
  - `mean_annual_net`, `median_annual_net`, `mean_sharpe_net`, `median_sharpe_net`

---

## 4. 核心流程

### 4.1 组合级 WFO 流程

入口：`applications/run_combo_wfo.py`

1. **加载配置**：`configs/combo_wfo_config.yaml`
2. **数据加载**：`core.data_loader.DataLoader` 从原始 CSV/缓存中加载 43 只 ETF 的 OHLCV
3. **因子计算**：`core.precise_factor_library_v2.PreciseFactorLibrary`
   - 18 个因子，例如：`MOM_20D`, `RSI_14`, `RET_VOL_20D`, `MAX_DD_60D` 等
4. **横截面标准化**：`core.cross_section_processor.CrossSectionProcessor`
   - 每天对所有 ETF 的因子横截面做 winsorize + z-score 标准化
5. **组合生成**：
   - 因子集合 F=18，组合规模 k ∈ {2,3,4,5}
   - 组合数约 12,597
6. **WFO 窗口切分**：
   - IS = 252 交易日，OOS = 60 交易日
   - 在 2020-01-01 至最新交易日间滚动，共 ~19 个窗口
7. **窗口内评估**：
   - 对每个组合，在每个窗口内评估 OOS IC / Sharpe / 正收益比例等
8. **综合打分与显著性**：
   - 聚合得到 `mean_oos_ic`, `stability_score` 等
   - 使用 FDR 控制进行显著性标记（筛选稳健组合）
9. **输出结果**：
   - 写入 `results/run_*/all_combos.parquet`

### 4.2 ML 排序流程

入口：`strategies/ml_ranker/pipeline.py`（被 `run_combo_wfo.py` 或 `run_ranking_pipeline.py` 调用）

1. **数据加载**：
   - 从 `all_combos.parquet` 读取组合级特征
2. **特征工程** (`strategies/ml_ranker/feature_engineer.py`)
   - 标量特征：
     - `combo_size`, `mean_oos_ic`, `oos_ic_std`, `oos_ic_ir`
     - `positive_rate`, `mean_oos_sharpe`, `oos_sharpe_std` 等
   - 序列特征：
     - 从 `oos_ic_list` 等序列字段中提取均值、波动率、偏度、尾部特征
   - 交叉特征：
     - 如 `mean_oos_ic * stability_score` 等
3. **模型训练/加载** (`strategies/ml_ranker/ltr_model.py`)
   - 使用 LightGBM Ranker 做 LTR
   - 生产运行时通常直接加载已训练好的模型
4. **评分与排序**：
   - 对每个组合计算 `ltr_score`
   - 生成 `ranking_ml_top2000.parquet`，只保留前 N 组（默认 2000）

### 4.3 真实回测流程

入口：`real_backtest/run_profit_backtest.py`

1. **参数与配置**：
   - `--ranking-file`：用于回测的排序结果文件
   - `--topk`：回测组合个数（若为 None 则回测全部）
   - `--slippage-bps`：滑点（基点）
2. **数据与因子加载**：
   - 从最新的 `results/run_*` 中加载因子与收益
3. **选择组合**：
   - 若显式指定 `--ranking-file`：
     - 读取对应文件，若传入 `--topk`，只截取前 K 行
   - 若未指定：
     - 根据配置中的 `ranking.method` 选择 ML / WFO 排序文件
4. **回测逻辑**：
   - 对每个组合：
     - 按组合内因子在日度横截面上生成打分
     - 根据 `best_rebalance_freq` 或 `--force-freq` 进行定期换仓
     - 计算含佣金的基准净值
     - 再应用滑点模型得到净值曲线
5. **输出**：
   - 组合级别的完整指标（含净值/年化/Sharpe/最大回撤/胜率等）
   - SUMMARY JSON 给监控/报表使用

---

## 5. 配置与运行

### 5.1 核心配置文件

- `configs/combo_wfo_config.yaml`
  - ETF 池
  - 日期区间
  - 因子列表
  - WFO 窗口长度（IS/OOS）
  - 排序方法（`ml` / `wfo`）
- `configs/ranking_datasets.yaml`
  - ML 排序训练数据源与切分
- `configs/combo_wfo_config_compound.yaml`
  - 复合策略/多层策略配置

### 5.2 一键生产流程

1. **完整 WFO + ML 排序**

```bash
cd etf_rotation_experiments
source ../.venv/bin/activate
python applications/run_combo_wfo.py --config configs/combo_wfo_config.yaml
```

1. **基于最新 run 做 TopK 回测**

```bash
# 示例：基于最新 run，回测 Top 100 组合
cd etf_rotation_experiments
source ../.venv/bin/activate
python real_backtest/run_profit_backtest.py \
  --ranking-file results/run_20251116_142927/ranking_ml_top2000.parquet \
  --topk 100 \
  --slippage-bps 10
```

---

## 6. 监控与回归检查

线上修改任何与信号/成本相关的逻辑前，建议至少执行：

1. 完整 WFO + 排序：

```bash
python applications/run_combo_wfo.py --config configs/combo_wfo_config.yaml
```

1. 最小 TopK 回测（例如 Top 5）：

```bash
python real_backtest/run_profit_backtest.py \
  --ranking-file results/run_latest/ranking_ml_top2000.parquet \
  --topk 5 \
  --slippage-bps 10
```

若需要进一步验证再跑测试：

```bash
cd etf_rotation_experiments
pytest tests/ -v
```

---

## 7. 变更影响边界

- 修改 `core/`：影响从因子到 WFO 的全链路，需要重跑 WFO + 回测
- 修改 `strategies/wfo/`：改变组合评估与筛选，需要至少重跑 WFO
- 修改 `strategies/ml_ranker/`：改变排序逻辑，需要重训/重打分 + 回测
- 修改 `real_backtest/`：只影响盈亏评估，需要重跑回测

保持这些边界清晰，可以避免一次修改拖垮整条流水线。
