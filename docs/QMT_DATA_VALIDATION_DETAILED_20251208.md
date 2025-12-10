# QMT 数据验证与对齐详细说明

**日期**: 2025-12-08

## 一、概述

本文档详尽记录了对来自 MINIQMT（下称 QMT）下载的 ETF 日线数据做全面验证、修复与策略复现的全过程，包含问题定位、修复方法、验证结果、关键脚本与生产建议。目标是确保 QMT 数据能替换或并行用于现有生产流水线并保证策略可复现性。


## 二、关键结论（要点）

- 已处理 ETF 数量：46（覆盖策略中 43 只 + 3 只额外）
- 修复后的生产目录：`raw/ETF/daily_qmt_aligned/`（推荐用于生产）
- Top100 策略（封板）在对齐后平均收益差异：-3.70%，相关性 0.8234
- 最佳策略（Top1）在对齐后差异：+2.04%（封板 237.45% vs QMT 239.49%）
- 主要问题来源：复权因子基准差异（已用 Daily 的 `adj_factor` 进行对齐），其次为少量日内价格微差对累积类因子（如 OBV）的放大效应


## 三、时间线与主要操作

1. 发现问题（2025-12-08）：QMT 数据 `ts_code` 列全为 None，且 `adj_factor` 与现有 `raw/ETF/daily/`（下称 Daily）存在系统性差异。
2. 初步错误修复：曾尝试把 `adj_factor` 固定为最新值 —— 导致部分 QDII ETF（如 513100、513500）收益异常（已回退该错误方案）。
3. 正确修复步骤：
   - 从文件名提取并填充 `ts_code`（生成 `raw/ETF/daily_qmt_fixed/`）。
   - 发现差异根源为复权基准日不同（Daily 的 `adj_factor` 为阶梯式，QMT 为每天动态）。
   - 采用 Daily 的 `adj_factor` 替换 QMT 的 `adj_factor`，重新计算 `adj_close` 等字段，生成 `raw/ETF/daily_qmt_aligned/`。
4. 使用完整 VEC 回测逻辑（与封板时代一致的 Net-New / timing 逻辑）对 Top100 封板策略进行逐条验证。
5. 输出验证结果并更新配置文件 `configs/combo_wfo_config_qmt.yaml` 指向对齐后的目录。


## 四、详细问题与根因

1. ts_code 缺失
   - 问题：QMT 原始 parquet 文件 `ts_code` 字段为空。
   - 解决：从文件名（例如 `513100.SZ_daily_...parquet`）提取代码并写回，输出到 `raw/ETF/daily_qmt_fixed/`。

2. 复权因子基准差异（主要问题）
   - 现象：部分 ETF（如 `510050.SH`, `512690.SH`, `510300.SH`）的 `adj_factor` 与 Daily 存在较大相对差异（最高 ~10%）；`adj_close` 相对差异放大到 ~3% 以上。
   - 根因：Daily 使用阶梯式后复权（只在除权/分红日改变），而 QMT 提供的复权因子是“每天动态”的后复权，二者基准不同导致不可比。
   - 解决：以 Daily 的 `adj_factor` 为基准替换 QMT 的 `adj_factor`（对共同日期使用 Daily 的值，超出 Daily 范围的日期使用 Daily 最后一个 `adj_factor`），并重新计算 `adj_open`, `adj_high`, `adj_low`, `adj_close`。

3. 累积类因子的敏感性
   - 说明：OBV/OBV_SLOPE、累积成交量相关因子对价格或方向的单日微差（0.01–0.04 元）极为敏感，因为 OBV 是累积量；单日方向不同会造成长期差异放大。
   - 观察：513100 在 2021-07-23 有 0.037 的价格差异，导致 OBV 累积差异，但因子总体相关性仍然很高（≈0.9999）。


## 五、验证方法（技术细节）

1. 数据加载
   - 使用 `DataLoader.load_ohlcv(etf_codes, start_date, end_date)` 加载 OHLCV（前复权列为 `adj_*`）。
   - 支持设置：`ETF_DATA_DIR` 或在配置文件 `configs/combo_wfo_config_qmt.yaml` 中设置 `data.data_dir`。

2. 因子计算
   - 因子库：`src/etf_strategy/core/precise_factor_library_v2.py`（18 因子），对齐时使用同一套实现来计算 Daily 与 QMT 数据上的因子。

3. 回测（VEC）
   - 使用与封板一致的 VEC 实现（见 `scripts/full_vec_bt_comparison.py` 的 VEC 段）
   - 关键实现细节：
     - 使用 `generate_rebalance_schedule(total_periods=T, lookback_window=LOOKBACK, freq=FREQ)` 保证与 BT 的调仓日一致
     - Net-New 买入逻辑、卖出后回收现金、timing_ratio 的 shift（`shift_timing_signal`）均与封板逻辑一致
     - 参数锁定（v3.1）：`FREQ=3`, `POS_SIZE=2`, `COMMISSION_RATE=0.0002`, `LOOKBACK=252`。

4. Top 100 验证流程
   - 读取 `results/top100_audit.csv`，遍历每个 `combo`（因子组合），在 Daily 与 QMT_aligned 上分别执行 VEC 回测，记录 `total_return`, `sharpe`, `max_dd` 等，并输出对比表 `results/qmt_aligned_validation_top100.csv`。


## 六、关键结果（数值）

- Top100 验证（使用 `raw/ETF/daily_qmt_aligned/`, 区间 2020-01-01 ~ 2025-10-14）
  - 封板记录平均收益（orig): 167.37%
  - QMT_aligned 平均收益: 163.67%
  - 平均差异: -3.70%
  - 相关性: 0.8234

- Top1（最佳组合）
  - 因子: `ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARPE_RATIO_20D`
  - 封板: 237.45%  QMT_aligned: 239.49%（差异 +2.04%）

- 受影响较大的 ETF（示例）:
  - `510050.SH` (adj_factor 相对差异 ~10.29%)
  - `512690.SH` (~8.37%)
  - `510300.SH` (~3.23%)
  - 共计 14 个 ETF 的 `adj_factor` 相对差异 > 0.1%


## 七、产生的文件与脚本（清单）

- 生成 / 修改文件：
  - `raw/ETF/daily_qmt_fixed/`  —— ts_code 修复后的 QMT 数据（中间产物）
  - `raw/ETF/daily_qmt_aligned/` —— 使用 Daily `adj_factor` 对齐后的 QMT 数据（推荐用于生产）
  - `results/qmt_validation_top100.csv`（初版）
  - `results/qmt_validation_top100_full.csv`（使用原 VEC 实现的对比）
  - `results/qmt_aligned_validation_top100.csv`（最终对齐后 Top100 验证结果）
  - `docs/QMT_DATA_VALIDATION_REPORT.md`（简要报告 v2.0，已更新）
  - `docs/QMT_DATA_VALIDATION_DETAILED_20251208.md`（本详细文档）

- 关键脚本与路径：
  - `scripts/validate_qmt_data.py` —— 数据检查 + ts_code 修复
  - `scripts/verify_qmt_strategy.py` —— 策略级验证脚本（含简化回测）
  - `scripts/full_vec_bt_comparison.py` —— 完整 VEC ↔ BT 对比脚本，包含 VEC 完整实现


## 八、操作命令（可复现）

1) 检查修复后的数据目录内容：

```bash
ls -la raw/ETF/daily_qmt_aligned/ | wc -l
```

2) 在 QMT 对齐数据上运行 Top100 验证（示例命令，仓内已有脚本）：

```bash
cd /home/sensen/dev/projects/-0927
uv run python scripts/verify_qmt_strategy.py --input results/top100_audit.csv --data_dir raw/ETF/daily_qmt_aligned
```

（注：脚本参数名示例，实际脚本已在本仓中使用内联方式运行，上述命令可据需调整）


## 九、建议与后续工作

1. 生产配置：将生产 pipeline 或 Cron 作业中的数据源替换为：

```yaml
data_dir: /home/sensen/dev/projects/-0927/raw/ETF/daily_qmt_aligned
```

2. 持续监控：对每日新增 QMT 数据采取以下流程：
   - 下载到 `raw/ETF/daily_qmt/`
   - 运行 `scripts/validate_qmt_data.py`（填充 `ts_code`）
   - 运行 `scripts/align_qmt_adj_factors.py`（或本次实现的对齐脚本）生成 `daily_qmt_aligned` 的增量更新
   - 在非生产环境对 Top 若干策略（例如 Top100 / Top10）做回归验证

3. 对敏感因子（如 OBV_SLOPE）增加容差或对输入做抖动容错（可选）：
   - 在计算累积型因子时，使用对齐后的 `adj_close` 并在极小价差（< 0.01 元）上忽略方向差异，以减少微小差异放大效应

4. 文档与审计：将本文件与 `docs/QMT_DATA_VALIDATION_REPORT.md` 一并提交审计记录，并在 `results/` 下保留对比 CSV 作为可追溯证据。


## 十、复现与联系方式

如需我把对齐脚本封装为可复用的 `scripts/align_qmt_adj_factors.py` 并加入到 CI 流程里，我可以继续实现并提交 PR。

维护者 / 联系人: `sensen`（仓内）

---
文档生成时间：2025-12-08
