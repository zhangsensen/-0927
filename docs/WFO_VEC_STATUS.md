# WFO & VEC 对齐状态（不含 BT）

## 当前结论
- WFO 与 VEC 使用统一参数：`FREQ=3`、`POS=2`、`LOOKBACK=252`，训练截止日（Holdout）至 `2025-05-31`。
- 信号链路无前视：因子、择时、vol_regime 均在 t-1 计算，调仓在 t 收盘执行。
- 调仓日程：首个调仓日 ≥ LOOKBACK，调仓频率与持仓数与配置一致。
- 动态杠杆默认禁用（零杠杆原则），无额外复杂性。

## 近期产出
- 全量 VEC 结果（含对齐收益/Sharpe）：`results/vec_full_space_20251211_192612/full_space_results.csv`
- 无前视自检报告：`results/diagnostics/wfo_vec_no_lookahead_check.txt`
- BT 对齐暂不作为阻塞，Top500 BT 已完成但当前重点是 WFO/VEC 逻辑验证。

## 发现
- 未发现前视或时序错误；决策信息均来自 t-1，执行价为 t 收盘。
- WFO/VEC 因子、择时、vol_regime 与主配置保持一致，数据窗口与 Holdout 边界一致。

## 后续建议
- 若继续做 Alpha 开发，可直接在 VEC 上迭代；需要 BT 时再单独跑针对性子集（Top-N）。 


