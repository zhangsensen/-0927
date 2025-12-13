Python 约定：3.11+，4 空格缩进，snake_case，绝对导入（`from etf_strategy.core...`）。

工程约束：
- 核心引擎 `src/etf_strategy/core/` 默认视为冻结/禁改（除非 bug 修复且不改结果）。
- 避免不必要的大规模 BT 审计；优先 WFO+VEC Alpha 开发与对齐验证；需要审计时只跑 Top-N。
- 结果输出优先 Parquet（CSV 仅兼容读取）。

关键坑：set 遍历不确定性、信号必须滞后（`shift_timing_signal`）、统一调仓日程（`generate_rebalance_schedule`）、避免浮点直接相等比较。
