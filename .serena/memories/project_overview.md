项目：ETF 高频轮动策略平台（v3.1），目标是通过 WFO（筛选层）+ VEC（复算层）+ BT（审计层）形成可复现、可交付的 43 ETF 轮动策略，并重点关注 VEC↔BT 对齐、无前视偏差和确定性（deterministic）运行。

关键约束：交易规则锁死 FREQ=3、POS=2、不止损、不 cash；严禁移除 5 只 QDII ETF（513100/513500/159920/513050/513130）。

代码结构：源码在 src/；操作脚本在 scripts/；配置在 configs/；数据在 raw/ETF/；结果在 results/；对齐测试在 tests/。
