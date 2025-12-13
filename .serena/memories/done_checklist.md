完成任务前检查：
- 代码可解析、脚本 exit code=0
- 指标合理且无前视偏差
- VEC↔BT 对齐（若涉及对齐）：平均差异 < 0.10pp（含 MAX_DD_60D 组合更严格）
- 输出产物存在且为 Parquet/Markdown
- 运行 `make format && make lint && make test`（或至少跑与变更相关的 pytest 子集）
