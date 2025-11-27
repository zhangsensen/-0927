Goal: 移除 production/profit 回测中对 etf_rotation_optimized 的隐式依赖，并补全阶段4-6的记录。

Scope:
1. `strategies/backtest/production_backtest.py` 仅依赖 PROJECT_ROOT 与配置指定路径，删除硬编码缓存/结果目录及 `sys.path` 注入。
2. `etf_rotation_experiments/real_backtest/run_profit_backtest.py` 仅从 experiments 树加载配置与结果目录，提供环境变量或参数覆盖。
3. `MIGRATION_LOG.md` 记录阶段4-6的工具、文档、测试步骤，包含 `tools/validate_combo_config.py` 等产出与验证命令。

Constraints:
- 不新增全局副作用，优先复用现有配置结构。
- 保持回测 CLI 兼容；若路径需默认值，统一读 PROJECT_ROOT。
- 记录所有新脚本与测试命令，确保可追溯。

Deliverables:
- 更新的回测脚本与日志。
- 通过的 `tools/validate_combo_config.py` 校验。
- MIGRATION_LOG 中的阶段4-6条目及测试记录。
