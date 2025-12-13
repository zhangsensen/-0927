推荐命令（UV 工作流）：
- 安装依赖：`uv sync --dev`；可选 editable：`uv pip install -e .`
- Step0 数据更新（QMT Bridge）：`uv run python scripts/update_daily_from_qmt_bridge.py --all`
- Step1 WFO：`uv run python src/etf_strategy/run_combo_wfo.py`
- Step2 VEC（读取最新 WFO 输出）：`uv run python scripts/run_full_space_vec_backtest.py`
- Step3 策略筛选（IC 门槛 + 综合得分）：`uv run python scripts/select_strategy_v2.py`
- BT 审计（TopN，小规模）：`uv run python scripts/batch_bt_backtest.py`
- 代码质量：`make format`、`make lint`、`make test`

一致性/对齐：
- VEC/BT 对齐指南：`docs/VEC_BT_ALIGNMENT_GUIDE.md`
