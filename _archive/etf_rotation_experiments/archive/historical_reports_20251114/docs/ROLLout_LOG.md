# Rollout Log

## 2025-11-11 Unlimited 排序上线

- 模型目录：`results/models_wfo_only_v2/`
- 默认排名：`results/run_20251111_201500/ranking_blends/ranking_two_stage_unlimited.parquet`
- 调整内容：
  - 更新 `batch_backtest.py` 默认排名为 Unlimited
  - `run_profit_backtest.py` 自动优先使用 Unlimited 排名
  - `blend_summary.parquet` 增加 `is_default` 标记
- 备注：观测期 2 周，若 Gate 未通过将回滚至 Baseline 排序

