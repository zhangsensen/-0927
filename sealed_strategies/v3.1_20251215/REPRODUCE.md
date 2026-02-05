# 复现指南

本封板版本的产物已被复制到本目录，优先以本目录 artifacts 为准。

## 1. 校验（防篡改）

```bash
cd sealed_strategies/v3.1_20251215
sha256sum -c CHECKSUMS.sha256
```

## 2. 环境准备

本封板包含完整的源码快照 (`locked/src`) 和环境定义 (`locked/pyproject.toml`)。

```bash
# 假设已安装 uv
cd locked
uv sync --dev
```

## 3. 运行复现

使用 locked 目录下的脚本与源码进行复现，确保不受主分支变更影响。

```bash
# 运行 BT 审计 (使用 locked 源码)
uv run python scripts/batch_bt_backtest.py
```

## 4. 关键产物

- artifacts/final_candidates.parquet
- artifacts/bt_results.parquet
- artifacts/production_candidates.parquet
- artifacts/PRODUCTION_REPORT.md
