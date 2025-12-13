# v3.2 封板实现点与核心设计方式

> 目标：把“能跑出来”升级为“可审计、不可质疑、可长期复现”。

## 1) 核心设计（Design Principles）

### A. 分层架构：Screening vs Ground Truth
- **Screening（筛选层）**：VEC + Rolling + Holdout
  - 负责高效探索与粗筛，允许一定程度的执行口径简化。
- **Ground Truth（审计层）**：BT（Backtrader，事件驱动）
  - 负责最终对外口径：收益、回撤、胜率、交易统计。

v3.2 的关键决策：**生产口径统一使用 BT**。当 VEC 与 BT 由于执行细节不同产生偏差时，BT 为准。

### B. Leakage Control（必须可证明）
- Rolling 一致性统计必须来自 **train-only summary**，不得混入 holdout 段。
- 任何基于 Rolling gate 的筛选，都必须能回答：“这条规则是否使用了训练期之外的信息？”

### C. Audit-Grade Split（训练/冷数据必须同口径）
- 训练期与冷数据对齐必须落实到 **BT 输出字段**：
  - `bt_train_return`
  - `bt_holdout_return`
- 这样在封板审计时，不会出现“用 BT 全区间去对比 VEC 训练段”的可质疑点。

## 2) 策略固定规则（Locked Rules）

- 调仓：FREQ=3（交易日）
- 持仓：POS=2
- 手续费：0.0002
- 不止损、不 cash（按现有引擎实现）

任何对上述规则的修改，都必须新开封板版本目录，不允许覆盖。

## 3) 关键实现点（Implementation Anchors）

### A. BT 分段收益输出（消除“跨区间对比”质疑）
- 实现文件：
  - scripts/batch_bt_backtest.py
- 关键行为：
  - 保留 TimeReturn 的日期索引
  - 基于 `training_end_date` 切分计算 `bt_train_return` / `bt_holdout_return`

### B. Production Pack 口径统一（BT Ground Truth）
- 实现文件：
  - scripts/generate_production_pack.py
- 关键行为：
  - 报告与排序基于 BT 分段收益（OOS-first）
  - 输出：TopN + All 候选，保证可追溯

### C. 产物归档与防篡改
- 归档工具：
  - scripts/seal_release.py
- 关键行为：
  - 复制关键产物到版本目录
  - 生成 MANIFEST.json（元信息）
  - 生成 CHECKSUMS.sha256（防篡改校验）

## 4) 本版本可复现入口

- 本目录仅保存“封板版本”的拷贝件：
  - artifacts/（最终结果）
  - locked/（关键脚本与配置）
- 复现步骤见：REPRODUCE.md
