# ETF Rotation Optimized – 项目实施指南

> 精确数据 → 精确定义因子 → 标准化处理 → IC/WFO → 约束筛选 → 端到端验证

## 1. 数据质量纪律（Step 1）

- **满窗原则**：任意滚动窗口内出现缺失即判无效；由 `DataValidator.check_full_window` 执行。
- **覆盖率阈值**：每个标的非 NaN 覆盖率 ≥ 97%，低于阈值的日期/标的在因子计算中剔除。
- **Amount 定义统一**：优先使用典型价格 `(high+low+close)/3 * volume`；缺失时 fallback 到 `close*volume`。
- **异常检测**：成交量 20 日 rolling z-score > 3 触发预警，仅记录，不自动修补。
- **NaN 传播**：验证阶段不填充缺失，所有 NaN 在后续模块保持原位。

## 2. 精确定义因子（Step 2）

- **因子库结构**：
  - `precise_factor_library.py`：26 个全量因子，兼容历史脚本。
  - `precise_factor_library_v2.py`：10 个精选核心因子，用于主流程（趋势/动量、价格位置、波动、量能、价量耦合、反转）。
- **实现规范**：
  - 明确窗口、输入列、公式、方向性（`high_is_good` / `low_is_good` / `neutral`）。
  - 缺失值与满窗不足 → 返回 NaN；禁止 `ffill`/`bfill`。
  - 有界因子（PRICE_POSITION、PV_CORR、RSI）显式声明 bounds，后续不做截断。
- **互斥与配额**：参照 `FACTOR_SELECTION_CONSTRAINTS.yaml` 的 `family_quotas`、`mutually_exclusive_pairs`，所有新增因子必须登记。

## 3. 横截面标准化（Step 3）

- **处理流程**：
  1. 无界因子：逐日横截面 z-score。
  2. Winsorize：同日分位 2.5% / 97.5%，限制极端值。
  3. 有界因子：直接透传，保留原始值域。
- **实现入口**：`CrossSectionProcessor.process_all_factors()`，输出 `{factor_name: DataFrame}`；NaN 计数前后必须一致。
- **测试要求**：验证均值≈0、标准差≈1、有界因子上下限未破、NaN 未被改动。

## 4. IC 计算与统计（Step 4 – IC Calculator）

- **相关系数选项**：
  - `pearson`（默认）：衡量线性关系，用于标准流程。
  - `spearman`：秩相关，极值鲁棒。
  - `kendall`：非参数，样本小但耗时大。
- **输出内容**：
  - IC 时间序列（按日期）。
  - 统计指标：mean、std、IR、t-stat、p-value、Sharpe、偏度、峰度、样本数。
  - 可选多前向周期（1/5/20 日）。
- **显著性门槛**：默认 `p-value < 0.05` 判显著；可根据策略调整。

## 5. WFO 前向回测框架（Step 4 – WalkForwardOptimizer）

- **默认参数**：`IS=252`、`OOS=60`、`step=20`、`ic_threshold=0.05`。
- **流程**：
  1. 滑动划分 IS/OOS 窗口。
  2. IS 内计算 IC，选择 `mean IC > threshold` 的因子。
  3. OOS 内重新计算 IC/IR/Sharpe，评估选中因子表现。
  4. 汇总前向平均表现 + 因子入选频率表。
- **复现要求**：同一输入与参数多次运行必须给出相同结果；集成测试覆盖基础流程、极端场景和性能基准。

## 6. 约束筛选（Step 5）

> 当前待实现，设计要求已固化：

- **家族配额**：每个维度的最大入选数量在 `family_quotas` 中定义；例如动量最多 4 个。
- **互斥对**：`mutually_exclusive_pairs` 定义互斥关系，冲突时保留 IC 更优的因子。
- **全局相关性**：入选后计算因子间 |ρ|，大于 0.75 时执行去冗余（除非明确互补）。
- **最终因子数量**：目标 12–15 个，覆盖至少 6 大维度。

## 7. 端到端验证（Step 6）

> 待实现；上线前必须满足：

- **测试覆盖**：端到端集成测试（含性能压力、极端样本、NaN 传播、重复运行一致性）。
- **性能目标**：500 日 × 30 标 × 15 因子流程耗时 < 10 秒，内存 < 1 GB。
- **回测验收**：样本外 Sharpe > 0.5 或日均超额收益 > 0.02%；否则回溯步骤检查。
- **输出物**：最终因子清单、WFO 报告、风险提示、上线 SOP。

## 8. QA / DevOps 纪律

- **测试覆盖率**：模块完成必须 ≥ 95%，核心模块 100%；任何改动必须跑全套测试（Step1–Step4 当前共 120 个用例）。
- **文档产出**：每个 Step 结束写 Completion Report + Summary；重大修复需补充 `CRITICAL_DEFECTS_FIXED.md`。
- **日志与告警**：
  - 验证阶段使用 `logger.warning`，保留前 10 条样例。
  - WFO 报告输出 IC、Sharpe、入选频率，便于稽核。
- **配置管理**：所有参数化约束、阈值放在 YAML / dataclass；严禁硬编码魔数。
- **复现流程**：提供 `verify_stepX_integration.py` 类脚本，保证外部调用者可一步完成该 Step 的验证。

## 9. 扩展建议

- 新增因子前，先补充 `CANDIDATE_FACTORS_PRECISE_DEFINITION.md` 与 `FACTOR_SELECTION_CONSTRAINTS.yaml`。
- 如需扩充到 35–40 个候选，按 family/互斥与缺失处理规范逐个实现，并增加对应单测。
- 可考虑开发因子规范 → 代码/测试骨架自动生成工具，降低批量扩展成本。

---

> 本指南是 `etf_rotation_optimized` 项目的唯一可信流程定义。后续所有模块、参数调整或策略扩展必须遵循上述规则，并同步更新本文件及关联配置。
