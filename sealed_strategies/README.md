# 封板策略仓库（Sealed Strategies Vault）

这个目录用于**长期保存“可复现、不可质疑”的封板策略版本**，将“结果 + 配置 + 关键脚本 + 说明文档”按版本归档，确保任何时候都能快速定位并复现。

## 目录规范

- 每个封板版本一个目录：`sealed_strategies/<version>_<yyyymmdd>/`
- 目录内必须包含：
  - `MANIFEST.json`：版本元信息（日期、区间、输入产物路径、策略数量等）
  - `CHECKSUMS.sha256`：归档文件的 sha256 校验，保证内容不可篡改
  - `IMPLEMENTATION_NOTES.md`：实现点与核心设计方式（必须写清楚）
  - `REPRODUCE.md`：复现步骤（必须可一键跑通）
  - 关键产物文件（parquet / md / yaml / py）

## 封板定义（Definition of Sealed Release）

封板=在不改变交易逻辑前提下，冻结以下内容：

1. **交易规则**：FREQ / POS / commission / execution timing 等规则与实现口径
2. **数据区间与切分点**：training_end_date、holdout 起止
3. **候选与生产清单产物**：候选（final candidates）、BT 审计、production pack
4. **审计口径**：最终收益、回撤、胜率等“对外口径”统一由哪一层引擎输出（v3.2 统一为 BT ground truth）
5. **关键脚本与配置**：用于生成上述产物的脚本与配置文件版本

> 允许：bugfix（不改逻辑）、数据更新、文档完善、性能优化（不改结果）
> 禁止：修改核心引擎/因子库/交易规则/ETF 池定义（尤其禁止移除任何 QDII）

## 未来封板流程（建议）

- 运行验证流水线（WFO/VEC/rolling/holdout/BT）产出最终结果
- 运行封板工具将产物归档到本目录（生成 manifest + checksums + 说明）
- 只要有任何指标口径或实现点改动，必须新开版本目录（禁止覆盖旧版本）

## 工具

- 封板脚本：`scripts/seal_release.py`
