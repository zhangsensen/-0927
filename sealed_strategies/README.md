# 封板策略仓库（Sealed Strategies Vault）

这个目录用于**长期保存“可复现、不可质疑”的封板策略版本**，将“结果 + 配置 + 关键脚本 + 说明文档”按版本归档，确保任何时候都能快速定位并复现。

> ⚠️ **重要**: 封板包**不包含虚拟环境**（`.venv/`），只包含配置文件（`pyproject.toml` + `uv.lock`）。  
> 用户需运行 `uv sync` 在本地自动生成环境，确保可复现性的同时保持包轻量（~15-30MB）。  
> 详见 [SEALING_GUIDELINES.md](SEALING_GUIDELINES.md)

## 目录规范

- 每个封板版本一个目录：`sealed_strategies/<version>_<yyyymmdd>/`
- 目录内必须包含：
  - `MANIFEST.json`：版本元信息（日期、区间、输入产物路径、策略数量等）
  - `CHECKSUMS.sha256`：归档文件的 sha256 校验，保证内容不可篡改
  - `REPRODUCE.md`：复现步骤（必须可一键跑通）⭐
  - `SEALING_GUIDELINES.md`：封板指南（如何正确封板）⭐
  - 关键产物文件（parquet / md / yaml / py）

> v3.2 起的最低复现标准：封板目录必须包含 `locked/pyproject.toml` + `locked/uv.lock` + `locked/src/`（源码快照）。  
> v3.4 起**禁止包含虚拟环境**（`.venv/`），确保封板包 < 50MB。

## 封板定义（Definition of Sealed Release）

封板=在不改变交易逻辑前提下，冻结以下内容：

1. **交易规则**：FREQ / POS / commission / execution timing 等规则与实现口径
2. **数据区间与切分点**：training_end_date、holdout 起止
3. **候选与生产清单产物**：候选（final candidates）、BT 审计、production pack
4. **审计口径**：最终收益、回撤、胜率等“对外口径”统一由哪一层引擎输出（v3.2 统一为 BT ground truth）
5. **关键脚本与配置**：用于生成上述产物的脚本与配置文件版本

> **允许**：bugfix（不改逻辑）、数据更新、文档完善、性能优化（不改结果）  
> **禁止**：修改核心引擎/因子库/交易规则/ETF 池定义（尤其禁止移除任何 QDII）  
> **v3.4+ 新增禁止**：包含虚拟环境（`.venv/`）到封板包 ⚠️

## 未来封板流程（建议）

- 运行验证流水线（WFO/VEC/rolling/holdout/BT）产出最终结果
- 运行封板工具将产物归档到本目录（生成 manifest + checksums + 说明）
- 只要有任何指标口径或实现点改动，必须新开版本目录（禁止覆盖旧版本）

## 工具

- 封板脚本：`scripts/seal_release.py`

## 为什么封板可复现

封板目录本质是一个“可搬走的自洽快照”，具备：

1. **结果快照**：`artifacts/` 下保存最终交付产物（候选、审计、报告）。
2. **逻辑快照**：`locked/scripts/` + `locked/configs/` + `locked/src/` 保存用于生成产物的关键实现与配置。
3. **环境快照**：`locked/pyproject.toml` + `locked/uv.lock` 冻结依赖版本（**不包含 `.venv/`**）。
4. **防篡改**：`CHECKSUMS.sha256` 对目录内全部文件做 sha256 校验。

只要满足以上四点，任何人拿到封板目录都可以在不依赖主仓库变化的前提下复现结果。

---

## 📦 封板包大小参考

| 版本 | 大小 | 说明 |
|------|------|------|
| v3.1 | 13MB | ✅ 标准（无虚拟环境） |
| v3.2 | 14MB | ✅ 标准（无虚拟环境） |
| v3.3 | 23MB | ✅ 包含较多脚本 |
| v3.4 | 16MB | ✅ 清理后（原 1.2GB，已删除 `.venv/`） |

> ⚠️ **如果封板包 > 100MB，必须检查是否误包含了虚拟环境或缓存！**

---

## 📖 相关文档

- **[SEALING_GUIDELINES.md](SEALING_GUIDELINES.md)**: 详细封板流程与规范（必读 ⭐）
- **各版本 REPRODUCE.md**: 每个版本的复现指南
