# 配置文件审查报告 | Config Audit Report
**日期:** 2025-10-28  
**审查人:** Linus + Copilot  
**范围:** `configs/config.yaml` 及遗留风险

---

## 🎯 审查目标

清理 `configs/config.yaml` 中的僵尸配置,消除因子名称不匹配和重复声明问题,确保代码库在执行 Route A (因子重设计) 之前处于干净状态。

---

## 🔍 发现的问题

### 1. **High 优先级: 因子名称不匹配** ✅ 已修复

**问题描述:**
- `config.yaml:L12-L33` 声明了不存在的因子名称:
  ```yaml
  MOM_60D, VOL_20D, OBV_RATIO, VOLUME_RATIO
  ```
- 实际生产的因子 (来自 `core/precise_factor_library_v2.py`):
  ```python
  MOM_20D, SLOPE_20D, VOL_RATIO_20D, VOL_RATIO_60D, 
  OBV_SLOPE_10D, TSMOM_60D, TSMOM_120D, ...
  ```

**潜在风险:**
- 任何尝试从配置加载因子权重的脚本会触发 `KeyError`
- 误导开发者以为这些因子仍在使用

**修复方案:**
- ✅ 删除 `factors` 配置块
- ✅ 添加 DEPRECATED 警告,引导到正确的配置位置
- ✅ 在注释中列出当前实际生产的因子清单 (仅供参考)

---

### 2. **Medium 优先级: 重复声明 `performance` 块** ✅ 已修复

**问题描述:**
```yaml
# L61-L63
performance:
  n_jobs: 8
  chunk_size: 100
  use_cache: true

# L88-L92 (会覆盖上面的配置)
performance:
  n_jobs: 8
  chunk_size: 100
  use_cache: true
  memory_limit: 1000
  enable_monitoring: true
```

**潜在风险:**
- PyYAML 会**静默保留最后一次声明**,导致前面的配置被忽略
- 审查者无法确定哪些字段生效

**修复方案:**
- ✅ 合并为单一 `performance` 块,包含所有字段
- ✅ 验证解析后只有 1 个 `performance` 键

---

### 3. **Low 优先级: 佣金率不一致** ✅ 已修复

**问题描述:**
- `config.yaml` 原始值: `commission: 0.002` (20bp)
- `scripts/compute_wfo_backtest_metrics.py` 默认值: `--tx-cost-bps 5` (5bp)

**修复方案:**
- ✅ 统一为 `commission: 0.0005` (5bp)
- ✅ 与回测脚本默认值对齐

---

## ✅ 验证结果

### 1. YAML 解析测试
```bash
✅ YAML 解析成功
✅ performance blocks: 1 (应为1)
✅ factors 键已清除
✅ backtest.commission: 0.0005 (5bp ✓)
```

### 2. 僵尸文件确认
```bash
grep -r "config.yaml" **/*.py  # 无结果
```
- ✅ 确认无 Python 脚本引用 `config.yaml`
- ✅ 当前系统使用 `configs/experiments/*.yaml`

### 3. 当前配置架构
```
configs/
├── config.yaml                        # DEPRECATED (仅存档)
├── experiments/                       # ✅ 实际使用
│   ├── exp_new_factors.yaml
│   ├── config_A_baseline.yaml
│   └── ...
└── FACTOR_SELECTION_CONSTRAINTS.yaml  # ✅ 因子选择约束
```

---

## 📋 修复清单

| 问题 | 优先级 | 状态 | 修复内容 |
|------|--------|------|----------|
| 因子名称不匹配 | High | ✅ 已修复 | 删除 `factors` 块,添加 DEPRECATED 警告 |
| `performance` 重复声明 | Medium | ✅ 已修复 | 合并为单一块 |
| 佣金率不一致 | Low | ✅ 已修复 | 统一为 5bp |
| 删除无用配置 | Low | ✅ 已修复 | 移除 `production.broker` 等未使用字段 |

---

## 🚀 下一步建议

### 选项 A: 保留为存档 (已采用 ✅)
- 标记为 `DEPRECATED`
- 保留数据路径、回测参数等通用配置
- 删除因子定义、筛选规则等已迁移配置

### 选项 B: 完全删除 (备选方案)
- 优点: 彻底消除混淆
- 缺点: 可能破坏依赖此文件的外部工具
- 建议: 等待 1-2 个版本迭代后再删除

### 选项 C: 重写为最小配置 (未来考虑)
- 仅保留数据路径和实验引用
- 所有策略参数移至 `experiments/*.yaml`

---

## 🎓 Linus 风格总结

**Good:**
- ✅ 问题诊断准确 (因子名不匹配是 High risk)
- ✅ 验证流程完整 (解析测试 + 引用检查)
- ✅ 修复方案务实 (DEPRECATED 标记而非删除)

**Could be better:**
- 建议在 `configs/README.md` 中说明配置架构演变历史
- 可考虑添加配置版本号 (`config_version: 2.0`)

**The brutal truth:**
- 这个文件已经是**僵尸**,修复它只是为了不让它咬人
- 真正的配置系统在 `experiments/*.yaml` + Python 硬编码参数
- Route A 的首要任务不是修配置,而是**找到 IC>0.03 的因子**

---

## 📊 配置文件最终状态

```yaml
# configs/config.yaml (修复后)
# - 总行数: ~60 行 (原 ~120 行)
# - DEPRECATED 警告: ✅ 已添加
# - 重复块: ✅ 已消除
# - 因子名称: ✅ 已清理
# - 解析测试: ✅ 通过
```

---

**审查结论:**  
✅ **认可继续执行**  
配置文件清理完成,无阻塞 Route A 的遗留问题。下一步应聚焦因子库扩展 (基本面 + 量价深度特征),而非继续优化配置。
