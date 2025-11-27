<!-- ALLOW-MD -->
# 项目清理与架构优化方案 (2025-11-26)

**当前状态**: 项目结构混乱，存在多份代码副本、大量临时文件和归档目录。
**目标状态**: 建立以 `etf_rotation_optimized` 为核心的生产环境，保留 `etf_rotation_experiments` 为研发沙盒，充分利用 GPU 资源。

---

## 1. 现状审计 (Audit Findings)

### 1.1 核心目录分析

| 目录 | 状态 | 判定 | 说明 |
|:---|:---|:---|:---|
| **`etf_rotation_optimized/`** | ✅ **核心** | **保留 (Production)** | 包含最新的回测引擎、因子库和配置。已集成 Light Timing 模块。 |
| **`etf_rotation_experiments/`** | ⚠️ **混杂** | **保留 (Research)** | 包含 ML Ranker (GPU)、实验性策略和大量历史结果。 |
| `etf_rotation_experiments/etf_rotation_optimized/` | ❌ **冗余** | **删除** | 空目录或旧副本，应移除。 |
| `real_backtest/` (根目录) | ❌ **过时** | **删除** | 仅含单脚本，功能已被 `optimized/real_backtest` 取代。 |
| `_archive*/`, `archive/` | ❌ **垃圾** | **删除** | 历史遗留，占用空间且干扰视线。 |
| `a_shares_strategy/` | ❌ **过时** | **归档/删除** | 旧策略代码。 |

### 1.2 关键组件定位

- **生产回测引擎**: `etf_rotation_optimized/real_backtest/run_production_backtest.py` (已验证)
- **因子库**: `etf_rotation_optimized/core/precise_factor_library_v2.py`
- **GPU 工作负载**: `etf_rotation_experiments/strategies/ml_ranker/` (包含 PyTorch/LGBM 模型)
- **数据管理**: `etf_download_manager/` (需确认是否集成，暂保留)

---

## 2. 清理行动计划 (Action Plan)

请指示执行 Agent 按以下顺序操作：

### 步骤 1: 清理根目录垃圾
```bash
# 删除归档和临时目录
rm -rf _archive* archive
rm -rf _archive_experiments_* _archive_optimized_*
rm -rf tmp_delete_test.txt _path_audit.txt
rm -rf untitled

# 删除过时项目目录
rm -rf a_shares_strategy
rm -rf strategies  # 根目录下的 strategies 似乎是空的或旧的
rm -rf real_backtest # 根目录下的，已被 optimized 内部取代
```

### 步骤 2: 整理文档
```bash
# 创建文档目录 (如果不存在)
mkdir -p docs/archive

# 移动根目录散落的 MD/TXT 到 docs
mv *.md docs/ 2>/dev/null
mv *.txt docs/ 2>/dev/null
mv *.log docs/archive/ 2>/dev/null

# 恢复关键文件到根目录 (保持项目可读性)
mv docs/README.md .
mv docs/Makefile .
mv docs/pyproject.toml .
mv docs/uv.lock .
mv docs/setup_vscode_proxy.sh .
```

### 步骤 3: 规范化 Experiments
```bash
# 删除 experiments 内部的冗余副本
rm -rf etf_rotation_experiments/etf_rotation_optimized

# 清理 experiments 中的旧结果 (保留最近的)
# (建议手动检查 results_combo_wfo 内容，或按日期清理)
```

---

## 3. GPU 利用与架构融合 (GPU Integration)

当前机器配置 (RTX 5070 Ti) 主要应用场景在 **ML Ranker**。

### 3.1 现状
- `etf_rotation_optimized` (生产): 纯 CPU (Numba 加速)。
- `etf_rotation_experiments` (研发): 包含 `ml_ranker` (GPU 潜力)。

### 3.2 融合路线图
1.  **迁移**: 将 `etf_rotation_experiments/strategies/ml_ranker` 移动到 `etf_rotation_optimized/strategies/ml_ranker`。
2.  **集成**: 修改 `run_production_backtest.py`，增加调用 ML 模型的接口 (类似 `timing_signal` 的方式)。
3.  **环境**: 确保 `pyproject.toml` 或 `uv` 环境中包含 `torch` (CUDA版) 和 `lightgbm` (GPU版)。

---

## 4. 最终目录结构预览

```text
/home/sensen/dev/projects/-0927/
├── etf_rotation_optimized/       <-- 🌟 唯一生产核心
│   ├── core/                     (因子库, 择时模块)
│   ├── real_backtest/            (回测引擎)
│   ├── configs/                  (配置文件)
│   └── strategies/
│       └── ml_ranker/            <-- (建议迁移至此)
├── etf_rotation_experiments/     <-- 🧪 研发沙盒
│   ├── notebooks/
│   └── legacy_tests/
├── etf_download_manager/         <-- 💾 数据工具
├── docs/                         <-- 📚 文档中心
│   ├── MACHINE_CONFIGURATION.md
│   └── AUDIT_REPORT_20251126.md
├── scripts/                      <-- 🛠 通用脚本
├── README.md
├── Makefile
└── pyproject.toml
```

---

**执行建议**:
请将此文档交给执行 Agent，并要求其严格按照 **步骤 1 -> 步骤 2** 执行清理。步骤 3 (GPU融合) 建议作为单独的开发任务进行。
