# 🏗️ 项目整合与重构深度方案 (Project Consolidation Plan)

**日期**: 2025-11-30
**目标**: 解决项目割裂问题，建立统一、标准、可维护的代码架构。

---

## 1. 现状深度诊断 (Deep Diagnosis)

经过对代码库的深度扫描（包括依赖关系、路径引用和配置加载机制），我们发现了以下核心问题：

### 1.1 架构割裂 (Fragmentation)
项目目前被人为分割为三个独立的“孤岛”，导致逻辑复用困难：
*   **策略孤岛**: `etf_rotation_optimized`（核心策略逻辑）
*   **数据孤岛**: `etf_download_manager`（数据下载与管理）
*   **审计孤岛**: `strategy_auditor`（回测审计）

### 1.2 路径依赖 (Path Hacking)
**严重问题**: 超过 20 个脚本（如 `scripts/batch_bt_backtest.py`）使用了 `sys.path.insert(0, ...)` 这种脆弱的方式来强行引用其他模块。
*   *风险*: 一旦文件夹移动或重命名，所有脚本都会立即崩溃。
*   *现状*: 脚本必须在特定目录下运行，否则找不到模块。

### 1.3 配置分散 (Config Scattering)
*   **策略配置**: 位于 `configs/combo_wfo_config.yaml`。
*   **数据配置**: 位于 `etf_download_manager/config/etf_config.yaml`。
*   **硬编码风险**: `etf_download_manager` 的代码中硬编码了 `Path(__file__).parent` 来寻找配置文件，这意味着如果移动代码，配置加载就会失效。

### 1.4 幽灵代码 (Ghost Code)
*   `tests/test_backtest_engine.py` 引用了不存在的 `Phase2BacktestEngine`，这是旧版本的残留，会误导开发者认为存在“隐藏引擎”。

---

## 2. 目标架构 (Target Architecture)

我们将采用 Python 行业标准的 `src/` 布局，将所有核心逻辑统一到一个包中，实现“一次安装，处处调用”。

```
project_root/
├── src/                          # ✅ 统一源码目录 (Single Source of Truth)
│   ├── etf_strategy/             # (原 etf_rotation_optimized)
│   │   ├── core/                 # 核心策略引擎
│   │   └── auditor/              # (原 strategy_auditor，作为子模块整合)
│   └── etf_data/                 # (原 etf_download_manager)
│       └── core/                 # 数据管理核心
├── scripts/                      # ✅ 统一执行入口 (Centralized Entry Points)
│   ├── run_wfo.py                # 策略优化入口
│   ├── run_backtest.py           # 回测入口
│   └── manage_data.py            # 数据管理入口
├── configs/                      # ✅ 统一配置中心 (Centralized Configs)
│   ├── strategy_config.yaml      # 策略参数
│   ├── data_config.yaml          # 数据下载参数
│   └── ...
├── pyproject.toml                # 项目定义文件
└── tests/                        # 统一测试目录
```

---

## 3. 执行路线图 (Execution Roadmap)

### 第一阶段：清理与准备 (Cleanup)
**目标**: 移除干扰项，确保环境干净。
1.  [ ] **删除幽灵测试**: 删除 `tests/test_backtest_engine.py`。
2.  [ ] **清理空目录**: 删除 `etf_rotation_optimized/results` 等无用目录。

### 第二阶段：物理迁移 (Physical Migration)
**目标**: 建立 `src` 结构，物理移动文件。
1.  [ ] **创建目录**: 建立 `src/` 目录。
2.  [ ] **迁移策略模块**: 将 `etf_rotation_optimized` 移动并重命名为 `src/etf_strategy`。
3.  [ ] **迁移数据模块**: 将 `etf_download_manager` 移动并重命名为 `src/etf_data`。
4.  [ ] **整合审计模块**: 将 `strategy_auditor` 移动到 `src/etf_strategy/auditor`。

### 第三阶段：代码修复 (Code Refactoring)
**目标**: 修复因移动文件导致的引用错误，彻底消除 `sys.path` hack。
1.  [ ] **安装项目**: 在 `pyproject.toml` 中配置项目，使用 `pip install -e .` 安装，使 `src` 下的包在全局可见。
2.  [ ] **修复导入**:
    *   将 `import etf_rotation_optimized.core` 批量替换为 `import etf_strategy.core`。
    *   将 `import etf_download_manager` 批量替换为 `import etf_data`。
3.  [ ] **移除 Hack**: 删除所有脚本中的 `sys.path.insert(...)` 代码。

### 第四阶段：配置统一 (Config Consolidation)
**目标**: 让所有模块都从 `configs/` 读取配置。
1.  [ ] **移动配置文件**: 将 `etf_download_manager/config/*.yaml` 移动到根目录 `configs/`。
2.  [ ] **重构加载逻辑**: 修改 `src/etf_data/core/config.py`，使其不再依赖相对路径，而是读取环境变量或默认的 `configs/` 路径。

### 第五阶段：验证 (Verification)
1.  [ ] **数据验证**: 运行 `python scripts/manage_data.py --action summary`。
2.  [ ] **策略验证**: 运行 `python scripts/run_wfo.py`。
3.  [ ] **回测验证**: 运行 `python scripts/run_backtest.py`。

---

## 4. 预期收益 (Benefits)

1.  **消除割裂感**: 所有代码都在 `src` 下，逻辑清晰，模块间调用自然。
2.  **提升稳定性**: 移除 `sys.path` hack 后，脚本可以在任何目录下运行，不再脆弱。
3.  **配置集中**: 所有参数都在 `configs/` 下一目了然，不再需要去深层目录找配置文件。
4.  **专业化**: 符合 Python 开源项目的标准结构，便于后续维护和交接。

---

## 5. 下一步建议 (Next Steps)

建议立即执行 **第一阶段（清理）**，然后按顺序推进。
是否开始执行第一阶段？
