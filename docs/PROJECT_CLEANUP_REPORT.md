# 项目清理报告

**日期**: 2024-11-27
**执行人**: GitHub Copilot

## 清理目标

- 删除过期/无用文件
- 保留核心回测框架和 Backtrader 验证系统
- 保持项目结构整洁

## 清理前备份

- Git tag: `backup-before-cleanup-YYYYMMDD_HHMMSS`
- Git commit: "BACKUP: 清理前备份 - 可复现性验证通过后的完整状态"

## 清理后项目结构

```
-0927/
├── configs/                    # 配置文件
├── docs/                       # 文档 (精简后)
├── etf_download_manager/       # ETF 数据下载管理
├── etf_rotation_experiments/   # 实验项目
│   ├── configs/
│   ├── core/
│   └── README.md
├── etf_rotation_optimized/     # 核心 WFO 系统 ★
│   ├── core/                   # 核心算法
│   ├── docs/                   # 系统文档
│   ├── real_backtest/          # 真实回测
│   ├── results/                # 运行结果
│   ├── scripts/                # 辅助脚本
│   ├── run_combo_wfo.py        # 组合 WFO 入口
│   └── run_unified_wfo.py      # 统一 WFO (可复现) ★
├── raw/                        # 原始数据
├── results/                    # 回测结果
│   ├── full_wfo_backtest_results.parquet  # 12,597 策略结果 ★
│   └── top5000_backtrader_verification_parallel.csv  # Backtrader 验证 ★
├── results_combo_wfo/          # WFO 运行结果
├── scripts/                    # 核心脚本
│   ├── full_wfo_backtest.py    # 全量回测
│   ├── full_wfo_backtest_v2.py # 全量回测 v2 (修复 NaN) ★
│   ├── verify_with_backtrader.py  # Backtrader 验证
│   └── parallel_verify_backtrader.py  # 并行验证
├── tests/                      # 测试文件
├── tools/                      # 工具脚本
├── .venv/                      # Python 虚拟环境
├── pyproject.toml              # 项目配置
├── Makefile                    # 构建命令
├── README.md                   # 项目说明
└── uv.lock                     # 依赖锁定
```

## 删除的内容

### 根目录
- `*.log` - 所有日志文件
- `test_adx_bug.py` - 临时测试脚本
- `final_audit_report.py` - 临时审计脚本
- `FINAL_CLEANUP_REPORT.md` - 旧清理报告
- `check_copilot.sh`, `QUICK_START.sh`, `setup_vscode_proxy.sh` - 临时脚本
- `.env`, `.pyscn.toml`, `.vulturerc`, `.mcp.json` - 配置文件

### scripts/ 目录
移动到 `scripts/archive/`:
- `analyze_random_trades.py`, `analyze_snowball_strategy.py` 等临时分析脚本

### docs/ 目录
- 删除过期文档 (MIGRATION_LOG.md, PROJECT_AUDIT_EXECUTIVE_SUMMARY.md 等)
- 删除 archive/, logs/, single_combo_dev/, etf_selection/, global-memory-bank/

### results/ 目录
移动到 `results/archive/`:
- snowball 相关分析结果
- calibrator 模型文件
- 临时 CSV 导出

### results_combo_wfo/ 目录
移动到 `results_combo_wfo/archive/`:
- 旧版本运行结果 (20251114, 20251124, 部分 20251126)

### etf_rotation_optimized/ 目录
- 删除 `__pycache__`, `.cache`, `.venv` 
- 删除 `uv.lock`, `pyproject.toml`, `Makefile`, `.gitignore` (使用根目录配置)
- 删除 `output/`, `factor_output/`
- 移动旧运行结果到 `results/archive/`

### etf_rotation_experiments/ 目录
- 删除大部分内容，仅保留核心 configs/ 和 core/

### 其他
- 删除 `.pytest_cache`, `quant_factor_trading.egg-info`
- 删除 `.roo`, `.windsurf`, `.cursor`, `.devcontainer`

## 保留的核心文件

### 可复现的回测系统
1. `etf_rotation_optimized/run_unified_wfo.py` - 统一规则 WFO
2. `scripts/full_wfo_backtest_v2.py` - 全量回测 (已修复 NaN)
3. `scripts/verify_with_backtrader.py` - Backtrader 验证
4. `scripts/parallel_verify_backtrader.py` - 并行 Backtrader 验证

### 关键结果
1. `results/full_wfo_backtest_results.parquet` - 12,597 策略回测结果
2. `results/top5000_backtrader_verification_parallel.csv` - Backtrader 验证结果

### Bug 修复记录
- `docs/BUG_SET_TRAVERSAL_NONDETERMINISM_20251127.md` - Set 遍历非确定性 bug 记录

## 验证

```bash
# 核心模块可导入
python -c "from etf_rotation_optimized.core import *"  # OK

# 结果文件完整
# 12,597 策略, Top 5 收益率: [134.2%, 123.4%, 116.1%, 115.7%, 101.7%]
```

## 项目统计

- 代码文件数: ~102 个 Python 文件 (不含 .venv)
- 项目大小: ~190MB (不含 raw/, .venv/, .git/)
