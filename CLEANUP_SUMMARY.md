# 项目清理执行摘要

## 🎯 核心问题

根目录混乱，包含：
- 6个临时测试脚本（已过时）
- 8个日志文件（无保存价值）
- 4个过时Shell脚本（功能重复）
- 3个无用目录（空或过期）
- 2个重复的ETF轮动项目（`etf_rotation_system` vs `etf_rotation_optimized`）

## 📋 清理清单

### 第1阶段：根目录清理（立即执行）

**删除临时脚本（6个）**
```
test_engine_init.py
code_quality_mcp_check.py
verify_9factors_dataflow.py
launch_wfo_real_backtest.py
start_real_backtest.py
test_signal_threshold_impact.py
```

**删除日志文件（8个）**
```
*.log (所有日志文件)
```

**删除无用目录（3个）**
```
factor_ready/
etf_cross_section_results/
production_factor_results/
```

**删除过时报告（1个）**
```
ETF_CODE_MISMATCH_REPORT.md
```

**删除过时Shell脚本（4个）**
```
monitor_wfo_backtest.sh
run_fixed_backtest.sh
run_real_backtest.sh
run_wfo_backtest.sh
```

### 第2阶段：scripts目录清理（8个脚本）

**删除过时脚本**
```
analyze_100k_results.py
analyze_top1000_strategies.py
analyze_top1000_strategies_fixed.py
etf_rotation_backtest.py
generate_etf_rotation_factors.py
linus_reality_check_report.py
validate_candlestick_patterns.py
test_full_pipeline_with_configmanager.py
```

### 第3阶段：factor_screening清理（可选）

**清空过期结果**
```
factor_system/factor_screening/screening_results/ (183个文件)
```

## 🚀 执行步骤

### 1. 备份项目
```bash
cd /Users/zhangshenshen/深度量化0927
git add -A
git commit -m "backup: before cleanup"
```

### 2. 运行清理脚本
```bash
bash cleanup.sh
```

### 3. 验证功能
```bash
make test
make lint
```

### 4. 提交更改
```bash
git add -A
git commit -m "cleanup: remove temporary files and scripts"
```

## 📊 预期效果

| 指标 | 清理前 | 清理后 | 改进 |
|------|--------|--------|------|
| 根目录文件 | 20+ | 10 | -50% |
| 临时脚本 | 6 | 0 | -100% |
| 日志文件 | 8 | 0 | -100% |
| 无用目录 | 3 | 0 | -100% |
| 磁盘空间 | ~100MB | ~50MB | -50% |

## ⚠️ 注意事项

1. **备份优先**：执行前必须备份项目
2. **逐步执行**：按阶段执行，每步后验证
3. **测试验证**：清理后运行完整测试套件
4. **Git追踪**：所有删除操作都在Git中可追踪

## 🔍 保留的核心项目

**etf_rotation_optimized** ✅
- 整洁的模块化架构
- 完整的测试覆盖
- 统一的配置管理
- 生产就绪

**factor_system** ✅
- 统一的因子计算引擎
- 专业的因子筛选系统
- 完整的数据提供者
- 高质量的代码

**scripts** ✅（清理后）
- 生产流程脚本
- 生产验证脚本
- 缓存清理工具
- CI检查工具

## 📝 后续建议

1. **配置整合**：统一所有配置到 `config/` 目录
2. **文档整理**：将过程文档归档到 `docs/archived/`
3. **项目整合**：评估是否完全迁移到 `etf_rotation_optimized`
4. **CI/CD**：建立自动化清理流程

---

**执行时间**：~5分钟  
**风险等级**：低（所有删除的都是临时/过时文件）  
**回滚难度**：简单（Git可恢复）
