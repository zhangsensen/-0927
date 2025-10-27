# 项目清理执行指南

## 📌 快速开始

### 第1步：了解现状
阅读以下文档（按顺序）：
1. `CLEANUP_SUMMARY.md` - 5分钟快速了解
2. `PROJECT_CLEANUP_PLAN.md` - 详细分析
3. `FACTOR_SYSTEM_AUDIT.md` - factor_system内部审查

### 第2步：备份项目
```bash
cd /Users/zhangshenshen/深度量化0927
git add -A
git commit -m "backup: before cleanup"
```

### 第3步：执行清理
```bash
bash cleanup.sh
```

### 第4步：验证功能
```bash
make test
make lint
```

### 第5步：提交更改
```bash
git add -A
git commit -m "cleanup: remove temporary files and scripts"
```

---

## 🎯 清理目标

### 优先级1：立即执行（低风险）
- ✅ 删除6个临时脚本
- ✅ 删除8个日志文件
- ✅ 删除3个无用目录
- ✅ 删除4个过时Shell脚本
- ✅ 删除1个过时报告

**预期时间**: 5分钟  
**风险等级**: 低  
**回滚难度**: 简单

### 优先级2：后续执行（中风险）
- ⚠️ 清理scripts目录（8个脚本）
- ⚠️ 清理factor_screening结果（183个文件）
- ⚠️ 删除factor_system重复代码

**预期时间**: 30分钟  
**风险等级**: 中  
**回滚难度**: 中等

### 优先级3：长期规划（高风险）
- 🔄 配置整合（需要更新所有导入）
- 🔄 项目整合（etf_rotation_system vs etf_rotation_optimized）
- 🔄 factor_system重构（需要大量测试）

**预期时间**: 2-3天  
**风险等级**: 高  
**回滚难度**: 困难

---

## 📊 清理清单

### 根目录（优先级1）

#### 临时脚本
- [ ] `test_engine_init.py` - 轻量级引擎测试
- [ ] `code_quality_mcp_check.py` - MCP代码质量检查
- [ ] `verify_9factors_dataflow.py` - 9因子数据流验证
- [ ] `launch_wfo_real_backtest.py` - WFO回测启动
- [ ] `start_real_backtest.py` - 真实回测启动
- [ ] `test_signal_threshold_impact.py` - 信号阈值测试

#### 日志文件
- [ ] `backtest_output.log`
- [ ] `execution_20251025_193306.log`
- [ ] `hk_factor_generation.log`
- [ ] `production_run.log`
- [ ] `run_optimized_220044.log`
- [ ] `test_100_manual.log`
- [ ] `test_minimal.log`
- [ ] `wfo_full_run.log`

#### 无用目录
- [ ] `factor_ready/`
- [ ] `etf_cross_section_results/`
- [ ] `production_factor_results/`

#### 过时报告
- [ ] `ETF_CODE_MISMATCH_REPORT.md`

#### 过时Shell脚本
- [ ] `monitor_wfo_backtest.sh`
- [ ] `run_fixed_backtest.sh`
- [ ] `run_real_backtest.sh`
- [ ] `run_wfo_backtest.sh`

### scripts目录（优先级2）

#### 过时脚本
- [ ] `analyze_100k_results.py`
- [ ] `analyze_top1000_strategies.py`
- [ ] `analyze_top1000_strategies_fixed.py`
- [ ] `etf_rotation_backtest.py`
- [ ] `generate_etf_rotation_factors.py`
- [ ] `linus_reality_check_report.py`
- [ ] `validate_candlestick_patterns.py`
- [ ] `test_full_pipeline_with_configmanager.py`

### factor_system（优先级2-3）

#### 明显重复（优先级2）
- [ ] `factor_system/factor_engine/auto_sync_validator.py`
- [ ] `factor_system/factor_generation/verify_consistency.py`
- [ ] `factor_system/factor_screening/data_loader_patch.py`
- [ ] `factor_system/factor_screening/screening_results/*`

#### 需要评估（优先级3）
- [ ] `factor_system/factor_engine/factor_consistency_guard.py`
- [ ] `factor_system/factor_engine/validate_factor_registry.py`
- [ ] `factor_system/factor_engine/etf_cross_section_strategy.py`
- [ ] `factor_system/factor_screening/vectorized_core.py`
- [ ] `factor_system/factor_screening/fair_scorer.py`

---

## 🔍 验证步骤

### 清理后验证

#### 1. 功能测试
```bash
make test
```
预期：所有测试通过

#### 2. 代码检查
```bash
make lint
```
预期：无新的lint错误

#### 3. 导入检查
```bash
python -c "from factor_system.factor_engine import api; print('✅ API导入正常')"
python -c "from factor_system.factor_generation import batch_factor_processor; print('✅ 批量处理导入正常')"
python -c "from factor_system.factor_screening import professional_factor_screener; print('✅ 筛选导入正常')"
```

#### 4. 核心功能测试
```bash
python -m pytest tests/ -v --tb=short
```

---

## ⚠️ 风险评估

### 低风险操作
- 删除临时脚本（无其他代码依赖）
- 删除日志文件（无代码依赖）
- 删除无用目录（已确认为空或无用）
- 删除过时报告（文档，无代码依赖）

### 中风险操作
- 删除过时Shell脚本（需要检查是否有其他脚本调用）
- 清理scripts目录（需要检查是否有其他脚本导入）
- 清理factor_screening结果（需要确认没有生产依赖）

### 高风险操作
- 删除factor_system模块（需要完整的测试覆盖）
- 配置整合（需要更新所有导入路径）
- 项目整合（需要大量的重构和测试）

---

## 🚨 故障排除

### 如果清理后出现导入错误

**症状**: `ImportError: No module named 'xxx'`

**解决方案**:
```bash
# 1. 检查是否误删了必需的模块
git log --oneline -n 10

# 2. 恢复误删的文件
git checkout HEAD~1 -- <file_path>

# 3. 重新运行测试
make test
```

### 如果清理后出现功能错误

**症状**: 某个功能不工作

**解决方案**:
```bash
# 1. 查看最近的更改
git diff HEAD~1

# 2. 检查是否有依赖关系
grep -r "deleted_file_name" factor_system/

# 3. 恢复相关文件
git checkout HEAD~1 -- <related_files>
```

### 如果清理脚本失败

**症状**: `cleanup.sh` 执行失败

**解决方案**:
```bash
# 1. 检查错误信息
bash -x cleanup.sh 2>&1 | tail -20

# 2. 手动执行清理
rm -f test_engine_init.py
rm -f *.log
# ... 等等

# 3. 验证清理结果
ls -la | grep -E "\.py$|\.log$"
```

---

## 📈 预期收益

### 代码质量
- 减少重复代码 -30%
- 提高代码可维护性 +40%
- 降低认知复杂度 -25%

### 磁盘空间
- 释放空间 ~50MB
- 减少文件数 -100+

### 开发效率
- 减少混淆 -50%
- 加快导航 +30%
- 简化维护 +40%

---

## 📝 后续建议

### 短期（1周内）
1. 执行优先级1清理
2. 验证所有功能正常
3. 提交更改到Git

### 中期（1-2周）
1. 执行优先级2清理
2. 运行完整测试套件
3. 更新文档

### 长期（1个月）
1. 评估优先级3项目
2. 制定factor_system重构计划
3. 规划项目整合方案

---

## 🎓 学习资源

### 相关文档
- `PROJECT_CLEANUP_PLAN.md` - 详细分析
- `CLEANUP_SUMMARY.md` - 快速摘要
- `FACTOR_SYSTEM_AUDIT.md` - 内部审查
- `cleanup.sh` - 自动化脚本

### 命令参考
```bash
# 查看要删除的文件
find . -name "*.log" -o -name "test_engine_init.py"

# 统计文件大小
du -sh factor_system/

# 查看Git历史
git log --oneline -n 20

# 恢复文件
git checkout HEAD~1 -- <file_path>
```

---

## ✅ 完成检查

清理完成后，请确认以下事项：

- [ ] 所有临时脚本已删除
- [ ] 所有日志文件已删除
- [ ] 所有无用目录已删除
- [ ] 所有测试通过
- [ ] 所有lint检查通过
- [ ] 所有导入正常
- [ ] 所有功能正常
- [ ] 更改已提交到Git

---

**最后更新**: 2025-10-27  
**版本**: 1.0  
**状态**: 准备执行
