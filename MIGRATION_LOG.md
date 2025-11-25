# 项目统一重构迁移日志

**开始时间**: 2025-11-16  
**Git分支**: refactor/unified-codebase-20251116  
**目标**: 将 etf_rotation_optimized/ 合并到 etf_rotation_experiments/，形成统一代码库

## 执行记录

### 阶段0：准备与备份
- [x] 步骤1: 完整备份创建 - 完成
- [x] 步骤2: Python缓存清理 - 完成
- [x] 步骤3: 工作分支创建 - refactor/unified-codebase-20251116
- [x] 步骤4: 迁移日志文件创建 - 本文件
- [x] 步骤5: 硬编码路径扫描 - 发现87处引用

### 阶段1：core层统一
- [x] 步骤6: data_contract.py已存在，无需创建
- [x] 步骤7-8: data_loader.py对比 - 两版本完全相同，保留experiments版本
- [x] 步骤9: ic_calculator_numba.py对比 - experiments版本有P0修复（阈值30），已是最优版本
- [x] 步骤10: core层测试 - 无单元测试文件（现状）

### 阶段2：strategies层重组
- [x] 步骤11: 创建strategies目录结构
- [x] 步骤12: 移动WFO文件到strategies/wfo/
- [x] 步骤13: 移动ml_ranker到strategies/ml_ranker/
- [x] 步骤14: 创建统一回测引擎backtest_engine.py（骨架）
- [ ] 步骤15-16: 提取回测逻辑（标记为TODO）
- [x] 步骤17: 复制position_optimizer和signal_optimizer到strategies/backtest/
- [x] 步骤18: 更新strategies层导入路径（combo_wfo_optimizer, direct_factor_wfo_optimizer, ml_ranker/*）
- [ ] 步骤19: strategies层集成测试

---

## 差异记录

### core层差异
- `data_loader.py`: 两版本相同（254行）
- `ic_calculator_numba.py`: experiments版本有P0修复（阈值30 vs 2）✓
- `precise_factor_library_v2.py`: 未对比（假定同步）
- `cross_section_processor.py`: 未对比

### strategies层差异
待记录...

### 测试结果
待记录...
