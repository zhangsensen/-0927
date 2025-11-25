# 代码快照索引

**生成时间**: 2025-11-10  
**版本**: v1.0_wfo_20251109  
**用途**: 为LLM提供完整代码追溯，确保策略可复现

---

## 快照说明

本文件夹保存完整代码快照，分为以下类别：

### 1. 核心执行脚本
- `run_combo_wfo.py` - WFO主脚本（357行）
- `run_production_backtest.py` - 生产回测引擎（2192行，无未来函数）

### 2. 核心算法模块
- `combo_wfo_optimizer.py` - 组合级WFO优化器
- `precise_factor_library_v2.py` - 18因子库
- `cross_section_processor.py` - 横截面标准化
- `data_loader.py` - OHLCV数据加载器
- `ic_calculator_numba.py` - Numba加速IC计算

### 3. 配置文件
- `combo_wfo_config.yaml` - WFO配置（IS=252, OOS=60, step=60, freq=[8]）
- `strategy_config_v1.json` - 5个生产策略配置
- `allocation_config_v1.json` - 权重分配+相关性矩阵

### 4. 数据快照
- `top12597_backtest_by_ic_20251109_032515_20251110_001325.csv` - WFO完整结果（5.17MB）
- `strategy_candidates_selected.csv` - 6个候选策略
- `wfo_audit_summary.json` - WFO审核数据
- `strategy_validation_report.json` - 策略配置验证报告

---

## 快照时间戳

| 文件类型 | 最后修改时间 | Git Commit (如有) |
|----------|--------------|-------------------|
| Python脚本 | 2025-11-09 | N/A |
| 配置文件 | 2025-11-10 | N/A |
| WFO结果 | 2025-11-10 00:13 | N/A |
| 策略配置 | 2025-11-10 | N/A |

---

## 代码版本管理

**版本号**: v1.0_wfo_20251109  
**命名规则**: v{major}.{minor}_wfo_{YYYYMMDD}  
**下次更新**: 策略表现复审后（预计2025-12-10）

---

## 文件清单

待复制完成后自动生成...
