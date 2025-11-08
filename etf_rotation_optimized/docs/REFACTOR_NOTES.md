<!-- ALLOW-MD --># Real Backtest 真实回测目录重构说明

## 🔍 问题发现

1. **重复的 core 目录**: `real_backtest/core/` 与父目录 `etf_rotation_optimized/core/` 完全相同（SHA1校验一致）
2. **测试风格命名**: 脚本名称带有 `test_` 前缀，容易被误认为临时测试文件

## ✅ 已执行的重构

### 1. 删除重复 core 目录

```bash
# 验证两个目录完全相同
cd etf_rotation_optimized/real_backtest
shasum core/*.py ../core/*.py | sort -k2
# 所有文件SHA1一致 ✓

# 删除重复目录
rm -rf core
```

**原因**: 
- `real_backtest/` 下的脚本使用 `from core.xxx import yyy` 引用父级 core
- 子目录的 core 完全是冗余拷贝，增加维护成本

### 2. 脚本重命名

| 原文件名 | 新文件名 | 说明 |
|---------|---------|------|
| `test_freq_no_lookahead.py` | `run_production_backtest.py` | 生产级回测主脚本 |
| `top500_pos_grid_search.py` | `run_position_grid_search.py` | 持仓数网格搜索 |

**重命名理由**:
- `test_` 前缀易与单元测试混淆
- `run_` 前缀明确表明为生产运行脚本
- 防止被误删除（测试文件通常被认为可随意删除）

### 3. 更新引用

- 修正 `run_position_grid_search.py` 中的导入：
  ```python
  from run_production_backtest import backtest_no_lookahead
  ```
- 修正因子库导入：
  ```python
  from core.precise_factor_library_v2 import PreciseFactorLibrary  # 原为 factor_library
  ```

## 📁 当前目录结构

```
real_backtest/
├── configs/                        # 配置文件
│   ├── combo_wfo_config.yaml      # 回测配置（含佣金率）
│   ├── default.yaml
│   └── FACTOR_SELECTION_CONSTRAINTS.yaml
├── scripts/
│   └── cleanup.sh
├── results/                        # 回测结果输出
├── output/                         # 临时输出
├── run_production_backtest.py      # 【主】生产级回测脚本
├── run_position_grid_search.py     # 持仓数网格搜索
└── README.md
```

**注意**: 不再维护本地 `core/` 目录，统一使用父级 `../core/`

## 🔄 影响的文档

需要后续更新以下文档中的脚本引用：

1. `etf_rotation_optimized/README.md`
2. `etf_rotation_optimized/docs/PROJECT_OVERVIEW.md`
3. `etf_rotation_optimized/docs/MODULE_MAP.md`
4. `etf_rotation_optimized/QUICK_REFERENCE.md`

## ✅ 验证

```bash
# 语法检查通过
python -m py_compile run_production_backtest.py
python -m py_compile run_position_grid_search.py
```

## 📌 核心优势

- **减少冗余**: 删除 9 个重复的 core 模块文件
- **语义明确**: `run_` 前缀清楚表明生产脚本
- **维护性强**: 单一 core 目录，避免版本不一致
- **防误删除**: 正式命名防止被当作测试文件清理

---
**Created**: 2025-11-06  
**Status**: ✅ 完成
