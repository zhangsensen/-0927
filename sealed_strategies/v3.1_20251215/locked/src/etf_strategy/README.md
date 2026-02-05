# ETF 轮动策略模块 (`etf_strategy`)

> **版本**: v1.1 | **策略封板**: 2025-11-28 | **架构更新**: 2025-11-30

## ⚠️ 重要提示

**本模块是主力策略系统，核心逻辑已封板。请先阅读根目录 `AGENTS.md`！**

## 📁 目录结构

```
src/etf_strategy/
├── run_combo_wfo.py        # 🎯 WFO 入口脚本
├── core/                   # 核心引擎（🔒 禁止修改）
│   ├── combo_wfo_optimizer.py      # 滚动 WFO 优化器
│   ├── precise_factor_library_v2.py  # 18 因子库
│   ├── data_loader.py              # 数据加载
│   ├── backtester_vectorized.py    # VEC 回测引擎
│   ├── cross_section_processor.py  # 横截面处理
│   ├── ic_calculator_numba.py      # IC 计算（Numba）
│   ├── market_timing.py            # 择时模块
│   └── utils/rebalance.py          # 共享工具
└── auditor/                # BT 审计模块
    └── core/engine.py      # Backtrader 策略
```

## 🚀 快速运行

```bash
# 确保已安装项目（在项目根目录）
uv pip install -e .

# 运行 WFO 筛选
uv run python src/etf_strategy/run_combo_wfo.py
```

## 🔒 核心约束

1. **唯一入口**: `run_combo_wfo.py` 是唯一的 WFO 入口
2. **禁止修改**: `core/` 目录下的任何逻辑
3. **Import 规范**: 使用 `from etf_strategy.core.xxx import yyy`

## 📊 最佳策略（已锁定）

```
因子组合: CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D
收益率: 121.02%
胜率: 54.59%
盈亏比: 1.414
```

## 📖 详细文档

- 根目录 `AGENTS.md` - AI Agent 指南（必读！）
- 根目录 `README.md` - 项目说明
- `docs/BEST_STRATEGY_43ETF_UNIFIED.md` - 最佳策略详情
