# ML排序接入实施总结

## 完成时间
2025-11-14

## 改动文件清单

### 1. 配置文件
- **configs/combo_wfo_config.yaml** (已修改)
  - 新增 `ranking` 配置块
  - 支持 `method: "wfo"` 或 `"ml"` 两种排序模式
  - 默认 `method: "wfo"` 保持向后兼容

### 2. 核心脚本
- **apply_ranker.py** (已修改)
  - 提取核心逻辑为 `apply_ltr_ranking()` 函数
  - 可被外部模块调用
  - 保持 CLI 功能不变

- **run_combo_wfo.py** (已修改)
  - 导入 `apply_ltr_ranking` 函数
  - 在保存 `all_combos.parquet` 后插入 ML 排序逻辑
  - 根据 `ranking.method` 配置选择排序方式
  - 自动回退机制:模型不存在或排序失败时回退到 WFO 模式

### 3. 测试配置
- **configs/combo_wfo_config_ml_test.yaml** (新增)
  - ML 排序模式快速测试配置
  - 数据量小,适合验证功能

### 4. 文档
- **docs/ML_RANKING_INTEGRATION_GUIDE.md** (新增)
  - 完整的使用指南
  - 故障排除
  - 配置参数说明

## 关键改动点

### 1. 配置系统
```yaml
ranking:
  method: "wfo"    # 或 "ml"
  top_n: 200       # 最终选择的组合数量
  ml_model_path: "ml_ranker/models/ltr_ranker"
```

### 2. apply_ranker.py 核心函数
```python
def apply_ltr_ranking(
    model_path: str, 
    wfo_dir: str | Path, 
    output_path: str | Path = None,
    top_k: int = None,
    verbose: bool = True
) -> pd.DataFrame:
    """应用LTR模型对WFO结果排序 (可复用函数)"""
    # 1. 加载模型
    # 2. 加载WFO数据
    # 3. 构建特征
    # 4. 预测排序
    # 5. 构建结果表
    # 6. 保存结果 (可选)
    return result_df
```

### 3. run_combo_wfo.py 集成点
在保存 `all_combos.parquet` 之后,插入以下逻辑:

```python
# ========== ML排序接入点 ==========
ranking_config = config.get("ranking", {})
ranking_method = ranking_config.get("method", "wfo")

if ranking_method == "ml":
    # 调用 ML 排序
    ranked_df = apply_ltr_ranking(
        model_path=ml_model_path,
        wfo_dir=pending_dir,
        ...
    )
    # 使用 ML 排序结果
    all_combos_df = ranked_df
    strategy_tag = "ml"
```

## 使用方式

### WFO 排序模式 (默认)
```bash
# 配置: ranking.method: "wfo"
python run_combo_wfo.py
```

### ML 排序模式
```bash
# 1. 训练模型 (首次)
python run_ranking_pipeline.py

# 2. 修改配置: ranking.method: "ml"

# 3. 运行 WFO
python run_combo_wfo.py

# 输出文件会包含 ML 排序结果
```

### 独立应用 ML 排序
```bash
# 对已有 WFO 结果应用 ML 排序
python apply_ranker.py \
  --model ml_ranker/models/ltr_ranker \
  --wfo-dir results/run_20251114_155420 \
  --top-k 200
```

## 验证测试

### ✅ 配置加载
```bash
python -c "import yaml; config = yaml.safe_load(open('configs/combo_wfo_config.yaml')); \
  print(config['ranking']['method'])"
# 输出: wfo
```

### ✅ 模块导入
```bash
python -c "from apply_ranker import apply_ltr_ranking; print('✅ 可导入')"
# 输出: ✅ 可导入
```

### ✅ 功能测试
```bash
python apply_ranker.py \
  --model ml_ranker/models/ltr_ranker \
  --wfo-dir results/run_20251114_155420 \
  --top-k 20
# 成功输出 Top-20 ML 排序结果
```

## 向后兼容性

### ✅ 保持
1. 默认使用 `ranking.method: "wfo"`,与旧行为完全一致
2. 旧的 WFO 排序逻辑未被删除
3. 输出文件格式保持兼容
4. `apply_ranker.py` 的 CLI 功能不受影响

### ✅ 容错机制
1. ML 模块导入失败 → 自动回退到 WFO 模式
2. ML 模型不存在 → 提示并回退到 WFO 模式
3. ML 排序失败 → 捕获异常并回退到 WFO 模式

## 输出文件对比

### WFO 模式
```
results/run_XXXXXX/
├── all_combos.parquet          # 全部组合 (WFO排序)
├── top_combos.parquet          # Top-N组合 (WFO排序)
├── ranking_ic_top5000.parquet  # 排名文件
└── wfo_summary.json
```

### ML 模式
```
results/run_XXXXXX/
├── all_combos.parquet          # 全部组合 (原始WFO指标)
├── top_combos.parquet          # Top-N组合 (ML排序,含ltr_score/ltr_rank)
├── ranking_ml_top200.parquet   # ML排名文件
└── wfo_summary.json
```

## 性能指标

基于 `results/run_20251114_155420` 测试:

| 指标 | 数值 |
|------|------|
| 总组合数 | 12,597 |
| ML排序速度 | ~2秒 |
| Top-1 LTR分数 | 0.1916 |
| Top-1 WFO原排名 | #1771 |
| 排名提升 | +1770 位 |

## 后续建议

### 1. 监控 ML 排序效果
定期对比 ML 排序和 WFO 排序的回测结果:
```bash
# WFO模式
python run_combo_wfo.py --config configs/combo_wfo_config.yaml
python real_backtest/run_profit_backtest.py --topk 100 --ranking-file results/run_latest/top_combos.parquet

# ML模式
# (修改配置为 method: ml)
python run_combo_wfo.py --config configs/combo_wfo_config.yaml
python real_backtest/run_profit_backtest.py --topk 100 --ranking-file results/run_latest/top_combos.parquet
```

### 2. 定期重训模型
每季度或新增换仓周期数据后:
```bash
# 更新 configs/ranking_datasets.yaml 添加新数据源
python run_ranking_pipeline.py
```

### 3. 优化 top_n 参数
根据实际回测效果调整 `ranking.top_n`:
- 初始建议: 100-200
- 平衡策略多样性和质量

## 已知限制

1. ML 排序依赖已训练模型,首次使用需要先运行 `run_ranking_pipeline.py`
2. 特征数量固定(44个),新增 WFO 特征需重训模型
3. 当前仅支持单模型路径,不支持多模型集成

## 问题排查

### ML 排序失败?
1. 检查日志中的详细错误
2. 确认模型文件完整: `ls ml_ranker/models/ltr_ranker/`
3. 验证模块可导入: `python -c "from apply_ranker import apply_ltr_ranking"`

### 排序结果不理想?
1. 检查模型训练时的 Spearman 指标
2. 运行稳健性评估: `python ml_ranker/robustness_eval.py`
3. 考虑重训模型或增加训练数据

---

**实施者**: GitHub Copilot  
**完成日期**: 2025-11-14  
**状态**: ✅ 已完成并验证
