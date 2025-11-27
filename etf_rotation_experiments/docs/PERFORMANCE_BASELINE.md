# 性能基线与监控

**建立日期**: 2025-11-16  
**目的**: 记录 WFO/ML 流程的性能指标，作为后续优化和回归测试的基准

---

## 当前基线 (2025-11-16)

### 硬件环境
- CPU: 未记录（建议执行 `lscpu | grep "Model name"` 记录）
- 内存: 未记录（建议执行 `free -h` 记录）
- 存储: SSD（推测，基于路径）
- Python: 3.12.3

### WFO 性能指标

| run_id | 排序方式 | 组合数 | WFO耗时 | 吞吐量 | 峰值内存 | 回测耗时 |
|--------|----------|--------|---------|--------|----------|----------|
| 035732 | ML | 12597 | 49s | 257 combo/s | 未测量 | 12s (2000组合) |
| 132810 | WFO | 12597 | 47s | 268 combo/s | 未测量 | 11s (2000组合) |

**观察**:
- WFO 核心评估耗时稳定在 47~49s
- ML 排序额外开销 ~2s (特征构建 + 模型预测)
- 回测单组合平均耗时 ~6ms

### 内存使用 (待测量)

建议在下次运行时添加内存监控:
```bash
/usr/bin/time -v python applications/run_combo_wfo.py --config configs/combo_wfo_config.yaml 2>&1 | grep "Maximum resident"
```

### 磁盘 I/O

- 缓存读取: ~50ms (ohlcv_*.pkl, 43 ETF × 1399 天)
- Parquet 写入: ~100ms (all_combos.parquet, 7.9M)
- 因子数据写入: ~200ms (18 个 parquet 文件)

---

## 性能目标 (优化后)

| 指标 | 当前 | 目标 | 策略 |
|------|------|------|------|
| WFO 吞吐量 | 257 combo/s | **500+ combo/s** | Polars 向量化 + Numba JIT |
| 峰值内存 | 未测量 | <2GB | 增量计算 + 内存映射 |
| 缓存命中率 | ~95% | 99% | 优化缓存键生成 |
| ML 排序耗时 | 2s | <1s | 特征预计算 + 模型量化 |

---

## 监控检查点

### 每次 WFO 运行后自动记录

在 `run_combo_wfo.py` 末尾添加性能日志:
```python
import time
import psutil

perf_log = {
    'timestamp': timestamp,
    'wfo_duration': wfo_end - wfo_start,
    'combo_count': len(all_combos_df),
    'throughput': len(all_combos_df) / (wfo_end - wfo_start),
    'peak_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
    'ranking_method': ranking_method,
}

with open(pending_dir / 'perf_metrics.json', 'w') as f:
    json.dump(perf_log, f, indent=2)
```

### 回归检测

在 CI/CD 中添加性能回归测试:
```bash
# 对比最近两次 run 的性能
python tools/check_perf_regression.py \
  --baseline results/run_20251116_035732/perf_metrics.json \
  --current results/run_LATEST/perf_metrics.json \
  --threshold 0.2  # 性能劣化超过 20% 时告警
```

---

## 历史性能趋势

| 日期 | run_id | WFO耗时 | 吞吐量 | 优化项 |
|------|--------|---------|--------|--------|
| 2025-11-16 | 035732 | 49s | 257/s | 基线 |
| 2025-11-16 | 132810 | 47s | 268/s | 基线 |
| TBD | - | - | - | Polars 重构 |
| TBD | - | - | - | Numba JIT IC 计算 |

---

## 性能剖析（待执行）

### CPU Profiling
```bash
python -m cProfile -o wfo.prof applications/run_combo_wfo.py --config configs/combo_wfo_config.yaml
python -m pstats wfo.prof
# 查看 top 10 热点函数
```

### 内存 Profiling
```bash
python -m memory_profiler applications/run_combo_wfo.py --config configs/combo_wfo_config.yaml
```

### 向量化率检测
```bash
# 检测循环和 .apply() 调用
grep -rn "\.apply(" etf_rotation_experiments/strategies/ etf_rotation_experiments/core/
grep -rn "for .* in .*:" etf_rotation_experiments/strategies/ | grep -v "# vectorized"
```

---

## 告警阈值

自动触发性能告警的条件:
- WFO 吞吐量 < 200 combo/s (降低 >20%)
- 峰值内存 > 4GB (超过合理范围)
- 缓存命中率 < 90% (I/O 瓶颈)
- 单次运行崩溃/超时 (>300s)

**通知渠道**: 日志 + 邮件 (生产环境)

---

## 下次性能审查

**日期**: 2025-12-01  
**目标**: 完成 Polars 重构后的性能对比  
**交付**: 更新本文档 + 性能报告
