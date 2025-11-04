#!/bin/bash
# 纯向量化回测 - 快速启动脚本

set -e

echo "=================================================="
echo "纯向量化回测系统 - 暴力测试所有因子组合"
echo "=================================================="

# 进入工作目录
cd "$(dirname "$0")"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

# 默认参数
WORKERS=${WORKERS:-7}  # 默认7个进程
TOP_K=${TOP_K:-100}    # 默认保存Top-100
LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "运行参数:"
echo "  - 并行进程数: $WORKERS"
echo "  - 保存Top-K: $TOP_K"
echo "  - 日志级别: $LOG_LEVEL"
echo ""

# 运行回测
python3 run_backtest.py \
    --config configs/backtest_config.yaml \
    --workers $WORKERS \
    --top-k $TOP_K \
    --log-level $LOG_LEVEL

echo ""
echo "=================================================="
echo "回测完成！查看 results/ 目录获取结果"
echo "=================================================="
