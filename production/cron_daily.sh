#!/bin/bash
# 每日定时任务脚本（无硬编码路径）
# 用法：crontab 示例
# 0 18 * * * /path/to/repo/production/cron_daily.sh

set -euo pipefail
umask 0027

# 设置文件权限掩码（日志/文件权限控制）
umask 0027

# 解析脚本所在目录 -> 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR%/production}"
cd "$PROJECT_DIR"

# 日志目录/文件（相对项目根目录）
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/production_$(date +%Y%m%d_%H%M%S).log"

# 优先激活本地虚拟环境
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.venv/bin/activate"
fi

# 运行流水线
{
  echo "========================================"
  echo "开始时间: $(date)"
  echo "项目根: $PROJECT_DIR"
  echo "Python: $(command -v python3 || true)"
  echo "========================================"
  
  # 依赖检查
  python3 scripts/tools/deps_check.py || { echo "[ERROR] Python依赖缺失"; exit 2; }

  python3 scripts/production_pipeline.py

  EXIT_CODE=$?

  echo "========================================"
  echo "结束时间: $(date)"
  echo "退出码: $EXIT_CODE"
  echo "========================================"
} 2>&1 | tee -a "$LOG_FILE"

# 清理旧日志（保留最近 30 天）
find "$LOG_DIR" -name "production_*.log" -mtime +30 -delete || true

exit ${EXIT_CODE:-0}
