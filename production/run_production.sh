#!/bin/bash
# 生产环境统一入口（无硬编码路径）
# - 自动推导项目根目录
# - 优先激活本地虚拟环境 .venv

set -euo pipefail
umask 0027

# 设置文件权限掩码（日志/文件权限控制）
umask 0027

# 解析脚本所在目录 -> 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/production}"
cd "$PROJECT_ROOT"

# 尝试激活本地虚拟环境
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.venv/bin/activate"
fi

# 检查 Python 依赖
if ! python3 -c 'import pandas, pyarrow, yaml' 2>/dev/null; then
  echo "❌ Python 依赖检查失败，请安装: pandas, pyarrow, pyyaml"
  exit 1
fi

echo "=== 生产流水线启动 ==="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "项目根: $PROJECT_ROOT"
echo "Python: $(command -v python3 || true)"
echo ""

# 依赖检查（关键包）
python3 scripts/tools/deps_check.py || { echo "[ERROR] Python依赖缺失"; exit 2; }

# 调用主调度脚本（位于 scripts/ 下）
python3 scripts/production_pipeline.py "$@"

echo ""
echo "=== 流水线执行完成 ==="
