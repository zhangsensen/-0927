#!/usr/bin/env bash
set -euo pipefail

# 获取脚本所在目录（etf_rotation_optimized/scripts）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Git 仓库根目录（上上级）
REPO_ROOT_DIR="$(cd "$SCRIPT_DIR"/../.. && pwd)"

HOOK_SRC="$SCRIPT_DIR/pre-commit-md-guard.sh"
HOOK_DST="$REPO_ROOT_DIR/.git/hooks/pre-commit"

if [[ ! -d "$REPO_ROOT_DIR/.git" ]]; then
  echo "❌ 未检测到 .git 目录，请在 Git 仓库根目录执行。"
  exit 1
fi

if [[ ! -f "$HOOK_SRC" ]]; then
  echo "❌ 未找到钩子脚本：$HOOK_SRC"
  exit 1
fi

cp "$HOOK_SRC" "$HOOK_DST"
chmod +x "$HOOK_DST"

echo "✅ 已安装 pre-commit 钩子：阻止随意新建 Markdown 文档"
echo "   规则：仅允许在 docs/ 新建 .md，且文档开头需包含允许标记（例如 <!-- ALLOW-MD -->）"
