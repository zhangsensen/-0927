#!/usr/bin/env bash
# MD creation guard: block adding new Markdown files unless they meet the policy.
# Policy:
# - Only allow adding new .md under docs/ directory.
# - New docs/*.md must contain an allow marker within first 20 lines:
#   one of: "<!-- ALLOW-MD -->", "[ALLOW-MD]", or a YAML line "ALLOW_MD: true".
# - Root README.md is always allowed (commonly exists/updated). New READMEs in subdirs are blocked unless docs/ + marker.

set -euo pipefail

# Collect newly added files in index (A = added, C = copied)
mapfile -t ADDED_MD < <(git diff --cached --name-status --diff-filter=AC | awk '$2 ~ /\.(md|MD)$/ {print $2}')

if [[ ${#ADDED_MD[@]} -eq 0 ]]; then
  exit 0
fi

REQUIRED_MARKERS=("<!-- ALLOW-MD -->" "[ALLOW-MD]" "ALLOW_MD: true")

err=0
for f in "${ADDED_MD[@]}"; do
  # Always allow the root README.md (common case); but block subdir READMEs as new additions
  if [[ "$f" == "README.md" ]]; then
    continue
  fi

  # Only allow creating new markdown in any 'docs/' directory
  # (e.g., docs/, project/docs/, etf_rotation_optimized/docs/, etc.)
  if [[ ! "$f" =~ /docs/ ]] && [[ "$f" != docs/* ]]; then
    echo "❌ 拦截：禁止在非 docs/ 目录新增 Markdown：$f" >&2
    echo "   允许位置：*/docs/*.md 或 docs/*.md" >&2
    err=1
    continue
  fi

  # Require an explicit allow marker within first 20 lines
  if [[ -f "$f" ]]; then
    head20=$(head -n 20 "$f" || true)
    has_marker=0
    for m in "${REQUIRED_MARKERS[@]}"; do
      if grep -qF "$m" <<<"$head20"; then
        has_marker=1
        break
      fi
    done
    if [[ $has_marker -eq 0 ]]; then
      echo "❌ 拦截：新增 $f 需在文档开头加入允许标记（任选其一）：" >&2
      echo "    <!-- ALLOW-MD -->  或  [ALLOW-MD]  或  ALLOW_MD: true" >&2
      err=1
    fi
  else
    echo "❌ 拦截：找不到文件（可能未添加到工作区）：$f" >&2
    err=1
  fi

done

if [[ $err -ne 0 ]]; then
  echo "—— Markdown 新建策略 ——" >&2
  echo "仅允许在 docs/ 目录新建 .md，且文档前 20 行包含允许标记。" >&2
  echo "如需临时放行，可先添加标记或在 docs/ 下创建。" >&2
  exit 1
fi

exit 0
