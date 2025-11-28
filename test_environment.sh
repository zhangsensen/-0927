#!/bin/bash

echo "=== 环境对比测试 ==="
echo "当前时间: $(date)"
echo "当前目录: $(pwd)"
echo "当前用户: $(whoami)"
echo

echo "=== PATH 对比 ==="
echo "系统完整PATH:"
echo "$PATH" | tr ':' '\n' | nl
echo

echo "=== 虚拟环境状态 ==="
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✓ 在虚拟环境中: $VIRTUAL_ENV"
    echo "虚拟环境Python: $(which python)"
    echo "虚拟环境Python版本: $(python --version)"
else
    echo "✗ 不在虚拟环境中"
    echo "系统Python: $(which python3)"
    echo "系统Python版本: $(python3 --version)"
fi
echo

echo "=== Node.js 环境 ==="
echo "Node.js路径: $(which node)"
echo "Node.js版本: $(node --version)"
echo "Claude CLI路径: $(which claude)"
echo

echo "=== MCP 配置检查 ==="
if [ -f ".mcp.json" ]; then
    echo "✓ 找到项目MCP配置: .mcp.json"
    echo "MCP服务器数量: $(grep -c '"type"' .mcp.json)"
else
    echo "✗ 未找到项目MCP配置"
fi

if [ -f "$HOME/.claude/settings.json" ]; then
    echo "✓ 找到Claude全局配置"
else
    echo "✗ 未找到Claude全局配置"
fi
echo

echo "=== 环境变量检查 ==="
echo "DISPLAY: $DISPLAY"
echo "SHELL: $SHELL"
echo "TERM: $TERM"
echo "LANG: $LANG"
echo

echo "=== VSCode 特定检查 ==="
if [ -n "$VSCODE_PID" ]; then
    echo "✓ 在VSCode环境中运行 (PID: $VSCODE_PID)"
else
    echo "✗ 不在VSCode环境中运行"
fi

if [ -n "$WSL_DISTRO_NAME" ]; then
    echo "✓ 在WSL环境中: $WSL_DISTRO_NAME"
else
    echo "✗ 不在WSL环境中"
fi