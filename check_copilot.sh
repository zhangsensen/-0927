#!/bin/bash
# GitHub Copilot 连接诊断工具

echo "🔍 GitHub Copilot 连接诊断"
echo "=========================================="
echo ""

# 1. 检查代理端口
echo "📡 1. 代理端口检查"
if netstat -tlnp 2>/dev/null | grep -q "127.0.0.1:10809"; then
    echo "   ✅ Xray 代理正在监听 10809 端口"
else
    echo "   ❌ Xray 代理未在 10809 端口监听"
    echo "   💡 请检查 Xray 服务是否运行"
fi
echo ""

# 2. 检查 VSCode Server 配置
echo "📝 2. VSCode Server 配置检查"
VSCODE_CONFIG=~/.vscode-server/data/Machine/settings.json
if [ -f "$VSCODE_CONFIG" ]; then
    PROXY_SETTING=$(grep -o '"http.proxy": "[^"]*"' "$VSCODE_CONFIG" | cut -d'"' -f4)
    if [ "$PROXY_SETTING" = "http://127.0.0.1:10809" ]; then
        echo "   ✅ VSCode Server 代理配置正确: $PROXY_SETTING"
    else
        echo "   ⚠️  VSCode Server 代理配置: $PROXY_SETTING"
        echo "   💡 期望值: http://127.0.0.1:10809"
        echo "   💡 运行 bash setup_vscode_proxy.sh 修复"
    fi
else
    echo "   ❌ VSCode Server 配置文件不存在"
    echo "   💡 运行 bash setup_vscode_proxy.sh 创建配置"
fi
echo ""

# 3. 检查环境变量
echo "🌐 3. 终端环境变量检查"
if [ "$http_proxy" = "http://127.0.0.1:10809" ]; then
    echo "   ✅ http_proxy: $http_proxy"
else
    echo "   ⚠️  http_proxy: ${http_proxy:-未设置}"
fi
if [ "$https_proxy" = "http://127.0.0.1:10809" ]; then
    echo "   ✅ https_proxy: $https_proxy"
else
    echo "   ⚠️  https_proxy: ${https_proxy:-未设置}"
fi
echo ""

# 4. 测试网络连接
echo "🌍 4. 网络连接测试"
if curl -s --connect-timeout 5 --proxy http://127.0.0.1:10809 https://api.github.com/zen > /dev/null 2>&1; then
    echo "   ✅ 通过代理可以访问 GitHub API"
else
    echo "   ❌ 无法通过代理访问 GitHub API"
    echo "   💡 请检查 Xray 配置和网络连接"
fi
echo ""

# 5. 总结
echo "=========================================="
echo "📋 诊断总结"
echo ""
ISSUES=0

if ! netstat -tlnp 2>/dev/null | grep -q "127.0.0.1:10809"; then
    echo "❌ 需要启动 Xray 代理服务"
    ISSUES=$((ISSUES + 1))
fi

if [ ! -f "$VSCODE_CONFIG" ] || ! grep -q "http://127.0.0.1:10809" "$VSCODE_CONFIG"; then
    echo "❌ 需要运行: bash setup_vscode_proxy.sh"
    ISSUES=$((ISSUES + 1))
fi

if [ $ISSUES -eq 0 ]; then
    echo "✅ 所有检查通过！"
    echo ""
    echo "如果 Copilot 仍然不可用，请尝试："
    echo "   1. 在 VSCode 中执行 'Reload Window' (Ctrl+Shift+P)"
    echo "   2. 重新连接 SSH"
    echo "   3. 检查 VSCode Copilot 扩展是否已登录"
else
    echo "⚠️  发现 $ISSUES 个问题需要修复"
fi
echo ""
