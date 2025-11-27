#!/bin/bash

# 创建VSCode Server机器配置目录
mkdir -p ~/.vscode-server/data/Machine

# 创建或更新VSCode Server机器配置文件
cat > ~/.vscode-server/data/Machine/settings.json << 'EOF'
{
    "http.proxy": "http://127.0.0.1:10809",
    "https.proxy": "http://127.0.0.1:10809",
    "http.proxyStrictSSL": false,
    "http.proxyAuthorization": null,
    "http.proxySupport": "on"
}
EOF

echo "✅ VSCode Server代理配置已更新为 10809 端口"