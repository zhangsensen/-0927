# MCP 服务器同步指南

## 问题说明

当你在家目录（桌面）安装新的 MCP 服务器后，项目目录可能无法自动使用这些服务器，因为项目有自己的 `.mcp.json` 配置文件。

## 解决方案

### 方法 1：手动同步配置

由于项目使用特定的 MCP 服务器配置，建议手动管理配置：

1. 查看全局安装的 MCP 服务器：
```bash
cd ~
claude mcp list
```

2. 根据需要将服务器添加到项目的 `.mcp.json` 文件中

项目已预配置了量化交易相关的 MCP 服务器：
- **quantconnect**: QuantConnect 平台集成
- **yahoo-finance**: 金融数据获取
- **filesystem**: 文件系统操作
- **web-search**: 网络搜索功能

### 方法 2：完全继承全局配置

如果你希望项目完全使用全局配置，可以备份并删除项目的 `.mcp.json` 文件：
```bash
# 备份项目配置
cp .mcp.json .mcp.json.backup

# 删除项目配置文件（谨慎使用）
# rm .mcp.json
```

## 项目特定 MCP 服务器

项目预配置了以下量化交易相关的 MCP 服务器：

- **quantconnect**: QuantConnect 云平台集成，支持算法回测和实盘交易
- **yahoo-finance**: Yahoo Finance 数据接口，获取实时和历史金融数据
- **filesystem**: 文件系统操作，管理数据和代码文件
- **memory**: 知识图谱管理，存储和检索量化知识
- **sequential-thinking**: 顺序思考工具，用于复杂量化问题分析

## 常见问题

**Q: 为什么项目需要自己的 MCP 配置？**
A: 项目需要特定的量化工具（如 QuantConnect、Yahoo Finance），这些不在默认全局配置中。

**Q: 如何添加新的 MCP 服务器？**
A: 直接编辑 `.mcp.json` 文件，或使用全局配置后手动合并。

**Q: 项目配置会影响全局配置吗？**
A: 不会，项目配置仅在当前项目目录中生效。

---
**更新**: 2025-10-14 | **状态**: ✅ 生产就绪