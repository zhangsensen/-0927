# Zen MCP 使用指南

## 🚀 快速开始

### 1. 服务器已配置状态
- **AI提供商**: DeepSeek API (deepseek-chat)
- **可用工具**: 12个核心工具
- **服务器状态**: ✅ 运行正常

### 2. 可用工具列表

#### 🔥 AI对话工具
- **`chat`** - 与AI对话聊天
- **`thinkdeep`** - 深度思考分析
- **`consensus`** - 多AI共识决策

#### 🛠️ 开发工具
- **`planner`** - 智能规划助手
- **`codereview`** - 代码审查
- **`debug`** - 调试助手
- **`precommit`** - 提交前检查
- **`challenge`** - 批判性思维挑战

#### 🔧 实用工具
- **`apilookup`** - API查询
- **`listmodels`** - 查看可用模型
- **`version`** - 版本信息
- **`clink`** - 命令行集成

## 💡 使用方法

### 方法1: 通过Claude Code直接使用
重启Claude Code后，可以直接说：
- "用Zen MCP的chat工具帮我分析这个代码"
- "调用thinkdeep工具深度思考这个问题"
- "使用planner工具制定一个开发计划"

### 方法2: 手动启动服务器
```bash
cd /Users/zhangshenshen/.zen-mcp-server
./venv/bin/python server.py
```

### 方法3: 查看可用模型
```bash
# 启动服务器后调用listmodels工具
# 查看所有可用的AI模型
```

## 🎯 实际使用示例

### 示例1: 代码分析
```
用户: "用Zen MCP的codereview工具分析这个量化交易策略"

期望结果:
- DeepSeek会分析代码质量
- 识别潜在问题
- 提供改进建议
```

### 示例2: 深度思考
```
用户: "用thinkdeep工具深度分析ETF轮动策略的优化空间"

期望结果:
- DeepSeek进入深度思考模式
- 分析策略的多维度问题
- 提供系统性建议
```

### 示例3: 规划助手
```
用户: "用planner工具为这个项目制定开发计划"

期望结果:
- 结构化的开发计划
- 时间节点安排
- 风险评估
```

## ⚙️ 配置说明

### DeepSeek配置
```bash
CUSTOM_API_URL=https://api.deepseek.com
CUSTOM_API_KEY=sk-1623056992064d37ab38a3dd30d0bdde
CUSTOM_MODEL_NAME=deepseek-chat
DEFAULT_MODEL=deepseek-chat
```

### 模型选择
- **deepseek-chat**: 标准模式，快速响应
- **deepseek-reasoner**: 思考模式，深度推理

## 🔍 故障排除

### 常见问题
1. **服务器启动失败**: 检查venv环境
2. **API连接失败**: 验证API密钥
3. **工具不可用**: 检查DISABLED_TOOLS配置

### 日志查看
```bash
tail -f /Users/zhangshenshen/.zen-mcp-server/logs/mcp_server.log
```

## 🎉 开始使用

现在你可以：
1. 重启Claude Code让Zen MCP生效
2. 直接通过对话调用Zen工具
3. 享受DeepSeek强大的中文AI能力

有问题随时询问！