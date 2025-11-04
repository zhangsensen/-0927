#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Zen MCP与DeepSeek集成
"""

import asyncio
import json
import sys
from pathlib import Path

# 添加Zen MCP服务器路径
sys.path.insert(0, str(Path("/Users/zhangshenshen/.zen-mcp-server")))


async def test_deepseek_integration():
    """测试DeepSeek集成"""
    print("🚀 测试Zen MCP与DeepSeek集成")
    print("=" * 50)

    try:
        # 模拟MCP工具调用
        from providers.custom import CustomModelProvider
        from utils.config import Config

        # 加载配置
        config = Config()
        print(f"✅ 配置加载成功")
        print(f"   API URL: {config.custom_api_url}")
        print(f"   模型: {config.custom_model_name}")
        print(f"   默认模型: {config.default_model}")

        # 创建提供商
        provider = CustomModelProvider(config)
        print(f"✅ DeepSeek提供商创建成功")

        # 测试模型列表
        models = await provider.list_models()
        print(f"✅ 可用模型: {models}")

        # 测试简单对话
        print(f"🔄 测试对话功能...")
        response = await provider.generate_response(
            messages=[{"role": "user", "content": "请用中文回答：你是哪个AI模型？"}],
            model="deepseek-chat",
            max_tokens=100,
        )

        print(f"✅ DeepSeek响应:")
        print(f"   {response[:100]}...")

        print(f"\n🎉 Zen MCP与DeepSeek集成测试成功！")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_deepseek_integration())
