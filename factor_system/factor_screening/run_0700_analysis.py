#!/usr/bin/env python3
"""
0700.HK 多时间框架因子分析快速启动脚本
Linus式简单设计：一个脚本，直接启动，无多余配置
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """直接启动0700.HK多时间框架分析"""

    # 配置文件路径
    config_file = project_root / "configs" / "0700_multi_timeframe_config.yaml"

    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        sys.exit(1)

    print(f"🚀 启动0700.HK多时间框架因子分析")
    print(f"📄 配置文件: {config_file}")
    print("=" * 60)

    # 直接导入并启动
    try:
        from professional_factor_screener import main as screener_main

        # 设置命令行参数
        sys.argv = [
            "professional_factor_screener.py",
            "--config", str(config_file)
        ]

        # 启动分析
        screener_main()

    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("请确保在正确的项目目录中运行此脚本")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 分析启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()