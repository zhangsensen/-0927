#!/usr/bin/env python3
"""
调试VectorBT STOCH参数
"""

import inspect

import vectorbt as vbt


def debug_stoch_params():
    """调试STOCH参数"""
    print("🔍 VectorBT STOCH参数检查...")

    # 检查STOCH.run的参数
    sig = inspect.signature(vbt.STOCH.run)
    print(f"STOCH.run参数:")
    for name, param in sig.parameters.items():
        if name != "self":
            print(
                f"  - {name}: {param.default if param.default != inspect.Parameter.empty else 'required'}"
            )

    # 检查VectorBT中是否有WILLR
    print(f"\nVectorBT中有WILLR吗? {hasattr(vbt, 'WILLR')}")

    # 列出所有VectorBT指标
    indicators = [
        attr for attr in dir(vbt) if attr.isupper() and not attr.startswith("_")
    ]
    print(f"可用指标: {sorted(indicators)}")


if __name__ == "__main__":
    debug_stoch_params()
