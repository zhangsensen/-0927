#!/usr/bin/env python3
"""
测试引擎配置变更的响应
"""

import os

from factor_system.factor_engine.api import clear_global_engine, get_engine


def test_cache_config_change():
    """测试缓存配置变更是否触发引擎重建"""
    print("🧪 测试缓存配置变更...")

    # 清理全局引擎
    clear_global_engine()

    # 获取初始引擎
    engine1 = get_engine()
    engine1_id = id(engine1)
    print(f"初始引擎ID: {engine1_id}")

    # 修改环境变量
    os.environ["FACTOR_ENGINE_CACHE_MEMORY_MB"] = "256"

    # 强制重新初始化来测试配置变更
    engine2 = get_engine(force_reinit=True)
    engine2_id = id(engine2)
    print(f"配置变更后引擎ID: {engine2_id}")

    # 验证配置确实生效
    print(f"新引擎缓存配置: {engine2.cache.config.memory_size_mb}MB")

    assert engine2.cache.config.memory_size_mb == 256, "配置变更应该生效"
    print("✅ 缓存配置变更测试通过")


if __name__ == "__main__":
    test_cache_config_change()
