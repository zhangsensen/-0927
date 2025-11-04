"""
DataLoader配置测试 | DataLoader Configuration Tests

测试环境变量优先级和默认路径回退机制

作者: Linus Test
日期: 2025-01-17
"""

import os
from pathlib import Path

import pytest
from core.data_loader import DataLoader


class TestDataLoaderConfiguration:
    """数据加载器配置测试"""

    def test_environment_variable_override(self, tmp_path):
        """测试环境变量优先级"""
        # 创建临时目录
        custom_data = tmp_path / "custom_data"
        custom_cache = tmp_path / "custom_cache"
        custom_data.mkdir()
        custom_cache.mkdir()

        os.environ["ETF_DATA_DIR"] = str(custom_data)
        os.environ["ETF_CACHE_DIR"] = str(custom_cache)

        try:
            loader = DataLoader()

            # 验证使用环境变量路径
            assert str(loader.data_dir) == str(custom_data)
            assert str(loader.cache_dir) == str(custom_cache)

        finally:
            # 清理环境变量
            if "ETF_DATA_DIR" in os.environ:
                del os.environ["ETF_DATA_DIR"]
            if "ETF_CACHE_DIR" in os.environ:
                del os.environ["ETF_CACHE_DIR"]

    def test_default_path_fallback(self):
        """测试环境变量不存在时使用默认路径"""
        # 确保环境变量不存在
        if "ETF_DATA_DIR" in os.environ:
            del os.environ["ETF_DATA_DIR"]
        if "ETF_CACHE_DIR" in os.environ:
            del os.environ["ETF_CACHE_DIR"]

        loader = DataLoader()

        # 验证使用默认路径（应该包含特定关键字）
        data_dir_str = str(loader.data_dir)
        cache_dir_str = str(loader.cache_dir)

        # 数据目录应包含"etf"或"data"
        assert "etf" in data_dir_str.lower() or "data" in data_dir_str.lower()

        # 缓存目录应包含"cache"
        assert "cache" in cache_dir_str.lower()

        # 应该是绝对路径
        assert Path(loader.data_dir).is_absolute()
        assert Path(loader.cache_dir).is_absolute()

    def test_environment_variable_partial_override(self, tmp_path):
        """测试只设置部分环境变量"""
        custom_data = tmp_path / "custom_data"
        custom_data.mkdir()

        os.environ["ETF_DATA_DIR"] = str(custom_data)
        # 不设置ETF_CACHE_DIR

        try:
            loader = DataLoader()

            # 数据目录使用环境变量
            assert str(loader.data_dir) == str(custom_data)

            # 缓存目录使用默认值
            assert "cache" in str(loader.cache_dir).lower()

        finally:
            if "ETF_DATA_DIR" in os.environ:
                del os.environ["ETF_DATA_DIR"]
