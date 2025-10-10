"""
FactorEngine配置管理

统一管理所有配置路径和参数，支持环境变量覆盖。
确保不同环境部署时只改配置不改代码。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field


class DataPaths(BaseModel):
    """数据路径配置"""
    raw_data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("FACTOR_ENGINE_RAW_DATA_DIR", "raw")),
        description="原始OHLCV数据目录"
    )
    cache_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("FACTOR_ENGINE_CACHE_DIR", "cache/factor_engine")),
        description="因子缓存目录"
    )
    registry_file: Path = Field(
        default_factory=lambda: Path(os.getenv("FACTOR_ENGINE_REGISTRY_FILE", "factor_system/factor_engine/data/registry.json")),
        description="因子注册表文件路径"
    )

    def ensure_dirs(self) -> None:
        """确保所有目录存在"""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)


class CacheConfig(BaseModel):
    """缓存配置"""
    memory_size_mb: int = Field(
        default=int(os.getenv("FACTOR_ENGINE_MEMORY_MB", "500")),
        description="内存缓存大小(MB)",
        gt=0,  # 必须大于0
        le=10240  # 最大10GB
    )
    disk_cache_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("FACTOR_ENGINE_DISK_CACHE_DIR", "cache/factor_engine")),
        description="磁盘缓存目录"
    )
    ttl_hours: int = Field(
        default=int(os.getenv("FACTOR_ENGINE_TTL_HOURS", "24")),
        description="缓存生存时间(小时)",
        gt=0,  # 必须大于0
        le=168  # 最大7天
    )
    enable_memory: bool = Field(
        default=os.getenv("FACTOR_ENGINE_ENABLE_MEMORY", "true").lower() == "true",
        description="是否启用内存缓存"
    )
    enable_disk: bool = Field(
        default=os.getenv("FACTOR_ENGINE_ENABLE_DISK", "true").lower() == "true",
        description="是否启用磁盘缓存"
    )
    copy_mode: str = Field(
        default=os.getenv("FACTOR_ENGINE_COPY_MODE", "view"),
        description="数据复制模式: view/copy/deepcopy"
    )


class EngineConfig(BaseModel):
    """引擎配置"""
    n_jobs: int = Field(
        default=int(os.getenv("FACTOR_ENGINE_N_JOBS", "1")),
        description="并行计算任务数",
        ge=-1,  # -1表示使用所有核心
        le=32  # 最大32个核心
    )
    chunk_size: int = Field(
        default=int(os.getenv("FACTOR_ENGINE_CHUNK_SIZE", "1000")),
        description="数据块大小",
        gt=0,  # 必须大于0
        le=10000  # 最大10000
    )
    validate_data: bool = Field(
        default=os.getenv("FACTOR_ENGINE_VALIDATE_DATA", "true").lower() == "true",
        description="是否验证数据完整性"
    )
    log_level: str = Field(
        default=os.getenv("FACTOR_ENGINE_LOG_LEVEL", "INFO"),
        description="日志级别"
    )


class FactorEngineSettings(BaseModel):
    """FactorEngine完整配置"""
    data_paths: DataPaths = Field(default_factory=DataPaths)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)

    class Config:
        arbitrary_types_allowed = True  # 允许Path类型

    @classmethod
    def from_env(cls) -> "FactorEngineSettings":
        """从环境变量加载配置"""
        return cls()

    def ensure_directories(self) -> None:
        """确保所有必要的目录存在"""
        self.data_paths.ensure_dirs()
        self.cache.disk_cache_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> "FactorEngineSettings":
        """从字典创建配置"""
        return cls(**data)


# 全局配置实例
_global_settings: Optional[FactorEngineSettings] = None


def get_settings() -> FactorEngineSettings:
    """
    获取全局配置实例（单例模式）

    Returns:
        FactorEngineSettings: 配置实例
    """
    global _global_settings
    if _global_settings is None:
        _global_settings = FactorEngineSettings.from_env()
    return _global_settings


def reset_settings() -> None:
    """重置全局配置（主要用于测试）"""
    global _global_settings
    _global_settings = None


# 便捷的环境变量设置函数
def set_raw_data_dir(path: str | Path) -> None:
    """设置原始数据目录"""
    os.environ["FACTOR_ENGINE_RAW_DATA_DIR"] = str(path)
    reset_settings()


def set_cache_dir(path: str | Path) -> None:
    """设置缓存目录"""
    os.environ["FACTOR_ENGINE_CACHE_DIR"] = str(path)
    reset_settings()


def set_registry_file(path: str | Path) -> None:
    """设置注册表文件路径"""
    os.environ["FACTOR_ENGINE_REGISTRY_FILE"] = str(path)
    reset_settings()


def set_memory_size_mb(size_mb: int) -> None:
    """设置内存缓存大小"""
    os.environ["FACTOR_ENGINE_MEMORY_MB"] = str(size_mb)
    reset_settings()


# 预定义配置模板
def get_development_config() -> FactorEngineSettings:
    """开发环境配置"""
    return FactorEngineSettings(
        cache=CacheConfig(
            memory_size_mb=200,  # 开发环境内存较小
            ttl_hours=2,         # 缓存时间较短，便于调试
        ),
        engine=EngineConfig(
            n_jobs=1,            # 开发环境单线程，便于调试
            validate_data=True,  # 开发环境开启数据验证
            log_level="DEBUG",   # 详细日志
        )
    )


def get_production_config() -> FactorEngineSettings:
    """生产环境配置"""
    return FactorEngineSettings(
        cache=CacheConfig(
            memory_size_mb=1024,  # 生产环境内存较大
            ttl_hours=168,        # 7天缓存
        ),
        engine=EngineConfig(
            n_jobs=-1,            # 生产环境使用所有CPU核心
            validate_data=False,  # 生产环境假设数据已验证
            log_level="WARNING",  # 只输出警告和错误
        )
    )


def get_research_config() -> FactorEngineSettings:
    """研究环境配置"""
    return FactorEngineSettings(
        cache=CacheConfig(
            memory_size_mb=512,
            ttl_hours=24,          # 1天缓存
            enable_memory=True,
            enable_disk=True,
        ),
        engine=EngineConfig(
            n_jobs=4,              # 研究环境使用部分CPU核心
            validate_data=True,
            log_level="INFO",
        )
    )