# -*- coding: utf-8 -*-
"""依赖检查工具"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DependencyChecker:
    """检查依赖是否可用"""

    REQUIRED_DEPS = ["pandas", "numpy", "scipy", "vectorbt", "talib", "sklearn"]

    OPTIONAL_DEPS = {
        "performance": ["polars", "numba"],
        "visualization": ["matplotlib", "plotly", "seaborn"],
        "web": ["dash"],
        "database": ["sqlalchemy", "redis"],
        "scheduling": ["schedule"],
    }

    @classmethod
    def check_dependencies(cls) -> Tuple[List[str], Dict[str, List[str]]]:
        """检查依赖状态."""

        missing_required: List[str] = []
        missing_optional: Dict[str, List[str]] = {}

        for dep in cls.REQUIRED_DEPS:
            try:
                module_name = dep.replace("-", "_")
                importlib.import_module(module_name)
                logger.debug("✅ %s 已安装", dep)
            except ImportError:
                missing_required.append(dep)
                logger.warning("❌ %s 未安装", dep)

        for category, deps in cls.OPTIONAL_DEPS.items():
            missing: List[str] = []
            for dep in deps:
                try:
                    importlib.import_module(dep.replace("-", "_"))
                    logger.debug("✅ %s (%s) 已安装", dep, category)
                except ImportError:
                    missing.append(dep)
                    logger.debug("⚠️ %s (%s) 未安装", dep, category)
            if missing:
                missing_optional[category] = missing

        return missing_required, missing_optional

    @classmethod
    def print_dependency_status(cls) -> None:
        """打印依赖状态."""

        missing_required, missing_optional = cls.check_dependencies()

        print("=== 依赖状态检查 ===")

        if missing_required:
            print(f"❌ 缺失必需依赖: {', '.join(missing_required)}")
            print("请运行: pip install " + " ".join(missing_required))
        else:
            print("✅ 所有必需依赖已安装")

        if missing_optional:
            print("\n⚠️  可选依赖状态:")
            for category, deps in missing_optional.items():
                print(f"  {category}: 缺失 {', '.join(deps)}")
                print(f"  安装命令: pip install {' '.join(deps)}")
        else:
            print("✅ 所有可选依赖已安装")

    @classmethod
    def get_installation_commands(cls) -> Dict[str, str]:
        """获取安装命令."""

        missing_required, missing_optional = cls.check_dependencies()

        commands: Dict[str, str] = {}

        if missing_required:
            commands["required"] = "pip install " + " ".join(missing_required)

        for category, deps in missing_optional.items():
            if deps:
                commands[category] = "pip install " + " ".join(deps)

        return commands

    @classmethod
    def check_version_compatibility(cls) -> Dict[str, str]:
        """检查版本兼容性."""

        version_info: Dict[str, str] = {}

        for dep in cls.REQUIRED_DEPS:
            try:
                module = importlib.import_module(dep.replace("-", "_"))
                version_info[dep] = getattr(module, "__version__", "Unknown")
            except ImportError:
                version_info[dep] = "Not installed"

        return version_info

    @classmethod
    def validate_environment(cls) -> bool:
        """验证环境是否满足最低要求."""

        missing_required, _ = cls.check_dependencies()

        if missing_required:
            logger.error("环境验证失败: 缺失必需依赖 %s", missing_required)
            return False

        if sys.version_info < (3, 8):
            logger.error("Python版本过低: %s, 需要 >= 3.8", sys.version_info)
            return False

        logger.info("✅ 环境验证通过")
        return True
