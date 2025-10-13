# -*- coding: utf-8 -*-
\"\"\"依赖检查工具\"\"\"

import importlib
import logging
import sys
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DependencyChecker:
    \"\"\"检查依赖是否可用\"\"\"

    REQUIRED_DEPS = [\"pandas\", \"numpy\", \"scipy\", \"vectorbt\", \"talib\", \"sklearn\"]

    OPTIONAL_DEPS = {
        \"performance\": [\"polars\", \"numba\"],
        \"visualization\": [\"matplotlib\", \"plotly\", \"seaborn\"],
        \"web\": [\"dash\"],
        \"database\": [\"sqlalchemy\", \"redis\"],
        \"scheduling\": [\"schedule\"],
    }

    @classmethod
    def check_dependencies(cls) -> Tuple[List[str], Dict[str, List[str]]]:
        \"\"\"
        检查依赖状态

        Returns:
            Tuple[List[str], Dict[str, List[str]]]: (缺失的必需依赖, 各功能模块的缺失依赖)
        \"\"\"
        missing_required = []
        missing_optional = {}

        # 检查必需依赖
        for dep in cls.REQUIRED_DEPS:
            try:
                # 处理特殊的模块名映射
                module_name = dep
                if dep == \"talib\":
                    module_name = \"talib\"
                elif dep == \"sklearn\":
                    module_name = \"sklearn\"

                importlib.import_module(module_name)
                logger.debug(f\"✅ {dep} 已安装\")
            except ImportError:
                missing_required.append(dep)
                logger.warning(f\"❌ {dep} 未安装\")

        # 检查可选依赖
        for category, deps in cls.OPTIONAL_DEPS.items():
            missing = []
            for dep in deps:
                try:
                    importlib.import_module(dep.replace(\"-\", \"_\"))
                    logger.debug(f\"✅ {dep} ({category}) 已安装\")
                except ImportError:
                    missing.append(dep)
                    logger.debug(f\"⚠️ {dep} ({category}) 未安装\")
            if missing:
                missing_optional[category] = missing

        return missing_required, missing_optional

    @classmethod
    def print_dependency_status(cls):
        \"\"\"打印依赖状态\"\"\"
        missing_required, missing_optional = cls.check_dependencies()

        print(\"=== 依赖状态检查 ===\")

        if missing_required:
            print(f\"❌ 缺失必需依赖: {', '.join(missing_required)}\")
            print(\"请运行: pip install \" + \" \".join(missing_required))
        else:
            print(\"✅ 所有必需依赖已安装\")

        if missing_optional:
            print(\"\\n⚠️  可选依赖状态:\")
            for category, deps in missing_optional.items():
                print(f\"  {category}: 缺失 {', '.join(deps)}\")
                print(f\"  安装命令: pip install {' '.join(deps)}\")
        else:
            print(\"✅ 所有可选依赖已安装\")

    @classmethod
    def get_installation_commands(cls) -> Dict[str, str]:
        \"\"\"
        获取安装命令

        Returns:
            Dict[str, str]: 各类依赖的安装命令
        \"\"\"
        missing_required, missing_optional = cls.check_dependencies()

        commands = {}

        if missing_required:
            commands[\"required\"] = f\"pip install {' '.join(missing_required)}\"

        for category, deps in missing_optional.items():
            if deps:
                commands[category] = f\"pip install {' '.join(deps)}\"

        return commands

    @classmethod
    def check_version_compatibility(cls) -> Dict[str, str]:
        \"\"\"
        检查版本兼容性

        Returns:
            Dict[str, str]: 模块版本信息
        \"\"\"
        version_info = {}

        for dep in cls.REQUIRED_DEPS:
            try:
                module = importlib.import_module(dep.replace(\"-\", \"_\"))
                version = getattr(module, \"__version__\", \"Unknown\")
                version_info[dep] = version
            except ImportError:
                version_info[dep] = \"Not installed\"

        return version_info

    @classmethod
    def validate_environment(cls) -> bool:
        \"\"\"
        验证环境是否满足最低要求

        Returns:
            bool: 环境是否有效
        \"\"\"
        missing_required, _ = cls.check_dependencies()

        if missing_required:
            logger.error(f\"环境验证失败: 缺失必需依赖 {missing_required}\")
            return False

        # 检查Python版本
        if sys.version_info < (3, 8):
            logger.error(f\"Python版本过低: {sys.version_info}, 需要 >= 3.8\")
            return False

        logger.info(\"✅ 环境验证通过\")
        return True