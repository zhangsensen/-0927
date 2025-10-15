"""
因子注册表模块 - 动态因子类映射
避免大型硬编码字典，提供动态因子注册和查询功能
"""

import logging
from typing import Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class FactorRegistry:
    """因子注册表类 - 管理因子ID到类的映射"""

    def __init__(self):
        self._factor_map: Dict[str, Type] = {}
        self._initialized = False

    def register_factor(self, factor_id: str, factor_class: Type) -> None:
        """注册单个因子

        Args:
            factor_id: 因子ID
            factor_class: 因子类
        """
        if factor_id not in self._factor_map:
            self._factor_map[factor_id] = factor_class
            logger.debug(f"注册因子: {factor_id} -> {factor_class.__name__}")

    def register_factors(self, factor_classes: List[Type]) -> None:
        """批量注册因子类

        Args:
            factor_classes: 因子类列表
        """
        for factor_class in factor_classes:
            factor_id = factor_class.__name__
            self.register_factor(factor_id, factor_class)

    def get_factor_class(self, factor_id: str) -> Optional[Type]:
        """获取因子类

        Args:
            factor_id: 因子ID

        Returns:
            因子类或None
        """
        return self._factor_map.get(factor_id)

    def list_factors(self) -> List[str]:
        """列出所有已注册的因子ID

        Returns:
            因子ID列表
        """
        return list(self._factor_map.keys())

    def get_factor_map(self) -> Dict[str, Type]:
        """获取完整的因子映射字典

        Returns:
            因子ID到类的映射字典
        """
        return self._factor_map.copy()

    def is_registered(self, factor_id: str) -> bool:
        """检查因子是否已注册

        Args:
            factor_id: 因子ID

        Returns:
            是否已注册
        """
        return factor_id in self._factor_map

    def auto_register_from_module(self, module_name: str) -> None:
        """从模块自动注册所有因子类

        Args:
            module_name: 模块名
        """
        try:
            import sys

            module = sys.modules.get(module_name)
            if module:
                for name in dir(module):
                    obj = getattr(module, name)
                    if (
                        isinstance(obj, type)
                        and hasattr(obj, "calculate")
                        and not name.startswith("_")
                    ):
                        self.register_factor(name, obj)
                        logger.debug(f"自动注册因子: {name}")
        except Exception as e:
            logger.warning(f"从模块 {module_name} 自动注册因子失败: {e}")


# 全局因子注册表实例
_registry = FactorRegistry()


def initialize_factor_registry():
    """初始化因子注册表"""
    global _registry

    if _registry._initialized:
        return

    try:
        # 从当前模块自动注册所有因子
        import sys

        current_module = sys.modules[__name__.replace(".factor_registry", "")]

        for name in dir(current_module):
            obj = getattr(current_module, name)
            if (
                isinstance(obj, type)
                and hasattr(obj, "calculate")
                and not name.startswith("_")
            ):
                _registry.register_factor(name, obj)

        _registry._initialized = True
        logger.info(f"因子注册表初始化完成，注册了 {len(_registry._factor_map)} 个因子")

    except Exception as e:
        logger.error(f"因子注册表初始化失败: {e}")


def get_factor_registry() -> FactorRegistry:
    """获取全局因子注册表实例

    Returns:
        因子注册表实例
    """
    if not _registry._initialized:
        initialize_factor_registry()
    return _registry


def get_factor_class(factor_id: str) -> Optional[Type]:
    """获取因子类（便捷函数）

    Args:
        factor_id: 因子ID

    Returns:
        因子类或None
    """
    return get_factor_registry().get_factor_class(factor_id)


def get_all_factors() -> List[str]:
    """获取所有因子ID列表（便捷函数）

    Returns:
        因子ID列表
    """
    return get_factor_registry().list_factors()


def get_factor_class_map() -> Dict[str, Type]:
    """获取因子映射字典（便捷函数）

    Returns:
        因子ID到类的映射字典
    """
    return get_factor_registry().get_factor_map()
