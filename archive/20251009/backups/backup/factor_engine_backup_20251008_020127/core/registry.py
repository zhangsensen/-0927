"""因子注册表管理"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

from factor_system.factor_engine.core.base_factor import BaseFactor, FactorMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactorRequest:
    """标准化的因子请求"""

    factor_id: str
    parameters: Dict

    def cache_key(self) -> str:
        """生成缓存键"""
        if not self.parameters:
            return self.factor_id
        params_key = json.dumps(self.parameters, sort_keys=True)
        return f"{self.factor_id}|{params_key}"


class FactorRegistry:
    """
    因子注册表

    职责:
    - 管理因子元数据
    - 注册和发现因子
    - 版本管理

    严格要求:
    - 所有因子必须使用标准命名（无参数后缀）
    - 参数通过字典传递，不嵌入在因子名中
    - 不支持别名解析
    """

    def __init__(self, registry_file: Optional[Path] = None):
        """
        初始化注册表

        Args:
            registry_file: factor_registry.json路径
        """
        self.registry_file = registry_file or Path("factor_system/factor_engine/data/registry.json")
        self.factors: Dict[str, Type[BaseFactor]] = {}
        self.metadata: Dict[str, Dict] = {}
        self._factor_sets: Dict[str, Dict] = {}

        if self.registry_file.exists():
            self._load_registry()

    def _load_registry(self):
        """从文件加载注册表"""
        try:
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 加载因子元数据
            factors_data = data.get('factors', {})
            self.metadata = factors_data if isinstance(factors_data, dict) else {}

            # 加载因子集
            self._factor_sets = data.get('factor_sets', {})

            logger.info(f"加载因子注册表: {len(self.metadata)}个因子, {len(self._factor_sets)}个因子集")
        except FileNotFoundError:
            logger.warning(f"注册表文件不存在: {self.registry_file}")
            self.metadata = {}
            self._factor_sets = {}
        except Exception as e:
            logger.error(f"加载注册表失败: {e}")
            self.metadata = {}
            self._factor_sets = {}
    
    def register(
        self,
        factor_class: Type[BaseFactor],
        metadata: Optional[Dict] = None,
    ):
        """
        注册因子

        Args:
            factor_class: 因子类
            metadata: 元数据（可选，从类属性自动提取）

        Raises:
            ValueError: 如果因子ID格式不正确
        """
        factor_id = factor_class.factor_id

        if not factor_id:
            raise ValueError(f"因子类必须定义factor_id: {factor_class}")

        # 验证因子ID格式（不能包含参数）
        self._validate_factor_id(factor_id)

        # 检查重复注册
        if factor_id in self.factors:
            existing_class = self.factors[factor_id]
            if existing_class != factor_class:
                logger.warning(f"因子 {factor_id} 已存在，将被覆盖")
            else:
                logger.debug(f"因子 {factor_id} 已存在，跳过注册")
                return

        self.factors[factor_id] = factor_class

        # 更新或创建元数据
        if metadata:
            self.metadata[factor_id] = metadata
        elif factor_id not in self.metadata:
            # 从类属性自动提取元数据
            self.metadata[factor_id] = {
                'factor_id': factor_id,
                'version': getattr(factor_class, 'version', 'v1.0'),
                'category': getattr(factor_class, 'category', 'unknown'),
                'description': getattr(factor_class, 'description', ''),
                'status': 'registered',
                'dependencies': [],
            }

        logger.info(f"注册因子: {factor_id}")

    def _validate_factor_id(self, factor_id: str) -> None:
        """验证因子ID格式"""
        # 不允许包含参数后缀
        if '_' in factor_id and any(c.isdigit() for c in factor_id):
            raise ValueError(f"因子ID不能包含参数后缀: {factor_id}")

        # 不允许TA_前缀
        if factor_id.startswith('TA_'):
            raise ValueError(f"因子ID不能使用TA_前缀: {factor_id}")

        # 检查格式是否合理
        if not factor_id.replace('_', '').isalnum():
            raise ValueError(f"因子ID只能包含字母、数字和下划线: {factor_id}")

    def get_factor(self, factor_id: str, **params) -> BaseFactor:
        """
        获取因子实例

        Args:
            factor_id: 标准因子ID
            **params: 因子参数

        Returns:
            因子实例

        Raises:
            ValueError: 如果因子未注册
        """
        # 首先检查动态注册的因子类
        if factor_id in self.factors:
            factor_class = self.factors[factor_id]
            return factor_class(**params)

        # 然后检查metadata中的因子，动态加载
        if factor_id in self.metadata:
            return self._load_factor_from_metadata(factor_id, **params)

        available_factors = sorted(list(self.factors.keys()) + list(self.metadata.keys()))
        raise ValueError(
            f"未注册的因子: '{factor_id}'\n"
            f"可用因子: {available_factors[:20]}{'...' if len(available_factors) > 20 else ''}\n"
            f"请确保使用标准因子名，参数通过**params传递"
        )

    def _load_factor_from_metadata(self, factor_id: str, **params) -> BaseFactor:
        """
        从元数据动态加载因子类

        Args:
            factor_id: 因子ID
            **params: 因子参数

        Returns:
            因子实例

        Raises:
            ValueError: 如果因子加载失败
        """
        metadata = self.metadata.get(factor_id)
        if not metadata:
            raise ValueError(f"因子 {factor_id} 的元数据不存在")

        # 构建因子类导入路径
        category = metadata.get('category', 'technical')
        module_name = f"factor_system.factor_engine.factors.{category}.{factor_id.lower()}"

        try:
            # 动态导入模块
            import importlib
            module = importlib.import_module(module_name)

            # 获取因子类（假设类名与因子ID相同）
            factor_class = getattr(module, factor_id)

            # 创建实例并缓存
            factor_instance = factor_class(**params)
            self.factors[factor_id] = factor_class  # 缓存类以供后续使用

            logger.info(f"动态加载因子: {factor_id} ({module_name})")
            return factor_instance

        except ImportError as e:
            raise ValueError(f"无法导入因子模块 {module_name}: {e}")
        except AttributeError as e:
            raise ValueError(f"因子类 {factor_id} 在模块 {module_name} 中不存在: {e}")
        except Exception as e:
            raise ValueError(f"加载因子 {factor_id} 失败: {e}")

    def list_factors(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[str]:
        """
        列出已注册因子

        Args:
            category: 过滤分类
            status: 过滤状态

        Returns:
            因子ID列表
        """
        result = []

        # 只列出已注册的因子
        for factor_id in sorted(self.factors.keys()):
            meta = self.metadata.get(factor_id, {})

            if category and meta.get('category') != category:
                continue
            if status and meta.get('status') != status:
                continue

            result.append(factor_id)

        return result
    
    def get_metadata(self, factor_id: str) -> Optional[Dict]:
        """获取因子元数据"""
        return self.metadata.get(factor_id)
    
    def get_dependencies(self, factor_id: str) -> List[str]:
        """
        获取因子依赖（递归）
        
        Args:
            factor_id: 因子ID
        
        Returns:
            包含所有递归依赖的因子ID列表
        """
        all_deps = set()
        visited = set()
        
        def _recursive_deps(fid: str):
            if fid in visited:
                return
            visited.add(fid)
            
            meta = self.get_metadata(fid)
            if meta:
                deps = meta.get('dependencies', [])
                for dep in deps:
                    all_deps.add(dep)
                    _recursive_deps(dep)
        
        _recursive_deps(factor_id)
        return list(all_deps)
    
    def get_factor_set(self, set_id: str) -> Optional[Dict]:
        """获取因子集定义"""
        return self._factor_sets.get(set_id)
    
    def list_factor_sets(self) -> List[str]:
        """列出所有因子集ID"""
        return list(self._factor_sets.keys())
    
    def save_registry(self):
        """保存注册表到文件"""
        try:
            # 读取现有数据
            if self.registry_file.exists():
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {'metadata': {}, 'factors': {}, 'factor_sets': {}, 'changelog': []}

            # 更新factors部分
            data['factors'] = self.metadata

            # 写回文件
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"保存因子注册表: {len(self.metadata)}个因子")
        except Exception as e:
            logger.error(f"保存注册表失败: {e}")

    def create_factor_requests(self, factor_configs: List[Dict]) -> List[FactorRequest]:
        """
        创建标准化的因子请求

        Args:
            factor_configs: 因子配置列表，格式: [{'factor_id': 'RSI', 'parameters': {'timeperiod': 14}}, ...]

        Returns:
            标准化的因子请求列表

        Raises:
            ValueError: 如果因子未注册或配置格式错误
        """
        requests = []

        for config in factor_configs:
            if not isinstance(config, dict):
                raise ValueError(f"因子配置必须是字典: {config}")

            factor_id = config.get('factor_id')
            if not factor_id:
                raise ValueError(f"因子配置缺少factor_id: {config}")

            parameters = config.get('parameters', {})
            if not isinstance(parameters, dict):
                raise ValueError(f"parameters必须是字典: {parameters}")

            # 验证因子是否存在（检查metadata和动态注册的factors）
            if factor_id not in self.factors and factor_id not in self.metadata:
                available_factors = sorted(list(self.factors.keys()) + list(self.metadata.keys()))
                raise ValueError(
                    f"未注册的因子: '{factor_id}'\n"
                    f"可用因子: {available_factors[:20]}{'...' if len(available_factors) > 20 else ''}"
                )

            requests.append(FactorRequest(
                factor_id=factor_id,
                parameters=parameters
            ))

        return requests


# 全局注册表实例
_global_registry: Optional[FactorRegistry] = None


def get_global_registry() -> FactorRegistry:
    """获取全局注册表"""
    global _global_registry
    if _global_registry is None:
        _global_registry = FactorRegistry()
    return _global_registry
