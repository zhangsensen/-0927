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
        self.registry_file = registry_file or Path(
            "factor_system/factor_engine/data/registry.json"
        )
        self.factors: Dict[str, Type[BaseFactor]] = {}
        self.metadata: Dict[str, Dict] = {}
        self._factor_sets: Dict[str, Dict] = {}

        if self.registry_file.exists():
            self._load_registry()

    def _load_registry(self):
        """从文件加载注册表"""
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 加载因子元数据
            factors_data = data.get("factors", {})
            self.metadata = factors_data if isinstance(factors_data, dict) else {}

            # 加载因子集
            self._factor_sets = data.get("factor_sets", {})

            logger.info(
                f"加载因子注册表: {len(self.metadata)}个因子, {len(self._factor_sets)}个因子集"
            )
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
        支持enhanced_factor_calculator的所有因子格式

        Args:
            factor_class: 因子类
            metadata: 元数据（可选，从类属性自动提取）

        Raises:
            ValueError: 如果因子ID格式不正确
        """
        factor_id = factor_class.factor_id

        if not factor_id:
            raise ValueError(f"因子类必须定义factor_id: {factor_class}")

        # 验证因子ID格式（支持enhanced_factor_calculator的所有格式）
        self._validate_factor_id(factor_id)

        # 增强元数据提取，支持enhanced_factor_calculator的特殊格式
        enhanced_metadata = self._extract_enhanced_metadata(factor_class, factor_id)

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
            # 合并用户提供的元数据和增强元数据
            merged_metadata = {**enhanced_metadata, **metadata}
            self.metadata[factor_id] = merged_metadata
        elif factor_id not in self.metadata:
            # 使用增强元数据
            self.metadata[factor_id] = enhanced_metadata
        else:
            # 更新现有元数据
            self.metadata[factor_id].update(enhanced_metadata)

        logger.info(f"注册因子: {factor_id}")

    def _extract_enhanced_metadata(
        self, factor_class: Type[BaseFactor], factor_id: str
    ) -> Dict:
        """
        提取enhanced_factor_calculator兼容的元数据
        """
        base_metadata = {
            "factor_id": factor_id,
            "version": getattr(factor_class, "version", "v1.0"),
            "category": getattr(factor_class, "category", "unknown"),
            "description": getattr(factor_class, "description", f"{factor_id} 因子"),
            "status": "registered",
            "dependencies": [],
        }

        # 尝试从类属性中提取参数信息
        if hasattr(factor_class, "__doc__") and factor_class.__doc__:
            doc_text = factor_class.__doc__
            # 提取参数信息（如果有的话）
            import re

            params_match = re.search(r"参数[:\s]*\{([^}]+)\}", doc_text)
            if params_match:
                try:
                    params_str = params_match.group(1)
                    # 简单解析参数字符串
                    base_metadata["parameters"] = params_str
                except Exception:
                    pass  # 忽略解析错误

        # 为特殊因子添加额外元数据
        if factor_id.startswith("TA_"):
            base_metadata["library"] = "TA-Lib"
        elif factor_id.startswith("BB_"):
            base_metadata["indicator_type"] = "Bollinger Bands"
            if "Upper" in factor_id:
                base_metadata["component"] = "Upper Band"
            elif "Lower" in factor_id:
                base_metadata["component"] = "Lower Band"
            elif "Middle" in factor_id:
                base_metadata["component"] = "Middle Band"
            elif "Width" in factor_id:
                base_metadata["component"] = "Band Width"
        elif any(prefix in factor_id for prefix in ["MA", "EMA", "SMA", "WMA"]):
            base_metadata["indicator_type"] = "Moving Average"
        elif any(prefix in factor_id for prefix in ["RSI", "MACD", "STOCH", "WILLR"]):
            base_metadata["indicator_type"] = "Momentum Oscillator"

        return base_metadata

    def _validate_factor_id(self, factor_id: str) -> None:
        """验证因子ID格式 - 兼容enhanced_factor_calculator的所有因子"""
        # 基本合理性检查：不能为空
        if not factor_id or not factor_id.strip():
            raise ValueError("因子ID不能为空")

        # 不允许以下划线开头或结尾
        if factor_id.startswith("_") or factor_id.endswith("_"):
            raise ValueError(f"因子ID不能以下划线开头或结尾: {factor_id}")

        # 不允许连续多个下划线
        if "__" in factor_id:
            raise ValueError(f"因子ID不能包含连续下划线: {factor_id}")

        # 允许enhanced_factor_calculator的所有现有格式：
        # - 基本格式：RSI, MACD, STOCH
        # - 简单数字后缀：RSI14, ATR14
        # - 参数化格式：MACD_12_26_9, STOCH_14_20
        # - 布林带格式：BB_20_2_0_Upper, BB_20_2_0_Middle, BB_20_2_0_Lower
        # - 方向格式：TA_AROON_14_up, TA_AROON_14_down
        # - 长格式：TA_STOCHRSI_fastd_period3_fastk_period5_timeperiod14_K

        # 允许enhanced_factor_calculator的所有现有ID格式
        # 基本字符：字母、数字、下划线
        # 特殊情况：允许BB因子中的小数点（如BB_20_2_0_Upper中的2.0）
        if factor_id.startswith("BB_"):
            # 对布林带因子，允许小数点
            allowed_chars = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."
            )
        else:
            # 其他因子只允许基本字符
            allowed_chars = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
            )

        if not all(c in allowed_chars for c in factor_id):
            raise ValueError(f"因子ID只能包含字母、数字和下划线: {factor_id}")

        # 记录支持的格式（用于调试）
        supported_patterns = [
            "RSI, MACD, STOCH",  # 基本格式
            "RSI14, ATR14",  # 简单数字后缀
            "MACD_12_26_9, STOCH_14_20",  # 参数化格式
            "BB_20_2_0_Upper",  # 布林带格式
            "TA_AROON_14_up",  # 方向格式
        ]
        logger.debug(
            f"因子ID {factor_id} 格式验证通过，支持的格式: {supported_patterns}"
        )

    def get_factor(self, factor_id: str, **params) -> BaseFactor:
        """
        获取因子实例 - 支持参数化因子映射

        Args:
            factor_id: 标准因子ID或参数化变体
            **params: 因子参数

        Returns:
            因子实例

        Raises:
            ValueError: 如果因子未注册
        """
        # 🔧 优先使用精确匹配 - 支持完整的参数化因子ID
        if factor_id in self.factors:
            factor_class = self.factors[factor_id]
            return factor_class(**params)

        # 如果精确匹配失败，尝试解析参数化因子ID并映射到基础因子
        standard_id, parsed_params = self._parse_parameterized_factor_id(factor_id)
        if parsed_params and standard_id in self.factors:
            # 合并解析的参数和传入参数
            params = {**parsed_params, **params}
            factor_class = self.factors[standard_id]
            return factor_class(**params)

        # 检查metadata中的因子，动态加载
        if factor_id in self.metadata:
            return self._load_factor_from_metadata(factor_id, **params)

        available_factors = sorted(
            list(self.factors.keys()) + list(self.metadata.keys())
        )
        raise ValueError(
            f"未注册的因子: '{factor_id}'\n"
            f"可用因子: {available_factors[:20]}{'...' if len(available_factors) > 20 else ''}\n"
            f"请使用标准因子名（如RSI, MACD, STOCH）或检查factor_generation中的参数化变体"
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
        category = metadata.get("category", "technical")
        module_name = (
            f"factor_system.factor_engine.factors.{category}.{factor_id.lower()}"
        )

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

    def _parse_parameterized_factor_id(self, factor_id: str) -> Tuple[str, Dict]:
        """
        解析参数化因子ID，映射到标准因子和参数
        支持enhanced_factor_calculator的所有因子格式

        Args:
            factor_id: 参数化因子ID，如RSI14, MACD_12_26_9, BB_20_2_0_Upper

        Returns:
            (标准因子ID, 解析的参数字典) 的元组
        """
        import re

        # 特殊处理AROON方向因子
        if factor_id.startswith("TA_AROON_") and factor_id.endswith("_up"):
            # TA_AROON_14_up -> TA_AROON with period=14, direction='up'
            match = re.match(r"^TA_AROON_(\d+)_up$", factor_id)
            if match:
                return "TA_AROON", {
                    "timeperiod": int(match.group(1)),
                    "direction": "up",
                }

        if factor_id.startswith("TA_AROON_") and factor_id.endswith("_down"):
            # TA_AROON_14_down -> TA_AROON with period=14, direction='down'
            match = re.match(r"^TA_AROON_(\d+)_down$", factor_id)
            if match:
                return "TA_AROON", {
                    "timeperiod": int(match.group(1)),
                    "direction": "down",
                }

        # 特殊处理长格式STOCHRSI
        if factor_id.startswith("TA_STOCHRSI_"):
            # TA_STOCHRSI_fastd_period3_fastk_period5_timeperiod14_K
            match = re.match(
                r"^TA_STOCHRSI_fastd_period(\d+)_fastk_period(\d+)_timeperiod(\d+)_(K|D)$",
                factor_id,
            )
            if match:
                return "TA_STOCHRSI", {
                    "fastd_period": int(match.group(1)),
                    "fastk_period": int(match.group(2)),
                    "timeperiod": int(match.group(3)),
                    "component": match.group(4),
                }

        # 特殊处理ULTOSC
        if factor_id.startswith("TA_ULTOSC_"):
            # TA_ULTOSC_timeperiod17_timeperiod214_timeperiod328
            match = re.match(
                r"^TA_ULTOSC_timeperiod(\d+)_timeperiod(\d+)_timeperiod(\d+)$",
                factor_id,
            )
            if match:
                return "TA_ULTOSC", {
                    "timeperiod1": int(match.group(1)),
                    "timeperiod2": int(match.group(2)),
                    "timeperiod3": int(match.group(3)),
                }

        # 特殊处理APO
        if factor_id.startswith("TA_APO_"):
            # TA_APO_fastperiod12_matype0_slowperiod26
            match = re.match(
                r"^TA_APO_fastperiod(\d+)_matype(\d+)_slowperiod(\d+)$", factor_id
            )
            if match:
                return "TA_APO", {
                    "fastperiod": int(match.group(1)),
                    "matype": int(match.group(2)),
                    "slowperiod": int(match.group(3)),
                }

        # 特殊处理布林带变体
        if factor_id.startswith("BB_") and any(
            suffix in factor_id for suffix in ["_Upper", "_Middle", "_Lower", "_Width"]
        ):
            # BB_20_2_0_Upper -> BB with parameters
            match = re.match(
                r"^BB_(\d+)_(\d+)_(\d+)_(Upper|Middle|Lower|Width)$", factor_id
            )
            if match:
                return "BB", {
                    "timeperiod": int(match.group(1)),
                    "nbdevup": float(match.group(2)),
                    "nbdevdn": float(match.group(3)),
                    "component": match.group(4),
                }

        # 特殊处理AROONOSC
        if factor_id.startswith("TA_AROONOSC_"):
            # TA_AROONOSC_14
            match = re.match(r"^TA_AROONOSC_(\d+)$", factor_id)
            if match:
                return "TA_AROONOSC", {"timeperiod": int(match.group(1))}

        # 定义enhanced_factor_calculator兼容的参数化因子映射规则
        mapping_rules = {
            # 基本技术指标变体
            r"^RSI(\d+)$": ("RSI", {"timeperiod": int}),
            r"^WILLR(\d+)$": ("WILLR", {"timeperiod": int}),
            r"^CCI(\d+)$": ("CCI", {"timeperiod": int}),
            r"^ATR(\d+)$": ("ATR", {"timeperiod": int}),
            r"^MSTD(\d+)$": ("MSTD", {"timeperiod": int}),
            r"^FSTD(\d+)$": ("FSTD", {"timeperiod": int}),
            # MACD变体 - 使用下划线命名
            r"^MACD_(\d+)_(\d+)_(\d+)$": (
                "MACD",
                {"fast_period": int, "slow_period": int, "signal_period": int},
            ),
            # STOCH变体
            r"^STOCH_(\d+)_(\d+)$": (
                "STOCH",
                {"fastk_period": int, "slowk_period": int},
            ),
            # 移动平均变体
            r"^MA(\d+)$": ("MA", {"timeperiod": int}),
            r"^EMA(\d+)$": ("EMA", {"timeperiod": int}),
            r"^SMA(\d+)$": ("SMA", {"timeperiod": int}),
            # 修复因子变体
            r"^FIXLB(\d+)$": ("FIXLB", {"lookback": int}),
            r"^FMEAN(\d+)$": ("FMEAN", {"window": int}),
            r"^FMIN(\d+)$": ("FMIN", {"window": int}),
            r"^FMAX(\d+)$": ("FMAX", {"window": int}),
            r"^LEXLB(\d+)$": ("LEXLB", {"lookback": int}),
            r"^MEANLB(\d+)$": ("MEANLB", {"lookback": int}),
            r"^TRENDLB(\d+)$": ("TRENDLB", {"lookback": int}),
            # 动量因子变体
            r"^Momentum(\d+)$": ("Momentum", {"period": int}),
            r"^Position(\d+)$": ("Position", {"period": int}),
            r"^Trend(\d+)$": ("Trend", {"period": int}),
            # 成交量因子变体
            r"^Volume_Ratio(\d+)$": ("Volume_Ratio", {"period": int}),
            r"^Volume_Momentum(\d+)$": ("Volume_Momentum", {"period": int}),
            r"^VWAP(\d+)$": ("VWAP", {"period": int}),
            r"^OBV_SMA(\d+)$": ("OBV_SMA", {"period": int}),
            # TA-Lib标准因子变体（简化版）
            r"^TA_T3_(\d+)$": ("TA_T3", {"timeperiod": int}),
            r"^TA_MIDPRICE_(\d+)$": ("TA_MIDPRICE", {"timeperiod": int}),
            r"^TA_ADX_(\d+)$": ("TA_ADX", {"timeperiod": int}),
            r"^TA_ADXR_(\d+)$": ("TA_ADXR", {"timeperiod": int}),
            r"^TA_DX_(\d+)$": ("TA_DX", {"timeperiod": int}),
            r"^TA_MFI_(\d+)$": ("TA_MFI", {"timeperiod": int}),
            r"^TA_MOM_(\d+)$": ("TA_MOM", {"timeperiod": int}),
            r"^TA_ROC_(\d+)$": ("TA_ROC", {"timeperiod": int}),
            r"^TA_ROCP_(\d+)$": ("TA_ROCP", {"timeperiod": int}),
            r"^TA_ROCR_(\d+)$": ("TA_ROCR", {"timeperiod": int}),
            r"^TA_ROCR100_(\d+)$": ("TA_ROCR100", {"timeperiod": int}),
            r"^TA_RSI_(\d+)$": ("TA_RSI", {"timeperiod": int}),
            r"^TA_TRIX_(\d+)$": ("TA_TRIX", {"timeperiod": int}),
            r"^TA_WILLR_(\d+)$": ("TA_WILLR", {"timeperiod": int}),
            r"^TA_SAR$": ("TA_SAR", {}),
            r"^TA_CCI_(\d+)$": ("TA_CCI", {"timeperiod": int}),
            # TA-Lib STOCH变体
            r"^TA_STOCHF_(K|D)$": lambda m: (f"TA_STOCHF_{m.group(1)}", {}),
            r"^TA_STOCH_(K|D)$": lambda m: (f"TA_STOCH_{m.group(1)}", {}),
        }

        for pattern, rule in mapping_rules.items():
            match = re.match(pattern, factor_id)
            if match:
                if callable(rule):
                    # 处理lambda函数规则
                    return rule(match)
                else:
                    # 处理普通规则
                    base_id, param_types = rule
                    param_values = match.groups()
                    parsed_params = {}

                    # 根据参数类型映射参数名
                    param_names = list(param_types.keys())
                    for i, (name, converter) in enumerate(param_types.items()):
                        if i < len(param_values):
                            parsed_params[name] = converter(param_values[i])

                    return base_id, parsed_params

        # 如果没有匹配的规则，返回原始ID（保持向后兼容）
        return factor_id, {}

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

            if category and meta.get("category") != category:
                continue
            if status and meta.get("status") != status:
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
                deps = meta.get("dependencies", [])
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
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {
                    "metadata": {},
                    "factors": {},
                    "factor_sets": {},
                    "changelog": [],
                }

            # 更新factors部分
            data["factors"] = self.metadata

            # 写回文件
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_file, "w", encoding="utf-8") as f:
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

            factor_id = config.get("factor_id")
            if not factor_id:
                raise ValueError(f"因子配置缺少factor_id: {config}")

            parameters = config.get("parameters", {})
            if not isinstance(parameters, dict):
                raise ValueError(f"parameters必须是字典: {parameters}")

            # 验证因子是否存在（检查metadata和动态注册的factors）
            if factor_id not in self.factors and factor_id not in self.metadata:
                available_factors = sorted(
                    list(self.factors.keys()) + list(self.metadata.keys())
                )
                raise ValueError(
                    f"未注册的因子: '{factor_id}'\n"
                    f"可用因子: {available_factors[:20]}{'...' if len(available_factors) > 20 else ''}"
                )

            requests.append(FactorRequest(factor_id=factor_id, parameters=parameters))

        return requests


# 全局注册表实例
_global_registry: Optional[FactorRegistry] = None


def get_global_registry() -> FactorRegistry:
    """获取全局注册表"""
    global _global_registry
    if _global_registry is None:
        _global_registry = FactorRegistry()
    return _global_registry
