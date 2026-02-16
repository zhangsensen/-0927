"""
因子注册中心 | Factor Zoo Registry
================================================================================
Layer 1: 统一管理所有因子（手工 + 挖掘发现），提供注册、查询、持久化。

每个因子由 FactorEntry 描述，包含元数据、来源、质检报告引用。
通过 _seed_from_library() 可导入现有 PreciseFactorLibrary 的 25 个手工因子。
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FactorEntry:
    """因子注册条目"""

    name: str
    source: str  # 'hand_crafted' | 'algebraic' | 'window_variant' | 'transform'
    expression: str = ""  # 人类可读的公式描述
    parent_factors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None
    passed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FactorEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class FactorZoo:
    """
    因子动物园 — 注册中心

    职责:
    - 管理所有因子的元数据（不存储因子值）
    - 导入手工因子、注册挖掘因子
    - 记录质检状态
    - JSON 持久化
    """

    def __init__(self):
        self._registry: Dict[str, FactorEntry] = {}

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def seed_from_library(self, library) -> int:
        """
        从 PreciseFactorLibrary 导入手工因子。

        参数:
            library: PreciseFactorLibrary 实例

        返回:
            导入的因子数量
        """
        meta_dict = library.list_factors()
        count = 0
        for name, meta in sorted(meta_dict.items()):
            if name in self._registry:
                continue
            entry = FactorEntry(
                name=name,
                source="hand_crafted",
                expression=meta.description,
                parent_factors=[],
                metadata={
                    "dimension": meta.dimension,
                    "direction": meta.direction,
                    "bounded": meta.bounded,
                    "window": meta.window,
                    "production_ready": meta.production_ready,
                },
            )
            self._registry[name] = entry
            count += 1

        logger.info("Seeded %d hand-crafted factors from PreciseFactorLibrary", count)
        return count

    def register(self, entry: FactorEntry) -> None:
        """注册单个因子（重复注册抛异常）"""
        if entry.name in self._registry:
            raise ValueError(f"Factor '{entry.name}' already registered")
        self._registry[entry.name] = entry

    def register_batch(self, entries: List[FactorEntry]) -> int:
        """批量注册，跳过已存在的，返回新增数量"""
        count = 0
        for entry in entries:
            if entry.name not in self._registry:
                self._registry[entry.name] = entry
                count += 1
        return count

    def get(self, name: str) -> Optional[FactorEntry]:
        return self._registry.get(name)

    def list_all(self) -> List[FactorEntry]:
        return [self._registry[k] for k in sorted(self._registry)]

    def list_passed(self) -> List[FactorEntry]:
        """返回质检通过的因子"""
        return [
            e for k, e in sorted(self._registry.items()) if e.passed
        ]

    def list_by_source(self, source: str) -> List[FactorEntry]:
        return [
            e for k, e in sorted(self._registry.items()) if e.source == source
        ]

    def update_quality(self, name: str, score: float, passed: bool) -> None:
        """更新因子质检结果"""
        if name not in self._registry:
            raise KeyError(f"Factor '{name}' not in registry")
        self._registry[name].quality_score = score
        self._registry[name].passed = passed

    def export_registry(self, path: Path) -> None:
        """导出为 JSON"""
        path = Path(path)
        data = {name: entry.to_dict() for name, entry in sorted(self._registry.items())}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("Exported %d factors to %s", len(data), path)

    def import_registry(self, path: Path) -> int:
        """从 JSON 导入，返回新增数量"""
        path = Path(path)
        data = json.loads(path.read_text())
        count = 0
        for name, d in data.items():
            if name not in self._registry:
                self._registry[name] = FactorEntry.from_dict(d)
                count += 1
        logger.info("Imported %d new factors from %s", count, path)
        return count
