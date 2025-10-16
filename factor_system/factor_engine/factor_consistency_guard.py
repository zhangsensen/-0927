#!/usr/bin/env python3
"""
因子一致性保护机制
确保FactorEngine严格继承factor_generation的所有因子，防止不一致修改
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class FactorSnapshot:
    """因子快照数据结构"""

    factors: List[str]
    source_hash: str
    timestamp: str
    source_file: str
    line_count: int


class FactorConsistencyGuard:
    """因子一致性守护器"""

    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path(__file__).parent.parent.parent
        self.snapshot_file = self.root_dir / ".factor_consistency_snapshot.json"
        self.lock_file = self.root_dir / ".factor_consistency_lock"

    def scan_factor_generation_factors(self) -> Dict[str, FactorSnapshot]:
        """扫描factor_generation中的所有因子"""
        logger.info("🔍 扫描factor_generation中的因子...")

        factors = {}
        gen_dir = self.root_dir / "factor_system" / "factor_generation"

        # 扫描enhanced_factor_calculator.py中的因子
        enhanced_file = gen_dir / "enhanced_factor_calculator.py"
        if enhanced_file.exists():
            factors.update(
                self._extract_factors_from_file(
                    enhanced_file, "enhanced_factor_calculator.py"
                )
            )

        # 扫描factor_generation_factors_list.txt中的因子清单
        factors_list_file = gen_dir.parent / "factor_generation_factors_list.txt"
        if factors_list_file.exists():
            factors.update(self._extract_factors_from_list_file(factors_list_file))

        # 扫描FACTOR_REGISTRY.md中的完整因子清单
        registry_file = gen_dir.parent / "FACTOR_REGISTRY.md"
        if registry_file.exists():
            factors.update(self._extract_factors_from_registry(registry_file))

        logger.info(f"✅ 从factor_generation发现 {len(factors)} 个因子源")
        return factors

    def scan_factor_engine_factors(self) -> Dict[str, FactorSnapshot]:
        """扫描FactorEngine中的因子"""
        logger.info("🔍 扫描FactorEngine中的因子...")

        factors = {}
        engine_dir = self.root_dir / "factor_system" / "factor_engine"

        # 扫描factors目录下的所有因子文件
        factors_dir = engine_dir / "factors"
        if factors_dir.exists():
            for factor_file in factors_dir.rglob("*.py"):
                if factor_file.name != "__init__.py":
                    factors.update(
                        self._extract_factors_from_file(
                            factor_file, str(factor_file.relative_to(self.root_dir))
                        )
                    )

        logger.info(f"✅ 从FactorEngine发现 {len(factors)} 个因子实现")
        return factors

    def _extract_factors_from_file(
        self, file_path: Path, relative_name: str
    ) -> Dict[str, FactorSnapshot]:
        """从Python文件中提取因子信息"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 计算文件哈希
            file_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

            # 提取因子类
            factors = {}
            lines = content.split("\n")

            # 扩展的关键因子模式，覆盖更多因子类型
            key_patterns = [
                "MACD",
                "RSI",
                "STOCH",
                "WILLR",
                "CCI",
                "ATR",
                "ADX",
                "MFI",
                "OBV",
                "SMA",
                "EMA",
                "WMA",
                "DEMA",
                "TEMA",
                "BBANDS",
                "SAR",
                "KAMA",
                "TRIMA",
                "T3",
                "ROC",
                "MOM",
                "TRIX",
                "ULTOSC",
                "APO",
                "PPO",
                "CMO",
                "DX",
                "MINUS_DM",
                "PLUS_DM",
                "MINUS_DI",
                "PLUS_DI",
                "ADXR",
                "AROON",
                "AROONOSC",
                "NATR",
                "TRANGE",
                "AD",
                "ADOSC",
                "BOP",
                "ROCP",
                "ROCR",
                "ROCR100",
                "STOCHRSI",
                "STOCHF",
            ]

            for i, line in enumerate(lines):
                for pattern in key_patterns:
                    if pattern in line and ("class" in line or "def" in line):
                        # 提取因子名称
                        factor_name = self._extract_factor_name(line, pattern)
                        if factor_name:
                            factors[factor_name] = FactorSnapshot(
                                factors=[factor_name],
                                source_hash=file_hash,
                                timestamp=str(file_path.stat().st_mtime),
                                source_file=relative_name,
                                line_count=len(lines),
                            )
                        break

            return factors

        except Exception as e:
            logger.error(f"❌ 读取文件失败 {file_path}: {e}")
            return {}

    def _extract_factors_from_list_file(
        self, file_path: Path
    ) -> Dict[str, FactorSnapshot]:
        """从factor_generation_factors_list.txt中提取因子"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 计算文件哈希
            file_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

            factors = {}
            lines = content.split("\n")

            for line in lines:
                line = line.strip()
                # 跳过注释和空行
                if line.startswith("#") or not line or ":" in line:
                    continue

                # 提取因子名（去掉参数部分，如RSI14 -> RSI）
                factor_name = self._extract_base_factor_name(line)
                if factor_name:
                    factors[factor_name] = FactorSnapshot(
                        factors=[factor_name],
                        source_hash=file_hash,
                        timestamp=str(file_path.stat().st_mtime),
                        source_file="factor_generation_factors_list.txt",
                        line_count=len(lines),
                    )

            return factors

        except Exception as e:
            logger.error(f"❌ 读取因子清单失败 {file_path}: {e}")
            return {}

    def _extract_factors_from_registry(
        self, file_path: Path
    ) -> Dict[str, FactorSnapshot]:
        """从FACTOR_REGISTRY.md中提取因子"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 计算文件哈希
            file_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

            factors = {}
            lines = content.split("\n")

            for line in lines:
                line = line.strip()
                # 查找因子名行（以- 开头的列表项）
                if line.startswith("- `") and line.endswith("`"):
                    # 提取因子名，如- `RSI` -> RSI
                    factor_name = line[3:-1].split("`")[0]
                    factor_name = self._extract_base_factor_name(factor_name)

                    if factor_name:
                        factors[factor_name] = FactorSnapshot(
                            factors=[factor_name],
                            source_hash=file_hash,
                            timestamp=str(file_path.stat().st_mtime),
                            source_file="FACTOR_REGISTRY.md",
                            line_count=len(lines),
                        )

            return factors

        except Exception as e:
            logger.error(f"❌ 读取注册表失败 {file_path}: {e}")
            return {}

    def _extract_base_factor_name(self, factor_name: str) -> str:
        """提取基础因子名，去掉参数后缀"""
        import re

        # 处理参数化因子名，如RSI14 -> RSI, MACD_12_26_9 -> MACD
        if re.match(r"^[A-Z]+[a-z]*\d+$", factor_name):
            # RSI14 -> RSI
            return re.sub(r"\d+$", "", factor_name)
        elif "_" in factor_name:
            # MACD_12_26_9 -> MACD
            return factor_name.split("_")[0]
        else:
            return factor_name

    def _extract_factor_name(self, line: str, pattern: str) -> Optional[str]:
        """从代码行中提取因子名称"""
        line = line.strip()

        # 类定义
        if line.startswith("class "):
            parts = line.split("(")[0].split()
            if len(parts) >= 2:
                class_name = parts[1]
                if pattern in class_name:
                    return class_name

        # 函数定义
        elif line.startswith("def "):
            parts = line.split("(")[0].split()
            if len(parts) >= 2:
                func_name = parts[1]
                if pattern in func_name:
                    return func_name

        return None

    def create_baseline_snapshot(self) -> bool:
        """创建基准快照"""
        logger.info("📸 创建因子一致性基准快照...")

        # 获取factor_generation中的因子（基准）
        gen_factors = self.scan_factor_generation_factors()

        # 获取FactorEngine中的因子（当前状态）
        engine_factors = self.scan_factor_engine_factors()

        # 创建快照
        snapshot = {
            "baseline": {
                name: asdict(snapshot) for name, snapshot in gen_factors.items()
            },
            "current": {
                name: asdict(snapshot) for name, snapshot in engine_factors.items()
            },
            "metadata": {
                "baseline_count": len(gen_factors),
                "current_count": len(engine_factors),
                "consistency_check": (
                    "PASS"
                    if self._check_consistency(gen_factors, engine_factors)
                    else "FAIL"
                ),
            },
        }

        # 保存快照
        try:
            with open(self.snapshot_file, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ 基准快照已保存: {self.snapshot_file}")
            logger.info(f"   基准因子数: {len(gen_factors)}")
            logger.info(f"   当前因子数: {len(engine_factors)}")

            return True

        except Exception as e:
            logger.error(f"❌ 保存快照失败: {e}")
            return False

    def _check_consistency(
        self, baseline: Dict[str, FactorSnapshot], current: Dict[str, FactorSnapshot]
    ) -> bool:
        """检查一致性"""
        baseline_factors = set(baseline.keys())
        current_factors = set(current.keys())

        # 检查是否所有基准因子都存在
        missing_factors = baseline_factors - current_factors
        extra_factors = current_factors - baseline_factors

        if missing_factors:
            logger.warning(f"❌ 缺失因子: {missing_factors}")

        if extra_factors:
            logger.warning(f"⚠️  多余因子: {extra_factors}")

        return len(missing_factors) == 0

    def validate_consistency(self) -> bool:
        """验证一致性"""
        logger.info("🔒 验证因子一致性...")

        if not self.snapshot_file.exists():
            logger.error("❌ 未找到基准快照，请先运行 create_baseline_snapshot()")
            return False

        try:
            with open(self.snapshot_file, "r", encoding="utf-8") as f:
                snapshot = json.load(f)

            baseline_factors = set(snapshot["baseline"].keys())

            # 获取当前FactorEngine状态
            current_factors = self.scan_factor_engine_factors()
            current_factor_names = set(current_factors.keys())

            # 检查一致性
            missing = baseline_factors - current_factor_names
            extra = current_factor_names - baseline_factors

            is_consistent = len(missing) == 0

            if is_consistent:
                logger.info("✅ 因子一致性验证通过")
                if extra:
                    logger.warning(f"⚠️  发现多余因子: {extra}")
            else:
                logger.error(f"❌ 因子一致性验证失败")
                logger.error(f"   缺失因子: {missing}")
                if extra:
                    logger.error(f"   多余因子: {extra}")

            return is_consistent

        except Exception as e:
            logger.error(f"❌ 验证过程失败: {e}")
            return False

    def enforce_consistency(self) -> bool:
        """强制执行一致性（修复FactorEngine）"""
        logger.info("⚡ 强制执行因子一致性...")

        if not self.snapshot_file.exists():
            logger.error("❌ 未找到基准快照")
            return False

        # 读取基准快照
        with open(self.snapshot_file, "r", encoding="utf-8") as f:
            snapshot = json.load(f)

        baseline_factors = set(snapshot["baseline"].keys())

        # 获取当前状态
        current_factors = self.scan_factor_engine_factors()
        current_factor_names = set(current_factors.keys())

        # 需要删除的额外因子
        extra_factors = current_factor_names - baseline_factors

        if extra_factors:
            logger.warning(f"🗑️  将删除多余因子: {extra_factors}")

            # 这里可以添加自动删除额外因子的逻辑
            # 但为了安全起见，我们只报告，不自动删除
            logger.error("❌ 发现不一致，请手动删除多余因子")
            return False

        logger.info("✅ FactorEngine已与factor_generation保持一致")
        return True

    def generate_report(self) -> Dict:
        """生成一致性报告"""
        logger.info("📊 生成因子一致性报告...")

        # 获取当前状态
        gen_factors = self.scan_factor_generation_factors()
        engine_factors = self.scan_factor_engine_factors()

        gen_factor_names = set(gen_factors.keys())
        engine_factor_names = set(engine_factors.keys())

        report = {
            "timestamp": str(Path().cwd()),
            "factor_generation": {
                "source": "factor_system/factor_generation",
                "factor_count": len(gen_factor_names),
                "factors": sorted(list(gen_factor_names)),
            },
            "factor_engine": {
                "source": "factor_system/factor_engine",
                "factor_count": len(engine_factor_names),
                "factors": sorted(list(engine_factor_names)),
            },
            "consistency_analysis": {
                "missing_in_engine": sorted(
                    list(gen_factor_names - engine_factor_names)
                ),
                "extra_in_engine": sorted(list(engine_factor_names - gen_factor_names)),
                "common_factors": sorted(list(gen_factor_names & engine_factor_names)),
                "is_consistent": len(gen_factor_names - engine_factor_names) == 0,
            },
        }

        return report


def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO)

    guard = FactorConsistencyGuard()

    print("🔒 因子一致性保护机制")
    print("=" * 50)

    # 生成报告
    report = guard.generate_report()

    print(f"📊 factor_generation因子数: {report['factor_generation']['factor_count']}")
    print(f"📊 FactorEngine因子数: {report['factor_engine']['factor_count']}")
    print(f"📊 共同因子数: {len(report['consistency_analysis']['common_factors'])}")

    if report["consistency_analysis"]["missing_in_engine"]:
        print(
            f"❌ FactorEngine缺失: {report['consistency_analysis']['missing_in_engine']}"
        )

    if report["consistency_analysis"]["extra_in_engine"]:
        print(
            f"⚠️  FactorEngine多余: {report['consistency_analysis']['extra_in_engine']}"
        )

    if report["consistency_analysis"]["is_consistent"]:
        print("✅ 因子一致性验证通过")
    else:
        print("❌ 因子一致性验证失败")
        print("\n🔧 建议操作:")
        print("1. 运行: python factor_consistency_guard.py create-baseline")
        print("2. 运行: python factor_consistency_guard.py enforce")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        guard = FactorConsistencyGuard()

        if command == "create-baseline":
            guard.create_baseline_snapshot()
        elif command == "validate":
            guard.validate_consistency()
        elif command == "enforce":
            guard.enforce_consistency()
        elif command == "report":
            report = guard.generate_report()
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print("可用命令: create-baseline, validate, enforce, report")
    else:
        main()
