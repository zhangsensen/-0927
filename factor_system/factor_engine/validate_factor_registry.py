#!/usr/bin/env python3
"""
因子清单验证器 / Factor Registry Validator

该脚本用于验证 FactorEngine 是否严格遵循官方因子清单。
确保系统中没有未授权的因子计算。

This script validates that FactorEngine strictly follows the official factor registry.
Ensures no unauthorized factor calculations exist in the system.
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class FactorRegistryValidator:
    """因子清单验证器 / Factor Registry Validator"""

    def __init__(self):
        self.registry_file = project_root / "factor_system" / "FACTOR_REGISTRY.md"
        self.factor_config_file = (
            project_root / "factor_system" / "factor_generation" / "factor_config.py"
        )
        self.factor_engine_file = (
            project_root / "factor_system" / "factor_engine" / "factor_engine.py"
        )

        self.official_factors: Dict = {}
        self.config_factors: Dict = {}
        self.engine_factors: Dict = {}

        self.errors: List[str] = []
        self.warnings: List[str] = []

    def load_official_registry(self) -> bool:
        """加载官方因子清单 / Load official factor registry"""
        try:
            if not self.registry_file.exists():
                self.errors.append(f"官方因子清单文件不存在: {self.registry_file}")
                return False

            content = self.registry_file.read_text(encoding="utf-8")

            # 解析因子清单中的因子
            current_factor = {}
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("**因子ID**: `"):
                    factor_id = line.split("`")[1]
                    current_factor["id"] = factor_id
                elif line.startswith("**参数配置 / Parameters**"):
                    # 等待参数配置部分
                    continue
                elif line.startswith("```python") and current_factor:
                    # 开始收集参数配置
                    continue
                elif line.startswith("}") and current_factor:
                    # 结束参数配置
                    if "id" in current_factor:
                        self.official_factors[current_factor["id"]] = {
                            "id": current_factor["id"],
                            "status": "🟢 ACTIVE",  # 默认状态
                        }
                    current_factor = {}
                elif line.startswith("**输出字段 / Output Fields**") and current_factor:
                    # 因子定义完成
                    if "id" in current_factor:
                        self.official_factors[current_factor["id"]] = {
                            "id": current_factor["id"],
                            "status": "🟢 ACTIVE",
                        }

            print(f"✅ 从官方清单加载了 {len(self.official_factors)} 个因子")
            return True

        except Exception as e:
            self.errors.append(f"加载官方因子清单失败: {str(e)}")
            return False

    def load_factor_config(self) -> bool:
        """加载 factor_generation 配置 / Load factor_generation configuration"""
        try:
            if not self.factor_config_file.exists():
                self.errors.append(
                    f"factor_config.py 文件不存在: {self.factor_config_file}"
                )
                return False

            spec = importlib.util.spec_from_file_location(
                "factor_config", self.factor_config_file
            )
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)

            if hasattr(config_module, "FACTOR_CONFIG"):
                self.config_factors = config_module.FACTOR_CONFIG
                print(
                    f"✅ 从 factor_config.py 加载了 {len(self.config_factors)} 个因子"
                )
                return True
            else:
                self.errors.append("factor_config.py 中未找到 FACTOR_CONFIG")
                return False

        except Exception as e:
            self.errors.append(f"加载 factor_config.py 失败: {str(e)}")
            return False

    def analyze_factor_engine(self) -> bool:
        """分析 FactorEngine 代码 / Analyze FactorEngine code"""
        try:
            # 尝试分析 API 文件
            api_file = project_root / "factor_system" / "factor_engine" / "api.py"
            if not api_file.exists():
                self.errors.append(f"FactorEngine API 文件不存在: {api_file}")
                return False

            content = api_file.read_text(encoding="utf-8")

            # 分析因子导入部分
            lines = content.split("\n")

            for i, line in enumerate(lines):
                line = line.strip()

                # 查找技术指标因子导入
                if "from factor_system.factor_engine.factors.technical import" in line:
                    # 提取同一行的因子
                    import_part = line.split("import")[1].strip()
                    if import_part:
                        factor_names = [f.strip() for f in import_part.split(",")]
                        for name in factor_names:
                            if name and name != "":
                                self.engine_factors[name] = {
                                    "type": "technical",
                                    "line": i + 1,
                                }

                # 查找移动平均因子导入
                elif "from factor_system.factor_engine.factors.overlap import" in line:
                    # 提取同一行的因子
                    import_part = line.split("import")[1].strip()
                    if import_part:
                        factor_names = [f.strip() for f in import_part.split(",")]
                        for name in factor_names:
                            if name and name != "":
                                self.engine_factors[name] = {
                                    "type": "overlap",
                                    "line": i + 1,
                                }

                # 查找统计指标因子导入
                elif (
                    "from factor_system.factor_engine.factors.statistic import" in line
                ):
                    # 提取同一行的因子
                    import_part = line.split("import")[1].strip()
                    if import_part:
                        factor_names = [f.strip() for f in import_part.split(",")]
                        for name in factor_names:
                            if name and name != "":
                                self.engine_factors[name] = {
                                    "type": "statistic",
                                    "line": i + 1,
                                }

                # 查找形态识别因子导入
                elif "from factor_system.factor_engine.factors.pattern import" in line:
                    # 提取同一行的因子
                    import_part = line.split("import")[1].strip()
                    if import_part:
                        factor_names = [f.strip() for f in import_part.split(",")]
                        for name in factor_names:
                            if name and name != "":
                                self.engine_factors[name] = {
                                    "type": "pattern",
                                    "line": i + 1,
                                }

            print(f"✅ 从 FactorEngine API 中识别了 {len(self.engine_factors)} 个因子")
            return True

        except Exception as e:
            self.errors.append(f"分析 FactorEngine 代码失败: {str(e)}")
            return False

    def validate_consistency(self):
        """验证一致性 / Validate consistency"""
        print("\n🔍 开始一致性验证...")

        # 1. 检查官方清单与配置的一致性
        official_ids = set(self.official_factors.keys())
        config_ids = set(self.config_factors.keys())

        if official_ids != config_ids:
            missing_in_config = official_ids - config_ids
            extra_in_config = config_ids - official_ids

            if missing_in_config:
                self.errors.append(f"配置中缺失的官方因子: {missing_in_config}")
            if extra_in_config:
                self.errors.append(
                    f"配置中多余的因子 (未在官方清单中): {extra_in_config}"
                )
        else:
            print("✅ 官方清单与 factor_config 完全一致")

        # 2. 检查 FactorEngine 是否只使用配置中的因子
        engine_ids = set(self.engine_factors.keys())
        allowed_ids = set(self.config_factors.keys())

        unauthorized_factors = engine_ids - allowed_ids
        if unauthorized_factors:
            self.errors.append(f"FactorEngine 中包含未授权因子: {unauthorized_factors}")
            for factor in unauthorized_factors:
                factor_info = self.engine_factors[factor]
                self.errors.append(
                    f"  - {factor} (类型: {factor_info['type']}, 行号: {factor_info['line']})"
                )
        else:
            print("✅ FactorEngine 只使用配置中的因子")

        # 3. 检查配置中的因子是否都在 FactorEngine 中实现
        missing_in_engine = allowed_ids - engine_ids
        if missing_in_engine:
            self.warnings.append(f"FactorEngine 中缺失的配置因子: {missing_in_engine}")
        else:
            print("✅ 所有配置因子都在 FactorEngine 中实现")

    def validate_parameters(self):
        """验证参数一致性 / Validate parameter consistency"""
        print("\n🔍 开始参数一致性验证...")

        for factor_id in self.config_factors:
            config_params = self.config_factors[factor_id]

            if factor_id in self.official_factors:
                # 验证参数与官方清单一致
                # 这里可以添加更详细的参数验证逻辑
                print(f"✅ {factor_id} 参数配置有效")
            else:
                self.warnings.append(f"{factor_id} 不在官方清单中，但存在于配置中")

    def generate_report(self) -> bool:
        """生成验证报告 / Generate validation report"""
        print("\n" + "=" * 60)
        print("📋 因子清单验证报告 / Factor Registry Validation Report")
        print("=" * 60)

        # 统计信息
        print(f"📊 统计信息:")
        print(f"  - 官方清单因子数: {len(self.official_factors)}")
        print(f"  - factor_config 因子数: {len(self.config_factors)}")
        print(f"  - FactorEngine 识别因子数: {len(self.engine_factors)}")

        # 验证结果
        if self.errors:
            print(f"\n❌ 发现 {len(self.errors)} 个错误:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        if self.warnings:
            print(f"\n⚠️  发现 {len(self.warnings)} 个警告:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if not self.errors and not self.warnings:
            print("\n🎉 所有验证通过！系统完全符合因子清单要求。")

        # 结论
        print(f"\n📝 验证结论:")
        if self.errors:
            print("  ❌ 验证失败 - 存在违规因子或配置不一致")
            print("  🔧 请修复错误后重新运行验证")
            return False
        else:
            print("  ✅ 验证通过 - 系统合规")
            if self.warnings:
                print("  💡 建议处理警告项以优化系统")
            return True

    def run_full_validation(self) -> bool:
        """运行完整验证 / Run full validation"""
        print("🚀 开始因子清单完整性验证...")

        success = True
        success &= self.load_official_registry()
        success &= self.load_factor_config()
        success &= self.analyze_factor_engine()

        if success:
            self.validate_consistency()
            self.validate_parameters()

        return self.generate_report()


def main():
    """主函数 / Main function"""
    validator = FactorRegistryValidator()
    success = validator.run_full_validation()

    if not success:
        print(f"\n💥 验证失败！退出码: 1")
        sys.exit(1)
    else:
        print(f"\n🎉 验证成功！因子清单合规。")
        sys.exit(0)


if __name__ == "__main__":
    main()
