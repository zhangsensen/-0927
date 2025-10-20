#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未来函数防护组件 - 核心类
作者：量化首席工程师
版本：1.0.0
日期：2025-10-17

功能：
- 统一的未来函数防护入口
- 多种使用模式支持
- 完整的安全检查流程
- 综合报告生成
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd

from .config import GuardConfig
from .exceptions import ConfigurationError, FutureFunctionGuardError
from .health_monitor import HealthMonitor
from .runtime_validator import RuntimeValidator
from .static_checker import StaticChecker
from .utils import setup_logging


class FutureFunctionGuard:
    """
    未来函数防护组件核心类

    提供统一的未来函数防护功能，支持静态检查、运行时验证、健康监控等多层次保护。
    """

    def __init__(self, config: Optional[GuardConfig] = None):
        """
        初始化未来函数防护组件

        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        # 获取配置
        if config is None:
            self.config = GuardConfig.from_environment()
        else:
            self.config = config.copy()

        # 验证配置
        self.config.validate()

        # 设置日志
        self.logger = setup_logging(self.config.logging)

        # 初始化子模块
        self.static_checker = StaticChecker(self.config.static_check)
        self.runtime_validator = RuntimeValidator(self.config.runtime_validation)
        self.health_monitor = HealthMonitor(self.config.health_monitor)

        # 统计信息
        self.stats = {
            "init_time": datetime.now(),
            "static_checks": 0,
            "runtime_validations": 0,
            "health_checks": 0,
            "issues_detected": 0,
            "alerts_generated": 0,
        }

        self.logger.info(f"FutureFunctionGuard initialized in {self.config.mode} mode")
        self.logger.debug(f"Static check enabled: {self.config.static_check.enabled}")
        self.logger.debug(
            f"Runtime validation enabled: {self.config.runtime_validation.enabled}"
        )
        self.logger.debug(
            f"Health monitoring enabled: {self.config.health_monitor.enabled}"
        )

    # ==================== 静态检查方法 ====================

    def check_code_for_future_functions(
        self,
        target: Union[str, Path, List[Union[str, Path]]],
        recursive: bool = True,
        pattern: str = "*.py",
    ) -> Dict[str, Any]:
        """
        检查代码中的未来函数使用

        Args:
            target: 检查目标（文件、目录或文件列表）
            recursive: 是否递归检查子目录
            pattern: 文件匹配模式

        Returns:
            检查结果
        """
        if not self.config.enabled or not self.config.static_check.enabled:
            return {"status": "disabled", "message": "Static checking is disabled"}

        start_time = time.time()
        self.stats["static_checks"] += 1

        try:
            if isinstance(target, (list, tuple)):
                # 批量检查文件列表
                result = self.static_checker.check_files(target)
            elif isinstance(target, (str, Path)):
                target_path = Path(target)
                if target_path.is_file():
                    # 检查单个文件
                    result = self.static_checker.check_file(target_path)
                elif target_path.is_dir():
                    # 检查目录
                    result = self.static_checker.check_directory(
                        target_path, recursive, pattern
                    )
                else:
                    raise FutureFunctionGuardError(f"Invalid target: {target}")
            else:
                raise FutureFunctionGuardError(
                    f"Unsupported target type: {type(target)}"
                )

            # 更新统计
            scan_time = time.time() - start_time
            result["scan_time"] = scan_time
            self.stats["issues_detected"] += result.get("total_issues", 0)

            self.logger.info(
                f"Static check completed: {result.get('files_checked', 0)} files, "
                f"{result.get('total_issues', 0)} issues, {scan_time:.3f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Static check failed: {e}")
            raise FutureFunctionGuardError(f"Static check failed: {e}", cause=e) from e

    def generate_static_report(
        self,
        target: Union[str, Path, List[Union[str, Path]]],
        output_format: str = "text",
        save_to_file: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        生成静态检查报告

        Args:
            target: 检查目标
            output_format: 输出格式 (text, json, markdown)
            save_to_file: 保存到文件路径

        Returns:
            报告内容
        """
        check_result = self.check_code_for_future_functions(target)
        report = self.static_checker.generate_report(check_result, output_format)

        if save_to_file:
            save_path = Path(save_to_file)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report)
            self.logger.info(f"Static check report saved to: {save_path}")

        return report

    # ==================== 运行时验证方法 ====================

    def validate_time_series_data(
        self,
        data: pd.DataFrame,
        factor_ids: Optional[List[str]] = None,
        timeframe: str = "unknown",
    ) -> Dict[str, Any]:
        """
        验证时间序列数据

        Args:
            data: 时间序列数据
            factor_ids: 因子ID列表
            timeframe: 时间框架

        Returns:
            验证结果
        """
        if not self.config.enabled or not self.config.runtime_validation.enabled:
            return {"status": "disabled", "message": "Runtime validation is disabled"}

        start_time = time.time()
        self.stats["runtime_validations"] += 1

        try:
            if factor_ids is None:
                factor_ids = list(data.columns)

            result = self.runtime_validator.validate_batch_factors(
                data, factor_ids, timeframe
            )

            validation_time = time.time() - start_time
            result_dict = result.to_dict()
            result_dict["validation_time"] = validation_time

            # 更新统计
            if not result.is_valid:
                self.stats["issues_detected"] += 1

            self.logger.debug(
                f"Runtime validation completed: {len(factor_ids)} factors, "
                f"{'PASS' if result.is_valid else 'FAIL'}, {validation_time:.3f}s"
            )

            return result_dict

        except Exception as e:
            self.logger.error(f"Runtime validation failed: {e}")
            raise FutureFunctionGuardError(
                f"Runtime validation failed: {e}", cause=e
            ) from e

    def validate_factor_calculation(
        self,
        factor_data: pd.Series,
        factor_id: str,
        timeframe: str = "unknown",
        reference_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        验证单个因子计算

        Args:
            factor_data: 因子数据
            factor_id: 因子ID
            timeframe: 时间框架
            reference_data: 参考数据

        Returns:
            验证结果
        """
        if not self.config.enabled or not self.config.runtime_validation.enabled:
            return {"status": "disabled", "message": "Runtime validation is disabled"}

        start_time = time.time()
        self.stats["runtime_validations"] += 1

        try:
            result = self.runtime_validator.validate_factor_calculation(
                factor_data, factor_id, timeframe, reference_data
            )

            validation_time = time.time() - start_time
            result_dict = result.to_dict()
            result_dict["validation_time"] = validation_time

            # 更新统计
            if not result.is_valid:
                self.stats["issues_detected"] += 1

            self.logger.debug(
                f"Factor validation completed: {factor_id}, "
                f"{'PASS' if result.is_valid else 'FAIL'}, {validation_time:.3f}s"
            )

            return result_dict

        except Exception as e:
            self.logger.error(f"Factor validation failed: {e}")
            raise FutureFunctionGuardError(
                f"Factor validation failed: {e}", cause=e
            ) from e

    # ==================== 健康监控方法 ====================

    def check_factor_health(
        self, factor_data: pd.Series, factor_id: str, strict_mode: bool = False
    ) -> Dict[str, Any]:
        """
        检查因子健康状况

        Args:
            factor_data: 因子数据
            factor_id: 因子ID
            strict_mode: 严格模式

        Returns:
            健康检查结果
        """
        if not self.config.enabled or not self.config.health_monitor.enabled:
            return {"status": "disabled", "message": "Health monitoring is disabled"}

        start_time = time.time()
        self.stats["health_checks"] += 1

        try:
            health_metrics = self.health_monitor.check_factor_health(
                factor_data, factor_id, strict_mode
            )

            check_time = time.time() - start_time
            result = health_metrics.to_dict()
            result["check_time"] = check_time

            # 更新统计
            initial_alert_count = len(self.health_monitor.alerts)
            current_alert_count = len(self.health_monitor.alerts)
            new_alerts = current_alert_count - initial_alert_count
            self.stats["alerts_generated"] += new_alerts

            self.logger.debug(
                f"Health check completed: {factor_id}, "
                f"quality_score={health_metrics.get_quality_score():.1f}, "
                f"alerts={new_alerts}, {check_time:.3f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            raise FutureFunctionGuardError(f"Health check failed: {e}", cause=e) from e

    def check_batch_factors_health(
        self,
        factor_panel: pd.DataFrame,
        factor_ids: Optional[List[str]] = None,
        strict_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        批量检查因子健康状况

        Args:
            factor_panel: 因子面板数据
            factor_ids: 因子ID列表
            strict_mode: 严格模式

        Returns:
            批量健康检查结果
        """
        if not self.config.enabled or not self.config.health_monitor.enabled:
            return {"status": "disabled", "message": "Health monitoring is disabled"}

        start_time = time.time()
        self.stats["health_checks"] += 1

        try:
            health_results = self.health_monitor.check_batch_factors_health(
                factor_panel, factor_ids, strict_mode
            )

            check_time = time.time() - start_time

            # 更新统计
            initial_alert_count = len(self.health_monitor.alerts)
            current_alert_count = len(self.health_monitor.alerts)
            new_alerts = current_alert_count - initial_alert_count
            self.stats["alerts_generated"] += new_alerts

            # 转换结果为字典格式
            result = {
                "check_time": check_time,
                "factors_checked": len(health_results),
                "health_summary": self.health_monitor.get_health_summary(),
                "factor_results": {
                    factor_id: metrics.to_dict()
                    for factor_id, metrics in health_results.items()
                },
            }

            self.logger.info(
                f"Batch health check completed: {len(health_results)} factors, "
                f"alerts={new_alerts}, {check_time:.3f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Batch health check failed: {e}")
            raise FutureFunctionGuardError(
                f"Batch health check failed: {e}", cause=e
            ) from e

    # ==================== 综合安全检查方法 ====================

    def comprehensive_security_check(
        self,
        code_targets: Optional[List[Union[str, Path]]] = None,
        data_targets: Optional[Dict[str, pd.DataFrame]] = None,
        generate_report: bool = True,
    ) -> Dict[str, Any]:
        """
        综合安全检查

        Args:
            code_targets: 代码检查目标列表
            data_targets: 数据检查目标字典 {name: data}
            generate_report: 是否生成综合报告

        Returns:
            综合检查结果
        """
        start_time = time.time()
        results = {
            "start_time": datetime.now().isoformat(),
            "check_components": [],
            "overall_status": "passed",
            "total_issues": 0,
            "total_alerts": 0,
        }

        try:
            # 1. 静态代码检查
            if code_targets and self.config.static_check.enabled:
                self.logger.info("Starting static code analysis...")
                static_result = self.check_code_for_future_functions(code_targets)
                results["static_check"] = static_result
                results["check_components"].append("static_check")
                results["total_issues"] += static_result.get("total_issues", 0)

                if static_result.get("total_issues", 0) > 0:
                    results["overall_status"] = "warning"

            # 2. 运行时数据验证
            if data_targets and self.config.runtime_validation.enabled:
                self.logger.info("Starting runtime data validation...")
                runtime_results = {}
                for name, data in data_targets.items():
                    try:
                        if isinstance(data, pd.Series):
                            runtime_result = self.validate_factor_calculation(
                                data, name
                            )
                        else:
                            runtime_result = self.validate_time_series_data(data)
                        runtime_results[name] = runtime_result

                        if not runtime_result.get("is_valid", True):
                            results["overall_status"] = "failed"

                    except Exception as e:
                        self.logger.error(f"Runtime validation failed for {name}: {e}")
                        runtime_results[name] = {"status": "error", "error": str(e)}
                        results["overall_status"] = "failed"

                results["runtime_validation"] = runtime_results
                results["check_components"].append("runtime_validation")

            # 3. 健康监控检查
            if data_targets and self.config.health_monitor.enabled:
                self.logger.info("Starting health monitoring...")
                health_results = {}
                for name, data in data_targets.items():
                    try:
                        if isinstance(data, pd.Series):
                            health_result = self.check_factor_health(data, name)
                        else:
                            health_result = self.check_batch_factors_health(data)
                        health_results[name] = health_result

                    except Exception as e:
                        self.logger.error(f"Health check failed for {name}: {e}")
                        health_results[name] = {"status": "error", "error": str(e)}

                results["health_monitoring"] = health_results
                results["check_components"].append("health_monitoring")
                results["total_alerts"] = len(self.health_monitor.alerts)

                if results["total_alerts"] > 0:
                    if results["overall_status"] == "passed":
                        results["overall_status"] = "warning"

            # 完成检查
            total_time = time.time() - start_time
            results["end_time"] = datetime.now().isoformat()
            results["total_time"] = total_time
            results["statistics"] = self.get_statistics()

            self.logger.info(
                f"Comprehensive security check completed in {total_time:.3f}s, "
                f"status: {results['overall_status']}, "
                f"issues: {results['total_issues']}, "
                f"alerts: {results['total_alerts']}"
            )

            # 生成综合报告
            if generate_report:
                results["report"] = self.generate_comprehensive_report(results)

            return results

        except Exception as e:
            self.logger.error(f"Comprehensive security check failed: {e}")
            results["overall_status"] = "error"
            results["error"] = str(e)
            results["total_time"] = time.time() - start_time
            return results

    # ==================== 上下文管理器支持 ====================

    @contextmanager
    def protect(self, mode: str = "auto"):
        """
        上下文管理器：保护代码块

        Args:
            mode: 保护模式 (auto, strict, permissive)

        Yields:
            None
        """
        # 保存原始配置
        original_config = self.config.copy()

        try:
            # 根据模式调整配置
            if mode == "strict":
                self.config.runtime_validation.strict_mode = (
                    self.config.runtime_validation.strict_mode.ENFORCED
                )
                self.config.health_monitor.real_time_alerts = True
            elif mode == "permissive":
                self.config.runtime_validation.strict_mode = (
                    self.config.runtime_validation.strict_mode.WARN_ONLY
                )
                self.config.health_monitor.real_time_alerts = False

            self.logger.debug(f"Entered protection context in {mode} mode")
            yield

        finally:
            # 恢复原始配置
            self.config = original_config
            self.logger.debug(f"Exited protection context")

    # ==================== 报告生成方法 ====================

    def generate_comprehensive_report(
        self, check_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        生成综合安全报告

        Args:
            check_results: 检查结果（如果为None则生成当前状态报告）

        Returns:
            报告内容
        """
        if check_results is None:
            # 生成当前状态报告
            check_results = {
                "statistics": self.get_statistics(),
                "health_summary": self.health_monitor.get_health_summary(),
                "alert_count": len(self.health_monitor.alerts),
            }

        lines = [
            "Future Function Guard - 综合安全报告",
            "=" * 60,
            f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"运行模式: {self.config.mode}",
            f"整体状态: {check_results.get('overall_status', 'unknown')}",
            "",
        ]

        # 统计信息
        stats = check_results.get("statistics", {})
        lines.extend(
            [
                "使用统计:",
                f"- 静态检查次数: {stats.get('static_checks', 0)}",
                f"- 运行时验证次数: {stats.get('runtime_validations', 0)}",
                f"- 健康检查次数: {stats.get('health_checks', 0)}",
                f"- 检测问题总数: {stats.get('issues_detected', 0)}",
                f"- 生成报警总数: {stats.get('alerts_generated', 0)}",
                "",
            ]
        )

        # 静态检查结果
        if "static_check" in check_results:
            static = check_results["static_check"]
            lines.extend(
                [
                    "静态代码检查:",
                    f"- 检查文件数: {static.get('files_checked', 0)}",
                    f"- 发现问题数: {static.get('total_issues', 0)}",
                    f"- 检查耗时: {static.get('scan_time', 0):.3f}秒",
                    "",
                ]
            )

        # 运行时验证结果
        if "runtime_validation" in check_results:
            runtime = check_results["runtime_validation"]
            passed_count = sum(1 for r in runtime.values() if r.get("is_valid", True))
            total_count = len(runtime)
            lines.extend(
                [
                    "运行时数据验证:",
                    f"- 验证数据集数: {total_count}",
                    f"- 通过验证数: {passed_count}",
                    f"- 验证失败数: {total_count - passed_count}",
                    "",
                ]
            )

        # 健康监控结果
        if "health_monitoring" in check_results:
            health = check_results["health_monitoring"]
            lines.extend(
                [
                    "健康监控:",
                    f"- 监控数据集数: {len(health)}",
                    f"- 活跃报警数: {len(self.health_monitor.alerts)}",
                    "",
                ]
            )

        # 最近报警
        recent_alerts = (
            self.health_monitor.alerts[-5:] if self.health_monitor.alerts else []
        )
        if recent_alerts:
            lines.append("最近报警:")
            for alert in recent_alerts:
                severity_icon = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}.get(
                    alert.severity, "⚠️"
                )
                lines.append(
                    f"{severity_icon} {alert.timestamp.strftime('%H:%M:%S')} - {alert.message}"
                )
        else:
            lines.append("✅ 无活跃报警")

        return "\n".join(lines)

    def export_data(
        self, file_path: Union[str, Path], include_alerts: bool = True
    ) -> None:
        """
        导出防护数据到文件

        Args:
            file_path: 导出文件路径
            include_alerts: 是否包含报警数据
        """
        export_path = Path(file_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 准备导出数据
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "config": self.config.to_dict(),
                "statistics": self.get_statistics(),
                "health_summary": self.health_monitor.get_health_summary(),
            }

            if include_alerts:
                export_data["alerts"] = [
                    alert.to_dict() for alert in self.health_monitor.alerts
                ]

            # 保存到文件
            import json

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"Guard data exported to: {export_path}")

        except Exception as e:
            raise FutureFunctionGuardError(f"Failed to export data: {e}") from e

    # ==================== 统计和管理方法 ====================

    def get_statistics(self) -> Dict[str, Any]:
        """获取使用统计信息"""
        current_time = datetime.now()
        uptime = current_time - self.stats["init_time"]

        return {
            **self.stats,
            "uptime_seconds": uptime.total_seconds(),
            "uptime_human": str(uptime).split(".")[0],
            "current_time": current_time.isoformat(),
            "cache_info": {
                "static_check": self.static_checker.get_cache_info(),
                "health_monitor": self.health_monitor.cache.get_size_info(),
            },
        }

    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.stats = {
            "init_time": datetime.now(),
            "static_checks": 0,
            "runtime_validations": 0,
            "health_checks": 0,
            "issues_detected": 0,
            "alerts_generated": 0,
        }
        self.logger.info("Statistics reset")

    def clear_caches(self) -> None:
        """清空所有缓存"""
        self.static_checker.clear_cache()
        self.health_monitor.cache.clear()
        self.logger.info("All caches cleared")

    def clear_alerts(self, older_than_days: Optional[int] = None) -> int:
        """清理报警记录"""
        count = self.health_monitor.clear_alerts(older_than_days)
        self.logger.info(f"Cleared {count} alerts")
        return count

    def update_config(self, new_config: GuardConfig) -> None:
        """
        更新配置

        Args:
            new_config: 新配置
        """
        new_config.validate()
        self.config = new_config

        # 重新初始化子模块
        self.static_checker = StaticChecker(self.config.static_check)
        self.runtime_validator = RuntimeValidator(self.config.runtime_validation)
        self.health_monitor = HealthMonitor(self.config.health_monitor)

        self.logger.info("Configuration updated")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if exc_type is not None:
            self.logger.error(f"Exception in guard context: {exc_val}")
        return False  # 不抑制异常

    def __str__(self) -> str:
        """字符串表示"""
        return f"FutureFunctionGuard(mode={self.config.mode}, enabled={self.config.enabled})"

    def __repr__(self) -> str:
        """详细字符串表示"""
        return (
            f"FutureFunctionGuard(mode={self.config.mode}, "
            f"enabled={self.config.enabled}, "
            f"static_checks={self.stats['static_checks']}, "
            f"runtime_validations={self.stats['runtime_validations']}, "
            f"health_checks={self.stats['health_checks']})"
        )
