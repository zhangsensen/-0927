#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未来函数防护组件 - 静态检查模块
作者：量化首席工程师
版本：1.0.0
日期：2025-10-17

功能：
- 基于AST的未来函数静态检查
- 正则表达式快速模式匹配
- 缓存优化的文件扫描
- 详细的检查报告生成
"""

from __future__ import annotations

import ast
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .config import StaticCheckConfig
from .exceptions import FutureFunctionDetectedError, StaticCheckError
from .utils import FileCache, batch_processing, get_file_hash


class FutureFunctionVisitor(ast.NodeVisitor):
    """AST访问器，检测未来函数使用"""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.issues: List[Dict[str, Any]] = []
        self.current_function: Optional[str] = None
        self.current_class: Optional[str] = None
        self.imports: Set[str] = set()
        self.future_function_patterns: Set[str] = {
            "shift",
            "lead",
            "future",
            "ahead",
            "lookahead",
            "lead_",
            "future_",
        }

    def visit_Import(self, node: ast.Import) -> None:
        """记录导入的模块"""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """记录从模块导入的内容"""
        if node.module:
            for alias in node.names:
                self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """访问函数定义"""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """访问异步函数定义"""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """访问类定义"""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Call(self, node: ast.Call) -> None:
        """检查函数调用"""
        self._check_function_call(node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """检查属性访问"""
        self._check_attribute_access(node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """检查变量名"""
        var_name = node.id.lower()

        # 检查前缀匹配（如 future_xxx, lead_xxx）
        if var_name.startswith("future_") or var_name.startswith("lead_"):
            self._add_issue(
                node.lineno,
                "future_variable_name",
                f"发现可疑的未来函数变量名: {var_name}",
                variable_name=var_name,
                variable_type="name",
            )

        # 检查变量名是否包含未来函数模式
        elif any(pattern in var_name for pattern in ["future", "lead", "ahead"]):
            self._add_issue(
                node.lineno,
                "future_variable_name",
                f"发现包含未来函数模式的变量名: {var_name}",
                variable_name=var_name,
                variable_type="name",
            )

        self.generic_visit(node)

    def _check_function_call(self, node: ast.Call) -> None:
        """检查函数调用中的未来函数"""
        # 检查直接调用未来函数
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
            if func_name in self.future_function_patterns:
                self._add_issue(
                    node.lineno,
                    "future_function_call",
                    f"发现未来函数调用: {func_name}()",
                    function_name=func_name,
                    call_type="direct",
                )

        # 检查方法调用中的参数
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr.lower()
            if method_name == "shift" and node.args:
                self._check_shift_arguments(node, method_name)

    def _check_attribute_access(self, node: ast.Attribute) -> None:
        """检查属性访问中的未来函数"""
        attr_name = node.attr.lower()

        # 检查完整匹配
        if attr_name in self.future_function_patterns:
            self._add_issue(
                node.lineno,
                "future_function_attribute",
                f"发现未来函数属性: .{attr_name}",
                attribute_name=attr_name,
                access_type="attribute",
            )

        # 检查前缀匹配（如 future_xxx, lead_xxx）
        elif attr_name.startswith("future_") or attr_name.startswith("lead_"):
            self._add_issue(
                node.lineno,
                "future_function_attribute",
                f"发现可疑的未来函数变量: {attr_name}",
                attribute_name=attr_name,
                access_type="attribute",
            )

        # 检查变量名是否包含未来函数模式
        elif any(pattern in attr_name for pattern in ["future", "lead", "ahead"]):
            self._add_issue(
                node.lineno,
                "future_function_attribute",
                f"发现包含未来函数模式的变量: {attr_name}",
                attribute_name=attr_name,
                access_type="attribute",
            )

    def _check_shift_arguments(self, node: ast.Call, method_name: str) -> None:
        """检查shift函数的参数"""
        if not node.args:
            return

        # 检查第一个参数是否为负数
        first_arg = node.args[0]
        if isinstance(first_arg, ast.UnaryOp) and isinstance(first_arg.op, ast.USub):
            # 对于负数，如 -1，operand.value 会是 1（正数），但因为是负数操作，所以整体是负数
            if isinstance(first_arg.operand, ast.Constant):
                value = first_arg.operand.value
                if (
                    isinstance(value, int) and value >= 0
                ):  # 注意：对于 -1，value 是 1，但整体表示 -1
                    self._add_issue(
                        node.lineno,
                        "negative_shift",
                        f"发现负数shift: shift(-{value})",
                        function_name=method_name,
                        parameters=f"-{value}",
                        severity="high",
                    )
            elif isinstance(first_arg.operand, ast.Num):  # Python < 3.8
                value = first_arg.operand.n
                if isinstance(value, int) and value >= 0:  # 同上逻辑
                    self._add_issue(
                        node.lineno,
                        "negative_shift",
                        f"发现负数shift: shift(-{value})",
                        function_name=method_name,
                        parameters=f"-{value}",
                        severity="high",
                    )

        # 检查变量参数（可能包含负数）
        for arg in node.args[:1]:  # 只检查第一个参数
            if isinstance(arg, ast.Name):
                self._add_issue(
                    node.lineno,
                    "variable_shift",
                    f"shift使用变量参数，可能包含负数: shift({arg.id})",
                    function_name=method_name,
                    parameters=arg.id,
                    severity="medium",
                )

    def _add_issue(
        self,
        line: int,
        issue_type: str,
        message: str,
        severity: str = "medium",
        **kwargs,
    ) -> None:
        """添加问题记录"""
        issue = {
            "file_path": str(self.file_path),
            "line": line,
            "column": 0,  # AST不提供列信息
            "issue_type": issue_type,
            "message": message,
            "severity": severity,
            "function": self.current_function,
            "class": self.current_class,
            "imports": list(self.imports),
            **kwargs,
        }
        self.issues.append(issue)


class StaticChecker:
    """静态检查器"""

    def __init__(self, config: StaticCheckConfig):
        self.config = config
        self.cache = FileCache(
            cache_dir=Path.home() / ".future_function_guard_cache" / "static_check",
            config=type(
                "obj",
                (object,),
                {"ttl_hours": config.cache_ttl_hours, "compression_enabled": True},
            )(),
        )
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> List[re.Pattern]:
        """预编译正则表达式模式"""
        patterns = []
        for pattern_str in self.config.check_patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                patterns.append(pattern)
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{pattern_str}': {e}")
        return patterns

    def _should_exclude_file(self, file_path: Path) -> bool:
        """检查文件是否应该被排除"""
        file_str = str(file_path)
        for pattern in self.config.exclude_patterns:
            try:
                if re.search(pattern, file_str):
                    return True
            except re.error:
                pass
        return False

    def _check_file_size(self, file_path: Path) -> bool:
        """检查文件大小是否在限制范围内"""
        if not file_path.exists():
            return False

        size_mb = file_path.stat().st_size / (1024 * 1024)
        return size_mb <= self.config.max_file_size_mb

    def _get_cache_key(self, file_path: Path) -> str:
        """生成缓存键"""
        try:
            file_hash = get_file_hash(file_path)
            return f"static_check:{file_path}:{file_hash}"
        except Exception:
            # 如果无法计算哈希，使用修改时间
            mtime = file_path.stat().st_mtime
            return f"static_check:{file_path}:{mtime}"

    def _regex_scan(self, file_path: Path) -> List[Dict[str, Any]]:
        """使用正则表达式快速扫描"""
        issues = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    content = f.read()
            except Exception:
                return []

        lines = content.split("\n")

        for pattern in self._compiled_patterns:
            for line_num, line in enumerate(lines, 1):
                matches = pattern.finditer(line)
                for match in matches:
                    issues.append(
                        {
                            "file_path": str(file_path),
                            "line": line_num,
                            "column": match.start(),
                            "issue_type": "regex_match",
                            "message": f"发现可疑模式: {match.group()}",
                            "severity": "medium",
                            "matched_text": match.group(),
                            "pattern": pattern.pattern,
                        }
                    )

        return issues

    def _ast_scan(self, file_path: Path) -> List[Dict[str, Any]]:
        """使用AST深度扫描"""
        issues = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    content = f.read()
            except Exception:
                return []

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            issues.append(
                {
                    "file_path": str(file_path),
                    "line": e.lineno or 0,
                    "column": e.offset or 0,
                    "issue_type": "syntax_error",
                    "message": f"语法错误: {e.msg}",
                    "severity": "low",
                    "syntax_error": str(e),
                }
            )
            return issues

        visitor = FutureFunctionVisitor(file_path)
        visitor.visit(tree)
        return visitor.issues

    def check_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        检查单个文件

        Args:
            file_path: 文件路径

        Returns:
            检查结果
        """
        file_path = Path(file_path)

        # 基本检查
        if not file_path.exists():
            raise StaticCheckError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise StaticCheckError(f"Path is not a file: {file_path}")

        if not file_path.suffix == ".py":
            raise StaticCheckError(f"Not a Python file: {file_path}")

        if self._should_exclude_file(file_path):
            return {
                "file_path": str(file_path),
                "status": "excluded",
                "issues": [],
                "scan_time": 0.0,
                "message": "File excluded by pattern",
            }

        if not self._check_file_size(file_path):
            raise StaticCheckError(
                f"File too large: {file_path} (max: {self.config.max_file_size_mb}MB)",
                file_path=str(file_path),
            )

        # 缓存检查
        cache_key = self._get_cache_key(file_path)
        if self.config.cache_enabled:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                cached_result["from_cache"] = True
                return cached_result

        # 执行检查
        start_time = time.time()

        try:
            # 正则扫描
            regex_issues = (
                self._regex_scan(file_path) if self._compiled_patterns else []
            )

            # AST扫描
            ast_issues = self._ast_scan(file_path)

            # 合并结果
            all_issues = regex_issues + ast_issues

            # 去重
            unique_issues = self._deduplicate_issues(all_issues)

            scan_time = time.time() - start_time

            result = {
                "file_path": str(file_path),
                "status": "completed",
                "issues": unique_issues,
                "scan_time": scan_time,
                "issue_count": len(unique_issues),
                "severity_counts": self._count_by_severity(unique_issues),
                "from_cache": False,
            }

            # 缓存结果
            if self.config.cache_enabled:
                try:
                    self.cache.set(cache_key, result)
                except Exception:
                    pass  # 缓存失败不影响检查结果

            return result

        except Exception as e:
            scan_time = time.time() - start_time
            raise StaticCheckError(
                f"Failed to scan file: {e}", file_path=str(file_path), cause=e
            ) from e

    def check_directory(
        self, directory: Union[str, Path], recursive: bool = True, pattern: str = "*.py"
    ) -> Dict[str, Any]:
        """
        检查目录中的Python文件

        Args:
            directory: 目录路径
            recursive: 是否递归检查子目录
            pattern: 文件匹配模式

        Returns:
            检查结果汇总
        """
        directory = Path(directory)

        if not directory.exists():
            raise StaticCheckError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise StaticCheckError(f"Path is not a directory: {directory}")

        # 查找Python文件
        if recursive:
            python_files = list(directory.rglob(pattern))
        else:
            python_files = list(directory.glob(pattern))

        # 过滤排除的文件
        python_files = [f for f in python_files if not self._should_exclude_file(f)]

        if not python_files:
            return {
                "directory": str(directory),
                "status": "no_files",
                "files_checked": 0,
                "total_issues": 0,
                "total_scan_time": 0.0,
                "results": [],
            }

        # 批量检查
        start_time = time.time()
        all_results = []
        total_issues = 0

        for file_path in python_files:
            try:
                result = self.check_file(file_path)
                all_results.append(result)
                total_issues += result["issue_count"]
            except StaticCheckError as e:
                # 记录失败的文件，但继续检查其他文件
                all_results.append(
                    {
                        "file_path": str(file_path),
                        "status": "error",
                        "error": str(e),
                        "issues": [],
                        "scan_time": 0.0,
                        "issue_count": 0,
                    }
                )

        total_scan_time = time.time() - start_time

        return {
            "directory": str(directory),
            "status": "completed",
            "files_checked": len(python_files),
            "total_issues": total_issues,
            "total_scan_time": total_scan_time,
            "average_scan_time": (
                total_scan_time / len(python_files) if python_files else 0
            ),
            "results": all_results,
            "severity_counts": self._aggregate_severity_counts(all_results),
        }

    def check_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        批量检查文件列表

        Args:
            file_paths: 文件路径列表

        Returns:
            检查结果汇总
        """
        if not file_paths:
            return {
                "status": "no_files",
                "files_checked": 0,
                "total_issues": 0,
                "total_scan_time": 0.0,
                "results": [],
            }

        start_time = time.time()
        all_results = []
        total_issues = 0

        for file_path in file_paths:
            try:
                result = self.check_file(file_path)
                all_results.append(result)
                total_issues += result["issue_count"]
            except StaticCheckError as e:
                all_results.append(
                    {
                        "file_path": str(file_path),
                        "status": "error",
                        "error": str(e),
                        "issues": [],
                        "scan_time": 0.0,
                        "issue_count": 0,
                    }
                )

        total_scan_time = time.time() - start_time

        return {
            "status": "completed",
            "files_checked": len(file_paths),
            "total_issues": total_issues,
            "total_scan_time": total_scan_time,
            "average_scan_time": total_scan_time / len(file_paths),
            "results": all_results,
            "severity_counts": self._aggregate_severity_counts(all_results),
        }

    def _deduplicate_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去除重复的问题"""
        seen = set()
        unique_issues = []

        for issue in issues:
            # 创建唯一标识
            key = (
                issue["file_path"],
                issue["line"],
                issue["issue_type"],
                issue.get("matched_text", ""),
            )
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)

        return unique_issues

    def _count_by_severity(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """按严重程度统计问题数量"""
        counts = {"high": 0, "medium": 0, "low": 0}
        for issue in issues:
            severity = issue.get("severity", "medium")
            if severity in counts:
                counts[severity] += 1
        return counts

    def _aggregate_severity_counts(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """聚合多个文件的严重程度统计"""
        total_counts = {"high": 0, "medium": 0, "low": 0}
        for result in results:
            if "severity_counts" in result:
                for severity, count in result["severity_counts"].items():
                    total_counts[severity] += count
        return total_counts

    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return self.cache.get_size_info()

    def generate_report(
        self, results: Dict[str, Any], output_format: str = "text"
    ) -> str:
        """
        生成检查报告

        Args:
            results: 检查结果
            output_format: 输出格式 (text, json, markdown)

        Returns:
            格式化的报告字符串
        """
        if output_format == "json":
            import json

            return json.dumps(results, indent=2, ensure_ascii=False, default=str)

        elif output_format == "markdown":
            return self._generate_markdown_report(results)

        else:  # text
            return self._generate_text_report(results)

    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """生成文本格式报告"""
        lines = []

        if "directory" in results:
            # 目录扫描报告
            lines.append(f"静态检查报告 - 目录: {results['directory']}")
        else:
            # 文件扫描报告
            lines.append(f"静态检查报告")

        lines.append("=" * 60)
        lines.append(f"检查文件数: {results['files_checked']}")
        lines.append(f"发现问题数: {results['total_issues']}")
        lines.append(f"总耗时: {results['total_scan_time']:.3f}秒")

        if results["total_issues"] > 0:
            severity_counts = results.get("severity_counts", {})
            lines.append(
                f"严重程度分布: 高({severity_counts.get('high', 0)}) "
                f"中({severity_counts.get('medium', 0)}) "
                f"低({severity_counts.get('low', 0)})"
            )

            lines.append("\n问题详情:")
            lines.append("-" * 40)

            for result in results.get("results", []):
                if result.get("status") == "error":
                    lines.append(
                        f"\n❌ {result['file_path']}: {result.get('error', 'Unknown error')}"
                    )
                    continue

                issues = result.get("issues", [])
                if issues:
                    lines.append(f"\n📁 {result['file_path']} ({len(issues)}个问题)")

                    for issue in issues:
                        severity_icon = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}.get(
                            issue.get("severity", "medium"), "⚠️"
                        )
                        lines.append(
                            f"  {severity_icon} 第{issue['line']}行: {issue['message']}"
                        )
                        if issue.get("matched_text"):
                            lines.append(f"     匹配文本: {issue['matched_text']}")

        else:
            lines.append("\n✅ 未发现未来函数问题")

        return "\n".join(lines)

    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """生成Markdown格式报告"""
        lines = ["# 未来函数静态检查报告\n"]

        if "directory" in results:
            lines.append(f"**检查目录**: `{results['directory']}`\n")

        lines.append("## 检查统计\n")
        lines.append(f"- **检查文件数**: {results['files_checked']}")
        lines.append(f"- **发现问题数**: {results['total_issues']}")
        lines.append(f"- **总耗时**: {results['total_scan_time']:.3f}秒")

        if results["total_issues"] > 0:
            severity_counts = results.get("severity_counts", {})
            lines.append(f"- **严重程度分布**:")
            lines.append(f"  - 🔴 高: {severity_counts.get('high', 0)}")
            lines.append(f"  - 🟡 中: {severity_counts.get('medium', 0)}")
            lines.append(f"  - 🟢 低: {severity_counts.get('low', 0)}")

            lines.append("\n## 问题详情\n")

            for result in results.get("results", []):
                if result.get("status") == "error":
                    lines.append(f"### ❌ {result['file_path']}")
                    lines.append(f"**错误**: {result.get('error', 'Unknown error')}\n")
                    continue

                issues = result.get("issues", [])
                if issues:
                    lines.append(
                        f"### 📁 {result['file_path']} ({len(issues)}个问题)\n"
                    )

                    for issue in issues:
                        severity_emoji = {
                            "high": "🔴",
                            "medium": "🟡",
                            "low": "🟢",
                        }.get(issue.get("severity", "medium"), "🟡")
                        lines.append(
                            f"- {severity_emoji} **第{issue['line']}行**: {issue['message']}"
                        )
                        if issue.get("matched_text"):
                            lines.append(f"  - 匹配文本: `{issue['matched_text']}`")
                    lines.append("")

        else:
            lines.append("\n## ✅ 结果\n")
            lines.append("未发现未来函数问题")

        return "\n".join(lines)
