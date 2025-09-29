#!/usr/bin/env python3
"""
质量保证检查 - 专业级因子筛选系统
作者：量化首席工程师
版本：2.0.0
日期：2025-09-29

检查内容：
1. 代码质量审查
2. 测试覆盖率验证
3. 性能基准验证
4. 配置系统验证
5. 文档完整性检查
6. 系统集成验证
"""

import sys
import subprocess
import time
from pathlib import Path
import importlib.util
import ast
import re
from typing import Dict, List, Tuple, Any

class QualityAssuranceChecker:
    """质量保证检查器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {}
        
    def check_code_quality(self) -> Dict[str, Any]:
        """代码质量检查"""
        print("🔍 代码质量检查")
        print("-" * 60)
        
        quality_results = {
            'file_structure': self.check_file_structure(),
            'import_structure': self.check_import_structure(),
            'function_complexity': self.check_function_complexity(),
            'documentation': self.check_code_documentation(),
            'naming_conventions': self.check_naming_conventions()
        }
        
        return quality_results
    
    def check_file_structure(self) -> Dict[str, Any]:
        """检查文件结构"""
        print("📁 检查文件结构...")
        
        required_files = [
            'professional_factor_screener.py',
            'config_loader.py',
            'test_basic_functionality.py',
            'performance_benchmark.py',
            'README.md',
            'config/screening_config.yaml'
        ]
        
        existing_files = []
        missing_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
                print(f"  ✅ {file_path}")
            else:
                missing_files.append(file_path)
                print(f"  ❌ {file_path} (缺失)")
        
        return {
            'existing_files': existing_files,
            'missing_files': missing_files,
            'structure_score': len(existing_files) / len(required_files)
        }
    
    def check_import_structure(self) -> Dict[str, Any]:
        """检查导入结构"""
        print("📦 检查导入结构...")
        
        main_file = self.project_root / 'professional_factor_screener.py'
        if not main_file.exists():
            return {'error': 'Main file not found'}
        
        try:
            with open(main_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            # 检查关键依赖
            required_imports = [
                'pandas', 'numpy', 'scipy', 'statsmodels', 'vectorbt'
            ]
            
            missing_imports = []
            for req in required_imports:
                if not any(req in imp for imp in imports):
                    missing_imports.append(req)
            
            print(f"  📊 总导入数: {len(imports)}")
            print(f"  ✅ 必需依赖: {len(required_imports) - len(missing_imports)}/{len(required_imports)}")
            
            if missing_imports:
                print(f"  ❌ 缺失依赖: {missing_imports}")
            
            return {
                'total_imports': len(imports),
                'required_imports': required_imports,
                'missing_imports': missing_imports,
                'import_score': (len(required_imports) - len(missing_imports)) / len(required_imports)
            }
            
        except Exception as e:
            print(f"  ❌ 导入检查失败: {str(e)}")
            return {'error': str(e)}
    
    def check_function_complexity(self) -> Dict[str, Any]:
        """检查函数复杂度"""
        print("🔧 检查函数复杂度...")
        
        main_file = self.project_root / 'professional_factor_screener.py'
        if not main_file.exists():
            return {'error': 'Main file not found'}
        
        try:
            with open(main_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            functions = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # 计算函数行数
                    lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
                    functions.append({
                        'name': node.name,
                        'lines': lines,
                        'args': len(node.args.args)
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    })
            
            # 分析复杂度
            long_functions = [f for f in functions if f['lines'] > 50]
            complex_functions = [f for f in functions if f['args'] > 8]
            
            print(f"  📊 总函数数: {len(functions)}")
            print(f"  📊 总类数: {len(classes)}")
            print(f"  ⚠️  长函数 (>50行): {len(long_functions)}")
            print(f"  ⚠️  复杂函数 (>8参数): {len(complex_functions)}")
            
            return {
                'total_functions': len(functions),
                'total_classes': len(classes),
                'long_functions': long_functions,
                'complex_functions': complex_functions,
                'complexity_score': 1 - (len(long_functions) + len(complex_functions)) / max(len(functions), 1)
            }
            
        except Exception as e:
            print(f"  ❌ 复杂度检查失败: {str(e)}")
            return {'error': str(e)}
    
    def check_code_documentation(self) -> Dict[str, Any]:
        """检查代码文档"""
        print("📝 检查代码文档...")
        
        main_file = self.project_root / 'professional_factor_screener.py'
        if not main_file.exists():
            return {'error': 'Main file not found'}
        
        try:
            with open(main_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 统计文档字符串
            docstring_pattern = r'"""[\s\S]*?"""'
            docstrings = re.findall(docstring_pattern, content)
            
            # 统计注释
            comment_pattern = r'#.*'
            comments = re.findall(comment_pattern, content)
            
            # 统计代码行
            lines = content.split('\n')
            code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            
            doc_coverage = len(docstrings) / max(len(code_lines) / 20, 1)  # 估算文档覆盖率
            
            print(f"  📊 文档字符串: {len(docstrings)}")
            print(f"  📊 注释行数: {len(comments)}")
            print(f"  📊 代码行数: {len(code_lines)}")
            print(f"  📈 文档覆盖率: {doc_coverage:.1%}")
            
            return {
                'docstrings': len(docstrings),
                'comments': len(comments),
                'code_lines': len(code_lines),
                'doc_coverage': doc_coverage,
                'doc_score': min(doc_coverage, 1.0)
            }
            
        except Exception as e:
            print(f"  ❌ 文档检查失败: {str(e)}")
            return {'error': str(e)}
    
    def check_naming_conventions(self) -> Dict[str, Any]:
        """检查命名规范"""
        print("🏷️  检查命名规范...")
        
        main_file = self.project_root / 'professional_factor_screener.py'
        if not main_file.exists():
            return {'error': 'Main file not found'}
        
        try:
            with open(main_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # 检查类名 (PascalCase)
            class_names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_names.append(node.name)
            
            # 检查函数名 (snake_case)
            function_names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_names.append(node.name)
            
            # 验证命名规范
            pascal_case_pattern = r'^[A-Z][a-zA-Z0-9]*$'
            snake_case_pattern = r'^[a-z_][a-z0-9_]*$'
            
            valid_class_names = [name for name in class_names if re.match(pascal_case_pattern, name)]
            valid_function_names = [name for name in function_names if re.match(snake_case_pattern, name) or name.startswith('_')]
            
            class_score = len(valid_class_names) / max(len(class_names), 1)
            function_score = len(valid_function_names) / max(len(function_names), 1)
            
            print(f"  📊 类名规范: {len(valid_class_names)}/{len(class_names)} ({class_score:.1%})")
            print(f"  📊 函数名规范: {len(valid_function_names)}/{len(function_names)} ({function_score:.1%})")
            
            return {
                'class_names': class_names,
                'function_names': function_names,
                'valid_class_names': valid_class_names,
                'valid_function_names': valid_function_names,
                'class_score': class_score,
                'function_score': function_score,
                'naming_score': (class_score + function_score) / 2
            }
            
        except Exception as e:
            print(f"  ❌ 命名检查失败: {str(e)}")
            return {'error': str(e)}
    
    def check_test_coverage(self) -> Dict[str, Any]:
        """检查测试覆盖率"""
        print("\n🧪 测试覆盖率检查")
        print("-" * 60)
        
        test_files = [
            'test_basic_functionality.py',
            'tests/test_professional_screener.py'
        ]
        
        existing_tests = []
        test_results = {}
        
        for test_file in test_files:
            test_path = self.project_root / test_file
            if test_path.exists():
                existing_tests.append(test_file)
                print(f"  ✅ {test_file}")
                
                # 运行基础功能测试
                if test_file == 'test_basic_functionality.py':
                    try:
                        print(f"  🏃 运行 {test_file}...")
                        result = subprocess.run(
                            [sys.executable, str(test_path)],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        
                        if result.returncode == 0:
                            print(f"    ✅ 测试通过")
                            test_results[test_file] = 'PASSED'
                        else:
                            print(f"    ❌ 测试失败")
                            test_results[test_file] = 'FAILED'
                            
                    except subprocess.TimeoutExpired:
                        print(f"    ⏰ 测试超时")
                        test_results[test_file] = 'TIMEOUT'
                    except Exception as e:
                        print(f"    ❌ 测试错误: {str(e)}")
                        test_results[test_file] = 'ERROR'
            else:
                print(f"  ❌ {test_file} (缺失)")
        
        coverage_score = len([r for r in test_results.values() if r == 'PASSED']) / max(len(test_files), 1)
        
        return {
            'existing_tests': existing_tests,
            'test_results': test_results,
            'coverage_score': coverage_score
        }
    
    def check_performance_benchmarks(self) -> Dict[str, Any]:
        """检查性能基准"""
        print("\n⚡ 性能基准检查")
        print("-" * 60)
        
        benchmark_file = self.project_root / 'performance_benchmark.py'
        
        if not benchmark_file.exists():
            print("  ❌ 性能基准文件不存在")
            return {'error': 'Benchmark file not found'}
        
        print("  ✅ 性能基准文件存在")
        
        # 检查性能报告
        report_files = list(self.project_root.glob('performance_report_*.txt'))
        
        if report_files:
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            print(f"  ✅ 最新性能报告: {latest_report.name}")
            
            # 解析性能报告
            try:
                with open(latest_report, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取关键性能指标
                ic_performance = re.search(r'IC计算性能: 🟢 优秀 \((\d+\.?\d*) 因子/秒\)', content)
                memory_efficiency = re.search(r'内存使用效率: (\S+) \((\d+\.?\d*)%\)', content)
                parallel_efficiency = re.search(r'并行处理效率: (\S+) \((\d+\.?\d*)%\)', content)
                
                performance_metrics = {}
                
                if ic_performance:
                    throughput = float(ic_performance.group(1))
                    performance_metrics['ic_throughput'] = throughput
                    print(f"    📊 IC计算吞吐量: {throughput} 因子/秒")
                
                if memory_efficiency:
                    mem_grade = memory_efficiency.group(1)
                    mem_percent = float(memory_efficiency.group(2))
                    performance_metrics['memory_efficiency'] = mem_percent
                    print(f"    📊 内存效率: {mem_percent}% ({mem_grade})")
                
                if parallel_efficiency:
                    par_grade = parallel_efficiency.group(1)
                    par_percent = float(parallel_efficiency.group(2))
                    performance_metrics['parallel_efficiency'] = par_percent
                    print(f"    📊 并行效率: {par_percent}% ({par_grade})")
                
                # 计算性能得分
                performance_score = 0
                if 'ic_throughput' in performance_metrics:
                    performance_score += min(performance_metrics['ic_throughput'] / 500, 1.0) * 0.5
                if 'memory_efficiency' in performance_metrics:
                    performance_score += performance_metrics['memory_efficiency'] / 100 * 0.3
                if 'parallel_efficiency' in performance_metrics:
                    performance_score += performance_metrics['parallel_efficiency'] / 100 * 0.2
                
                return {
                    'report_exists': True,
                    'latest_report': latest_report.name,
                    'performance_metrics': performance_metrics,
                    'performance_score': performance_score
                }
                
            except Exception as e:
                print(f"  ❌ 性能报告解析失败: {str(e)}")
                return {'error': f'Report parsing failed: {str(e)}'}
        else:
            print("  ⚠️  未找到性能报告")
            return {'warning': 'No performance reports found'}
    
    def check_configuration_system(self) -> Dict[str, Any]:
        """检查配置系统"""
        print("\n⚙️  配置系统检查")
        print("-" * 60)
        
        config_files = [
            'config/screening_config.yaml',
            'config_loader.py'
        ]
        
        config_results = {}
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                print(f"  ✅ {config_file}")
                config_results[config_file] = 'EXISTS'
            else:
                print(f"  ❌ {config_file} (缺失)")
                config_results[config_file] = 'MISSING'
        
        # 测试配置加载
        try:
            config_loader_path = self.project_root / 'config_loader.py'
            if config_loader_path.exists():
                print("  🔧 测试配置加载...")
                
                result = subprocess.run(
                    [sys.executable, str(config_loader_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print("    ✅ 配置加载测试通过")
                    config_results['loader_test'] = 'PASSED'
                else:
                    print("    ❌ 配置加载测试失败")
                    config_results['loader_test'] = 'FAILED'
            
        except Exception as e:
            print(f"    ❌ 配置测试错误: {str(e)}")
            config_results['loader_test'] = 'ERROR'
        
        config_score = len([r for r in config_results.values() if r in ['EXISTS', 'PASSED']]) / len(config_results)
        
        return {
            'config_results': config_results,
            'config_score': config_score
        }
    
    def check_documentation_completeness(self) -> Dict[str, Any]:
        """检查文档完整性"""
        print("\n📚 文档完整性检查")
        print("-" * 60)
        
        doc_files = [
            'README.md',
            'factor_system_optimization_plan.md'
        ]
        
        doc_results = {}
        
        for doc_file in doc_files:
            doc_path = self.project_root / doc_file
            if doc_path.exists():
                print(f"  ✅ {doc_file}")
                
                # 检查文档长度
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    word_count = len(content.split())
                    
                print(f"    📊 字数: {word_count}")
                
                if word_count > 1000:
                    doc_results[doc_file] = 'COMPREHENSIVE'
                elif word_count > 500:
                    doc_results[doc_file] = 'ADEQUATE'
                else:
                    doc_results[doc_file] = 'BRIEF'
            else:
                print(f"  ❌ {doc_file} (缺失)")
                doc_results[doc_file] = 'MISSING'
        
        # 检查README关键章节
        readme_path = self.project_root / 'README.md'
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
            
            required_sections = [
                '系统概述', '快速开始', '5维度筛选框架', 
                '配置系统', '性能基准', '使用示例'
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in readme_content:
                    missing_sections.append(section)
            
            section_score = (len(required_sections) - len(missing_sections)) / len(required_sections)
            
            print(f"  📊 README章节完整性: {section_score:.1%}")
            if missing_sections:
                print(f"    ⚠️  缺失章节: {missing_sections}")
            
            doc_results['readme_sections'] = section_score
        
        doc_score = len([r for r in doc_results.values() if r in ['COMPREHENSIVE', 'ADEQUATE']]) / max(len(doc_files), 1)
        
        return {
            'doc_results': doc_results,
            'doc_score': doc_score
        }
    
    def run_system_integration_test(self) -> Dict[str, Any]:
        """运行系统集成测试"""
        print("\n🔗 系统集成测试")
        print("-" * 60)
        
        integration_results = {}
        
        # 测试主要模块导入
        modules_to_test = [
            'professional_factor_screener',
            'config_loader'
        ]
        
        for module_name in modules_to_test:
            module_path = self.project_root / f'{module_name}.py'
            if module_path.exists():
                try:
                    print(f"  🔧 测试模块导入: {module_name}")
                    
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    print(f"    ✅ 导入成功")
                    integration_results[module_name] = 'IMPORT_SUCCESS'
                    
                except Exception as e:
                    print(f"    ❌ 导入失败: {str(e)}")
                    integration_results[module_name] = 'IMPORT_FAILED'
            else:
                print(f"  ❌ 模块文件不存在: {module_name}")
                integration_results[module_name] = 'FILE_MISSING'
        
        integration_score = len([r for r in integration_results.values() if r == 'IMPORT_SUCCESS']) / len(modules_to_test)
        
        return {
            'integration_results': integration_results,
            'integration_score': integration_score
        }
    
    def generate_quality_report(self, all_results: Dict[str, Any]) -> str:
        """生成质量报告"""
        report = []
        report.append("="*100)
        report.append("专业级因子筛选系统 - 质量保证报告")
        report.append("="*100)
        report.append(f"检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"版本: 2.0.0")
        report.append("")
        
        # 计算总体质量得分
        total_score = 0
        score_count = 0
        
        # 代码质量
        if 'code_quality' in all_results:
            report.append("🔍 代码质量评估")
            report.append("-" * 50)
            
            cq = all_results['code_quality']
            
            if 'file_structure' in cq and 'structure_score' in cq['file_structure']:
                score = cq['file_structure']['structure_score']
                report.append(f"文件结构: {score:.1%} {'🟢' if score > 0.8 else '🟡' if score > 0.6 else '🔴'}")
                total_score += score
                score_count += 1
            
            if 'import_structure' in cq and 'import_score' in cq['import_structure']:
                score = cq['import_structure']['import_score']
                report.append(f"导入结构: {score:.1%} {'🟢' if score > 0.8 else '🟡' if score > 0.6 else '🔴'}")
                total_score += score
                score_count += 1
            
            if 'function_complexity' in cq and 'complexity_score' in cq['function_complexity']:
                score = cq['function_complexity']['complexity_score']
                report.append(f"函数复杂度: {score:.1%} {'🟢' if score > 0.8 else '🟡' if score > 0.6 else '🔴'}")
                total_score += score
                score_count += 1
            
            if 'documentation' in cq and 'doc_score' in cq['documentation']:
                score = cq['documentation']['doc_score']
                report.append(f"代码文档: {score:.1%} {'🟢' if score > 0.8 else '🟡' if score > 0.6 else '🔴'}")
                total_score += score
                score_count += 1
            
            if 'naming_conventions' in cq and 'naming_score' in cq['naming_conventions']:
                score = cq['naming_conventions']['naming_score']
                report.append(f"命名规范: {score:.1%} {'🟢' if score > 0.8 else '🟡' if score > 0.6 else '🔴'}")
                total_score += score
                score_count += 1
            
            report.append("")
        
        # 测试覆盖率
        if 'test_coverage' in all_results:
            report.append("🧪 测试覆盖率")
            report.append("-" * 50)
            
            tc = all_results['test_coverage']
            if 'coverage_score' in tc:
                score = tc['coverage_score']
                report.append(f"测试覆盖率: {score:.1%} {'🟢' if score > 0.8 else '🟡' if score > 0.6 else '🔴'}")
                total_score += score
                score_count += 1
            
            if 'test_results' in tc:
                for test_file, result in tc['test_results'].items():
                    status = "✅" if result == "PASSED" else "❌"
                    report.append(f"  {status} {test_file}: {result}")
            
            report.append("")
        
        # 性能基准
        if 'performance' in all_results:
            report.append("⚡ 性能基准")
            report.append("-" * 50)
            
            perf = all_results['performance']
            if 'performance_score' in perf:
                score = perf['performance_score']
                report.append(f"性能得分: {score:.1%} {'🟢' if score > 0.8 else '🟡' if score > 0.6 else '🔴'}")
                total_score += score
                score_count += 1
            
            if 'performance_metrics' in perf:
                metrics = perf['performance_metrics']
                for key, value in metrics.items():
                    if key == 'ic_throughput':
                        report.append(f"  IC计算吞吐量: {value} 因子/秒")
                    elif key == 'memory_efficiency':
                        report.append(f"  内存效率: {value}%")
                    elif key == 'parallel_efficiency':
                        report.append(f"  并行效率: {value}%")
            
            report.append("")
        
        # 配置系统
        if 'configuration' in all_results:
            report.append("⚙️ 配置系统")
            report.append("-" * 50)
            
            config = all_results['configuration']
            if 'config_score' in config:
                score = config['config_score']
                report.append(f"配置系统: {score:.1%} {'🟢' if score > 0.8 else '🟡' if score > 0.6 else '🔴'}")
                total_score += score
                score_count += 1
            
            report.append("")
        
        # 文档完整性
        if 'documentation' in all_results:
            report.append("📚 文档完整性")
            report.append("-" * 50)
            
            doc = all_results['documentation']
            if 'doc_score' in doc:
                score = doc['doc_score']
                report.append(f"文档完整性: {score:.1%} {'🟢' if score > 0.8 else '🟡' if score > 0.6 else '🔴'}")
                total_score += score
                score_count += 1
            
            report.append("")
        
        # 系统集成
        if 'integration' in all_results:
            report.append("🔗 系统集成")
            report.append("-" * 50)
            
            integration = all_results['integration']
            if 'integration_score' in integration:
                score = integration['integration_score']
                report.append(f"系统集成: {score:.1%} {'🟢' if score > 0.8 else '🟡' if score > 0.6 else '🔴'}")
                total_score += score
                score_count += 1
            
            report.append("")
        
        # 总体评估
        if score_count > 0:
            overall_score = total_score / score_count
            
            report.append("🏆 总体质量评估")
            report.append("-" * 50)
            report.append(f"总体得分: {overall_score:.1%}")
            
            if overall_score >= 0.9:
                grade = "🟢 优秀 (A级)"
                recommendation = "系统质量优秀，可以投入生产使用"
            elif overall_score >= 0.8:
                grade = "🟡 良好 (B级)"
                recommendation = "系统质量良好，建议进行小幅优化后使用"
            elif overall_score >= 0.7:
                grade = "🟠 合格 (C级)"
                recommendation = "系统基本合格，需要进行优化改进"
            else:
                grade = "🔴 需改进 (D级)"
                recommendation = "系统需要重大改进才能投入使用"
            
            report.append(f"质量等级: {grade}")
            report.append(f"建议: {recommendation}")
        
        report.append("")
        report.append("="*100)
        
        return "\n".join(report)

def run_quality_assurance():
    """运行质量保证检查"""
    print("🚀 启动专业级因子筛选系统质量保证检查")
    print("="*100)
    
    checker = QualityAssuranceChecker()
    all_results = {}
    
    try:
        # 1. 代码质量检查
        all_results['code_quality'] = checker.check_code_quality()
        
        # 2. 测试覆盖率检查
        all_results['test_coverage'] = checker.check_test_coverage()
        
        # 3. 性能基准检查
        all_results['performance'] = checker.check_performance_benchmarks()
        
        # 4. 配置系统检查
        all_results['configuration'] = checker.check_configuration_system()
        
        # 5. 文档完整性检查
        all_results['documentation'] = checker.check_documentation_completeness()
        
        # 6. 系统集成测试
        all_results['integration'] = checker.run_system_integration_test()
        
        # 7. 生成质量报告
        quality_report = checker.generate_quality_report(all_results)
        
        # 输出报告
        print("\n" + quality_report)
        
        # 保存报告
        report_path = checker.project_root / f"quality_assurance_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(quality_report)
        
        print(f"\n📄 质量保证报告已保存: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 质量保证检查失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_quality_assurance()
    sys.exit(0 if success else 1)

