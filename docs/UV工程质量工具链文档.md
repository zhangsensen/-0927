# UV工程质量工具链文档

## 概述

基于现代Python包管理器UV构建的专业级量化交易系统工程质量工具链，集成性能分析、代码质量保证、测试框架和部署自动化。

## 🚀 UV包管理器

### 核心优势
- **极速依赖解析**: 比pip快10-100倍
- **确定性构建**: 锁定文件保证环境一致性
- **现代Python工具链**: 无需pip、virtualenv、venv
- **零配置**: 开箱即用的项目管理

### 安装配置
```bash
# 安装UV (推荐方法)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用pip安装
pip install uv

# 初始化项目
uv init quantitative-trading
cd quantitative-trading
```

### 项目配置 (pyproject.toml)
```toml
[project]
name = "quantitative-trading"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "vectorbt>=0.28.1",
    "pandas>=2.3.2",
    "numpy>=2.3.3",
    "polars>=1.0.0",
    "numba>=0.60.0",
    "ta-lib>=0.6.7",
    "scikit-learn>=1.7.2",
    "scipy>=1.16.2",
    "pyarrow>=21.0.0",
    "yfinance>=0.2.66",
    "matplotlib>=3.10.6",
    "seaborn>=0.13.2",
    "plotly>=5.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "black>=24.0.0",
    "isort>=5.12.0",
    "mypy>=1.8.0",
    "ruff>=0.1.0",
    "line-profiler>=4.0.0",
    "memory-profiler>=0.61.0",
    "pre-commit>=3.5.0",
]
test = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.5.0",
    "pytest-benchmark>=4.0.0",
]
performance = [
    "line-profiler>=4.0.0",
    "memory-profiler>=0.61.0",
    "py-spy>=0.3.14",
    "snakeviz>=2.2.0",
]

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

[tool.mypy]
python_version = "3.11"
strict = true
disallow_untyped_defs = true
warn_return_any = true
warn_unused_configs = true

[tool.ruff]
target-version = "py311"
line-length = 88
select = ["E", "F", "W", "I", "N", "B", "C90"]
```

## 🔧 性能分析工具链

### 1. Line Profiler - 逐行性能分析
```bash
# 安装line-profiler
uv add --dev line-profiler

# 分析函数级性能
uv run kernprof -l -v script.py

# 生成详细报告
uv run python -m line_profiler script.py.lprof
```

**示例输出**:
```
Timer unit: 1e-06 s

Total time: 6.934 s
File: professional_factor_screener.py
Function: calculate_rolling_ic at line 1053

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
  1053                                               def calculate_rolling_ic(self, factors, returns, window=20):
  1206     19600      6934000    353.7     99.6          # O(n²) 循环 - 性能瓶颈
```

### 2. Memory Profiler - 内存使用分析
```bash
# 安装memory-profiler
uv add --dev memory-profiler

# 分析内存使用
uv run python -m memory_profiler script.py

# 生成内存报告
uv run mprof run script.py
uv run mprof plot
```

### 3. Performance Benchmarking
```bash
# 安装pytest-benchmark
uv add --dev pytest-benchmark

# 运行性能基准测试
uv run pytest tests/test_performance.py --benchmark-only

# 生成性能报告
uv run pytest tests/test_performance.py --benchmark-json=benchmark.json
```

## 📊 代码质量工具

### 1. Black - 代码格式化
```bash
# 格式化所有代码
uv run black factor_system/ data-resampling/

# 检查格式 (不修改文件)
uv run black --check factor_system/

# 显示差异
uv run black --diff factor_system/
```

### 2. isort - 导入排序
```bash
# 排序导入
uv run isort factor_system/ data-resampling/

# 检查导入顺序
uv run isort --check-only factor_system/

# 与black配合使用
uv run isort --profile black factor_system/
```

### 3. Ruff - 快速代码检查
```bash
# 运行代码检查
uv run ruff check factor_system/

# 自动修复简单问题
uv run ruff check --fix factor_system/

# 详细输出
uv run ruff check --verbose factor_system/
```

### 4. MyPy - 静态类型检查
```bash
# 严格类型检查
uv run mypy factor_system/ --strict

# 检查特定模块
uv run mypy factor_system/factor_screening/professional_factor_screener.py

# 生成类型报告
uv run mypy factor_system/ --html-report mypy-report
```

## 🧪 测试框架

### 1. Pytest - 测试运行器
```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/test_professional_screener.py

# 运行特定测试函数
uv run pytest tests/test_professional_screener.py::test_calculate_rolling_ic

# 详细输出
uv run pytest -v

# 停止在第一个失败
uv run pytest -x
```

### 2. Coverage - 测试覆盖率
```bash
# 运行测试并生成覆盖率报告
uv run pytest --cov=factor_system --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html

# 设置覆盖率阈值
uv run pytest --cov=factor_system --cov-fail-under=80
```

### 3. Parameterized Tests - 参数化测试
```python
import pytest

@pytest.mark.parametrize("stock_code,timeframe,expected_factors", [
    ("0700.HK", "5min", 217),
    ("0700.HK", "15min", 245),
    ("0700.HK", "60min", 252),
    ("0700.HK", "daily", 204),
])
def test_factor_count_by_timeframe(stock_code, timeframe, expected_factors):
    """测试不同时间框架的因子数量"""
    screener = ProfessionalFactorScreener()
    factors = screener.generate_factors(stock_code, timeframe)
    assert len(factors.columns) >= expected_factors
```

## 🚀 部署自动化

### 1. Pre-commit Hooks - Git提交钩子
```bash
# 安装pre-commit
uv add --dev pre-commit

# 创建.pre-commit-config.yaml
uv run pre-commit install

# 手动运行所有钩子
uv run pre-commit run --all-files
```

**配置文件**:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### 2. UV Scripts - 项目脚本
```toml
[project.scripts]
# 开发工具脚本
dev-setup = "scripts.dev_setup:main"
test-all = "scripts.test:main"
quality-check = "scripts.quality:main"
performance-test = "scripts.performance:main"

# 量化交易脚本
factor-screen = "factor_system.factor_screening.cli:main"
backtest = "factor_system.backtest.engine:main"
analyze-stock = "factor_system.analysis.single_stock:main"
```

### 3. 环境管理
```bash
# 创建开发环境
uv sync --dev

# 创建生产环境
uv sync --no-dev

# 运行特定环境中的命令
uv run --dev python -m pytest
uv run python factor_screening.py

# 锁定依赖版本
uv lock

# 导出依赖
uv pip freeze > requirements.txt
```

## 📈 性能监控与优化

### 1. 自定义性能分析脚本
```python
#!/usr/bin/env python3
"""
性能分析脚本 - 针对量化系统优化
"""
import time
import psutil
import numpy as np
from contextlib import contextmanager

@contextmanager
def performance_monitor(operation_name):
    """性能监控上下文管理器"""
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    yield

    end_time = time.perf_counter()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    print(f"🚀 {operation_name} 性能指标:")
    print(f"   ⏱️  执行时间: {end_time - start_time:.4f}s")
    print(f"   💾 内存变化: {end_memory - start_memory:.2f}MB")
    print(f"   📊 最终内存: {end_memory:.2f}MB")

def benchmark_factor_calculation(stock_code="0700.HK", timeframe="5min"):
    """因子计算性能基准测试"""
    with performance_monitor(f"{stock_code} {timeframe} 因子计算"):
        screener = ProfessionalFactorScreener()
        factors = screener.generate_factors(stock_code, timeframe)

    print(f"   📈 生成因子数: {len(factors.columns)}")
    print(f"   📏 数据样本数: {len(factors)}")

    return factors

if __name__ == "__main__":
    benchmark_factor_calculation()
```

### 2. 自动化性能测试
```python
import pytest
import time
from factor_system.factor_screening.professional_factor_screener import ProfessionalFactorScreener

class TestPerformance:
    """性能测试套件"""

    @pytest.mark.benchmark
    def test_rolling_ic_performance(self, benchmark):
        """测试滚动IC计算性能"""
        screener = ProfessionalFactorScreener()
        factors, returns = self._load_test_data()

        result = benchmark(
            screener.calculate_rolling_ic,
            factors, returns, window=20
        )

        assert len(result) > 0

    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        screener = ProfessionalFactorScreener()
        factors = screener.generate_factors("0700.HK", "5min")

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

        assert memory_increase < 100, f"内存增长过多: {memory_increase:.2f}MB"
```

## 🛠️ 实用开发脚本

### 1. dev_setup.py - 开发环境设置
```python
#!/usr/bin/env python3
"""
开发环境自动设置脚本
"""
import subprocess
import sys
from pathlib import Path

def main():
    """设置开发环境"""
    print("🚀 设置量化交易系统开发环境...")

    # 安装依赖
    print("📦 安装依赖...")
    subprocess.run(["uv", "sync", "--dev"], check=True)

    # 安装pre-commit钩子
    print("🪝 设置Git钩子...")
    subprocess.run(["uv", "run", "pre-commit", "install"], check=True)

    # 创建必要目录
    directories = ["logs", "results", "cache", "reports"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 创建目录: {dir_name}")

    # 运行质量检查
    print("✅ 运行质量检查...")
    subprocess.run(["uv", "run", "black", "--check", "factor_system/"])
    subprocess.run(["uv", "run", "isort", "--check-only", "factor_system/"])
    subprocess.run(["uv", "run", "ruff", "check", "factor_system/"])

    print("🎉 开发环境设置完成!")

if __name__ == "__main__":
    main()
```

### 2. quality_check.py - 代码质量检查
```python
#!/usr/bin/env python3
"""
代码质量检查脚本
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """运行命令并处理结果"""
    print(f"🔍 {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ {description} 失败:")
        print(result.stdout)
        print(result.stderr)
        return False
    else:
        print(f"✅ {description} 通过")
        return True

def main():
    """运行所有质量检查"""
    print("🚀 运行代码质量检查...")

    checks = [
        ("uv run black --check factor_system/", "代码格式检查"),
        ("uv run isort --check-only factor_system/", "导入顺序检查"),
        ("uv run ruff check factor_system/", "代码质量检查"),
        ("uv run mypy factor_system/ --strict", "类型检查"),
        ("uv run pytest --cov=factor_system", "测试覆盖率检查"),
    ]

    failed_checks = []
    for cmd, desc in checks:
        if not run_command(cmd, desc):
            failed_checks.append(desc)

    if failed_checks:
        print(f"\n❌ 质量检查失败: {', '.join(failed_checks)}")
        sys.exit(1)
    else:
        print("\n🎉 所有质量检查通过!")

if __name__ == "__main__":
    main()
```

### 3. performance_test.py - 性能测试脚本
```python
#!/usr/bin/env python3
"""
性能测试脚本
"""
import time
import psutil
from contextlib import contextmanager
from factor_system.factor_screening.professional_factor_screener import ProfessionalFactorScreener

@contextmanager
def performance_monitor(name):
    """性能监控器"""
    start = time.perf_counter()
    start_mem = psutil.Process().memory_info().rss / 1024 / 1024

    yield

    elapsed = time.perf_counter() - start
    end_mem = psutil.Process().memory_info().rss / 1024 / 1024

    print(f"\n📊 {name} 性能指标:")
    print(f"   ⏱️  执行时间: {elapsed:.4f}s")
    print(f"   💾 内存使用: {end_mem - start_mem:+.2f}MB")
    print(f"   📈 总内存: {end_mem:.2f}MB")

def run_performance_tests():
    """运行性能测试"""
    print("🚀 运行性能测试...")

    screener = ProfessionalFactorScreener()

    # 测试不同时间框架的性能
    test_cases = [
        ("0700.HK", "5min"),
        ("0700.HK", "15min"),
        ("0700.HK", "60min"),
        ("0700.HK", "daily"),
    ]

    for stock_code, timeframe in test_cases:
        with performance_monitor(f"{stock_code} {timeframe} 因子生成"):
            factors = screener.generate_factors(stock_code, timeframe)
            print(f"   📊 因子数量: {len(factors.columns)}")
            print(f"   📏 数据样本: {len(factors)}")

if __name__ == "__main__":
    run_performance_tests()
```

## 📋 工程质量最佳实践

### 1. 开发工作流
```bash
# 1. 设置开发环境
uv run python scripts/dev_setup.py

# 2. 开发新功能
git checkout -b feature/new-factor

# 3. 运行质量检查
uv run python scripts/quality_check.py

# 4. 运行测试
uv run pytest

# 5. 性能测试
uv run python scripts/performance_test.py

# 6. 提交代码
git add .
git commit -m "feat: add new factor calculation"

# 7. 推送和PR
git push origin feature/new-factor
```

### 2. 性能优化流程
```bash
# 1. 性能分析
uv run kernprof -l -v factor_system/factor_screening/professional_factor_screener.py

# 2. 内存分析
uv run python -m memory_profiler factor_system/factor_screening/professional_factor_screener.py

# 3. 运行基准测试
uv run pytest tests/test_performance.py --benchmark-only

# 4. 验证优化效果
uv run python scripts/performance_test.py
```

### 3. 代码质量门禁
```yaml
# .github/workflows/quality.yml
name: Quality Check
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        run: uv sync --dev

      - name: Run quality checks
        run: uv run python scripts/quality_check.py

      - name: Run tests
        run: uv run pytest --cov=factor_system --cov-fail-under=80
```

## 🎯 总结

基于UV的工程质量工具链提供了：

- **🚀 极速依赖管理**: UV比传统pip快10-100倍
- **📊 全面性能分析**: Line profiler、Memory profiler、Benchmarking
- **🔧 严格质量保证**: Black、isort、Ruff、MyPy四重保障
- **🧪 完整测试框架**: Pytest、Coverage、参数化测试
- **⚙️ 自动化部署**: Pre-commit hooks、CI/CD集成
- **📈 持续监控**: 性能监控、内存追踪、优化建议

这套工具链确保了量化交易系统的代码质量、性能优化和工程效率，为专业级量化开发提供了坚实的基础设施支持。