<!-- .github/copilot-instructions.md
简短说明：为 AI 编码代理提供可执行、基于本仓库可发现事实的指导。仅包含可验证的模式、关键入口和常用命令示例。 -->

# 快速上手 — 给 AI 编码代理的说明

目标：快速让代理在本仓库内编写、修改和调试代码时立即有价值。优先级是可执行命令、关键文件引用、约定与不可做的操作。

1) 主要入口与职责（必读）
 - 单一因子计算入口：`factor_system/factor_engine/api.py` —— 所有因子计算与列举接口都应通过它调用。
 - 路径与资源：`factor_system/utils/project_paths.py`（或 `factor_system/utils` 中的 path helpers）——**绝对禁止硬编码路径**，必须使用统一路径 API。
 - 因子注册/元数据：`factor_system/factor_engine/factor_registry.py` 或 `factor_system/factor_engine/core/registry.py` —— 因子 id、类别与元数据由注册表管理。
 - ETF 横截面：`factor_system/factor_engine/factors/etf_cross_section/`（尤其 `unified_manager.py`, `batch_factor_calculator.py`）——复杂且受严格一致性约束的子系统。
 - 配置加载：`factor_system/factor_generation/config_loader.py` —— YAML / Pydantic 配置的单一来源。
 - 异常与安全调用：`factor_system/utils/error_utils.py`（`safe_operation`, `FactorSystemError`）——遵循包装/上抛约定。

2) 常用开发命令（在仓库根目录或通过 Makefile）
 - 环境与依赖：`uv sync`，激活虚拟环境 `source .venv/bin/activate`，开发安装 `pip install -e .`
 - 快速任务：`make install`、`make format`、`make lint`、`make test`、`make run-example`
 - 代码质量：`pre-commit run --all-files`、`black .`（line-length 88）、`isort .`、`mypy factor_system/`
 - 运行测试：`pytest -v`，一致性验证：`python tests/test_factor_engine_consistency.py`

3) 项目约定（严格遵守）
 - 路径：从不使用相对/硬编码路径（例如 `../raw`）；使用项目路径 helpers。参考 `CLAUDE.md` 的 "Unified Path Management System" 段落。
 - 时间序列与时帧：支持 `1min,5min,15min,30min,60min,120min,240min,daily,weekly,monthly`。在因子调用中显式传 timeframe。
 - 输出/文件命名：遵循仓库中存在的格式（例如 `{SYMBOL}_{TIMEFRAME}_factors_YYYY-MM-DD_HH-MM-SS.parquet`）。
 - 配置优先：参数化（YAML + Pydantic），不要在代码中写常量参数；请使用 `factor_generation/config_loader.py`。
 - 异常处理：使用 `safe_operation` 装饰器封装可能失败的外部调用并抛出 `FactorSystemError` / `ConfigurationError`。

4) 性能与实现风格（可验证的规则）
 - 向量化优先：目标 >95% 向量化率；避免使用 `DataFrame.apply()`；优先 Pandas/NumPy/Polars/VectorBT。
 - 函数限制：尽量短小（仓库实践倾向 <50 行），并保持明确类型注解（项目 `pyproject.toml` 要求 strict mypy）。
 - 不允许未来函数 / lookahead：时间对齐规则必须严格遵循（见 CLAUDE.md 的 Quantitative Engineering Standards）。

5) 重要的集成点与依赖
 - VectorBT：性能与回测集成（在 `pyproject.toml` 中声明）。
 - ta-lib / technical indicators：因子实现依赖 `ta-lib` 与自定义实现（`factor_system/factor_engine/factors/technical`）。
 - 数据提供者：`factor_system/factor_engine/providers/`（Parquet / CSV）；修改时注意 provider contract。
 - 环境变量：常见变量 `FACTOR_ENGINE_RAW_DATA_DIR`, `FACTOR_ENGINE_CACHE_DIR`, `FACTOR_ENGINE_N_JOBS` 等可改变运行时目录/并行度。

6) 快速示例（可执行、直接引用仓库 API）
 - 计算因子（推荐）：
 ```python
 from factor_system.factor_engine import api
 from datetime import datetime

 api.calculate_factors(
     factor_ids=["RSI", "MACD"],
     symbols=["0700.HK"],
     timeframe="15min",
     start_date=datetime(2025,9,1),
     end_date=datetime(2025,9,30),
 )
 ```
 - 路径检查（调试）：
 ```bash
 python -c "from factor_system.utils import get_project_root; print(get_project_root())"
 ```

7) 调试与验证快捷方式
 - 若怀疑路径/配置，先运行路径验证脚本（见 CLAUDE.md 示例）。
 - 使用 `pytest -k consistency` 或直接运行 `tests/test_factor_engine_consistency.py` 做因子一致性验证。
 - 对性能回归，优先运行 VectorBT 单元或 `scripts/comprehensive_smoke_test.py` 中的烟雾测试。

8) 常见非建议行为（不要做）
 - 不要写硬编码路径；不要使用 DataFrame.apply() 或其他明显非向量化模式；不要引入 lookahead。
 - 不要跳过配置加载或直接修改全局配置对象；优先通过 YAML/loader 修改。

9) 参考文件（优先阅读顺序）
 - `CLAUDE.md`（项目总览、操作命令与设计原则）
 - `pyproject.toml`（依赖、mypy、black 配置）
 - `Makefile`（常用 make 目标）
 - `factor_system/factor_engine/api.py`、`factor_system/utils/project_paths.py`、`factor_system/factor_engine/factors/etf_cross_section/`
 - `docs/FACTOR_ENGINE_DEPLOYMENT_GUIDE.md`、`docs/INTRADAY_RESAMPLE_CN.md`、`docs/FACTOR_SETS.md`

---
请审阅此文件并指出任何遗漏或不够明确的部分（例如：需要补充的脚本路径、特定子系统的运行步骤或额外的 quick commands）。我可以据此迭代更新。 
