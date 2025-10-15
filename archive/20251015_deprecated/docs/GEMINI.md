# GEMINI.md - 深度量化0927

## Project Overview

This project, "深度量化0927," is a professional-grade quantitative trading framework focused on A-shares and Hong Kong stocks. It provides a complete, production-level factor engineering system for developing and backtesting quantitative trading strategies. The core of the project is a factor engine that supports a wide range of technical and fundamental factors, with a recent focus on money flow analysis.

The project is built using Python and a rich ecosystem of scientific computing and financial libraries, including:

*   **Core Libraries:** `pandas`, `numpy`, `scipy`, `scikit-learn`
*   **Backtesting:** `vectorbt`
*   **Technical Analysis:** `ta-lib`
*   **Data Handling:** `pyarrow`, `fastparquet`, `polars`
*   **Development and Quality:** `pytest`, `black`, `isort`, `mypy`, `ruff`, `pre-commit`

The project is well-structured and follows modern software engineering best practices, with a clear separation of concerns between data pipelines, factor calculation, backtesting, and analysis.

## Building and Running

The project uses a `Makefile` to streamline common development tasks. Here are the key commands:

*   **Install dependencies:**
    ```bash
    make install
    ```
*   **Format code:**
    ```bash
    make format
    ```
*   **Run linters and type checkers:**
    ```bash
    make lint
    ```
*   **Run tests:**
    ```bash
    make test
    ```
*   **Run a factor screening example:**
    ```bash
    make run-example
    ```

## Development Conventions

The project enforces a strict set of development conventions to ensure code quality and consistency:

*   **Code Style:** Code is formatted using `black` and `isort`.
*   **Linting:** `flake8` and `ruff` are used to check for code quality issues.
*   **Type Checking:** `mypy` is used for static type checking.
*   **Testing:** `pytest` is used for unit and integration testing.
*   **Pre-commit Hooks:** `pre-commit` is used to automatically run checks before each commit.

## Key Files and Directories

*   `README.md`: Provides a high-level overview of the project, its goals, and its current status.
*   `pyproject.toml`: Defines the project's dependencies and development tools.
*   `Makefile`: Contains a set of commands for building, testing, and running the project.
*   `factor_system/`: The main source code directory, containing the core logic for factor engineering, backtesting, and analysis.
*   `factor_system/FACTOR_REGISTRY.md`: A comprehensive registry of all the factors implemented in the system.
*   `a_shares_strategy/`: Contains code specific to A-shares strategies.
*   `hk_midfreq/`: Contains code specific to Hong Kong mid-frequency trading strategies.
*   `data/`: The directory for storing raw and processed data.
*   `tests/`: Contains the project's test suite.

## Usage

This project is intended to be used as a framework for developing and backtesting quantitative trading strategies. To get started, you can:

1.  Install the dependencies using `make install`.
2.  Explore the example factor screening task by running `make run-example`.
3.  Develop your own factors and strategies by extending the existing codebase.
4.  Use the provided testing and quality assurance tools to ensure the correctness and robustness of your code.
