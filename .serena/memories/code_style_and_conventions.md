# Code Style & Conventions

- Language: Python 3.10+
- Libraries: pandas, numpy, pyarrow, yaml, sklearn (planned), lightgbm (optional), ty Loaders.
- Styling: prefer vectorized operations, avoid .apply loops; functions < 50 LOC; explicit arguments, dataclasses for metadata; logging via standard logging module.
- Naming: UPPER_SNAKE for constants, lower_snake for functions/variables, PascalCase for classes.
- Type hints: encouraged across modules; dataclasses used for structured records.
- Documentation: module docstrings summarizing responsibilities; inline comments only for non-obvious logic.
- Error handling: raise explicit ValueError/RuntimeError with contextual message; no silent pass.
- Testing: pytest under tests/ (needs expansion for new features).
