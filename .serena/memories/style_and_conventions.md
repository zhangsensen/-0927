# Code Style & Conventions
- Language: Python 3.11+, adhere to PEP 8 with pragmatic deviations when performance critical.
- Type hints: used where helpful; LightGBM/NumPy code mixes typed numpy arrays and pandas frames.
- Documentation: Markdown-heavy, README-driven; code comments used sparingly to explain nontrivial logic.
- Testing expectation: new features require pytest coverage and doc updates.
- Architecture: modules organized by subsystem; prefer vectorized pandas/numpy operations, avoid loops; maintain no-forward-looking data via strict execution timing (14:30 freeze, T+1 trading).
- Git: concise commits tied to business outcomes; respect performance-freeze rules unless explicitly unlocked.