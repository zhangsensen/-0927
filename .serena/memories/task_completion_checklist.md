# Task Completion Checklist
- Ensure affected subsystems pass `pytest -v` or targeted test commands.
- Run formatting (`black`, `isort`) and lint hooks (`pre-commit run --all-files`) when modifying Python code.
- Update relevant Markdown documentation if features or workflows change.
- Regenerate necessary parquet/csv artifacts only when explicitly required; avoid committing large data outputs.
- Summarize verification steps (tests run, key metrics) in the delivery note or PR description.