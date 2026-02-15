"""Execution model configuration for backtesting engines.

Controls how orders are filled across all three engines (WFO/VEC/BT).

Modes:
    COC (Cheat-On-Close): Signal at close(T-1) → fill at close(T). Legacy mode.
    T1_OPEN: Signal at close(T-1) → fill at open(T+1). Realistic execution.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionModel:
    mode: str = "T1_OPEN"  # "COC" or "T1_OPEN"

    @property
    def is_t1_open(self) -> bool:
        return self.mode == "T1_OPEN"

    @property
    def is_coc(self) -> bool:
        return self.mode == "COC"

    def __post_init__(self):
        if self.mode not in ("COC", "T1_OPEN"):
            raise ValueError(f"Unknown execution mode: {self.mode!r}. Use 'COC' or 'T1_OPEN'.")


def load_execution_model(config: dict) -> ExecutionModel:
    """Load ExecutionModel from config dict.

    Checks both top-level and backtest section for execution_model key.
    """
    mode = config.get("execution_model")
    if mode is None:
        mode = config.get("backtest", {}).get("execution_model", "COC")
    return ExecutionModel(mode=mode)
