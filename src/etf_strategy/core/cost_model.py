"""Cost model configuration for backtesting engines.

Supports two modes:
    UNIFIED: Single commission rate for all ETFs (legacy compatibility).
    SPLIT_MARKET: Different costs for A-share vs QDII ETFs, with low/med/high tiers.

Three-tier cost schedule (one-way, bps):
    | Tier | A-share | QDII | Description                            |
    |------|---------|------|----------------------------------------|
    | low  |  10     |  30  | Optimistic: lowest commissions + spread|
    | med  |  20     |  50  | Baseline: commission + spread + impact |
    | high |  30     |  80  | Pessimistic: redemption + cross-border |
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostTier:
    """One-way transaction cost for a market segment (decimal, e.g. 0.0020 = 20bps)."""

    a_share: float
    qdii: float

    def __post_init__(self):
        if self.a_share < 0 or self.qdii < 0:
            raise ValueError(
                f"Cost rates must be non-negative: a_share={self.a_share}, qdii={self.qdii}"
            )


# Default tier definitions (one-way, decimal)
_DEFAULT_TIERS: dict[str, CostTier] = {
    "low": CostTier(a_share=0.0010, qdii=0.0030),
    "med": CostTier(a_share=0.0020, qdii=0.0050),
    "high": CostTier(a_share=0.0030, qdii=0.0080),
}

_VALID_MODES = ("UNIFIED", "SPLIT_MARKET")
_VALID_TIERS = ("low", "med", "high")


@dataclass(frozen=True)
class CostModel:
    """Cost model for the three-tier engine (WFO/VEC/BT).

    Attributes:
        mode: "UNIFIED" (single rate) or "SPLIT_MARKET" (A-share vs QDII).
        tier: Active cost tier ("low", "med", "high").
        tiers: Mapping of tier name -> CostTier.
        unified_rate: Fallback rate used in UNIFIED mode (decimal).
    """

    mode: str = "SPLIT_MARKET"
    tier: str = "med"
    tiers: tuple[tuple[str, float, float], ...] = ()  # (name, a_share, qdii)
    unified_rate: float = 0.0002

    def __post_init__(self):
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown cost model mode: {self.mode!r}. Use one of {_VALID_MODES}."
            )
        if self.tier not in _VALID_TIERS:
            raise ValueError(
                f"Unknown cost tier: {self.tier!r}. Use one of {_VALID_TIERS}."
            )

    @property
    def is_split_market(self) -> bool:
        return self.mode == "SPLIT_MARKET"

    @property
    def active_tier(self) -> CostTier:
        """Return the CostTier for the currently selected tier."""
        if not self.is_split_market:
            return CostTier(a_share=self.unified_rate, qdii=self.unified_rate)
        for name, a, q in self.tiers:
            if name == self.tier:
                return CostTier(a_share=a, qdii=q)
        return _DEFAULT_TIERS.get(self.tier, _DEFAULT_TIERS["med"])

    def get_cost(self, etf_code: str, qdii_codes: set[str]) -> float:
        """Return one-way cost (decimal) for a given ETF."""
        t = self.active_tier
        if etf_code in qdii_codes:
            return t.qdii
        return t.a_share

    def with_tier(self, tier: str) -> CostModel:
        """Return a new CostModel with a different active tier."""
        return CostModel(
            mode=self.mode,
            tier=tier,
            tiers=self.tiers,
            unified_rate=self.unified_rate,
        )


def load_cost_model(config: dict) -> CostModel:
    """Load CostModel from config dict.

    Falls back to UNIFIED mode using backtest.commission_rate when
    the cost_model section is absent.
    """
    cm_cfg = config.get("backtest", {}).get("cost_model")
    if cm_cfg is None:
        # Legacy fallback: use scalar commission_rate
        rate = float(config.get("backtest", {}).get("commission_rate", 0.0002))
        return CostModel(mode="UNIFIED", tier="med", tiers=(), unified_rate=rate)

    mode = cm_cfg.get("mode", "SPLIT_MARKET")
    tier = cm_cfg.get("tier", "med")

    # Parse tiers from config
    tiers_cfg = cm_cfg.get("tiers", {})
    tiers_list: list[tuple[str, float, float]] = []
    for tier_name in _VALID_TIERS:
        if tier_name in tiers_cfg:
            t = tiers_cfg[tier_name]
            tiers_list.append(
                (tier_name, float(t["a_share"]), float(t["qdii"]))
            )
        elif tier_name in _DEFAULT_TIERS:
            d = _DEFAULT_TIERS[tier_name]
            tiers_list.append((tier_name, d.a_share, d.qdii))

    unified_rate = float(config.get("backtest", {}).get("commission_rate", 0.0002))

    return CostModel(
        mode=mode,
        tier=tier,
        tiers=tuple(tiers_list),
        unified_rate=unified_rate,
    )


def build_cost_array(
    cost_model: CostModel,
    etf_codes: list[str],
    qdii_codes: set[str],
) -> np.ndarray:
    """Build a 1D cost array (shape N,) for use in VEC/WFO kernels.

    Each element is the one-way transaction cost (decimal) for the corresponding ETF.
    """
    n = len(etf_codes)
    cost_arr = np.empty(n, dtype=np.float64)
    for i, code in enumerate(etf_codes):
        cost_arr[i] = cost_model.get_cost(code, qdii_codes)
    return cost_arr
