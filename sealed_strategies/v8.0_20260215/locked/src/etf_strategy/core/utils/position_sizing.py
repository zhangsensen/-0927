"""Shared helpers for dynamic position sizing.

Both VEC and BT engines must use the same mapping to maintain alignment.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple


def resolve_dynamic_pos_size(
    regime_exposure: float,
    *,
    regime_map: Sequence[Tuple[float, int]],
    base_pos_size: int = 2,
    min_pos_size: int = 1,
    max_pos_size: int = 4,
) -> int:
    """Determine effective position count from regime exposure.

    Parameters
    ----------
    regime_exposure : float
        Current regime-gate exposure value in [0, 1].
    regime_map : sequence of (threshold, pos_size)
        Sorted ascending by threshold.  Each entry means:
        "if exposure <= threshold, use this pos_size".
        Example: [(0.4, 1), (0.7, 2), (1.0, 3)]
    base_pos_size : int
        Fallback if no regime_map entry matches.
    min_pos_size, max_pos_size : int
        Hard bounds.

    Returns
    -------
    int : effective position count, clamped to [min_pos_size, max_pos_size].
    """
    pos = base_pos_size
    for threshold, size in regime_map:
        if regime_exposure <= threshold:
            pos = size
            break
    else:
        # exposure exceeds all thresholds — use the last entry's size
        if regime_map:
            pos = regime_map[-1][1]

    return max(min_pos_size, min(max_pos_size, pos))


def parse_dynamic_pos_config(config: Dict) -> Dict:
    """Parse dynamic_pos_size config section into a normalized dict.

    Parameters
    ----------
    config : dict
        The ``backtest`` section of the YAML config.

    Returns
    -------
    dict with keys:
        enabled (bool), mode (str), base_pos_size (int),
        min_pos_size (int), max_pos_size (int),
        regime_map (list of (threshold, pos_size) tuples)
    """
    dps = config.get("dynamic_pos_size") or {}
    enabled = bool(dps.get("enabled", False))

    base_pos_size = int(config.get("pos_size", 2))

    if not enabled:
        return {
            "enabled": False,
            "mode": "fixed",
            "base_pos_size": base_pos_size,
            "min_pos_size": base_pos_size,
            "max_pos_size": base_pos_size,
            "regime_map": [],
        }

    mode = str(dps.get("mode", "regime"))
    min_ps = int(dps.get("min_pos_size", 1))
    max_ps = int(dps.get("max_pos_size", 4))

    # Build regime_map: list of (threshold, pos_size), ascending by threshold
    regime_map_raw = dps.get("regime_map") or {}
    # Default mapping if not specified
    if not regime_map_raw:
        regime_map_raw = {
            "high_vol": 1,
            "mid_vol": 2,
            "low_vol": 3,
        }

    # Convert named zones to (threshold, pos_size)
    # Exposure mapping: high_vol → exposure <= 0.4, mid_vol → <= 0.7, low_vol → <= 1.0
    _ZONE_THRESHOLDS = {
        "high_vol": 0.4,
        "mid_vol": 0.7,
        "low_vol": 1.0,
    }

    regime_map = []
    for zone, ps in sorted(regime_map_raw.items(), key=lambda kv: _ZONE_THRESHOLDS.get(kv[0], 1.0)):
        threshold = _ZONE_THRESHOLDS.get(zone, 1.0)
        regime_map.append((threshold, int(ps)))

    return {
        "enabled": True,
        "mode": mode,
        "base_pos_size": base_pos_size,
        "min_pos_size": min_ps,
        "max_pos_size": max_ps,
        "regime_map": regime_map,
    }


def resolve_pos_size_for_day(
    dps_config: Dict,
    regime_exposure: float,
) -> int:
    """Convenience wrapper: resolve pos_size for a given day.

    Parameters
    ----------
    dps_config : dict
        Output from ``parse_dynamic_pos_config()``.
    regime_exposure : float
        Current regime-gate exposure for the day.

    Returns
    -------
    int : effective position count.
    """
    if not dps_config.get("enabled", False):
        return dps_config["base_pos_size"]

    return resolve_dynamic_pos_size(
        regime_exposure,
        regime_map=dps_config["regime_map"],
        base_pos_size=dps_config["base_pos_size"],
        min_pos_size=dps_config["min_pos_size"],
        max_pos_size=dps_config["max_pos_size"],
    )
