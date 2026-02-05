#!/usr/bin/env python3
"""Print the pipeline registry as a formatted table.

Usage:
    uv run python scripts/view_registry.py
"""

import json
from pathlib import Path

REGISTRY = Path(__file__).parent.parent / "results" / "REGISTRY.jsonl"


def main() -> None:
    if not REGISTRY.exists():
        print("No registry found. Run the full pipeline first.")
        return

    entries = [json.loads(line) for line in REGISTRY.read_text().splitlines() if line.strip()]
    if not entries:
        print("Registry is empty.")
        return

    hdr = f"{'ID':<28} {'Date':<12} {'Cands':>5}  {'Train%':>7}  {'Holdout%':>8}  {'Gate':<4}  {'Top Combo'}"
    print(hdr)
    print("-" * len(hdr))

    for e in entries:
        s = e.get("result_summary", {})
        pid = e.get("pipeline_id", "?")
        date = e.get("created_at", "")[:10]
        cands = s.get("final_candidates", 0)
        train_pct = f"{s.get('top_train_return', 0) * 100:.1f}%"
        holdout_pct = f"{s.get('top_holdout_return', 0) * 100:.1f}%"
        gate = e.get("regime_gate", "?")
        combo = s.get("top_combo", "")
        # Abbreviate combo: keep only short factor names
        short_combo = " + ".join(
            f.replace("_20D", "").replace("_60D", "").replace("_14D", "").replace("_120D", "")
            for f in combo.split(" + ")
        )
        if len(short_combo) > 50:
            short_combo = short_combo[:47] + "..."
        print(f"{pid:<28} {date:<12} {cands:>5}  {train_pct:>7}  {holdout_pct:>8}  {gate:<4}  {short_combo}")


if __name__ == "__main__":
    main()
