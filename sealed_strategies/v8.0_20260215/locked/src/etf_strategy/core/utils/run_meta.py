"""Pipeline step metadata writer.

Each pipeline step calls ``write_step_meta()`` to record a compact
``_meta.json`` next to its outputs so that downstream consumers (and
the registry viewer) know exactly where the data came from.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def _git_short_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def write_step_meta(
    output_dir: Path,
    *,
    step: str,
    inputs: dict[str, str] | None = None,
    config: str | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    """Write ``_meta.json`` into *output_dir*.

    Parameters
    ----------
    output_dir : Path
        The directory where results were saved.
    step : str
        Pipeline step name, e.g. ``"wfo"``, ``"vec"``, ``"bt"``.
    inputs : dict, optional
        Mapping of input name to path / directory used.
    config : str, optional
        Path to the config YAML that governed this run.
    extras : dict, optional
        Arbitrary extra key/value pairs (counts, thresholds, etc.).
    """
    meta: dict[str, Any] = {
        "step": step,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_short_hash(),
    }
    if config:
        meta["config"] = config
    if inputs:
        meta["inputs"] = inputs
    if extras:
        meta.update(extras)

    path = Path(output_dir) / "_meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    return path
