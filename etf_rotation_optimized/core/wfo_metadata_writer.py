"""
WFO元数据记录器

记录WFO运行的完整上下文：配置、环境、Git提交、网格参数等
确保结果可复现、可追溯
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml


class WFOMetadataWriter:
    """WFO元数据记录器"""

    @staticmethod
    def _get_git_info() -> Dict[str, str]:
        """获取Git信息"""
        try:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )

            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )

            is_dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )

            return {"commit_hash": commit_hash, "branch": branch, "is_dirty": is_dirty}
        except Exception:
            return {"commit_hash": "unknown", "branch": "unknown", "is_dirty": False}

    @staticmethod
    def _get_env_info() -> Dict[str, str]:
        """获取环境信息"""
        import sys

        return {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        }

    @staticmethod
    def write_metadata(
        out_dir: Path,
        config_path: Optional[Path] = None,
        phase2_params: Optional[Dict[str, Any]] = None,
        wfo_results_count: int = 0,
        strategies_count: int = 0,
    ):
        """
        写入元数据到metadata.json

        Args:
            out_dir: 输出目录
            config_path: 配置文件路径
            phase2_params: Phase 2参数
            wfo_results_count: WFO窗口数
            strategies_count: 枚举策略数
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "git": WFOMetadataWriter._get_git_info(),
            "environment": WFOMetadataWriter._get_env_info(),
            "wfo": {
                "windows_count": wfo_results_count,
                "strategies_enumerated": strategies_count,
            },
        }

        # 加载配置快照
        if config_path and config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                metadata["config_snapshot"] = yaml.safe_load(f)

        # Phase 2参数
        if phase2_params:
            metadata["phase2_params"] = phase2_params

        # 写入
        metadata_path = out_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
