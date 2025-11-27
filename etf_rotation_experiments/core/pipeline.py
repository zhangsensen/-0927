"""ETF轮动系统精简流水线。

该版本去除了历史遗留的多阶段处理，仅保留核心流程：
1. 加载OHLCV数据并计算精确因子
2. 对因子进行横截面标准化
3. 执行直接因子级WFO并输出窗口指标

目标是提供一个可维护、易理解的基线实现。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from .cross_section_processor import CrossSectionProcessor
from .data_loader import DataLoader
from strategies.wfo.direct_factor_wfo_optimizer import DirectFactorWFOOptimizer
from .precise_factor_library_v2 import PreciseFactorLibrary

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """管线配置包装。"""

    run_id: str
    data: Dict
    cross_section: Dict
    wfo: Dict
    output_root: Path


class Pipeline:
    """ETF轮动系统的最小流水线封装。"""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.output_root = config.output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.cross_section_dir = self.output_root / "cross_section"
        self.factor_selection_dir = self.output_root / "factor_selection"
        self.wfo_dir = self.output_root / "wfo"
        for path in (self.cross_section_dir, self.factor_selection_dir, self.wfo_dir):
            path.mkdir(parents=True, exist_ok=True)

        self.ohlcv_data: Optional[Dict[str, pd.DataFrame]] = None
        self.factors_dict: Optional[Dict[str, pd.DataFrame]] = None
        self.standardized_factors: Optional[Dict[str, pd.DataFrame]] = None
        self.wfo_results: Optional[pd.DataFrame] = None

        self._setup_logging()

    @classmethod
    def from_config(cls, config_path: Path) -> "Pipeline":
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        pipeline_cfg = PipelineConfig(
            run_id=cfg.get("run_id", "ETF_PIPELINE"),
            data=cfg.get("data", {}),
            cross_section=cfg.get("cross_section", {}),
            wfo=cfg.get("wfo", {}),
            output_root=Path(__file__).parent.parent / cfg.get("output_root", "results"),
        )
        return cls(pipeline_cfg)

    def _setup_logging(self) -> None:
        self.log_dir = self.output_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / "pipeline.log"

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
            force=True,
        )
        logger.info("ETF轮动系统流水线初始化完毕 | 输出目录=%s", self.output_root)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info("开始执行流水线 run_id=%s", self.config.run_id)
        self._run_cross_section()
        self._run_factor_selection()
        self._run_wfo()
        logger.info("流水线执行完成")

    # ------------------------------------------------------------------
    # Step 1: 数据加载与因子计算
    # ------------------------------------------------------------------
    def _run_cross_section(self) -> None:
        if self.ohlcv_data is None:
            loader = DataLoader(
                data_dir=self.config.data.get("data_dir"),
                cache_dir=self.config.data.get("cache_dir"),
            )
            symbols = self.config.data.get("symbols")
            start_date = self.config.data.get("start_date")
            end_date = self.config.data.get("end_date")
            self.ohlcv_data = loader.load_ohlcv(
                etf_codes=symbols,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
            )
            logger.info(
                "数据加载完成: %d 只ETF × %d 个交易日",
                len(self.ohlcv_data["close"].columns),
                len(self.ohlcv_data["close"]),
            )

        factor_lib = PreciseFactorLibrary()
        factors_df = factor_lib.compute_all_factors(prices=self.ohlcv_data)

        self.factors_dict = {
            name: factors_df[name] for name in factor_lib.list_factors()
        }
        factor_path = self.cross_section_dir / "factors"
        factor_path.mkdir(parents=True, exist_ok=True)
        for name, df in self.factors_dict.items():
            df.to_parquet(factor_path / f"{name}.parquet")

        ohlcv_path = self.cross_section_dir / "ohlcv"
        ohlcv_path.mkdir(parents=True, exist_ok=True)
        for field, df in self.ohlcv_data.items():
            df.to_parquet(ohlcv_path / f"{field}.parquet")

        metadata = {
            "run_id": self.config.run_id,
            "symbols": list(self.ohlcv_data["close"].columns),
            "date_range": [
                str(self.ohlcv_data["close"].index[0].date()),
                str(self.ohlcv_data["close"].index[-1].date()),
            ],
            "factor_count": len(self.factors_dict),
        }
        with open(
            self.cross_section_dir / "metadata.json", "w", encoding="utf-8"
        ) as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)
        logger.info("因子计算完成，共 %d 个因子", len(self.factors_dict))

    # ------------------------------------------------------------------
    # Step 2: 因子标准化
    # ------------------------------------------------------------------
    def _run_factor_selection(self) -> None:
        if not self.factors_dict:
            raise RuntimeError("未找到因子结果，请先执行 _run_cross_section")

        winsor_lower = float(self.config.cross_section.get("winsorize_lower", 0.025))
        winsor_upper = float(self.config.cross_section.get("winsorize_upper", 0.975))
        if winsor_lower <= 1.0:
            winsor_lower *= 100.0
        if winsor_upper <= 1.0:
            winsor_upper *= 100.0
        processor = CrossSectionProcessor(
            lower_percentile=winsor_lower,
            upper_percentile=winsor_upper,
            verbose=False,
        )

        bounded_override = self.config.cross_section.get("bounded_factors")
        if bounded_override:
            processor.BOUNDED_FACTORS = set(bounded_override)

        standardized = processor.process_all_factors(self.factors_dict)
        self.standardized_factors = standardized
        output_dir = self.factor_selection_dir / "standardized"
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in standardized.items():
            df.to_parquet(output_dir / f"{name}.parquet")
        logger.info("因子标准化完成，共保存 %d 个文件", len(standardized))

    # ------------------------------------------------------------------
    # Step 3: WFO
    # ------------------------------------------------------------------
    def _run_wfo(self) -> None:
        if not self.standardized_factors:
            raise RuntimeError("未找到标准化因子，请先执行 _run_factor_selection")
        if self.ohlcv_data is None:
            raise RuntimeError("缺少OHLCV数据")

        factor_names = sorted(self.standardized_factors.keys())
        factor_arrays: List[np.ndarray] = [
            self.standardized_factors[name].values for name in factor_names
        ]
        factors_tensor = np.stack(factor_arrays, axis=-1)  # (T, N, F)

        returns_df = self.ohlcv_data["close"].pct_change(fill_method=None)
        returns_array = returns_df.values

        warmup = self.config.wfo.get("warmup", 0)
        if warmup > 0:
            factors_tensor = factors_tensor[warmup:]
            returns_array = returns_array[warmup:]
            logger.info("丢弃前 %d 天数据以满足指标预热", warmup)

        optimizer = DirectFactorWFOOptimizer(
            factor_weighting=self.config.wfo.get("factor_weighting", "ic_weighted"),
            min_factor_ic=self.config.wfo.get("min_factor_ic", 0.01),
            ic_floor=self.config.wfo.get("ic_floor", 0.0),
            verbose=True,
        )

        window_results, summary_df = optimizer.run_wfo(
            factors_data=factors_tensor,
            returns=returns_array,
            factor_names=factor_names,
            is_period=int(self.config.wfo.get("is_period", 252)),
            oos_period=int(self.config.wfo.get("oos_period", 60)),
            step_size=int(self.config.wfo.get("step_size", 20)),
        )

        self.wfo_results = summary_df
        summary_path = self.wfo_dir / "wfo_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        detail_path = self.wfo_dir / "window_results.json"
        with open(detail_path, "w", encoding="utf-8") as fh:
            json.dump(
                [
                    {
                        "window_index": item.window_index,
                        "is_start": item.is_start,
                        "is_end": item.is_end,
                        "oos_start": item.oos_start,
                        "oos_end": item.oos_end,
                        "selected_factors": item.selected_factors,
                        "factor_weights": item.factor_weights,
                        "oos_ic": item.oos_ic,
                        "oos_ir": item.oos_ir,
                        "positive_rate": item.positive_rate,
                    }
                    for item in window_results
                ],
                fh,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("WFO 完成，结果已保存至 %s", summary_path)
