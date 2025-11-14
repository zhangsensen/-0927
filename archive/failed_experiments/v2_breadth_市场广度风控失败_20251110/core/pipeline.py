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
from .correlation_monitor import CorrelationMonitor
from .data_loader import DataLoader
from .direct_factor_wfo_optimizer import DirectFactorWFOOptimizer
from .market_breadth import MarketBreadthMonitor
from .precise_factor_library_v2 import PreciseFactorLibrary
from .volatility_target import VolatilityTargeting

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """管线配置包装。"""

    run_id: str
    data: Dict
    cross_section: Dict
    wfo: Dict
    output_root: Path
    risk_control: Optional[Dict] = None  # V2新增：风控配置


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

        # V2新增：风控模块
        self.breadth_monitor: Optional[MarketBreadthMonitor] = None
        self.vol_target: Optional[VolatilityTargeting] = None
        self.corr_monitor: Optional[CorrelationMonitor] = None
        self._init_risk_control()

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
            output_root=Path(cfg.get("output_root", "results")),
            risk_control=cfg.get("risk_control"),  # V2新增
        )
        return cls(pipeline_cfg)

    def _init_risk_control(self) -> None:
        """初始化风控模块（V2新增）"""
        if not self.config.risk_control:
            logger.info("未配置风控模块，跳过初始化")
            return

        rc = self.config.risk_control

        # 市场广度
        if rc.get("market_breadth", {}).get("enabled", False):
            mb_cfg = rc["market_breadth"]
            self.breadth_monitor = MarketBreadthMonitor(
                breadth_floor=mb_cfg.get("breadth_floor", 0.25),
                score_threshold=mb_cfg.get("score_threshold", 0.0),
                defensive_scale=mb_cfg.get("defensive_scale", 0.5),
                verbose=mb_cfg.get("verbose", True),
            )
            logger.info(
                "✅ 市场广度监控已启用: floor=%.0f%%, defensive_scale=%.0f%%",
                mb_cfg.get("breadth_floor", 0.25) * 100,
                mb_cfg.get("defensive_scale", 0.5) * 100,
            )

        # 波动率目标
        if rc.get("volatility_target", {}).get("enabled", False):
            vt_cfg = rc["volatility_target"]
            self.vol_target = VolatilityTargeting(
                target_vol=vt_cfg.get("target_vol", 0.30),
                min_window=vt_cfg.get("min_window", 20),
                max_scale=vt_cfg.get("max_scale", 1.0),
                min_scale=vt_cfg.get("min_scale", 0.3),
                verbose=vt_cfg.get("verbose", True),
            )
            logger.info(
                "✅ 波动率目标已启用: target=%.0f%%",
                vt_cfg.get("target_vol", 0.30) * 100,
            )

        # 相关性监控
        if rc.get("correlation_monitor", {}).get("enabled", False):
            cm_cfg = rc["correlation_monitor"]
            self.corr_monitor = CorrelationMonitor(
                corr_threshold=cm_cfg.get("corr_threshold", 0.65),
                window=cm_cfg.get("window", 20),
                min_penalty=cm_cfg.get("min_penalty", 0.5),
                verbose=cm_cfg.get("verbose", True),
            )
            logger.info(
                "✅ 相关性监控已启用: threshold=%.2f",
                cm_cfg.get("corr_threshold", 0.65),
            )

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

        # V2新增：应用风控层
        if any([self.breadth_monitor, self.vol_target, self.corr_monitor]):
            self._apply_risk_control(
                factors_tensor=factors_tensor,
                returns_array=returns_array,
                factor_names=factor_names,
                window_results=window_results,
                dates=returns_df.index[warmup:] if warmup > 0 else returns_df.index,
            )

    def _apply_risk_control(
        self,
        factors_tensor: np.ndarray,  # (T, N, F)
        returns_array: np.ndarray,  # (T, N)
        factor_names: List[str],
        window_results: List,
        dates: pd.DatetimeIndex,
    ) -> None:
        """应用风控层并记录触发日志（V2新增）
        
        逻辑：
        1. 根据WFO窗口结果重建每日复合因子得分
        2. 调用三个监控器计算position_scale
        3. 根据综合策略（min/multiply/max）合并
        4. 记录触发日志到CSV
        """
        logger.info("开始应用风控层...")
        
        T, N, F = factors_tensor.shape
        rc_cfg = self.config.risk_control or {}
        combine_strategy = rc_cfg.get("combine_strategy", "min")
        
        # 1. 重建每日复合因子得分 (T, N)
        composite_scores = self._rebuild_composite_scores(
            factors_tensor, factor_names, window_results, dates
        )
        
        # 2. 初始化日志记录
        log_records = []
        
        # 3. 逐日应用风控
        for t, date in enumerate(dates):
            daily_scores = composite_scores[t]  # (N,)
            daily_returns = returns_array[t] if t > 0 else None
            
            scales = []
            breadth_val = 1.0
            vol_val = 1.0
            corr_val = 1.0
            
            # 市场广度
            if self.breadth_monitor:
                breadth_signal = self.breadth_monitor.calculate_breadth(
                    factor_scores=daily_scores, date=date
                )
                breadth_val = breadth_signal.position_scale
                scales.append(breadth_val)
            
            # 波动率目标
            if self.vol_target and daily_returns is not None and t >= self.vol_target.min_window:
                vol_signal = self.vol_target.calculate_volatility(
                    returns=returns_array[:t+1], date=date
                )
                vol_val = vol_signal.position_scale
                scales.append(vol_val)
            
            # 相关性监控
            if self.corr_monitor and t >= self.corr_monitor.window:
                corr_signal = self.corr_monitor.calculate_correlation(
                    factors=factors_tensor[t-self.corr_monitor.window+1:t+1],
                    date=date,
                )
                corr_val = corr_signal.penalty
                scales.append(corr_val)
            
            # 4. 综合策略
            if scales:
                if combine_strategy == "min":
                    final_scale = min(scales)
                elif combine_strategy == "multiply":
                    final_scale = np.prod(scales)
                elif combine_strategy == "max":
                    final_scale = max(scales)
                else:
                    final_scale = min(scales)
                
                # 记录
                log_records.append({
                    "date": date,
                    "breadth_scale": breadth_val,
                    "vol_scale": vol_val,
                    "corr_penalty": corr_val,
                    "final_scale": final_scale,
                    "combine_strategy": combine_strategy,
                })
        
        # 5. 保存日志
        if log_records:
            log_df = pd.DataFrame(log_records)
            log_path = self.wfo_dir / "risk_control_log.csv"
            log_df.to_csv(log_path, index=False)
            logger.info("风控日志已保存至 %s（%d天）", log_path, len(log_df))
            
            # 统计触发情况
            triggered = log_df[log_df["final_scale"] < 1.0]
            if not triggered.empty:
                logger.info(
                    "触发防守: %d天/%.1f%% | 平均缩仓: %.0f%%",
                    len(triggered),
                    len(triggered) / len(log_df) * 100,
                    (1 - triggered["final_scale"].mean()) * 100,
                )
        
        # 6. 输出各模块统计
        if self.breadth_monitor:
            stats = self.breadth_monitor.get_statistics()
            logger.info(
                "市场广度: 触发%d天 | 平均%.1f%% | 最低%.1f%%",
                stats["defensive_days"],
                stats["mean_breadth"] * 100,
                stats["min_breadth"] * 100,
            )
        
        if self.vol_target:
            stats = self.vol_target.get_statistics()
            logger.info(
                "波动率目标: 触发%d天 | 平均波动%.1f%% | 最高%.1f%%",
                stats["high_vol_days"],
                stats["mean_volatility"] * 100,
                stats["max_volatility"] * 100,
            )
        
        if self.corr_monitor:
            stats = self.corr_monitor.get_statistics()
            logger.info(
                "相关性监控: 触发%d天 | 平均相关%.2f | 最高%.2f",
                stats["high_corr_days"],
                stats["mean_correlation"],
                stats["max_correlation"],
            )
    
    def _rebuild_composite_scores(
        self,
        factors_tensor: np.ndarray,  # (T, N, F)
        factor_names: List[str],
        window_results: List,
        dates: pd.DatetimeIndex,
    ) -> np.ndarray:
        """根据WFO窗口结果重建每日复合因子得分 (T, N)
        
        逻辑：每个WFO窗口在OOS期间应用固定的因子权重，
             将(N, F)因子矩阵与(F,)权重向量点积得到(N,)得分。
        """
        T, N, F = factors_tensor.shape
        composite_scores = np.zeros((T, N), dtype=np.float32)
        
        for window in window_results:
            oos_start_idx = window.oos_start
            oos_end_idx = window.oos_end
            
            # 构建权重向量 (F,)
            weights = np.zeros(F, dtype=np.float32)
            for fname, w in window.factor_weights.items():
                if fname in factor_names:
                    idx = factor_names.index(fname)
                    weights[idx] = w
            
            # 向量化计算：(T_oos, N, F) @ (F,) -> (T_oos, N)
            oos_factors = factors_tensor[oos_start_idx:oos_end_idx+1]  # (T_oos, N, F)
            oos_scores = np.dot(oos_factors, weights)  # (T_oos, N)
            composite_scores[oos_start_idx:oos_end_idx+1] = oos_scores
        
        return composite_scores

