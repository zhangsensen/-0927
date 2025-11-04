"""
流程编排器 | Pipeline Orchestrator

统一执行入口，替代 scripts/step*.py 的手动流程

工作流:
  配置文件 → Pipeline.from_config()
    ↓
  横截面加工 (cross_section)
    ↓
  因子筛选 (factor_selection)
    ↓
  WFO验证 (wfo)
    ↓
  VBT回测 (backtest)

作者: Linus Refactor
日期: 2025-10-28
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from .cross_section_processor import CrossSectionProcessor
from .precise_factor_library_v2 import PreciseFactorLibrary

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """流程配置"""

    run_id: str
    data: Dict
    cross_section: Dict
    factor_selection: Dict
    wfo: Dict
    backtest: Dict
    output_root: Path


class Pipeline:
    """
    ETF轮动系统流程编排器

    负责按顺序执行: 横截面 -> 因子筛选 -> WFO -> 回测
    每个阶段输出落盘，支持断点续跑
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_date = self.timestamp[:8]

        # 输出目录
        self.output_root = config.output_root
        self.cross_section_dir = (
            self.output_root / "cross_section" / self.run_date / self.timestamp
        )
        self.factor_selection_dir = (
            self.output_root / "factor_selection" / self.run_date / self.timestamp
        )
        self.wfo_dir = self.output_root / "wfo" / self.run_date / self.timestamp
        self.backtest_dir = (
            self.output_root / "backtest" / self.run_date / self.timestamp
        )

        # 创建目录
        for d in [
            self.cross_section_dir,
            self.factor_selection_dir,
            self.wfo_dir,
            self.backtest_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # 日志
        self._setup_logging()

        # 数据容器
        self.ohlcv_data: Optional[Dict[str, pd.DataFrame]] = None
        self.factors_dict: Optional[Dict[str, pd.DataFrame]] = None
        self.standardized_factors: Optional[Dict[str, pd.DataFrame]] = None
        self.selected_factors: Optional[List[str]] = None
        self.wfo_results: Optional[pd.DataFrame] = None

    @classmethod
    def from_config(cls, config_path: Path) -> "Pipeline":
        """从配置文件创建Pipeline"""
        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        pipeline_config = PipelineConfig(
            run_id=config_dict.get("run_id", "DEFAULT_RUN"),
            data=config_dict.get("data", {}),
            cross_section=config_dict.get("cross_section", {}),
            factor_selection=config_dict.get("factor_selection", {}),
            wfo=config_dict.get("wfo", {}),
            backtest=config_dict.get("backtest", {}),
            output_root=Path(config_dict.get("output_root", "results")),
        )

        return cls(config=pipeline_config)

    def _setup_logging(self):
        """配置日志"""
        log_file = self.output_root / "logs" / f"pipeline_{self.timestamp}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

        logger.info("=" * 80)
        logger.info("ETF轮动系统 - 流程启动")
        logger.info("=" * 80)
        logger.info(f"Run ID: {self.config.run_id}")
        logger.info(f"时间戳: {self.timestamp}")
        logger.info(f"输出根目录: {self.output_root}")
        logger.info("")

    def run(self):
        """运行完整流程"""
        logger.info("🚀 开始完整流程")
        logger.info("")

        self.run_step("cross_section")
        self.run_step("factor_selection")
        self.run_step("wfo")
        self.run_step("backtest")

        logger.info("=" * 80)
        logger.info("✅ 完整流程执行完成")
        logger.info("=" * 80)

    def run_step(self, step: str):
        """运行单个步骤"""
        if step == "cross_section":
            self._run_cross_section()
        elif step == "factor_selection":
            self._run_factor_selection()
        elif step == "wfo":
            self._run_wfo()
        elif step == "backtest":
            self._run_backtest()
        else:
            raise ValueError(f"未知步骤: {step}")

    def _run_cross_section(self):
        """横截面加工 - 加载数据并计算因子"""
        logger.info("-" * 80)
        logger.info("Step 1: 横截面加工")
        logger.info("-" * 80)

        # 1. 加载数据
        from .data_loader import DataLoader

        loader = DataLoader()
        symbols = self.config.data.get("symbols", [])
        start_date = self.config.data.get("start_date")
        end_date = self.config.data.get("end_date")

        logger.info(f"加载数据: {len(symbols)} 只标的")
        logger.info(f"日期范围: {start_date} -> {end_date}")

        self.ohlcv_data = loader.load_ohlcv(
            etf_codes=symbols, start_date=start_date, end_date=end_date
        )

        # 数据契约验证
        from .data_contract import DataContract

        DataContract.validate_ohlcv(self.ohlcv_data)

        data_summary = loader.get_summary(self.ohlcv_data)
        logger.info(f"✅ 数据加载完成: {data_summary['total_dates']} 天")
        logger.info("")

        # 2. 计算因子
        logger.info("计算精确因子...")
        lib = PreciseFactorLibrary()
        factors_df = lib.compute_all_factors(prices=self.ohlcv_data)

        # 转换为字典
        self.factors_dict = {}
        for factor_name in lib.list_factors():
            self.factors_dict[factor_name] = factors_df[factor_name]

        logger.info(f"✅ 因子计算完成: {len(self.factors_dict)} 个因子")
        logger.info("")

        # 3. 保存OHLCV
        ohlcv_dir = self.cross_section_dir / "ohlcv"
        ohlcv_dir.mkdir(exist_ok=True)

        for col_name, df in self.ohlcv_data.items():
            df.to_parquet(ohlcv_dir / f"{col_name}.parquet")

        logger.info(f"✅ OHLCV已保存: {ohlcv_dir}")

        # 4. 保存因子
        factors_dir = self.cross_section_dir / "factors"
        factors_dir.mkdir(exist_ok=True)

        for fname, fdata in self.factors_dict.items():
            if isinstance(fdata, pd.Series):
                df_to_save = fdata.to_frame(name=fname)
            else:
                df_to_save = fdata
            df_to_save.to_parquet(factors_dir / f"{fname}.parquet")

        logger.info(f"✅ 因子已保存: {factors_dir}")

        # 5. 保存元数据
        metadata = {
            "timestamp": self.timestamp,
            "step": "cross_section",
            "symbols": symbols,
            "date_range": [start_date, end_date],
            "factor_count": len(self.factors_dict),
            "factor_names": list(self.factors_dict.keys()),
        }

        with open(self.cross_section_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Step 1 完成: {self.cross_section_dir}")
        logger.info("")

    def _run_factor_selection(self):
        """因子筛选 - 标准化处理"""
        logger.info("-" * 80)
        logger.info("Step 2: 因子筛选（标准化）")
        logger.info("-" * 80)

        # 加载横截面数据（如果未加载）
        if self.factors_dict is None:
            self._load_cross_section_data()

        logger.info(
            f"📊 待处理因子数量: {len(self.factors_dict) if self.factors_dict else 0}"
        )

        # 标准化
        processor = CrossSectionProcessor()
        self.standardized_factors = processor.process_all_factors(self.factors_dict)

        logger.info(f"✅ 因子标准化完成: {len(self.standardized_factors)} 个")
        logger.info("")

        # 保存标准化因子
        standardized_dir = self.factor_selection_dir / "standardized"
        standardized_dir.mkdir(parents=True, exist_ok=True)

        for fname, fdata in self.standardized_factors.items():
            fdata.to_parquet(standardized_dir / f"{fname}.parquet")

        logger.info(f"✅ 标准化因子已保存: {standardized_dir}")

        # 保存元数据
        metadata = {
            "timestamp": self.timestamp,
            "step": "factor_selection",
            "standardized_factor_count": len(self.standardized_factors),
            "standardized_factor_names": list(self.standardized_factors.keys()),
        }

        with open(
            self.factor_selection_dir / "metadata.json", "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Step 2 完成: {self.factor_selection_dir}")
        logger.info("")

    def _run_wfo(self):
        """WFO验证 - 集成前向回测"""
        logger.info("-" * 80)
        logger.info("Step 3: WFO验证")
        logger.info("-" * 80)

        # 加载OHLCV数据（如果未加载）
        if self.ohlcv_data is None:
            self._load_cross_section_data()

        # 加载标准化因子（如果未加载）
        if self.standardized_factors is None:
            self._load_factor_selection_data()

        # 准备WFO数据
        factor_names = sorted(self.standardized_factors.keys())
        factor_arrays = []

        for factor_name in factor_names:
            factor_df = self.standardized_factors[factor_name]
            factor_arrays.append(factor_df.values)

        # 堆叠: (K, T, N) → (T, N, K)
        import numpy as np

        factors_array = np.stack(factor_arrays, axis=0)
        factors_array = np.transpose(factors_array, (1, 2, 0))

        # 提取收益率
        returns_df = self.ohlcv_data["close"].pct_change(fill_method=None)
        returns_array = returns_df.values

        # 跳过因子预热期（根据实际测试需要371天）
        # 原因：VOL_RATIO_60D需要119天 + IS窗口252天 = 371天
        warmup_offset = 371
        if factors_array.shape[0] > warmup_offset:
            logger.info(f"⚠️  跳过前{warmup_offset}天因子预热期")
            factors_array = factors_array[warmup_offset:]
            returns_array = returns_array[warmup_offset:]

        logger.info(f"WFO数据: {factors_array.shape}")
        logger.info("")

        # 加载约束配置
        constraints_path = Path("configs/FACTOR_SELECTION_CONSTRAINTS.yaml")
        if constraints_path.exists():
            with open(constraints_path, encoding="utf-8") as f:
                constraints_config = yaml.safe_load(f)
        else:
            constraints_config = {}

        # 运行Direct Factor WFO (新版本 - 直接因子级加权)
        wfo_config = self.config.wfo
        from core.direct_factor_wfo_optimizer import DirectFactorWFOOptimizer

        optimizer = DirectFactorWFOOptimizer(
            factor_weighting=wfo_config.get("factor_weighting", "ic_weighted"),
            min_factor_ic=wfo_config.get("min_factor_ic", 0.01),
            ic_floor=wfo_config.get("ic_floor", 0.0),
            verbose=True,
        )

        wfo_results_list, wfo_summary_df = optimizer.run_wfo(
            factors_data=factors_array,
            returns=returns_array,
            factor_names=factor_names,
            is_period=wfo_config.get("is_period", 252),
            oos_period=wfo_config.get("oos_period", 60),
            step_size=wfo_config.get("step_size", 20),
        )

        self.wfo_results = wfo_results_list

        # 保存结果
        wfo_summary_df.to_csv(self.wfo_dir / "wfo_summary.csv", index=False)
        logger.info(f"   - 结果已保存: {self.wfo_dir / 'wfo_summary.csv'}")

        # 🔧 Phase 1: 计算真实收益和KPI（事件驱动 T+1 Top-N）
        logger.info("   - Phase 1: 计算真实收益和KPI...")
        from .wfo_performance_evaluator_basic import WfoPerformanceEvaluator

        backtest_cfg = self.config.backtest or {}
        top_n = int(backtest_cfg.get("top_n", 6))

        # dates 对齐到 warmup 之后
        dates_aligned = returns_df.index[-factors_array.shape[0] :]

        evaluator = WfoPerformanceEvaluator(top_n=top_n)
        evaluator.evaluate_and_save(
            results_list=wfo_results_list,
            factors=factors_array,
            returns=returns_array,
            factor_names=factor_names,
            dates=dates_aligned,
            out_dir=self.wfo_dir,
        )

        # 🔧 Phase 2: 多策略枚举 + Top-5 组合选择（严格T+1真实收益）
        logger.info("\n   - Phase 2: 多策略枚举 + Top-5 组合选择…")
        from .wfo_multi_strategy_selector import WFOMultiStrategySelector

        phase2_cfg = (self.config.wfo or {}).get("phase2", {})
        selector = WFOMultiStrategySelector(
            min_factor_freq=phase2_cfg.get("min_factor_freq", 0.3),
            min_factors=phase2_cfg.get("min_factors", 3),
            max_factors=phase2_cfg.get("max_factors", 5),
            subset_mode=phase2_cfg.get("subset_mode", "enumerate"),
            tau_grid=phase2_cfg.get("tau_grid", [0.7, 1.0, 1.5]),
            topn_grid=phase2_cfg.get("topn_grid", [top_n]),
            signal_z_threshold_grid=phase2_cfg.get("signal_z_threshold_grid", [None]),
            max_strategies=phase2_cfg.get("max_strategies", 200),
            non_overlap_oos=phase2_cfg.get("non_overlap_oos", False),
            turnover_penalty=phase2_cfg.get("turnover_penalty", 0.0),
            coverage_penalty_coef=phase2_cfg.get("coverage_penalty_coef", 1.0),
            coverage_min=phase2_cfg.get("coverage_min", 0.0),
            avg_turnover_max=phase2_cfg.get("avg_turnover_max", None),
            rank_by=phase2_cfg.get("rank_by", "score"),
            stratified_by_k=phase2_cfg.get("stratified_by_k", False),
            k_quota=phase2_cfg.get("k_quota", None),
            subset_shuffle=phase2_cfg.get("subset_shuffle", False),
            random_seed=phase2_cfg.get("random_seed", None),
        )

        top5_df = selector.select_and_save(
            results_list=wfo_results_list,
            factors=factors_array,
            returns=returns_array,
            factor_names=factor_names,
            dates=dates_aligned,
            out_dir=self.wfo_dir,
        )

        # 🔧 写入元数据
        logger.info("\n   - 写入元数据...")
        from .wfo_metadata_writer import WFOMetadataWriter

        WFOMetadataWriter.write_metadata(
            out_dir=self.wfo_dir,
            config_path=Path("configs/default.yaml"),
            wfo_results_count=len(wfo_results_list),
            strategies_count=len(top5_df) if top5_df is not None else 0,
            phase2_params={
                "min_factor_freq": phase2_cfg.get("min_factor_freq", 0.3),
                "min_factors": phase2_cfg.get("min_factors", 3),
                "max_factors": phase2_cfg.get("max_factors", 5),
                "subset_mode": phase2_cfg.get("subset_mode", "enumerate"),
                "tau_grid": phase2_cfg.get("tau_grid", [0.7, 1.0, 1.5]),
                "topn_grid": phase2_cfg.get("topn_grid", [top_n]),
                "signal_z_threshold_grid": phase2_cfg.get(
                    "signal_z_threshold_grid", [None]
                ),
                "max_strategies": phase2_cfg.get("max_strategies", 200),
                "non_overlap_oos": phase2_cfg.get("non_overlap_oos", False),
                "turnover_penalty": phase2_cfg.get("turnover_penalty", 0.0),
                "coverage_penalty_coef": phase2_cfg.get("coverage_penalty_coef", 1.0),
                "coverage_min": phase2_cfg.get("coverage_min", 0.0),
                "avg_turnover_max": phase2_cfg.get("avg_turnover_max", None),
                "rank_by": phase2_cfg.get("rank_by", "score"),
                "stratified_by_k": phase2_cfg.get("stratified_by_k", False),
                "k_quota": phase2_cfg.get("k_quota", None),
                "subset_shuffle": phase2_cfg.get("subset_shuffle", False),
                "random_seed": phase2_cfg.get("random_seed", None),
            },
        )

        logger.info("\n✅ WFO完整流程完成")
        logger.info(f"   - 总窗口数: {len(self.wfo_results)}")
        logger.info(f"   - 平均OOS IC: {wfo_summary_df['oos_ensemble_ic'].mean():.4f}")
        logger.info("\n⚠️  信号延迟说明 (P0-2):")
        logger.info("   - WFO阶段: 无延迟 (纯信号IC验证)")
        logger.info("   - 回测阶段: T+1延迟由VectorBT层或portfolio_constructor应用")
        logger.info("   - 配置位置: configs/default.yaml::backtest.signal_delay_days=1")
        logger.info("   - 生效证据: 回测结果中信号[t]对应持仓[t+1]")
        logger.info(f"✅ Step 3 完成: {self.wfo_dir}")
        logger.info("")

    def _run_backtest(self):
        """VBT回测 - 暴力测试"""
        logger.info("-" * 80)
        logger.info("Step 4: VBT回测")
        logger.info("-" * 80)

        # 回测模块独立运行
        # 使用: python vectorbt_backtest/run_backtest.py --config configs/backtest_config.yaml
        logger.info("✓ 回测模块位于 vectorbt_backtest/")
        logger.info("  运行命令: python vectorbt_backtest/run_backtest.py")
        logger.info("  配置文件: vectorbt_backtest/configs/backtest_config.yaml")
        logger.info("")
        logger.info("⚠️  回测模块独立运行，不集成到Pipeline中")
        logger.info("  原因: 回测需要完整历史数据，与WFO验证逻辑分离")
        logger.info("")

    def _load_cross_section_data(self):
        """加载横截面数据"""
        logger.info("加载横截面数据...")

        # 查找最新的cross_section结果（如果当前目录为空）
        ohlcv_dir = self.cross_section_dir / "ohlcv"
        factors_dir = self.cross_section_dir / "factors"

        if not ohlcv_dir.exists() or not list(factors_dir.glob("*.parquet")):
            results_base = self.output_root / "cross_section"
            if results_base.exists():
                date_dirs = sorted(
                    [d for d in results_base.glob("*") if d.is_dir()], reverse=True
                )
                for date_dir in date_dirs:
                    # 找到第一个有数据的时间戳目录
                    time_dirs = sorted(
                        [d for d in date_dir.glob("*") if d.is_dir()], reverse=True
                    )
                    for time_dir in time_dirs:
                        test_factors_dir = time_dir / "factors"
                        if test_factors_dir.exists() and list(
                            test_factors_dir.glob("*.parquet")
                        ):
                            self.cross_section_dir = time_dir
                            ohlcv_dir = self.cross_section_dir / "ohlcv"
                            factors_dir = self.cross_section_dir / "factors"
                            logger.info(f"使用最新横截面结果: {self.cross_section_dir}")
                            break
                    if (
                        self.cross_section_dir
                        != self.output_root
                        / "cross_section"
                        / self.run_date
                        / self.timestamp
                    ):
                        break

        # 加载OHLCV
        self.ohlcv_data = {}
        for col_name in ["open", "high", "low", "close", "volume"]:
            parquet_path = ohlcv_dir / f"{col_name}.parquet"
            if parquet_path.exists():
                self.ohlcv_data[col_name] = pd.read_parquet(parquet_path)

        # 加载因子
        factors_dir = self.cross_section_dir / "factors"
        self.factors_dict = {}
        for factor_file in factors_dir.glob("*.parquet"):
            factor_name = factor_file.stem
            self.factors_dict[factor_name] = pd.read_parquet(factor_file)

        logger.info(f"✅ 加载完成: {len(self.factors_dict)} 个因子")

    def _load_factor_selection_data(self):
        """加载因子筛选数据"""
        logger.info("加载因子筛选数据...")

        # 查找最新的factor_selection结果（如果当前目录为空）
        standardized_dir = self.factor_selection_dir / "standardized"

        if not standardized_dir.exists() or not list(
            standardized_dir.glob("*.parquet")
        ):
            results_base = self.output_root / "factor_selection"
            if results_base.exists():
                date_dirs = sorted(
                    [d for d in results_base.glob("*") if d.is_dir()], reverse=True
                )
                for date_dir in date_dirs:
                    # 找到第一个有数据的时间戳目录
                    time_dirs = sorted(
                        [d for d in date_dir.glob("*") if d.is_dir()], reverse=True
                    )
                    for time_dir in time_dirs:
                        test_standardized_dir = time_dir / "standardized"
                        if test_standardized_dir.exists() and list(
                            test_standardized_dir.glob("*.parquet")
                        ):
                            self.factor_selection_dir = time_dir
                            standardized_dir = test_standardized_dir
                            logger.info(
                                f"使用最新因子筛选结果: {self.factor_selection_dir}"
                            )
                            break
                    if standardized_dir.exists() and list(
                        standardized_dir.glob("*.parquet")
                    ):
                        break

        # 加载标准化因子
        if not standardized_dir.exists() or not list(
            standardized_dir.glob("*.parquet")
        ):
            raise FileNotFoundError(
                f"未找到标准化因子目录: {standardized_dir}\n"
                "请先运行横截面处理和因子筛选步骤"
            )

        self.standardized_factors = {}
        for factor_file in standardized_dir.glob("*.parquet"):
            factor_name = factor_file.stem
            self.standardized_factors[factor_name] = pd.read_parquet(factor_file)

        logger.info(f"✅ 加载完成: {len(self.standardized_factors)} 个标准化因子")
