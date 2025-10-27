"""
标准真实数据 WFO 回测流程

遵循 PROJECT_GUIDELINES.md 中的步骤，将横截面建设、因子筛选、WFO 结果按阶段落盘，
便于多次回测对比与复用。
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.constrained_walk_forward_optimizer import ConstrainedWalkForwardOptimizer
from core.cross_section_processor import CrossSectionProcessor
from core.precise_factor_library_v2 import PreciseFactorLibrary
from utils.factor_cache import FactorCache

from scripts.standard_real_data_loader import StandardRealDataLoader


class StandardRealBacktest:
    """标准真实数据回测流程，分阶段落盘输出以便多次 WFO 对比。"""

    def __init__(
        self,
        data_dir: str = None,
        output_root: str | Path = None,
    ):
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "raw" / "ETF" / "daily"
        if output_root is None:
            output_root = Path(__file__).parent.parent / "results"

        self.data_dir = data_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_date = self.timestamp[:8]
        self.output_root = Path(output_root).resolve()

        self.cross_section_dir = (
            self.output_root / "cross_section" / self.run_date / self.timestamp
        )
        self.factor_selection_dir = (
            self.output_root / "factor_selection" / self.run_date / self.timestamp
        )
        self.wfo_dir = self.output_root / "wfo" / self.timestamp
        self.log_dir = self.output_root / "logs"

        for directory in (
            self.cross_section_dir,
            self.factor_selection_dir,
            self.wfo_dir,
            self.log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"standard_real_backtest_{self.timestamp}.log"
        self.logger = logging.getLogger(f"StandardRealBacktest.{self.timestamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.ohlcv_data = None
        self.factor_panel = None
        self.factors_dict: dict[str, pd.DataFrame] = {}
        self.standardized_factors: dict[str, pd.DataFrame] = {}
        self.wfo_results = None
        self.data_summary = None
        self.nan_stats: dict[str, dict[str, int | bool]] = {}
        self.factor_frequency: pd.DataFrame | None = None

        # 🔥 缓存系统
        cache_dir = Path(__file__).parent.parent / "cache" / "factors"
        self.cache = FactorCache(cache_dir=cache_dir, use_timestamp=True)

        self.logger.info("=" * 80)
        self.logger.info("标准真实数据 WFO 回测流程启动")
        self.logger.info("Cross Section 输出目录: %s", self.cross_section_dir)
        self.logger.info("Factor Selection 输出目录: %s", self.factor_selection_dir)
        self.logger.info("WFO 输出目录: %s", self.wfo_dir)
        self.logger.info("日志文件: %s", self.log_file)

    def step1_load_data(
        self,
        etf_codes=None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
    ):
        """Step 1: 加载真实数据（保留 NaN、使用前复权价格）。"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Step 1: 加载真实数据（标准流程）")
        self.logger.info("=" * 80)

        loader = StandardRealDataLoader(data_dir=self.data_dir)
        self.ohlcv_data = loader.load_ohlcv(
            etf_codes=etf_codes,
            start_date=start_date,
            end_date=end_date,
        )

        self.data_summary = loader.get_summary(self.ohlcv_data)
        self.logger.info("✅ 数据加载完成")
        self.logger.info("   日期: %s 天", self.data_summary["total_dates"])
        self.logger.info("   标的: %s 只", self.data_summary["total_symbols"])
        self.logger.info("   日期范围: %s -> %s", *self.data_summary["date_range"])

        low_coverage = {
            code: ratio
            for code, ratio in self.data_summary["coverage_ratio"].items()
            if ratio < 0.97
        }
        if low_coverage:
            self.logger.warning("⚠️  %s 只 ETF 覆盖率 < 97%%", len(low_coverage))
            for code, ratio in list(low_coverage.items())[:5]:
                self.logger.warning("     %s: %.2f%%", code, ratio * 100)

        return self

    def step2_compute_factors(self):
        """Step 2: 计算精确定义因子（不做标准化/截断）。"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Step 2: 计算因子（PreciseFactorLibrary v2）")
        self.logger.info("=" * 80)

        if self.ohlcv_data is None:
            raise RuntimeError("请先执行 step1_load_data()")

        lib = PreciseFactorLibrary()

        # 🔥 尝试加载缓存
        cached_factors = self.cache.load_factors(
            ohlcv=self.ohlcv_data, lib_class=lib.__class__, stage="raw"
        )

        if cached_factors is not None:
            self.logger.info("✅ 使用因子缓存（跳过计算）")
            self.factors_dict = cached_factors
            # 重建 factor_panel
            self.factor_panel = pd.concat(
                [
                    df.to_frame(name=fname) if isinstance(df, pd.Series) else df
                    for fname, df in self.factors_dict.items()
                ],
                axis=1,
            )
        else:
            self.logger.info("🔄 缓存未命中，开始计算因子...")
            factors_df = lib.compute_all_factors(prices=self.ohlcv_data)
            self.factor_panel = factors_df
            self.factors_dict = {}

            for factor_name in lib.list_factors():
                self.factors_dict[factor_name] = factors_df[factor_name]

            # 🔥 保存缓存
            self.cache.save_factors(
                factors=self.factors_dict,
                ohlcv=self.ohlcv_data,
                lib_class=lib.__class__,
                stage="raw",
            )

        self.logger.info("✅ 因子计算完成")
        self.logger.info("   因子数: %s", len(self.factors_dict))
        for idx, (fname, fdata) in enumerate(self.factors_dict.items(), start=1):
            nan_ratio = fdata.isna().sum().sum() / fdata.size
            self.logger.info(
                "     %02d. %-25s NaN率: %.2f%%", idx, fname, nan_ratio * 100
            )

        self._persist_cross_section_outputs()
        return self

    def step3_standardize(self):
        """Step 3: 横截面标准化 + Winsorize（不破坏 NaN）。"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Step 3: 横截面标准化（CrossSectionProcessor）")
        self.logger.info("=" * 80)

        if not self.factors_dict:
            raise RuntimeError("请先执行 step2_compute_factors()")

        processor = CrossSectionProcessor()

        # 🔥 尝试加载标准化因子缓存
        cached_std_factors = self.cache.load_factors(
            ohlcv=self.ohlcv_data, lib_class=processor.__class__, stage="standardized"
        )

        if cached_std_factors is not None:
            self.logger.info("✅ 使用标准化因子缓存（跳过处理）")
            processed_factors = cached_std_factors
        else:
            self.logger.info("🔄 缓存未命中，开始标准化...")
            processed_factors = processor.process_all_factors(self.factors_dict)

            # 🔥 保存标准化因子缓存
            self.cache.save_factors(
                factors=processed_factors,
                ohlcv=self.ohlcv_data,
                lib_class=processor.__class__,
                stage="standardized",
            )

        self.standardized_factors = processed_factors
        self.nan_stats = {}

        issues = []
        for fname, original_df in self.factors_dict.items():
            standardized_df = processed_factors[fname]
            orig_nan = int(original_df.isna().sum().sum())
            std_nan = int(standardized_df.isna().sum().sum())
            self.nan_stats[fname] = {
                "original_nan_count": orig_nan,
                "standardized_nan_count": std_nan,
                "nan_preserved": orig_nan == std_nan,
            }
            if orig_nan != std_nan:
                issues.append((fname, orig_nan, std_nan))

        if issues:
            for fname, orig_nan, std_nan in issues:
                self.logger.error(
                    "❌ %s NaN 数量不一致: %s -> %s", fname, orig_nan, std_nan
                )
            raise RuntimeError("标准化阶段检测到 NaN 不一致，请检查日志。")

        self.logger.info("✅ 标准化完成，NaN 保留检查通过")
        self._persist_factor_selection_outputs()
        return self

    def step4_wfo_with_constraints(
        self,
        in_sample_days: int = 252,
        out_of_sample_days: int = 60,
        step_days: int = 20,
        target_factor_count: int = 5,
        ic_threshold: float | None = None,
    ):
        """Step 4+5: 约束 WFO 因子筛选。"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Step 4: 约束 WFO 因子筛选")
        self.logger.info("=" * 80)

        if not self.standardized_factors:
            raise RuntimeError("请先执行 step3_standardize()")

        factor_names = list(self.standardized_factors.keys())
        close_df = self.ohlcv_data["close"]
        returns_df = close_df.pct_change()

        n_dates = len(close_df)
        n_symbols = len(close_df.columns)
        n_factors = len(factor_names)

        factors_3d = np.full((n_dates, n_symbols, n_factors), np.nan)
        for idx, fname in enumerate(factor_names):
            factors_3d[:, :, idx] = self.standardized_factors[fname].values

        self.logger.info("   数据形状: T=%s, N=%s, F=%s", n_dates, n_symbols, n_factors)
        self.logger.info(
            "   WFO参数: IS=%s, OOS=%s, step=%s, target_factor_count=%s",
            in_sample_days,
            out_of_sample_days,
            step_days,
            target_factor_count,
        )

        optimizer = ConstrainedWalkForwardOptimizer()
        wfo_df, constraint_reports = optimizer.run_constrained_wfo(
            factors_data=factors_3d,
            returns=returns_df.values,
            factor_names=factor_names,
            is_period=in_sample_days,
            oos_period=out_of_sample_days,
            step_size=step_days,
            target_factor_count=target_factor_count,
        )

        self.wfo_results = {
            "results_df": wfo_df,
            "constraint_reports": constraint_reports,
            "windows": len(wfo_df),
            "config": {
                "in_sample_days": in_sample_days,
                "out_of_sample_days": out_of_sample_days,
                "step_days": step_days,
                "target_factor_count": target_factor_count,
                "ic_threshold": ic_threshold,
            },
        }

        self.logger.info("✅ WFO 回测完成，窗口数: %s", len(wfo_df))
        if len(wfo_df) > 0:
            self.logger.info("   平均 OOS IC: %.4f", wfo_df["oos_ic_mean"].mean())
            self.logger.info("   平均 IC 衰减: %.4f", wfo_df["ic_drop"].mean())

            selected_lists = [
                [factor.strip() for factor in factors_str.split(",") if factor.strip()]
                for factors_str in wfo_df["selected_factors"]
            ]
            selected_flat = [factor for factors in selected_lists for factor in factors]
            if selected_flat:
                factor_counts = (
                    pd.Series(selected_flat).value_counts().sort_values(ascending=False)
                )
                self.factor_frequency = factor_counts.to_frame(name="count")
                self.factor_frequency["selection_rate"] = self.factor_frequency[
                    "count"
                ] / len(wfo_df)

                self.logger.info("")
                self.logger.info("   TOP 5 因子选中频率:")
                for factor, row in self.factor_frequency.head(5).iterrows():
                    self.logger.info(
                        "     %-25s: %.2f%% (%s/%s)",
                        factor,
                        row["selection_rate"] * 100,
                        int(row["count"]),
                        len(wfo_df),
                    )
            else:
                self.factor_frequency = None
        else:
            self.factor_frequency = None

        return self

    def save_results(self):
        """按阶段落盘所有结果（cross_section / factor_selection / wfo）。"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("保存结果")
        self.logger.info("=" * 80)

        if not self.wfo_results or "results_df" not in self.wfo_results:
            self.logger.warning("暂无 WFO 结果可保存，跳过写盘。")
            return self

        wfo_df = self.wfo_results["results_df"]
        wfo_file = self.wfo_dir / "wfo_results.csv"
        wfo_df.to_csv(wfo_file, index=False, encoding="utf-8")
        self.logger.info("✅ WFO 结果: %s", wfo_file)

        if self.wfo_results.get("constraint_reports") is not None:
            constraint_file = self.wfo_dir / "constraint_reports.json"
            # 将 dataclass 对象转换为字典以便 JSON 序列化
            from dataclasses import asdict

            serializable_reports = [
                asdict(report) for report in self.wfo_results["constraint_reports"]
            ]
            with open(constraint_file, "w", encoding="utf-8") as f:
                json.dump(
                    serializable_reports,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            self.logger.info("✅ 约束报告: %s", constraint_file)

        freq_file, factors_file, freq_summary = self._persist_factor_selection_summary(
            wfo_df
        )

        metadata = {
            "timestamp": self.timestamp,
            "paths": {
                "cross_section": str(
                    self.cross_section_dir.relative_to(self.output_root)
                ),
                "factor_selection": str(
                    self.factor_selection_dir.relative_to(self.output_root)
                ),
                "wfo": str(self.wfo_dir.relative_to(self.output_root)),
                "log_file": str(self.log_file.relative_to(self.output_root)),
            },
            "wfo": {
                "windows": int(len(wfo_df)),
                "oos_ic_mean": (
                    float(wfo_df["oos_ic_mean"].mean()) if len(wfo_df) else None
                ),
                "ic_drop_mean": (
                    float(wfo_df["ic_drop"].mean()) if len(wfo_df) else None
                ),
                "config": self.wfo_results.get("config", {}),
            },
            "factor_selection_summary": {
                "frequency_file": (
                    str(freq_file.relative_to(self.output_root)) if freq_file else None
                ),
                "selected_factors_file": (
                    str(factors_file.relative_to(self.output_root))
                    if factors_file
                    else None
                ),
                "summary_file": (
                    str(freq_summary.relative_to(self.output_root))
                    if freq_summary
                    else None
                ),
            },
        }

        metadata_path = self.wfo_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.logger.info("✅ WFO 元数据: %s", metadata_path)

        self._write_summary_file(wfo_df, metadata)
        self.logger.info("📦 完整输出位于: %s", self.wfo_dir)
        return self

    def run_full_pipeline(
        self,
        etf_codes=None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        **wfo_kwargs,
    ):
        """执行完整流程。"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("开始标准真实数据 WFO 回测")
        self.logger.info("=" * 80)

        try:
            self.step1_load_data(etf_codes, start_date, end_date)
            self.step2_compute_factors()
            self.step3_standardize()
            self.step4_wfo_with_constraints(**wfo_kwargs)
            self.save_results()

            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("✅ 完整流程执行成功")
            self.logger.info("=" * 80)
        except Exception:
            self.logger.exception("❌ 流程执行失败")
            raise

        return self

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------
    def _persist_cross_section_outputs(self):
        if (
            self.ohlcv_data is None
            or self.factor_panel is None
            or self.data_summary is None
        ):
            return

        ohlcv_dir = self.cross_section_dir / "ohlcv"
        ohlcv_dir.mkdir(parents=True, exist_ok=True)

        for field, df in self.ohlcv_data.items():
            file_path = ohlcv_dir / f"{field}.parquet"
            df.to_parquet(file_path)

        factor_panel_path = self.cross_section_dir / "factor_panel.parquet"
        self.factor_panel.to_parquet(factor_panel_path)

        metadata = {
            "timestamp": self.timestamp,
            "total_dates": int(self.data_summary["total_dates"]),
            "total_symbols": int(self.data_summary["total_symbols"]),
            "date_range": list(self.data_summary["date_range"]),
            "factor_count": len(self.factors_dict),
            "factor_names": list(self.factors_dict.keys()),
            "coverage_ratio": {
                symbol: float(ratio)
                for symbol, ratio in self.data_summary["coverage_ratio"].items()
            },
            "missing_ratio": {
                symbol: float(ratio)
                for symbol, ratio in self.data_summary["missing_ratio"].items()
            },
        }

        metadata_path = self.cross_section_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info("✅ 横截面数据写入: %s", self.cross_section_dir)

    def _persist_factor_selection_outputs(self):
        if not self.standardized_factors:
            return

        std_dir = self.factor_selection_dir / "standardized_factors"
        std_dir.mkdir(parents=True, exist_ok=True)

        for fname, fdata in self.standardized_factors.items():
            fdata.to_parquet(std_dir / f"{fname}.parquet")

        metadata = {
            "timestamp": self.timestamp,
            "factor_names": list(self.standardized_factors.keys()),
            "nan_stats": self.nan_stats,
        }

        metadata_path = self.factor_selection_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info("✅ 因子筛选阶段数据写入: %s", self.factor_selection_dir)

    def _persist_factor_selection_summary(self, wfo_df):
        if self.factor_frequency is None or self.factor_frequency.empty:
            return None, None, None

        freq_file = (
            self.factor_selection_dir
            / f"selected_factor_frequency_{self.timestamp}.csv"
        )
        self.factor_frequency.to_csv(freq_file, encoding="utf-8")

        factors_file = (
            self.factor_selection_dir / f"selected_factors_{self.timestamp}.txt"
        )
        with open(factors_file, "w", encoding="utf-8") as f:
            for factor in self.factor_frequency.index.tolist():
                f.write(f"{factor}\n")

        summary_payload = {
            "timestamp": self.timestamp,
            "windows": int(len(wfo_df)),
            "top_factors": [
                {
                    "factor": factor,
                    "count": int(row["count"]),
                    "selection_rate": float(row["selection_rate"]),
                }
                for factor, row in self.factor_frequency.head(10).iterrows()
            ],
        }
        summary_file = (
            self.factor_selection_dir / f"selection_summary_{self.timestamp}.json"
        )
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)

        self.logger.info("✅ 因子选中频率写入: %s", freq_file)
        return freq_file, factors_file, summary_file

    def _write_summary_file(self, wfo_df, metadata):
        summary_file = self.wfo_dir / "SUMMARY.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("ETF轮动回测 - 执行摘要\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"执行时间: {self.timestamp}\n")
            if self.data_summary:
                start, end = self.data_summary["date_range"]
                f.write(f"数据范围: {start} ~ {end}\n")
                f.write(f"标的数量: {self.data_summary['total_symbols']}\n")
                f.write(f"交易日数: {self.data_summary['total_dates']}\n")
            if self.factors_dict:
                f.write(f"因子候选: {len(self.factors_dict)}\n")
            f.write(f"WFO窗口: {metadata['wfo']['windows']}\n")

            if len(wfo_df) > 0:
                f.write("\nWFO性能:\n")
                f.write(f"  平均OOS IC: {wfo_df['oos_ic_mean'].mean():.4f}\n")
                f.write(f"  平均IC衰减: {wfo_df['ic_drop'].mean():.4f}\n")
                if (
                    self.factor_frequency is not None
                    and not self.factor_frequency.empty
                ):
                    f.write("\n  因子选中 TOP5:\n")
                    for factor, row in self.factor_frequency.head(5).iterrows():
                        f.write(
                            f"    {factor:25s}: {row['selection_rate']*100:5.2f}% "
                            f"({int(row['count'])}/{len(wfo_df)})\n"
                        )

            f.write("\n输出目录:\n")
            f.write(f"  Cross Section: {metadata['paths']['cross_section']}\n")
            f.write(f"  Factor Selection: {metadata['paths']['factor_selection']}\n")
            f.write(f"  WFO: {metadata['paths']['wfo']}\n")
            f.write(f"  日志: {metadata['paths']['log_file']}\n")

        self.logger.info("✅ WFO 摘要: %s", summary_file)


def main():
    """命令行入口。"""
    output_root = Path(__file__).parent.parent / "results"
    backtest = StandardRealBacktest(output_root=output_root)

    # 使用数据文件中实际存在的43个ETF代码（2024-10更新）
    etf_codes = [
        # 深圳ETF (19只)
        "159801",
        "159819",
        "159859",
        "159883",
        "159915",
        "159920",
        "159928",
        "159949",
        "159992",
        "159995",
        "159998",
        # 上海ETF - 51X系列 (7只)
        "510050",
        "510300",
        "510500",
        "511010",
        "511260",
        "511380",
        # 上海ETF - 512系列 (10只)
        "512010",
        "512100",
        "512400",
        "512480",
        "512660",
        "512690",
        "512720",
        "512800",
        "512880",
        "512980",
        # 上海ETF - 513系列 (4只)
        "513050",
        "513100",
        "513130",
        "513500",
        # 上海ETF - 515系列 (5只)
        "515030",
        "515180",
        "515210",
        "515650",
        "515790",
        # 上海ETF - 516/518系列 (3只)
        "516090",
        "516160",
        "516520",
        # 上海ETF - 518/588系列 (4只)
        "518850",
        "518880",
        "588000",
        "588200",
    ]  # 共43只ETF，全部有数据文件支持

    backtest.run_full_pipeline(
        etf_codes=etf_codes,
        start_date="2020-01-01",
        end_date="2025-10-14",  # 🔥 修复：使用原始数据的真实结束日期
        in_sample_days=252,
        out_of_sample_days=60,
        step_days=20,
        target_factor_count=5,
        ic_threshold=0.05,
    )


if __name__ == "__main__":
    main()
