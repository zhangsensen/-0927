#!/usr/bin/env python3
"""
因子挖掘全流程 | Factor Mining Pipeline
================================================================================
Layer 6: 完整流程入口。

用法:
  uv run python scripts/run_factor_mining.py                 # 完整流程 (~30min)
  uv run python scripts/run_factor_mining.py --skip-discovery # 仅质检现有因子 (~3min)
  uv run python scripts/run_factor_mining.py --algebraic-only # 仅代数搜索

流程:
  1. Load Data
  2. Compute Factors (PreciseFactorLibrary)
  3. Quality Analysis (10 维度)
  4. Discovery (代数/窗口/变换搜索 + FDR)
  5. Prefilter (可组合性预筛: 6-gate + 正交性)
  6. Selection (聚类去冗余)
  7. Save Results

输出: results/factor_mining_YYYYMMDD_HHMMSS/
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.regime_detector import RegimeDetector
from etf_strategy.core.factor_mining import (
    FactorDiscoveryPipeline,
    FactorPrefilter,
    FactorQualityAnalyzer,
    FactorSelector,
    FactorZoo,
    PrefilterConfig,
)

# ── 配置 ──────────────────────────────────────────────────
SEP = "=" * 80
THIN = "-" * 80
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "combo_wfo_config.yaml"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="因子挖掘全流程")
    parser.add_argument(
        "--skip-discovery", action="store_true", help="跳过发现阶段，仅质检现有因子"
    )
    parser.add_argument("--algebraic-only", action="store_true", help="仅执行代数搜索")
    parser.add_argument(
        "--max-factors", type=int, default=40, help="最终精选因子数量上限 (default: 40)"
    )
    parser.add_argument(
        "--max-correlation", type=float, default=0.7, help="聚类相关阈值 (default: 0.7)"
    )
    parser.add_argument(
        "--fdr-alpha", type=float, default=0.05, help="FDR 校正 alpha (default: 0.05)"
    )
    parser.add_argument(
        "--skip-prefilter",
        action="store_true",
        help="跳过可组合性预筛（默认开启）",
    )
    parser.add_argument(
        "--orth-threshold",
        type=float,
        default=0.6,
        help="正交性阈值: max|rank_corr| with active factors (default: 0.6)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（默认: configs/combo_wfo_config.yaml）",
    )
    return parser.parse_args()


def load_config(config_path: str | None) -> dict:
    """Load YAML config for active factors list."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("factor_mining")

    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "results" / f"factor_mining_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config for active factors
    config = load_config(args.config)
    active_factors = sorted(config.get("active_factors", []))

    # ── Step 1: Load Data ──────────────────────────────────
    print(f"\n{SEP}")
    print("Step 1: 加载数据")
    print(SEP)

    data_dir = str(PROJECT_ROOT / "raw" / "ETF" / "daily")
    loader = DataLoader(data_dir=data_dir)
    ohlcv = loader.load_ohlcv(start_date="2020-01-01", end_date="2025-12-31")
    close = ohlcv["close"]
    volume = ohlcv["volume"]
    print(f"  日期范围: {close.index[0].date()} ~ {close.index[-1].date()}")
    print(f"  ETF 数量: {close.shape[1]}")
    print(f"  交易日数: {len(close)}")

    # ── Step 2: Compute Factors ────────────────────────────
    print(f"\n{SEP}")
    print("Step 2: 计算因子")
    print(SEP)

    lib = PreciseFactorLibrary()
    raw = lib.compute_all_factors(prices=ohlcv)
    factor_names = sorted(lib.list_factors().keys())
    metadata = lib.list_factors()

    # 提取为 dict
    factors_dict = {}
    for name in factor_names:
        factors_dict[name] = raw[name]
    print(f"  原始因子数: {len(factors_dict)}")

    # 标准化
    proc = CrossSectionProcessor(verbose=False)
    std_factors = proc.process_all_factors(factors_dict)
    print(f"  标准化完成: {len(std_factors)} 个因子")

    # ── Step 2b: Regime Detection ──────────────────────────
    print(f"\n{THIN}")
    print("  检测市场环境...")
    detector = RegimeDetector()
    regime_series, _ = detector.detect_regime(ohlcv)
    for regime_val in sorted(regime_series.unique()):
        count = (regime_series == regime_val).sum()
        print(
            f"    {regime_val}: {count} days ({count / len(regime_series) * 100:.1f}%)"
        )

    # ── Step 3: Quality Analysis ───────────────────────────
    print(f"\n{SEP}")
    print("Step 3: 因子质检 (10 维度)")
    print(SEP)

    # Read freq from config (production-aligned)
    freq = config.get("freq", config.get("rebalance", {}).get("freq", 5))

    analyzer = FactorQualityAnalyzer(
        close=close,
        regime_series=regime_series,
        freq=freq,
    )

    # 注册手工因子到 Zoo
    zoo = FactorZoo()
    zoo.seed_from_library(lib)
    print(f"  注册手工因子: {len(zoo)} 个")

    # 质检所有手工因子
    reports = {}
    for name in factor_names:
        meta = metadata[name]
        report = analyzer.analyze(
            factor_name=name,
            factor_df=std_factors[name],
            direction=meta.direction,
            production_ready=meta.production_ready,
        )
        reports[name] = report
        zoo.update_quality(name, report.quality_score, report.passed)

        status = "PASS" if report.passed else "FAIL"
        print(
            f"  [{status}] {name:35s}  score={report.quality_score:+.1f}  "
            f"IC={report.mean_ic:+.4f}  p={report.p_value:.4f}  "
            f"mono={report.monotonicity_score:.2f}"
        )

    passed_count = sum(1 for r in reports.values() if r.passed)
    print(f"\n  质检通过: {passed_count} / {len(reports)}")

    # ── Step 4: Discovery ──────────────────────────────────
    discovery_factors = {}

    if not args.skip_discovery:
        print(f"\n{SEP}")
        print("Step 4: 因子发现")
        print(SEP)

        pipeline = FactorDiscoveryPipeline(
            analyzer=analyzer,
            close=close,
            volume=volume,
            fdr_alpha=args.fdr_alpha,
        )

        enable_algebraic = True
        enable_window = not args.algebraic_only
        enable_transform = not args.algebraic_only

        new_entries, new_factors = pipeline.run(
            factors_dict=std_factors,
            enable_algebraic=enable_algebraic,
            enable_window=enable_window,
            enable_transform=enable_transform,
        )

        print(f"\n  FDR 通过: {len(new_entries)} 个新因子")

        # 质检通过 FDR 的新因子
        for entry in new_entries:
            if entry.name in new_factors:
                direction = "high_is_good"  # 新发现因子默认方向
                report = analyzer.analyze(
                    factor_name=entry.name,
                    factor_df=new_factors[entry.name],
                    direction=direction,
                    production_ready=True,
                )
                reports[entry.name] = report
                entry.quality_score = report.quality_score
                entry.passed = report.passed

        zoo.register_batch(new_entries)
        discovery_factors = new_factors

        # 更新 zoo 质检状态
        for entry in new_entries:
            if entry.name in reports:
                zoo.update_quality(
                    entry.name,
                    reports[entry.name].quality_score,
                    reports[entry.name].passed,
                )

        new_passed = sum(1 for e in new_entries if e.passed)
        print(f"  新因子质检通过: {new_passed} / {len(new_entries)}")
    else:
        print(f"\n{SEP}")
        print("Step 4: 因子发现 (已跳过)")
        print(SEP)

    # ── Step 5: Prefilter (可组合性预筛) ──────────────────
    # 合并所有因子值 (base + algebraic)
    all_factors = {**std_factors, **discovery_factors}

    prefilter_survivors = None  # None = no prefilter, use all

    if not args.skip_prefilter and active_factors:
        print(f"\n{SEP}")
        print("Step 5: 可组合性预筛 (6-gate cascade + orthogonality)")
        print(SEP)

        pf_config = PrefilterConfig(orthogonality_max=args.orth_threshold)
        prefilter = FactorPrefilter(
            active_factors=active_factors,
            config=pf_config,
        )

        # Build quality DataFrame for prefilter
        quality_rows = []
        for name, r in reports.items():
            quality_rows.append(r.to_dict())
        quality_df = pd.DataFrame(quality_rows)

        pf_result = prefilter.run(
            quality_df=quality_df,
            registry={e.name: {"source": e.source,
                               "parent_factors": e.parent_factors,
                               "quality_score": e.quality_score}
                      for e in zoo.list_all()},
            factor_values=all_factors,
        )

        # Print cascade
        for gate, eliminated, remaining in pf_result.gate_cascade:
            print(f"  {gate:20s}: -{eliminated:4d} → {remaining:4d} remain")
        print(f"\n  Survivors: {len(pf_result.survivors)}")

        # Bucket distribution
        print(f"\n  Bucket distribution:")
        for bucket, count in sorted(pf_result.bucket_distribution.items()):
            print(f"    {bucket:30s}: {count}")
        if pf_result.unmapped_survivors:
            print(f"    UNMAPPED: {len(pf_result.unmapped_survivors)}")

        # Warnings
        for w in pf_result.bucket_warnings:
            print(f"  ⚠ {w}")

        # Save prefilter results + survivor matrices
        full_registry = {e.name: {
            "name": e.name, "source": e.source,
            "expression": e.expression,
            "parent_factors": e.parent_factors,
            "metadata": e.metadata,
            "quality_score": e.quality_score, "passed": e.passed,
        } for e in zoo.list_all()}
        prefilter.save(
            output_dir, full_registry, pf_result,
            factor_values=all_factors,
        )
        print(f"\n  Saved: prefilter artifacts + survivor matrices")

        prefilter_survivors = pf_result.survivors
    else:
        print(f"\n{SEP}")
        print("Step 5: 可组合性预筛 (已跳过)")
        print(SEP)

    # ── Step 6: Selection (聚类去冗余) ────────────────────
    print(f"\n{SEP}")
    print("Step 6: 因子筛选 (聚类去冗余)")
    print(SEP)

    # If prefilter ran, only select from survivors
    if prefilter_survivors is not None:
        entries_for_select = [
            e for e in zoo.list_all()
            if e.name in prefilter_survivors
        ]
        factors_for_select = {
            k: v for k, v in all_factors.items()
            if k in prefilter_survivors
        }
        reports_for_select = {
            k: v for k, v in reports.items()
            if k in prefilter_survivors
        }
        print(f"  Input: {len(entries_for_select)} prefiltered survivors")
    else:
        entries_for_select = zoo.list_all()
        factors_for_select = all_factors
        reports_for_select = reports
        print(f"  Input: {len(entries_for_select)} factors (no prefilter)")

    selector = FactorSelector(
        max_correlation=args.max_correlation,
        min_quality_score=2.0,
        max_factors=args.max_factors,
    )

    selected_names, corr_matrix = selector.select(
        entries=entries_for_select,
        reports=reports_for_select,
        factors_dict=factors_for_select,
    )

    print(f"\n  精选因子池 ({len(selected_names)} 个):")
    for name in selected_names:
        r = reports.get(name)
        source = zoo.get(name).source if zoo.get(name) else "?"
        if r:
            print(
                f"    {name:40s}  [{source:14s}]  score={r.quality_score:+.1f}  IC={r.mean_ic:+.4f}"
            )
        else:
            print(f"    {name:40s}  [{source:14s}]")

    # ── Step 7: Save Results ──────────────────────────────
    print(f"\n{SEP}")
    print(f"Step 7: 保存结果 → {output_dir}")
    print(SEP)

    # 7a. Factor registry
    zoo.export_registry(output_dir / "factor_registry.json")
    print(f"  factor_registry.json ({len(zoo)} factors)")

    # 7b. Quality reports
    report_rows = [r.to_dict() for r in reports.values()]
    reports_df = pd.DataFrame(report_rows)
    reports_df.to_parquet(output_dir / "quality_reports.parquet", index=False)
    print(f"  quality_reports.parquet ({len(reports_df)} rows)")

    # 7c. Discovery summary
    if not args.skip_discovery:
        discovery_rows = []
        for entry in zoo.list_all():
            if entry.source != "hand_crafted":
                row = {
                    "name": entry.name,
                    "source": entry.source,
                    "expression": entry.expression,
                    "quality_score": entry.quality_score,
                    "passed": entry.passed,
                }
                discovery_rows.append(row)
        if discovery_rows:
            disc_df = pd.DataFrame(discovery_rows)
            disc_df.to_parquet(output_dir / "discovery_summary.parquet", index=False)
            print(f"  discovery_summary.parquet ({len(disc_df)} rows)")

    # 7d. Selected factors
    selected_data = {
        "timestamp": timestamp,
        "n_selected": len(selected_names),
        "max_correlation": args.max_correlation,
        "max_factors": args.max_factors,
        "prefiltered": prefilter_survivors is not None,
        "factors": selected_names,
        "factor_details": {},
    }
    for name in selected_names:
        r = reports.get(name)
        entry = zoo.get(name)
        if r and entry:
            selected_data["factor_details"][name] = {
                "source": entry.source,
                "quality_score": r.quality_score,
                "mean_ic": r.mean_ic,
                "ic_ir": r.ic_ir,
                "p_value": r.p_value,
                "hit_rate": r.hit_rate,
                "monotonicity": r.monotonicity_score,
            }
    (output_dir / "selected_factors.json").write_text(
        json.dumps(selected_data, indent=2, ensure_ascii=False)
    )
    print(f"  selected_factors.json ({len(selected_names)} factors)")

    # 7e. Correlation matrix
    if not corr_matrix.empty:
        corr_matrix.to_parquet(output_dir / "correlation_matrix.parquet")
        print(
            f"  correlation_matrix.parquet ({corr_matrix.shape[0]}x{corr_matrix.shape[1]})"
        )

    elapsed = time.time() - t0
    print(f"\n{SEP}")
    print(f"完成! 总耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"结果目录: {output_dir}")
    print(SEP)


if __name__ == "__main__":
    main()
