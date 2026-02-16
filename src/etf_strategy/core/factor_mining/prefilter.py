"""
可组合性预筛器 | Composability Pre-filter
==========================================
在 discovery → registry 之后运行，筛选出与现有 active 因子正交的高质量候选。

设计原则:
  - 因子层只做"排除明确垃圾"，不预测组合价值
  - WFO 的 Rolling OOS 才是金标准验证
  - 正交性是核心门槛（Phase-0 诊断确认：压倒性筛选器，淘汰 73%）

Pipeline 位置:
  Discovery → Quality → **Prefilter** → Selection → Save

输入: 全量 quality reports + factor values + active 因子列表
输出: prefiltered survivors + 诊断报告 + 桶分布
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────

@dataclass
class PrefilterConfig:
    """预筛器阈值配置 (v1, 基于 Phase-0 诊断)."""
    # Stage 1: Quality hard gates
    nan_rate_max: float = 0.20
    ic_p_value_max: float = 0.05
    monotonicity_min: float = 0.8
    rank_autocorr_min: float = 0.7
    rolling_ic_pos_min: float = 0.55
    # Stage 2: Composability gate
    orthogonality_max: float = 0.6


# ── Result ───────────────────────────────────────────────────

@dataclass
class PrefilterResult:
    """预筛结果."""
    survivors: List[str]
    total_input: int
    gate_cascade: List[Tuple[str, int, int]]  # (gate, eliminated, remaining)
    bucket_distribution: Dict[str, int]
    unmapped_survivors: List[str]
    bucket_warnings: List[str]
    factor_bucket_map: Dict[str, str] = field(default_factory=dict)  # factor → bucket
    orthogonality_series: Optional[pd.Series] = field(default=None, repr=False)


# ── Core ─────────────────────────────────────────────────────

class FactorPrefilter:
    """可组合性预筛器.

    用法:
        pf = FactorPrefilter(active_factors=ACTIVE_17)
        result = pf.run(quality_df, registry, factor_values)
        pf.save(output_dir, registry, result)
    """

    def __init__(
        self,
        active_factors: List[str],
        config: Optional[PrefilterConfig] = None,
    ):
        self.active = sorted(active_factors)
        self.config = config or PrefilterConfig()

    def run(
        self,
        quality_df: pd.DataFrame,
        registry: dict,
        factor_values: Dict[str, pd.DataFrame],
    ) -> PrefilterResult:
        """运行 6-gate 预筛.

        Args:
            quality_df: 质量报告 (须含 factor_name 列)
            registry: 全量因子注册表 dict
            factor_values: {factor_name: DataFrame(dates × symbols)}

        Returns:
            PrefilterResult
        """
        cfg = self.config

        # Index by factor_name
        if "factor_name" in quality_df.columns:
            qr = quality_df.set_index("factor_name")
        else:
            qr = quality_df

        # Exclude active factors (baseline, not candidates)
        candidate_names = sorted(n for n in qr.index if n not in self.active)
        candidates = qr.loc[candidate_names]
        logger.info(f"Prefilter input: {len(candidate_names)} candidates "
                     f"(excl. {len(self.active)} active)")

        # ── Stage 1: Quality hard gates ──
        remaining = pd.Series(True, index=candidates.index)
        cascade: List[Tuple[str, int, int]] = []

        gates = [
            ("nan_ok",        candidates["nan_rate"] <= cfg.nan_rate_max),
            ("ic_sig",        candidates["p_value"] < cfg.ic_p_value_max),
            ("monotonic",     candidates["monotonicity_score"] >= cfg.monotonicity_min),
            ("rank_autocorr", candidates["rank_autocorrelation"] >= cfg.rank_autocorr_min),
            ("ic_stable",     candidates["rolling_ic_positive_rate"] >= cfg.rolling_ic_pos_min),
        ]

        for gate_name, mask in gates:
            before = int(remaining.sum())
            remaining = remaining & mask
            after = int(remaining.sum())
            cascade.append((gate_name, before - after, after))
            logger.info(f"  Gate {gate_name}: {before} → {after} (-{before - after})")

        # ── Stage 2: Orthogonality gate ──
        orth_series = self._compute_orthogonality(factor_values, candidates.index)

        before = int(remaining.sum())
        remaining = remaining & (orth_series.reindex(candidates.index) <= cfg.orthogonality_max)
        after = int(remaining.sum())
        cascade.append(("orthogonal", before - after, after))
        logger.info(f"  Gate orthogonal: {before} → {after} (-{before - after})")

        survivors = sorted(remaining[remaining].index.tolist())

        # ── Bucket analysis ──
        bucket_dist, unmapped, warnings, factor_map = self._analyze_buckets(
            survivors, registry)

        return PrefilterResult(
            survivors=survivors,
            total_input=len(candidate_names),
            gate_cascade=cascade,
            bucket_distribution=bucket_dist,
            unmapped_survivors=unmapped,
            bucket_warnings=warnings,
            factor_bucket_map=factor_map,
            orthogonality_series=orth_series,
        )

    # ── Orthogonality computation ────────────────────────────

    def _compute_orthogonality(
        self,
        factor_values: Dict[str, pd.DataFrame],
        candidate_names: pd.Index,
    ) -> pd.Series:
        """Compute max|rank_corr| with active factors for each factor.

        Vectorized: ranks along symbol axis, then matrix-multiply per date.
        """
        # Active factors present in values
        active_in = [n for n in self.active if n in factor_values]
        if not active_in:
            logger.warning("No active factors found in factor_values!")
            return pd.Series(dtype=float)

        # Build name list: candidates ∪ active (for correlation matrix)
        all_names = sorted(set(list(candidate_names) + active_in))
        all_names = [n for n in all_names if n in factor_values]
        N = len(all_names)
        K = len(active_in)

        # Common dates/symbols
        ref = factor_values[active_in[0]]
        common_dates = ref.index
        common_symbols = ref.columns
        for n in active_in[1:]:
            df = factor_values[n]
            common_dates = common_dates.intersection(df.index)
            common_symbols = common_symbols.intersection(df.columns)

        T = len(common_dates)
        S = len(common_symbols)
        mid_rank = (S + 1) / 2.0
        logger.info(f"  Orthogonality: {N} factors × {T} dates × {S} symbols")

        # Build ranked 3D array
        all_ranks = np.full((T, S, N), mid_rank, dtype=np.float32)
        for i, name in enumerate(all_names):
            df = factor_values[name]
            aligned = df.reindex(index=common_dates, columns=common_symbols)
            ranked = aligned.rank(axis=1, method="average", na_option="keep")
            all_ranks[:, :, i] = ranked.fillna(mid_rank).values

        active_idx = [all_names.index(n) for n in active_in]

        # Vectorized cross-sectional correlation
        corr_sum = np.zeros((N, K), dtype=np.float64)
        for t in range(T):
            r_all = all_ranks[t]              # (S, N)
            r_act = r_all[:, active_idx]      # (S, K)

            # Center
            r_all_c = r_all - r_all.mean(axis=0, keepdims=True)
            r_act_c = r_act - r_act.mean(axis=0, keepdims=True)

            # Normalize
            n_all = np.sqrt(np.sum(r_all_c ** 2, axis=0, keepdims=True))
            n_act = np.sqrt(np.sum(r_act_c ** 2, axis=0, keepdims=True))

            r_all_c = np.where(n_all > 1e-10, r_all_c / n_all, 0)
            r_act_c = np.where(n_act > 1e-10, r_act_c / n_act, 0)

            corr_sum += r_all_c.T @ r_act_c

        mean_corr = corr_sum / T
        max_abs = np.max(np.abs(mean_corr), axis=1)

        # For active factors: exclude self-correlation
        for j, act_name in enumerate(active_in):
            if act_name in all_names:
                i = all_names.index(act_name)
                row = np.abs(mean_corr[i, :]).copy()
                row[j] = 0
                max_abs[i] = row.max()

        return pd.Series(max_abs, index=all_names)

    # ── Bucket analysis ──────────────────────────────────────

    @staticmethod
    def _analyze_buckets(
        survivors: List[str],
        registry: dict,
    ) -> Tuple[Dict[str, int], List[str], List[str], Dict[str, str]]:
        """Analyze bucket distribution of survivors.

        Returns:
            (bucket_dist, unmapped, warnings, factor_bucket_map)
        """
        from etf_strategy.core.factor_buckets import FACTOR_BUCKETS, FACTOR_TO_BUCKET

        bucket_dist: Dict[str, int] = {}
        unmapped: List[str] = []
        factor_map: Dict[str, str] = {}  # per-factor → bucket

        for name in survivors:
            if name in FACTOR_TO_BUCKET:
                b = FACTOR_TO_BUCKET[name]
                bucket_dist[b] = bucket_dist.get(b, 0) + 1
                factor_map[name] = b
            elif registry.get(name, {}).get("source") == "algebraic":
                parents = registry[name].get("parent_factors", [])
                parent_buckets = {
                    FACTOR_TO_BUCKET[p]
                    for p in parents
                    if p in FACTOR_TO_BUCKET
                }
                if parent_buckets:
                    # Assign to least-populated parent bucket
                    primary = min(
                        parent_buckets,
                        key=lambda b: bucket_dist.get(b, 0),
                    )
                    bucket_dist[primary] = bucket_dist.get(primary, 0) + 1
                    factor_map[name] = primary
                else:
                    unmapped.append(name)
            else:
                unmapped.append(name)

        # Warnings
        warnings: List[str] = []
        all_buckets = set(FACTOR_BUCKETS.keys())
        covered = set(bucket_dist.keys())
        missing = all_buckets - covered
        if missing:
            warnings.append(f"MISSING buckets (0 survivors): {sorted(missing)}")
        for b in sorted(bucket_dist):
            if bucket_dist[b] > 15:
                warnings.append(f"OVER-REPRESENTED: {b} = {bucket_dist[b]} factors")

        return bucket_dist, unmapped, warnings, factor_map

    # ── Save results ─────────────────────────────────────────

    def save(
        self,
        output_dir: Path,
        registry: dict,
        result: PrefilterResult,
        factor_values: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        """Save prefiltered results to output_dir.

        Args:
            output_dir: output directory
            registry: full factor registry
            result: PrefilterResult from run()
            factor_values: if provided, export survivor matrices
        """
        output_dir = Path(output_dir)

        # 1. Prefiltered registry (only survivors)
        prefiltered_reg = {
            k: v for k, v in registry.items()
            if k in result.survivors
        }
        with open(output_dir / "factor_registry_prefiltered.json", "w") as f:
            json.dump(prefiltered_reg, f, indent=2, ensure_ascii=False)

        # 2. Flat factor list
        with open(output_dir / "prefiltered_factor_list.txt", "w") as f:
            for name in sorted(result.survivors):
                f.write(f"{name}\n")

        # 3. Diagnostic report (includes factor_bucket_map)
        report = {
            "total_input": result.total_input,
            "n_survivors": len(result.survivors),
            "gate_cascade": [
                {"gate": g, "eliminated": e, "remaining": r}
                for g, e, r in result.gate_cascade
            ],
            "bucket_distribution": result.bucket_distribution,
            "unmapped_survivors": result.unmapped_survivors,
            "bucket_warnings": result.bucket_warnings,
            "factor_bucket_map": result.factor_bucket_map,
            "config": asdict(self.config),
        }
        with open(output_dir / "prefilter_report.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 4. UNMAPPED survivors (for manual bucket assignment)
        if result.unmapped_survivors:
            rows = []
            for name in sorted(result.unmapped_survivors):
                info = registry.get(name, {})
                rows.append({
                    "factor_name": name,
                    "source": info.get("source", ""),
                    "expression": info.get("expression", ""),
                    "quality_score": info.get("quality_score", 0),
                })
            pd.DataFrame(rows).to_csv(
                output_dir / "unmapped_survivors.csv", index=False)

        # 5. Survivor factor matrices (for WFO consumption)
        if factor_values and result.survivors:
            self._export_survivor_matrices(
                output_dir, result.survivors, factor_values,
                result.factor_bucket_map,
            )

        logger.info(
            f"Prefilter saved: {len(result.survivors)} survivors "
            f"→ {output_dir}"
        )

    def _export_survivor_matrices(
        self,
        output_dir: Path,
        survivors: List[str],
        factor_values: Dict[str, pd.DataFrame],
        factor_bucket_map: Dict[str, str],
    ) -> None:
        """Export survivor factor matrices as parquet (audit) + npz (WFO).

        Files produced:
          survivors_factors.parquet  — long table (date, symbol, factor_name, value)
          survivors_3d.npz           — {data(T,N,F), dates, symbols, factor_names}
          survivors_meta.json        — factor_names, bucket_map, shapes
        """
        available = sorted(n for n in survivors if n in factor_values)
        if not available:
            logger.warning("No survivor factor values available for export")
            return

        # Determine common dates/symbols from first factor
        ref = factor_values[available[0]]
        dates = ref.index
        symbols = sorted(ref.columns)

        # ── Parquet long table ──
        long_parts = []
        for name in available:
            df = factor_values[name].reindex(index=dates, columns=symbols)
            melted = df.stack(dropna=False).reset_index()
            melted.columns = ["date", "symbol", "value"]
            melted["factor_name"] = name
            long_parts.append(melted)
        long_df = pd.concat(long_parts, ignore_index=True)
        long_df.to_parquet(output_dir / "survivors_factors.parquet", index=False)

        # ── NPZ 3D tensor ──
        T, N, F = len(dates), len(symbols), len(available)
        tensor = np.full((T, N, F), np.nan, dtype=np.float32)
        for i, name in enumerate(available):
            df = factor_values[name].reindex(index=dates, columns=symbols)
            tensor[:, :, i] = df.values

        np.savez_compressed(
            output_dir / "survivors_3d.npz",
            data=tensor,
            dates=np.array([str(d.date()) for d in dates]),
            symbols=np.array(symbols),
            factor_names=np.array(available),
        )

        # ── Metadata JSON ──
        meta = {
            "factor_names": available,
            "n_factors": len(available),
            "n_dates": T,
            "n_symbols": N,
            "date_range": [str(dates[0].date()), str(dates[-1].date())],
            "symbols": symbols,
            "factor_bucket_map": {
                n: factor_bucket_map.get(n, "UNMAPPED")
                for n in available
            },
        }
        with open(output_dir / "survivors_meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(
            f"  Exported survivor matrices: {F} factors × {T} dates × {N} symbols"
        )
