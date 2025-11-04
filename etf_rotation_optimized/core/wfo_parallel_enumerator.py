"""
WFO并行枚举器

职责：
- 并行枚举策略
- 增量计算（支持中断恢复）
- Parquet高效存储
- 进度显示
"""

from __future__ import annotations

import json
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .wfo_multi_strategy_selector import StrategySpec
from .wfo_strategy_evaluator import WFOStrategyEvaluator

logger = logging.getLogger(__name__)


class WFOParallelEnumerator:
    """并行枚举器：支持并行、增量、Parquet"""

    def __init__(
        self,
        n_workers: Optional[int] = None,
        chunk_size: int = 50,
        use_parquet: bool = True,
        enable_incremental: bool = True,
    ):
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.chunk_size = chunk_size
        self.use_parquet = use_parquet
        self.enable_incremental = enable_incremental

    def enumerate_strategies(
        self,
        specs: List[StrategySpec],
        results_list: List,
        factors: np.ndarray,
        returns: np.ndarray,
        factor_names: List[str],
        out_dir: Path,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """并行枚举策略"""

        output_file = out_dir / (
            "strategies_ranked.parquet" if self.use_parquet else "strategies_ranked.csv"
        )

        # 1. 增量计算：检查已存在结果
        existing_keys = set()
        if self.enable_incremental and output_file.exists():
            logger.info("检测到已存在结果，启用增量计算...")
            if self.use_parquet:
                existing_df = pd.read_parquet(output_file)
            else:
                existing_df = pd.read_csv(output_file)
            existing_keys = set(existing_df["_key"].tolist())
            logger.info(f"已计算{len(existing_keys)}个策略，跳过")

        # 过滤已计算的策略
        specs_to_compute = [s for s in specs if s.key() not in existing_keys]
        if not specs_to_compute:
            logger.info("所有策略已计算完成，直接读取结果")
            if self.use_parquet:
                return pd.read_parquet(output_file)
            else:
                return pd.read_csv(output_file)

        logger.info(f"待计算策略: {len(specs_to_compute)}/{len(specs)}")

        # 2. 分片
        chunks = [
            specs_to_compute[i : i + self.chunk_size]
            for i in range(0, len(specs_to_compute), self.chunk_size)
        ]

        # 3. 并行计算
        logger.info(f"启动{self.n_workers}个进程并行计算...")
        all_recs = []
        per_strategy_returns = {}

        if self.n_workers > 1:
            # 多进程并行 - 使用imap_unordered显示进度
            from functools import partial

            with Pool(processes=self.n_workers) as pool:
                evaluate_fn = partial(
                    WFOStrategyEvaluator.evaluate_chunk,
                    results_list=results_list,
                    factors=factors,
                    returns=returns,
                    factor_names=factor_names,
                    dates=dates,
                )

                # 使用imap_unordered来实时获取结果
                total_chunks = len(chunks)
                completed = 0
                chunk_results = []  # 先收集所有chunk结果

                for chunk_res in pool.imap_unordered(evaluate_fn, chunks, chunksize=1):
                    completed += 1
                    chunk_results.append(chunk_res)

                    # 每10个chunk输出一次进度
                    if completed % 10 == 0 or completed == total_chunks:
                        progress_pct = 100 * completed / total_chunks
                        logger.info(
                            f"进度: {completed}/{total_chunks} chunks ({progress_pct:.1f}%)"
                        )

                # 批量合并结果（更快）
                logger.info("合并结果...")
                for chunk_res in chunk_results:
                    for rec, daily_ret in chunk_res:
                        all_recs.append(rec)
                        per_strategy_returns[rec["_key"]] = daily_ret

        else:
            # 单进程（调试用）
            for i, chunk in enumerate(chunks):
                chunk_res = WFOStrategyEvaluator.evaluate_chunk(
                    chunk, results_list, factors, returns, factor_names, dates
                )
                for rec, daily_ret in chunk_res:
                    all_recs.append(rec)
                    per_strategy_returns[rec["_key"]] = daily_ret

                if (i + 1) % 10 == 0:
                    logger.info(
                        f"进度: {(i+1)*self.chunk_size}/{len(specs_to_compute)} ({(i+1)*self.chunk_size/len(specs_to_compute)*100:.1f}%)"
                    )

        # 4. 构建DataFrame
        df_new = pd.DataFrame(all_recs)

        # 5. 合并已存在结果
        if existing_keys:
            if self.use_parquet:
                df_existing = pd.read_parquet(output_file)
            else:
                df_existing = pd.read_csv(output_file)
            df = pd.concat([df_existing, df_new], ignore_index=True)
            logger.info(f"合并结果: {len(df_existing)} + {len(df_new)} = {len(df)}")
        else:
            df = df_new

        # 6. 保存（注意：这里只保存未排序的数据，排序由主选择器完成）
        if self.use_parquet:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, output_file, compression="snappy")
            logger.info(f"保存Parquet（未排序）: {output_file} (压缩格式: snappy)")
        else:
            df.to_csv(output_file, index=False)
            logger.info(f"保存CSV（未排序）: {output_file}")

        # 7. 返回结果和收益字典（不在此处保存，由主选择器处理）
        return df, per_strategy_returns

    @staticmethod
    def save_enumeration_audit(audit_data: Dict, out_dir: Path):
        """保存枚举审计报告"""
        with open(out_dir / "enumeration_audit.json", "w") as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False)
