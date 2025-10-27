#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 存储架构模块 - 实现时间戳文件夹和 TOP 300 保存

功能:
1. 按启动时间建立时间戳文件夹
2. 只保存 TOP 300 策略结果
3. 完整日志记录
4. 分周期结果保存
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class OptimizedResultStorage:
    """优化版结果存储管理器"""

    def __init__(self, base_dir: str, strategy_limit: int = 300):
        """
        初始化存储管理器

        Args:
            base_dir: 基础存储目录
            strategy_limit: 保存策略数限制 (默认 TOP 300)
        """
        self.base_dir = Path(base_dir)
        self.strategy_limit = strategy_limit

        # 创建时间戳文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / f"{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        self.results_dir = self.run_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.wfo_dir = self.run_dir / "wfo_periods"
        self.wfo_dir.mkdir(exist_ok=True)

        # 初始化日志
        self.logger = self._init_logger()
        self.logger.info(f"创建运行目录: {self.run_dir}")

    def _init_logger(self) -> logging.Logger:
        """初始化日志系统"""
        logger = logging.getLogger(__name__)

        # 文件处理器
        log_file = self.run_dir / "run_log.txt"
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger

    def save_config(self, config: Dict[str, Any]) -> None:
        """保存回测配置"""
        config_file = self.run_dir / "config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            # 简单序列化为 JSON (YAML 需要额外依赖)
            json.dump(config, f, indent=2, default=str, ensure_ascii=False)
        self.logger.info(f"配置已保存: {config_file}")

    def save_results(
        self, results: List[Dict[str, Any]], period_id: Optional[int] = None
    ) -> None:
        """
        保存结果 (只保存 TOP N)

        Args:
            results: 回测结果列表
            period_id: WFO 周期 ID (如果是分周期保存)
        """
        if not results:
            self.logger.warning("没有结果可保存")
            return

        # 按 Sharpe 排序
        sorted_results = sorted(
            results, key=lambda x: x.get("sharpe_ratio", 0), reverse=True
        )

        # 取 TOP N
        top_results = sorted_results[: self.strategy_limit]

        if period_id:
            # 分周期保存
            period_dir = self.wfo_dir / f"period_{period_id}"
            period_dir.mkdir(exist_ok=True)

            # 区分 IS 和 OOS
            is_results = [r for r in top_results if r.get("data_type") == "IS"]
            oos_results = [r for r in top_results if r.get("data_type") == "OOS"]

            # 保存 CSV
            if is_results:
                csv_file = period_dir / "top_30_is.csv"
                df = pd.DataFrame(is_results)
                df.to_csv(csv_file, index=False, encoding="utf-8")
                self.logger.info(f"已保存 IS 结果: {csv_file} ({len(is_results)})")

            if oos_results:
                csv_file = period_dir / "top_30_oos.csv"
                df = pd.DataFrame(oos_results)
                df.to_csv(csv_file, index=False, encoding="utf-8")
                self.logger.info(f"已保存 OOS 结果: {csv_file} ({len(oos_results)})")
        else:
            # 全局保存

            # CSV 格式
            csv_file = self.results_dir / f"top_{self.strategy_limit}_detailed.csv"
            df = pd.DataFrame(top_results)
            df.to_csv(csv_file, index=False, encoding="utf-8")
            self.logger.info(f"已保存详细结果: {csv_file}")

            # 🟢 Parquet 格式 (新标准格式，不覆盖历史数据)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parquet_file = self.results_dir / f"wfo_results_{timestamp}.parquet"
            df = pd.DataFrame(top_results)
            df.to_parquet(parquet_file, compression="snappy", index=False)
            self.logger.info(f"✓ 已保存 Parquet 格式 (新标准): {parquet_file}")

            # ⚠️ 保留 Pickle 格式以兼容现有代码
            pkl_file = self.results_dir / f"top_{self.strategy_limit}.pkl"
            with open(pkl_file, "wb") as f:
                pickle.dump(top_results, f)
            self.logger.info(f"⚠️ 已保存 Pickle 格式 (兼容): {pkl_file}")

    def save_summary(self, summary: Dict[str, Any]) -> None:
        """保存统计摘要"""
        summary_file = self.results_dir / "summary_stats.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        self.logger.info(f"统计摘要已保存: {summary_file}")

    def save_optimization_metadata(self, metadata: Dict[str, Any]) -> None:
        """保存优化元数据"""
        meta_file = self.run_dir / "optimization_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "run_dir": str(self.run_dir),
                    "strategy_limit": self.strategy_limit,
                    **metadata,
                },
                f,
                indent=2,
                default=str,
                ensure_ascii=False,
            )
        self.logger.info(f"优化元数据已保存: {meta_file}")

    def get_run_info(self) -> Dict[str, str]:
        """获取运行信息"""
        return {
            "run_dir": str(self.run_dir),
            "results_dir": str(self.results_dir),
            "wfo_dir": str(self.wfo_dir),
            "timestamp": self.run_dir.name,
        }


# ============================================================================
# 使用示例
# ============================================================================


def example_usage():
    """示例用法"""

    # 1. 初始化存储
    storage = OptimizedResultStorage(
        base_dir="/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/vbtwfo",
        strategy_limit=300,
    )

    # 2. 保存配置
    config = {
        "rebalance_freq_list": [5, 10, 15],
        "top_n_list": [1],
        "total_strategies": 270000,
        "optimization": "frequency_topn_focused",
    }
    storage.save_config(config)

    # 3. 保存分周期结果
    for period_id in range(1, 11):
        period_results = [
            {
                "period_id": period_id,
                "data_type": "IS",
                "sharpe_ratio": 1.5 + period_id * 0.1,
                "return": 0.05 + period_id * 0.01,
            },
            {
                "period_id": period_id,
                "data_type": "OOS",
                "sharpe_ratio": 6.5 + period_id * 0.1,
                "return": 0.23 + period_id * 0.02,
            },
        ]
        storage.save_results(period_results, period_id=period_id)

    # 4. 保存全局结果
    all_results = [
        {"rank": i, "sharpe_ratio": 7.0 - i * 0.01, "return": 0.28 - i * 0.0001}
        for i in range(1, 301)
    ]
    storage.save_results(all_results)

    # 5. 保存摘要
    summary = {
        "total_strategies": 270000,
        "top_strategies_saved": 300,
        "mean_sharpe": 6.59,
        "mean_return": 0.234,
    }
    storage.save_summary(summary)

    # 6. 保存元数据
    metadata = {
        "optimization_applied": "frequency_topn_focused",
        "expected_speedup": "9.4×",
        "expected_storage_reduction": "96%",
    }
    storage.save_optimization_metadata(metadata)

    # 打印运行信息
    info = storage.get_run_info()
    print("\n📂 运行信息:")
    for key, val in info.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    example_usage()
    print("\n✅ 存储系统测试完成")
