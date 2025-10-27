#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分批回测执行器 - 按调仓周期分批运行大规模回测

用于避免内存溢出，将大规模回测任务分解为多个小批次。
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


class BatchBacktestRunner:
    """分批回测执行器"""

    def __init__(self, config_file: str = "parallel_backtest_config.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.results_dir = Path(self.config["data_paths"]["output_dir"])

    def _load_config(self) -> dict:
        """加载配置文件"""
        with open(self.config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _save_config(self, config: dict):
        """保存配置文件"""
        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.dump(
                config, f, allow_unicode=True, default_flow_style=False, sort_keys=False
            )

    def _estimate_strategies(self, rebalance_freqs: list) -> int:
        """估算策略数量"""
        n_top_n = len(self.config["backtest_config"]["top_n_list"])
        n_rebalance = len(rebalance_freqs)
        n_weights = self.config["weight_grid"].get("max_combinations", 10000)
        return n_top_n * n_rebalance * n_weights

    def run_single_batch(
        self, batch_rebalance_freqs: list, batch_idx: int, total_batches: int
    ):
        """运行单个批次"""
        n_strategies = self._estimate_strategies(batch_rebalance_freqs)

        print(f"\n{'='*80}")
        print(f"📦 批次 {batch_idx}/{total_batches}")
        print(f"   调仓周期: {batch_rebalance_freqs}")
        print(f"   预计策略数: {n_strategies:,}")
        print(f"{'='*80}\n")

        # 临时修改配置文件
        original_freqs = self.config["backtest_config"]["rebalance_freq_list"].copy()
        self.config["backtest_config"]["rebalance_freq_list"] = batch_rebalance_freqs
        self._save_config(self.config)

        try:
            # 运行回测
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, "parallel_backtest_configurable.py"],
                cwd=self.config_file.parent,
                capture_output=False,  # 显示实时输出
                text=True,
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                print(
                    f"\n✅ 批次 {batch_idx}/{total_batches} 完成 (耗时: {elapsed/60:.1f}分钟)"
                )
                return True
            else:
                print(
                    f"\n❌ 批次 {batch_idx}/{total_batches} 失败 (退出码: {result.returncode})"
                )
                return False

        finally:
            # 恢复原始配置
            self.config["backtest_config"]["rebalance_freq_list"] = original_freqs
            self._save_config(self.config)

    def merge_results(self, batch_results: list) -> Path:
        """合并所有批次的结果"""
        print(f"\n{'='*80}")
        print("📊 合并所有批次结果...")
        print(f"{'='*80}\n")

        all_dfs = []
        for result_dir in batch_results:
            csv_file = result_dir / "results.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
                print(f"  ✓ 加载: {result_dir.name} ({len(df):,} 策略)")

        if not all_dfs:
            print("❌ 没有找到任何结果文件！")
            return None

        # 合并并重新排序
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df = merged_df.sort_values("sharpe_ratio", ascending=False)

        # 截取Top N结果
        save_top = self.config["output_config"]["save_top_results"]
        merged_df = merged_df.head(save_top)

        # 保存合并结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_dir = self.results_dir / f"backtest_merged_{timestamp}"
        merged_dir.mkdir(parents=True, exist_ok=True)

        output_csv = merged_dir / "results.csv"
        merged_df.to_csv(output_csv, index=False)

        print(f"\n✅ 合并完成:")
        print(f"   总批次: {len(all_dfs)}")
        print(f"   总策略: {sum(len(df) for df in all_dfs):,}")
        print(f"   保存Top: {len(merged_df)}")
        print(f"   输出位置: {output_csv}")

        # 生成摘要
        self._generate_summary(merged_df, merged_dir, len(all_dfs))

        return merged_dir

    def _generate_summary(self, df: pd.DataFrame, output_dir: Path, n_batches: int):
        """生成合并结果摘要"""
        summary_file = output_dir / "summary.txt"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("分批回测合并结果摘要\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"合并批次数: {n_batches}\n")
            f.write(f"总策略数: {len(df):,}\n")
            f.write(
                f"配置: {self.config['backtest_config']['top_n_list']} Top-N × "
                f"{self.config['backtest_config']['rebalance_freq_list']} 调仓周期\n\n"
            )

            f.write("-" * 80 + "\n")
            f.write("Top 10 策略:\n")
            f.write("-" * 80 + "\n")

            top10 = df.head(10)
            for idx, row in top10.iterrows():
                f.write(f"\n#{idx+1}\n")
                f.write(f"  Sharpe: {row['sharpe_ratio']:.3f}\n")
                f.write(f"  总收益: {row['total_return']:.2f}%\n")
                # annual_return可能不存在，使用total_return替代
                if "annual_return" in df.columns:
                    f.write(f"  年化收益: {row['annual_return']:.2%}\n")
                f.write(f"  最大回撤: {row['max_drawdown']:.2f}%\n")
                f.write(f"  Top-N: {row['top_n']} | 调仓: {row['rebalance_freq']}日\n")
                if "factors" in df.columns:
                    f.write(f"  因子: {row['factors']}\n")
                f.write(f"  权重: {row['weights']}\n")

        print(f"   摘要文件: {summary_file}")

    def run_batched(self, batch_size: int = 1):
        """
        分批运行回测

        Args:
            batch_size: 每批次包含的调仓周期数（默认1，即每个调仓周期单独运行）
        """
        rebalance_freqs = self.config["backtest_config"]["rebalance_freq_list"]

        # 计算批次
        batches = []
        for i in range(0, len(rebalance_freqs), batch_size):
            batch = rebalance_freqs[i : i + batch_size]
            batches.append(batch)

        total_batches = len(batches)
        total_strategies = self._estimate_strategies(rebalance_freqs)

        print(f"\n{'='*80}")
        print(f"🚀 分批回测启动")
        print(f"{'='*80}")
        print(f"总调仓周期: {len(rebalance_freqs)} → {rebalance_freqs}")
        print(f"分批方案: {total_batches} 个批次，每批 {batch_size} 个调仓周期")
        print(f"总策略数: {total_strategies:,}")
        print(f"预计每批次策略数: ~{total_strategies//total_batches:,}")
        print(f"{'='*80}\n")

        # 记录成功的批次结果目录
        successful_results = []

        # 逐批次执行
        for batch_idx, batch_freqs in enumerate(batches, 1):
            # 执行前获取现有结果目录
            existing_dirs = (
                set(self.results_dir.glob("backtest_*"))
                if self.results_dir.exists()
                else set()
            )

            success = self.run_single_batch(batch_freqs, batch_idx, total_batches)

            if success:
                # 查找新生成的结果目录
                new_dirs = set(self.results_dir.glob("backtest_*")) - existing_dirs
                if new_dirs:
                    newest = max(new_dirs, key=lambda p: p.stat().st_mtime)
                    successful_results.append(newest)
                    print(f"   结果保存至: {newest.name}")
            else:
                print(f"⚠️  批次 {batch_idx} 失败，但继续执行后续批次...")

            # 批次间短暂休息，释放资源
            if batch_idx < total_batches:
                print(f"\n⏸  暂停5秒，释放系统资源...\n")
                time.sleep(5)

        # 合并结果
        if successful_results:
            merged_dir = self.merge_results(successful_results)

            print(f"\n{'='*80}")
            print(f"🎉 分批回测全部完成！")
            print(f"{'='*80}")
            print(f"成功批次: {len(successful_results)}/{total_batches}")
            print(f"最终结果: {merged_dir}")
            print(f"{'='*80}\n")

            return merged_dir
        else:
            print(f"\n❌ 所有批次均失败，无结果可合并")
            return None


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="分批回测执行器")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="每批次包含的调仓周期数（默认1=每个周期单独运行）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="parallel_backtest_config.yaml",
        help="配置文件路径",
    )

    args = parser.parse_args()

    runner = BatchBacktestRunner(config_file=args.config)
    runner.run_batched(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
