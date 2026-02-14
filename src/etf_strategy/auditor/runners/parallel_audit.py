import backtrader as bt
import pandas as pd
import numpy as np
import sys
import os
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import multiprocessing
from functools import partial

import logging
import time
import gc
from contextlib import contextmanager

# Add project root to path
sys.path.insert(0, os.getcwd())

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    generate_rebalance_schedule,
    shift_timing_signal,
)
from etf_strategy.regime_gate import compute_regime_gate_arr
from etf_strategy.auditor.core.engine import (
    COMMISSION_RATE,
    FREQ,
    INITIAL_CAPITAL,
    LOOKBACK,
    POS_SIZE,
    GenericStrategy,
    PandasData,
)

# Global variables for workers (Copy-on-Write optimization)
GLOBAL_DATA_FEEDS = {}
GLOBAL_STD_FACTORS = {}
GLOBAL_TIMING = None
GLOBAL_REBALANCE_SCHEDULE = None
GLOBAL_FREQ = None
GLOBAL_POS_SIZE = None
GLOBAL_ETF_CODES = []
GLOBAL_INDEX = None
# GLOBAL_OUTPUT_DIR removed as we return data now


@contextmanager
def timer(logger, description):
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"⏱️ {description}: {elapsed:.2f}s")


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def worker_task(strategy_row):
    """
    Worker function to run a single backtest.
    Uses global variables to access data (efficient on Linux via fork).
    """
    t_start = time.time()
    metrics = {}

    try:
        combo_name = strategy_row["combo"]
        orig_return = strategy_row["total_return"]
        # Support both 'real_rank' and 'rank' column names
        rank = strategy_row.get("real_rank", strategy_row.get("rank", 0))

        factors = [f.strip() for f in combo_name.split(" + ")]

        # Parse factor_signs (IC-sign-aware direction)
        factor_signs_raw = strategy_row.get("factor_signs")
        if factor_signs_raw and pd.notna(factor_signs_raw):
            factor_signs = [int(s) for s in str(factor_signs_raw).split(",")]
        else:
            factor_signs = [1] * len(factors)

        # Combine Scores
        t_score = time.time()
        combined_score = pd.DataFrame(0.0, index=GLOBAL_INDEX, columns=GLOBAL_ETF_CODES)
        valid_factors = True
        for f, sign in zip(factors, factor_signs):
            if f not in GLOBAL_STD_FACTORS:
                valid_factors = False
                break
            if sign < 0:
                combined_score = combined_score.add(-GLOBAL_STD_FACTORS[f], fill_value=0)
            else:
                combined_score = combined_score.add(GLOBAL_STD_FACTORS[f], fill_value=0)
        metrics["score_calc"] = time.time() - t_score

        if not valid_factors:
            return None

        # Run Backtrader
        t_setup = time.time()
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(INITIAL_CAPITAL)
        # 使用 1.0 杠杆，匹配实际无杠杆交易场景
        cerebro.broker.setcommission(commission=COMMISSION_RATE, leverage=1.0)
        cerebro.broker.set_coc(True)
        cerebro.broker.set_checksubmit(False)

        # Add data feeds (referencing global dict)
        for ticker, df in GLOBAL_DATA_FEEDS.items():
            data = PandasData(dataname=df, name=ticker)
            cerebro.adddata(data)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

        cerebro.addstrategy(
            GenericStrategy,
            scores=combined_score,
            timing=GLOBAL_TIMING,
            etf_codes=GLOBAL_ETF_CODES,
            freq=GLOBAL_FREQ,
            pos_size=GLOBAL_POS_SIZE,
            rebalance_schedule=GLOBAL_REBALANCE_SCHEDULE,
        )
        metrics["bt_setup"] = time.time() - t_setup

        # Run without output
        t_run = time.time()
        start_val = cerebro.broker.getvalue()
        results = cerebro.run()
        end_val = cerebro.broker.getvalue()
        strat = results[0]
        metrics["bt_run"] = time.time() - t_run

        bt_return = (end_val / start_val) - 1
        diff = bt_return - orig_return

        # Prepare DataFrames for return (don't save to file here)
        trades_df = None
        equity_df = None

        t_process = time.time()
        # Extract Trades
        if strat.trades:
            trades_df = pd.DataFrame(strat.trades)
            # Add identifier columns
            trades_df["rank"] = rank
            trades_df["combo"] = combo_name

        # Extract Equity Curve
        timereturns = strat.analyzers.timereturn.get_analysis()
        equity_data = list(timereturns.items())
        if equity_data:
            equity_df = pd.DataFrame(equity_data, columns=["date", "return"])
            equity_df["date"] = pd.to_datetime(equity_df["date"])
            equity_df = equity_df.set_index("date").sort_index()
            equity_df["equity"] = INITIAL_CAPITAL * (1 + equity_df["return"]).cumprod()
            equity_df["rank"] = rank
            equity_df = equity_df.reset_index()
        metrics["result_process"] = time.time() - t_process

        metrics["total"] = time.time() - t_start

        # Memory cleanup - critical for preventing OOM in long runs
        del cerebro, results, strat, combined_score
        gc.collect()

        return {
            "summary": {
                "real_rank": rank,
                "combo": combo_name,
                "orig_return": orig_return,
                "bt_return": bt_return,
                "diff": diff,
                "bt_final_equity": end_val,
                "time_total": metrics["total"],
                "time_bt_run": metrics["bt_run"],
            },
            "trades": trades_df,
            "equity": equity_df,
            "metrics": metrics,
        }
    except Exception:
        import traceback

        combo_name = None
        try:
            combo_name = strategy_row.get("combo")
        except Exception:
            combo_name = None

        return {
            "summary": {
                "combo": combo_name,
                "error": traceback.format_exc(),
            },
            "trades": None,
            "equity": None,
            "metrics": {"total": time.time() - t_start},
        }


def run_audit(input_file: str, output_file: str = None, top_n: int = None):
    # Create Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("etf_strategy.auditor/results")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup Logging
    log_file = run_dir / "audit.log"
    logger = setup_logging(log_file)

    logger.info(f"🚀 启动 Backtrader 独立审计...")
    logger.info(f"📂 输入: {input_file}")
    logger.info(f"📂 输出目录: {run_dir}")

    # 1. Load Strategies
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error("❌ Input file not found!")
        return

    if input_path.suffix == ".parquet":
        df_summary = pd.read_parquet(input_path)
    else:
        df_summary = pd.read_csv(input_path)

    if top_n:
        strategies_to_run = df_summary.head(top_n)
        logger.info(f"🔍 仅审计 Top {top_n} 策略")
    else:
        strategies_to_run = df_summary
        logger.info(f"🔍 审计全量 {len(strategies_to_run)} 策略")

    # 2. Load Config & Data
    config_path = Path("configs/combo_wfo_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # 3. Compute Factors
    with timer(logger, "计算因子"):
        factor_lib = PreciseFactorLibrary()
        raw_factors_df = factor_lib.compute_all_factors(ohlcv)

    with timer(logger, "标准化因子"):
        factor_names = raw_factors_df.columns.get_level_values(0).unique().tolist()
        raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}

        processor = CrossSectionProcessor(verbose=False)
        std_factors = processor.process_all_factors(raw_factors)

    # 4. Timing
    timing_module = LightTimingModule()
    # 使用共享 helper shift_timing_signal (T-1的信号决定T的仓位)，与 VEC 引擎对齐
    timing_series_raw = timing_module.compute_position_ratios(ohlcv["close"])
    timing_series = pd.Series(
        shift_timing_signal(timing_series_raw.values), index=timing_series_raw.index
    )

    # 5. Prepare Globals
    global GLOBAL_DATA_FEEDS, GLOBAL_STD_FACTORS, GLOBAL_TIMING
    global GLOBAL_REBALANCE_SCHEDULE, GLOBAL_FREQ, GLOBAL_POS_SIZE
    global GLOBAL_ETF_CODES, GLOBAL_INDEX

    backtest_cfg = config.get("backtest", {})
    GLOBAL_FREQ = int(backtest_cfg.get("freq", FREQ))
    GLOBAL_POS_SIZE = int(backtest_cfg.get("pos_size", POS_SIZE))

    # ✅ v3.2: Regime gate（可选），统一缩放 timing
    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], timing_series.index, backtest_config=backtest_cfg
    )
    timing_series = timing_series * pd.Series(gate_arr, index=timing_series.index)

    GLOBAL_STD_FACTORS = std_factors
    GLOBAL_TIMING = timing_series
    GLOBAL_ETF_CODES = ohlcv["close"].columns.tolist()
    GLOBAL_INDEX = std_factors[factor_names[0]].index

    # ✅ Use the shared rebalance schedule convention (indices), aligned with VEC/BT.
    GLOBAL_REBALANCE_SCHEDULE = generate_rebalance_schedule(
        total_periods=len(GLOBAL_INDEX),
        lookback_window=LOOKBACK,
        freq=GLOBAL_FREQ,
    )

    with timer(logger, "准备数据 Feeds"):
        for ticker in GLOBAL_ETF_CODES:
            df = pd.DataFrame(
                {
                    "open": ohlcv["open"][ticker],
                    "high": ohlcv["high"][ticker],
                    "low": ohlcv["low"][ticker],
                    "close": ohlcv["close"][ticker],
                    "volume": ohlcv["volume"][ticker],
                }
            )
            df = df.reindex(GLOBAL_INDEX)
            df = df.ffill().fillna(0.01)
            GLOBAL_DATA_FEEDS[ticker] = df

    # 6. Parallel Execution
    # Ryzen 9 9950X has 16 cores / 32 threads.
    # WARNING: Each backtrader process can use 1.5-2GB RAM due to fork() copy-on-write
    # With 46GB RAM, using 24+ processes caused OOM and system crash!
    # SAFE LIMIT: 16 processes = ~32GB max, leaving ~14GB for system + buffers
    total_cores = multiprocessing.cpu_count()
    num_cores = min(total_cores, 16)  # HARD LIMIT: 16 to prevent OOM
    logger.info(f"🔥 使用 {num_cores} 个核心并行回测 (Total Cores: {total_cores})...")

    summary_results = []
    trades_buffer = []
    equity_buffer = []
    chunk_counter = 0

    strategies_list = [row for _, row in strategies_to_run.iterrows()]
    # Smaller chunksize for better load balancing
    chunksize = max(1, len(strategies_list) // (num_cores * 8))

    # Create directories for partitioned parquet
    (run_dir / "trades").mkdir(exist_ok=True, parents=True)
    (run_dir / "equity").mkdir(exist_ok=True, parents=True)

    t_start_parallel = time.time()

    metrics_total = {
        "score_calc": 0,
        "bt_setup": 0,
        "bt_run": 0,
        "result_process": 0,
        "total": 0,
    }
    metrics_count = 0

    with multiprocessing.Pool(processes=num_cores) as pool:
        for res in tqdm(
            pool.imap(worker_task, strategies_list, chunksize=chunksize),
            total=len(strategies_list),
            desc="Auditing",
        ):
            if res:
                summary_results.append(res["summary"])

                # Aggregate metrics
                if "metrics" in res:
                    for k in metrics_total:
                        if k in res["metrics"]:
                            metrics_total[k] += res["metrics"][k]
                    metrics_count += 1

                if res["trades"] is not None:
                    trades_buffer.append(res["trades"])

                if res["equity"] is not None:
                    equity_buffer.append(res["equity"])

                # Flush every 500 results
                if len(summary_results) % 500 == 0:
                    # Save Summary Partial
                    pd.DataFrame(summary_results).to_csv(
                        run_dir / "summary_partial.csv", index=False
                    )

                    # Save Trades Chunk
                    if trades_buffer:
                        pd.concat(trades_buffer).to_parquet(
                            run_dir / "trades" / f"part_{chunk_counter}.parquet",
                            index=False,
                        )
                        trades_buffer = []

                    # Save Equity Chunk
                    if equity_buffer:
                        pd.concat(equity_buffer).to_parquet(
                            run_dir / "equity" / f"part_{chunk_counter}.parquet",
                            index=False,
                        )
                        equity_buffer = []

                    chunk_counter += 1

    elapsed_parallel = time.time() - t_start_parallel
    logger.info(
        f"⏱️ 并行回测总耗时: {elapsed_parallel:.2f}s (平均 {elapsed_parallel/len(strategies_list):.4f}s/策略)"
    )

    if metrics_count > 0:
        logger.info("📊 性能耗时分布 (平均每策略):")
        for k, v in metrics_total.items():
            avg_time = v / metrics_count
            pct = (v / metrics_total["total"]) * 100
            logger.info(f"   - {k:<15}: {avg_time*1000:.2f}ms ({pct:.1f}%)")

    # Final Flush
    if trades_buffer:
        pd.concat(trades_buffer).to_parquet(
            run_dir / "trades" / f"part_{chunk_counter}.parquet", index=False
        )
    if equity_buffer:
        pd.concat(equity_buffer).to_parquet(
            run_dir / "equity" / f"part_{chunk_counter}.parquet", index=False
        )

    # Final Summary
    df_res = pd.DataFrame(summary_results)
    summary_file = run_dir / "summary.csv"
    df_res.to_csv(summary_file, index=False)

    logger.info("-" * 100)
    logger.info(f"💾 审计报告已保存至: {summary_file}")
    logger.info(f"💾 交易记录已保存至: {run_dir}/trades/ (Parquet)")
    logger.info(f"💾 净值曲线已保存至: {run_dir}/equity/ (Parquet)")
    logger.info(f"📝 运行日志已保存至: {log_file}")

    if df_res.empty:
        logger.warning(
            "⚠️ 未产生任何有效 summary 结果（可能所有策略在 worker 内报错）。"
        )
        logger.info(f"✅ 审计完成")
        return

    if "diff" in df_res.columns:
        avg_diff = df_res["diff"].mean()
        logger.info(f"📊 平均差异: {avg_diff*100:.2f}%")
    else:
        logger.warning("⚠️ summary.csv 中缺少 diff 字段（可能存在 worker error）。")

    if "bt_return" in df_res.columns:
        logger.info("🏆 Backtrader 验证后 Top 20:")
        df_res_sorted = df_res.sort_values("bt_return", ascending=False).head(20)
        cols = [
            c
            for c in ["real_rank", "bt_return", "orig_return", "diff", "combo", "error"]
            if c in df_res_sorted.columns
        ]
        logger.info("\n" + df_res_sorted[cols].to_string(index=False))
    else:
        logger.warning("⚠️ summary.csv 中缺少 bt_return 字段（可能存在 worker error）。")

    logger.info(f"✅ 审计完成")


if __name__ == "__main__":
    # Default behavior if run directly
    run_audit("results/top5000_summary.csv", "results/audit_report.csv")
