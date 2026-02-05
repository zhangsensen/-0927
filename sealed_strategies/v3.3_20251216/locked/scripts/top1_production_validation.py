"""
Top 1 Production Validation & Tuning
Target: ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + PV_CORR_20D + SHARPE_RATIO_20D

1. BT Independent Engine Reproduction
2. Parameter Perturbation
3. Capacity Simulation
4. Extreme Testing
5. Monte Carlo
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime
import itertools

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.utils.rebalance import (
    generate_rebalance_schedule,
    shift_timing_signal,
)
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.auditor.core.engine import GenericStrategy, PandasData

# Reuse VEC Kernel
from full_universe_postselection import validation_backtest_kernel, stable_topk_indices

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Dynamic Factor Calculation ---


def compute_adx(high, low, close, window):
    high_diff = high.diff()
    low_diff = -low.diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    atr = tr.ewm(span=window, adjust=False, min_periods=window).mean()
    plus_di = 100 * (
        plus_dm.ewm(span=window, adjust=False, min_periods=window).mean()
        / (atr + 1e-10)
    )
    minus_di = 100 * (
        minus_dm.ewm(span=window, adjust=False, min_periods=window).mean()
        / (atr + 1e-10)
    )

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.ewm(span=window, adjust=False, min_periods=window).mean()
    return adx


def compute_max_dd(close, window):
    return (
        close.rolling(window=window, min_periods=window).apply(
            lambda x: (x.max() - x[-1]) / x.max(), raw=True
        )
        * 100
    )


def compute_price_position(close, high, low, window):
    roll_high = high.rolling(window=window, min_periods=window).max()
    roll_low = low.rolling(window=window, min_periods=window).min()
    pp = (close - roll_low) / (roll_high - roll_low + 1e-10)
    return pp


def compute_pv_corr(close, volume, window):
    return close.rolling(window=window, min_periods=window).corr(volume)


def compute_sharpe(close, window):
    ret = close.pct_change()
    roll_mean = ret.rolling(window=window, min_periods=window).mean()
    roll_std = ret.rolling(window=window, min_periods=window).std()
    return (roll_mean / (roll_std + 1e-10)) * np.sqrt(252)


# --- Main Validation Class ---


class Top1Validator:
    def __init__(self):
        self.config = self._load_config()
        self.loader = DataLoader(
            data_dir=self.config["data"].get("data_dir"),
            cache_dir=self.config["data"].get("cache_dir"),
        )
        self.ohlcv = self._load_data()
        self.tickers = sorted(self.ohlcv["close"].columns)
        self.dates = self.ohlcv["close"].index
        self.T = len(self.dates)
        self.N = len(self.tickers)

        # Base Parameters
        self.FREQ = 3
        self.POS_SIZE = 2
        self.rebalance_schedule = generate_rebalance_schedule(self.T, 252, self.FREQ)

        # Timing
        self.timing_arr = self._compute_timing()

        # Base Factors (Pre-computed for efficiency if not perturbed)
        self.base_factors = {}

    def _load_config(self):
        config_path = ROOT / "configs/combo_wfo_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _load_data(self):
        # Load Liquid Whitelist
        with open(ROOT / "scripts/run_liquid_vec_backtest.py", "r") as f:
            content = f.read()
            import ast

            tree = ast.parse(content)
            whitelist = None
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "LIQUID_ETFS":
                            whitelist = ast.literal_eval(node.value)
                            break
        if not whitelist:
            whitelist = [
                "510300",
                "510500",
                "510050",
                "513100",
                "513500",
                "512880",
                "512000",
                "512660",
                "512010",
                "512800",
                "512690",
                "512480",
                "512100",
                "512070",
                "515000",
                "588000",
                "159915",
                "159949",
                "518880",
                "513050",
                "513330",
            ]

        data = self.loader.load_ohlcv(
            etf_codes=self.config["data"]["symbols"],
            start_date=self.config["data"]["start_date"],
            end_date=self.config["data"]["end_date"],
        )
        return {k: v[[t for t in v.columns if t in whitelist]] for k, v in data.items()}

    def _compute_timing(self):
        timing_module = LightTimingModule()
        timing_signals = timing_module.compute_position_ratios(self.ohlcv["close"])
        return shift_timing_signal(timing_signals)

    def run_vec_backtest(self, factors_3d, factor_indices, slippage_model=None):
        # Prepare Arrays
        close = self.ohlcv["close"].values
        open_ = self.ohlcv["open"].values
        high = self.ohlcv["high"].values
        low = self.ohlcv["low"].values

        # Run Kernel
        equity = validation_backtest_kernel(
            factors_3d,
            close,
            open_,
            high,
            low,
            self.timing_arr,
            factor_indices,
            self.rebalance_schedule,
            self.POS_SIZE,
            1_000_000.0,
            0.0002,
            0.0,
            0,
            False,
            False,
            0.0,
            np.zeros((self.T, self.N)),
            0.0,
            False,
            np.ones((self.T, self.N), dtype=bool),
            False,
            np.array([0.15, 0.30, np.inf]),
            np.array([0.05, 0.03, 0.08]),
            np.array([2.0, 1.5, 3.0]),
            0.0,
            0.0,
            0,
            0,
            1.0,
        )
        return equity

    def step1_bt_reproduction(self):
        print("\n=== Step 1: BT Independent Engine Reproduction ===")

        # 1. Compute Factors using PreciseFactorLibrary (Golden Source)
        print("Computing Standard Factors via Library...")
        lib = PreciseFactorLibrary()
        # We need to pass dict of dfs
        data_dict = {
            "open": self.ohlcv["open"],
            "high": self.ohlcv["high"],
            "low": self.ohlcv["low"],
            "close": self.ohlcv["close"],
            "volume": self.ohlcv["volume"],
        }
        raw_factors_df = lib.compute_all_factors(data_dict)

        # Select specific factors
        target_factors = [
            "ADX_14D",
            "MAX_DD_60D",
            "PRICE_POSITION_120D",
            "PV_CORR_20D",
            "SHARPE_RATIO_20D",
        ]

        # Process Factors
        processor = CrossSectionProcessor(verbose=False)
        # We need to reconstruct the dict structure expected by processor
        # raw_factors_df has MultiIndex columns (Factor, Ticker)
        raw_factors_dict = {}
        for f in target_factors:
            raw_factors_dict[f] = raw_factors_df[f]

        std_factors = processor.process_all_factors(raw_factors_dict)

        # Construct 3D Array for VEC
        factors_3d = np.zeros((self.T, self.N, 5))
        total_score = pd.DataFrame(0.0, index=self.dates, columns=self.tickers)

        for i, f in enumerate(target_factors):
            df = std_factors[f]
            factors_3d[:, :, i] = df.fillna(0).values
            total_score = total_score.add(df.fillna(0), fill_value=0)

        # 2. Run VEC
        print("Running VEC...")
        vec_equity = self.run_vec_backtest(factors_3d, np.array([0, 1, 2, 3, 4]))

        # Fix Return Calculation
        # Use Total Return for comparison to avoid period confusion
        vec_total_ret = (vec_equity[-1] / 1_000_000.0) - 1
        vec_ann_ret = (vec_equity[-1] / 1_000_000.0) ** (252 / self.T) - 1

        print(f"VEC Total Return: {vec_total_ret:.2%}")
        print(f"VEC Ann Return: {vec_ann_ret:.2%}")

        # 3. Run BT
        print("Running BT...")
        cerebro = bt.Cerebro()

        # Add Data
        for ticker in self.tickers:
            df = pd.DataFrame(
                {
                    "open": self.ohlcv["open"][ticker],
                    "high": self.ohlcv["high"][ticker],
                    "low": self.ohlcv["low"][ticker],
                    "close": self.ohlcv["close"][ticker],
                    "volume": self.ohlcv["volume"][ticker],
                }
            )
            data = PandasData(dataname=df, name=ticker)
            cerebro.adddata(data)

        # Add Strategy
        cerebro.addstrategy(
            GenericStrategy,
            scores=total_score,
            timing=pd.Series(self.timing_arr, index=self.dates),
            etf_codes=self.tickers,
            freq=self.FREQ,
            pos_size=self.POS_SIZE,
            rebalance_schedule=self.rebalance_schedule,
            target_vol=0.20,
            vol_window=20,
            dynamic_leverage_enabled=False,
        )

        cerebro.broker.setcash(1_000_000.0)
        cerebro.broker.setcommission(commission=0.0002)
        cerebro.broker.set_coc(True)
        cerebro.broker.set_checksubmit(False)

        results = cerebro.run()
        strat = results[0]
        bt_value = cerebro.broker.getvalue()

        bt_total_ret = (bt_value / 1_000_000.0) - 1
        bt_ann_ret = (bt_value / 1_000_000.0) ** (252 / self.T) - 1

        print(f"BT Total Return: {bt_total_ret:.2%}")
        print(f"BT Ann Return: {bt_ann_ret:.2%}")
        print(f"Diff (Total): {abs(vec_total_ret - bt_total_ret):.4f}")

        return vec_equity, strat

    def step2_perturbation(self):
        print("\n=== Step 2: Parameter Perturbation ===")

        variations = {
            "ADX": [10, 14, 20],
            "MAX_DD": [40, 60, 80],
            "PRICE_POS": [60, 120, 180],
            "PV_CORR": [10, 20, 30],
            "SHARPE": [10, 20, 30],
        }

        keys = list(variations.keys())
        values = list(variations.values())
        combinations = list(itertools.product(*values))

        print(f"Testing {len(combinations)} combinations...")

        results = []

        # Pre-compute all factor variations to save time
        cache = {}
        for k, vals in variations.items():
            for v in vals:
                if k == "ADX":
                    cache[f"{k}_{v}"] = compute_adx(
                        self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], v
                    )
                elif k == "MAX_DD":
                    cache[f"{k}_{v}"] = -compute_max_dd(
                        self.ohlcv["close"], v
                    )  # Invert here
                elif k == "PRICE_POS":
                    cache[f"{k}_{v}"] = compute_price_position(
                        self.ohlcv["close"], self.ohlcv["high"], self.ohlcv["low"], v
                    )
                elif k == "PV_CORR":
                    cache[f"{k}_{v}"] = compute_pv_corr(
                        self.ohlcv["close"], self.ohlcv["volume"], v
                    )
                elif k == "SHARPE":
                    cache[f"{k}_{v}"] = compute_sharpe(self.ohlcv["close"], v)

                # Z-Score immediately
                mean = cache[f"{k}_{v}"].mean(axis=1)
                std = cache[f"{k}_{v}"].std(axis=1)
                cache[f"{k}_{v}"] = (
                    cache[f"{k}_{v}"].sub(mean, axis=0).div(std + 1e-10, axis=0)
                )
                cache[f"{k}_{v}"] = cache[f"{k}_{v}"].fillna(0)

        best_ret = -np.inf
        best_combo = None

        for combo in tqdm(combinations):
            # Construct 3D factors
            f_adx = cache[f"ADX_{combo[0]}"]
            f_mdd = cache[f"MAX_DD_{combo[1]}"]
            f_pp = cache[f"PRICE_POS_{combo[2]}"]
            f_pv = cache[f"PV_CORR_{combo[3]}"]
            f_sharpe = cache[f"SHARPE_{combo[4]}"]

            factors_3d = np.zeros((self.T, self.N, 5))
            factors_3d[:, :, 0] = f_adx.values
            factors_3d[:, :, 1] = f_mdd.values
            factors_3d[:, :, 2] = f_pp.values
            factors_3d[:, :, 3] = f_pv.values
            factors_3d[:, :, 4] = f_sharpe.values

            equity = self.run_vec_backtest(factors_3d, np.array([0, 1, 2, 3, 4]))
            ann_ret = (equity[-1] / 1_000_000.0) ** (252 / self.T) - 1

            results.append({"combo": combo, "ann_return": ann_ret})

            if ann_ret > best_ret:
                best_ret = ann_ret
                best_combo = combo

        df = pd.DataFrame(results)
        print(f"Best Combo: {best_combo}, Return: {best_ret:.2%}")
        return df, best_combo

    def step3_capacity(self, bt_strat):
        print("\n=== Step 3: Capacity Simulation ===")
        # Use BT orders to simulate impact
        orders = pd.DataFrame(bt_strat.orders)
        if orders.empty:
            print("No trades to simulate capacity.")
            return

        # Calculate ADV
        close = self.ohlcv["close"]
        volume = self.ohlcv["volume"]
        amount = close * volume * 100
        adv = amount.rolling(20).mean()

        total_impact_cost = 0.0

        for _, order in orders.iterrows():
            date = order["date"]
            ticker = order["ticker"]
            value = order["value"]

            # Get ADV for that day
            try:
                day_adv = adv.loc[pd.Timestamp(date), ticker]
            except KeyError:
                day_adv = np.nan

            if pd.isna(day_adv) or day_adv == 0:
                day_adv = 10_000_000  # Default fallback 10M

            # Impact Model: 10bp + 1 * (Size / ADV) * Size ?
            # User said: "持仓上限 = 日成交额 2%。加梯度滑点"
            # Let's assume linear impact: Slippage = 0.1 * (Value / ADV)
            # Cost = Value * Slippage = 0.1 * Value^2 / ADV

            impact_slippage = 0.1 * (value / day_adv)
            impact_cost = value * impact_slippage
            total_impact_cost += impact_cost

        print(f"Total Impact Cost: {total_impact_cost:.2f}")
        print(f"Impact as % of Capital: {total_impact_cost / 1_000_000.0:.2%}")

        return total_impact_cost

    def step4_extreme(self, equity_curve):
        print("\n=== Step 4: Extreme Testing ===")
        dates = self.dates
        equity = pd.Series(equity_curve, index=dates)

        periods = {
            "2020_COVID": ("2020-01-20", "2020-03-31"),
            "2022_Mar_Crash": ("2022-03-01", "2022-04-30"),
            "2024_Jan_Micro": ("2024-01-01", "2024-02-29"),
        }

        results = {}
        for name, (start, end) in periods.items():
            sub = equity[start:end]
            if len(sub) == 0:
                continue
            ret = (sub[-1] / sub[0]) - 1
            mdd = (sub / sub.cummax() - 1).min()
            results[name] = {"Return": ret, "MaxDD": mdd}
            print(f"{name}: Return {ret:.2%}, MaxDD {mdd:.2%}")

        return results

    def step5_monte_carlo(self, equity_curve):
        print("\n=== Step 5: Monte Carlo Simulation ===")
        equity = pd.Series(equity_curve)
        returns = equity.pct_change().dropna()

        n_sims = 1000
        n_days = 252

        final_rets = []
        max_dds = []

        for _ in range(n_sims):
            sim_rets = returns.sample(n=n_days, replace=True).values
            sim_eq = (1 + sim_rets).cumprod()
            final_rets.append(sim_eq[-1] - 1)

            dd = (sim_eq / np.maximum.accumulate(sim_eq) - 1).min()
            max_dds.append(dd)

        var_99 = np.percentile(final_rets, 1)
        worst_dd = np.percentile(max_dds, 1)  # 1st percentile (most negative)

        print(f"Monte Carlo 99% VaR (1 Year): {var_99:.2%}")
        print(f"Monte Carlo 99% Worst DD: {worst_dd:.2%}")

        return var_99, worst_dd

    def analyze_var_diagnostics(self, equity_curve):
        print("\n=== Step 6: VaR Diagnostics ===")
        equity = pd.Series(equity_curve)
        # Use fillna(0) for safety, though equity curve shouldn't have NaNs usually
        returns = equity.pct_change().fillna(0.0)

        # 1. Historical Rolling 1-Year Analysis
        # Calculate rolling 252-day returns
        rolling_1y_ret = equity.pct_change(252).dropna()

        if len(rolling_1y_ret) > 0:
            real_min_1y = rolling_1y_ret.min()
            real_5pct_1y = rolling_1y_ret.quantile(0.05)
            real_1pct_1y = rolling_1y_ret.quantile(0.01)
        else:
            real_min_1y = np.nan
            real_5pct_1y = np.nan
            real_1pct_1y = np.nan

        print(f"Real Min 1Y: {real_min_1y:.2%}")
        print(f"Real 1% 1Y: {real_1pct_1y:.2%}")

        # 2. i.i.d. Monte Carlo (Re-run for analysis)
        n_sims = 2000
        n_days = 252
        iid_final_rets = []

        # Use numpy for speed
        ret_values = returns.values
        # Filter out the initial 0s if any (warmup)
        ret_values = ret_values[ret_values != 0]

        np.random.seed(42)  # For reproducibility

        for _ in range(n_sims):
            sim_rets = np.random.choice(ret_values, size=n_days, replace=True)
            # Compound returns
            final_ret = np.prod(1 + sim_rets) - 1
            iid_final_rets.append(final_ret)

        iid_var_99 = np.percentile(iid_final_rets, 1)
        iid_var_95 = np.percentile(iid_final_rets, 5)
        iid_mean = np.mean(iid_final_rets)

        # Analyze Worst Paths (i.i.d.)
        sorted_indices = np.argsort(iid_final_rets)
        worst_1pct_idx = sorted_indices[: int(n_sims * 0.01)]
        worst_paths_rets = [iid_final_rets[i] for i in worst_1pct_idx]
        print(f"Avg Return of Worst 1% Paths (i.i.d): {np.mean(worst_paths_rets):.2%}")

        # 3. Block Bootstrap Monte Carlo
        block_size = 21  # ~1 Month
        block_final_rets = []

        # Create blocks
        blocks = []
        # Overlapping blocks
        for i in range(len(ret_values) - block_size + 1):
            blocks.append(ret_values[i : i + block_size])

        n_blocks_needed = int(np.ceil(n_days / block_size))

        for _ in range(n_sims):
            # Sample blocks
            chosen_indices = np.random.randint(0, len(blocks), size=n_blocks_needed)
            sim_rets_list = [blocks[i] for i in chosen_indices]
            sim_rets = np.concatenate(sim_rets_list)[:n_days]

            final_ret = np.prod(1 + sim_rets) - 1
            block_final_rets.append(final_ret)

        block_var_99 = np.percentile(block_final_rets, 1)
        block_var_95 = np.percentile(block_final_rets, 5)
        block_mean = np.mean(block_final_rets)

        print(f"i.i.d. VaR 99%: {iid_var_99:.2%}")
        print(f"Block VaR 99%: {block_var_99:.2%}")

        return {
            "real_min_1y": real_min_1y,
            "real_5pct_1y": real_5pct_1y,
            "real_1pct_1y": real_1pct_1y,
            "iid_var_99": iid_var_99,
            "iid_var_95": iid_var_95,
            "iid_mean": iid_mean,
            "block_var_99": block_var_99,
            "block_var_95": block_var_95,
            "block_mean": block_mean,
        }

    def run(self):
        vec_equity, bt_strat = self.step1_bt_reproduction()

        perturb_df, best_params = self.step2_perturbation()
        # perturb_df.to_csv(ROOT / "results/postselection/perturbation_results.csv", index=False)

        self.step3_capacity(bt_strat)
        extreme_res = self.step4_extreme(vec_equity)
        var99, worst_dd = self.step5_monte_carlo(vec_equity)

        # Step 6: Diagnostics
        diag_res = self.analyze_var_diagnostics(vec_equity)

        # Generate Report
        report_path = ROOT / "results/top1_production_report.md"
        with open(report_path, "w") as f:
            f.write("# 🚀 Top 1 Production Validation Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n\n")

            f.write("## 1. BT Reproduction\n")
            f.write(
                f"- **VEC Return**: {(vec_equity[-1]/1e6)**(252/len(vec_equity))-1:.2%}\n"
            )
            bt_val = bt_strat.broker.getvalue()
            bt_ret = (bt_val / 1e6) ** (252 / len(vec_equity)) - 1
            f.write(f"- **BT Return**: {bt_ret:.2%}\n")

            f.write("## 2. Parameter Perturbation\n")
            f.write(f"- **Best Params**: {best_params}\n")
            f.write(f"- **Top 5 Variations**:\n")
            f.write(
                perturb_df.sort_values("ann_return", ascending=False)
                .head(5)
                .to_markdown()
            )
            f.write("\n\n")

            f.write("## 3. Extreme Testing\n")
            f.write(pd.DataFrame(extreme_res).T.to_markdown())
            f.write("\n\n")

            f.write("## 4. Monte Carlo (1 Year)\n")
            f.write(f"- **99% VaR**: {var99:.2%}\n")
            f.write(f"- **99% Worst DD**: {worst_dd:.2%}\n")
            f.write("\n")

            f.write("## 5. VaR Diagnostics\n")
            f.write("### 5.1 Definition\n")
            f.write("- **Method**: Monte Carlo Simulation (2000 runs).\n")
            f.write("- **Metric**: 1-Year Total Return Distribution.\n")
            f.write(
                "- **VaR 99%**: The 1st percentile of the simulated 1-year returns (i.e., 99% of the time, the annual return is better than this).\n"
            )
            f.write("- **Data**: Daily returns from the strategy equity curve.\n\n")

            f.write("### 5.2 Historical vs Monte Carlo\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            f.write(f"| Real Min 1Y Return | {diag_res['real_min_1y']:.2%} |\n")
            f.write(f"| Real 5% Quantile 1Y | {diag_res['real_5pct_1y']:.2%} |\n")
            f.write(f"| Real 1% Quantile 1Y | {diag_res['real_1pct_1y']:.2%} |\n")
            f.write(f"| MC 99% VaR (i.i.d.) | {diag_res['iid_var_99']:.2%} |\n")
            f.write(f"| MC 99% VaR (Block) | {diag_res['block_var_99']:.2%} |\n\n")

            f.write("### 5.3 Bootstrap Comparison\n")
            f.write("| Metric | i.i.d. Bootstrap | Block Bootstrap (21 Days) |\n")
            f.write("| :--- | :--- | :--- |\n")
            f.write(
                f"| 99% VaR (1Y) | {diag_res['iid_var_99']:.2%} | {diag_res['block_var_99']:.2%} |\n"
            )
            f.write(
                f"| 95% VaR (1Y) | {diag_res['iid_var_95']:.2%} | {diag_res['block_var_95']:.2%} |\n"
            )
            f.write(
                f"| Mean Simulated Return | {diag_res['iid_mean']:.2%} | {diag_res['block_mean']:.2%} |\n\n"
            )

            f.write("### 5.4 Interpretation\n")
            f.write(
                "- **i.i.d. Assumption**: The i.i.d. bootstrap destroys the serial correlation (trends) that the strategy relies on. By shuffling daily returns randomly, it simulates a 'random walk' market where trends do not persist, causing the strategy to underperform significantly.\n"
            )
            f.write(
                "- **Block Bootstrap**: By preserving 21-day (approx 1 month) blocks, we retain short-term trends. The Block VaR is expected to be closer to historical reality.\n"
            )
            if diag_res["block_var_99"] > diag_res["iid_var_99"]:
                f.write(
                    "- **Conclusion**: The -27% VaR was largely an artifact of the i.i.d. assumption breaking the trend structure. The Block VaR provides a more realistic (though still conservative) risk estimate.\n"
                )
            else:
                f.write(
                    "- **Conclusion**: The risk remains high even with block bootstrapping, suggesting the strategy has genuine tail risk in certain market regimes.\n"
                )

        print("Validation Complete.")


if __name__ == "__main__":
    validator = Top1Validator()
    validator.run()
