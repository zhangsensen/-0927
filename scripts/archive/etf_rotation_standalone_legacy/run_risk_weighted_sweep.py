#!/usr/bin/env python3
"""
风险加权策略扫描 | Risk-Weighted Strategy Sweep

对全量 WFO 策略进行风险评估，输出三张榜单：
1. 稳健型 (Conservative): 最大回撤 < 15%，优先保本
2. 平衡型 (Balanced): 综合评分最高，收益与风险平衡
3. 进取型 (Aggressive): 最大回撤 < 30%，追求高收益

评分公式:
Score = 40% × Calmar + 30% × Sharpe + 30% × WinRate

作者: Linus
日期: 2025-11-29
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.core.backtester_vectorized import run_vec_backtest_with_risk


def compute_composite_score(row, weights=None):
    """
    计算综合评分
    
    Args:
        row: DataFrame row with metrics
        weights: dict with weights for each metric
        
    默认权重:
        - calmar: 40%  (卡玛比率 = 年化收益/最大回撤)
        - sharpe: 30%  (夏普比率)
        - win_rate: 30% (胜率)
    """
    if weights is None:
        weights = {"calmar": 0.4, "sharpe": 0.3, "win_rate": 0.3}
    
    # Calmar Ratio
    calmar = row["annual_return"] / max(row["max_dd"], 0.01)
    
    # Normalize metrics to [0, 1] scale (approximately)
    # Calmar: good range is 0.5-3.0, clip and scale
    calmar_norm = np.clip(calmar, 0, 5) / 5.0
    
    # Sharpe: good range is 0.5-2.5, clip and scale
    sharpe_norm = np.clip(row["sharpe"], 0, 3) / 3.0
    
    # Win Rate: already in [0, 1]
    wr_norm = row["win_rate"]
    
    score = (
        weights["calmar"] * calmar_norm +
        weights["sharpe"] * sharpe_norm +
        weights["win_rate"] * wr_norm
    )
    
    return score, calmar


def main():
    print("=" * 80)
    print("📊 风险加权策略扫描 | Risk-Weighted Strategy Sweep")
    print("=" * 80)

    # 1. Load Configuration
    config_path = Path(__file__).parent / "configs/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 2. Load WFO Combos
    wfo_results_dir = Path("results")
    wfo_files = sorted(wfo_results_dir.glob("unified_wfo_*/all_combos.parquet"))
    if not wfo_files:
        print("❌ No WFO results found in results/unified_wfo_*")
        return
    
    wfo_file = wfo_files[-1]
    print(f"📂 Loading WFO Combos from: {wfo_file}")
    df_combos = pd.read_parquet(wfo_file)
    print(f"   Loaded {len(df_combos)} combos.")

    # 3. Load Data
    print("\n[Step 1] Loading Data...")
    risk_off_asset = "518880"
    symbols = config["data"]["symbols"]
    if risk_off_asset not in symbols:
        symbols.append(risk_off_asset)
        
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=symbols,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    # Extract Risk-Off Asset
    if risk_off_asset in ohlcv["close"].columns:
        risk_off_prices = ohlcv["close"][risk_off_asset].ffill().bfill().values
        rotation_symbols = [s for s in ohlcv["close"].columns if s != risk_off_asset]
        ohlcv_rotation = {col: df[rotation_symbols] for col, df in ohlcv.items()}
    else:
        risk_off_prices = None
        ohlcv_rotation = ohlcv

    # 4. Compute Factors
    print("\n[Step 2] Computing Factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv_rotation)
    
    # 5. Process Factors
    print("\n[Step 3] Processing Factors...")
    processor = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=config["cross_section"]["winsorize_upper"] * 100,
        verbose=False
    )
    std_factors = processor.process_all_factors({
        name: raw_factors_df[name] 
        for name in raw_factors_df.columns.get_level_values(0).unique()
    })

    factor_names = sorted(std_factors.keys())
    factor_map = {name: i for i, name in enumerate(factor_names)}
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = ohlcv_rotation["close"][std_factors[factor_names[0]].columns].ffill().bfill().values
    open_prices = ohlcv_rotation["open"][std_factors[factor_names[0]].columns].ffill().bfill().values
    
    # 6. Compute Timing
    print("\n[Step 4] Computing Timing Signal...")
    timing_module = LightTimingModule(
        extreme_threshold=config["backtest"]["timing"]["extreme_threshold"],
        extreme_position=config["backtest"]["timing"]["extreme_position"]
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr = shift_timing_signal(timing_series.reindex(std_factors[factor_names[0]].index).fillna(1.0).values)

    # 7. Pre-filter combos
    print("\n[Step 5] Preparing combos...")
    combo_indices_list = []
    valid_combos = []
    
    for idx, row in df_combos.iterrows():
        combo_str = row["combo"]
        parts = [p.strip() for p in combo_str.split("+")]
        try:
            indices = [factor_map[p] for p in parts]
            combo_indices_list.append(indices)
            valid_combos.append(combo_str)
        except KeyError:
            continue
            
    print(f"   Valid combos: {len(valid_combos)} / {len(df_combos)}")
    
    # 8. Define sweep parameters
    # 频率范围: 5-30天 (剔除过于激进的1-4天)
    freq_range = list(range(5, 31))
    
    # 硬过滤条件
    MIN_TRADES = 50  # 最少交易次数
    MAX_DD_HARD = 0.35  # 硬性最大回撤上限 (35%)
    
    print(f"\n[Step 6] Running sweep ({len(valid_combos)} combos x {len(freq_range)} freqs)...")
    
    results = []
    start_time = time.time()
    
    total_tasks = len(valid_combos) * len(freq_range)
    pbar = tqdm(total=total_tasks, unit="bt")
    
    for freq in freq_range:
        for i, combo_indices in enumerate(combo_indices_list):
            ret, wr, pf, trades, mdd, sharpe, ann_ret, vol = run_vec_backtest_with_risk(
                factors_3d=factors_3d,
                close_prices=close_prices,
                open_prices=open_prices,
                timing_arr=timing_arr,
                factor_indices=combo_indices,
                risk_off_prices=risk_off_prices,
                freq=freq,
                pos_size=config["backtest"]["position_size"],
                initial_capital=config["backtest"].get("initial_capital", 1_000_000.0),
                commission_rate=config["backtest"].get("commission_rate", 0.0002),
                lookback=config["backtest"].get("lookback_window", 252)
            )
            
            results.append({
                "combo": valid_combos[i],
                "freq": freq,
                "total_return": ret,
                "annual_return": ann_ret,
                "win_rate": wr,
                "profit_factor": pf,
                "trades": trades,
                "max_dd": mdd,
                "sharpe": sharpe,
                "volatility": vol,
            })
            pbar.update(1)
            
    pbar.close()
    elapsed = time.time() - start_time
    print(f"\n✅ Sweep completed in {elapsed:.2f}s ({total_tasks/elapsed:.0f} bt/s)")

    # 9. Process Results
    df_res = pd.DataFrame(results)
    
    # 硬过滤
    print(f"\n[Step 7] Applying hard filters...")
    print(f"   Before: {len(df_res)} rows")
    
    df_filtered = df_res[
        (df_res["trades"] >= MIN_TRADES) &
        (df_res["max_dd"] <= MAX_DD_HARD) &
        (df_res["max_dd"] > 0)  # 排除回撤为0的异常情况
    ].copy()
    
    print(f"   After:  {len(df_filtered)} rows (filtered out {len(df_res) - len(df_filtered)})")
    
    # 计算综合评分和卡玛比率
    scores = []
    calmars = []
    for _, row in df_filtered.iterrows():
        score, calmar = compute_composite_score(row)
        scores.append(score)
        calmars.append(calmar)
    
    df_filtered["score"] = scores
    df_filtered["calmar"] = calmars
    
    # 10. Generate Three Rankings
    print("\n" + "=" * 80)
    print("📊 策略排行榜 | Strategy Rankings")
    print("=" * 80)
    
    # === 稳健型 (Conservative) ===
    # 最大回撤 < 15%, 按 Calmar 排序
    df_conservative = df_filtered[df_filtered["max_dd"] < 0.15].copy()
    df_conservative = df_conservative.sort_values("calmar", ascending=False)
    
    print("\n🛡️ 【稳健型 Conservative】 (MaxDD < 15%)")
    print("-" * 60)
    if len(df_conservative) > 0:
        top_conservative = df_conservative.head(10)
        for i, (_, row) in enumerate(top_conservative.iterrows(), 1):
            print(f"{i:2}. {row['combo']:<50} | Freq={row['freq']:2} | "
                  f"Ret={row['total_return']*100:6.1f}% | MDD={row['max_dd']*100:5.1f}% | "
                  f"Calmar={row['calmar']:5.2f} | WR={row['win_rate']*100:4.1f}%")
    else:
        print("   (无符合条件的策略)")
    
    # === 平衡型 (Balanced) ===
    # 最大回撤 < 25%, 按综合评分排序
    df_balanced = df_filtered[df_filtered["max_dd"] < 0.25].copy()
    df_balanced = df_balanced.sort_values("score", ascending=False)
    
    print("\n⚖️ 【平衡型 Balanced】 (MaxDD < 25%, 综合评分)")
    print("-" * 60)
    if len(df_balanced) > 0:
        top_balanced = df_balanced.head(10)
        for i, (_, row) in enumerate(top_balanced.iterrows(), 1):
            print(f"{i:2}. {row['combo']:<50} | Freq={row['freq']:2} | "
                  f"Ret={row['total_return']*100:6.1f}% | MDD={row['max_dd']*100:5.1f}% | "
                  f"Score={row['score']:5.3f} | Sharpe={row['sharpe']:5.2f}")
    else:
        print("   (无符合条件的策略)")
    
    # === 进取型 (Aggressive) ===
    # 最大回撤 < 30%, 按收益率排序
    df_aggressive = df_filtered[df_filtered["max_dd"] < 0.30].copy()
    df_aggressive = df_aggressive.sort_values("total_return", ascending=False)
    
    print("\n🚀 【进取型 Aggressive】 (MaxDD < 30%, 追求高收益)")
    print("-" * 60)
    if len(df_aggressive) > 0:
        top_aggressive = df_aggressive.head(10)
        for i, (_, row) in enumerate(top_aggressive.iterrows(), 1):
            print(f"{i:2}. {row['combo']:<50} | Freq={row['freq']:2} | "
                  f"Ret={row['total_return']*100:6.1f}% | MDD={row['max_dd']*100:5.1f}% | "
                  f"Sharpe={row['sharpe']:5.2f} | PF={row['profit_factor']:5.2f}")
    else:
        print("   (无符合条件的策略)")
    
    # 11. Save Results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    parquet_path = output_dir / f"risk_sweep_full_{timestamp}.parquet"
    df_filtered.to_parquet(parquet_path)
    print(f"\n💾 Full results saved to {parquet_path}")
    
    # Save rankings to CSV
    rankings = {
        "conservative": df_conservative.head(50) if len(df_conservative) > 0 else pd.DataFrame(),
        "balanced": df_balanced.head(50) if len(df_balanced) > 0 else pd.DataFrame(),
        "aggressive": df_aggressive.head(50) if len(df_aggressive) > 0 else pd.DataFrame(),
    }
    
    for name, df in rankings.items():
        if len(df) > 0:
            csv_path = output_dir / f"ranking_{name}_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"💾 {name.capitalize()} ranking saved to {csv_path}")
    
    # 12. Summary Statistics
    print("\n" + "=" * 80)
    print("📈 扫描统计 | Sweep Statistics")
    print("=" * 80)
    print(f"总扫描数:      {total_tasks:,}")
    print(f"有效策略数:    {len(df_filtered):,}")
    print(f"稳健型候选:    {len(df_conservative):,}")
    print(f"平衡型候选:    {len(df_balanced):,}")
    print(f"进取型候选:    {len(df_aggressive):,}")
    
    if len(df_filtered) > 0:
        print(f"\n最佳卡玛比率:  {df_filtered['calmar'].max():.2f}")
        print(f"最佳夏普比率:  {df_filtered['sharpe'].max():.2f}")
        print(f"最高总收益率:  {df_filtered['total_return'].max()*100:.1f}%")
        print(f"最低最大回撤:  {df_filtered['max_dd'].min()*100:.1f}%")


if __name__ == "__main__":
    main()
