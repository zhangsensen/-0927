#!/usr/bin/env python3
"""
提取Top2000组合的逐日净值序列

从已有的回测CSV中读取组合列表，重新运行回测以获取每个组合的逐日净值序列。
输出: results/combo_daily_nav_top2000.pkl (DataFrame: index=date, columns=combo_id)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def _ensure_stable_paths():
    """将稳定仓加入 sys.path"""
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    stable_root = repo_root / "etf_rotation_optimized"
    stable_rb = stable_root / "real_backtest"
    for p in (stable_root, stable_rb):
        sp = str(p.resolve())
        if sp not in sys.path:
            sys.path.insert(0, sp)


_ensure_stable_paths()

from core.cross_section_processor import CrossSectionProcessor  # type: ignore
from core.data_loader import DataLoader  # type: ignore
from core.precise_factor_library_v2 import PreciseFactorLibrary  # type: ignore
from real_backtest.run_production_backtest import backtest_no_lookahead  # type: ignore


def load_config() -> dict:
    """加载配置文件（仅使用experiments项目的配置）"""
    here = Path(__file__).resolve()
    cfg_path = here.parent.parent / "configs" / "combo_wfo_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def parse_combo_string(combo_str: str) -> List[str]:
    """解析组合字符串，例如 'ADX_14D + CMF_20D + RSI_14' -> ['ADX_14D', 'CMF_20D', 'RSI_14']"""
    return [f.strip() for f in combo_str.split("+")]


def main():
    print("=" * 100)
    print("提取Top2000组合逐日净值序列")
    print("=" * 100)
    print()

    # 1. 加载配置和数据
    print("加载配置...")
    cfg = load_config()
    
    print("加载数据...")
    loader = DataLoader(
        data_dir=cfg["data"].get("data_dir"),
        cache_dir=cfg["data"].get("cache_dir")
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=cfg["data"]["symbols"],
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        use_cache=True,
    )
    dates = ohlcv["close"].index
    print(f"✓ 数据: {len(dates)}天 × {len(ohlcv['close'].columns)}只ETF")
    
    print("计算因子...")
    factor_lib = PreciseFactorLibrary()
    factors_df = factor_lib.compute_all_factors(prices=ohlcv)
    factors_dict = {name: factors_df[name] for name in factor_lib.list_factors()}
    print(f"✓ 因子: {len(factors_dict)}个")
    
    print("横截面标准化...")
    processor = CrossSectionProcessor(
        lower_percentile=cfg["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=cfg["cross_section"]["winsorize_upper"] * 100,
        verbose=False,
    )
    standardized = processor.process_all_factors(factors_dict)
    factor_names = sorted(standardized.keys())
    factors_data = np.stack([standardized[n].values for n in factor_names], axis=-1)
    returns = ohlcv["close"].pct_change(fill_method=None).values
    etf_names = list(ohlcv["close"].columns)
    print("✓ 标准化完成")
    print()

    # 2. 读取Top2000组合列表
    print("读取Top2000组合列表...")
    here = Path(__file__).resolve()
    results_dir = here.parent.parent / "results_combo_wfo"
    csv_files = sorted(results_dir.glob("*/top2000_profit_backtest_slip0bps_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到Top2000回测CSV文件，路径: {results_dir}")
    
    csv_path = csv_files[-1]  # 最新的
    print(f"✓ 读取: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 可选: 限制组合数量用于测试
    import os
    test_limit = os.environ.get("TEST_LIMIT", "").strip()
    if test_limit and test_limit.isdigit():
        df = df.head(int(test_limit))
        print(f"⚠ 测试模式: 仅处理前 {len(df)} 个组合")
    
    print(f"✓ 组合数: {len(df)}")
    print()

    # 3. 提取每个组合的逐日净值
    print("开始提取逐日净值...")
    commission_rate = cfg["backtest"].get("commission_rate", 0.000005)
    
    nav_dict = {}
    failed_combos = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="回测进度"):
        combo_str = row["combo"]
        combo_id = f"combo_{idx}"
        freq = int(row["test_freq"])
        
        try:
            factor_list = parse_combo_string(combo_str)
            
            # 找到因子索引并提取对应的因子数据
            factor_indices = []
            for fname in factor_list:
                if fname in factor_names:
                    factor_indices.append(factor_names.index(fname))
                else:
                    raise ValueError(f"因子 {fname} 不在因子列表中")
            
            # 提取选定因子的数据切片 (T, N, F_selected)
            combo_factors_data = factors_data[:, :, factor_indices]
            
            # 运行回测
            result = backtest_no_lookahead(
                factors_data=combo_factors_data,
                returns=returns,
                etf_names=etf_names,
                rebalance_freq=freq,
                lookback_window=252,
                position_size=5,
                initial_capital=1000000.0,
                commission_rate=commission_rate,
                commission_min=0.0,
            )
            
            # 提取净值序列
            nav = result.get("nav", np.array([]))
            if len(nav) > 0:
                nav_dict[combo_id] = nav
            else:
                failed_combos.append((combo_id, combo_str, "空净值序列"))
                
        except Exception as e:
            failed_combos.append((combo_id, combo_str, str(e)))
            continue
    
    print()
    print(f"✓ 成功提取: {len(nav_dict)}/{len(df)} 个组合")
    if failed_combos:
        print(f"✗ 失败: {len(failed_combos)} 个组合")
        for combo_id, combo_str, error in failed_combos[:5]:
            print(f"  - {combo_id}: {combo_str[:50]}... | {error}")
    print()

    # 4. 构建DataFrame并保存
    print("构建DataFrame...")
    # 确保所有净值序列长度一致
    # 注意: nav序列长度 = T - start_idx + 1，其中start_idx通常是lookback_window
    # 所以nav长度会比dates短
    nav_data = {}
    nav_lengths = [len(nav) for nav in nav_dict.values()]
    if nav_lengths:
        max_nav_len = max(nav_lengths)
        min_nav_len = min(nav_lengths)
        print(f"  - 净值序列长度范围: {min_nav_len} ~ {max_nav_len}")
        
        # 使用最常见的长度作为标准长度
        from collections import Counter
        common_len = Counter(nav_lengths).most_common(1)[0][0]
        print(f"  - 使用标准长度: {common_len}")
        
        for combo_id, nav in nav_dict.items():
            if len(nav) == common_len:
                nav_data[combo_id] = nav
            elif len(nav) > common_len:
                # 截断
                nav_data[combo_id] = nav[:common_len]
            else:
                # 后向填充（用最后一个值填充）
                aligned_nav = np.full(common_len, nav[-1] if len(nav) > 0 else np.nan)
                aligned_nav[:len(nav)] = nav
                nav_data[combo_id] = aligned_nav
        
        # 使用对应的日期索引（从lookback_window开始）
        nav_dates = dates[len(dates) - common_len:]
    else:
        nav_dates = dates
    
    nav_df = pd.DataFrame(nav_data, index=nav_dates)
    print(f"✓ DataFrame shape: {nav_df.shape}")
    print()

    # 5. 保存结果
    output_dir = here.parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "combo_daily_nav_top2000.pkl"
    
    nav_df.to_pickle(output_path)
    print(f"✓ 已保存: {output_path}")
    print(f"  - 行数(日期): {len(nav_df)}")
    print(f"  - 列数(组合): {len(nav_df.columns)}")
    print()
    
    # 6. 基本统计
    print("基本统计:")
    print(f"  - 首日净值均值: {nav_df.iloc[0].mean():.2f}")
    print(f"  - 末日净值均值: {nav_df.iloc[-1].mean():.2f}")
    print(f"  - 平均总收益率: {(nav_df.iloc[-1] / nav_df.iloc[0] - 1).mean():.2%}")
    print()
    
    print("=" * 100)
    print("✅ 完成！")
    print("=" * 100)


if __name__ == "__main__":
    main()

