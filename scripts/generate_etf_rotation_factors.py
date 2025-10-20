#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成真正的ETF轮动因子面板 - 快速版本

只计算长周期动量和关键技术指标
"""
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("生成ETF轮动因子面板...")
    print("="*80)
    
    # 加载价格数据
    prices = []
    for f in sorted(glob.glob('raw/ETF/daily/*.parquet')):
        df = pd.read_parquet(f)
        symbol = f.split('/')[-1].split('_')[0]
        df['symbol'] = symbol
        df['date'] = pd.to_datetime(df['trade_date'])
        # 统一列名：vol -> volume
        if 'vol' in df.columns and 'volume' not in df.columns:
            df['volume'] = df['vol']
        prices.append(df[['date', 'close', 'high', 'low', 'volume', 'symbol']])
    
    price_df = pd.concat(prices, ignore_index=True)
    
    # 按symbol分组计算
    factors_list = []
    symbols = price_df['symbol'].unique()
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] 计算 {symbol}")
        
        symbol_data = price_df[price_df['symbol'] == symbol].sort_values('date')
        close = symbol_data['close'].values
        high = symbol_data['high'].values
        low = symbol_data['low'].values
        volume = symbol_data['volume'].values
        dates = symbol_data['date'].values
        
        factors = pd.DataFrame(index=symbol_data.index)
        factors['date'] = dates
        factors['symbol'] = symbol
        
        # === 长周期动量（ETF轮动核心） ===
        for period in [20, 63, 126, 252]:  # 1月、3月、6月、12月
            # T+1安全：shift(period+1)
            mom = pd.Series(close) / pd.Series(close).shift(period+1) - 1
            factors[f'MOMENTUM_{period}D'] = mom.values
        
        # === 波动率 ===
        for window in [20, 60, 120]:
            ret = pd.Series(close).pct_change()
            vol = ret.rolling(window).std() * np.sqrt(252)
            factors[f'VOLATILITY_{window}D'] = vol.values
        
        # === 最大回撤 ===
        for window in [63, 126]:
            # 修复：使用min_periods=1确保前期数据也能计算
            s = pd.Series(close)
            rolling_max = s.rolling(window, min_periods=1).max()
            dd = (s - rolling_max) / rolling_max
            factors[f'DRAWDOWN_{window}D'] = dd.values
        
        # === 动量加速度 ===
        mom_short = pd.Series(close) / pd.Series(close).shift(64) - 1
        mom_long = pd.Series(close) / pd.Series(close).shift(253) - 1
        factors['MOM_ACCEL'] = mom_short - mom_long
        
        # === RSI ===
        for window in [6, 14, 24]:
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            factors[f'RSI_{window}'] = rsi.values
        
        # === 价格位置 ===
        for window in [20, 60, 120]:
            roll_high = pd.Series(high).rolling(window).max()
            roll_low = pd.Series(low).rolling(window).min()
            pos = (close - roll_low) / (roll_high - roll_low + 1e-10)
            factors[f'PRICE_POSITION_{window}D'] = pos.values
        
        # === 成交量比率 ===
        for window in [20, 60]:
            vol_ma = pd.Series(volume).rolling(window).mean()
            ratio = volume / (vol_ma + 1e-10)
            factors[f'VOLUME_RATIO_{window}D'] = ratio.values
        
        factors_list.append(factors)
    
    # 合并
    panel = pd.concat(factors_list, ignore_index=True)
    panel = panel.set_index(['symbol', 'date']).sort_index()
    
    # 统计
    print("\n面板统计:")
    print(f"  ETF数量: {panel.index.get_level_values('symbol').nunique()}")
    print(f"  因子数量: {len(panel.columns)}")
    print(f"  数据点数: {len(panel)}")
    print(f"  覆盖率: {panel.notna().mean().mean():.2%}")
    
    # 保存
    output_dir = Path('factor_output/etf_rotation_true')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    panel_file = output_dir / 'panel_etf_rotation_20200102_20251014.parquet'
    panel.to_parquet(panel_file)
    print(f"\n✅ 面板已保存: {panel_file}")
    
    # 保存元数据
    meta = {
        'etf_count': panel.index.get_level_values('symbol').nunique(),
        'factor_count': len(panel.columns),
        'data_points': len(panel),
        'coverage_rate': float(panel.notna().mean().mean()),
        'factors': panel.columns.tolist(),
        'description': '真正的ETF轮动因子：长周期动量 + 风险指标'
    }
    
    import json
    with open(output_dir / 'panel_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"✅ 元数据已保存: {output_dir / 'panel_meta.json'}")
    print("\n因子列表:")
    for i, col in enumerate(panel.columns, 1):
        print(f"  {i:2}. {col}")

if __name__ == '__main__':
    main()

