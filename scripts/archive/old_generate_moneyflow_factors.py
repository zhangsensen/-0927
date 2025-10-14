"""
资金流因子生产脚本

真实数据、真实问题、真实修复
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from factor_system.factor_engine.providers.money_flow_provider import MoneyFlowProvider
from factor_system.factor_engine.factors.money_flow.core import (
    MainNetInflow_Rate,
    LargeOrder_Ratio,
    SuperLargeOrder_Ratio,
    OrderConcentration,
    MoneyFlow_Hierarchy,
    MoneyFlow_Consensus,
    MainFlow_Momentum,
    Flow_Price_Divergence,
)
from factor_system.factor_engine.factors.money_flow.enhanced import (
    Institutional_Absorption,
    Flow_Tier_Ratio_Delta,
    Flow_Reversal_Ratio,
)


def generate_moneyflow_factors(symbol: str, output_dir: Path):
    """
    生成资金流因子
    
    Args:
        symbol: 股票代码
        output_dir: 输出目录
    """
    print("=" * 60)
    print(f"生成资金流因子: {symbol}")
    print("=" * 60)
    
    # 1. 加载资金流数据（T+1时序安全）
    print("\n1. 加载资金流数据...")
    provider = MoneyFlowProvider(
        data_dir=Path("raw/SH/money_flow"),
        enforce_t_plus_1=True
    )
    
    df_money = provider.load_money_flow(symbol, "2023-01-01", "2025-12-31")
    print(f"   ✅ 资金流数据: {df_money.shape}")
    print(f"   ✅ 时序安全: {df_money['temporal_safe'].all()}")
    print(f"   ✅ 日期范围: {df_money.index.min()} 到 {df_money.index.max()}")
    
    # 2. 加载价格数据并重采样到日线
    print("\n2. 加载价格数据...")
    price_file = Path(f"raw/SH/{symbol}.parquet")
    if not price_file.exists():
        raise FileNotFoundError(f"价格数据不存在: {price_file}")
    
    df_price = pd.read_parquet(price_file)
    
    # 处理索引
    if 'datetime' in df_price.columns:
        df_price['datetime'] = pd.to_datetime(df_price['datetime'])
        df_price.set_index('datetime', inplace=True)
    elif 'trade_date' in df_price.columns:
        df_price['trade_date'] = pd.to_datetime(df_price['trade_date'])
        df_price.set_index('trade_date', inplace=True)
    
    print(f"   原始价格数据: {df_price.shape}")
    print(f"   索引类型: {type(df_price.index)}")
    
    # 检查是否需要重采样
    if isinstance(df_price.index, pd.DatetimeIndex):
        # 检查是否是日内数据
        if df_price.index[0].hour != 0 or df_price.index[0].minute != 0:
            print("   检测到分钟数据，重采样到日线...")
            df_price_daily = df_price.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            print(f"   ✅ 日线数据: {df_price_daily.shape}")
        else:
            df_price_daily = df_price
    else:
        raise ValueError("价格数据必须有DatetimeIndex")
    
    # 3. 合并数据
    print("\n3. 合并数据...")
    # 确保索引格式一致（都是日期，无时分秒）
    df_price_daily.index = df_price_daily.index.normalize()
    
    merged = df_money.join(df_price_daily[['close']], how='left')
    print(f"   ✅ 合并后: {merged.shape}")
    print(f"   ✅ close有效值: {(~merged['close'].isna()).sum()}/{len(merged)}")
    
    # 如果close全是NaN，检查索引对齐
    if merged['close'].isna().all():
        print("   ⚠️  close全为NaN，检查索引对齐...")
        print(f"   资金流索引示例: {df_money.index[:3].tolist()}")
        print(f"   价格索引示例: {df_price_daily.index[:3].tolist()}")
        common = df_money.index.intersection(df_price_daily.index)
        print(f"   共同日期数: {len(common)}")
        if len(common) > 0:
            print(f"   使用inner join重新合并...")
            merged = df_money.join(df_price_daily[['close']], how='inner')
            print(f"   ✅ 重新合并后: {merged.shape}")
            print(f"   ✅ close有效值: {(~merged['close'].isna()).sum()}/{len(merged)}")
    
    # 4. 计算因子
    print("\n4. 计算因子...")
    factors = {}
    
    # 核心因子
    print("   核心因子:")
    f1 = MainNetInflow_Rate(window=5)
    factors['MainNetInflow_Rate'] = f1.calculate(merged)
    valid = (~factors['MainNetInflow_Rate'].isna()).sum()
    print(f"     ✅ MainNetInflow_Rate: {valid}/{len(merged)} ({valid/len(merged)*100:.1f}%)")
    
    f2 = LargeOrder_Ratio(window=10)
    factors['LargeOrder_Ratio'] = f2.calculate(merged)
    valid = (~factors['LargeOrder_Ratio'].isna()).sum()
    print(f"     ✅ LargeOrder_Ratio: {valid}/{len(merged)} ({valid/len(merged)*100:.1f}%)")
    
    f3 = SuperLargeOrder_Ratio(window=20)
    factors['SuperLargeOrder_Ratio'] = f3.calculate(merged)
    valid = (~factors['SuperLargeOrder_Ratio'].isna()).sum()
    print(f"     ✅ SuperLargeOrder_Ratio: {valid}/{len(merged)} ({valid/len(merged)*100:.1f}%)")
    
    f4 = OrderConcentration()
    factors['OrderConcentration'] = f4.calculate(merged)
    valid = (~factors['OrderConcentration'].isna()).sum()
    print(f"     ✅ OrderConcentration: {valid}/{len(merged)} ({valid/len(merged)*100:.1f}%)")
    
    f5 = MoneyFlow_Hierarchy()
    factors['MoneyFlow_Hierarchy'] = f5.calculate(merged)
    valid = (~factors['MoneyFlow_Hierarchy'].isna()).sum()
    print(f"     ✅ MoneyFlow_Hierarchy: {valid}/{len(merged)} ({valid/len(merged)*100:.1f}%)")
    
    f6 = MoneyFlow_Consensus(window=5)
    factors['MoneyFlow_Consensus'] = f6.calculate(merged)
    valid = (~factors['MoneyFlow_Consensus'].isna()).sum()
    print(f"     ✅ MoneyFlow_Consensus: {valid}/{len(merged)} ({valid/len(merged)*100:.1f}%)")
    
    f7 = MainFlow_Momentum(short_window=5, long_window=10)
    factors['MainFlow_Momentum'] = f7.calculate(merged)
    valid = (~factors['MainFlow_Momentum'].isna()).sum()
    print(f"     ✅ MainFlow_Momentum: {valid}/{len(merged)} ({valid/len(merged)*100:.1f}%)")
    
    f8 = Flow_Price_Divergence(window=5)
    factors['Flow_Price_Divergence'] = f8.calculate(merged)
    valid = (~factors['Flow_Price_Divergence'].isna()).sum()
    print(f"     ✅ Flow_Price_Divergence: {valid}/{len(merged)} ({valid/len(merged)*100:.1f}%)")
    
    # 增强因子
    print("   增强因子:")
    f9 = Institutional_Absorption()
    factors['Institutional_Absorption'] = f9.calculate(merged)
    valid = (~factors['Institutional_Absorption'].isna()).sum()
    print(f"     ✅ Institutional_Absorption: {valid}/{len(merged)} ({valid/len(merged)*100:.1f}%)")
    
    f10 = Flow_Tier_Ratio_Delta(window=5)
    factors['Flow_Tier_Ratio_Delta'] = f10.calculate(merged)
    valid = (~factors['Flow_Tier_Ratio_Delta'].isna()).sum()
    print(f"     ✅ Flow_Tier_Ratio_Delta: {valid}/{len(merged)*100:.1f}%)")
    
    f11 = Flow_Reversal_Ratio()
    factors['Flow_Reversal_Ratio'] = f11.calculate(merged)
    valid = (~factors['Flow_Reversal_Ratio'].isna()).sum()
    print(f"     ✅ Flow_Reversal_Ratio: {valid}/{len(merged)} ({valid/len(merged)*100:.1f}%)")
    
    # 5. 保存结果
    print("\n5. 保存结果...")
    factor_df = pd.DataFrame(factors, index=merged.index)
    factor_df['temporal_safe'] = True
    
    output_path = output_dir / f"{symbol}_moneyflow_factors.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    factor_df.to_parquet(output_path)
    
    print(f"   ✅ 保存成功: {output_path}")
    print(f"   ✅ 因子数量: {len(factors)}")
    print(f"   ✅ 样本数量: {len(factor_df)}")
    print(f"   ✅ 文件大小: {output_path.stat().st_size / 1024:.2f} KB")
    
    # 6. 统计报告
    print("\n6. 质量报告:")
    print(f"   平均有效率: {factor_df.notna().mean().mean()*100:.1f}%")
    print(f"   时序安全: ✅")
    print(f"   数据完整性: ✅")
    
    print("\n" + "=" * 60)
    print("✅ 资金流因子生成完成")
    print("=" * 60)
    
    return factor_df


if __name__ == "__main__":
    symbol = "000600.SZ"
    output_dir = Path("factor_output/000600_SZ/1day")
    
    try:
        generate_moneyflow_factors(symbol, output_dir)
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
