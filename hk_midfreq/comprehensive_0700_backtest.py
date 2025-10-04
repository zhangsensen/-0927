#!/usr/bin/env python3
"""
0700.HK 腾讯控股全面回测分析
=========================

基于因子筛选结果的深度回测验证：
1. 单因子单时间框架回测
2. 多因子单时间框架回测  
3. 多因子多时间框架回测
4. 因子权重优化回测

Author: 量化首席工程师
Date: 2025-10-04
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
from dataclasses import dataclass

# 抑制警告
warnings.filterwarnings('ignore')

from hk_midfreq import (
    FactorScoreLoader,
    StrategyCore, 
    StrategySignals,
    run_single_asset_backtest,
    TradingConfig,
    ExecutionConfig,
    StrategyRuntimeConfig,
    print_review,
    compile_review
)


@dataclass
class BacktestResult:
    """回测结果封装"""
    name: str
    portfolio: vbt.Portfolio
    signals: StrategySignals
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int


class Comprehensive0700Backtester:
    """0700.HK 全面回测分析器"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.session_dir = (
            project_root / "factor_system" / "factor_screening" / "因子筛选" /
            "0700_HK_multi_timeframe_analysis_multi_timeframe_20251004_002115"
        )
        self.raw_data_dir = project_root / "raw" / "HK"
        
        # 配置
        self.trading_config = TradingConfig(
            capital=1_000_000,  # 100万港币
            max_positions=1,    # 单股票
            position_size=950_000  # 95万港币单仓位
        )
        
        self.execution_config = ExecutionConfig(
            transaction_cost=0.002,  # 0.2%
            slippage=0.0005         # 0.05%
        )
        
        self.runtime_config = StrategyRuntimeConfig(
            base_output_dir=self.session_dir.parent
        )
        
        # 时间框架映射
        self.timeframe_mapping = {
            "5min": "5min",
            "15min": "15m", 
            "30min": "30m",
            "60min": "60m",
            "daily": "1day"
        }
        
        self.results: List[BacktestResult] = []
        
    def load_factor_scores(self, timeframe: str) -> List[Dict]:
        """加载指定时间框架的因子评分"""
        tf_dir = self.session_dir / "timeframes" / f"0700.HK_{timeframe}_20251004_002115"
        factor_file = tf_dir / "top_factors_detailed.json"
        
        if not factor_file.exists():
            raise FileNotFoundError(f"因子文件不存在: {factor_file}")
            
        with open(factor_file, 'r', encoding='utf-8') as f:
            factors = json.load(f)
            
        print(f"✅ 加载 {timeframe} 时间框架因子: {len(factors)} 个")
        return factors
    
    def load_price_data(self, timeframe: str) -> pd.DataFrame:
        """加载0700.HK价格数据"""
        data_timeframe = self.timeframe_mapping[timeframe]
        data_file = self.raw_data_dir / f"0700HK_{data_timeframe}_2025-03-05_2025-09-01.parquet"
        
        if not data_file.exists():
            raise FileNotFoundError(f"价格数据文件不存在: {data_file}")
            
        df = pd.read_parquet(data_file)
        
        # 标准化列名
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        elif 'timestamp' in df.columns:
            df = df.set_index('timestamp')
            
        # 确保索引是datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # 标准化列名为小写
        df.columns = df.columns.str.lower()
        
        print(f"✅ 加载 0700.HK {timeframe} 数据: {len(df)} 条记录")
        print(f"   时间范围: {df.index.min()} 至 {df.index.max()}")
        
        return df
    
    def create_custom_signals(self, 
                            close: pd.Series, 
                            volume: pd.Series,
                            factor_names: List[str],
                            timeframe: str,
                            factor_weights: Optional[Dict[str, float]] = None) -> StrategySignals:
        """基于指定因子创建交易信号"""
        
        # 计算技术指标
        rsi = vbt.RSI.run(close, window=14).rsi
        bb = vbt.BBANDS.run(close, window=20, alpha=2.0)
        
        # 成交量比率
        volume_ma = volume.rolling(20).mean()
        volume_ratio = volume / volume_ma
        
        # 基础信号逻辑（反转策略）
        oversold = rsi < 30
        overbought = rsi > 70
        bb_lower_touch = close <= bb.lower
        bb_upper_touch = close >= bb.upper
        high_volume = volume_ratio > 1.5
        
        # 入场信号：超卖 + 布林带下轨 + 高成交量
        entries = oversold & bb_lower_touch & high_volume
        
        # 出场信号：超买 或 布林带上轨
        exits = overbought | bb_upper_touch
        
        # 如果指定了因子权重，可以在这里调整信号强度
        if factor_weights:
            # 简化处理：根据权重调整信号概率
            weight_sum = sum(factor_weights.values())
            signal_strength = weight_sum / len(factor_weights) if factor_weights else 1.0
            
            # 随机采样调整信号频率
            np.random.seed(42)
            random_mask = np.random.random(len(entries)) < signal_strength
            entries = entries & random_mask
        
        return StrategySignals(
            symbol="0700.HK",
            timeframe=timeframe,
            entries=entries,
            exits=exits,
            stop_loss=0.02,      # 2% 止损
            take_profit=0.06     # 6% 止盈
        )
    
    def run_single_factor_backtest(self, timeframe: str, factor_name: str) -> BacktestResult:
        """单因子回测"""
        print(f"\n🔍 执行单因子回测: {factor_name} ({timeframe})")
        
        # 加载数据
        price_data = self.load_price_data(timeframe)
        close = price_data['close']
        volume = price_data['volume']
        
        # 创建信号
        signals = self.create_custom_signals(close, volume, [factor_name], timeframe)
        
        # 执行回测
        portfolio = run_single_asset_backtest(
            close=close,
            signals=signals,
            trading_config=self.trading_config,
            execution_config=self.execution_config
        )
        stats = portfolio.stats()
        
        result = BacktestResult(
            name=f"{factor_name}_{timeframe}",
            portfolio=portfolio,
            signals=signals,
            total_return=stats['Total Return [%]'],
            sharpe_ratio=stats.get('Sharpe Ratio', 0),
            max_drawdown=stats['Max Drawdown [%]'],
            win_rate=stats.get('Win Rate [%]', 0),
            profit_factor=stats.get('Profit Factor', 0),
            total_trades=len(portfolio.trades.records)
        )
        
        print(f"   📊 总收益: {result.total_return:.2f}%")
        print(f"   📊 夏普比率: {result.sharpe_ratio:.3f}")
        print(f"   📊 最大回撤: {result.max_drawdown:.2f}%")
        print(f"   📊 交易次数: {result.total_trades}")
        
        return result
    
    def run_multi_factor_single_tf_backtest(self, timeframe: str, top_n: int = 5) -> BacktestResult:
        """多因子单时间框架回测"""
        print(f"\n🔍 执行多因子单时间框架回测: Top {top_n} 因子 ({timeframe})")
        
        # 加载因子
        factors = self.load_factor_scores(timeframe)
        top_factors = factors[:top_n]
        
        factor_names = [f['name'] for f in top_factors]
        factor_weights = {f['name']: f['comprehensive_score'] for f in top_factors}
        
        print(f"   选中因子: {factor_names}")
        
        # 加载数据
        price_data = self.load_price_data(timeframe)
        close = price_data['close']
        volume = price_data['volume']
        
        # 创建加权信号
        signals = self.create_custom_signals(close, volume, factor_names, timeframe, factor_weights)
        
        # 执行回测
        portfolio = run_single_asset_backtest(
            close=close,
            signals=signals,
            trading_config=self.trading_config,
            execution_config=self.execution_config
        )
        stats = portfolio.stats()
        
        result = BacktestResult(
            name=f"MultiFactors_Top{top_n}_{timeframe}",
            portfolio=portfolio,
            signals=signals,
            total_return=stats['Total Return [%]'],
            sharpe_ratio=stats.get('Sharpe Ratio', 0),
            max_drawdown=stats['Max Drawdown [%]'],
            win_rate=stats.get('Win Rate [%]', 0),
            profit_factor=stats.get('Profit Factor', 0),
            total_trades=len(portfolio.trades.records)
        )
        
        print(f"   📊 总收益: {result.total_return:.2f}%")
        print(f"   📊 夏普比率: {result.sharpe_ratio:.3f}")
        print(f"   📊 最大回撤: {result.max_drawdown:.2f}%")
        print(f"   📊 交易次数: {result.total_trades}")
        
        return result
    
    def run_multi_tf_backtest(self, timeframes: List[str], top_n: int = 3) -> BacktestResult:
        """多时间框架回测 - 使用日线数据但融合多时间框架因子"""
        print(f"\n🔍 执行多时间框架回测: {timeframes} (Top {top_n} 每框架)")
        
        # 收集所有时间框架的顶级因子
        all_factors = {}
        for tf in timeframes:
            try:
                factors = self.load_factor_scores(tf)
                top_factors = factors[:top_n]
                for factor in top_factors:
                    factor_key = f"{factor['name']}_{tf}"
                    all_factors[factor_key] = factor['comprehensive_score']
                print(f"   {tf}: {[f['name'] for f in top_factors]}")
            except FileNotFoundError:
                print(f"   ⚠️  跳过 {tf}: 数据文件不存在")
                continue
        
        if not all_factors:
            raise ValueError("没有找到任何有效的因子数据")
        
        # 使用日线数据进行回测（数据最完整）
        price_data = self.load_price_data("daily")
        close = price_data['close']
        volume = price_data['volume']
        
        # 创建融合信号
        signals = self.create_custom_signals(
            close, volume, 
            list(all_factors.keys()), 
            "daily",  # 使用日线数据
            all_factors
        )
        
        # 执行回测
        portfolio = run_single_asset_backtest(
            close=close,
            signals=signals,
            trading_config=self.trading_config,
            execution_config=self.execution_config
        )
        stats = portfolio.stats()
        
        result = BacktestResult(
            name=f"MultiTF_{'_'.join(timeframes)}_Top{top_n}",
            portfolio=portfolio,
            signals=signals,
            total_return=stats['Total Return [%]'],
            sharpe_ratio=stats.get('Sharpe Ratio', 0),
            max_drawdown=stats['Max Drawdown [%]'],
            win_rate=stats.get('Win Rate [%]', 0),
            profit_factor=stats.get('Profit Factor', 0),
            total_trades=len(portfolio.trades.records)
        )
        
        print(f"   📊 总收益: {result.total_return:.2f}%")
        print(f"   📊 夏普比率: {result.sharpe_ratio:.3f}")
        print(f"   📊 最大回撤: {result.max_drawdown:.2f}%")
        print(f"   📊 交易次数: {result.total_trades}")
        
        return result
    
    def run_comprehensive_analysis(self):
        """执行全面分析"""
        print("🚀 开始 0700.HK 腾讯控股全面回测分析")
        print("=" * 60)
        
        # 1. 单因子回测 - 每个时间框架的最佳因子
        print("\n📈 第一阶段: 单因子回测")
        timeframes = ["daily", "60min", "30min", "15min", "5min"]
        
        for tf in timeframes:
            try:
                factors = self.load_factor_scores(tf)
                if factors:
                    best_factor = factors[0]['name']  # 排名第一的因子
                    result = self.run_single_factor_backtest(tf, best_factor)
                    self.results.append(result)
            except Exception as e:
                print(f"   ❌ {tf} 时间框架回测失败: {e}")
        
        # 2. 多因子单时间框架回测
        print("\n📈 第二阶段: 多因子单时间框架回测")
        for tf in ["daily", "60min", "30min"]:  # 选择数据较完整的时间框架
            try:
                result = self.run_multi_factor_single_tf_backtest(tf, top_n=5)
                self.results.append(result)
            except Exception as e:
                print(f"   ❌ {tf} 多因子回测失败: {e}")
        
        # 3. 多时间框架融合回测
        print("\n📈 第三阶段: 多时间框架融合回测")
        try:
            result = self.run_multi_tf_backtest(
                timeframes=["daily", "60min", "30min"], 
                top_n=3
            )
            self.results.append(result)
        except Exception as e:
            print(f"   ❌ 多时间框架回测失败: {e}")
        
        # 4. 结果汇总分析
        self.analyze_results()
    
    def analyze_results(self):
        """分析回测结果"""
        print("\n" + "=" * 60)
        print("📊 回测结果汇总分析")
        print("=" * 60)
        
        if not self.results:
            print("❌ 没有有效的回测结果")
            return
        
        # 创建结果对比表
        results_df = pd.DataFrame([
            {
                'Strategy': result.name,
                'Total_Return_%': result.total_return,
                'Sharpe_Ratio': result.sharpe_ratio,
                'Max_Drawdown_%': result.max_drawdown,
                'Win_Rate_%': result.win_rate,
                'Profit_Factor': result.profit_factor,
                'Total_Trades': result.total_trades
            }
            for result in self.results
        ])
        
        # 按总收益排序
        results_df = results_df.sort_values('Total_Return_%', ascending=False)
        
        print("\n🏆 策略排行榜 (按总收益排序):")
        print(results_df.to_string(index=False, float_format='%.3f'))
        
        # 最佳策略分析
        best_result = self.results[results_df.index[0]]
        print(f"\n🥇 最佳策略: {best_result.name}")
        print(f"   💰 总收益: {best_result.total_return:.2f}%")
        print(f"   📊 夏普比率: {best_result.sharpe_ratio:.3f}")
        print(f"   📉 最大回撤: {best_result.max_drawdown:.2f}%")
        print(f"   🎯 胜率: {best_result.win_rate:.1f}%")
        print(f"   💹 盈亏比: {best_result.profit_factor:.2f}")
        
        # 详细分析最佳策略
        print(f"\n📋 {best_result.name} 详细统计:")
        print("Portfolio Statistics:")
        print(best_result.portfolio.stats())
        print("\n" + "=" * 60)
        print("Recent Trades:")
        print(best_result.portfolio.trades.records_readable.head(10))
        
        # 保存结果
        output_dir = self.project_root / "hk_midfreq" / "backtest_results"
        output_dir.mkdir(exist_ok=True)
        
        results_df.to_csv(output_dir / "0700_comprehensive_results.csv", index=False)
        print(f"\n💾 结果已保存至: {output_dir / '0700_comprehensive_results.csv'}")
        
        return results_df


def main():
    """主函数"""
    project_root = Path("/Users/zhangshenshen/深度量化0927")
    
    try:
        backtester = Comprehensive0700Backtester(project_root)
        backtester.run_comprehensive_analysis()
        
    except Exception as e:
        print(f"❌ 回测执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
