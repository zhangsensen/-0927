# A股技术分析框架

## 📋 框架概述

本框架基于Python技术栈，专为A股市场设计，提供系统化的技术分析方法和可重复使用的分析逻辑。

### 🎯 设计目标
- **系统性**: 建立标准化的技术分析流程
- **可重复性**: 提供可复现的分析结果
- **实用性**: 专注于中短期投资决策
- **风险控制**: 内置风险评估机制

### 🛠️ 技术栈
- **Python 3.8+**: 主要编程语言
- **Pandas**: 数据处理和分析
- **NumPy**: 数值计算
- **Matplotlib/Seaborn**: 数据可视化
- **技术指标库**: TA-Lib、pandas-ta

---

## 📊 数据处理模块

### 数据格式标准化

#### CSV数据结构
```csv
Date,Close,High,Low,Open,Volume
股票代码,股票代码,股票代码,股票代码,股票代码,股票代码
2025-04-01,20.87,21.14,20.86,21.00,9805664
```

#### 数据加载函数
```python
def load_stock_data(file_path):
    """加载股票数据，跳过文件头部的重复行"""
    # 读取数据，跳过前两行（标题行和重复行）
    df = pd.read_csv(file_path, skiprows=2)

    # 重命名列
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    # 转换日期格式
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # 确保数值类型正确
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 按日期排序
    df.sort_index(inplace=True)

    return df
```

#### 数据质量检查
- **完整性检查**: 确保没有缺失值
- **数据类型验证**: 确保价格和成交量为数值类型
- **日期范围验证**: 确保数据时间跨度合理
- **异常值检测**: 识别和处理异常价格数据

---

## 📈 技术指标计算模块

### 核心技术指标

#### 1. 趋势指标
```python
def calculate_trend_indicators(df):
    """计算趋势指标"""
    # 移动平均线
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()

    # 均线排列判断
    df['MA_Arrangement'] = np.where(
        (df['MA5'] > df['MA10']) &
        (df['MA10'] > df['MA20']) &
        (df['MA20'] > df['MA60']),
        'Perfect_Bullish',
        np.where(
            (df['MA5'] > df['MA10']) &
            (df['MA10'] > df['MA20']),
            'Short_Term_Bullish',
            'Mixed'
        )
    )

    return df
```

#### 2. 动量指标
```python
def calculate_momentum_indicators(df):
    """计算动量指标"""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # MACD信号
    df['MACD_Signal_State'] = np.where(df['MACD'] > df['Signal'], 'Golden_Cross', 'Death_Cross')

    return df
```

#### 3. 波动率指标
```python
def calculate_volatility_indicators(df):
    """计算波动率指标"""
    # 布林带
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

    # 布林带位置
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # 波动率
    df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean() * 100

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['ATR'] = tr.rolling(window=14).mean()

    return df
```

#### 4. 成交量指标
```python
def calculate_volume_indicators(df):
    """计算成交量指标"""
    # 成交量移动平均
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

    # 成交量相对强度
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    # 成交量价格趋势
    df['Volume_Price_Trend'] = np.where(
        (df['Close'] > df['Close'].shift(1)) &
        (df['Volume'] > df['Volume_MA20']),
        '价涨量增',
        np.where(
            (df['Close'] < df['Close'].shift(1)) &
            (df['Volume'] > df['Volume_MA20']),
            '价跌量增',
            '其他'
        )
    )

    return df
```

---

## 📊 性能评估模块

### 关键性能指标

#### 收益率指标
```python
def calculate_return_metrics(df):
    """计算收益率指标"""
    returns = df['Close'].pct_change()

    # 累计收益
    df['Cumulative_Return'] = (1 + returns).cumprod() - 1

    # 年化收益率
    days_held = len(df)
    annual_return = (1 + df['Cumulative_Return'].iloc[-1]) ** (252 / days_held) - 1

    # 最大回撤
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # 夏普比率
    risk_free_rate = 0.02  # 假设无风险利率为2%
    excess_returns = returns.mean() * 252 - risk_free_rate
    sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252))

    # 索提诺比率
    downside_returns = returns[returns < 0]
    sortino_ratio = excess_returns / (downside_returns.std() * np.sqrt(252))

    return {
        'Total_Return': df['Cumulative_Return'].iloc[-1] * 100,
        'Annualized_Return': annual_return * 100,
        'Max_Drawdown': max_drawdown * 100,
        'Sharpe_Ratio': sharpe_ratio,
        'Sortino_Ratio': sortino_ratio,
        'Volatility': returns.std() * np.sqrt(252) * 100
    }
```

#### 风险指标
```python
def calculate_risk_metrics(df):
    """计算风险指标"""
    returns = df['Close'].pct_change()

    # VaR (95%置信度)
    var_95 = returns.quantile(0.05)

    # CVaR (Expected Shortfall)
    cvar_95 = returns[returns <= var_95].mean()

    # 最大连续上涨/下跌天数
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max

    # 风险评级
    volatility = returns.std() * np.sqrt(252) * 100
    if volatility > 40:
        risk_level = '高风险'
    elif volatility > 25:
        risk_level = '中等风险'
    else:
        risk_level = '低风险'

    return {
        'VaR_95': var_95 * 100,
        'CVaR_95': cvar_95 * 100,
        'Risk_Level': risk_level,
        'Max_Consecutive_Loss': drawdown.min() * 100
    }
```

---

## 🎯 支撑阻力位分析模块

### 斐波那契回撤位
```python
def calculate_fibonacci_levels(df):
    """计算斐波那契回撤位"""
    max_price = df['High'].max()
    min_price = df['Low'].min()
    price_range = max_price - min_price

    fib_levels = {
        '0%': max_price,
        '23.6%': max_price - price_range * 0.236,
        '38.2%': max_price - price_range * 0.382,
        '50.0%': max_price - price_range * 0.5,
        '61.8%': max_price - price_range * 0.618,
        '78.6%': max_price - price_range * 0.786,
        '100%': min_price
    }

    return fib_levels
```

### 支撑阻力位识别
```python
def identify_support_resistance(df, window=20):
    """识别支撑阻力位"""
    # 使用局部高低点
    highs = df['High'].rolling(window=window, center=True).max()
    lows = df['Low'].rolling(window=window, center=True).min()

    # 找到阻力位（高点聚集区）
    resistance_levels = []
    for i in range(len(highs)):
        if highs.iloc[i] == df['High'].iloc[i]:
            resistance_levels.append(highs.iloc[i])

    # 找到支撑位（低点聚集区）
    support_levels = []
    for i in range(len(lows)):
        if lows.iloc[i] == df['Low'].iloc[i]:
            support_levels.append(lows.iloc[i])

    # 聚类分析，找到重要的支撑阻力位
    from sklearn.cluster import DBSCAN

    if resistance_levels:
        resistance_array = np.array(resistance_levels).reshape(-1, 1)
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(resistance_array)
        unique_resistance = []
        for cluster_id in set(clustering.labels_):
            if cluster_id != -1:
                cluster_prices = resistance_array[clustering.labels_ == cluster_id]
                unique_resistance.append(np.mean(cluster_prices))

    return {
        'resistance': sorted(unique_resistance, reverse=True),
        'support': sorted(unique_resistance),
        'fibonacci': calculate_fibonacci_levels(df)
    }
```

---

## 🧠 信号生成模块

### 技术信号评分系统

#### 信号定义
```python
def generate_technical_signals(df):
    """生成技术信号"""
    signals = []
    signal_strength = 0

    # RSI信号
    current_rsi = df['RSI'].iloc[-1]
    if current_rsi > 70:
        signals.append('RSI超买')
        signal_strength -= 1
    elif current_rsi < 30:
        signals.append('RSI超卖')
        signal_strength += 1
    elif current_rsi > 50:
        signals.append('RSI强势')
        signal_strength += 0.5

    # MACD信号
    if df['MACD_Signal_State'].iloc[-1] == 'Golden_Cross':
        signals.append('MACD金叉')
        signal_strength += 1
    else:
        signals.append('MACD死叉')
        signal_strength -= 1

    # 均线信号
    if df['MA_Arrangement'].iloc[-1] == 'Perfect_Bullish':
        signals.append('均线完美多头')
        signal_strength += 1.5
    elif 'Bullish' in df['MA_Arrangement'].iloc[-1]:
        signals.append('均线多头')
        signal_strength += 0.5

    # 布林带信号
    bb_position = df['BB_Position'].iloc[-1]
    if bb_position > 0.8:
        signals.append('价格接近布林带上轨')
        signal_strength -= 0.5
    elif bb_position < 0.2:
        signals.append('价格接近布林带下轨')
        signal_strength += 0.5

    # 成交量信号
    if df['Volume_Ratio'].iloc[-1] > 1.2:
        signals.append('成交量放大')
        signal_strength += 0.5
    elif df['Volume_Ratio'].iloc[-1] < 0.8:
        signals.append('成交量萎缩')
        signal_strength -= 0.5

    return {
        'signals': signals,
        'signal_strength': signal_strength,
        'recommendation': 'Buy' if signal_strength > 1 else 'Sell' if signal_strength < -1 else 'Hold'
    }
```

#### 趋势强度评估
```python
def assess_trend_strength(df, window=20):
    """评估趋势强度"""
    recent_prices = df['Close'].tail(window)

    # 线性回归斜率
    x = np.arange(len(recent_prices))
    slope, intercept = np.polyfit(x, recent_prices, 1)

    # R²
    y_pred = slope * x + intercept
    ss_res = np.sum((recent_prices - y_pred) ** 2)
    ss_tot = np.sum((recent_prices - np.mean(recent_prices)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # 趋势强度评级
    if abs(slope) > 0.5:
        strength = '强'
    elif abs(slope) > 0.2:
        strength = '中等'
    else:
        strength = '弱'

    direction = '上升' if slope > 0 else '下降'

    return {
        'trend_direction': direction,
        'trend_strength': strength,
        'slope': slope,
        'r_squared': r_squared
    }
```

---

## 💡 决策支持模块

### 综合评分系统

#### 多维度评分
```python
def comprehensive_scoring(df, metrics, signals):
    """综合评分系统"""
    scores = {}

    # 趋势评分 (0-5分)
    trend_assessment = assess_trend_strength(df)
    if trend_assessment['trend_direction'] == '上升':
        trend_score = 3 if trend_assessment['trend_strength'] == '强' else 2
    else:
        trend_score = 1 if trend_assessment['trend_strength'] == '弱' else 0
    scores['trend'] = trend_score

    # 动量评分 (0-5分)
    momentum_score = 0
    if df['MACD_Signal_State'].iloc[-1] == 'Golden_Cross':
        momentum_score += 2
    if 30 < df['RSI'].iloc[-1] < 70:
        momentum_score += 1
    if df['RSI'].iloc[-1] > 50:
        momentum_score += 1
    scores['momentum'] = min(momentum_score, 5)

    # 成交量评分 (0-5分)
    volume_score = 0
    if df['Volume_Ratio'].iloc[-1] > 1.2:
        volume_score += 2
    if df['Volume_Price_Trend'].iloc[-1] == '价涨量增':
        volume_score += 2
    scores['volume'] = min(volume_score, 5)

    # 风险评分 (0-5分，越低越好)
    volatility = metrics['Volatility']
    if volatility < 20:
        risk_score = 5
    elif volatility < 30:
        risk_score = 4
    elif volatility < 40:
        risk_score = 3
    else:
        risk_score = 1
    scores['risk'] = risk_score

    # 综合评分
    total_score = (scores['trend'] + scores['momentum'] + scores['volume'] + scores['risk']) / 4

    return {
        'total_score': total_score,
        'trend_score': scores['trend'],
        'momentum_score': scores['momentum'],
        'volume_score': scores['volume'],
        'risk_score': scores['risk'],
        'max_score': 5
    }
```

### 投资建议生成

#### 策略推荐
```python
def generate_investment_recommendation(df, metrics, signals, scores):
    """生成投资建议"""
    current_price = df['Close'].iloc[-1]

    # 基础推荐
    base_recommendation = signals['recommendation']

    # 根据综合评分调整
    if scores['total_score'] >= 4:
        recommendation = '强烈买入'
        action = '积极建仓'
    elif scores['total_score'] >= 3:
        recommendation = '买入'
        action = '逢低买入'
    elif scores['total_score'] >= 2:
        recommendation = '持有'
        action = '观望'
    else:
        recommendation = '卖出'
        action = '逢高减仓'

    # 目标价位设定
    fib_levels = calculate_fibonacci_levels(df)
    targets = {
        'resistance_1': current_price * 1.05,
        'resistance_2': current_price * 1.10,
        'resistance_3': current_price * 1.15,
        'support_1': current_price * 0.95,
        'support_2': current_price * 0.90,
        'support_3': current_price * 0.85,
        'fib_382': fib_levels['38.2%'],
        'fib_500': fib_levels['50.0%'],
        'fib_618': fib_levels['61.8%']
    }

    # 止损建议
    volatility = metrics['Volatility']
    if volatility > 40:
        stop_loss_pct = 0.15  # 高波动股票，止损15%
    elif volatility > 25:
        stop_loss_pct = 0.10  # 中等波动股票，止损10%
    else:
        stop_loss_pct = 0.08  # 低波动股票，止损8%

    stop_loss_price = current_price * (1 - stop_loss_pct)

    return {
        'recommendation': recommendation,
        'action': action,
        'confidence': scores['total_score'] / 5,
        'targets': targets,
        'stop_loss': stop_loss_price,
        'stop_loss_pct': stop_loss_pct * 100,
        'position_size': min(30, 50 / volatility)  # 根据波动率调整仓位
    }
```

---

## 📊 报告生成模块

### 标准化报告格式

#### Markdown报告模板
```python
def generate_markdown_report(stock_code, df, metrics, signals, scores, recommendation):
    """生成Markdown格式报告"""
    report = f"""# {stock_code} 技术分析报告

**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 性能指标
- **总收益率**: {metrics['Total_Return']:.2f}%
- **年化收益率**: {metrics['Annualized_Return']:.2f}%
- **年化波动率**: {metrics['Volatility']:.2f}%
- **最大回撤**: {metrics['Max_Drawdown']:.2f}%
- **夏普比率**: {metrics['Sharpe_Ratio']:.2f}
- **当前价格**: {df['Close'].iloc[-1]:.2f}元

## 📈 技术状态
- **RSI指标**: {df['RSI'].iloc[-1]:.1f}
- **MACD状态**: {df['MACD_Signal_State'].iloc[-1]}
- **均线排列**: {df['MA_Arrangement'].iloc[-1]}
- **布林带位置**: {df['BB_Position'].iloc[-1]:.2f}
- **成交量比率**: {df['Volume_Ratio'].iloc[-1]:.2f}

## 🎯 投资建议
- **综合建议**: {recommendation['recommendation']}
- **操作建议**: {recommendation['action']}
- **信心指数**: {recommendation['confidence']:.1%}
- **建议仓位**: {recommendation['position_size']:.1f}%

### 关键价位
- **阻力位1**: {recommendation['targets']['resistance_1']:.2f}元
- **阻力位2**: {recommendation['targets']['resistance_2']:.2f}元
- **支撑位1**: {recommendation['targets']['support_1']:.2f}元
- **支撑位2**: {recommendation['targets']['support_2']:.2f}元
- **止损价位**: {recommendation['stop_loss']:.2f}元

## ⚠️ 风险提示
- **风险等级**: {metrics['Risk_Level']}
- **波动率**: {metrics['Volatility']:.2f}%
- **止损比例**: {recommendation['stop_loss_pct']:.1f}%

**免责声明**: 本分析仅供参考，不构成投资建议。投资者应根据自身风险承受能力和投资目标做出独立决策。
"""
    return report
```

---

## 🛠️ 使用指南

### 快速开始

#### 1. 数据准备
```python
# 设置数据路径
data_path = "/Users/zhangshenshen/深度量化0927/a股/STOCK_CODE/STOCK_CODE_1d_2025-09-28.csv"

# 加载数据
df = load_stock_data(data_path)
```

#### 2. 技术分析
```python
# 计算技术指标
df = calculate_trend_indicators(df)
df = calculate_momentum_indicators(df)
df = calculate_volatility_indicators(df)
df = calculate_volume_indicators(df)

# 计算性能指标
metrics = calculate_return_metrics(df)
risk_metrics = calculate_risk_metrics(df)
metrics.update(risk_metrics)

# 生成信号
signals = generate_technical_signals(df)

# 综合评分
scores = comprehensive_scoring(df, metrics, signals)

# 投资建议
recommendation = generate_investment_recommendation(df, metrics, signals, scores)
```

#### 3. 生成报告
```python
# 生成Markdown报告
report = generate_markdown_report("300450.SZ", df, metrics, signals, scores, recommendation)

# 保存报告
with open(f"/Users/zhangshenshen/深度量化0927/a股/300450.SZ_技术分析报告.md", 'w', encoding='utf-8') as f:
    f.write(report)
```

### 批量分析

#### 多股票分析
```python
def batch_analysis(stock_list):
    """批量分析多只股票"""
    results = {}

    for stock_code in stock_list:
        try:
            # 数据路径
            data_path = f"/Users/zhangshenshen/深度量化0927/a股/{stock_code}/{stock_code}_1d_2025-09-28.csv"

            # 加载和分析
            df = load_stock_data(data_path)
            # ... (分析过程)

            # 存储结果
            results[stock_code] = {
                'metrics': metrics,
                'signals': signals,
                'recommendation': recommendation
            }

        except Exception as e:
            print(f"分析 {stock_code} 时出错: {e}")

    return results
```

---

## 📈 框架优势

### 技术优势
1. **系统性**: 标准化的分析流程
2. **可重复性**: 相同输入产生相同输出
3. **可扩展性**: 易于添加新的技术指标
4. **风险控制**: 内置风险评估机制

### 实用优势
1. **数据驱动**: 基于客观数据而非主观判断
2. **实时性**: 可用于实时数据流分析
3. **自动化**: 支持批量处理和自动化
4. **可定制**: 根据不同需求调整参数

### 应用场景
- **个股分析**: 详细的技术面分析
- **组合管理**: 投资组合风险评估
- **策略验证**: 量化策略回测
- **市场监控**: 实时市场变化监控

---

## 🔄 维护和更新

### 定期维护
1. **数据源更新**: 确保数据接口稳定
2. **指标优化**: 根据市场变化调整指标参数
3. **模型验证**: 定期验证模型有效性
4. **文档更新**: 保持文档与代码同步

### 版本控制
1. **语义化版本**: 使用版本号管理
2. **变更日志**: 记录重要变更
3. **兼容性**: 保持向后兼容
4. **测试覆盖**: 确保代码质量

---

## 📝 实战案例

### 案例：300450.SZ 分析
详见 `300450_sz_technical_analysis.py` 和 `300450.SZ_技术分析报告.md`

**关键结果**:
- 总收益率: 213.31%
- 技术面: 完美多头排列
- 风险评估: 高风险
- 投资建议: 逢低买入

### 参数调优建议
- **RSI周期**: 14天（标准）
- **MACD参数**: 12,26,9（标准）
- **移动平均线**: 5,10,20,30,60日
- **布林带**: 20日均线，2倍标准差

---

## ⚠️ 使用限制和风险提示

### 数据限制
1. **数据质量**: 依赖输入数据的准确性
2. **时间跨度**: 历史数据可能不反映未来表现
3. **市场环境**: 不同市场环境下表现可能不同

### 模型限制
1. **技术分析局限性**: 不能预测黑天鹅事件
2. **参数敏感性**: 结果对参数设置敏感
3. **过拟合风险**: 可能对历史数据过度优化

### 使用建议
1. **结合基本面**: 建议与基本面分析结合使用
2. **风险控制**: 严格执行止损策略
3. **持续学习**: 根据市场变化调整策略
4. **专业建议**: 建议咨询专业投资顾问

---

**最后更新**: 2025年9月28日
**版本**: 1.0.0
**维护者**: Claude量化分析系统