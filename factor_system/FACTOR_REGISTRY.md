# Factor Registry - 统一因子注册表

本文档定义了量化系统中所有可用因子的权威清单，确保FactorEngine和factor_generation两个系统的一致性。

## 因子分类体系

### 1. 移动平均类 (Moving Averages) - 33个因子

#### 标准移动平均
- `MA3`, `MA5`, `MA8`, `MA10`, `MA12`, `MA15`, `MA20`, `MA25`, `MA30`, `MA40`, `MA50`, `MA60`, `MA80`, `MA100`, `MA120`, `MA150`, `MA200`
- `SMA{period}` - 简单移动平均的参数化版本

#### 指数移动平均
- `EMA3`, `EMA5`, `EMA8`, `EMA12`, `EMA15`, `EMA20`, `EMA26`, `EMA30`, `EMA40`, `EMA50`, `EMA60`
- `EMA{period}` - 指数移动平均的参数化版本

#### 高级移动平均
- `DEMA` - 双指数移动平均
- `TEMA` - 三指数移动平均
- `KAMA` - 考夫曼自适应移动平均
- `MAMA` - MESA自适应移动平均
- `WMA` - 加权移动平均
- `TRIMA` - 三角移动平均

### 2. MACD指标类 - 4个因子

#### 标准MACD
- `MACD` - MACD线 (EMA_fast - EMA_slow)
- `MACD_SIGNAL` - MACD信号线 (MACD的EMA)
- `MACD_HIST` - MACD柱状图 (MACD - Signal)

#### 参数化版本
- `MACD_{fast}_{slow}_{signal}` - 例如: `MACD_12_26_9`, `MACD_5_35_5`

#### 扩展MACD
- `MACDEXT` - 扩展MACD
- `MACDFIX` - 固定周期MACD

### 3. RSI指标类 - 10个因子

#### 标准RSI
- `RSI` - 标准RSI (默认周期14)

#### 参数化版本
- `RSI3`, `RSI6`, `RSI9`, `RSI12`, `RSI14`, `RSI18`, `RSI21`, `RSI25`, `RSI30`
- `RSI{period}` - RSI的参数化版本

### 4. 随机指标类 - 3个因子

#### 标准随机指标
- `STOCH` - 随机指标 (%K, %D)
- `STOCHF` - 快速随机指标
- `STOCHRSI` - RSI随机指标

#### 参数化版本
- `STOCH_{k}_{d}` - 例如: `STOCH_14_3`
- `STOCHF_{k}_{d}` - 例如: `STOCHF_5_3`
- `STOCHRSI_{fastk}_{fastk}_{timeperiod}` - 例如: `STOCHRSI_3_5_14`

### 5. 布林带类 - 1个因子

#### 标准布林带
- `BBANDS` - 布林带 (上轨、中轨、下轨)

#### 衍生指标
- `BOLB_{period}` - 布林带位置 (收盘价相对于布林带的位置)
- `{name}_Upper`, `{name}_Middle`, `{name}_Lower`, `{name}_Width` - 布林带组件

### 6. 威廉指标类 - 2个因子

#### 标准威廉指标
- `WILLR` - 威廉%R指标

#### 参数化版本
- `WILLR{period}` - 威廉指标的参数化版本

### 7. 商品通道指标类 - 2个因子

#### 标准CCI
- `CCI` - 商品通道指标

#### 参数化版本
- `CCI{period}` - CCI的参数化版本

### 8. ATR指标类 - 1个因子

#### 标准ATR
- `ATR` - 平均真实波幅

### 9. 波动率指标类 - 1个因子

#### 移动标准差
- `MSTD` - 移动标准差
- `FSTD` - 固定窗口标准差

### 10. 成交量指标类 - 6个因子

#### 标准成交量指标
- `OBV` - 能量潮指标
- `OBV_` - OBV的变体

#### VWAP指标
- `VWAP` - 成交量加权平均价格
- `VWAP{window}` - 参数化VWAP

#### 成交量动量
- `Volume_Momentum{window}` - 成交量动量
- `Volume_Ratio{window}` - 成交量比率

### 11. 动量指标类 - 1个因子

#### 价格动量
- `Momentum{period}` - 价格动量指标

### 12. 位置指标类 - 1个因子

#### 价格位置
- `Position{window}` - 价格在窗口中的相对位置

### 13. 趋势强度类 - 1个因子

#### 趋势强度
- `Trend{window}` - 趋势强度指标

### 14. 方向性指标类 - 6个因子

#### ADX系列
- `ADX` - 平均趋向指标
- `ADXR` - 平滑ADX

#### 方向性运动
- `PLUS_DM` - 上升方向运动
- `MINUS_DM` - 下降方向运动
- `PLUS_DI` - 上升方向指标
- `MINUS_DI` - 下降方向指标
- `DX` - 方向性运动指标

### 15. 阿隆指标类 - 2个因子

#### 阿隆指标
- `AROON` - 阿隆指标
- `AROONOSC` - 阿隆振荡器

### 16. 其他技术指标类 - 20个因子

#### 价格振荡器
- `APO` - 绝对价格振荡器
- `PPO` - 百分比价格振荡器

#### 变化率指标
- `ROC` - 变化率
- `ROCP` - 变化率百分比
- `ROCR` - 变化率比率
- `ROCR100` - 变化率比率×100

#### 动量指标
- `MOM` - 动量
- `TRIX` - 三重指数平滑移动平均

#### 终极振荡器
- `ULTOSC` - 终极振荡器

#### 其他指标
- `CMO` - 钱德动量摆动指标
- `MFI` - 资金流量指标
- `NATR` - 归一化平均真实波幅
- `TRANGE` - 真实波幅
- `AD` - 累积/派发线
- `BOP` - 均势指标

### 17. 统计指标类 - 15个因子

#### 价格统计
- `MEDPRICE` - 中位价
- `TYPPRICE` - 典型价格
- `WCLPRICE` - 加权收盘价
- `AVGPRICE` - 平均价格
- `MIDPRICE` - 中间价
- `MIDPOINT` - 中点价

#### 回归分析
- `LINEARREG` - 线性回归
- `LINEARREG_SLOPE` - 线性回归斜率
- `LINEARREG_INTERCEPT` - 线性回归截距
- `LINEARREG_ANGLE` - 线性回归角度
- `TSF` - 时间序列预测

#### 相关性分析
- `CORREL` - 皮尔逊相关系数
- `BETA` - 贝塔系数

#### 分布统计
- `STDDEV` - 标准差
- `VAR` - 方差

### 18. 希尔伯特变换类 - 7个因子

#### 希尔伯特变换指标
- `HT_TRENDMODE` - 希尔伯特变换趋势模式
- `HT_TRENDLINE` - 希尔伯特变换趋势线
- `HT_SINE` - 希尔伯特变换正弦波
- `HT_PHASOR` - 希尔伯特变换相量
- `HT_DCPERIOD` - 希尔伯特变换DC周期
- `HT_DCPHASE` - 希尔伯特变换DC相位

### 19. K线形态类 - 60+个因子

#### 强势形态
- `CDL3WHITESOLDIERS` - 三个白兵
- `CDLMORNINGSTAR` - 启明星
- `CDLMORNINGDOJISTAR` - 十字启明星
- `CDL3LINESTRIKE` - 三线打击
- `CDLDRAGONFLYDOJI` - 蜻蜓十字

#### 弱势形态
- `CDL3BLACKCROWS` - 三只乌鸦
- `CDLEVENINGSTAR` - 暮星
- `CDLEVENINGDOJISTAR` - 十字暮星
- `CDLDARKCLOUDCOVER` - 乌云压顶

#### 反转形态
- `CDLENGULFING` - 吞噬形态
- `CDLHARAMI` - 孕线形态
- `CDLHARAMICROSS` - 十字孕线
- `CDLINVERTEDHAMMER` - 倒锤头
- `CDLHAMMER` - 锤头

#### 持续形态
- `CDL3INSIDE` - 三内部形态
- `CDL3OUTSIDE` - 三外部形态
- `CDLRISEFALL3METHODS` - 上升下降三法

#### 其他形态 (50+个)
- 包括所有TA-Lib提供的K线形态识别指标

### 20. 自定义指标类 - 50+个因子

#### 前沿/后沿指标
- `MEANLB{period}` - 均值回溯指标
- `FIXLB{period}` - 固定回溯指标
- `TRENDLB{period}` - 趋势回溯指标
- `LEXLB{period}` - 线性回溯指标

#### 统计函数
- `FMEAN`, `FMAX`, `FMIN`, `FSTD` - 固定窗口统计
- `RPROB{window}`, `RPROBCX`, `RPROBNX`, `RPROBX` - 概率指标
- `STCX`, `STX`, `OHLCSTCX`, `OHLCSTX` - 统计阈值指标

#### 其他自定义指标
- `BETWEEN` - 区间指标
- `TOP_N`, `BOTTOM_N` - 排名指标
- `RAND`, `RANDNX`, `RANDX` - 随机指标

## 参数化规则

### 标准参数化格式
- `{FACTOR_NAME}_{param1}_{param2}` - 多参数因子
- `{FACTOR_NAME}{period}` - 单参数因子
- `{name}_{component}` - 多组件因子 (如布林带的_Upper, _Lower等)

### 常用参数范围
- 移动平均周期: 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200
- RSI周期: 3, 6, 9, 12, 14, 18, 21, 25, 30
- MACD参数: (5,35,5), (12,26,9), (快速, 慢速, 信号)
- 随机指标: (5,3), (14,3), (%K周期, %D周期)

## 系统映射规则

### FactorEngine映射
- FactorEngine使用标准化名称 (如: RSI, MACD, STOCH)
- 通过参数化构造函数支持多种参数组合
- 统一使用共享计算器确保计算一致性

### factor_generation映射
- factor_generation使用具体参数化名称 (如: RSI14, MACD_12_26_9)
- 通过配置文件控制启用的因子和参数组合
- 支持批量生成多种参数变体

### 一致性保证
- 两个系统都通过SHARED_CALCULATORS确保计算逻辑一致
- 使用相同的TA-Lib/VectorBT底层实现
- 通过factor_consistency_guard.py验证一致性

## 总计

- **总因子数**: 246+ (包括参数化变体)
- **核心因子**: 约100个 (不含参数化变体)
- **因子类别**: 20个主要类别
- **参数化组合**: 1500+ 种可能的参数组合

---

*最后更新: 2025-10-08*
*版本: v1.0*
*维护者: 量化系统团队*