# 故障排除和常见问题

## 环境配置问题

### FactorEngine安装问题
```
问题: ModuleNotFoundError: No module named 'factor_system.factor_engine'
解决方案:
1. 确保在项目根目录下
2. 安装开发版本: pip install -e .
3. 检查虚拟环境是否激活: which python
4. 验证安装: python -c "from factor_system.factor_engine import api"

问题: TA-Lib安装失败
解决方案:
1. macOS: brew install ta-lib
2. Ubuntu: sudo apt-get install libta-lib-dev
3. Windows: 下载预编译版本
4. 或使用conda: conda install -c conda-forge ta-lib

问题: VectorBT安装缓慢
解决方案:
1. 使用国内镜像: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vectorbt
2. 预编译版本: pip install --only-binary=all vectorbt
3. 分步安装: 先安装numba, 再安装vectorbt
```

### 依赖版本冲突
```
问题: pandas版本冲突
解决方案:
1. 检查当前版本: pip show pandas
2. 更新到兼容版本: pip install pandas>=2.3.2
3. 清理缓存: pip cache purge
4. 重新安装: pip install --force-reinstall pandas

问题: NumPy版本不兼容
解决方案:
1. 检查NumPy版本: python -c "import numpy; print(numpy.__version__)"
2. 升级NumPy: pip install numpy>=2.3.3
3. 如果有conda环境: conda update numpy

问题: uv sync失败
解决方案:
1. 清理uv缓存: uv cache clean
2. 删除锁定文件: rm uv.lock
3. 重新同步: uv sync
4. 检查Python版本: python --version (需要3.11+)
```

## 数据访问问题

### 文件路径错误
```
问题: FileNotFoundError: 数据文件不存在
诊断步骤:
1. 检查数据路径配置: factor_generation/config.yaml
2. 验证文件存在: ls -la /path/to/data/
3. 检查文件权限: ls -l /path/to/data/file.parquet
4. 确认文件格式正确: file /path/to/data/file.parquet

解决方案:
1. 更新配置文件中的数据路径
2. 使用绝对路径而非相对路径
3. 检查符号链接是否有效
4. 确保数据文件未损坏

问题: Parquet文件读取错误
解决方案:
1. 检查PyArrow版本: pip show pyarrow
2. 验证文件完整性: python -c "import pyarrow.parquet as pq; pq.read_table('file.parquet')"
3. 重新生成文件: python batch_resample_hk.py
4. 检查磁盘空间: df -h
```

### 数据格式问题
```
问题: OHLCV数据格式错误
常见错误:
- 开盘价 > 收盘价
- 最低价 > 开盘价/收盘价
- 成交量为负数
- 时间戳不连续

诊断代码:
```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.describe())
print("价格关系检查:", (df['low'] > df[['open', 'close']].min(axis=1)).sum())
print("成交量检查:", (df['volume'] < 0).sum())
print("时间戳检查:", df.index.is_monotonic_increasing)
```

解决方案:
1. 数据清洗: 使用data_validator.py
2. 异常值处理: 设置合理阈值
3. 时间戳对齐: 重新索引
4. 缺失值填充: 前向填充或插值
```

## 因子计算问题

### 内存不足
```
症状: MemoryError 或进程被杀死
诊断:
1. 监控内存使用: htop 或 ps aux
2. 检查数据规模: df.shape
3. 分析内存占用: df.memory_usage(deep=True)

解决方案:
1. 减少批量处理大小
2. 使用数据类型优化: df.astype('float32')
3. 启用增量处理: chunk_size=1000
4. 清理缓存: api.clear_cache()
5. 增加虚拟内存: sudo swapon /swapfile

配置优化:
```python
# factor_generation/config.yaml
analysis:
  memory_limit_mb: 4096  # 降低内存限制
  chunk_size: 500       # 减小处理块
  enable_parallel: False # 禁用并行处理
```
```

### 计算缓慢
```
症状: 因子计算时间过长
诊断:
1. 性能分析: python -m cProfile script.py
2. 检查CPU使用: top -p $(pgrep python)
3. 监控I/O: iostat -x 1

优化方案:
1. 启用并行计算: n_jobs=-1
2. 使用VectorBT加速
3. 预热缓存: api.prewarm_cache()
4. 减少因子数量
5. 使用更小的时间范围

性能调优配置:
```python
# 提升计算性能
settings = {
    'n_jobs': -1,           # 使用所有CPU核心
    'chunk_size': 2000,     # 适当增大块大小
    'cache_size': '1GB',    # 增大缓存
    'use_vectorbt': True    # 启用VectorBT
}
```
```

### 因子值异常
```
症状: 因子值为NaN或inf
诊断:
```python
# 检查异常值
print(df.isnull().sum())
print(np.isinf(df).sum())

# 检查特定因子
rsi = api.calculate_single_factor('RSI', symbol='0700.HK')
print(f"RSI统计: min={rsi.min()}, max={rsi.max()}, mean={rsi.mean()}")
print(f"异常值: {(rsi < 0).sum()}, {(rsi > 100).sum()}")
```

常见原因:
1. 数据不足: 样本数量少于计算窗口
2. 价格异常: 价格为零或负数
3. 除权除息: 价格跳空导致指标异常
4. 停牌数据: 长时间无交易

解决方案:
1. 增加数据范围: 扩大start_date和end_date
2. 数据清洗: 移除异常价格点
3. 调整参数: 减小计算窗口
4. 异常值处理: 使用np.nan_to_num()
```

## 缓存系统问题

### 缓存未命中
```
症状: 缓存命中率低
诊断:
```python
stats = api.get_cache_stats()
print(f"内存缓存命中率: {stats['memory_hit_rate']:.2%}")
print(f"磁盘缓存命中率: {stats['disk_hit_rate']:.2%}")
```

优化方案:
1. 预热常用缓存
2. 调整TTL设置
3. 增大缓存容量
4. 优化缓存键策略

缓存预热示例:
```python
# 预热热门因子和股票
api.prewarm_cache(
    factor_ids=['RSI', 'MACD', 'STOCH'],
    symbols=['0700.HK', '0005.HK', '0941.HK'],
    timeframe='15min',
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31)
)
```
```

### 缓存损坏
```
症状: 缓存数据读取错误
解决方案:
1. 清理缓存: api.clear_cache()
2. 删除缓存目录: rm -rf cache/factor_engine/
3. 重新计算因子
4. 检查磁盘完整性: fsck -f

预防措施:
1. 定期备份重要缓存
2. 使用校验和验证
3. 监控磁盘健康状态
4. 设置合理的缓存过期策略
```

## 筛选系统问题

### IC计算异常
```
症状: IC值为NaN或异常大/小
诊断:
```python
# 检查收益率数据
returns = data['close'].pct_change()
print(f"收益率统计: mean={returns.mean():.6f}, std={returns.std():.6f}")
print(f"异常收益率: {(returns.abs() > 0.5).sum()}")  # >50%的异常收益率

# 检查因子分布
factor_data = calculated_factors['RSI']
print(f"因子统计: min={factor_data.min():.2f}, max={factor_data.max():.2f}")
print(f"缺失值: {factor_data.isnull().sum()}")
```

解决方案:
1. 处理极端收益率: 使用winsorization
2. 填充缺失值: 前向填充或均值填充
3. 标准化因子数据: z-score标准化
4. 增加最小样本数要求

数据清洗示例:
```python
# 极端值处理
returns = returns.clip(returns.quantile(0.01), returns.quantile(0.99))

# 缺失值处理
factor_data = factor_data.fillna(method='ffill').fillna(method='bfill')

# 标准化
factor_data = (factor_data - factor_data.mean()) / factor_data.std()
```
```

### VIF计算错误
```
症状: VIF值为inf或计算失败
原因:
1. 完全共线性: 两个因子完全相关
2. 常数因子: 因子值无变化
3. 样本不足: 样本数少于因子数
4. 缺失值: 存在NaN值

解决方案:
```python
# 检查共线性
corr_matrix = factor_data.corr()
high_corr_pairs = np.where(np.abs(corr_matrix) > 0.99)
print("高相关性因子对:", high_corr_pairs)

# 移除常数因子
constant_factors = factor_data.columns[factor_data.var() == 0]
print("常数因子:", constant_factors)

# 处理缺失值
factor_data = factor_data.dropna()
```
```

## 外部集成问题

### QuantConnect连接问题
```
症状: 认证失败或API调用超时
诊断:
1. 检查网络连接: ping www.quantconnect.com
2. 验证API凭据: ~/.quantconnect/config.json
3. 测试API: ./quantconnect-mcp-wrapper.sh test

解决方案:
1. 更新API凭据
2. 检查网络代理设置
3. 增加超时时间
4. 使用VPN或代理

配置示例:
```bash
# 设置代理
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# 增加超时
export QUANTCONNECT_TIMEOUT=60
```
```

### yfinance数据问题
```
症状: 数据下载失败或数据异常
常见问题:
1. API限制: 请求过于频繁
2. 市场休市: 无交易数据
3. 股票代码错误: 代码不匹配
4. 网络超时: 连接不稳定

解决方案:
1. 限制请求频率: time.sleep(0.5)
2. 检查交易日历
3. 验证股票代码: 使用搜索功能
4. 增加重试机制

错误处理示例:
```python
import yfinance as yf
import time
from requests.exceptions import RequestException

def safe_download(symbol, period="1y"):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = yf.download(symbol, period=period)
            if not data.empty:
                return data
        except RequestException as e:
            print(f"下载失败 (尝试 {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
    return None
```
```

## 性能问题排查

### 内存泄漏
```
症状: 内存使用持续增长
诊断工具:
1. 内存监控: ps aux --sort=%mem
2. 内存分析: python -m memory_profiler script.py
3. 垃圾回收: gc.collect()

解决方案:
1. 及时删除大对象: del large_dataframe
2. 手动垃圾回收: import gc; gc.collect()
3. 使用生成器而非列表
4. 限制缓存大小

代码优化示例:
```python
# 避免
large_list = []
for item in generator:
    processed = expensive_function(item)
    large_list.append(processed)

# 推荐
def process_items():
    for item in generator:
        yield expensive_function(item)
```
```

### CPU使用率过高
```
症状: CPU使用率持续100%
诊断:
1. 进程监控: top -p $(pgrep python)
2. 线程分析: ps -T -p $(pgrep python)
3. 性能分析: py-spy top --pid $(pgrep python)

优化方案:
1. 减少并行进程数
2. 优化算法复杂度
3. 使用更高效的数据结构
4. 启用JIT编译

并行处理优化:
```python
# 避免过度并行
from multiprocessing import cpu_count
n_jobs = min(cpu_count() - 1, 4)  # 保留一个核心，最多4个并行
```
```

## 数据完整性问题

### 数据损坏检测
```
检测方法:
1. 文件校验和: md5sum datafile.parquet
2. 数据统计: df.describe()
3. 可视化检查: df.plot()
4. 业务规则验证

自动检测代码:
```python
def validate_data(df):
    """验证数据完整性"""
    issues = []

    # 检查缺失值
    missing_data = df.isnull().sum()
    if missing_data.any():
        issues.append(f"缺失值: {missing_data[missing_data > 0].to_dict()}")

    # 检查异常值
    for col in ['open', 'high', 'low', 'close']:
        if (df[col] <= 0).any():
            issues.append(f"{col}存在非正值")

    # 检查价格关系
    invalid_prices = (df['low'] > df['high']).sum()
    if invalid_prices > 0:
        issues.append(f"价格关系错误: {invalid_prices}条记录")

    # 检查时间序列
    if not df.index.is_monotonic_increasing:
        issues.append("时间戳非递增")

    return issues
```
```

### 数据恢复
```
恢复策略:
1. 从备份恢复
2. 重新下载原始数据
3. 数据插值和修复
4. 部分数据重建

自动恢复脚本:
```python
def recover_data(symbol, target_file):
    """尝试恢复损坏的数据文件"""
    try:
        # 尝试读取备份
        backup_file = f"{target_file}.backup"
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, target_file)
            return True

        # 重新下载
        data = download_data(symbol)
        data.to_parquet(target_file)
        return True

    except Exception as e:
        print(f"数据恢复失败: {e}")
        return False
```
```

## 系统监控和警报

### 健康检查脚本
```python
import psutil
import shutil
from factor_system.factor_engine import api

def system_health_check():
    """系统健康检查"""
    health_report = {
        'timestamp': datetime.now().isoformat(),
        'status': 'healthy',
        'issues': []
    }

    # 检查内存使用
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        health_report['status'] = 'warning'
        health_report['issues'].append(f"内存使用过高: {memory.percent:.1f}%")

    # 检查磁盘空间
    disk = shutil.disk_usage('.')
    disk_usage = (disk.used / disk.total) * 100
    if disk_usage > 95:
        health_report['status'] = 'critical'
        health_report['issues'].append(f"磁盘空间不足: {disk_usage:.1f}%")

    # 检查FactorEngine
    try:
        cache_stats = api.get_cache_stats()
        if cache_stats['memory_hit_rate'] < 80:
            health_report['issues'].append(f"缓存命中率低: {cache_stats['memory_hit_rate']:.1f}%")
    except Exception as e:
        health_report['status'] = 'error'
        health_report['issues'].append(f"FactorEngine异常: {e}")

    return health_report

# 定期健康检查
if __name__ == "__main__":
    health = system_health_check()
    print(f"系统状态: {health['status']}")
    if health['issues']:
        print("发现问题:")
        for issue in health['issues']:
            print(f"  - {issue}")
```

### 日志分析
```python
import logging
import re
from collections import defaultdict

def analyze_log_file(log_file):
    """分析日志文件，识别常见问题"""
    error_patterns = {
        'memory_error': r'MemoryError|OutOfMemoryError',
        'file_not_found': r'FileNotFoundError|No such file',
        'network_error': r'ConnectionError|Timeout|Network',
        'data_error': r'ValueError|DataError|Corruption',
        'permission_error': r'PermissionError|Access denied'
    }

    error_counts = defaultdict(int)
    error_details = defaultdict(list)

    with open(log_file, 'r') as f:
        for line in f:
            for error_type, pattern in error_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    error_counts[error_type] += 1
                    if len(error_details[error_type]) < 5:  # 只保留前5个示例
                        error_details[error_type].append(line.strip())

    return {
        'error_counts': dict(error_counts),
        'error_details': dict(error_details)
    }
```

这个故障排除指南涵盖了环境配置、数据访问、因子计算、缓存系统、筛选系统、外部集成、性能问题、数据完整性等各个方面的问题诊断和解决方案，为用户提供了全面的问题排查参考。