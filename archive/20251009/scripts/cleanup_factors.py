#!/usr/bin/env python3
"""
清理FactorEngine中不符合一致性要求的因子
"""

import os
import shutil


def get_valid_factor_files():
    """获取符合一致性要求的因子文件清单"""

    # 基于factor_generation中实际存在的因子
    # 这些因子对应的FactorEngine文件应该保留
    valid_factor_files = {
        # technical目录 - 核心技术指标
        "rsi.py",
        "macd.py",
        "stoch.py",
        "willr.py",
        "atr.py",
        "cci.py",
        "mfi.py",
        "adx.py",
        "obv.py",
        "mom.py",
        "roc.py",
        "trix.py",
        "ultosc.py",
        "stochf.py",
        "stochrsi.py",
        "trange.py",
        "adxr.py",
        "apo.py",
        "aroon.py",
        "aroonosc.py",
        "bop.py",
        "cmo.py",
        "dx.py",
        "minus_di.py",
        "minus_dm.py",
        "natr.py",
        "plus_di.py",
        "plus_dm.py",
        "ppo.py",
        "rocp.py",
        "rocr.py",
        "rocr100.py",
        # overlap目录 - 移动平均类指标
        "sma.py",
        "ema.py",
        "bbands.py",
        "dema.py",
        "tema.py",
        "trima.py",
        "wma.py",
        "kama.py",
        "mama.py",
        "t3.py",
        "midpoint.py",
        "midprice.py",
        "sar.py",
        "sarext.py",
    }

    return valid_factor_files


def cleanup_factor_engine():
    """清理FactorEngine中不符合要求的因子"""

    base_path = "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/factors"
    valid_files = get_valid_factor_files()

    removed_files = []
    kept_files = []

    # 清理technical目录
    tech_path = os.path.join(base_path, "technical")
    if os.path.exists(tech_path):
        for file in os.listdir(tech_path):
            if file.endswith(".py") and file != "__init__.py":
                if file not in valid_files:
                    file_path = os.path.join(tech_path, file)
                    removed_files.append(file_path)
                    print(f"🗑️  删除technical: {file}")
                else:
                    kept_files.append(f"technical/{file}")

    # 清理overlap目录
    overlap_path = os.path.join(base_path, "overlap")
    if os.path.exists(overlap_path):
        for file in os.listdir(overlap_path):
            if file.endswith(".py") and file != "__init__.py":
                if file not in valid_files:
                    file_path = os.path.join(overlap_path, file)
                    removed_files.append(file_path)
                    print(f"🗑️  删除overlap: {file}")
                else:
                    kept_files.append(f"overlap/{file}")

    return removed_files, kept_files


def update_init_files():
    """更新__init__.py文件，只包含有效的因子"""

    # 更新technical/__init__.py
    tech_init_path = "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/factors/technical/__init__.py"
    valid_tech_imports = [
        "from factor_system.factor_engine.factors.technical.ad import AD",
        "from factor_system.factor_engine.factors.technical.adosc import ADOSC",
        "from factor_system.factor_engine.factors.technical.adx import ADX",
        "from factor_system.factor_engine.factors.technical.adxr import ADXR",
        "from factor_system.factor_engine.factors.technical.apo import APO",
        "from factor_system.factor_engine.factors.technical.aroon import AROON",
        "from factor_system.factor_engine.factors.technical.aroonosc import AROONOSC",
        "from factor_system.factor_engine.factors.technical.atr import ATR",
        "from factor_system.factor_engine.factors.technical.bop import BOP",
        "from factor_system.factor_engine.factors.technical.cci import CCI",
        "from factor_system.factor_engine.factors.technical.cmo import CMO",
        "from factor_system.factor_engine.factors.technical.dx import DX",
        "from factor_system.factor_engine.factors.technical.macd import MACD, MACDSignal, MACDHistogram",
        "from factor_system.factor_engine.factors.technical.mfi import MFI",
        "from factor_system.factor_engine.factors.technical.minus_di import MINUS_DI",
        "from factor_system.factor_engine.factors.technical.minus_dm import MINUS_DM",
        "from factor_system.factor_engine.factors.technical.mom import MOM",
        "from factor_system.factor_engine.factors.technical.natr import NATR",
        "from factor_system.factor_engine.factors.technical.obv import OBV",
        "from factor_system.factor_engine.factors.technical.plus_di import PLUS_DI",
        "from factor_system.factor_engine.factors.technical.plus_dm import PLUS_DM",
        "from factor_system.factor_engine.factors.technical.ppo import PPO",
        "from factor_system.factor_engine.factors.technical.roc import ROC",
        "from factor_system.factor_engine.factors.technical.rocp import ROCP",
        "from factor_system.factor_engine.factors.technical.rocr import ROCR",
        "from factor_system.factor_engine.factors.technical.rocr100 import ROCR100",
        "from factor_system.factor_engine.factors.technical.rsi import RSI",
        "from factor_system.factor_engine.factors.technical.stoch import STOCH",
        "from factor_system.factor_engine.factors.technical.stochf import STOCHF",
        "from factor_system.factor_engine.factors.technical.stochrsi import STOCHRSI",
        "from factor_system.factor_engine.factors.technical.trange import TRANGE",
        "from factor_system.factor_engine.factors.technical.trix import TRIX",
        "from factor_system.factor_engine.factors.technical.ultosc import ULTOSC",
        "from factor_system.factor_engine.factors.technical.willr import WILLR",
    ]

    # 更新overlap/__init__.py
    overlap_init_path = "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/factors/overlap/__init__.py"
    valid_overlap_imports = [
        "from factor_system.factor_engine.factors.overlap.bbands import BBANDS",
        "from factor_system.factor_engine.factors.overlap.dema import DEMA",
        "from factor_system.factor_engine.factors.overlap.ema import EMA",
        "from factor_system.factor_engine.factors.overlap.kama import KAMA",
        "from factor_system.factor_engine.factors.overlap.mama import MAMA",
        "from factor_system.factor_engine.factors.overlap.midpoint import MIDPOINT",
        "from factor_system.factor_engine.factors.overlap.midprice import MIDPRICE",
        "from factor_system.factor_engine.factors.overlap.sar import SAR",
        "from factor_system.factor_engine.factors.overlap.sarext import SAREXT",
        "from factor_system.factor_engine.factors.overlap.sma import SMA",
        "from factor_system.factor_engine.factors.overlap.t3 import T3",
        "from factor_system.factor_engine.factors.overlap.tema import TEMA",
        "from factor_system.factor_engine.factors.overlap.trima import TRIMA",
        "from factor_system.factor_engine.factors.overlap.wma import WMA",
    ]

    return valid_tech_imports, valid_overlap_imports


def write_new_init_files():
    """写入新的__init__.py文件"""

    # 写入technical/__init__.py
    tech_init_content = '''"""技术指标因子"""

from factor_system.factor_engine.factors.technical.ad import AD
from factor_system.factor_engine.factors.technical.adosc import ADOSC
from factor_system.factor_engine.factors.technical.adx import ADX
from factor_system.factor_engine.factors.technical.adxr import ADXR
from factor_system.factor_engine.factors.technical.apo import APO
from factor_system.factor_engine.factors.technical.aroon import AROON
from factor_system.factor_engine.factors.technical.aroonosc import AROONOSC
from factor_system.factor_engine.factors.technical.atr import ATR
from factor_system.factor_engine.factors.technical.bop import BOP
from factor_system.factor_engine.factors.technical.cci import CCI
from factor_system.factor_engine.factors.technical.cmo import CMO
from factor_system.factor_engine.factors.technical.dx import DX
from factor_system.factor_engine.factors.technical.macd import MACD, MACDSignal, MACDHistogram
from factor_system.factor_engine.factors.technical.mfi import MFI
from factor_system.factor_engine.factors.technical.minus_di import MINUS_DI
from factor_system.factor_engine.factors.technical.minus_dm import MINUS_DM
from factor_system.factor_engine.factors.technical.mom import MOM
from factor_system.factor_engine.factors.technical.natr import NATR
from factor_system.factor_engine.factors.technical.obv import OBV
from factor_system.factor_engine.factors.technical.plus_di import PLUS_DI
from factor_system.factor_engine.factors.technical.plus_dm import PLUS_DM
from factor_system.factor_engine.factors.technical.ppo import PPO
from factor_system.factor_engine.factors.technical.roc import ROC
from factor_system.factor_engine.factors.technical.rocp import ROCP
from factor_system.factor_engine.factors.technical.rocr import ROCR
from factor_system.factor_engine.factors.technical.rocr100 import ROCR100
from factor_system.factor_engine.factors.technical.rsi import RSI
from factor_system.factor_engine.factors.technical.stoch import STOCH
from factor_system.factor_engine.factors.technical.stochf import STOCHF
from factor_system.factor_engine.factors.technical.stochrsi import STOCHRSI
from factor_system.factor_engine.factors.technical.trange import TRANGE
from factor_system.factor_engine.factors.technical.trix import TRIX
from factor_system.factor_engine.factors.technical.ultosc import ULTOSC
from factor_system.factor_engine.factors.technical.willr import WILLR

__all__ = ['AD', 'ADOSC', 'ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'ATR', 'BOP', 'CCI', 'CMO', 'DX', 'MACD', 'MACDSignal', 'MACDHistogram', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'NATR', 'OBV', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'TRANGE', 'TRIX', 'ULTOSC', 'WILLR']
'''

    # 写入overlap/__init__.py
    overlap_init_content = '''"""移动平均与覆盖指标"""

from factor_system.factor_engine.factors.overlap.bbands import BBANDS
from factor_system.factor_engine.factors.overlap.dema import DEMA
from factor_system.factor_engine.factors.overlap.ema import EMA
from factor_system.factor_engine.factors.overlap.kama import KAMA
from factor_system.factor_engine.factors.overlap.mama import MAMA
from factor_system.factor_engine.factors.overlap.midpoint import MIDPOINT
from factor_system.factor_engine.factors.overlap.midprice import MIDPRICE
from factor_system.factor_engine.factors.overlap.sar import SAR
from factor_system.factor_engine.factors.overlap.sarext import SAREXT
from factor_system.factor_engine.factors.overlap.sma import SMA
from factor_system.factor_engine.factors.overlap.t3 import T3
from factor_system.factor_engine.factors.overlap.tema import TEMA
from factor_system.factor_engine.factors.overlap.trima import TRIMA
from factor_system.factor_engine.factors.overlap.wma import WMA

__all__ = ['BBANDS', 'DEMA', 'EMA', 'KAMA', 'MAMA', 'MIDPOINT', 'MIDPRICE', 'SAR', 'SAREXT', 'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA']
'''

    # 写入文件
    with open(
        "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/factors/technical/__init__.py",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(tech_init_content)

    with open(
        "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/factors/overlap/__init__.py",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(overlap_init_content)

    print("✅ 已更新technical/__init__.py和overlap/__init__.py")


def main():
    """主函数"""
    print("🔧 开始清理FactorEngine中不符合一致性要求的因子...")

    # 1. 分析需要删除的文件
    print("\n📋 分析要删除的文件...")
    removed_files, kept_files = cleanup_factor_engine()

    # 2. 实际删除文件
    print(f"\n🗑️  删除 {len(removed_files)} 个文件...")
    for file_path in removed_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  ✅ 已删除: {os.path.basename(file_path)}")

    # 3. 更新__init__.py文件
    print(f"\n📝 更新__init__.py文件...")
    write_new_init_files()

    # 4. 生成清理报告
    print(f"\n📊 清理报告:")
    print(f"  - 删除文件: {len(removed_files)} 个")
    print(f"  - 保留文件: {len(kept_files)} 个")
    print(f"  - 保留的文件:")
    for file in sorted(kept_files):
        print(f"    ✅ {file}")

    # 5. 保存报告
    with open(
        "/Users/zhangshenshen/深度量化0927/factor_engine_cleanup_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("FactorEngine一致性修复报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"删除的文件 ({len(removed_files)}个):\n")
        for file in removed_files:
            f.write(f"  - {os.path.basename(file)}\n")
        f.write(f"\n保留的文件 ({len(kept_files)}个):\n")
        for file in sorted(kept_files):
            f.write(f"  - {file}\n")
        f.write("\n修复完成: FactorEngine现在只包含factor_generation中存在的因子\n")

    print(f"\n✅ 清理完成! 报告已保存至: factor_engine_cleanup_report.txt")

    return removed_files, kept_files


if __name__ == "__main__":
    removed_files, kept_files = main()
