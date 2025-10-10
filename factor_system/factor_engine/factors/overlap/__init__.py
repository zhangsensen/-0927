"""移动平均与覆盖指标"""

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
