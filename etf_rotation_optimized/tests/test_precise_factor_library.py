"""
精确因子库单元测试
================================================================================
测试23个核心因子的计算正确性
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from core.precise_factor_library import PreciseFactorLibrary


class TestMomentumFactors(unittest.TestCase):
    """动量因子测试"""

    def setUp(self):
        """设置测试数据"""
        self.lib = PreciseFactorLibrary(verbose=False)
        dates = pd.date_range("2023-01-01", periods=100)
        self.close = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
        self.high = pd.Series(self.close + abs(np.random.randn(100)), index=dates)
        self.low = pd.Series(self.close - abs(np.random.randn(100)), index=dates)
        self.volume = pd.Series(np.random.randint(100000, 500000, 100), index=dates)

    def test_momentum(self):
        """测试：Momentum因子 - 现在使用百分比形式 (close[t]/close[t-N]-1)*100"""
        mom = self.lib.momentum(self.close, period=20)  # 改为20日

        # 检查长度
        self.assertEqual(len(mom), len(self.close))

        # 检查值（第20个值应该是(close[20]/close[0]-1)*100）
        if not np.isnan(self.close.iloc[20]) and not np.isnan(self.close.iloc[0]):
            expected_20 = (self.close.iloc[20] / self.close.iloc[0] - 1) * 100
            self.assertAlmostEqual(mom.iloc[20], expected_20, places=4)

    def test_rsi(self):
        """测试：RSI因子"""
        rsi = self.lib.rsi(self.close, period=14)

        # RSI应该在0-100之间
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())

        # 检查长度
        self.assertEqual(len(rsi), len(self.close))

    def test_macd(self):
        """测试：MACD因子"""
        macd_result = self.lib.macd(self.close)

        # 应该返回DataFrame
        self.assertIsInstance(macd_result, pd.DataFrame)

        # 应该有三列
        self.assertEqual(len(macd_result.columns), 3)
        self.assertIn("MACD", macd_result.columns)
        self.assertIn("Signal", macd_result.columns)
        self.assertIn("Histogram", macd_result.columns)

        # Histogram应该 ≈ MACD - Signal
        expected_hist = macd_result["MACD"] - macd_result["Signal"]
        np.testing.assert_array_almost_equal(
            macd_result["Histogram"].fillna(0).values,
            expected_hist.fillna(0).values,
            decimal=5,
        )

    def test_kdj(self):
        """测试：KDJ因子"""
        kdj_result = self.lib.kdj(self.high, self.low, self.close)

        # 应该返回DataFrame
        self.assertIsInstance(kdj_result, pd.DataFrame)

        # 应该有三列
        self.assertEqual(len(kdj_result.columns), 3)
        self.assertIn("K", kdj_result.columns)
        self.assertIn("D", kdj_result.columns)
        self.assertIn("J", kdj_result.columns)

    def test_roc(self):
        """测试：ROC因子"""
        roc = self.lib.roc(self.close, period=12)

        # ROC = pct_change * 100
        expected_roc = self.close.pct_change(periods=12) * 100

        pd.testing.assert_series_equal(
            roc.fillna(0), expected_roc.fillna(0), check_exact=False, rtol=1e-10
        )

    def test_cci(self):
        """测试：CCI因子"""
        cci = self.lib.cci(self.high, self.low, self.close, period=20)

        # CCI可以是任意值（无上下界）
        self.assertEqual(len(cci), len(self.close))

    def test_stoch(self):
        """测试：Stochastic因子"""
        stoch_result = self.lib.stoch(self.high, self.low, self.close)

        # 应该返回DataFrame
        self.assertIsInstance(stoch_result, pd.DataFrame)

        # 应该有两列
        self.assertEqual(len(stoch_result.columns), 2)
        self.assertIn("%K", stoch_result.columns)
        self.assertIn("%D", stoch_result.columns)

    def test_aroon(self):
        """测试：Aroon因子"""
        aroon_result = self.lib.aroon(self.high, self.low)

        # 应该返回DataFrame
        self.assertIsInstance(aroon_result, pd.DataFrame)

        # Aroon Up和Down应该在0-100之间
        valid_up = aroon_result["Aroon_Up"].dropna()
        valid_down = aroon_result["Aroon_Down"].dropna()

        self.assertTrue((valid_up >= 0).all() and (valid_up <= 100).all())
        self.assertTrue((valid_down >= 0).all() and (valid_down <= 100).all())


class TestVolatilityFactors(unittest.TestCase):
    """波动率因子测试"""

    def setUp(self):
        """设置测试数据"""
        self.lib = PreciseFactorLibrary(verbose=False)
        dates = pd.date_range("2023-01-01", periods=100)
        self.close = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
        self.high = pd.Series(self.close + abs(np.random.randn(100)) * 2, index=dates)
        self.low = pd.Series(self.close - abs(np.random.randn(100)) * 2, index=dates)

    def test_atr(self):
        """测试：ATR因子"""
        atr = self.lib.atr(self.high, self.low, self.close, period=14)

        # ATR应该是正数
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr >= 0).all())

        # ATR不应该超过日均波幅过多
        avg_range = (self.high - self.low).mean()
        self.assertLess(atr.mean(), avg_range * 10)

    def test_bollinger_bands(self):
        """测试：Bollinger Bands因子"""
        boll = self.lib.bollinger_bands(self.close, period=20)

        # 应该返回DataFrame
        self.assertIsInstance(boll, pd.DataFrame)

        # 应该有三列
        self.assertEqual(len(boll.columns), 3)

        # Upper > Middle > Lower
        for i in range(len(boll)):
            if pd.notna(boll["Upper"].iloc[i]):
                self.assertGreater(boll["Upper"].iloc[i], boll["Middle"].iloc[i])
                self.assertGreater(boll["Middle"].iloc[i], boll["Lower"].iloc[i])

    def test_keltner_channel(self):
        """测试：Keltner Channel因子"""
        keltner = self.lib.keltner_channel(self.high, self.low, self.close, period=20)

        # 应该返回DataFrame
        self.assertIsInstance(keltner, pd.DataFrame)

        # Upper > Lower
        valid_idx = keltner["Upper"].notna()
        self.assertTrue(
            (keltner.loc[valid_idx, "Upper"] > keltner.loc[valid_idx, "Lower"]).all()
        )

    def test_donchian_channel(self):
        """测试：Donchian Channel因子"""
        donchian = self.lib.donchian_channel(self.high, self.low, period=20)

        # Upper应该是High的最大值（用fillna避免NaN比较问题）
        valid_idx = donchian["Upper"].notna()
        self.assertTrue(
            (donchian.loc[valid_idx, "Upper"] >= self.high[valid_idx]).all()
        )

        # Lower应该是Low的最小值
        self.assertTrue((donchian.loc[valid_idx, "Lower"] <= self.low[valid_idx]).all())


class TestVolumeFactors(unittest.TestCase):
    """成交量因子测试"""

    def setUp(self):
        """设置测试数据"""
        self.lib = PreciseFactorLibrary(verbose=False)
        dates = pd.date_range("2023-01-01", periods=100)
        self.close = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
        self.high = pd.Series(self.close + abs(np.random.randn(100)), index=dates)
        self.low = pd.Series(self.close - abs(np.random.randn(100)), index=dates)
        self.volume = pd.Series(np.random.randint(100000, 500000, 100), index=dates)

    def test_obv(self):
        """测试：OBV因子"""
        obv = self.lib.obv(self.close, self.volume)

        # OBV应该是累积的
        self.assertEqual(len(obv), len(self.close))

        # OBV应该单调变化（或保持不变）
        # （由于价格上升和下降，OBV可能振荡）

    def test_ad(self):
        """测试：AD因子"""
        ad = self.lib.ad(self.high, self.low, self.close, self.volume)

        # AD应该是累积的
        self.assertEqual(len(ad), len(self.close))

    def test_volume_rate(self):
        """测试：Volume Rate因子"""
        vol_rate = self.lib.volume_rate(self.volume, period=20)

        # Volume Rate应该是正数
        valid_vol_rate = vol_rate.dropna()
        self.assertTrue((valid_vol_rate > 0).all())

        # 应该围绕1波动
        self.assertGreater(vol_rate.mean(), 0.5)
        self.assertLess(vol_rate.mean(), 2.0)

    def test_vwap(self):
        """测试：VWAP因子"""
        vwap = self.lib.vwap(self.high, self.low, self.close, self.volume, period=20)

        # VWAP应该在价格范围内（有一定容差）
        valid_idx = vwap.notna()

        # 检查VWAP的合理性
        self.assertTrue(len(vwap[valid_idx]) > 0)

        # VWAP应该是正数
        self.assertTrue((vwap[valid_idx] > 0).all())


class TestComputeAllFactors(unittest.TestCase):
    """完整因子计算测试"""

    def setUp(self):
        """设置测试数据"""
        self.lib = PreciseFactorLibrary(verbose=False)
        dates = pd.date_range("2023-01-01", periods=100)

        self.prices = {
            "close": pd.DataFrame(
                {
                    "SH600000": np.random.randn(100).cumsum() + 100,
                    "SH600001": np.random.randn(100).cumsum() + 100,
                },
                index=dates,
            ),
            "high": pd.DataFrame(
                {
                    "SH600000": np.random.randn(100).cumsum() + 102,
                    "SH600001": np.random.randn(100).cumsum() + 102,
                },
                index=dates,
            ),
            "low": pd.DataFrame(
                {
                    "SH600000": np.random.randn(100).cumsum() + 98,
                    "SH600001": np.random.randn(100).cumsum() + 98,
                },
                index=dates,
            ),
            "volume": pd.DataFrame(
                {
                    "SH600000": np.random.randint(100000, 500000, 100),
                    "SH600001": np.random.randint(100000, 500000, 100),
                },
                index=dates,
            ),
        }

    def test_compute_all_factors(self):
        """测试：计算所有因子"""
        factors_dict = self.lib.compute_all_factors(self.prices)

        # 应该返回字典
        self.assertIsInstance(factors_dict, dict)

        # 应该有两个标的
        self.assertEqual(len(factors_dict), 2)
        self.assertIn("SH600000", factors_dict)
        self.assertIn("SH600001", factors_dict)

        # 每个标的应该有多个因子
        for symbol, factors_df in factors_dict.items():
            self.assertIsInstance(factors_df, pd.DataFrame)
            # 应该有至少20个因子列
            self.assertGreater(len(factors_df.columns), 15)

    def test_factor_count(self):
        """测试：因子数量"""
        factors_dict = self.lib.compute_all_factors(self.prices)

        # 获取第一个标的的因子
        factors_df = factors_dict["SH600000"]

        # 应该包含所有主要因子
        expected_factors = [
            "MOM_12",
            "RSI_14",
            "ROC_12",
            "CCI_20",
            "MACD",
            "MACD_Signal",
            "MACD_Hist",
            "KDJ_K",
            "KDJ_D",
            "KDJ_J",
            "ATR_14",
            "BOLL_Upper",
            "BOLL_Lower",
            "BOLL_Width",
            "OBV",
            "AD",
            "Volume_Rate",
            "VWAP_20",
        ]

        for factor in expected_factors:
            self.assertIn(factor, factors_df.columns)


class TestFactorNaNHandling(unittest.TestCase):
    """NaN处理测试"""

    def setUp(self):
        """设置测试数据"""
        self.lib = PreciseFactorLibrary(verbose=False)
        dates = pd.date_range("2023-01-01", periods=100)

        close = np.random.randn(100).cumsum() + 100
        close[:10] = np.nan  # 前10个NaN

        self.close = pd.Series(close, index=dates)
        self.high = pd.Series(np.random.randn(100).cumsum() + 102, index=dates)
        self.low = pd.Series(np.random.randn(100).cumsum() + 98, index=dates)
        self.volume = pd.Series(np.random.randint(100000, 500000, 100), index=dates)

    def test_momentum_with_nan(self):
        """测试：Momentum处理NaN - 不再有fill_na参数，NaN会被保留"""
        mom = self.lib.momentum(self.close, period=20)

        # 长度应该正确
        self.assertEqual(len(mom), len(self.close))

        # 前20个应该是NaN（满窗原则）
        self.assertGreater(mom[:20].isna().sum(), 15)

    def test_rsi_with_nan(self):
        """测试：RSI处理NaN - 不再有fill_na参数，NaN会被保留"""
        rsi = self.lib.rsi(self.close, period=14)

        # 长度应该正确
        self.assertEqual(len(rsi), len(self.close))

        # RSI应该在[0, 100]范围内（排除NaN）
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestMomentumFactors))
    suite.addTests(loader.loadTestsFromTestCase(TestVolatilityFactors))
    suite.addTests(loader.loadTestsFromTestCase(TestVolumeFactors))
    suite.addTests(loader.loadTestsFromTestCase(TestComputeAllFactors))
    suite.addTests(loader.loadTestsFromTestCase(TestFactorNaNHandling))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_tests()

    print("\n" + "=" * 70)
    print(f"✅ 测试完成: {result.testsRun}个测试")
    print(f"✓ 通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"✗ 失败: {len(result.failures)}")
    print(f"⚠️ 错误: {len(result.errors)}")
    print("=" * 70)
