import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import talib

ROOT = Path("/home/sensen/dev/projects/-0927")


def load_data(code):
    data_dir = ROOT / "raw/ETF/daily"
    files = list(data_dir.glob(f"{code}*.parquet"))
    if not files:
        print(f"❌ {code} not found")
        return None
    df = pd.read_parquet(files[0])
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.set_index("date").sort_index()
    df["close"] = df["adj_close"]
    df["high"] = df["adj_high"]
    df["low"] = df["adj_low"]
    df["open"] = df["adj_open"]
    df["volume"] = df["vol"]
    return df


def calc_factors(df):
    # ADX_14D
    df["ADX_14D"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)

    # MAX_DD_60D
    roll_max = df["close"].rolling(60).max()
    dd = (df["close"] / roll_max) - 1
    df["MAX_DD_60D"] = dd.rolling(60).min()

    # PRICE_POSITION_120D
    roll_min_120 = df["close"].rolling(120).min()
    roll_max_120 = df["close"].rolling(120).max()
    df["PRICE_POSITION_120D"] = (df["close"] - roll_min_120) / (
        roll_max_120 - roll_min_120
    )

    # SHARPE_RATIO_20D
    ret = df["close"].pct_change()
    roll_mean = ret.rolling(20).mean() * 252
    roll_std = ret.rolling(20).std() * np.sqrt(252)
    df["SHARPE_RATIO_20D"] = roll_mean / roll_std

    # VOL_RATIO_60D
    vol_ma_60 = df["volume"].rolling(60).mean()
    vol_ma_5 = (
        df["volume"].rolling(5).mean()
    )  # Assuming short window is 5? Or just vol / ma60?
    # Usually VOL_RATIO is Volume / MA_Volume. Let's assume Vol / MA60
    df["VOL_RATIO_60D"] = df["volume"] / vol_ma_60

    return df


def main():
    codes = ["159949", "516090", "516160"]
    date_target = "2025-10-15"

    results = []

    for code in codes:
        print(f"Processing {code}...")
        df = load_data(code)
        if df is None:
            continue

        try:
            df = calc_factors(df)
        except Exception as e:
            print(f"Error calculating factors for {code}: {e}")
            continue

        if date_target in df.index:
            row = df.loc[date_target]
            res = {
                "code": code,
                "ADX_14D": row["ADX_14D"],
                "MAX_DD_60D": row["MAX_DD_60D"],
                "PRICE_POSITION_120D": row["PRICE_POSITION_120D"],
                "SHARPE_RATIO_20D": row["SHARPE_RATIO_20D"],
                "VOL_RATIO_60D": row["VOL_RATIO_60D"],
                "close": row["close"],
            }
            results.append(res)
        else:
            print(f"❌ Date {date_target} not found for {code}")

    res_df = pd.DataFrame(results)
    print(f"\n📊 Factor Values on {date_target}")
    print(res_df)

    # Simple Ranking (Higher is Better, except maybe MaxDD?)
    # Usually factors are standardized.
    # MAX_DD: Closer to 0 is better (higher value). e.g. -0.05 > -0.10. So Higher is Better.
    # ADX: Higher is Better (Strong Trend).
    # PRICE_POS: Higher is Better.
    # SHARPE: Higher is Better.
    # VOL_RATIO: Higher is Better (Volume Breakout)? Or Lower?
    # In trend following, usually High Volume is good.

    # Let's sum the ranks
    ranks = res_df.rank(numeric_only=True)
    print("\n🏆 Ranks (Higher is Better)")
    print(ranks)

    res_df["score"] = (
        ranks["ADX_14D"]
        + ranks["MAX_DD_60D"]
        + ranks["PRICE_POSITION_120D"]
        + ranks["SHARPE_RATIO_20D"]
        + ranks["VOL_RATIO_60D"]
    )
    print("\n🏁 Total Score")
    print(res_df[["code", "score"]].sort_values("score", ascending=False))


if __name__ == "__main__":
    main()
