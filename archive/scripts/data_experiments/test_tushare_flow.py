
import tushare as ts
import pandas as pd
import os

def test_fund_flow():
    print("\n=== Testing Tushare Fund Flow ===")
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        print("TUSHARE_TOKEN not set")
        return

    ts.set_token(token)
    pro = ts.pro_api()

    # Try 510300.SH
    print("Fetching moneyflow for 510300.SH (20240101-20240110)")
    try:
        df = pro.moneyflow(ts_code='510300.SH', start_date='20240101', end_date='20240110')
        print(f"Result:\n{df}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_fund_flow()
