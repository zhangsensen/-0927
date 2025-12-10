
import pandas as pd
from pathlib import Path

def check_parquet_data():
    file_path = Path("/home/sensen/dev/projects/-0927/raw/ETF/daily/512010.SH_daily_20200102_20251014.parquet")
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"Reading {file_path}...")
    df = pd.read_parquet(file_path)
    
    # Check columns
    print("Columns:", df.columns.tolist())
    
    # Check specific date
    target_date = "20250605"
    # Ensure trade_date is string or convert if needed
    # Parquet might store it as int or string
    print(f"trade_date dtype: {df['trade_date'].dtype}")
    
    df['trade_date_str'] = df['trade_date'].astype(str)
        
    row = df[df['trade_date_str'] == target_date]
    
    if not row.empty:
        print(f"\nData for {target_date}:")
        print(row[['trade_date', 'open', 'high', 'low', 'close', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'vol']])
    else:
        print(f"\nNo data found for {target_date}")
        # Print surrounding dates
        print("\nSurrounding dates:")
        df['trade_date_dt'] = pd.to_datetime(df['trade_date_str'])
        target_dt = pd.to_datetime(target_date)
        surrounding = df[(df['trade_date_dt'] >= target_dt - pd.Timedelta(days=5)) & 
                         (df['trade_date_dt'] <= target_dt + pd.Timedelta(days=5))]
        print(surrounding[['trade_date', 'close', 'adj_close']])

if __name__ == "__main__":
    check_parquet_data()
