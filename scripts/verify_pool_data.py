#!/usr/bin/env python3
"""
Verify that all ETFs defined in etf_pools.yaml exist in the raw data directory.
"""
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def main():
    print("=" * 80)
    print("🔍 Verifying ETF Pool Data Availability")
    print("=" * 80)

    # 1. Load Config
    config_path = ROOT / "configs/etf_pools.yaml"
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 2. Check Data Directory
    raw_dir = ROOT / "raw/ETF/daily"
    if not raw_dir.exists():
        print(f"❌ Data directory not found: {raw_dir}")
        return

    # 3. Iterate Pools
    all_good = True
    total_etfs = 0
    missing_etfs = 0

    for pool_name, pool_info in config['pools'].items():
        print(f"\nChecking Pool: {pool_name} ({pool_info.get('name', 'Unknown')})")
        symbols = pool_info.get('symbols', [])
        
        for sym in symbols:
            total_etfs += 1
            # Check for parquet file
            # Pattern: {code}_daily_*.parquet
            # We'll just check if any file starts with the code
            found = False
            for f in raw_dir.glob(f"{sym}_daily_*.parquet"):
                found = True
                break
            
            if found:
                print(f"  ✅ {sym}: Found")
            else:
                print(f"  ❌ {sym}: MISSING in {raw_dir}")
                all_good = False
                missing_etfs += 1

    print("\n" + "=" * 80)
    if all_good:
        print(f"✅ All {total_etfs} ETFs found in data directory.")
    else:
        print(f"❌ Verification FAILED. {missing_etfs}/{total_etfs} ETFs missing.")
        print("Please update the download list or remove missing ETFs from the pool config.")

if __name__ == "__main__":
    main()
