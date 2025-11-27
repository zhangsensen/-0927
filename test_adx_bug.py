
import pandas as pd
import numpy as np

def test_adx_logic():
    # Create dummy data for 2 ETFs
    dates = pd.date_range('2023-01-01', periods=5)
    data = {
        'A': [10, 11, 12, 11, 13],
        'B': [20, 19, 18, 19, 20]
    }
    close_df = pd.DataFrame(data, index=dates)
    high_df = close_df + 1
    low_df = close_df - 1
    
    print("Close DF:")
    print(close_df)
    
    # Simulate the logic in _adx_14d_batch
    prev_close = close_df.shift(1)
    tr1 = high_df - low_df
    tr2 = (high_df - prev_close).abs()
    tr3 = (low_df - prev_close).abs()
    
    print("\nTR1:")
    print(tr1)
    
    # The suspicious block
    print("\n--- Suspicious Block ---")
    concatenated = pd.concat([tr1, tr2, tr3], axis=1)
    print("Concatenated (axis=1):")
    print(concatenated)
    
    tr_max = concatenated.max(axis=1)
    print("\nMax (axis=1):")
    print(tr_max)
    
    tr_final = tr_max.to_frame().reindex(columns=close_df.columns, fill_value=0)
    print("\nFinal TR (reindexed):")
    print(tr_final)
    
    # Correct logic (using np.maximum)
    print("\n--- Correct Logic ---")
    tr_correct = np.maximum(np.maximum(tr1, tr2), tr3)
    print(tr_correct)

if __name__ == "__main__":
    test_adx_logic()
