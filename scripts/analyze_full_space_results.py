
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load results
df = pd.read_csv('results/vec_from_wfo_20251212_004403/full_space_results.csv')

# Sort by Calmar
df_sorted = df.sort_values('vec_calmar_ratio', ascending=False)

# Top 100 Analysis
top_100 = df_sorted.head(100)
bottom_100 = df_sorted.tail(100)

def count_factors(df_subset):
    factors = []
    for combo in df_subset['combo']:
        factors.extend([f.strip() for f in combo.split('+')])
    return Counter(factors)

top_counts = count_factors(top_100)
bottom_counts = count_factors(bottom_100)

print("="*40)
print("TOP 100 FACTOR FREQUENCY")
print("="*40)
for f, c in top_counts.most_common(10):
    print(f"{f}: {c}%")

print("\n" + "="*40)
print("BOTTOM 100 FACTOR FREQUENCY")
print("="*40)
for f, c in bottom_counts.most_common(10):
    print(f"{f}: {c}%")

print("\n" + "="*40)
print("TOP 5 STRATEGIES")
print("="*40)
print(top_100[['combo', 'vec_return', 'vec_max_drawdown', 'vec_calmar_ratio']].head(5).to_string())
