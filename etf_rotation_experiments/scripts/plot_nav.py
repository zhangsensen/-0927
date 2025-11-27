import json
import sys
import numpy as np

def print_ascii_chart(series, height=15, width=60):
    min_val = min(series)
    max_val = max(series)
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1
    
    normalized = [(x - min_val) / range_val for x in series]
    
    # Downsample to width
    step = len(series) / width
    downsampled = []
    for i in range(width):
        start = int(i * step)
        end = int((i + 1) * step)
        chunk = normalized[start:end]
        if chunk:
            downsampled.append(sum(chunk) / len(chunk))
        else:
            downsampled.append(downsampled[-1] if downsampled else 0)
            
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    for x, y_norm in enumerate(downsampled):
        y = int(y_norm * (height - 1))
        grid[height - 1 - y][x] = '*'
        
    print(f"NAV Chart (Min: {min_val:.2f}, Max: {max_val:.2f})")
    print("-" * (width + 2))
    for row in grid:
        print("|" + "".join(row) + "|")
    print("-" * (width + 2))

def main():
    file_path = sys.argv[1]
    with open(file_path, 'r') as f:
        nav = json.load(f)
    
    print(f"Loaded NAV series with {len(nav)} points")
    print(f"Start: {nav[0]:.4f}, End: {nav[-1]:.4f}")
    print(f"Total Return: {(nav[-1]/nav[0] - 1)*100:.2f}%")
    
    print_ascii_chart(nav)
    
    # Yearly returns approximation (assuming 252 days/year)
    print("\nApproximate Yearly Returns (Recent to Oldest):")
    days_per_year = 252
    total_days = len(nav)
    
    # Split into chunks of 252 days from the end
    chunks = []
    for i in range(total_days, 0, -days_per_year):
        start = max(0, i - days_per_year)
        end = i
        chunks.append(nav[start:end])
    
    for i, chunk in enumerate(chunks):
        if not chunk: continue
        ret = (chunk[-1] / chunk[0]) - 1
        print(f"Year -{i} (Days {len(chunk)}): {ret*100:.2f}%")

if __name__ == "__main__":
    main()
