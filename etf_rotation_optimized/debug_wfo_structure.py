import pickle
import pprint
from pathlib import Path

# 加载WFO结果
wfo_path = Path("results/wfo/20251027_163940/wfo_results.pkl")
with open(wfo_path, "rb") as f:
    wfo_results = pickle.load(f)

print("WFO结果数据类型:", type(wfo_results))
print("\nWFO结果键（如果是字典）:")
if isinstance(wfo_results, dict):
    print(wfo_results.keys())
    print("\n前2个窗口结果示例:")
    window_results = wfo_results.get("window_results", [])
    for i, wr in enumerate(window_results[:2]):
        print(f"\n窗口{i}:")
        print(f"  类型: {type(wr)}")
        if isinstance(wr, dict):
            print(f"  键: {wr.keys()}")
            pprint.pprint(wr, depth=2)
        else:
            print(f"  属性: {dir(wr)}")
