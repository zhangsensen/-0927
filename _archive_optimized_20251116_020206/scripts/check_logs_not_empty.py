#!/usr/bin/env python3
"""日志完整性检查脚本

功能:
1. 遍历 results/run_* 目录，检测必备产物是否存在：READY、top100_by_ic.parquet、all_combos.parquet、wfo_summary.json
2. 检查 factors/ 目录是否非空且因子数量 >= 5
3. 检查 real_backtest 下的基线摘要文件是否存在 (top200_baseline_partial_summary.json)
4. 汇总日志文件行数 (results/logs/*.log) 并判断是否为空
5. 输出机器可读 JSON + 人类可读表格

退出码:
- 0: 所有检查通过
- 1: 存在缺失项目

用法:
    python scripts/check_logs_not_empty.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime

BASE_RESULTS = Path("results")
REAL_BACKTEST_DIR = Path("real_backtest")
LOG_DIR = BASE_RESULTS / "logs"

REQUIRED_FILES = [
    "top100_by_ic.parquet",
    "all_combos.parquet",
    "wfo_summary.json",
]

def check_run(run_dir: Path) -> dict:
    ok = True
    notes = []
    ready = (run_dir / "READY").exists()
    if not ready:
        ok = False
        notes.append("READY missing")
    for rf in REQUIRED_FILES:
        if not (run_dir / rf).exists():
            ok = False
            notes.append(f"missing {rf}")
    factors_dir = run_dir / "factors"
    if not factors_dir.exists():
        ok = False
        notes.append("factors/ missing")
        n_factors = 0
    else:
        n_factors = len(list(factors_dir.glob("*.parquet")))
        if n_factors < 5:
            ok = False
            notes.append(f"too few factors ({n_factors})")
    return {
        "run": run_dir.name,
        "ready": ready,
        "n_factors": n_factors,
        "status": "OK" if ok else "FAIL",
        "issues": notes,
    }


def collect_logs() -> dict:
    if not LOG_DIR.exists():
        return {"log_dir": str(LOG_DIR), "exists": False, "total_lines": 0, "files": []}
    files = list(LOG_DIR.glob("*.log"))
    total_lines = 0
    file_meta = []
    for f in files:
        try:
            n = sum(1 for _ in f.open("r", encoding="utf-8", errors="ignore"))
        except Exception:
            n = -1
        total_lines += max(0, n)
        file_meta.append({"file": f.name, "lines": n})
    return {"log_dir": str(LOG_DIR), "exists": True, "total_lines": total_lines, "files": file_meta}


def main():
    runs = sorted([d for d in BASE_RESULTS.glob("run_*") if d.is_dir()])
    run_reports = [check_run(d) for d in runs]
    baseline_file = REAL_BACKTEST_DIR / "top200_baseline_partial_summary.json"
    baseline_exists = baseline_file.exists()

    logs_report = collect_logs()

    overall_ok = all(r["status"] == "OK" for r in run_reports) and baseline_exists and logs_report.get("total_lines",0) > 0

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "runs": run_reports,
        "baseline_summary_exists": baseline_exists,
        "logs": logs_report,
        "overall_status": "OK" if overall_ok else "FAIL",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 人类可读输出
    print("\n=== RUN CHECK SUMMARY ===")
    for r in run_reports:
        print(f"{r['run']}: {r['status']} factors={r['n_factors']} issues={','.join(r['issues']) if r['issues'] else '-'}")
    print(f"Baseline summary: {'OK' if baseline_exists else 'MISSING'}")
    print(f"Logs total lines: {logs_report.get('total_lines',0)} (files={len(logs_report.get('files',[]))})")
    if not overall_ok:
        print("\nFAIL: Some integrity checks failed. See JSON above.")
        sys.exit(1)
    else:
        print("\nOK: All integrity checks passed.")

if __name__ == "__main__":
    main()
