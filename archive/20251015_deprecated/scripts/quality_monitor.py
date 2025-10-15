#!/usr/bin/env python3
"""
量化交易系统代码质量持续监控工具
用于跟踪代码质量变化趋势
"""

import argparse
import glob
import json
import os
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


class QualityMonitor:
    """代码质量监控器"""

    def __init__(self, db_path: str = "reports/quality_trends.db"):
        self.db_path = db_path
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS quality_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                git_commit TEXT,
                total_files INTEGER,
                complexity_issues INTEGER,
                clone_issues INTEGER,
                dead_code_issues INTEGER,
                high_risk_functions INTEGER,
                total_clones INTEGER,
                avg_complexity REAL,
                quality_score REAL
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS file_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER,
                file_path TEXT NOT NULL,
                complexity INTEGER,
                clones_count INTEGER,
                lines_of_code INTEGER,
                FOREIGN KEY (snapshot_id) REFERENCES quality_snapshots (id)
            )
        """
        )

        conn.commit()
        conn.close()

    def capture_snapshot(self, target_path: str = "factor_system/") -> Dict[str, Any]:
        """捕获当前代码质量快照"""
        print(f"📸 正在捕获 {target_path} 的质量快照...")

        # 获取当前 git commit
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True
            ).strip()
        except:
            git_commit = "unknown"

        # 运行 pyscn 分析
        try:
            cmd = ["pyscn", "analyze", target_path, "--json", "--no-open"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

            if result.returncode != 0:
                print(f"❌ pyscn 分析失败: {result.stderr}")
                return {"success": False}

            # pyscn 会生成 JSON 文件，需要读取最新的报告文件
            import glob

            json_files = glob.glob(".pyscn/reports/analyze_*.json")
            if json_files:
                latest_file = max(json_files, key=lambda x: os.path.getctime(x))
                with open(latest_file, "r") as f:
                    data = json.load(f)
                metrics = self._extract_metrics(data)
            else:
                print("❌ 找不到 pyscn 生成的 JSON 报告文件")
                return {"success": False}

            # 保存到数据库
            snapshot_id = self._save_snapshot(
                timestamp=datetime.now().isoformat(), git_commit=git_commit, **metrics
            )

            print(f"✅ 质量快照已保存 (ID: {snapshot_id})")
            return {"success": True, "snapshot_id": snapshot_id, "metrics": metrics}

        except Exception as e:
            print(f"❌ 捕获快照时出错: {e}")
            return {"success": False}

    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从 pyscn 数据中提取关键指标"""
        metrics = {
            "total_files": 0,
            "complexity_issues": 0,
            "clone_issues": 0,
            "dead_code_issues": 0,
            "high_risk_functions": 0,
            "total_clones": 0,
            "avg_complexity": 0.0,
            "quality_score": 0.0,
        }

        # 复杂度指标
        if "complexity" in data:
            complexity_data = data["complexity"]
            functions = complexity_data.get("functions", [])
            metrics["total_files"] = len(set(f.get("file", "") for f in functions))
            metrics["high_risk_functions"] = len(
                [f for f in functions if f.get("complexity", 0) > 10]
            )
            metrics["complexity_issues"] = len(
                [f for f in functions if f.get("complexity", 0) > 5]
            )

            if functions:
                metrics["avg_complexity"] = sum(
                    f.get("complexity", 0) for f in functions
                ) / len(functions)

        # 克隆指标
        if "clones" in data:
            clones_data = data["clones"]
            metrics["total_clones"] = len(clones_data.get("clones", []))
            metrics["clone_issues"] = len(clones_data.get("clones", []))

        # 死代码指标
        if "dead_code" in data:
            dead_code_data = data["dead_code"]
            metrics["dead_code_issues"] = len(dead_code_data.get("issues", []))

        # 计算质量分数 (0-100)
        metrics["quality_score"] = self._calculate_quality_score(metrics)

        return metrics

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """计算代码质量分数 (0-100)"""
        score = 100.0

        # 复杂度扣分
        if metrics.get("high_risk_functions", 0) > 0:
            score -= metrics["high_risk_functions"] * 5

        if metrics.get("avg_complexity", 0) > 10:
            score -= (metrics["avg_complexity"] - 10) * 2

        # 克隆扣分
        if metrics.get("total_clones", 0) > 0:
            score -= min(metrics["total_clones"] * 2, 20)

        # 死代码扣分
        if metrics.get("dead_code_issues", 0) > 0:
            score -= metrics["dead_code_issues"] * 3

        return max(0.0, min(100.0, score))

    def _save_snapshot(self, **kwargs) -> int:
        """保存快照到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO quality_snapshots
            (timestamp, git_commit, total_files, complexity_issues, clone_issues,
             dead_code_issues, high_risk_functions, total_clones, avg_complexity, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                kwargs["timestamp"],
                kwargs["git_commit"],
                kwargs["total_files"],
                kwargs["complexity_issues"],
                kwargs["clone_issues"],
                kwargs["dead_code_issues"],
                kwargs["high_risk_functions"],
                kwargs["total_clones"],
                kwargs["avg_complexity"],
                kwargs["quality_score"],
            ),
        )

        snapshot_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return snapshot_id

    def get_trends(self, days: int = 30) -> Dict[str, Any]:
        """获取质量趋势数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT * FROM quality_snapshots
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """,
            (cutoff_date,),
        )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"trends": [], "summary": "No data available"}

        trends = []
        for row in rows:
            trends.append(
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "git_commit": row[2][:8] if row[2] else "unknown",
                    "quality_score": row[9],
                    "complexity_issues": row[4],
                    "clone_issues": row[5],
                    "dead_code_issues": row[6],
                }
            )

        # 计算趋势
        if len(trends) >= 2:
            latest = trends[-1]
            earliest = trends[0]
            score_change = latest["quality_score"] - earliest["quality_score"]

            summary = f"质量分数变化: {score_change:+.1f} ({earliest['quality_score']:.1f} → {latest['quality_score']:.1f})"
        else:
            summary = "需要更多数据来分析趋势"

        return {"trends": trends, "summary": summary}

    def generate_trend_report(self, days: int = 30) -> str:
        """生成趋势报告"""
        data = self.get_trends(days)
        trends = data["trends"]

        if not trends:
            return "# 代码质量趋势报告\n\n暂无数据。\n"

        report = []
        report.append("# 代码质量趋势报告\n")
        report.append(f"**时间范围**: 最近 {days} 天\n")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**数据点数**: {len(trends)}\n")
        report.append(f"**趋势摘要**: {data['summary']}\n")

        if len(trends) >= 2:
            # 详细趋势分析
            report.append("## 📈 质量分数趋势\n")
            report.append(
                "| 日期 | Commit | 质量分数 | 复杂度问题 | 克隆问题 | 死代码 |\n"
            )
            report.append(
                "|------|--------|----------|------------|----------|--------|\n"
            )

            for trend in trends:
                date = datetime.fromisoformat(trend["timestamp"]).strftime(
                    "%m-%d %H:%M"
                )
                report.append(
                    f"| {date} | {trend['git_commit']} | {trend['quality_score']:.1f} | "
                    f"{trend['complexity_issues']} | {trend['clone_issues']} | "
                    f"{trend['dead_code_issues']} |\n"
                )

            # 趋势分析
            latest = trends[-1]
            earliest = trends[0]

            report.append("\n## 📊 趋势分析\n")

            score_trend = (
                "📈 上升"
                if latest["quality_score"] > earliest["quality_score"]
                else "📉 下降"
            )
            report.append(
                f"- **质量分数**: {score_trend} ({earliest['quality_score']:.1f} → {latest['quality_score']:.1f})"
            )

            complexity_trend = (
                "📈 减少"
                if latest["complexity_issues"] < earliest["complexity_issues"]
                else "📉 增加"
            )
            report.append(
                f"- **复杂度问题**: {complexity_trend} ({earliest['complexity_issues']} → {latest['complexity_issues']})"
            )

            clone_trend = (
                "📈 减少"
                if latest["clone_issues"] < earliest["clone_issues"]
                else "📉 增加"
            )
            report.append(
                f"- **代码克隆**: {clone_trend} ({earliest['clone_issues']} → {latest['clone_issues']})"
            )

            # 建议
            report.append("\n## 💡 改进建议\n")
            if latest["quality_score"] < 80:
                report.append(
                    "- 当前质量分数偏低，建议重点解决高复杂度函数和代码克隆问题"
                )
            if latest["complexity_issues"] > 10:
                report.append("- 复杂度问题较多，建议重构复杂函数")
            if latest["clone_issues"] > 5:
                report.append("- 代码克隆较多，建议提取公共逻辑")

        return "\n".join(report)

    def setup_monitoring(self, interval_hours: int = 24) -> None:
        """设置定时监控 (需要外部调度器支持)"""
        print(f"🔧 设置质量监控，间隔: {interval_hours} 小时")
        print("💡 建议使用 cron 或其他调度器定期运行:")
        print(
            f"   0 */{interval_hours} * * * cd {Path.cwd()} && python scripts/quality_monitor.py --capture"
        )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="代码质量监控工具")
    parser.add_argument("--capture", action="store_true", help="捕获当前质量快照")
    parser.add_argument("--trends", action="store_true", help="显示质量趋势")
    parser.add_argument("--days", type=int, default=30, help="趋势分析天数 (默认: 30)")
    parser.add_argument("--report", action="store_true", help="生成趋势报告")
    parser.add_argument(
        "--target", default="factor_system/", help="分析目标路径 (默认: factor_system/)"
    )

    args = parser.parse_args()

    monitor = QualityMonitor()

    if args.capture:
        monitor.capture_snapshot(args.target)
    elif args.trends:
        trends = monitor.get_trends(args.days)
        print(f"\n📈 质量趋势 ({args.days} 天):")
        print(trends["summary"])

        if trends["trends"]:
            print("\n最近5个数据点:")
            for trend in trends["trends"][-5:]:
                print(
                    f"  {trend['timestamp'][:16]} | {trend['git_commit']} | "
                    f"质量分数: {trend['quality_score']:.1f}"
                )
    elif args.report:
        report = monitor.generate_trend_report(args.days)
        report_file = (
            Path("reports")
            / f"quality_trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"📄 趋势报告已生成: {report_file}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
