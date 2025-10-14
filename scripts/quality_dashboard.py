#!/usr/bin/env python3
"""
量化交易系统代码质量仪表板
生成交互式质量报告和可视化图表
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import subprocess


class QualityDashboard:
    """代码质量仪表板生成器"""

    def __init__(self):
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.db_path = self.reports_dir / "quality_trends.db"

    def generate_html_dashboard(self, days: int = 30) -> str:
        """生成 HTML 质量仪表板"""
        # 获取趋势数据
        trends_data = self._get_trends_data(days)
        current_metrics = self._get_current_metrics()

        html_content = self._create_html_dashboard(trends_data, current_metrics, days)

        dashboard_file = self.reports_dir / f"quality_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(dashboard_file)

    def _get_trends_data(self, days: int) -> List[Dict]:
        """获取趋势数据"""
        if not self.db_path.exists():
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute('''
            SELECT timestamp, quality_score, complexity_issues, clone_issues,
                   dead_code_issues, high_risk_functions, total_clones, avg_complexity
            FROM quality_snapshots
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        ''', (cutoff_date,))

        rows = cursor.fetchall()
        conn.close()

        trends = []
        for row in rows:
            trends.append({
                "timestamp": row[0],
                "quality_score": row[1],
                "complexity_issues": row[2],
                "clone_issues": row[3],
                "dead_code_issues": row[4],
                "high_risk_functions": row[5],
                "total_clones": row[6],
                "avg_complexity": row[7]
            })

        return trends

    def _get_current_metrics(self) -> Dict[str, Any]:
        """获取当前质量指标"""
        try:
            cmd = ["pyscn", "analyze", "factor_system/", "--format", "json"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                return self._parse_pyscn_data(data)
            else:
                return {"error": "Failed to run pyscn analysis"}
        except Exception as e:
            return {"error": str(e)}

    def _parse_pyscn_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """解析 pyscn 数据"""
        metrics = {
            "total_functions": 0,
            "high_risk_functions": 0,
            "medium_risk_functions": 0,
            "total_clones": 0,
            "high_similarity_clones": 0,
            "dead_code_issues": 0,
            "avg_complexity": 0,
            "max_complexity": 0,
            "files_analyzed": set()
        }

        # 复杂度数据
        if "complexity" in data:
            complexity_data = data["complexity"]
            functions = complexity_data.get("functions", [])
            metrics["total_functions"] = len(functions)

            complexities = [f.get("complexity", 0) for f in functions]
            if complexities:
                metrics["avg_complexity"] = sum(complexities) / len(complexities)
                metrics["max_complexity"] = max(complexities)

            metrics["high_risk_functions"] = len([f for f in functions if f.get("complexity", 0) > 10])
            metrics["medium_risk_functions"] = len([f for f in functions if 5 < f.get("complexity", 0) <= 10])

            for f in functions:
                metrics["files_analyzed"].add(f.get("file", ""))

        # 克隆数据
        if "clones" in data:
            clones_data = data["clones"]
            clones = clones_data.get("clones", [])
            metrics["total_clones"] = len(clones)
            metrics["high_similarity_clones"] = len([c for c in clones if c.get("similarity", 0) > 0.9])

        # 死代码数据
        if "dead_code" in data:
            dead_code_data = data["dead_code"]
            metrics["dead_code_issues"] = len(dead_code_data.get("issues", []))

        metrics["files_analyzed"] = len(metrics["files_analyzed"])

        # 计算质量分数
        metrics["quality_score"] = self._calculate_quality_score(metrics)

        return metrics

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """计算质量分数"""
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

    def _create_html_dashboard(self, trends_data: List[Dict], current_metrics: Dict, days: int) -> str:
        """创建 HTML 仪表板"""
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化交易系统 - 代码质量仪表板</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
            transition: transform 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
        }}

        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}

        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .quality-excellent {{ color: #28a745; }}
        .quality-good {{ color: #ffc107; }}
        .quality-poor {{ color: #dc3545; }}

        .charts-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .chart-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        .chart-title {{
            font-size: 1.3em;
            margin-bottom: 20px;
            color: #333;
            text-align: center;
        }}

        .chart-canvas {{
            max-height: 300px;
        }}

        .issues-list {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        .issues-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #333;
        }}

        .issue-item {{
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #dc3545;
            background: #f8f9fa;
            border-radius: 5px;
        }}

        .issue-severity {{
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.8em;
            margin-bottom: 5px;
        }}

        .issue-description {{
            color: #666;
        }}

        .footer {{
            text-align: center;
            color: white;
            margin-top: 40px;
            opacity: 0.8;
        }}

        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            }}

            .charts-container {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏆 代码质量仪表板</h1>
            <p>量化交易系统 • 最近 {days} 天</p>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="metrics-grid">
            {self._generate_metric_cards(current_metrics)}
        </div>

        <div class="charts-container">
            <div class="chart-card">
                <div class="chart-title">📈 质量分数趋势</div>
                <canvas id="qualityChart" class="chart-canvas"></canvas>
            </div>

            <div class="chart-card">
                <div class="chart-title">📊 问题分布</div>
                <canvas id="issuesChart" class="chart-canvas"></canvas>
            </div>
        </div>

        <div class="charts-container">
            <div class="chart-card">
                <div class="chart-title">🎯 复杂度分析</div>
                <canvas id="complexityChart" class="chart-canvas"></canvas>
            </div>

            <div class="chart-card">
                <div class="chart-title">🔄 代码克隆趋势</div>
                <canvas id="clonesChart" class="chart-canvas"></canvas>
            </div>
        </div>

        <div class="issues-list">
            <div class="issues-title">🚨 主要质量问题</div>
            {self._generate_issues_list(current_metrics)}
        </div>

        <div class="footer">
            <p>📊 基于 pyscn 深度代码质量分析 • Control Flow Graph + APTED 算法</p>
        </div>
    </div>

    <script>
        // Chart.js 配置
        Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

        // 质量分数趋势图
        const qualityCtx = document.getElementById('qualityChart').getContext('2d');
        new Chart(qualityCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps([d['timestamp'][:10] for d in trends_data[-10:]])},
                datasets: [{{
                    label: '质量分数',
                    data: {json.dumps([d['quality_score'] for d in trends_data[-10:]])},
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});

        // 问题分布图
        const issuesCtx = document.getElementById('issuesChart').getContext('2d');
        new Chart(issuesCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['复杂度问题', '代码克隆', '死代码'],
                datasets: [{{
                    data: [
                        {current_metrics.get('high_risk_functions', 0) + current_metrics.get('medium_risk_functions', 0)},
                        {current_metrics.get('total_clones', 0)},
                        {current_metrics.get('dead_code_issues', 0)}
                    ],
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 206, 86, 0.8)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false
            }}
        }});

        // 复杂度分析图
        const complexityCtx = document.getElementById('complexityChart').getContext('2d');
        new Chart(complexityCtx, {{
            type: 'bar',
            data: {{
                labels: ['低风险', '中风险', '高风险'],
                datasets: [{{
                    label: '函数数量',
                    data: [
                        {current_metrics.get('total_functions', 0) - current_metrics.get('high_risk_functions', 0) - current_metrics.get('medium_risk_functions', 0)},
                        {current_metrics.get('medium_risk_functions', 0)},
                        {current_metrics.get('high_risk_functions', 0)}
                    ],
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(255, 206, 86, 0.8)',
                        'rgba(255, 99, 132, 0.8)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }}
            }}
        }});

        // 代码克隆趋势图
        const clonesCtx = document.getElementById('clonesChart').getContext('2d');
        new Chart(clonesCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps([d['timestamp'][:10] for d in trends_data[-10:]])},
                datasets: [{{
                    label: '代码克隆数',
                    data: {json.dumps([d['total_clones'] for d in trends_data[-10:]])},
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        return html

    def _generate_metric_cards(self, metrics: Dict) -> str:
        """生成指标卡片 HTML"""
        quality_score = metrics.get('quality_score', 0)
        quality_class = (
            'quality-excellent' if quality_score >= 80 else
            'quality-good' if quality_score >= 60 else
            'quality-poor'
        )

        cards = f"""
        <div class="metric-card">
            <div class="metric-value {quality_class}">{quality_score:.1f}</div>
            <div class="metric-label">质量分数</div>
        </div>

        <div class="metric-card">
            <div class="metric-value">{metrics.get('total_functions', 0)}</div>
            <div class="metric-label">函数总数</div>
        </div>

        <div class="metric-card">
            <div class="metric-value" style="color: #dc3545;">{metrics.get('high_risk_functions', 0)}</div>
            <div class="metric-label">高风险函数</div>
        </div>

        <div class="metric-card">
            <div class="metric-value" style="color: #ffc107;">{metrics.get('total_clones', 0)}</div>
            <div class="metric-label">代码克隆</div>
        </div>

        <div class="metric-card">
            <div class="metric-value">{metrics.get('avg_complexity', 0):.1f}</div>
            <div class="metric-label">平均复杂度</div>
        </div>

        <div class="metric-card">
            <div class="metric-value">{metrics.get('files_analyzed', 0)}</div>
            <div class="metric-label">分析文件数</div>
        </div>
        """
        return cards

    def _generate_issues_list(self, metrics: Dict) -> str:
        """生成问题列表 HTML"""
        issues = []

        if metrics.get('high_risk_functions', 0) > 0:
            issues.append(f"""
            <div class="issue-item">
                <div class="issue-severity" style="color: #dc3545;">高风险</div>
                <div class="issue-description">
                    发现 {metrics.get('high_risk_functions', 0)} 个高风险函数（复杂度 > 10）
                </div>
            </div>
            """)

        if metrics.get('total_clones', 0) > 5:
            issues.append(f"""
            <div class="issue-item">
                <div class="issue-severity" style="color: #ffc107;">中风险</div>
                <div class="issue-description">
                    发现 {metrics.get('total_clones', 0)} 个代码克隆，建议重构重复代码
                </div>
            </div>
            """)

        if metrics.get('dead_code_issues', 0) > 0:
            issues.append(f"""
            <div class="issue-item">
                <div class="issue-severity" style="color: #17a2b8;">信息</div>
                <div class="issue-description">
                    发现 {metrics.get('dead_code_issues', 0)} 处死代码，可以清理
                </div>
            </div>
            """)

        if not issues:
            issues.append("""
            <div class="issue-item" style="border-left-color: #28a745; background: #d4edda;">
                <div class="issue-severity" style="color: #28a745;">优秀</div>
                <div class="issue-description">
                    代码质量良好，未发现严重问题！
                </div>
            </div>
            """)

        return "".join(issues)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成代码质量仪表板")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="趋势分析天数 (默认: 30)"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="生成后自动打开仪表板"
    )

    args = parser.parse_args()

    dashboard = QualityDashboard()
    dashboard_file = dashboard.generate_html_dashboard(args.days)

    print(f"🎯 质量仪表板已生成: {dashboard_file}")

    if args.open:
        import webbrowser
        webbrowser.open(f"file://{Path.cwd()}/{dashboard_file}")


if __name__ == "__main__":
    main()