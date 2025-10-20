"""会话索引管理器 - 统一管理所有回测会话记录"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionIndexManager:
    """会话索引管理器 - 维护所有回测会话的中心化索引"""

    def __init__(self, backtest_results_dir: Path) -> None:
        self.backtest_results_dir = backtest_results_dir
        self.index_path = backtest_results_dir / "RUNS_INDEX.json"
        self.index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """加载现有索引"""
        if not self.index_path.exists():
            logger.info(f"索引文件不存在，创建新索引: {self.index_path}")
            return {}

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            logger.info(f"加载索引: {len(index)}个会话")
            return index
        except Exception as e:
            logger.warning(f"加载索引失败: {e}，返回空索引")
            return {}

    def _save_index(self) -> None:
        """保存索引到文件"""
        try:
            self.backtest_results_dir.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
            logger.debug(f"索引已保存: {len(self.index)}个会话")
        except Exception as e:
            logger.error(f"保存索引失败: {e}")

    def register_session(
        self,
        session_id: str,
        metadata: Dict[str, Any],
        overwrite: bool = False,
    ) -> None:
        """注册新会话到索引"""
        if session_id in self.index and not overwrite:
            logger.warning(f"会话已存在: {session_id}，跳过注册")
            return

        self.index[session_id] = {
            "registered_at": datetime.now().isoformat(),
            "session_dir": str(self.backtest_results_dir / session_id),
            **metadata,
        }
        self._save_index()
        logger.info(f"已注册会话: {session_id}")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取指定会话的元数据"""
        return self.index.get(session_id)

    def list_sessions(
        self,
        symbol: Optional[str] = None,
        session_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """列出会话，支持过滤和限制"""
        sessions = []

        for session_id, metadata in self.index.items():
            # 过滤
            if symbol and metadata.get("symbol") != symbol:
                continue
            if session_type and metadata.get("type") != session_type:
                continue

            sessions.append({"session_id": session_id, **metadata})

        # 按注册时间倒序排序
        sessions.sort(key=lambda x: x.get("registered_at", ""), reverse=True)

        # 限制数量
        if limit:
            sessions = sessions[:limit]

        return sessions

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """更新会话元数据"""
        if session_id not in self.index:
            logger.warning(f"会话不存在: {session_id}，无法更新")
            return

        self.index[session_id].update(updates)
        self._save_index()
        logger.debug(f"已更新会话: {session_id}")

    def remove_session(self, session_id: str) -> None:
        """从索引中移除会话（不删除实际文件）"""
        if session_id in self.index:
            del self.index[session_id]
            self._save_index()
            logger.info(f"已从索引移除会话: {session_id}")

    def rebuild_index_from_disk(self) -> int:
        """从磁盘扫描并重建索引"""
        logger.info("开始从磁盘重建索引...")
        rebuilt_count = 0

        for session_dir in self.backtest_results_dir.iterdir():
            if not session_dir.is_dir():
                continue

            session_id = session_dir.name

            # 跳过特殊目录
            if session_id.startswith(".") or session_id == "RUNS_INDEX.json":
                continue

            # 尝试读取summary.json
            summary_path = session_dir / "summary.json"
            metadata = {}

            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    metadata = {
                        "type": (
                            "combination"
                            if "total_combinations" in summary
                            else "multi_tf"
                        ),
                        "symbol": summary.get("symbol", "unknown"),
                        "timeframe": summary.get("timeframe", "unknown"),
                        "total_combinations": summary.get("total_combinations"),
                        "best_score": summary.get("best_score"),
                    }
                except Exception as e:
                    logger.warning(f"读取summary.json失败: {session_id}, {e}")

            # 从目录名解析基本信息
            parts = session_id.split("_")
            if not metadata.get("symbol") and len(parts) >= 1:
                metadata["symbol"] = parts[0].replace("_", ".")
            if not metadata.get("type"):
                if "combo" in session_id:
                    metadata["type"] = "combination"
                elif "multi" in session_id or "tf" in session_id:
                    metadata["type"] = "multi_tf"
                else:
                    metadata["type"] = "unknown"

            # 注册到索引
            self.register_session(session_id, metadata, overwrite=True)
            rebuilt_count += 1

        logger.info(f"索引重建完成: {rebuilt_count}个会话")
        return rebuilt_count

    def generate_summary_report(self) -> str:
        """生成索引摘要报告"""
        lines = [
            "=" * 80,
            "回测会话索引摘要",
            "=" * 80,
            f"总会话数: {len(self.index)}",
            "",
        ]

        # 按类型统计
        type_counts: Dict[str, int] = {}
        symbol_counts: Dict[str, int] = {}

        for metadata in self.index.values():
            session_type = metadata.get("type", "unknown")
            type_counts[session_type] = type_counts.get(session_type, 0) + 1

            symbol = metadata.get("symbol", "unknown")
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        lines.append("=== 会话类型分布 ===")
        for session_type, count in sorted(type_counts.items()):
            lines.append(f"  {session_type}: {count}个")

        lines.append("")
        lines.append("=== 标的分布 ===")
        for symbol, count in sorted(symbol_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {symbol}: {count}个")

        lines.append("")
        lines.append("=== 最近10次运行 ===")
        recent_sessions = self.list_sessions(limit=10)
        for session in recent_sessions:
            session_id = session["session_id"]
            session_type = session.get("type", "unknown")
            symbol = session.get("symbol", "unknown")
            registered_at = session.get("registered_at", "unknown")
            lines.append(f"  {session_id[:50]}")
            lines.append(
                f"    类型: {session_type}, 标的: {symbol}, 时间: {registered_at}"
            )

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)


def rebuild_session_index(backtest_results_dir: Path) -> None:
    """重建会话索引的便捷函数"""
    manager = SessionIndexManager(backtest_results_dir)
    count = manager.rebuild_index_from_disk()
    print(f"✅ 索引重建完成: {count}个会话")
    print(manager.generate_summary_report())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        from hk_midfreq.config import PathConfig

        config = PathConfig()
        rebuild_session_index(config.backtest_output_dir)
    else:
        print("使用方法: python -m hk_midfreq.session_index_manager rebuild")
