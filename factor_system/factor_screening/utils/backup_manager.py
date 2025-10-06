#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据备份管理器 - P3-3数据备份策略
提供自动备份、版本控制、恢复流程
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BackupManager:
    """数据备份管理器"""

    def __init__(
        self,
        backup_root: Path = Path("./backups"),
        max_backups: int = 10,
        retention_days: int = 30,
    ):
        """
        初始化备份管理器

        Args:
            backup_root: 备份根目录
            max_backups: 最大备份数量
            retention_days: 保留天数
        """
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)

        self.max_backups = max_backups
        self.retention_days = retention_days

        # 备份索引文件
        self.index_file = self.backup_root / "backup_index.json"
        self.backup_index = self._load_backup_index()

    def _load_backup_index(self) -> Dict[str, Any]:
        """加载备份索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载备份索引失败: {e}")
                return {"backups": []}
        return {"backups": []}

    def _save_backup_index(self) -> None:
        """保存备份索引"""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.backup_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存备份索引失败: {e}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """计算文件校验和"""
        sha256_hash = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"计算校验和失败 {file_path}: {e}")
            return ""

    def create_backup(
        self,
        source_path: Path,
        backup_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        创建备份

        Args:
            source_path: 源文件/目录路径
            backup_name: 备份名称（可选，默认使用时间戳）
            metadata: 备份元数据（可选）

        Returns:
            (是否成功, 备份ID或错误消息)
        """
        if not source_path.exists():
            return False, f"源路径不存在: {source_path}"

        # 生成备份ID和名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"backup_{timestamp}"
        backup_name = backup_name or source_path.name

        # 创建备份目录
        backup_dir = self.backup_root / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 复制文件/目录
            if source_path.is_file():
                dest_path = backup_dir / source_path.name
                shutil.copy2(source_path, dest_path)

                checksum = self._calculate_checksum(source_path)
                backup_type = "file"
                size_mb = source_path.stat().st_size / 1024 / 1024
            else:
                dest_path = backup_dir / source_path.name
                shutil.copytree(source_path, dest_path)

                checksum = ""  # 目录不计算单一校验和
                backup_type = "directory"

                # 计算目录总大小
                total_size = sum(
                    f.stat().st_size for f in dest_path.rglob("*") if f.is_file()
                )
                size_mb = total_size / 1024 / 1024

            # 记录备份信息
            backup_info = {
                "backup_id": backup_id,
                "backup_name": backup_name,
                "backup_type": backup_type,
                "source_path": str(source_path),
                "backup_path": str(backup_dir),
                "timestamp": timestamp,
                "iso_timestamp": datetime.now().isoformat(),
                "size_mb": size_mb,
                "checksum": checksum,
                "metadata": metadata or {},
            }

            # 更新备份索引
            self.backup_index["backups"].append(backup_info)
            self._save_backup_index()

            logger.info(f"✅ 备份创建成功: {backup_id} ({size_mb:.2f}MB)")

            # 检查并清理旧备份
            self._cleanup_old_backups()

            return True, backup_id

        except Exception as e:
            logger.error(f"创建备份失败: {e}")

            # 清理失败的备份
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)

            return False, str(e)

    def restore_backup(
        self, backup_id: str, restore_path: Optional[Path] = None
    ) -> Tuple[bool, str]:
        """
        恢复备份

        Args:
            backup_id: 备份ID
            restore_path: 恢复路径（可选，默认恢复到原位置）

        Returns:
            (是否成功, 恢复路径或错误消息)
        """
        # 查找备份信息
        backup_info = None
        for backup in self.backup_index["backups"]:
            if backup["backup_id"] == backup_id:
                backup_info = backup
                break

        if backup_info is None:
            return False, f"备份不存在: {backup_id}"

        backup_dir = Path(backup_info["backup_path"])

        if not backup_dir.exists():
            return False, f"备份目录不存在: {backup_dir}"

        try:
            # 确定恢复路径
            if restore_path is None:
                restore_path = Path(backup_info["source_path"])
            else:
                restore_path = Path(restore_path)

            # 如果目标已存在，创建临时备份
            temp_backup = None
            if restore_path.exists():
                temp_backup = restore_path.parent / f"{restore_path.name}.temp_backup"
                if restore_path.is_file():
                    shutil.copy2(restore_path, temp_backup)
                else:
                    shutil.copytree(restore_path, temp_backup)

                logger.info(f"创建临时备份: {temp_backup}")

            try:
                # 执行恢复
                backup_content = list(backup_dir.iterdir())[0]  # 备份内容

                if backup_info["backup_type"] == "file":
                    shutil.copy2(backup_content, restore_path)
                else:
                    if restore_path.exists():
                        shutil.rmtree(restore_path)
                    shutil.copytree(backup_content, restore_path)

                # 验证校验和（如果可用）
                if backup_info["checksum"] and backup_info["backup_type"] == "file":
                    restored_checksum = self._calculate_checksum(restore_path)
                    if restored_checksum != backup_info["checksum"]:
                        raise ValueError("校验和不匹配，数据可能已损坏")

                logger.info(f"✅ 备份恢复成功: {backup_id} → {restore_path}")

                # 删除临时备份
                if temp_backup and temp_backup.exists():
                    if temp_backup.is_file():
                        temp_backup.unlink()
                    else:
                        shutil.rmtree(temp_backup)

                return True, str(restore_path)

            except Exception:
                # 恢复失败，回滚到临时备份
                if temp_backup and temp_backup.exists():
                    logger.warning("恢复失败，回滚到临时备份")
                    if restore_path.exists():
                        if restore_path.is_file():
                            restore_path.unlink()
                        else:
                            shutil.rmtree(restore_path)

                    if temp_backup.is_file():
                        shutil.copy2(temp_backup, restore_path)
                    else:
                        shutil.copytree(temp_backup, restore_path)

                    # 清理临时备份
                    if temp_backup.is_file():
                        temp_backup.unlink()
                    else:
                        shutil.rmtree(temp_backup)

                raise

        except Exception as e:
            logger.error(f"恢复备份失败: {e}")
            return False, str(e)

    def list_backups(
        self, source_path: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        列出备份

        Args:
            source_path: 过滤源路径（可选）
            limit: 返回数量限制

        Returns:
            备份信息列表
        """
        backups = self.backup_index["backups"]

        # 过滤
        if source_path:
            backups = [b for b in backups if b["source_path"] == source_path]

        # 按时间戳倒序排序
        backups = sorted(backups, key=lambda x: x["timestamp"], reverse=True)

        return backups[:limit]

    def delete_backup(self, backup_id: str) -> Tuple[bool, str]:
        """
        删除备份

        Args:
            backup_id: 备份ID

        Returns:
            (是否成功, 消息)
        """
        # 查找备份信息
        backup_info = None
        backup_index = None

        for i, backup in enumerate(self.backup_index["backups"]):
            if backup["backup_id"] == backup_id:
                backup_info = backup
                backup_index = i
                break

        if backup_info is None:
            return False, f"备份不存在: {backup_id}"

        try:
            # 删除备份目录
            backup_dir = Path(backup_info["backup_path"])
            if backup_dir.exists():
                shutil.rmtree(backup_dir)

            # 从索引中移除
            del self.backup_index["backups"][backup_index]
            self._save_backup_index()

            logger.info(f"✅ 备份已删除: {backup_id}")
            return True, "备份删除成功"

        except Exception as e:
            logger.error(f"删除备份失败: {e}")
            return False, str(e)

    def _cleanup_old_backups(self) -> None:
        """清理旧备份"""
        backups = self.backup_index["backups"]

        # 按时间戳排序
        backups_sorted = sorted(backups, key=lambda x: x["timestamp"], reverse=True)

        # 计算需要删除的备份
        to_delete = []
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for i, backup in enumerate(backups_sorted):
            # 保留最近的max_backups个备份
            if i >= self.max_backups:
                to_delete.append(backup["backup_id"])
                continue

            # 删除超过保留期的备份
            backup_date = datetime.fromisoformat(backup["iso_timestamp"])
            if backup_date < cutoff_date:
                to_delete.append(backup["backup_id"])

        # 执行删除
        for backup_id in to_delete:
            self.delete_backup(backup_id)

    def get_backup_info(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """
        获取备份信息

        Args:
            backup_id: 备份ID

        Returns:
            备份信息字典或None
        """
        for backup in self.backup_index["backups"]:
            if backup["backup_id"] == backup_id:
                return backup
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取备份统计信息

        Returns:
            统计信息字典
        """
        backups = self.backup_index["backups"]

        total_size = sum(b.get("size_mb", 0) for b in backups)

        stats = {
            "total_backups": len(backups),
            "total_size_mb": total_size,
            "backup_root": str(self.backup_root),
            "max_backups": self.max_backups,
            "retention_days": self.retention_days,
            "oldest_backup": None,
            "newest_backup": None,
        }

        if backups:
            backups_sorted = sorted(backups, key=lambda x: x["timestamp"])
            stats["oldest_backup"] = backups_sorted[0]["iso_timestamp"]
            stats["newest_backup"] = backups_sorted[-1]["iso_timestamp"]

        return stats


# 全局备份管理器实例
_global_backup_manager: Optional[BackupManager] = None


def get_backup_manager(
    backup_root: Path = Path("./backups"),
    max_backups: int = 10,
    retention_days: int = 30,
) -> BackupManager:
    """
    获取全局备份管理器实例

    Args:
        backup_root: 备份根目录
        max_backups: 最大备份数量
        retention_days: 保留天数

    Returns:
        BackupManager实例
    """
    global _global_backup_manager
    if _global_backup_manager is None:
        _global_backup_manager = BackupManager(backup_root, max_backups, retention_days)
    return _global_backup_manager


# 便捷函数


def auto_backup(
    source_path: Path,
    backup_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """
    自动备份文件/目录

    Args:
        source_path: 源文件/目录路径
        backup_name: 备份名称（可选）
        metadata: 备份元数据（可选）

    Returns:
        (是否成功, 备份ID或错误消息)
    """
    manager = get_backup_manager()
    return manager.create_backup(source_path, backup_name, metadata)
