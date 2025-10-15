#!/usr/bin/env python3
"""
通知处理器
支持钉钉 Webhook / 邮件通知
"""
import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class NotificationHandler:
    """通知处理器"""

    def __init__(self):
        # 从环境变量读取配置
        self.dingtalk_webhook = os.getenv("DINGTALK_WEBHOOK")
        self.email_config = {
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.example.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "sender": os.getenv("EMAIL_SENDER"),
            "password": os.getenv("EMAIL_PASSWORD"),
            "recipients": os.getenv("EMAIL_RECIPIENTS", "").split(","),
        }

    def send_dingtalk(self, title: str, content: str, level: str = "INFO"):
        """发送钉钉通知"""
        if not self.dingtalk_webhook:
            logger.warning("⚠️  未配置钉钉 Webhook，跳过")
            return False

        # 构造消息
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "SUCCESS": "✅"}.get(
            level, "📢"
        )

        message = {
            "msgtype": "markdown",
            "markdown": {
                "title": f"{emoji} {title}",
                "text": f"### {emoji} {title}\n\n{content}\n\n---\n\n**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            },
        }

        try:
            response = requests.post(
                self.dingtalk_webhook,
                headers={"Content-Type": "application/json"},
                data=json.dumps(message),
                timeout=5,
            )

            if response.status_code == 200:
                logger.info(f"✅ 钉钉通知已发送: {title}")
                return True
            else:
                logger.error(f"❌ 钉钉通知失败: {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ 钉钉通知异常: {e}")
            return False

    def send_email(self, subject: str, body: str):
        """发送邮件通知"""
        if not self.email_config["sender"] or not self.email_config["password"]:
            logger.warning("⚠️  未配置邮件，跳过")
            return False

        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = self.email_config["sender"]
            msg["To"] = ", ".join(self.email_config["recipients"])

            with smtplib.SMTP(
                self.email_config["smtp_server"], self.email_config["smtp_port"]
            ) as server:
                server.starttls()
                server.login(self.email_config["sender"], self.email_config["password"])
                server.send_message(msg)

            logger.info(f"✅ 邮件通知已发送: {subject}")
            return True

        except Exception as e:
            logger.error(f"❌ 邮件通知异常: {e}")
            return False

    def notify_failure(self, task: str, error_msg: str):
        """失败通知"""
        title = f"❌ {task} 失败"
        content = f"**任务**: {task}\n\n**错误**: {error_msg}"

        self.send_dingtalk(title, content, level="ERROR")
        self.send_email(title, content)

    def notify_success(self, task: str, summary: str = ""):
        """成功通知"""
        title = f"✅ {task} 完成"
        content = f"**任务**: {task}\n\n{summary}"

        self.send_dingtalk(title, content, level="SUCCESS")


class SnapshotManager:
    """快照管理器"""

    def __init__(self, snapshot_dir: str = "snapshots", max_snapshots: int = 10):
        self.snapshot_dir = Path(snapshot_dir)
        self.max_snapshots = max_snapshots
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self, source_dir: str, tag: str = None):
        """创建快照"""
        source = Path(source_dir)
        if not source.exists():
            logger.warning(f"⚠️  源目录不存在: {source}")
            return None

        # 生成快照名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = (
            f"snapshot_{tag}_{timestamp}" if tag else f"snapshot_{timestamp}"
        )
        snapshot_path = self.snapshot_dir / snapshot_name

        # 复制文件
        import shutil

        shutil.copytree(source, snapshot_path)

        logger.info(f"✅ 快照已创建: {snapshot_path}")

        # 清理旧快照
        self._cleanup_old_snapshots()

        return snapshot_path

    def _cleanup_old_snapshots(self):
        """清理旧快照（保留最近 N 个）"""
        snapshots = sorted(
            [d for d in self.snapshot_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if len(snapshots) > self.max_snapshots:
            for old_snapshot in snapshots[self.max_snapshots :]:
                import shutil

                shutil.rmtree(old_snapshot)
                logger.info(f"🗑️  已删除旧快照: {old_snapshot.name}")


def main():
    """测试通知"""
    handler = NotificationHandler()

    # 测试钉钉
    handler.send_dingtalk(
        title="测试通知",
        content="这是一条测试消息\n\n- 项目: FactorEngine\n- 状态: 正常",
        level="INFO",
    )

    # 测试快照
    snapshot_mgr = SnapshotManager(max_snapshots=5)
    snapshot_mgr.create_snapshot(
        source_dir="factor_output/etf_rotation_production", tag="test"
    )


if __name__ == "__main__":
    main()
