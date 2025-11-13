# ============================================
# slack_notifier.py
# ============================================
import os
import requests
import logging

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

def send_slack_message(text: str):
    """Slackにテキストを送信（Webhook版）"""
    if not SLACK_WEBHOOK_URL:
        logging.warning("⚠️ SLACK_WEBHOOK_URLが未設定のためSlack通知をスキップ")
        return
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={"text": str(text)})
        if resp.status_code != 200:
            logging.error(f"Slack送信失敗: {resp.status_code} {resp.text}")
    except Exception as e:
        logging.error(f"Slack通知中に例外発生: {e}")


def send_file_notification(file_path: str, title: str = None):
    """ファイルを直接送信せずに通知のみ行う"""
    if not os.path.exists(file_path):
        logging.warning(f"通知対象ファイルが存在しません: {file_path}")
        return
    name = os.path.basename(file_path)
    message = f"📁 実験結果ファイル出力: `{name}`"
    if title:
        message = f"*{title}*\n{message}"
    send_slack_message(message)

def send_file_notification(file_path: str, title: str = None):
    """実験結果ファイルをSlackに添付せずに通知（リンク付き）"""
    if not os.path.exists(file_path):
        logging.warning(f"ファイルが存在しません: {file_path}")
        return
    msg = f" *{title or '新しい実験結果'}* が出力されました\n`{os.path.basename(file_path)}`"
    send_slack_message(msg)
