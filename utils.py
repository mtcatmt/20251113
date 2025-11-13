# utils.py
import logging
import sys

def setup_logging(log_path: str):
    # 既存ハンドラを全削除（再設定時の重複防止）
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # --- ファイル出力 ---
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(file_handler)

    # --- コンソール出力（ターミナル） ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(console)
