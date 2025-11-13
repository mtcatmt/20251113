# auto_run_later.py
# python auto_run_later.py
import subprocess
import time
import datetime

target_time = "23:59"  # 実行したい時刻（24h形式）
script_path = r"C:/research/20251023/sensitivity_precision_recall_f1_copy.py"

def wait_until(target_hhmm):
    target = datetime.datetime.strptime(target_hhmm, "%H:%M").time()
    while True:
        now = datetime.datetime.now().time()
        if now >= target:
            break
        time.sleep(30)

print(f"Waiting until {target_time} to start...")
wait_until(target_time)
print("Start execution!")
subprocess.run(["python", script_path, "--workers", "8"])
