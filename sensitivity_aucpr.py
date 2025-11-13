# ==========================================================
# sensitivity_aucpr.py（完全版）
# ==========================================================
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import setup_logging
from config import SIMULATION_CONFIG, GLOBAL_SEED
from evaluation import auc_pr_score
from data_generator import generate_artificial_data, create_dataset_iv_logit_score
from analysis import run_block_bootstrap_lingam
from pebble import ProcessPool
from concurrent.futures import as_completed, TimeoutError as PebbleTimeoutError
from slack_notifier import send_slack_message

# ==========================================================
# 🎯 感度分析対象
# ==========================================================
sensitivity_targets = {
    "N_VARS": [2, 3, 4, 6, 8],
    "N_SAMPLES": [500, 1000, 1500, 2000, 3000],
    "EDGE_PROB": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    "BOOTSTRAP_SAMPLES": [10, 25, 50, 100, 200],
    "BLOCK_SIZE": [25, 50, 75, 100, 200],
    "LAG": [1, 2, 3, 4],
    
}

output_root = "sensitivity_results_multi"
workers = os.cpu_count()

# ----------------------------------------------------------
def run_single_simulation_aucpr(sim_id: int, config: dict, seed: int):
    np.random.seed(seed)
    try:
        data_ts_cont, data_ts_mixed, B_true, base_names, lag, disc_names = generate_artificial_data(
            n_vars=config["N_VARS"],
            n_samples=config["N_SAMPLES"],
            lag=config["LAG"],
            seed=seed,
            edge_prob=config["EDGE_PROB"],
            
        )

        def build_edge_prob(df_edge, labels):
            idx = {lbl: i for i, lbl in enumerate(labels)}
            mat = np.zeros((len(labels), len(labels)))
            for _, r in df_edge.iterrows():
                f, t = idx.get(r["From"]), idx.get(r["To"])
                if f is not None and t is not None:
                    mat[t, f] = r["Probability"]
            return mat

        df_c, lbl_c = run_block_bootstrap_lingam(data_ts_cont, config["BLOCK_SIZE"], lag, base_names, config["BOOTSTRAP_SAMPLES"], seed)
        df_m, lbl_m = run_block_bootstrap_lingam(data_ts_mixed, config["BLOCK_SIZE"], lag, base_names, config["BOOTSTRAP_SAMPLES"], seed)
        data_ts_logit = create_dataset_iv_logit_score(data_ts_mixed, B_true, base_names, disc_names, lbl_m, lag)
        df_l, lbl_l = run_block_bootstrap_lingam(data_ts_logit, config["BLOCK_SIZE"], lag, base_names, config["BOOTSTRAP_SAMPLES"], seed)

        return {
            "ok": True,
            "AUC_PR_Continuous": auc_pr_score(B_true, build_edge_prob(df_c, lbl_c)),
            "AUC_PR_Mixed": auc_pr_score(B_true, build_edge_prob(df_m, lbl_m)),
            "AUC_PR_Logit": auc_pr_score(B_true, build_edge_prob(df_l, lbl_l)),
        }
    except Exception as e:
        logging.error(f"[Sim {sim_id}] {e}")
        return {"ok": False}

# ----------------------------------------------------------
def run_parallel_aucpr(config, n_workers, seeds):
    timeout = config.get("SIMULATION_TIMEOUT", 600)
    results = []
    with ProcessPool(max_workers=n_workers) as pool:
        futures = {pool.schedule(run_single_simulation_aucpr, args=[i, config, seeds[i]], timeout=timeout): i
                   for i in range(config["N_SIMULATIONS"])}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Simulations"):
            try:
                r = f.result()
                if isinstance(r, dict) and r.get("ok"):
                    results.append(r)
            except PebbleTimeoutError:
                logging.warning("Timeout")
    return results

# ----------------------------------------------------------
def format_config_str(config: dict, exclude_keys: list) -> str:
    """ファイル名用に、ベースパラメータを簡潔に連結する"""
    key_map = {
        "N_SIMULATIONS": "sim",
        "N_SAMPLES": "s",
        "LAG": "l",
        "N_VARS": "v",
        "EDGE_PROB": "edge",
        "BLOCK_SIZE": "blk",
        "BOOTSTRAP_SAMPLES": "bt",
        "PROBABILITY_THRESHOLD": "th",
        "DISCRETE_PARENT_MODE": "dp",
    }
    parts = []
    for k, v in config.items():
        if k in exclude_keys:
            continue
        short = key_map.get(k, k.lower())
        val = str(v).replace(".", "_")
        parts.append(f"{short}{val}")
    return "_".join(parts)

# ----------------------------------------------------------
def run_all_sensitivity():
    os.makedirs(output_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_root, f"sensitivity_multi_{timestamp}.log")
    setup_logging(log_path)

    base_config = SIMULATION_CONFIG.copy()
    np.random.seed(GLOBAL_SEED)
    overall_summary = []

    # --- 実行開始時: 基本設定をSlackに送信 ---
    base_info = "\n".join([f"{k}: {v}" for k, v in base_config.items()])
    send_slack_message(
        f"🚀 複数パラメータ感度分析を開始 ({len(sensitivity_targets)} セット)\n"
        f"使用コア数: {workers}\n"
        f"基本設定:\n```\n{base_info}\n```"
    )

    # --- 各パラメータで感度分析 ---
    for param, values in sensitivity_targets.items():
        send_slack_message(f"🔹 感度分析開始: *{param}* = {values}")
        subdir = os.path.join(output_root, param)
        os.makedirs(subdir, exist_ok=True)

        summaries = []
        for v in values:
            send_slack_message(f"⚙️ {param} = {v} の分析を開始します…")
            cfg = base_config.copy()
            cfg[param] = int(v) if isinstance(v, (int, np.integer)) else float(v)
            seeds = [np.random.randint(0, 2**31 - 1) for _ in range(cfg["N_SIMULATIONS"])]

            res = run_parallel_aucpr(cfg, workers, seeds)
            if not res:
                send_slack_message(f"⚠️ {param}={v}: 結果が空でした")
                continue

            df = pd.DataFrame(res)
            summary = {
                "value": v,
                "AUC_PR_Continuous_mean": df["AUC_PR_Continuous"].mean(),
                "AUC_PR_Continuous_std": df["AUC_PR_Continuous"].std(ddof=1),
                "AUC_PR_Mixed_mean": df["AUC_PR_Mixed"].mean(),
                "AUC_PR_Mixed_std": df["AUC_PR_Mixed"].std(ddof=1),
                "AUC_PR_Logit_mean": df["AUC_PR_Logit"].mean(),
                "AUC_PR_Logit_std": df["AUC_PR_Logit"].std(ddof=1),
            }
            summaries.append(summary)
            overall_summary.append({"param": param, **summary})

            msg = (
                f"📊 {param}={v} の中間結果\n"
                f"C={summary['AUC_PR_Continuous_mean']:.3f}, "
                f"M={summary['AUC_PR_Mixed_mean']:.3f}, "
                f"L={summary['AUC_PR_Logit_mean']:.3f}"
            )
            send_slack_message(msg)

        # --- 感度分析1セット完了時: Slackに表で送信 ---
        if summaries:
            df_sum = pd.DataFrame(summaries).sort_values("value")

            header = (
                f"{'Param':<10}"
                f"{'C_mean':>10}{'C_std':>10}"
                f"{'M_mean':>10}{'M_std':>10}"
                f"{'L_mean':>10}{'L_std':>10}"
            )
            table_lines = [header, "-" * len(header)]
            for _, r in df_sum.iterrows():
                def fmt(x): return f"{x:.3f}" if pd.notna(x) else "nan"
                table_lines.append(
                    f"{r['value']:<10.2f}"
                    f"{fmt(r['AUC_PR_Continuous_mean']):>10}{fmt(r['AUC_PR_Continuous_std']):>10}"
                    f"{fmt(r['AUC_PR_Mixed_mean']):>10}{fmt(r['AUC_PR_Mixed_std']):>10}"
                    f"{fmt(r['AUC_PR_Logit_mean']):>10}{fmt(r['AUC_PR_Logit_std']):>10}"
                )
            table_text = "\n".join(table_lines)

            slack_message = (
                f"✅ *{param}* の感度分析完了\n"
                f"対象値: {values}\n"
                f"総シミュレーション: {cfg['N_SIMULATIONS']}回\n"
                "```\n"
                f"{table_text}\n"
                "```"
            )
            send_slack_message(slack_message)

            # --- ファイル名拡張 ---
            base_tag = format_config_str(cfg, exclude_keys=[param])
            csv_name = f"{param}_summary_{base_tag}.csv"
            plot_name = f"{param}_plot_{base_tag}.png"

            # 保存
            df_sum.to_csv(os.path.join(subdir, csv_name), index=False)
            plt.figure(figsize=(7,4))
            plt.plot(df_sum["value"], df_sum["AUC_PR_Continuous_mean"], "o-", label="Continuous")
            plt.plot(df_sum["value"], df_sum["AUC_PR_Mixed_mean"], "s-", label="Mixed")
            plt.plot(df_sum["value"], df_sum["AUC_PR_Logit_mean"], "^-", label="Logit")
            plt.xlabel(param)
            plt.ylabel("AUC-PR")
            plt.title(f"Sensitivity: {param}")
            plt.ylim(0,1)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(subdir, plot_name))
            plt.close()

    # --- 総合まとめ ---
    pd.DataFrame(overall_summary).to_csv(
        os.path.join(output_root, f"all_sensitivity_summary_{timestamp}.csv"), index=False
    )
    send_slack_message("✅ 全ての感度分析が完了しました！")

# ----------------------------------------------------------
if __name__ == "__main__":
    run_all_sensitivity()
