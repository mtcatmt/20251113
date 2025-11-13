# ==========================================================
# sensitivity_precision_recall_f1.py（完全修正版・最終安定版）
# ==========================================================

# python sensitivity_precision_recall_f1.py --workers 6

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import as_completed, TimeoutError as PebbleTimeoutError

# --- ローカルモジュール ---
from slack_notifier import send_slack_message
from config import SIMULATION_CONFIG, GLOBAL_SEED
from utils import setup_logging
from data_generator import generate_artificial_data, create_dataset_iv_logit_score
from analysis import run_block_bootstrap_lingam
from evaluation import calculate_metrics


# ==========================================================
# 🎯 感度分析対象パラメータ
# ==========================================================
param_sweep = {
    "PROBABILITY_THRESHOLD": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
    # "N_VARS": [2, 3, 4, 6, 8],
}

output_root = "sensitivity_multi_results"
workers_default = os.cpu_count()


# ----------------------------------------------------------
# B行列構築（閾値対応）
# ----------------------------------------------------------
def build_B(df_edge: pd.DataFrame, labels: list, prob_threshold: float) -> np.ndarray:
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    B_est = np.zeros((n, n))
    if df_edge is None or df_edge.empty:
        return B_est
    df_filt = df_edge[df_edge["Probability"] >= prob_threshold]
    for _, row in df_filt.iterrows():
        f = label_to_idx.get(row["From"])
        t = label_to_idx.get(row["To"])
        if f is not None and t is not None:
            B_est[t, f] = row["Effect"]
    return B_est


# ----------------------------------------------------------
# 単一シミュレーション実行
# ----------------------------------------------------------
def run_single_simulation_metrics(sim_id: int, config: dict, seed: int):
    np.random.seed(seed)
    try:
        th = float(config["PROBABILITY_THRESHOLD"])

        data_ts_cont, data_ts_mixed, B_true, base_names, lag, disc_names = generate_artificial_data(
            n_vars=config["N_VARS"],
            n_samples=config["N_SAMPLES"],
            lag=config["LAG"],
            seed=seed,
            edge_prob=config["EDGE_PROB"],
            discrete_parent_mode=config.get("DISCRETE_PARENT_MODE", "default"),
        )

        # --- Continuous ---
        df_edge_cont, labels_cont = run_block_bootstrap_lingam(
            data=data_ts_cont,
            block_size=config["BLOCK_SIZE"],
            lag=lag,
            base_names_list=base_names,
            n_sampling=config["BOOTSTRAP_SAMPLES"],
            random_seed=seed,
        )
        B_est_cont = build_B(df_edge_cont, labels_cont, th)

        # --- Mixed ---
        df_edge_mixed, labels_mixed = run_block_bootstrap_lingam(
            data=data_ts_mixed,
            block_size=config["BLOCK_SIZE"],
            lag=lag,
            base_names_list=base_names,
            n_sampling=config["BOOTSTRAP_SAMPLES"],
            random_seed=seed,
        )
        B_est_mixed = build_B(df_edge_mixed, labels_mixed, th)

        # --- Logit Score ---
        data_ts_logit = create_dataset_iv_logit_score(
            data_ts_mixed=data_ts_mixed,
            B_true=B_true,
            base_names=base_names,
            discrete_variable_names=disc_names,
            labels=labels_mixed,
            lag=lag,
        )
        df_edge_logit, labels_logit = run_block_bootstrap_lingam(
            data=data_ts_logit,
            block_size=config["BLOCK_SIZE"],
            lag=lag,
            base_names_list=base_names,
            n_sampling=config["BOOTSTRAP_SAMPLES"],
            random_seed=seed,
        )
        B_est_logit = build_B(df_edge_logit, labels_logit, th)

        # --- Precision / Recall / F1 ---
        m_cont = calculate_metrics(B_true, B_est_cont)
        m_mixed = calculate_metrics(B_true, B_est_mixed)
        m_logit = calculate_metrics(B_true, B_est_logit)

        return {
            "ok": True,
            "sim_id": sim_id,
            "precision_cont": m_cont["precision"],
            "recall_cont": m_cont["recall"],
            "f1_cont": m_cont["f1_score"],
            "precision_mixed": m_mixed["precision"],
            "recall_mixed": m_mixed["recall"],
            "f1_mixed": m_mixed["f1_score"],
            "precision_logit": m_logit["precision"],
            "recall_logit": m_logit["recall"],
            "f1_logit": m_logit["f1_score"],
        }

    except Exception as e:
        logging.error(f"[Sim {sim_id}] error: {e}")
        return {"ok": False, "sim_id": sim_id}


# ----------------------------------------------------------
def run_parallel_metrics(config: dict, n_workers: int, seeds: list):
    results = []
    timeout = config.get("SIMULATION_TIMEOUT", 600)
    logging.info(f"並列実行開始 (Workers={n_workers}, Timeout={timeout}s)")

    with ProcessPool(max_workers=n_workers) as executor:
        futures = {
            executor.schedule(run_single_simulation_metrics, args=[i, config, seeds[i]], timeout=timeout): i
            for i in range(config["N_SIMULATIONS"])
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Simulations"):
            try:
                res = f.result()
                if isinstance(res, dict) and res.get("ok"):
                    results.append(res)
            except PebbleTimeoutError:
                logging.warning("タイムアウト発生")
            except Exception as e:
                logging.error(f"プールエラー: {e}")

    return results


# ----------------------------------------------------------
def run_sensitivity_analysis(param_name: str, param_values, output_dir="sensitivity_results_f1", workers=os.cpu_count()):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_config = SIMULATION_CONFIG.copy()
    send_slack_message(f"🚀 F1感度分析開始: {param_name} = {param_values}")

    summary_records = []

    for val in param_values:
        config = base_config.copy()
        config[param_name] = val
        np.random.seed(GLOBAL_SEED)
        seeds = [np.random.randint(0, 2**31 - 1) for _ in range(config["N_SIMULATIONS"])]

        # --- 実行設定をJSONで保存（出力ディレクトリ基準） ---
        import json
        cfg_json_path = os.path.join(output_dir, f"config_{param_name}_{val}_{timestamp}.json")
        try:
            with open(cfg_json_path, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logging.warning(f"設定保存に失敗: {e}")

        send_slack_message(f"⚙️ {param_name}={val} 実行中…")
        results = run_parallel_metrics(config, workers, seeds)
        if not results:
            send_slack_message(f"⚠️ {param_name}={val}: 結果なし")
            continue

        df = pd.DataFrame(results)
        summary = {"param_value": val}
        for metric in ["precision", "recall", "f1"]:
            for ds in ["cont", "mixed", "logit"]:
                summary[f"{metric}_{ds}_mean"] = np.nanmean(df[f"{metric}_{ds}"])
                summary[f"{metric}_{ds}_std"] = np.nanstd(df[f"{metric}_{ds}"], ddof=1)
        summary_records.append(summary)

        try:
            msg = (
                f"📊 {param_name}={val} 中間結果\n"
                f"Cont: F1={summary['f1_cont_mean']:.3f}, "
                f"Mixed: F1={summary['f1_mixed_mean']:.3f}, "
                f"Logit: F1={summary['f1_logit_mean']:.3f}"
            )
            send_slack_message(msg)
        except Exception as e:
            logging.warning(f"Slack通知失敗: {e}")

    if not summary_records:
        send_slack_message(f"❌ {param_name}: 全設定で空")
        return

    df_summary = pd.DataFrame(summary_records).sort_values("param_value")
    csv_path = os.path.join(output_dir, f"{param_name}_metrics_summary_{timestamp}.csv")
    df_summary.to_csv(csv_path, index=False)
    logging.info(f"保存: {csv_path}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        for ds, marker, label in [("cont", "o", "Continuous"), ("mixed", "s", "Mixed"), ("logit", "^", "LogitScore")]:
            y = df_summary[f"{metric}_{ds}_mean"].values
            s = df_summary[f"{metric}_{ds}_std"].values
            ax.plot(df_summary["param_value"], y, marker=marker, label=label)
            ax.fill_between(df_summary["param_value"], y - s, y + s, alpha=0.15)
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1)
        ax.legend()
    fig.suptitle(f"Sensitivity: {param_name} (Precision / Recall / F1)")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{param_name}_metrics_plot_{timestamp}.png")
    plt.savefig(fig_path)
    plt.close()
    send_slack_message(f"✅ {param_name} 感度分析完了. 出力: {csv_path}")


# ----------------------------------------------------------
# メインエントリ
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=workers_default)
    args = parser.parse_args()

    send_slack_message(f"🚀 Precision/Recall/F1 感度分析 全{len(param_sweep)}セット開始")
    for param_name, param_values in param_sweep.items():
        subdir = os.path.join(output_root, param_name)
        os.makedirs(subdir, exist_ok=True)
        run_sensitivity_analysis(param_name, param_values, subdir, args.workers)
    send_slack_message("🎯 全パラメータのF1感度分析が完了しました！")
