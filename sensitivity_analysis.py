# sensitivity_analysis.py
# python sensitivity_analysis.py --workers 6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
from config import SIMULATION_CONFIG, GLOBAL_SEED
from utils import setup_logging
from simulation import run_parallel_simulations

# ==========================================================
# 🎯 感度分析の対象パラメータを指定
# ==========================================================
param_name = "EDGE_PROB"
param_values = np.arange(0.0, 1.01, 0.1)
workers = os.cpu_count()
output_dir = "sensitivity_results"


def filter_edges(df_edge: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """確率閾値でエッジをフィルタ"""
    if df_edge is None or df_edge.empty:
        return df_edge
    return df_edge[df_edge["Probability"] >= threshold]


def run_sensitivity_analysis(param_name: str, param_values, output_dir="sensitivity_results", workers=os.cpu_count()):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"sensitivity_{param_name}_{timestamp}.log")
    setup_logging(log_path)

    logging.info(f"=== 感度分析を開始 ({param_name}) ===")

    base_config = SIMULATION_CONFIG.copy()
    metrics_summary = []

    for val in param_values:
        config = base_config.copy()
        config[param_name] = val
        th = config["PROBABILITY_THRESHOLD"]

        logging.info(f"\n--- {param_name} = {val} の実験を開始 ---")

        np.random.seed(GLOBAL_SEED)
        simulation_seeds = [np.random.randint(0, 2**31 - 1) for _ in range(config["N_SIMULATIONS"])]

        results_all, summary_counts = run_parallel_simulations(config, workers, simulation_seeds)
        logging.info(f"{param_name}={val} 完了: 成功={summary_counts['success']}, 部分成功={summary_counts['partial']}")

        if not results_all:
            continue

        # 結果整形
        df = pd.DataFrame(results_all)
        if "sim_type" not in df.columns:
            continue

        # 各データセット（Continuous, Mixed, LogitScore）すべてに閾値を適用
        df_filtered = []
        for sim_type in ["Continuous", "Mixed", "LogitScore"]:
            df_sub = df[df["sim_type"] == sim_type]
            if df_sub.empty:
                continue
            # 仮にProbability列があればここでfilter_edgesを適用（再評価用に拡張）
            df_filtered.append(df_sub)

        df_filtered = pd.concat(df_filtered, ignore_index=True)
        df_mean = df_filtered.groupby("sim_type")[["precision", "recall", "f1_score"]].mean().reset_index()
        df_mean[param_name] = val
        metrics_summary.append(df_mean)

    if not metrics_summary:
        logging.error("全ての設定で結果が得られませんでした。")
        return

    metrics_df = pd.concat(metrics_summary, ignore_index=True)
    csv_path = os.path.join(output_dir, f"{param_name}_sensitivity_{timestamp}.csv")
    metrics_df.to_csv(csv_path, index=False)
    logging.info(f"感度分析結果を保存しました: {csv_path}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["precision", "recall", "f1_score"]):
        for sim_type in metrics_df["sim_type"].unique():
            df_sub = metrics_df[metrics_df["sim_type"] == sim_type]
            ax.plot(df_sub[param_name], df_sub[metric], marker="o", label=sim_type)
        ax.set_title(metric.capitalize())
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()

    plt.suptitle(f"VAR-LiNGAM Sensitivity to {param_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(output_dir, f"{param_name}_sensitivity_plot_{timestamp}.png")
    plt.savefig(fig_path)
    plt.close()

if __name__ == "__main__":
    run_sensitivity_analysis(param_name, param_values, output_dir, workers)
