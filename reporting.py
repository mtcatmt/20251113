# reporting.py
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

def save_results_csv(results_df: pd.DataFrame, output_dir: str, base_name: str):
    """
    集計結果をCSVファイル (avg_scores, stats) として保存する。
    (元ファイル main_modified.py L290-L303 から移動)
    """
    try:
        avg_scores = results_df.groupby(["sim_type", "type"])[["precision", "recall", "f1_score"]].mean()
        stats = results_df.groupby(["sim_type", "type"])[["precision", "recall", "f1_score"]].agg(["mean", "std", "median"])

        avg_path = os.path.join(output_dir, f"avg_scores_{base_name}.csv")
        stats_path = os.path.join(output_dir, f"stats_{base_name}.csv")

        avg_scores.to_csv(avg_path)
        stats.to_csv(stats_path)

        logging.info(f"平均スコアを保存: {avg_path}")
        logging.info(f"統計スコアを保存: {stats_path}")
        
    except Exception as e:
        logging.error(f"CSV結果の保存に失敗: {e}")

def plot_results_boxplots(
    results_df: pd.DataFrame, 
    output_dir: str, 
    base_name: str, 
    n_simulations: int,
    config: dict
):
    """
    結果の箱ひげ図 を生成し、PNGファイルとして保存する。
    """
    
    plt.style.use("seaborn-v0_8-whitegrid")

    # # --- 1. データセット② (Mixed) の箱ひげ図 ---
    # try:
    #     df_mixed = results_df[results_df["sim_type"] == "Mixed"]
    #     if not df_mixed.empty:
    #         box_path_mixed = os.path.join(output_dir, f"boxplot_mixed_{base_name}.png")
    #         df_melted_mixed = df_mixed.melt(
    #             id_vars=["type"],
    #             value_vars=["precision", "recall", "f1_score"],
    #             var_name="Metric",
    #             value_name="Score",
    #         )
    #         fig_box_mixed, ax_box_mixed = plt.subplots(figsize=(12, 7))
    #         sns.boxplot(x="type", y="Score", hue="Metric", data=df_melted_mixed, ax=ax_box_mixed)
    #         ax_box_mixed.set(
    #             title=f"VAR-LiNGAM Metrics ({n_simulations} sims, Dataset 2: Mixed Data)",
    #             xlabel="Edge Type (Child Variable)",
    #             ylabel="Score",
    #             ylim=(-0.05, 1.05)
    #         )
    #         ax_box_mixed.legend(title="Metric")
    #         plt.tight_layout()
    #         fig_box_mixed.savefig(box_path_mixed)
    #         logging.info(f"箱ひげ図 (Mixed) を保存: {box_path_mixed}")
    #         plt.close(fig_box_mixed)
    # except Exception as e:
    #     logging.error(f"Failed to plot mixed boxplot: {e}")

    # --- 2. 結合した箱ひげ図 (メトリクス別) ---
        # --- 結合プロット用のデータを抽出 ---
        # (データセット② は 'All' のみ使用)
    try:
        logging.info("Generating combined boxplot (grouped by metric)...")
        
        df_cont = results_df[results_df["sim_type"] == "Continuous"].copy()
        df_mixed_all = results_df[(results_df["sim_type"] == "Mixed") & (results_df["type"] == "All")].copy()
        df_logit = results_df[results_df["sim_type"] == "LogitScore"].copy()

        # データセットタイプが空（シミュレーション失敗など）の場合、concatエラーを避ける
        dfs_to_concat = []
        if not df_cont.empty:
            df_cont["Dataset"] = "1: Continuous"
            dfs_to_concat.append(df_cont)
        if not df_mixed_all.empty:
            df_mixed_all["Dataset"] = "2: Mixed (All)"
            dfs_to_concat.append(df_mixed_all)
        
        if not df_logit.empty:
            df_logit["Dataset"] = "3: Logit Score"
            dfs_to_concat.append(df_logit)

        if not dfs_to_concat:
            logging.warning("Combined plot: No data available to plot.")
            return

        df_combined_for_plot = pd.concat(dfs_to_concat)
        df_melted = df_combined_for_plot.melt(
            id_vars=["Dataset"], 
            value_vars=["precision", "recall", "f1_score"], 
            var_name="Metric", 
            value_name="Score"
        )
        
        g = sns.catplot(
            x="Dataset", 
            y="Score", 
            col="Metric",
            data=df_melted, 
            kind="box",
            height=7, 
            aspect=1.0,
            hue="Dataset",
            order=["1: Continuous", "2: Mixed (All)", "3: Logit Score"], 
            hue_order=["1: Continuous", "2: Mixed (All)", "3: Logit Score"], # ★ 色の順序をx軸と一致させる
            col_order=["precision", "recall", "f1_score"]
        )
        
        g.fig.suptitle(f"VAR-LiNGAM Performance Metrics ({n_simulations} sims) - Grouped by Metric", y=1.03)
        g.set_axis_labels("Dataset Type", "Score")
        g.set_titles("{col_name}")
        g.set(ylim=(-0.05, 1.05))
        if g.legend: # 凡例が None でない場合のみ削除
            g.legend.remove()
        
        for ax in g.axes.flat:
            ticks = range(len(ax.get_xticklabels()))
            ax.set_xticks(ticks)
            ax.set_xticklabels([lbl.get_text() for lbl in ax.get_xticklabels()], rotation=15)

            ax.grid(True, linestyle='--', alpha=0.6)

        comb_box_path = os.path.join(output_dir, f"boxplot_combined_by_metric_{base_name}.png")
        g.savefig(comb_box_path)
        logging.info(f"メトリクス別 結合箱ひげ図を保存: {comb_box_path}")
        plt.close(g.fig)

    except Exception as e:
        logging.error(f"Failed to plot combined (by metric) boxplot: {e}")


def generate_reports(
    results_all: List[Dict[str, Any]], 
    output_dir: str, 
    param_str: str, 
    timestamp: str, 
    n_simulations: int,
    config: dict
):
    """
    シミュレーション結果のリストを受け取り、CSV保存とプロットを行う。
    """
    
    # --- 結果の整理 ---
    results_df = pd.DataFrame(results_all)
    results_df["simulation_run"] = results_df.groupby(["sim_type", "type"]).cumcount()

    base_name = f"parallel_results_{param_str}_{timestamp}"

    # --- CSV保存 ---
    save_results_csv(results_df, output_dir, base_name)
    
    # --- 可視化 ---
    plot_results_boxplots(results_df, output_dir, base_name, n_simulations, config)

def generate_reports_new_metrics(results_all, output_dir, param_str, timestamp):
    """
    新形式（AUC-PR, Brier, ECE, MSE, MAE）を含むレポート出力。
    simulation.py → save_consolidated_outputs() の結果に対応。
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import logging

    results = []
    for res in results_all:
        # 各シミュレーション結果から cont/mixed/logit の3種を抽出
        for tag in ["cont", "mixed", "logit"]:
            r = res.get(tag)
            if not r:
                continue
            results.append({
                "sim_id": r["sim_id"],
                "dataset": r["dataset"],
                "AUC_PR": r.get("AUC_PR", np.nan),
                "mse": r.get("mse", np.nan),
                "mae": r.get("mae", np.nan),
                "brier": r.get("brier", np.nan),
                "ece": r.get("ece", np.nan),
            })

    if not results:
        logging.warning("generate_reports_new_metrics: 出力対象データがありません。")
        return

    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"new_metrics_{param_str}_{timestamp}"

    # --- CSV保存 ---
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"新指標の結果を保存: {csv_path}")

    # --- 箱ひげ図で可視化 ---
    melted = df.melt(id_vars=["dataset"], value_vars=["AUC_PR", "brier", "ece", "mse", "mae"],
                     var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Metric", y="Score", hue="dataset", data=melted)
    plt.title("Performance Metrics (AUC-PR / Brier / ECE / MSE / MAE)")
    plt.ylim(0, 1)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{base_name}_boxplot.png")
    plt.savefig(fig_path)
    plt.close()
    logging.info(f"新指標の箱ひげ図を保存: {fig_path}")


# --- 追加: 3本柱の結果出力用 ---
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from evaluation import auc_pr_score, weight_errors, calibration_metrics

def generate_additional_reports(B_true, B_est, edge_probs, output_dir, base_name):
    """3本柱の補助レポートを作成して保存"""
    # === 1️⃣ AUC-PR ===
    auc_pr = auc_pr_score(B_true, edge_probs)
    pd.DataFrame([{"AUC_PR": auc_pr}]).to_csv(
        os.path.join(output_dir, f"auc_pr_results_{base_name}.csv"), index=False
    )

    # === 2️⃣ 重み誤差 ===
    err = weight_errors(B_true, B_est, where="support")
    pd.DataFrame([err]).to_csv(
        os.path.join(output_dir, f"weight_error_results_{base_name}.csv"), index=False
    )

    # === 3️⃣ キャリブレーション ===
    cal = calibration_metrics(B_true, edge_probs)
    pd.DataFrame([cal]).to_csv(
        os.path.join(output_dir, f"calibration_results_{base_name}.csv"), index=False
    )

    # === 信頼度曲線（Reliability図） ===
    y_true = (B_true.flatten() != 0).astype(int)
    y_prob = edge_probs.flatten()
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.plot(mean_pred, frac_pos, marker='o')
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability diagram")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"reliability_{base_name}.png"))
    plt.close()
# =============================================
# 3本柱の自動レポート出力（Mixed＋Logit用）
# =============================================
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from evaluation import auc_pr_score, weight_errors, calibration_metrics

def generate_additional_reports_for_datasets(
    B_true_mixed, B_est_mixed, edge_probs_mixed,
    B_true_logit, B_est_logit, edge_probs_logit,
    output_dir, base_name, param_str=None
):

    """MixedとLogitの両方の3本柱を生成・保存"""
    def _one_dataset_report(B_true, B_est, edge_probs, tag):
        auc_pr = auc_pr_score(B_true, edge_probs)
        err = weight_errors(B_true, B_est, where="support")
        cal = calibration_metrics(B_true, edge_probs)

        # CSV保存
        suffix = f"_{param_str}" if param_str else ""
        pd.DataFrame([{"AUC_PR": auc_pr, **err, **cal}]).to_csv(
            os.path.join(output_dir, f"metrics_{tag}_{base_name}{suffix}.csv"), index=False
        )
        plt.savefig(os.path.join(output_dir, f"reliability_{tag}_{base_name}{suffix}.png"))


        # 信頼度曲線
        y_true = (B_true.flatten() != 0).astype(int)
        y_prob = edge_probs.flatten()
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.figure(figsize=(5,5))
        plt.plot([0,1],[0,1],'--',color='gray')
        plt.plot(mean_pred, frac_pos, marker='o', label=tag)
        plt.xlabel("Predicted probability")
        plt.ylabel("Empirical accuracy")
        plt.title(f"Reliability: {tag}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"reliability_{tag}_{base_name}.png"))
        plt.close()

    _one_dataset_report(B_true_mixed, B_est_mixed, edge_probs_mixed, "Mixed")
    _one_dataset_report(B_true_logit, B_est_logit, edge_probs_logit, "LogitScore")
