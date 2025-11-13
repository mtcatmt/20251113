# main.py
# python main.py --workers 6

import argparse
import logging
import os
from datetime import datetime
import numpy as np
import random
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from config import GLOBAL_SEED, SIMULATION_CONFIG
from utils import setup_logging
from simulation import run_parallel_simulations
from reporting import generate_reports
from slack_notifier import send_slack_message




# ==========================================================
# 📊 Mixed vs Logit の統計的有意差検定関数
# ==========================================================
def compare_mixed_logit_significance(df: pd.DataFrame, output_dir: str):
    """
    precision / recall / f1 の Mixed vs Logit を対応のある t検定・Wilcoxon検定で比較
    """
    results = []
    metrics = ["precision", "recall", "f1_score"]

    for metric in metrics:
        col_m = f"{metric}_mixed"
        col_l = f"{metric}_logit"
        if col_m not in df.columns or col_l not in df.columns:
            logging.warning(f"{metric}: 列が見つかりません (skip)")
            continue

        x = df[col_m].values
        y = df[col_l].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if len(x) < 5:
            logging.warning(f"{metric}: データ不足 ({len(x)} ペア)")
            continue

        # --- 対応のある検定 (片側: logit > mixed) ---
        t_stat, t_p = ttest_rel(y, x, alternative='greater')
        try:
            w_stat, w_p = wilcoxon(y - x, alternative='greater')
        except ValueError:
            w_stat, w_p = np.nan, np.nan

        mean_diff = np.mean(y - x)
        better = "logit > mixed" if mean_diff > 0 else "mixed > logit"

        results.append({
            "Metric": metric.upper(),
            "Mean(Mixed)": np.mean(x),
            "Mean(Logit)": np.mean(y),
            "MeanDiff(Logit-Mixed)": mean_diff,
            "Better": better,
            "t_pval": t_p,
            "wilcoxon_pval": w_p
        })

    # --- 出力 ---
    res_df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "significance_mixed_logit.csv")
    res_df.to_csv(out_path, index=False, float_format="%.6f")

    logging.info(f"有意差検定結果を保存: {out_path}")
    send_slack_message(f"📊 Mixed vs Logit 有意差検定完了\n結果: `{os.path.basename(out_path)}`")
    logging.info(f"\n{res_df}\n")

    # Slackにも結果を概要で通知
    summary_lines = [
        f"{r['Metric']}: t_p={r['t_pval']:.4f}, w_p={r['wilcoxon_pval']:.4f}, better={r['Better']}"
        for _, r in res_df.iterrows()
    ]
    send_slack_message("```\n" + "\n".join(summary_lines) + "\n```")

    return res_df


# ==========================================================
# 🎯 メイン実行関数
# ==========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results", help="出力ディレクトリ")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="並列実行数")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    config = SIMULATION_CONFIG
    simulation_seeds = [np.random.randint(0, 2**31 - 1) for _ in range(config["N_SIMULATIONS"])]

    param_str = (
        f"sim{config['N_SIMULATIONS']}_s{config['N_SAMPLES']}_"
        f"l{config['LAG']}_v{config['N_VARS']}_bs{config['BOOTSTRAP_SAMPLES']}"
    )
    # --- config保存 ---
    import json
    with open(os.path.join(args.output_dir, f"config_{timestamp}.json"), "w") as f:
        json.dump(config, f, indent=2)

    logging.info(f"開始: {config['N_SIMULATIONS']} 回シミュレーションを {args.workers} 並列で実行します")
    logging.info(f"設定: {config}")
    send_slack_message(f"🚀 main.py 実行開始\n設定: {config}\nWorkers={args.workers}")

    try:
        results_all, summary_counts = run_parallel_simulations(config, args.workers, simulation_seeds)

        # --- ログサマリ ---
        n_total = config["N_SIMULATIONS"]
        n_completed = (
            summary_counts.get("success", 0)
            + summary_counts.get("partial", 0)
            + summary_counts.get("empty_success", 0)
            + summary_counts.get("internal_error", 0)
        )
        n_failed = summary_counts.get("timeout", 0) + summary_counts.get("pool_error", 0)
        n_processed = n_completed + n_failed

        logging.info(f"--- シミュレーション実行サマリ (処理済: {n_processed} / {n_total}) ---")
        logging.info(f"  [完了] {n_completed} 回 (タイムアウトなし)")
        logging.info(f"    - 成功 (全行): {summary_counts.get('success', 0)} 回")
        logging.info(f"    - 部分成功: {summary_counts.get('partial', 0)} 回")
        logging.info(f"    - LiNGAM失敗: {summary_counts.get('empty_success', 0)} 回")
        logging.info(f"    - 内部エラー: {summary_counts.get('internal_error', 0)} 回")
        logging.info(f"  [失敗] {n_failed} 回")
        logging.info(f"    - タイムアウト: {summary_counts.get('timeout', 0)} 回")
        logging.info(f"    - プールエラー: {summary_counts.get('pool_error', 0)} 回")
        logging.info(f"--- 合計収集結果 (総行数): {len(results_all)} ---")

        if not results_all:
            msg = "⚠️ 結果が空です。VAR-LiNGAMがすべて失敗した可能性があります。"
            logging.warning(msg)
            send_slack_message(msg)
            return

        # --- レポート生成 ---
        generate_reports(
            results_all=results_all,
            output_dir=args.output_dir,
            param_str=param_str,
            timestamp=timestamp,
            n_simulations=config["N_SIMULATIONS"],
            config=config
        )

        # --- 有意差検定の実行 ---
        df_results = pd.DataFrame(results_all)

        # 縦型データを横型（mixed/logit列）に変換
        df_wide = (
            df_results.pivot_table(
                index=df_results.index // 3,  # sim_idがない場合、ブロックごとにまとめる
                columns="sim_type",
                values=["precision", "recall", "f1_score"]
            )
        )
        df_wide.columns = [f"{m}_{t.lower()}" for m, t in df_wide.columns]
        df_wide.reset_index(drop=True, inplace=True)

        # 修正ポイント: df_wideをチェック・渡す
        # --- 列名を正規化して存在チェック ---
        df_wide.rename(columns=lambda c: c.lower().replace("logitscore", "logit"), inplace=True)
        if all(f"{m}_mixed" in df_wide.columns and f"{m}_logit" in df_wide.columns for m in ["precision", "recall", "f1_score"]):
            compare_mixed_logit_significance(df_wide, args.output_dir)
        else:
            logging.warning("Mixed/Logit列が不足しているため検定をスキップ")

        try:
            send_slack_message(f"✅ main.py 実行完了: {config['N_SIMULATIONS']} 回シミュレーション終了")
        except Exception as e:
            logging.error(f"Slack通知失敗: {e}")
        logging.info("Precision / Recall / F1 の有意差検定を含む全処理が完了しました。")

    except Exception as e:
        logging.error(f"main.py 実行中にエラー: {e}")
        send_slack_message(f"💥 main.py 実行中にエラー発生: {e}")


if __name__ == "__main__":
    main()
