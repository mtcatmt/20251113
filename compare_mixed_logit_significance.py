# compare_mixed_logit_significance.py
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
import os

def compare_mixed_logit_significance(csv_path: str, output_dir: str = "results"):
    """
    metrics_all_*.csv を読み込み、mixed と logit の間で
    AUC-PR, Brier, ECE の有意差を検定（paired t-test & Wilcoxon）
    """

    # --- データ読み込み ---
    df = pd.read_csv(csv_path)
    if "dataset" not in df.columns or "sim_id" not in df.columns:
        raise ValueError("CSVに 'dataset' または 'sim_id' 列がありません。")

    df_mixed = df[df["dataset"].str.lower().str.contains("mixed")].set_index("sim_id")
    df_logit = df[df["dataset"].str.lower().str.contains("logit")].set_index("sim_id")

    common_ids = df_mixed.index.intersection(df_logit.index)
    if len(common_ids) == 0:
        raise ValueError("mixed と logit の共通 sim_id が見つかりません。")

    df_mixed = df_mixed.loc[common_ids]
    df_logit = df_logit.loc[common_ids]

    metrics = [m for m in ["AUC_PR", "brier", "ece", "mse", "mae"] if m in df.columns]
    results = []

    # --- 各指標ごとに検定 ---
    for metric in metrics:
        x, y = df_mixed[metric].values, df_logit[metric].values

        # 欠損値を除去
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 5:
            print(f"[Warning] {metric}: 有効なペアが少ないためスキップ ({len(x)} ペア)")
            continue

        # paired t-test
        t_stat, t_p = ttest_rel(x, y)
        # Wilcoxon signed-rank
        try:
            w_stat, w_p = wilcoxon(x, y)
        except ValueError:
            w_stat, w_p = np.nan, np.nan

        # 平均差と方向性
        mean_diff = np.mean(y - x)  # logit - mixed
        better = "logit > mixed" if mean_diff > 0 else "mixed > logit"

        results.append({
            "Metric": metric,
            "Mean(mixed)": np.mean(x),
            "Mean(logit)": np.mean(y),
            "MeanDiff(logit-mixed)": mean_diff,
            "Better": better,
            "t_pval": t_p,
            "wilcoxon_pval": w_p
        })

    # --- 出力 ---
    res_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "significance_test_results.csv")
    res_df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"✅ 結果を保存しました: {out_path}")
    print(res_df)

    return res_df


if __name__ == "__main__":
    # 例: python compare_mixed_logit_significance.py --csv results/metrics_all_sim100_s1500_l1_v4_bs50_dpdefault.csv
    import argparse
    if __name__ == "__main__":
        csv_path = r"C:/research/20251023~/results/metrics_all_sim100_s1500_l1_v4_bs50_dpdefault.csv"# ←ここに自分のCSVパス
        compare_mixed_logit_significance(csv_path, "results")
