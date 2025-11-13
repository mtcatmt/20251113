# analysis.py
import numpy as np
import pandas as pd
from collections import defaultdict
from lingam import VARLiNGAM
from arch.bootstrap import CircularBlockBootstrap
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def run_block_bootstrap_lingam(
    data: pd.DataFrame, 
    block_size: int, 
    lag: int, 
    base_names_list: List[str], 
    n_sampling: int = 1, 
    random_seed: int = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    CircularBlockBootstrap を用いて VAR-LiNGAM を n_sampling 回適用し、
    エッジの出現確率と平均効果量を集計して返す。
    (元ファイル lingam_utils_modified.py L83-L131 から移動)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    labels = [f"{name}(t-{p})" if p > 0 else f"{name}(t)" for p in range(lag + 1) for name in base_names_list]
    results = []

    # --- 定数列チェック（ほぼゼロ分散の列に微小ノイズ） ---
    data = data.copy()
    stds = data.std()
    low_var_cols = stds[stds < 1e-8].index.tolist()
    if low_var_cols:
        logger.warning(f"Low-variance columns detected: {low_var_cols}. Adding small noise to prevent singularity.")
        for col in low_var_cols:
            data[col] += np.random.normal(0, 1e-6, size=len(data))

            
    bs = CircularBlockBootstrap(block_size, data.values)  # data -> ndarray expected

    for resampled_data in bs.bootstrap(n_sampling):
        boot_data = np.squeeze(resampled_data[0])
        model_var = VARLiNGAM(lags=lag)
        try:
            model_var.fit(boot_data)
            # adjacency_matrices_ : list of (n_nodes x n_nodes) for each lag (0..lag)
            results.append(model_var.adjacency_matrices_)
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.debug("VARLiNGAM fit failed for a bootstrap sample: %s", e)
            continue

    if not results:
        empty_df = pd.DataFrame(columns=["From", "To", "Count", "Effect", "Probability"])
        return empty_df, labels

    edge_data = defaultdict(lambda: {"count": 0, "total_effect": 0.0})
    n_successful = len(results)

    for matrices_list in results:
        for lag_idx, matrix in enumerate(matrices_list):
            to_indices, from_indices = np.where(matrix != 0)
            for from_node, to_node in zip(from_indices, to_indices):
                child_label = labels[to_node]  # t時点の変数
                parent_label = labels[from_node + lag_idx * len(base_names_list)]  # t-lag時点の変数

                edge_data[(parent_label, child_label)]["count"] += 1
                edge_data[(parent_label, child_label)]["total_effect"] += matrix[to_node, from_node]
        



    summary_list = []
    for edge, d in edge_data.items():
        summary_list.append({
            "From": edge[0],
            "To": edge[1],
            "Count": d["count"],
            "Effect": d["total_effect"] / d["count"],
            "Probability": d["count"] / n_successful,
        })

    df_edge = pd.DataFrame(summary_list)
    if not df_edge.empty:
        df_edge = df_edge.sort_values(by="Probability", ascending=False).reset_index(drop=True)
    return df_edge, labels


def render_estimated_graph(B_est: np.ndarray, labels: List[str], output_path: str):
    """
    lingam.utils.make_dot を使い、推定された行列を可視化して PNG (またはPDF)で保存する。
    make_dot が利用できない環境では何もしない（エラーは投げない）。

    """
    try:
        from lingam.utils import make_dot
    except Exception:
        logger.info("lingam.utils.make_dot is not available; skipping graph render.")
        return False

    try:
        dot = make_dot(B_est, labels=labels)
        dot.format = "png"
        dot.render(output_path, cleanup=True)
        return True
    except Exception as e:
        logger.exception("Failed to render estimated graph: %s", e)
        return False