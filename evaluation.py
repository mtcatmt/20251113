# evaluation.py
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)

# -------------------------
# 1️⃣ 構造復元 Precision/Recall/F1
# -------------------------
def calculate_metrics(true_matrix: np.ndarray, estimated_matrix: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
    true_adj = true_matrix != 0
    est_adj = estimated_matrix != 0
    if mask is not None:
        true_adj = true_adj[mask, :]
        est_adj = est_adj[mask, :]
    tp = np.sum(np.logical_and(est_adj, true_adj))
    fp = np.sum(np.logical_and(est_adj, np.logical_not(true_adj)))
    fn = np.sum(np.logical_and(np.logical_not(est_adj), true_adj))
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1_score": f1,
            "TP": int(tp), "FP": int(fp), "FN": int(fn)}

# -------------------------
# 2️⃣ タイプ別評価
# -------------------------
def evaluate_by_variable_type(B_true, B_est, base_names, discrete_names):
    n_vars = len(base_names)
    Bt, Be = B_true[:n_vars, :], B_est[:n_vars, :]
    mask_disc = np.array([n in discrete_names for n in base_names])
    mask_cont = ~mask_disc
    return {
        "all": calculate_metrics(Bt, Be),
        "continuous": calculate_metrics(Bt, Be, mask_cont) if np.any(mask_cont) else None,
        "discrete": calculate_metrics(Bt, Be, mask_disc) if np.any(mask_disc) else None,
    }

# -------------------------
# 3️⃣ AUC-PR（平均適合率）
# -------------------------
def auc_pr_score(B_true, edge_probs):
    y_true = (B_true.flatten() != 0).astype(int)
    y_score = edge_probs.flatten()
    return float(average_precision_score(y_true, y_score))

# -------------------------
# 4️⃣ 重み誤差（MSE/MAE）
# -------------------------
def weight_errors(B_true, B_est, where="support"):
    if where == "support":
        mask = (B_true != 0)
    elif where == "estimated":
        mask = (B_est != 0)
    else:
        mask = np.ones_like(B_true, dtype=bool)
    diff = (B_est - B_true)[mask]
    if diff.size == 0:
        return {"mse": np.nan, "mae": np.nan, "count": 0}
    return {"mse": float(np.mean(diff**2)), "mae": float(np.mean(np.abs(diff))), "count": int(diff.size)}

# -------------------------
# 5️⃣ キャリブレーション（Brier/ECE）
# -------------------------
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        idx = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if np.any(idx):
            acc = np.mean(y_true[idx])
            conf = np.mean(y_prob[idx])
            ece += (np.sum(idx) / len(y_prob)) * abs(acc - conf)
    return float(ece)

def calibration_metrics(B_true, edge_probs):
    y_true = (B_true.flatten() != 0).astype(int)
    y_score = edge_probs.flatten()
    return {"brier": float(brier_score_loss(y_true, y_score)), "ece": expected_calibration_error(y_true, y_score)}


# evaluation.py の末尾に追加
def calibration_bins(B_true: np.ndarray, edge_probs: np.ndarray, n_bins: int = 10):
    y_true = (B_true.flatten() != 0).astype(int)
    y_prob = edge_probs.flatten()
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        left, right = bins[i], bins[i+1]
        idx = (y_prob >= left) & (y_prob < right) if i < n_bins-1 else (y_prob >= left) & (y_prob <= right)
        n = int(np.sum(idx))
        if n == 0:
            out.append({"bin": i, "bin_left": float(left), "bin_right": float(right),
                        "mean_pred": np.nan, "frac_pos": np.nan, "n": 0})
        else:
            out.append({"bin": i, "bin_left": float(left), "bin_right": float(right),
                        "mean_pred": float(np.mean(y_prob[idx])),
                        "frac_pos": float(np.mean(y_true[idx])),
                        "n": n})
    return out  # list[dict]
