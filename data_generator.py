# data_generator.py
import pandas as pd
import numpy as np
import networkx as nx
import random
import logging
import warnings
from typing import Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

logger = logging.getLogger(__name__)

# ---------------------------
# データセット①, ②
# ---------------------------

def generate_artificial_data(
    n_vars: int,
    n_samples: int,
    lag: int,
    seed: int,
    edge_prob: float,
    discrete_parent_mode: str = "default" # ★ 1. 引数を追加
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[str], int, List[str]]: 
    """
    指定されたラグ構造を持つ連続時系列データ（①）と混合時系列データ（②）を生成する。
    """
    # --- reproducibility for multiple RNGs ---
    np.random.seed(seed)
    random.seed(seed)

    # 縮小パラメータ（現在は [0,1]）
    min_contemp, max_contemp = 0.0, 1.0
    min_lag, max_lag = 0.0, 1.0

    # 変数名と行列サイズ
    base_names = [f"x{i+1}" for i in range(n_vars)]
    n_nodes = n_vars * (lag + 1)
    
    # --- ★★★ 2. 再試行ループのセットアップ ★★★ ---
    max_retries = 1000 # 無限ループ防止
    graph_generated_ok = False

    for attempt in range(max_retries):
        B_true = np.zeros((n_nodes, n_nodes))

        # --- contemporaneous DAG を生成（有向非巡回） ---
        max_edges_contemp = np.random.randint(max(1, n_vars - 1), (n_vars * (n_vars - 1) // 2) + 1)
        while True:
            contemp_graph = nx.gnm_random_graph(n=n_vars, m=max_edges_contemp, directed=True)
            if nx.is_directed_acyclic_graph(contemp_graph):
                break

        B_contemporaneous = np.zeros((n_vars, n_vars))
        for u, v in contemp_graph.edges():
            sign = np.random.choice([-1, 1])
            B_contemporaneous[v, u] = sign * np.random.uniform(min_contemp, max_contemp)
        B_true[:n_vars, :n_vars] = B_contemporaneous

        # --- lagged effects ---
        if lag > 0:
            for l in range(1, lag + 1):
                B_lagged = np.zeros((n_vars, n_vars))
                edges = np.random.binomial(1, edge_prob, size=(n_vars, n_vars))
                for r, c in np.argwhere(edges == 1):
                    sign = np.random.choice([-1, 1])
                    B_lagged[r, c] = sign * np.random.uniform(min_lag, max_lag)
                B_true[:n_vars, n_vars * l : n_vars * (l + 1)] = B_lagged

        num_discrete = np.random.randint(1, n_vars) if n_vars > 1 else 0

        var_types = np.zeros(n_vars, dtype=int) # 0=連続, 1=離散
        discrete_indices = np.random.choice(n_vars, num_discrete, replace=False) if num_discrete > 0 else []
        var_types[discrete_indices] = 1
        discrete_names = [base_names[i] for i in discrete_indices]

        #--- ★★★ シンプル化: defaultモードのみ考慮 ★★★
        graph_generated_ok = True
        break

    if not graph_generated_ok:
        logger.warning(f"Graph generation failed after {max_retries} retries (seed={seed}) — using last graph.")



    # --- データ生成 --- (★これ以降は変更なし、インデントも元のまま)
    X_cont = np.zeros((n_samples, n_vars)) # データセット①
    X_mixed = np.zeros((n_samples, n_vars)) # データセット②
    contemp_order = list(nx.topological_sort(contemp_graph))

    # --- 1. データセット① (連続) の生成 ---
    # 初期値 (t < lag)
    for t in range(min(lag, n_samples)):
        for i in range(n_vars):
            X_cont[t, i] = np.random.laplace(0, 1) # 常に連続 (Laplace)

    # 時系列生成 (t >= lag)
    for t in range(lag, n_samples):
        lagged_influence = np.zeros(n_vars)
        for l in range(1, lag + 1):
            B_l = B_true[:n_vars, n_vars * l : n_vars * (l + 1)]
            lagged_influence += X_cont[t - l, :] @ B_l.T # X_cont を参照

        for i in contemp_order:
            contemp_influence = X_cont[t, :] @ B_contemporaneous[i, :] # X_cont を参照
            total_influence = lagged_influence[i] + contemp_influence
            noise = np.random.laplace(0, 1)
            X_cont[t, i] = total_influence + noise
    
    data_ts_cont = pd.DataFrame(X_cont, columns=base_names)

    # --- 2. データセット② (混合) の生成 ---
    # ※ X_cont から計算される「影響量」を参照する
    
    # 初期値 (t < lag)
    for t in range(min(lag, n_samples)):
        for i in range(n_vars):
            if var_types[i] == 1: # 離散
                X_mixed[t, i] = np.random.binomial(1, 0.5)
            else: # 連続
                X_mixed[t, i] = X_cont[t, i] # 連続の場合はデータセット①の初期値と同じ


    # 時系列生成 (t >= lag)
    for t in range(lag, n_samples):
        lagged_influence_from_cont = np.zeros(n_vars)
        for l in range(1, lag + 1):
            B_l = B_true[:n_vars, n_vars * l : n_vars * (l + 1)]
            # ★ 参照元は X_cont (データセット①)
            lagged_influence_from_cont += X_cont[t - l, :] @ B_l.T

        for i in contemp_order:
            # ★ 参照元は X_cont (データセット①)
            contemp_influence_from_cont = X_cont[t, :] @ B_contemporaneous[i, :]
            total_influence_from_cont = lagged_influence_from_cont[i] + contemp_influence_from_cont
            
            if var_types[i] == 1:  # 離散
                noise = np.random.logistic(0, 1)
                X_mixed[t, i] = int(total_influence_from_cont + noise > 0)
            else:  # 連続
                X_mixed[t, i] = X_cont[t, i]

    data_ts_mixed = pd.DataFrame(X_mixed, columns=base_names)
    
    return data_ts_cont, data_ts_mixed, B_true, base_names, lag, discrete_names


# ---------------------------
# データセット③
# ---------------------------

def create_dataset_iii_logistic(
    data_ts_mixed: pd.DataFrame, 
    B_true: np.ndarray, 
    base_names: List[str], 
    discrete_variable_names: List[str],
    labels: List[str],
    lag: int
) -> pd.DataFrame:
    """
    データセット② (混合) と正解グラフ (B_true) に基づき、
    離散変数を「親」からのロジスティック回帰で連続値 (確率) に置き換えた
    データセット③ (Regressed) を作成する。
    """
    
    # 警告を抑制 (liblinear が収束しない場合があるため)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    data_ts_regressed = data_ts_mixed.copy()
    n_vars = len(base_names)
    
    # ラベル (e.g., "x1(t-1)") から (変数名, ラグ) を引く辞書
    label_to_info = {}
    for p in range(lag + 1):
        for i, name in enumerate(base_names):
            label = f"{name}(t-{p})" if p > 0 else f"{name}(t)"
            label_to_info[label] = (name, p)

    # B_true (n_nodes x n_nodes) から親ラベルを取得するためのmap
    index_to_label = {i: lbl for i, lbl in enumerate(labels)}

    for discrete_name in discrete_variable_names:
        
        # 1. ターゲット (y) の準備
        y_target = data_ts_regressed[discrete_name]
        
        # 2. 説明変数 (X) の特定
        target_base_idx = base_names.index(discrete_name) # 0..n_vars-1
        
        # B_true[target_base_idx, :] が t 時点の子への親 (contemp + lagged)
        parent_row = B_true[target_base_idx, :]
        parent_indices_full = np.where(parent_row != 0)[0] # 0..n_nodes-1
        
        parent_labels = [index_to_label[i] for i in parent_indices_full]

        X_features = pd.DataFrame(index=data_ts_mixed.index)

        if not parent_labels:
            # 親がいない場合 (非常に稀だがあり得る)
            logger.warning(f"'{discrete_name}' has no parents in B_true. Filling with 0.5.")
            data_ts_regressed[discrete_name] = 0.5 + np.random.normal(0, 1e-6, size=len(y_target))
            continue

        # --- 単一クラスのチェックを追加 ---
        if np.unique(y_target).size < 2:
            logger.warning(f"'{discrete_name}' has only one class ({np.unique(y_target)[0]}). Filling with 0.5 + small noise.")
            data_ts_regressed[discrete_name] = 0.5 + np.random.normal(0, 1e-6, size=len(y_target))
            continue


            
        # 3. 説明変数のDataFrameを構築 (ラグを考慮)
        for p_label in parent_labels:
            p_name, p_lag = label_to_info[p_label]
            # data_ts_mixed (データセット②) から親の時系列を取得し、ラグ分ずらす
            X_features[p_label] = data_ts_mixed[p_name].shift(p_lag).fillna(0)

        # 4. ロジスティック回帰の実行
        try:
            model = LogisticRegression(solver='liblinear', random_state=1)
            model.fit(X_features, y_target)
            
            # 5. 確率 (クラス1) で列を置き換え
            y_pred_proba = model.predict_proba(X_features)[:, 1]
            data_ts_regressed[discrete_name] = y_pred_proba
            
        except Exception as e:
            logger.error(f"Logistic regression failed for '{discrete_name}': {e}")
            logger.warning(f"Falling back to original mixed data for '{discrete_name}'")
            # 失敗した場合は元の混合データ (0/1) がそのまま残る
            
    return data_ts_regressed

# ---------------------------
# データセット④ (Logit Score)
# ---------------------------

def create_dataset_iv_logit_score(
    data_ts_mixed: pd.DataFrame, 
    B_true: np.ndarray, 
    base_names: List[str], 
    discrete_variable_names: List[str],
    labels: List[str],
    lag: int
) -> pd.DataFrame:
    """
    データセット② (混合) と正解グラフ (B_true) に基づき、
    離散変数を「親」からのロジスティック回帰の「ロジットスコア (線形予測子)」に
    置き換えたデータセット④ (Logit Score) を作成する。
    
    ※ データセット③ (predict_proba) とは異なり、decision_function を使用する。
    """
    
    # 警告を抑制 (liblinear が収束しない場合があるため)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    data_ts_logit_score = data_ts_mixed.copy() # データセット④用のDataFrame
    n_vars = len(base_names)
    
    # ラベル (e.g., "x1(t-1)") から (変数名, ラグ) を引く辞書
    label_to_info = {}
    for p in range(lag + 1):
        for i, name in enumerate(base_names):
            label = f"{name}(t-{p})" if p > 0 else f"{name}(t)"
            label_to_info[label] = (name, p)

    # B_true (n_nodes x n_nodes) から親ラベルを取得するためのmap
    index_to_label = {i: lbl for i, lbl in enumerate(labels)}

    for discrete_name in discrete_variable_names:
        
        y_target = data_ts_logit_score[discrete_name]
        target_base_idx = base_names.index(discrete_name)
        parent_row = B_true[target_base_idx, :]
        parent_indices_full = np.where(parent_row != 0)[0]
        parent_labels = [index_to_label[i] for i in parent_indices_full]

        X_features = pd.DataFrame(index=data_ts_mixed.index)

        if not parent_labels:
            logger.warning(f"[Dataset IV] '{discrete_name}' has no parents in B_true. Filling with 0.0.")
            data_ts_logit_score[discrete_name] = np.random.normal(0, 1e-6, size=len(y_target))
            continue

        # --- 単一クラスのチェックを追加 ---
        if np.unique(y_target).size < 2:
            single_val = np.unique(y_target)[0]
            logger.warning(f"[Dataset IV] '{discrete_name}' has only one class ({single_val}). Filling with 0.0 + small noise.")
            data_ts_logit_score[discrete_name] = np.random.normal(0, 1e-6, size=len(y_target))
            continue
            
        for p_label in parent_labels:
            p_name, p_lag = label_to_info[p_label]
            X_features[p_label] = data_ts_mixed[p_name].shift(p_lag).fillna(0)

        try:
            model = LogisticRegression(solver='liblinear', random_state=1)
            model.fit(X_features, y_target)
            
            # --- ★★★ 変更点 ★★★ ---
            # 確率 (proba) の代わりにロジットスコア (decision_function) を使用
            y_pred_logit_score = model.decision_function(X_features)
            data_ts_logit_score[discrete_name] = y_pred_logit_score
            # --- ★★★★★★★★★★★ ---
            
        except Exception as e:
            logger.error(f"[Dataset IV] Logistic regression failed for '{discrete_name}': {e}")
            logger.warning(f"[Dataset IV] Falling back to original mixed data for '{discrete_name}'")
            
    return data_ts_logit_score