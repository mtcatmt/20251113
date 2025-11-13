# simulation.py
import logging
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError as PebbleTimeoutError, as_completed
from typing import List, Dict, Any, Tuple

# ローカルモジュール
from data_generator import generate_artificial_data, create_dataset_logit_lag_only
from analysis import run_block_bootstrap_lingam
from evaluation import evaluate_by_variable_type



def generate_unified_sample_csv(config: dict, output_dir: str, filename: str = "unified_sample.csv"):

    os.makedirs(output_dir, exist_ok=True)

    # --- ①②の生成 ---
    data_ts_cont, data_ts_mixed, B_true, base_names, lag, disc_names = generate_artificial_data(
        n_vars=config["N_VARS"],
        n_samples=config["N_SAMPLES"],
        lag=config["LAG"],
        seed=42,
        edge_prob=config["EDGE_PROB"],
        discrete_parent_mode="default"
    )

    # --- ④ LogitScore（ラグのみ × ロジット） ---
    data_ts_logit = create_dataset_logit_lag_only(
        data_ts_mixed=data_ts_mixed,
        base_names=base_names,
        discrete_variable_names=disc_names,
        lag=lag
    )

    # --- 列名をわかりやすく付ける ---
    df_cont = data_ts_cont.add_prefix("cont_")
    df_mixed = data_ts_mixed.add_prefix("mixed_")
    df_logit = data_ts_logit.add_prefix("logit_")

    # --- 横結合 ---
    df_all = pd.concat([df_cont, df_mixed, df_logit], axis=1)

    # --- B_true を縦長（tidy）に変換して append ---
    B_df = pd.DataFrame(B_true)
    B_df.insert(0, "row_index", range(len(B_df)))
    B_df["__type__"] = "B_true"

    # 区切り行を入れる
    sep_df = pd.DataFrame({"__separator__": ["--- B_true below this line ---"]})

    # 保存のためひとつにまとめる
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        df_all.to_csv(f, index=False)
        f.write("\n")  # 空行
        sep_df.to_csv(f, index=False)
        f.write("\n")
        B_df.to_csv(f, index=False)

    print(f"\n===== unified sample CSV created =====")
    print(f"→ {os.path.join(output_dir, filename)}")

    return df_all, B_true


# =========================================================
# 単一シミュレーション
# =========================================================
def run_single_simulation(sim_id: int, config: dict, sim_seed: int) -> Tuple[List[Dict[str, Any]], str]:
    """1回分のLiNGAM推定を実行し, Precision/Recall/F1 を返す"""
    np.random.seed(sim_seed)
    random.seed(sim_seed)

    try:
        # 1. データ生成 (Continuous + Mixed)
        data_ts_cont, data_ts_mixed, B_true, base_names, lag, disc_names = generate_artificial_data(
            n_vars=config["N_VARS"],
            n_samples=config["N_SAMPLES"],
            lag=config["LAG"],
            seed=sim_seed,
            edge_prob=config["EDGE_PROB"],
            discrete_parent_mode=config.get("DISCRETE_PARENT_MODE", "default")
        )

        results_all = []
        n_results_max = 0

        # -----------------------------------------------------
        # Dataset1 Continuous
        # -----------------------------------------------------
        n_results_max += 1
        df_edge_cont, labels_cont = run_block_bootstrap_lingam(
            data=data_ts_cont,
            block_size=config["BLOCK_SIZE"],
            lag=lag,
            base_names_list=base_names,
            n_sampling=config["BOOTSTRAP_SAMPLES"],
            random_seed=sim_seed,
        )

        if not df_edge_cont.empty:
            logging.info(f"[Sim {sim_id}] Continuous → success ({len(df_edge_cont)} edges)")
            # 推定結果の閾値適用
            df_edge_filtered = df_edge_cont[df_edge_cont["Probability"] >= config["PROBABILITY_THRESHOLD"]]
            label_to_index = {label: i for i, label in enumerate(labels_cont)}
            B_est = np.zeros((len(labels_cont), len(labels_cont)))
            for _, row in df_edge_filtered.iterrows():
                if row["From"] in label_to_index and row["To"] in label_to_index:
                    B_est[label_to_index[row["To"]], label_to_index[row["From"]]] = row["Effect"]

            evals = evaluate_by_variable_type(B_true, B_est, base_names, [])
            results_all.append({"sim_type": "Continuous", "type": "All", **evals["all"]})
        else:
            logging.warning(f"[Sim {sim_id}] Continuous → LiNGAM failed (empty result)")

        # -----------------------------------------------------
        # Dataset2 Mixed
        # -----------------------------------------------------
        n_results_max += 1
        df_edge_mixed, labels_mixed = run_block_bootstrap_lingam(
            data=data_ts_mixed,
            block_size=config["BLOCK_SIZE"],
            lag=lag,
            base_names_list=base_names,
            n_sampling=config["BOOTSTRAP_SAMPLES"],
            random_seed=sim_seed,
        )

        if not df_edge_mixed.empty:
            logging.info(f"[Sim {sim_id}] Mixed → success ({len(df_edge_mixed)} edges)")
            df_edge_filtered = df_edge_mixed[df_edge_mixed["Probability"] >= config["PROBABILITY_THRESHOLD"]]
            label_to_index = {label: i for i, label in enumerate(labels_mixed)}
            B_est = np.zeros((len(labels_mixed), len(labels_mixed)))
            for _, row in df_edge_filtered.iterrows():
                if row["From"] in label_to_index and row["To"] in label_to_index:
                    B_est[label_to_index[row["To"]], label_to_index[row["From"]]] = row["Effect"]

            evals = evaluate_by_variable_type(B_true, B_est, base_names, disc_names)
            results_all.append({"sim_type": "Mixed", "type": "All", **evals["all"]})
        else:
            logging.warning(f"[Sim {sim_id}] Mixed → LiNGAM failed (empty result)")

        # -----------------------------------------------------
        # Dataset3 LogitScore（パターン1：ラグのみ×ロジットスコア）
        # -----------------------------------------------------
        data_ts_logit = create_dataset_logit_lag_only(
            data_ts_mixed=data_ts_mixed,
            base_names=base_names,
            discrete_variable_names=disc_names,
            lag=lag
        )

        df_edge_logit, labels_logit = run_block_bootstrap_lingam(
            data=data_ts_logit,
            block_size=config["BLOCK_SIZE"],
            lag=lag,
            base_names_list=base_names,
            n_sampling=config["BOOTSTRAP_SAMPLES"],
            random_seed=sim_seed,
        )

        if not df_edge_logit.empty:
            logging.info(f"[Sim {sim_id}] LogitScore → success ({len(df_edge_logit)} edges)")
            df_edge_filtered = df_edge_logit[df_edge_logit["Probability"] >= config["PROBABILITY_THRESHOLD"]]
            label_to_index = {label: i for i, label in enumerate(labels_logit)}
            B_est = np.zeros((len(labels_logit), len(labels_logit)))
            for _, row in df_edge_filtered.iterrows():
                if row["From"] in label_to_index and row["To"] in label_to_index:
                    B_est[label_to_index[row["To"]], label_to_index[row["From"]]] = row["Effect"]

            evals = evaluate_by_variable_type(B_true, B_est, base_names, [])
            results_all.append({"sim_type": "LogitScore", "type": "All", **evals["all"]})
        else:
            logging.warning(f"[Sim {sim_id}] LogitScore → LiNGAM failed (empty result)")


        # ステータス
        if len(results_all) == n_results_max:
            status = "SUCCESS"
        elif len(results_all) > 0:
            status = "PARTIAL"
        else:
            status = "FAILED"

        return results_all, status

    except Exception as e:
        logging.error(f"[Sim {sim_id}] 内部エラー: {e}")
        return [], "ERROR"


# =========================================================
# 並列実行
# =========================================================
def run_parallel_simulations(config: dict, n_workers: int, simulation_seeds: List[int]):
    """run_single_simulation を並列で実行し, 結果と集計を返す"""
    results_all = []
    summary_counts = {"success": 0, "partial": 0, "failed": 0, "error": 0, "timeout": 0}
    SIM_TIMEOUT = config.get("SIMULATION_TIMEOUT", 600)

    logging.info(f"並列実行開始 (Workers={n_workers}, Timeout={SIM_TIMEOUT}s)")

    with ProcessPool(max_workers=n_workers) as pool:
        futures = {
            pool.schedule(run_single_simulation, args=[i, config, simulation_seeds[i]], timeout=SIM_TIMEOUT): i
            for i in range(config["N_SIMULATIONS"])
        }

        for f in tqdm(as_completed(futures), total=len(futures), desc="Simulations"):
            sim_id = futures[f]
            try:
                res, status = f.result()
                if res:
                    results_all.extend(res)
                summary_counts[status.lower()] = summary_counts.get(status.lower(), 0) + 1
            except PebbleTimeoutError:
                logging.warning(f"Sim {sim_id} timeout ({SIM_TIMEOUT}s)")
                summary_counts["timeout"] += 1
            except Exception as e:
                logging.error(f"Sim {sim_id} failed: {e}")
                summary_counts["error"] += 1

    return results_all, summary_counts
