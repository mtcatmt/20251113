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
from data_generator import generate_artificial_data, create_dataset_iv_logit_score
from analysis import run_block_bootstrap_lingam
from evaluation import evaluate_by_variable_type


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
        # Dataset① Continuous
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
        # Dataset② Mixed
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
        # Dataset④ LogitScore
        # -----------------------------------------------------
        n_results_max += 1
        data_ts_logit = create_dataset_iv_logit_score(
            data_ts_mixed=data_ts_mixed,
            B_true=B_true,
            base_names=base_names,
            discrete_variable_names=disc_names,
            labels=labels_mixed,
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
