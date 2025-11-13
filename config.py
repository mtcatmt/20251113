# config.py

# --- グローバルシードの設定 ---
GLOBAL_SEED = 42 # 好きな固定値に変更可能

# --- 設定パラメータ ---

SIMULATION_CONFIG = {
    "N_SIMULATIONS":100,
    "N_SAMPLES": 2000,
    "LAG":1, 
    "N_VARS": 4,
    "EDGE_PROB": 0.2,
    "BLOCK_SIZE": 50,
    "BOOTSTRAP_SAMPLES": 100,
    "PROBABILITY_THRESHOLD": 0.65,
    "SIMULATION_TIMEOUT": 1800, # (秒)
}

# SIMULATION_CONFIG = {
#     "N_SIMULATIONS":15,
#     "N_SAMPLES": 1000,
#     "LAG":1, 
#     "N_VARS": 4,
#     "EDGE_PROB": 0.2,
#     "BLOCK_SIZE": 50,
#     "BOOTSTRAP_SAMPLES": 10,
#     "PROBABILITY_THRESHOLD": 0.65,
#     "SIMULATION_TIMEOUT": 300, # (秒)
    
# }