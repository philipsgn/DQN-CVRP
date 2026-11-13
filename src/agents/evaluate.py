"""
MP-MOTO-CVRP (V9.2) — Benchmark Evaluation
Runs ablation and comparative benchmark on holdout test_scenarios.json (50 scenarios) across:
  1. Clarke-Wright (initial solution)
  2. Random Local Search (Ablation: non-RL operator selection)
  3. OR-Tools Guided Local Search (time limit 1.5s)
  4. Double DQN Agent (trained checkpoint, 4 actions)
Outputs metrics: Total distance (km), Savings vs CW (%), Gap vs OR-Tools (%), Latency ms.
"""

import sys
import json
import time
import random
import numpy as np
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.cvrp.matrix_manager import MatrixManager
from src.cvrp.motorbike_env import MotorbikeCVRPEnv, compute_total_cost
from src.cvrp.ortools_solver import solve_cvrp_ortools
from src.agents.train_motorbike_cvrp import MaskedQNetwork

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

def run_evaluation():
    print("=" * 70)
    print("MP-MOTO-CVRP (V9.2) — BENCHMARK EVALUATION (50 HOLDOUT SCENARIOS)")
    print("=" * 70)

    device = torch.device("cpu")
    mat_mgr = MatrixManager.get_instance()

    with open(DATA_DIR / "db" / "depots.json", "r", encoding="utf-8") as f:
        depots = {d["id"]: d for d in json.load(f)["depots"]}
    with open(DATA_DIR / "scenarios" / "test_scenarios.json", "r", encoding="utf-8") as f:
        test_scenarios = json.load(f)["scenarios"]

    model_path = MODELS_DIR / "best_val_model.pt"
    rl_agent = MaskedQNetwork(state_dim=28, num_actions=4).to(device)
    if model_path.exists():
        ckpt = torch.load(model_path, map_location=device)
        rl_agent.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded trained Double DQN from: {model_path} (Ep {ckpt.get('episode', '?')})")
    else:
        print("WARNING: Checkpoint not found, using untrained network.")
    rl_agent.eval()

    results = {
        "clarke_wright": {"distances": [], "times": []},
        "random_ls": {"distances": [], "times": []},
        "ortools": {"distances": [], "times": []},
        "double_dqn": {"distances": [], "times": []}
    }

    print(f"\nEvaluating on {len(test_scenarios)} geographic holdout test scenarios...")
    print(f"{'Scenario ID':<16} | {'Stores':<6} | {'CW (km)':<9} | {'Random (km)':<11} | {'OR-Tools':<9} | {'DDQN (km)':<9} | {'DDQN Sav%':<9}")
    print("-" * 80)

    for sc in test_scenarios:
        sc_id = sc["scenario_id"]
        d_id = sc["depot_id"]
        depot_obj = depots[d_id]
        node_ids = [depot_obj["snapped_node_id"]] + [st["snapped_node_id"] for st in sc["stores"]]
        sub_d, _ = mat_mgr.get_submatrix(node_ids)
        demands = [0.0] + [st["demand_kg"] for st in sc["stores"]]
        q_kg = sc["q_kg"]
        k_veh = sc["k_available"]

        # 1. Clarke-Wright
        t0 = time.time()
        cw_env = MotorbikeCVRPEnv(sub_d, demands, q_kg, k_veh, max_steps=1, seed=42)
        cw_env.reset()
        cw_dist = cw_env.initial_cost
        cw_time = (time.time() - t0) * 1000.0
        results["clarke_wright"]["distances"].append(cw_dist)
        results["clarke_wright"]["times"].append(cw_time)

        # 2. Random Local Search
        t0 = time.time()
        rnd_env = MotorbikeCVRPEnv(sub_d, demands, q_kg, k_veh, max_steps=20, seed=42)
        rnd_env.reset()
        rng = random.Random(42)
        for _ in range(rnd_env.max_steps):
            mask = rnd_env.get_action_mask()
            valid = [a for a in range(4) if mask[a]]
            if not valid:
                break
            _, _, done, _ = rnd_env.step(rng.choice(valid))
            if done:
                break
        rnd_dist = rnd_env.current_cost
        rnd_time = (time.time() - t0) * 1000.0
        results["random_ls"]["distances"].append(rnd_dist)
        results["random_ls"]["times"].append(rnd_time)

        # 3. OR-Tools
        t0 = time.time()
        ort_res = solve_cvrp_ortools(sub_d, demands, q_kg, k_veh, time_limit_s=1.0)
        ort_dist = ort_res["total_distance_m"]
        ort_time = (time.time() - t0) * 1000.0
        results["ortools"]["distances"].append(ort_dist)
        results["ortools"]["times"].append(ort_time)

        # 4. Double DQN
        t0 = time.time()
        dqn_env = MotorbikeCVRPEnv(sub_d, demands, q_kg, k_veh, max_steps=20, seed=42)
        s, mask = dqn_env.reset()
        for _ in range(dqn_env.max_steps):
            valid = [a for a in range(4) if mask[a]]
            if not valid:
                break
            with torch.no_grad():
                s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                m_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
                q_vals = rl_agent(s_t, m_t)
                a = int(torch.argmax(q_vals, dim=1).item())
            s, _, done, _ = dqn_env.step(a)
            mask = dqn_env.get_action_mask()
            if done:
                break
        dqn_dist = dqn_env.current_cost
        dqn_time = (time.time() - t0) * 1000.0
        results["double_dqn"]["distances"].append(dqn_dist)
        results["double_dqn"]["times"].append(dqn_time)

        ddqn_sav = ((cw_dist - dqn_dist) / max(1.0, cw_dist)) * 100.0
        print(
            f"{sc_id:<16} | {len(sc['stores']):<6} | "
            f"{cw_dist/1000.0:<9.2f} | {rnd_dist/1000.0:<11.2f} | "
            f"{ort_dist/1000.0:<9.2f} | {dqn_dist/1000.0:<9.2f} | "
            f"+{ddqn_sav:<8.2f}%"
        )

    print("=" * 80)
    print("SUMMARY BENCHMARK REPORT (50 SCENARIOS):")
    for method, data in results.items():
        avg_km = np.mean(data["distances"]) / 1000.0
        avg_ms = np.mean(data["times"])
        cw_base = np.mean(results["clarke_wright"]["distances"])
        sav_pct = ((cw_base - np.mean(data["distances"])) / cw_base) * 100.0
        ort_base = np.mean(results["ortools"]["distances"])
        gap_ort = ((np.mean(data["distances"]) - ort_base) / ort_base) * 100.0
        print(f"[{method.upper():<14}] Avg Dist: {avg_km:6.2f} km | Sav vs CW: {sav_pct:+5.2f}% | Gap vs ORT: {gap_ort:+5.2f}% | Latency: {avg_ms:6.1f} ms")
    print("=" * 80)

if __name__ == "__main__":
    run_evaluation()
