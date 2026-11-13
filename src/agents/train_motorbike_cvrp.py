"""
MP-MOTO-CVRP (V9.2) — Double DQN Training Pipeline (4 Actions)
Trains a Double DQN Agent with Action Masking and 3-Stage Curriculum Learning.
Evaluates periodically against OR-Tools reference solutions on 30 validation scenarios.
Saves best model to models/best_val_model.pt.
"""

import os
import sys
import json
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.cvrp.matrix_manager import MatrixManager
from src.cvrp.motorbike_env import MotorbikeCVRPEnv, compute_total_cost
from src.cvrp.ortools_solver import solve_cvrp_ortools

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "data"

class MaskedQNetwork(nn.Module):
    def __init__(self, state_dim: int = 28, num_actions: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q_values = self.fc(state)
        if action_mask is not None:
            q_values = torch.where(action_mask, q_values, torch.tensor(-1e9, device=q_values.device))
        return q_values

class ReplayBuffer:
    def __init__(self, capacity: int = 25000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(np.array(next_masks), dtype=torch.bool)
        )

    def __len__(self):
        return len(self.buffer)

def sample_curriculum_instance(
    stage: int,
    depots: List[Dict[str, Any]],
    stores: List[Dict[str, Any]],
    mat_mgr: MatrixManager,
    rng: random.Random
) -> Tuple[np.ndarray, List[float], float, int]:
    if stage == 1:
        n_stores = rng.randint(10, 20)
    elif stage == 2:
        n_stores = rng.randint(20, 35)
    else:
        n_stores = rng.randint(35, 50)

    train_depots = [d for d in depots if d.get("split") == "train"] or depots
    train_stores = [s for s in stores if s.get("split") == "train"] or stores

    depot = rng.choice(train_depots)
    selected_stores = rng.sample(train_stores, min(n_stores, len(train_stores)))

    q_kg = rng.choice([40.0, 50.0, 60.0, 80.0])

    demands = [0.0]
    node_ids = [depot["snapped_node_id"]]

    for st in selected_stores:
        n_pkg = rng.randint(st["typical_pkg_range"][0], st["typical_pkg_range"][1])
        w_pkg = 5.0
        d_i = n_pkg * w_pkg
        if d_i > q_kg:
            d_i = q_kg - 2.0
        demands.append(d_i)
        node_ids.append(st["snapped_node_id"])

    total_demand = sum(demands)
    k_min = math.ceil(total_demand / q_kg)
    k_vehicles = k_min + rng.randint(1, 3)

    sub_dist, _ = mat_mgr.get_submatrix(node_ids)
    return sub_dist, demands, q_kg, k_vehicles

def train_double_dqn(
    total_episodes: int = 1500,
    batch_size: int = 64,
    lr: float = 1e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.996,
    eval_interval: int = 150,
    save_path: Optional[str] = None
):
    print("=" * 65)
    print("STARTING MP-MOTO-CVRP DOUBLE DQN TRAINING (4 ACTIONS)")
    print(f"Total episodes: {total_episodes} | Batch size: {batch_size} | Eval interval: {eval_interval}")
    print("=" * 65)

    device = torch.device("cpu")
    print(f"Training on device: {device}")

    mat_mgr = MatrixManager.get_instance()
    with open(DATA_DIR / "db" / "depots.json", "r", encoding="utf-8") as f:
        depots = json.load(f)["depots"]
    with open(DATA_DIR / "db" / "stores.json", "r", encoding="utf-8") as f:
        stores = json.load(f)["stores"]
    with open(DATA_DIR / "scenarios" / "val_scenarios.json", "r", encoding="utf-8") as f:
        val_scenarios = json.load(f)["scenarios"]

    # Precompute / Load OR-Tools reference solutions for validation
    ref_ortools_path = DATA_DIR / "scenarios" / "reference_ortools.json"
    val_references = {}
    if ref_ortools_path.exists():
        with open(ref_ortools_path, "r", encoding="utf-8") as f:
            val_references = json.load(f)

    # Compute missing validation references
    missing_scens = [sc for sc in val_scenarios if sc["scenario_id"] not in val_references]
    if missing_scens:
        print(f"Computing OR-Tools reference solutions for {len(missing_scens)} validation scenarios...")
        for sc in missing_scens:
            sc_id = sc["scenario_id"]
            d_id = sc["depot_id"]
            depot_obj = next(d for d in depots if d["id"] == d_id)
            node_ids = [depot_obj["snapped_node_id"]] + [st["snapped_node_id"] for st in sc["stores"]]
            sub_d, _ = mat_mgr.get_submatrix(node_ids)
            demands = [0.0] + [st["demand_kg"] for st in sc["stores"]]
            sol = solve_cvrp_ortools(sub_d, demands, sc["q_kg"], sc["k_available"], time_limit_s=1.5)
            val_references[sc_id] = sol["total_distance_m"]
        with open(ref_ortools_path, "w", encoding="utf-8") as f:
            json.dump(val_references, f, indent=2)

    # Initialize Networks
    policy_net = MaskedQNetwork(state_dim=28, num_actions=4).to(device)
    target_net = MaskedQNetwork(state_dim=28, num_actions=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=25000)

    epsilon = epsilon_start
    best_val_gap = float("inf")
    rng = random.Random(42)

    curriculum_stage = 1
    recent_losses = deque(maxlen=100)

    for episode in range(1, total_episodes + 1):
        if episode <= int(total_episodes * 0.35):
            curriculum_stage = 1
        elif episode <= int(total_episodes * 0.70):
            curriculum_stage = 2
        else:
            curriculum_stage = 3

        sub_dist, demands, q_kg, k_veh = sample_curriculum_instance(
            curriculum_stage, depots, stores, mat_mgr, rng
        )

        env = MotorbikeCVRPEnv(
            dist_m=sub_dist,
            demands_kg=demands,
            capacity_kg=q_kg,
            k_vehicles=k_veh,
            max_steps=20,
            seed=rng.randint(1, 100000)
        )

        state, action_mask = env.reset()

        for t in range(env.max_steps):
            valid_actions = [i for i, v in enumerate(action_mask) if v]
            if not valid_actions:
                break

            if rng.random() < epsilon:
                action = rng.choice(valid_actions)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    mask_t = torch.tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
                    q_vals = policy_net(s_t, mask_t)
                    action = int(torch.argmax(q_vals, dim=1).item())

            next_state, reward, done, info = env.step(action)
            next_mask = env.get_action_mask()

            buffer.push(state, action, reward, next_state, done, next_mask)
            state = next_state
            action_mask = next_mask

            if len(buffer) >= batch_size:
                b_states, b_actions, b_rewards, b_next_states, b_dones, b_next_masks = buffer.sample(batch_size)
                b_states = b_states.to(device)
                b_actions = b_actions.to(device)
                b_rewards = b_rewards.to(device)
                b_next_states = b_next_states.to(device)
                b_dones = b_dones.to(device)
                b_next_masks = b_next_masks.to(device)

                current_q = policy_net(b_states).gather(1, b_actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_policy_q = policy_net(b_next_states, b_next_masks)
                    next_actions = torch.argmax(next_policy_q, dim=1, keepdim=True)
                    next_target_q = target_net(b_next_states).gather(1, next_actions).squeeze(1)
                    target_q = b_rewards + (1.0 - b_dones) * gamma * next_target_q

                loss = nn.SmoothL1Loss()(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

                recent_losses.append(loss.item())

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % 15 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Evaluation
        if episode % eval_interval == 0 or episode == total_episodes:
            val_gaps = []
            val_cw_savings = []

            policy_net.eval()
            for sc in val_scenarios:
                sc_id = sc["scenario_id"]
                d_id = sc["depot_id"]
                depot_obj = next(d for d in depots if d["id"] == d_id)
                node_ids = [depot_obj["snapped_node_id"]] + [st["snapped_node_id"] for st in sc["stores"]]
                sub_d, _ = mat_mgr.get_submatrix(node_ids)
                demands = [0.0] + [st["demand_kg"] for st in sc["stores"]]

                v_env = MotorbikeCVRPEnv(
                    dist_m=sub_d,
                    demands_kg=demands,
                    capacity_kg=sc["q_kg"],
                    k_vehicles=sc["k_available"],
                    max_steps=20,
                    seed=12345
                )

                v_state, v_mask = v_env.reset()
                cw_cost = v_env.initial_cost

                for _ in range(v_env.max_steps):
                    valid = [i for i, v in enumerate(v_mask) if v]
                    if not valid:
                        break
                    with torch.no_grad():
                        s_t = torch.tensor(v_state, dtype=torch.float32, device=device).unsqueeze(0)
                        m_t = torch.tensor(v_mask, dtype=torch.bool, device=device).unsqueeze(0)
                        q_vals = policy_net(s_t, m_t)
                        a = int(torch.argmax(q_vals, dim=1).item())
                    v_state, _, v_done, _ = v_env.step(a)
                    v_mask = v_env.get_action_mask()
                    if v_done:
                        break

                final_cost = v_env.current_cost
                ortools_ref = val_references.get(sc_id, cw_cost)

                gap = ((final_cost - ortools_ref) / max(1.0, ortools_ref)) * 100.0
                cw_saving = ((cw_cost - final_cost) / max(1.0, cw_cost)) * 100.0

                val_gaps.append(gap)
                val_cw_savings.append(cw_saving)

            policy_net.train()

            mean_val_gap = float(np.mean(val_gaps))
            mean_cw_saving = float(np.mean(val_cw_savings))
            avg_loss = float(np.mean(recent_losses)) if recent_losses else 0.0

            print(
                f"[Ep {episode:04d}/{total_episodes}] "
                f"Stage {curriculum_stage} | "
                f"Loss: {avg_loss:.4f} | "
                f"Eps: {epsilon:.3f} | "
                f"CW Saving: +{mean_cw_saving:.2f}% | "
                f"Val Gap vs OR-Tools: {mean_val_gap:+.2f}%"
            )

            if mean_val_gap < best_val_gap:
                best_val_gap = mean_val_gap
                target_save = save_path or str(MODELS_DIR / "best_val_model.pt")
                torch.save({
                    "model_state_dict": policy_net.state_dict(),
                    "val_gap": best_val_gap,
                    "cw_saving": mean_cw_saving,
                    "episode": episode,
                    "curriculum_stage": curriculum_stage
                }, target_save)
                print(f"  >>> New Best Model saved to {target_save} (Val Gap: {best_val_gap:+.2f}%)")

    print("=" * 65)
    print(f"TRAINING COMPLETE. Best Validation Gap vs OR-Tools: {best_val_gap:+.2f}%")
    print(f"Model saved at: {MODELS_DIR / 'best_val_model.pt'}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500, help="Number of pilot training episodes")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_interval", type=int, default=150)
    args = parser.parse_args()

    train_double_dqn(
        total_episodes=args.episodes,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval
    )
