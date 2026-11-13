"""Automated Training Script for DQN-CVRP Checkpoint.
Trains Double DQN agent on HCMC instances and saves trained weights to checkpoints/ddqn_cvrp_hcmc.pt.
"""
from collections import deque
import random
from pathlib import Path
import numpy as np
import torch

from src.cvrp.instance import CVRPInstance
from src.cvrp.environment import LocalSearchEnv
from src.agents.dqn_agent import DQNAgent


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = 64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


def train():
    base_dir = Path(__file__).resolve().parent.parent.parent
    checkpoints_dir = base_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / "ddqn_cvrp_hcmc.pt"

    print(">>> Training Double DQN on HCMC Instances...")
    inst_20 = CVRPInstance.from_json(base_dir / "data" / "hcmc_instances" / "hcmc_district_20.json")
    inst_40 = CVRPInstance.from_json(base_dir / "data" / "hcmc_instances" / "hcmc_fpt_40.json")
    inst_synth = CVRPInstance.generate_synthetic("Synth-24", num_customers=24, seed=42)

    instances = [inst_20, inst_40, inst_synth]

    agent = DQNAgent(algorithm="ddqn", state_dim=28, action_dim=4, hidden_dim=128, learning_rate=1e-3)
    replay_buffer = ReplayBuffer(capacity=10000)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.94

    num_episodes = 45
    losses = []

    for ep in range(num_episodes):
        inst = instances[ep % len(instances)]
        env = LocalSearchEnv(instance=inst, max_evals=640, patience=25, feature_patience=30.0, seed=ep)
        state, info = env.reset(seed=ep)

        ep_reward = 0.0
        steps = 0

        while True:
            action, q_vals = agent.act(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            steps += 1

            if len(replay_buffer) >= 64:
                batch = replay_buffer.sample(batch_size=64)
                loss = agent.update(*batch)
                losses.append(loss)

            if done or steps >= 35:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (ep + 1) % 15 == 0:
            avg_loss = float(np.mean(losses[-50:])) if losses else 0.0
            print(f"Episode {ep + 1}/{num_episodes} | Steps: {steps} | Reward: {ep_reward:.2f} | Loss: {avg_loss:.4f} | Eps: {epsilon:.2f}")

    agent.save(checkpoint_path)
    print(f"✅ Checkpoint saved successfully to: {checkpoint_path}")


if __name__ == "__main__":
    train()

