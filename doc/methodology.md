# Research Methodology

## Overview

This project investigates the application of **Deep Reinforcement Learning (DRL)** to dynamic vehicle route optimisation.

The research models the routing problem as a sequential decision-making task and evaluates whether a learned routing policy can make effective decisions under both static and changing travel conditions.

The primary learning approach is a **Double Deep Q-Network (Double DQN)** with invalid-action masking. Its performance is evaluated using established **CVRPLIB benchmark instances** and compared with classical optimisation and heuristic routing approaches.

The methodology consists of five main components:

1. formulation of the Capacitated Vehicle Routing Problem;
2. development of a Gymnasium-compatible routing environment;
3. implementation of the Double DQN agent;
4. simulation of dynamic travel disruption; and
5. benchmark evaluation against optimisation and heuristic baselines.

---

## 1. Problem Formulation

The research focuses on the **Capacitated Vehicle Routing Problem (CVRP)**.

A vehicle must serve a collection of geographically distributed customers while respecting vehicle-capacity constraints.

Each customer is associated with:

- a location;
- a demand quantity; and
- a service requirement.

The objective is to construct feasible routes while minimising total travel distance.

The depot represents the starting and returning location of the vehicle.

A route must therefore satisfy two fundamental constraints:

- each required customer must be served; and
- vehicle capacity must not be exceeded.

---

## 2. Reinforcement Learning Formulation

The routing problem is represented as a **Markov Decision Process (MDP)**.

At each decision step, the agent observes the current routing state and selects the next node to visit.

The environment then calculates the resulting travel cost, updates the vehicle state, and returns a reward.

This process continues until the routing episode is complete.

The implementation is built using **Gymnasium**, providing a standard interface for interaction between the environment and the learning agent.

---

## 3. State Representation

The observation supplied to the agent contains information required to make the next routing decision.

The state representation includes features describing:

- customer coordinates;
- customer demand;
- customers already visited;
- current vehicle position;
- remaining vehicle capacity;
- current-position representation; and
- travel-cost information associated with the current environment.

Under dynamic conditions, the observation also reflects changing travel multipliers.

This allows the policy to respond to the current routing state rather than relying exclusively on the original problem configuration.

---

## 4. Action Space

An action corresponds to selecting the **next routing node**.

Not every action is valid at every decision point.

For example, an action may be infeasible when:

- the customer has already been served; or
- the customer's demand exceeds the remaining vehicle capacity.

The implementation therefore uses **invalid-action masking**.

Invalid actions are removed from consideration during policy action selection.

Action masking is also incorporated into the learning process when calculating bootstrap targets.

This prevents the agent from assigning useful future value to routing decisions that violate problem constraints.

---

## 5. Reward Design

The reward function is based primarily on travel distance.

For each routing transition, the agent receives the negative travel cost associated with the selected movement.

Conceptually:

```text
reward = - travel_distance
```

Maximising cumulative reward therefore encourages the agent to minimise overall route distance.

This creates a direct relationship between the reinforcement-learning objective and the underlying vehicle-routing objective.

---

# Double Deep Q-Network

## 6. DQN Architecture

The main learning method is a **Double Deep Q-Network**.

A neural network approximates the action-value function:

```text
Q(s, a)
```

where:

- `s` represents the current routing state; and
- `a` represents a possible routing action.

The network estimates the expected long-term value of selecting each feasible action from the current state.

---

## 7. Double DQN

Standard DQN can overestimate action values because the same value estimates may influence both action selection and evaluation.

Double DQN reduces this effect by separating these operations.

The online network is used to select the next action, while the target network is used to evaluate the selected action.

This produces a more stable target estimate during training.

---

## 8. Experience Replay

Training transitions are stored in an **experience replay buffer**.

A transition contains information including:

```text
(state, action, reward, next_state, terminal_status)
```

Mini-batches are sampled from the replay buffer during training.

Experience replay improves learning efficiency and reduces the temporal correlation between consecutive training observations.

---

## 9. Target Network

A separate target network is maintained during training.

The target network provides more stable learning targets than using a rapidly changing online network for both prediction and target calculation.

Its parameters are periodically updated from the online Q-network.

---

## 10. Exploration

Training uses an **epsilon-greedy exploration strategy**.

During early training, the agent explores a larger proportion of available feasible actions.

As training progresses, epsilon decreases and the agent increasingly selects actions according to learned Q-values.

Evaluation is subsequently performed using greedy action selection rather than exploratory behaviour.

---

# Dynamic Routing Environment

## 11. Motivation

Static vehicle-routing benchmarks assume that travel conditions remain unchanged throughout execution.

Real logistics systems may experience changing conditions such as:

- congestion;
- temporary delays;
- varying road conditions; or
- other disruptions affecting travel cost.

The project therefore extends the static routing environment with controlled dynamic travel-cost changes.

---

## 12. Dynamic Travel Multipliers

Dynamic conditions are represented using travel multipliers.

Traffic multipliers are re-sampled every three environment steps.

The experimental framework contains three conditions:

| Condition | Travel-cost behaviour |
|---|---|
| Static | Multiplier remains at 1.0 |
| Low disruption | Log-normal disruption with σ = 0.15 |
| High disruption | Log-normal disruption with σ = 0.40 |

Random seeds are used so experimental conditions can be reproduced and competing methods can be evaluated under comparable disruption realisations.

---

# Benchmark Evaluation

## 13. CVRPLIB

The research uses established instances from **CVRPLIB**, a widely used benchmark collection for vehicle-routing research.

Eight benchmark instances of different sizes are included in the evaluation.

Using established benchmark instances allows the experimental solutions to be compared with known reference solution costs.

---

## 14. Baseline Methods

The learned Double DQN policy is evaluated against several alternative approaches.

### Google OR-Tools

Google OR-Tools provides the principal classical optimisation baseline.

Different experimental configurations investigate fixed planning and re-planning behaviour.

### Nearest-Neighbour Heuristic

A capacity-aware nearest-neighbour approach provides a simple deterministic heuristic baseline.

### Random Policy

A masked random policy provides a lower performance reference.

These baselines help establish whether improvements are attributable to learned routing behaviour rather than simply producing feasible routes.

---

# Experimental Reliability

## 15. Multiple Random Seeds

Reinforcement-learning performance can vary because of:

- neural-network initialisation;
- exploration;
- replay-buffer sampling; and
- stochastic environment behaviour.

The experimental pipeline therefore uses multiple random seeds.

This reduces reliance on a single favourable training run.

---

## 16. Statistical Evaluation

The analysis extends beyond reporting mean performance.

The project includes:

- descriptive statistics;
- **Wilcoxon signed-rank tests**;
- **rank-biserial effect sizes**;
- **bootstrap confidence intervals**; and
- disruption degradation analysis.

This allows differences between methods to be evaluated in terms of both observed performance and statistical evidence.

---

# Methodological Scope

The methodology is designed as a controlled experimental framework rather than a production logistics system.

The primary purpose is to investigate:

> **how a learned sequential routing policy compares with established optimisation and heuristic approaches when routing costs become dynamic.**

The framework therefore provides a foundation for further investigation into adaptive optimisation, reinforcement learning, uncertainty, and intelligent sequential decision-making.

---

## Implementation

The primary implementation can be found under:

```text
src/
├── agents/
├── analysis/
├── baselines/
├── envs/
└── experiments/
```

For the complete experimental configuration, see:

**[Experimental Design](experiments.md)**

For the empirical findings, see:

**[Results and Statistical Analysis](results.md)**

For instructions to reproduce the research, see:

**[Reproducibility Guide](reproducibility.md)**

For methodological limitations and possible extensions, see:

**[Limitations and Future Research](limitations.md)**