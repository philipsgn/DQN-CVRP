"""Implementation of the 5 Experimental Protocols from the Group 10 Final Report:
- A:  max_evals = 2560, patience = 20
- A2: max_evals = 2560, patience = None (Main exact budget)
- A3: max_evals = 2560, patience = None, feature_patience = 30 (Fixed feature shift)
- B:  time_limit = 5.0 s, patience = 100
- B2: time_limit = 0.1 s, patience = None (Wall-clock CPU budget)
"""
import time
from typing import Any, Dict, Optional
import numpy as np

from ..cvrp.instance import CVRPInstance
from ..cvrp.environment import LocalSearchEnv


def run_protocol(
    agent_or_policy: Any,
    instance: CVRPInstance,
    protocol: str = "A2",
    seed: int = 42
) -> Dict[str, Any]:
    """Run an optimization episode under one of the 5 protocols."""
    protocol = protocol.upper()

    # Configure protocol parameters
    if protocol == "A":
        max_evals = 2560
        patience = 20
        feature_patience = 20.0
        time_limit = None
    elif protocol == "A2":
        max_evals = 2560
        patience = None
        feature_patience = 1e6  # effectively streak ~ 0 as in original run
        time_limit = None
    elif protocol == "A3":
        max_evals = 2560
        patience = None
        feature_patience = 30.0  # Normalized correctly
        time_limit = None
    elif protocol == "B":
        max_evals = 1000000
        patience = 100
        feature_patience = 100.0
        time_limit = 5.0
    elif protocol == "B2":
        max_evals = 1000000
        patience = None
        feature_patience = 30.0
        time_limit = 0.1
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    env = LocalSearchEnv(
        instance=instance,
        max_evals=max_evals,
        patience=patience,
        feature_patience=feature_patience,
        seed=seed
    )

    state, info = env.reset(seed=seed)
    start_time = time.perf_counter()
    stop_reason = ""

    while True:
        # Check time limit if applicable
        elapsed = time.perf_counter() - start_time
        if time_limit is not None and elapsed >= time_limit:
            stop_reason = "time_limit"
            break

        # Select action
        if hasattr(agent_or_policy, "select_operator"):
            # Greedy policy
            action, _ = agent_or_policy.select_operator(env.routes, instance)
        elif hasattr(agent_or_policy, "act"):
            action, _ = agent_or_policy.act(state)
        else:
            action = 0

        next_state, reward, terminated, truncated, step_info = env.step(action)
        state = next_state

        if terminated or truncated:
            stop_reason = step_info.get("stop_reason", "completed")
            break

    total_time = time.perf_counter() - start_time
    is_feasible, feas_msg = instance.is_feasible(env.routes)

    impr_pct = ((env.initial_cost - env.current_cost) / max(1e-6, env.initial_cost)) * 100.0

    return {
        "protocol": protocol,
        "instance_name": instance.name,
        "initial_cost": float(env.initial_cost),
        "final_cost": float(env.current_cost),
        "improvement_pct": float(impr_pct),
        "evals_used": int(env.evals_used),
        "steps_taken": int(env.steps_taken),
        "runtime_sec": float(total_time),
        "stop_reason": stop_reason,
        "is_feasible": is_feasible,
        "feasibility_msg": feas_msg,
        "final_routes": env.routes
    }

