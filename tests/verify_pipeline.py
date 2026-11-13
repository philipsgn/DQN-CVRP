"""Verification test script for the DQN-CVRP pipeline and web server."""
from src.cvrp import CVRPInstance, clarke_wright_savings, best_move, StateExtractorF28, LocalSearchEnv
from src.agents import DQNAgent, RandomPolicy, GreedyPolicy
from src.evaluation import IndependentValidator
from src.protocols import run_protocol
from src.web.server import app

def main():
    print("=== Testing Core Imports and Models ===")
    assert app is not None
    print("[1/5] FastAPI app loaded successfully.")

    inst = CVRPInstance.from_json("data/hcmc_instances/hcmc_district_20.json")
    assert inst.num_customers == 20
    print(f"[2/5] Loaded HCMC-20 instance. Depot: {inst.node_names[0]}")

    routes = clarke_wright_savings(inst)
    cost = inst.solution_cost(routes)
    val = IndependentValidator.validate(inst, routes)
    assert val["is_valid"] is True
    print(f"[3/5] Clarke-Wright solution: {len(routes)} routes, {cost:.2f} km. Validator: {val['status']}")

    agent = DQNAgent(algorithm="ddqn")
    env = LocalSearchEnv(instance=inst, max_evals=2560, seed=42)
    s0, _ = env.reset(seed=42)
    assert s0.shape == (28,)
    a, q_vals = agent.act(s0)
    s1, r, done, trunc, info = env.step(a)
    print(f"[4/5] DDQN action: {a} ({info['operator_name']}), Delta: {info['delta_cost']:.2f} km, Q-values: {q_vals}")

    print("[5/5] Running sample episode under Protocol A2...")
    res = run_protocol(agent, inst, protocol="A2", seed=42)
    print(f"      Finished protocol A2: initial={res['initial_cost']:.2f} km, final={res['final_cost']:.2f} km, impr={res['improvement_pct']:.2f}%, evals={res['evals_used']}")
    assert res["is_feasible"] is True

    print("\n>>> ALL CHECKS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    main()

