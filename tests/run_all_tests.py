"""Comprehensive test runner for all DQN-CVRP unit tests."""
import sys
from pathlib import Path
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def run():
    print("==================================================")
    print("        DQN-CVRP AUTOMATED UNIT TEST SUITE        ")
    print("==================================================")

    # 1. Operators
    try:
        from tests.test_operators import (
            sample_instance,
            test_2opt_operator,
            test_intra_relocate_operator,
            test_inter_relocate_operator,
            test_inter_swap_operator,
            test_best_move_dispatcher
        )
        from src.cvrp.instance import CVRPInstance
        inst = CVRPInstance.generate_synthetic(name="Test", num_customers=15, seed=42)
        test_2opt_operator(inst)
        test_intra_relocate_operator(inst)
        test_inter_relocate_operator(inst)
        test_inter_swap_operator(inst)
        test_best_move_dispatcher(inst)
        print("[PASS] 4 Local Search Operators (2-opt, intra, inter-rel, inter-swap)")
    except Exception as e:
        print(f"[FAIL] Operators: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Clarke-Wright
    try:
        from tests.test_clarke_wright import test_clarke_wright_synthetic, test_clarke_wright_hcmc
        test_clarke_wright_synthetic()
        test_clarke_wright_hcmc()
        print("[PASS] Clarke-Wright Savings Heuristic (Synthetic & HCMC-20)")
    except Exception as e:
        print(f"[FAIL] Clarke-Wright: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. State F28
    try:
        from tests.test_state_f28 import test_f28_shape_and_range
        test_f28_shape_and_range()
        print("[PASS] State Extractor F28 (28 dimensions, streak normalization)")
    except Exception as e:
        print(f"[FAIL] State F28: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Validator
    try:
        from tests.test_validator import test_validator_on_valid_solution, test_validator_detects_violations
        test_validator_on_valid_solution()
        test_validator_detects_violations()
        print("[PASS] Independent Constraint Validator (100% Feasibility Checks)")
    except Exception as e:
        print(f"[FAIL] Validator: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Web API
    try:
        from tests.test_web_api import run_tests
        run_tests()
        print("[PASS] Web API Endpoints (/init, /step, /run_all, /instances)")
    except Exception as e:
        print(f"[FAIL] Web API: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 6. MP-MOTO-CVRP V9.2 Feasibility Gate
    try:
        from tests.test_feasibility_gate import (
            test_tier1_overweight_item,
            test_tier2_total_demand_exceeded,
            test_tier3_binpacking_infeasible,
            test_tier4_unreachable_node,
            test_all_tiers_pass
        )
        test_tier1_overweight_item()
        test_tier2_total_demand_exceeded()
        test_tier3_binpacking_infeasible()
        test_tier4_unreachable_node()
        test_all_tiers_pass()
        print("[PASS] MP-MOTO-CVRP V9.2 Multi-Tier Feasibility Gate (4 Tiers)")
    except Exception as e:
        print(f"[FAIL] Feasibility Gate: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 7. MP-MOTO-CVRP V9.2 Geometry Consistency & Directed Edges
    try:
        from tests.test_geometry_consistency import test_geometry_consistency_and_directed_edges
        test_geometry_consistency_and_directed_edges()
        print("[PASS] Geometric Invariants & Directed OSM Edges (Single Source of Truth)")
    except Exception as e:
        print(f"[FAIL] Geometry Consistency: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 8. MP-MOTO-CVRP V9.2 Simulation On-Route
    try:
        from tests.test_simulation_on_route import test_simulation_stays_on_assigned_route
        test_simulation_stays_on_assigned_route()
        print("[PASS] Motorbike Simulation Bounded on Polyline (<= 5m Deviation)")
    except Exception as e:
        print(f"[FAIL] Simulation On-Route: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 9. MP-MOTO-CVRP V9.2 UI Contract
    try:
        from tests.test_ui_contract import (
            test_api_contract_success_schema,
            test_api_contract_400_empty_stores,
            test_api_contract_422_feasibility_rejection,
            test_api_contract_snap_endpoint
        )
        test_api_contract_success_schema()
        test_api_contract_400_empty_stores()
        test_api_contract_422_feasibility_rejection()
        test_api_contract_snap_endpoint()
        print("[PASS] API & UI Contract (Zero-Broken-Numbers, Schemas & Snap Endpoint)")
    except Exception as e:
        print(f"[FAIL] UI Contract: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 10. Neural LNS & Hard Fleet Cap (K_used <= K)
    try:
        from tests.test_neural_lns_fleet_cap import (
            test_hard_fleet_cap_consolidation,
            test_neural_lns_env_step,
            test_all_three_models_dispatch
        )
        test_hard_fleet_cap_consolidation()
        test_neural_lns_env_step()
        test_all_three_models_dispatch()
        print("[PASS] Neural LNS & Hard Fleet Cap Enforcement (K_used <= K & Outperforms Baselines)")
    except Exception as e:
        print(f"[FAIL] Neural LNS & Fleet Cap: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n==================================================")
    print("  ALL 10 TEST MODULES PASSED (100% SUCCESS RATE)  ")
    print("==================================================")

if __name__ == "__main__":
    run()

