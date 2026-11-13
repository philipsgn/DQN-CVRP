"""Direct synchronous unit test for FastAPI endpoint functions."""
from src.web.server import (
    list_instances,
    initialize_instance,
    execute_step,
    run_all_steps,
    create_custom_instance,
    run_benchmark,
    compare_duel,
    filter_district,
    normalize_district,
    init_comparison,
    compare_step,
    InitRequest,
    StepRequest,
    RunAllRequest,
    CustomInstanceRequest,
    BenchmarkRequest,
    CompareDuelRequest,
    FilterDistrictRequest,
    InitComparisonRequest,
    CompareStepRequest,
    NodeDemandConfig,
    DynamicNode,
    CreateDynamicScenarioRequest,
    geocode_location,
    create_dynamic_scenario
)
from src.cvrp.osrm_service import DynamicOSRMService


def run_tests():
    print("[1/6] Testing list_instances...", flush=True)
    res_inst = list_instances()
    assert "instances" in res_inst
    assert len(res_inst["instances"]) >= 2
    print(f"      Got {len(res_inst['instances'])} instances.", flush=True)

    print("[2/6] Testing initialize_instance for HCMC-20...", flush=True)
    init_req = InitRequest(instance_name="HCMC-District-20")
    init_res = initialize_instance(init_req)
    assert init_res["is_feasible"] is True
    assert init_res["initial_cost"] > 0
    print(f"      Cost: {init_res['initial_cost']} km, routes: {init_res['num_active_routes']}, feasible: {init_res['is_feasible']}", flush=True)

    print("[3/6] Testing execute_step with DDQN...", flush=True)
    step_req = StepRequest(
        instance_name="HCMC-District-20",
        routes=init_res["routes"],
        algorithm="ddqn",
        evals_used=0,
        step_count=0,
        initial_cost=init_res["initial_cost"]
    )
    step_res = execute_step(step_req)
    assert step_res["step"] == 1
    assert "operator_name" in step_res
    assert step_res["is_feasible"] is True
    print(f"      Chose: [{step_res['operator_name']}], new_cost: {step_res['new_cost']} km, improved: {step_res['improved']}", flush=True)

    print("[4/6] Testing run_all_steps with Greedy...", flush=True)
    run_req = RunAllRequest(instance_name="HCMC-District-20", algorithm="greedy", max_steps=5)
    run_res = run_all_steps(run_req)
    assert run_res["is_feasible"] is True
    assert run_res["total_steps"] > 0
    print(f"      RunAll finished: steps={run_res['total_steps']}, final_cost={run_res['final_cost']} km, feasible: {run_res['is_feasible']}", flush=True)

    print("[5/6] Testing create_custom_instance (District: Quận 1, K=4, Q=30, N=12)...", flush=True)
    custom_req = CustomInstanceRequest(
        num_customers=12,
        capacity=30,
        max_vehicles=4,
        district="Quận 1",
        seed=42
    )
    custom_res = create_custom_instance(custom_req)
    assert custom_res["is_feasible"] is True
    assert custom_res["num_customers"] == 12
    assert len(custom_res["nodes"]) == 13
    print(f"      Custom instance: {custom_res['instance_name']} generated with {custom_res['num_customers']} customers, initial cost={custom_res['initial_cost']} km, feasible={custom_res['is_feasible']}", flush=True)

    print("[6/7] Testing run_benchmark (Head-to-head 5 algorithms)...", flush=True)
    bench_req = BenchmarkRequest(
        instance_name=custom_res["instance_name"],
        budget_steps=10
    )
    bench_res = run_benchmark(bench_req)
    assert len(bench_res["leaderboard"]) == 5
    assert bench_res["winner"]["rank"] == 1
    assert "display_name" in bench_res["winner"]
    print(f"      Benchmark winner: {bench_res['winner']['display_name']} with final_cost={bench_res['winner']['final_cost']} km (savings: +{bench_res['winner']['savings_pct']}%)", flush=True)

    print("[7/8] Testing compare_duel (DDQN vs Greedy)...", flush=True)
    duel_req = CompareDuelRequest(
        instance_name=custom_res["instance_name"],
        model_a="ddqn",
        model_b="greedy",
        budget_steps=10
    )
    duel_res = compare_duel(duel_req)
    assert "model_a" in duel_res
    assert "model_b" in duel_res
    assert duel_res["model_a"]["final_cost"] > 0
    assert duel_res["model_b"]["final_cost"] > 0
    assert duel_res["model_a"]["is_feasible"] is True
    assert duel_res["model_b"]["is_feasible"] is True
    print(f"      Duel result: A ({duel_res['model_a']['display_name']})={duel_res['model_a']['final_cost']} km vs B ({duel_res['model_b']['display_name']})={duel_res['model_b']['final_cost']} km (Diff: {duel_res['diff_km']} km, Winner: {duel_res['winner_name']})", flush=True)

    print("[8/8] Testing Exact District Normalization & Filter Endpoint...", flush=True)
    # 1. Verify exact normalization does not collide
    assert normalize_district("Quận 1") == "quan_1"
    assert normalize_district("Quận 11") == "quan_11"
    assert normalize_district("Quận 10") == "quan_10"
    assert normalize_district("Quận 12") == "quan_12"
    assert normalize_district("Quận 1") != normalize_district("Quận 11")
    assert normalize_district("Quận Bình Thạnh") == "quan_binh_thanh"
    assert normalize_district("TP. Thủ Đức") == "tp_thu_duc"
    print("      District normalization tests passed without collision.", flush=True)

    # 2. Filter District with 0 customers in HCMC-20: 'Quận 11'
    q11_req = FilterDistrictRequest(instance_name="HCMC-District-20", district="Quận 11", algorithm="ddqn")
    q11_res = filter_district(q11_req)
    assert q11_res["num_customers"] == 0
    assert q11_res["initial_cost"] == 0.0
    assert len(q11_res["routes"]) == 0
    assert len(q11_res["nodes"]) == 1 # Only Depot
    assert q11_res["action_mask"] == [False, False, False, False]
    print(f"      Filter 'Quận 11': 0 customers, cost=0.0 km, action_mask={q11_res['action_mask']}.", flush=True)

    # 3. Filter District with >= 1 customer in HCMC-20: 'Quận 1'
    q1_req = FilterDistrictRequest(instance_name="HCMC-District-20", district="Quận 1", algorithm="ddqn")
    q1_res = filter_district(q1_req)
    assert q1_res["num_customers"] > 0
    assert q1_res["k_min"] == 1
    assert q1_res["num_active_routes"] == 1
    assert q1_res["is_feasible"] is True
    # Action mask with 1 route must mask cross-route operators
    assert q1_res["action_mask"] == [True, True, False, False]
    assert q1_res["chosen_action"] in [0, 1]  # 2-opt or intra-relocate only
    print(f"      Filter 'Quận 1': {q1_res['num_customers']} customers, {q1_res['num_active_routes']} route, cost={q1_res['initial_cost']} km, action_mask={q1_res['action_mask']}, chosen={q1_res['operator_name']}.", flush=True)

    # 4. Step execution on filtered instance with action mask
    step_filt_req = StepRequest(
        instance_name=q1_res["instance_name"],
        routes=q1_res["routes"],
        algorithm="ddqn",
        evals_used=0,
        step_count=0,
        initial_cost=q1_res["initial_cost"]
    )
    step_filt_res = execute_step(step_filt_req)
    assert step_filt_res["action_mask"] == [True, True, False, False]
    assert step_filt_res["operator_name"] in ["2-opt", "intra-relocate"]
    assert step_filt_res["is_feasible"] is True
    print(f"      Step on filtered sub-instance: chose [{step_filt_res['operator_name']}] from masked actions, feasible={step_filt_res['is_feasible']}.", flush=True)

    print("[9/9] Testing Pre-Comparison Configuration Gate & Step-by-Step Dual Benchmarking...", flush=True)
    # 1. Validation test: < 2 points must fail
    from fastapi import HTTPException
    try:
        init_comparison(InitComparisonRequest(
            base_instance="HCMC-District-20",
            selected_nodes=[NodeDemandConfig(node_id=1, demand=10)],
            capacity=30
        ))
        assert False, "Should have raised HTTPException for < 2 points"
    except HTTPException as e:
        assert e.status_code == 400
        print("      Validation check: < 2 points rejected with 400.", flush=True)

    # 2. Validation test: demand <= 0 or > Q must fail
    try:
        init_comparison(InitComparisonRequest(
            base_instance="HCMC-District-20",
            selected_nodes=[
                NodeDemandConfig(node_id=1, demand=0),
                NodeDemandConfig(node_id=2, demand=10)
            ],
            capacity=30
        ))
        assert False, "Should have raised HTTPException for demand == 0"
    except HTTPException as e:
        assert e.status_code == 400
        print("      Validation check: demand <= 0 rejected with 400.", flush=True)

    # 3. Valid comparison setup: 3 points, demands: 8, 10, 6 -> total 24 <= 30 => K_min = 1
    init_comp_res = init_comparison(InitComparisonRequest(
        base_instance="HCMC-District-20",
        selected_nodes=[
            NodeDemandConfig(node_id=1, demand=8),
            NodeDemandConfig(node_id=2, demand=10),
            NodeDemandConfig(node_id=3, demand=6)
        ],
        capacity=30,
        model_a="ddqn",
        model_b="greedy"
    ))
    assert init_comp_res["num_customers"] == 3
    assert init_comp_res["total_demand"] == 24
    assert init_comp_res["k_min"] == 1
    assert init_comp_res["num_active_routes"] == 1
    assert init_comp_res["is_feasible"] is True
    assert init_comp_res["action_mask"] == [True, True, False, False]
    assert len(init_comp_res["nodes"]) == 4 # Depot + 3 customers
    print(f"      Init comparison: {init_comp_res['num_customers']} points, total {init_comp_res['total_demand']} cargo, K_min={init_comp_res['k_min']} vehicle, cost={init_comp_res['initial_cost']} km.", flush=True)

    # 4. Synchronous Step comparison: Step 1
    step_comp_req = CompareStepRequest(
        instance_name=init_comp_res["instance_name"],
        routes_a=init_comp_res["routes"],
        routes_b=init_comp_res["routes"],
        model_a="ddqn",
        model_b="greedy",
        step_count=0,
        initial_cost=init_comp_res["initial_cost"]
    )
    step_comp_res = compare_step(step_comp_req)
    assert step_comp_res["step"] == 1
    assert "model_a" in step_comp_res
    assert "model_b" in step_comp_res
    assert step_comp_res["model_a"]["is_feasible"] is True
    assert step_comp_res["model_b"]["is_feasible"] is True
    assert step_comp_res["model_a"]["operator_name"] in ["2-opt", "intra-relocate"]
    assert step_comp_res["model_b"]["operator_name"] in ["2-opt", "intra-relocate"]
    print(f"      Compare Step 1: Model A ({step_comp_res['model_a']['algorithm']})={step_comp_res['model_a']['cost']} km vs Model B ({step_comp_res['model_b']['algorithm']})={step_comp_res['model_b']['cost']} km (Diff: {step_comp_res['diff_km']} km, Winner: {step_comp_res['winner_key']}).", flush=True)

    # 10. Testing Dynamic Scenario Dispatch & OSRM Engine
    print("[10/10] Testing Dynamic Scenario Platform & OSRM Engine...", flush=True)
    osrm = DynamicOSRMService()
    test_coords = [[10.8415, 106.8098], [10.7725, 106.6980], [10.7951, 106.7218]]
    d_mat, t_mat = osrm.compute_matrices(test_coords)
    assert d_mat.shape == (3, 3)
    assert t_mat.shape == (3, 3)
    assert d_mat[0, 1] > 0
    print(f"      OSRM/Haversine Matrix: shape={d_mat.shape}, depot->c1={d_mat[0, 1]} km, time={t_mat[0, 1]} min.", flush=True)

    # Geocoding test
    geo_res = geocode_location("Chợ Bến Thành")
    assert "results" in geo_res
    assert len(geo_res["results"]) > 0
    print(f"      Geocode 'Chợ Bến Thành': found {len(geo_res['results'])} matches (Top: {geo_res['results'][0]['name']}).", flush=True)

    # Dynamic scenario validation: Capacity overflow
    depot_node = DynamicNode(name="Kho FPT", lat=10.8415, lng=106.8098, demand=0, district="TP. Thủ Đức")
    cust_nodes = [
        DynamicNode(name="Điểm A", lat=10.7725, lng=106.6980, demand=25, district="Quận 1"),
        DynamicNode(name="Điểm B", lat=10.7951, lng=106.7218, demand=20, district="Bình Thạnh"),
    ]
    # Total demand = 45. If max_vehicles = 1, capacity = 30 -> overflow!
    overflow_req = CreateDynamicScenarioRequest(
        depot=depot_node,
        customers=cust_nodes,
        max_vehicles=1,
        capacity=30
    )
    try:
        create_dynamic_scenario(overflow_req)
        assert False, "Should have rejected demand overflow > fleet capacity"
    except HTTPException as e:
        assert e.status_code == 400
        print(f"      Feasibility rejection verified: {e.detail}", flush=True)

    # Dynamic scenario valid dispatch: max_vehicles = 3, capacity = 30 -> K_min = ceil(45/30) = 2
    valid_dyn_req = CreateDynamicScenarioRequest(
        depot=depot_node,
        customers=cust_nodes,
        max_vehicles=3,
        capacity=30,
        scenario_name="Test-Dynamic-Dispatch"
    )
    dyn_res = create_dynamic_scenario(valid_dyn_req)
    assert dyn_res["status"] == "success"
    assert dyn_res["num_customers"] == 2
    assert dyn_res["total_demand"] == 45
    assert dyn_res["k_min"] == 2
    assert dyn_res["is_feasible"] is True
    assert dyn_res["k_min"] <= len(dyn_res["routes"]) <= dyn_res["max_vehicles"]
    print(f"      Dynamic Scenario Created: '{dyn_res['scenario_name']}', {dyn_res['num_customers']} pts, initial cost={dyn_res['initial_cost']} km, K_min={dyn_res['k_min']}.", flush=True)

    print("\n>>> ALL DIRECT ENDPOINT TESTS PASSED SYNCHRONOUSLY! <<<", flush=True)


if __name__ == "__main__":
    run_tests()
