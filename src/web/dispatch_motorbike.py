"""
MP-MOTO-CVRP (V9.2) — Warehouse Dispatch Motorbike Router
Provides:
  - POST /api/dispatch-motorbike (Full solver dispatch with single source of truth geometry)
  - POST /api/motorbike/snap (Map click snapping with 150m validation threshold)
  - GET /api/motorbike/depots
  - GET /api/motorbike/stores
  - GET /api/motorbike/meta
"""

import json
import math
import time
import numpy as np
import torch
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..cvrp.feasibility import check_feasibility
from ..cvrp.matrix_manager import MatrixManager
from ..cvrp.motorbike_env import MotorbikeCVRPEnv, compute_route_cost, compute_total_cost, consolidate_routes_to_k
from ..cvrp.neural_lns_env import NeuralLNSEnv
from ..agents.train_motorbike_cvrp import MaskedQNetwork
from ..agents.train_neural_lns import NeuralLNSNetwork

router = APIRouter(prefix="/api", tags=["Motorbike Dispatch"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

VEHICLE_COLORS = [
    "#007AFF", # iOS Blue
    "#FF3B30", # iOS Red
    "#34C759", # iOS Green
    "#FF9500", # iOS Orange
    "#AF52DE", # iOS Purple
    "#5856D6", # iOS Indigo
    "#00C7BE", # iOS Teal
    "#FF2D55", # iOS Pink
    "#FFCC00", # iOS Yellow
    "#8E8E93"  # iOS Gray
]

DEPOTS_CACHE: List[Dict[str, Any]] = []
STORES_CACHE: List[Dict[str, Any]] = []
GRAPH_CACHE: Optional[nx.DiGraph] = None
RL_FINETUNED_CACHE: Optional[MaskedQNetwork] = None
RL_BASE_CACHE: Optional[MaskedQNetwork] = None
RL_NEURAL_LNS_CACHE: Optional[NeuralLNSNetwork] = None

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def get_graph() -> nx.DiGraph:
    global GRAPH_CACHE
    if GRAPH_CACHE is None:
        # Reuse the graph already loaded by MatrixManager to avoid parsing the
        # GraphML twice on the first dispatch (keeps cold-start latency low).
        mat_mgr = MatrixManager.get_instance()
        if mat_mgr.graph is not None:
            GRAPH_CACHE = mat_mgr.graph
        else:
            graph_path = DATA_DIR / "graph" / "hcm_motorbike.graphml"
            if graph_path.exists():
                GRAPH_CACHE = nx.read_graphml(graph_path)
            else:
                raise RuntimeError("GraphML not found. Run scripts/build_dataset.py first.")
    return GRAPH_CACHE

def get_depots(force_reload: bool = False) -> List[Dict[str, Any]]:
    global DEPOTS_CACHE
    if not DEPOTS_CACHE or force_reload:
        depots_path = DATA_DIR / "db" / "depots.json"
        if depots_path.exists():
            with open(depots_path, "r", encoding="utf-8") as f:
                DEPOTS_CACHE = json.load(f)["depots"]
    return DEPOTS_CACHE

def get_stores(force_reload: bool = False) -> List[Dict[str, Any]]:
    global STORES_CACHE
    if not STORES_CACHE or force_reload:
        stores_path = DATA_DIR / "db" / "stores.json"
        if stores_path.exists():
            with open(stores_path, "r", encoding="utf-8") as f:
                STORES_CACHE = json.load(f)["stores"]
    return STORES_CACHE

def get_finetuned_model() -> Optional[MaskedQNetwork]:
    global RL_FINETUNED_CACHE
    if RL_FINETUNED_CACHE is None:
        model_path = MODELS_DIR / "best_val_model.pt"
        if model_path.exists():
            try:
                model = MaskedQNetwork(state_dim=28, num_actions=4)
                ckpt = torch.load(model_path, map_location="cpu")
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                RL_FINETUNED_CACHE = model
                print("[DispatchMotorbike] Loaded Finetuned Double DQN model.")
            except Exception as e:
                print(f"[DispatchMotorbike] Failed to load finetuned model: {e}")
                RL_FINETUNED_CACHE = None
    return RL_FINETUNED_CACHE

def get_base_model() -> Optional[MaskedQNetwork]:
    global RL_BASE_CACHE
    if RL_BASE_CACHE is None:
        model_path = MODELS_DIR / "base_ddqn_model.pt"
        if model_path.exists():
            try:
                model = MaskedQNetwork(state_dim=28, num_actions=4)
                ckpt = torch.load(model_path, map_location="cpu")
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                RL_BASE_CACHE = model
                print("[DispatchMotorbike] Loaded Base Double DQN model.")
            except Exception as e:
                print(f"[DispatchMotorbike] Failed to load base model: {e}")
                RL_BASE_CACHE = None
    return RL_BASE_CACHE

def get_neural_lns_model() -> Optional[NeuralLNSNetwork]:
    global RL_NEURAL_LNS_CACHE
    if RL_NEURAL_LNS_CACHE is None:
        model_path = MODELS_DIR / "best_neural_lns.pt"
        if model_path.exists():
            try:
                model = NeuralLNSNetwork(state_dim=36, num_actions=6)
                ckpt = torch.load(model_path, map_location="cpu")
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                RL_NEURAL_LNS_CACHE = model
                print("[DispatchMotorbike] Loaded Neural LNS model.")
            except Exception as e:
                print(f"[DispatchMotorbike] Failed to load Neural LNS model: {e}")
                RL_NEURAL_LNS_CACHE = None
    return RL_NEURAL_LNS_CACHE

# -------------------------------------------------------------
# Request Schemas
# -------------------------------------------------------------
class SnapRequest(BaseModel):
    lat: float
    lon: float

class StoreItem(BaseModel):
    id: str
    n_pkg: int = Field(default=1, ge=1)
    weight_per_pkg_kg: Optional[float] = None

class FleetConfig(BaseModel):
    # No artificial 6/10/30-vehicle ceiling: the fleet can scale with the batch.
    k: int = Field(default=3, ge=1, le=2000)
    q_kg: int = Field(default=50, ge=20, le=120)

class DispatchRequest(BaseModel):
    depot: Dict[str, Any]
    stores: List[StoreItem]
    fleet: FleetConfig
    options: Optional[Dict[str, Any]] = None

# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@router.get("/motorbike/depots")
def list_depots():
    return {"depots": get_depots()}

@router.get("/motorbike/stores")
def list_stores(district: Optional[str] = None):
    all_stores = get_stores()
    if district and district != "ALL":
        dist_list = [d.strip() for d in district.split(",") if d.strip()]
        if dist_list:
            filtered = [s for s in all_stores if s.get("district") in dist_list]
            return {"stores": filtered}
    return {"stores": all_stores}

@router.get("/motorbike/meta")
def get_meta():
    return {
        "num_depots": len(get_depots()),
        "num_stores": len(get_stores()),
        "graph_nodes": get_graph().number_of_nodes(),
        "graph_edges": get_graph().number_of_edges(),
        "supported_capacities_kg": [40, 50, 60, 80],
        "default_pkg_weight_kg": 5.0
    }

@router.post("/motorbike/snap")
def snap_coordinate(req: SnapRequest):
    """
    Snaps a user-clicked coordinate to the nearest node in the Largest SCC of HCMC Motorbike Graph.
    Enforces the <= 150m threshold rule.
    """
    G = get_graph()
    best_nid = None
    best_dist = float("inf")
    best_lat = 0.0
    best_lon = 0.0

    for nid, data in G.nodes(data=True):
        n_lat = float(data["lat"])
        n_lon = float(data["lon"])
        d = haversine_m(req.lat, req.lon, n_lat, n_lon)
        if d < best_dist:
            best_dist = d
            best_nid = int(nid)
            best_lat = n_lat
            best_lon = n_lon

    if best_dist > 150.0:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "SNAP_OUT_OF_BOUNDS",
                "reason": f"Vị trí vừa chọn nằm ngoài mạng lưới đường xe máy TP.HCM ({best_dist:.1f}m > 150m).",
                "metadata": {"snap_distance_m": round(best_dist, 1)}
            }
        )

    return {
        "snapped_node_id": best_nid,
        "lat": best_lat,
        "lon": best_lon,
        "snap_distance_m": round(best_dist, 1)
    }

@router.post("/dispatch-motorbike")
def dispatch_motorbike(req: DispatchRequest):
    t_start = time.time()

    if not req.stores:
        raise HTTPException(status_code=400, detail="Danh sách cửa hàng giao hàng đang trống.")

    depots_list = get_depots()
    stores_list = get_stores()
    stores_lookup = {s["id"]: s for s in stores_list}

    # Resolve depot
    depot_id = req.depot.get("id")
    depot_obj = next((d for d in depots_list if d["id"] == depot_id), None)
    if not depot_obj:
        if "lat" in req.depot and "lon" in req.depot:
            # Snap dynamic click
            snap_res = snap_coordinate(SnapRequest(lat=req.depot["lat"], lon=req.depot["lon"]))
            depot_obj = {
                "id": "CUSTOM_DEPOT",
                "name": req.depot.get("name", "Kho Chỉ Định"),
                "lat": snap_res["lat"],
                "lon": snap_res["lon"],
                "district": req.depot.get("district", "TP.HCM"),
                "snapped_node_id": snap_res["snapped_node_id"],
                "snap_distance_m": snap_res["snap_distance_m"]
            }
        else:
            raise HTTPException(status_code=400, detail=f"Không tìm thấy kho xuất phát id='{depot_id}'.")

    # Resolve stores
    w_default = 5.0
    if req.options and "w_pkg_default_kg" in req.options:
        w_default = float(req.options["w_pkg_default_kg"])

    demands_kg = [0.0]
    resolved_stores = []
    node_ids = [depot_obj["snapped_node_id"]]
    store_names = []

    for item in req.stores:
        st_data = stores_lookup.get(item.id)
        if not st_data:
            raise HTTPException(status_code=400, detail=f"Cửa hàng id='{item.id}' không tồn tại trong hệ thống.")

        w_pkg = item.weight_per_pkg_kg if item.weight_per_pkg_kg is not None else w_default
        d_kg = round(item.n_pkg * w_pkg, 1)

        demands_kg.append(d_kg)
        node_ids.append(st_data["snapped_node_id"])
        store_names.append(st_data["name"])

        resolved_stores.append({
            "id": st_data["id"],
            "name": st_data["name"],
            "address": st_data.get("address", ""),
            "district": st_data.get("district", ""),
            "lat": st_data["lat"],
            "lon": st_data["lon"],
            "n_pkg": item.n_pkg,
            "weight_per_pkg_kg": w_pkg,
            "demand_kg": d_kg,
            "snapped_node_id": st_data["snapped_node_id"]
        })

    # Adaptive solver budget: keep large batches (N > 50) responsive.
    n_stores = len(resolved_stores)
    fast_mode = n_stores > 50
    solve_steps = 8 if fast_mode else 25
    lns_steps = 10 if fast_mode else 25

    # 4-Tier Feasibility Gate
    scc_nodes = set(int(n) for n in get_graph().nodes())
    feas_result = check_feasibility(
        demands=demands_kg[1:],
        k_vehicles=req.fleet.k,
        capacity_kg=float(req.fleet.q_kg),
        node_ids=node_ids[1:],
        valid_scc_nodes=scc_nodes,
        store_names=store_names
    )

    if not feas_result.is_feasible:
        raise HTTPException(
            status_code=422,
            detail=feas_result.to_dict()
        )

    # Submatrix extraction
    mat_mgr = MatrixManager.get_instance()
    sub_dist_m, sub_time_s = mat_mgr.get_submatrix(node_ids)

    # Solve CVRP with MotorbikeCVRPEnv (4 Operators)
    env = MotorbikeCVRPEnv(
        dist_m=sub_dist_m,
        demands_kg=demands_kg,
        capacity_kg=float(req.fleet.q_kg),
        k_vehicles=req.fleet.k,
        max_steps=solve_steps,
        seed=42
    )

    state, mask = env.reset()
    baseline_cw_cost = env.initial_cost

    solver_mode = (req.options or {}).get("solver_mode", "ddqn_finetuned")
    last_q_values = None

    if solver_mode == "cw_greedy":
        solver_used = "cw_greedy"
        solver_display_name = "CW + Greedy"
        # Large batches use First-Improvement to avoid evaluating every operator deeply.
        for _ in range(env.max_steps):
            applied = False
            for a in range(4):
                if not mask[a]:
                    continue
                _, delta, improved = env._evaluate_operator(a)
                if improved and delta > 0:
                    state, _, done, _ = env.step(a)
                    mask = env.get_action_mask()
                    applied = True
                    if done:
                        break
                    break
            if not applied or done:
                break
        last_q_values = None

    elif solver_mode == "ddqn_base":
        solver_used = "ddqn_base"
        solver_display_name = "DDQN Gốc"
        base_model = get_base_model()
        if base_model is not None:
            for _ in range(env.max_steps):
                valid = [i for i, v in enumerate(mask) if v]
                if not valid:
                    break
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    m_dummy = torch.ones((1, 4), dtype=torch.bool)
                    q_vals = base_model(s_t, m_dummy)
                    pred_a = int(torch.argmax(q_vals, dim=1).item())
                    action = pred_a if mask[pred_a] else valid[0]
                    raw_q = q_vals.squeeze(0).tolist()
                    last_q_values = {
                        "2-opt": round(float(raw_q[0]), 3),
                        "intra-relocate": round(float(raw_q[1]), 3),
                        "inter-relocate": round(float(raw_q[2]), 3),
                        "inter-swap": round(float(raw_q[3]), 3)
                    }
                state, _, done, _ = env.step(action)
                mask = env.get_action_mask()
                if done:
                    break
        else:
            for _ in range(env.max_steps):
                valid = [a for a in range(4) if mask[a]]
                if not valid:
                    break
                state, _, done, _ = env.step(valid[0])
                mask = env.get_action_mask()
                if done:
                    break

    elif solver_mode in ["neural_lns", "ddqn_finetuned"]:
        solver_used = "neural_lns"
        solver_display_name = "Neural LNS (TP.HCM)"
        lns_env = NeuralLNSEnv(
            dist_m=sub_dist_m,
            demands_kg=demands_kg,
            capacity_kg=float(req.fleet.q_kg),
            k_vehicles=req.fleet.k,
            max_steps=lns_steps,
            seed=42
        )
        state_lns, mask_lns = lns_env.reset()
        lns_model = get_neural_lns_model()
        action_names = [
            "Worst + Greedy",
            "Worst + Regret-2",
            "Shaw + Greedy",
            "Shaw + Regret-2",
            "Route-Squeeze",
            "Worst + Regret-2 + 2opt"
        ]
        for _ in range(lns_env.max_steps):
            valid = [i for i, v in enumerate(mask_lns) if v]
            if not valid:
                break
            if lns_model is not None:
                with torch.no_grad():
                    s_t = torch.tensor(state_lns, dtype=torch.float32).unsqueeze(0)
                    m_t = torch.tensor(mask_lns, dtype=torch.bool).unsqueeze(0)
                    q_vals = lns_model(s_t, m_t)
                    action = int(torch.argmax(q_vals, dim=1).item())
                    raw_q = q_vals.squeeze(0).tolist()
                    last_q_values = {
                        action_names[idx]: round(float(raw_q[idx]), 3) for idx in range(6)
                    }
            else:
                action = valid[0]

            state_lns, _, done, _ = lns_env.step(action)
            mask_lns = lns_env.get_action_mask()
            if done:
                break

        final_routes = lns_env.best_routes
    else:
        solver_used = "cw_greedy"
        solver_display_name = "CW + Greedy"
        final_routes = env.routes

    if solver_mode not in ["neural_lns", "ddqn_finetuned"]:
        final_routes = env.routes

    # Enforce Hard Fleet Cap: strictly K_used <= K
    if len(final_routes) > req.fleet.k:
        final_routes = consolidate_routes_to_k(
            final_routes, demands_kg, float(req.fleet.q_kg), sub_dist_m, req.fleet.k
        )
    assert len(final_routes) <= req.fleet.k, f"K_used ({len(final_routes)}) exceeds fleet K ({req.fleet.k})"

    total_dist_m = compute_total_cost(final_routes, sub_dist_m)
    k_used = len(final_routes)

    # Single Source of Truth Geometry via reconstruct_leg_geometry
    vehicles = []
    max_route_time_s = 0.0

    for v_idx, route in enumerate(final_routes):
        veh_id = f"MOTO_{v_idx + 1:02d}"
        color = VEHICLE_COLORS[v_idx % len(VEHICLE_COLORS)]

        route_stores = [resolved_stores[c - 1] for c in route[1:-1]]
        load_kg = round(sum(st["demand_kg"] for st in route_stores), 1)
        load_pkgs = sum(st["n_pkg"] for st in route_stores)

        stops = []
        legs = []
        route_dist_m = 0.0
        route_time_s = 0.0

        # Depot start
        stops.append({
            "type": "depot",
            "id": depot_obj["id"],
            "name": depot_obj["name"],
            "lat": depot_obj["lat"],
            "lon": depot_obj["lon"],
            "packages": 0,
            "demand_kg": 0.0,
            "load_after_kg": load_kg
        })

        current_load = load_kg
        for step_idx in range(len(route) - 1):
            from_idx = route[step_idx]
            to_idx = route[step_idx + 1]

            from_nid = node_ids[from_idx]
            to_nid = node_ids[to_idx]

            leg_data = mat_mgr.reconstruct_leg_geometry(from_nid, to_nid)
            leg_dist = leg_data["distance_m"]
            leg_time = leg_data["time_s"]
            route_dist_m += leg_dist
            route_time_s += leg_time

            from_id = depot_obj["id"] if from_idx == 0 else resolved_stores[from_idx - 1]["id"]
            to_id = depot_obj["id"] if to_idx == 0 else resolved_stores[to_idx - 1]["id"]

            legs.append({
                "from": from_id,
                "to": to_id,
                "node_path": leg_data["node_path"],
                "geometry": leg_data["geometry"],
                "distance_m": leg_dist,
                "time_s": leg_time
            })

            if to_idx != 0:
                st = resolved_stores[to_idx - 1]
                current_load = round(current_load - st["demand_kg"], 1)
                stops.append({
                    "type": "store",
                    "id": st["id"],
                    "name": st["name"],
                    "address": st["address"],
                    "district": st["district"],
                    "lat": st["lat"],
                    "lon": st["lon"],
                    "packages": st["n_pkg"],
                    "demand_kg": st["demand_kg"],
                    "load_after_kg": max(0.0, current_load)
                })

        # Depot return
        stops.append({
            "type": "depot",
            "id": depot_obj["id"],
            "name": depot_obj["name"],
            "lat": depot_obj["lat"],
            "lon": depot_obj["lon"],
            "packages": 0,
            "demand_kg": 0.0,
            "load_after_kg": 0.0
        })

        if route_time_s > max_route_time_s:
            max_route_time_s = route_time_s

        dist_districts = list(set(st["district"] for st in route_stores))
        reasoning = (
            f"Phục vụ {len(route_stores)} cửa hàng tại {', '.join(dist_districts[:3])}. "
            f"Tải trọng đạt {round(load_kg / req.fleet.q_kg * 100)}% ({load_kg}/{req.fleet.q_kg} kg)."
        )

        vehicles.append({
            "vehicle_id": veh_id,
            "color": color,
            "stops": stops,
            "load_kg": load_kg,
            "load_pkgs": load_pkgs,
            "distance_km": round(route_dist_m / 1000.0, 2),
            "eta_min": round(route_time_s / 60.0, 1),
            "legs": legs,
            "reasoning": reasoning
        })

    # Unused vehicles waiting at depot
    for u_idx in range(k_used, req.fleet.k):
        veh_id = f"MOTO_{u_idx + 1:02d}"
        color = VEHICLE_COLORS[u_idx % len(VEHICLE_COLORS)]
        vehicles.append({
            "vehicle_id": veh_id,
            "color": color,
            "stops": [
                {
                    "type": "depot",
                    "id": depot_obj["id"],
                    "name": depot_obj["name"],
                    "lat": depot_obj["lat"],
                    "lon": depot_obj["lon"],
                    "packages": 0,
                    "demand_kg": 0.0,
                    "load_after_kg": 0.0
                }
            ],
            "load_kg": 0.0,
            "load_pkgs": 0,
            "distance_km": 0.0,
            "eta_min": 0.0,
            "legs": [],
            "reasoning": "Chờ tại kho (0 kiện) do đội xe sẵn sàng vượt mức tối ưu cần thiết."
        })

    total_dist_km = round(total_dist_m / 1000.0, 2)
    cw_dist_km = round(baseline_cw_cost / 1000.0, 2)
    saving_pct = round(((baseline_cw_cost - total_dist_m) / max(1.0, baseline_cw_cost)) * 100.0, 2)

    return {
        "k_used": k_used,
        "vehicles": vehicles,
        "total_distance_km": total_dist_km,
        "baseline_cw_distance_km": cw_dist_km,
        "saving_vs_cw_pct": saving_pct,
        "makespan_min": round(max_route_time_s / 60.0, 1),
        "solver": solver_used,
        "solver_mode": solver_mode,
        "solver_name": solver_display_name,
        "q_values": last_q_values,
        "fallback": (solver_used == "fallback_cw"),
        "data_source": "osm_poi",
        "fast_mode": fast_mode
    }
