"""FastAPI Backend Server for DQN-CVRP Production HCMC Map Demo.
Provides REST endpoints for:
- Instance loading with full administrative district/ward metadata
- Clarke-Wright initialization
- Step-by-step local search execution with Q-values
- Live HCMC administrative data integration (provinces.open-api.vn)
- Independent constraint validation
"""
import time
import urllib.request
import urllib.parse
import json
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..cvrp.instance import CVRPInstance, haversine_distance
from ..cvrp.clarke_wright import clarke_wright_savings
from ..cvrp.operators import best_move, OPERATOR_NAMES
from ..cvrp.state_f28 import StateExtractorF28
from ..cvrp.osrm_service import DynamicOSRMService
from ..agents.dqn_agent import DQNAgent
from ..agents.baselines import RandomPolicy, CyclicPolicy, GreedyPolicy
from ..evaluation.validator import IndependentValidator

osrm_service = DynamicOSRMService()

app = FastAPI(
    title="DQN-CVRP Ho Chi Minh City Map Demo",
    description="Interactive Reinforcement Learning CVRP Solver on HCMC Map with OpenStreetMap",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/") or request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = BASE_DIR / "data" / "hcmc_instances"

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount brand / icon assets (e.g. /assets/fast-delivery.png)
ASSETS_DIR = STATIC_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# Mount MP-MOTO-CVRP V9.2 Motorbike Dispatch Router
from .dispatch_motorbike import router as motorbike_router
app.include_router(motorbike_router)

# Caches
INSTANCES_CACHE: Dict[str, CVRPInstance] = {}
RAW_JSON_CACHE: Dict[str, Dict[str, Any]] = {}
AGENTS_CACHE: Dict[str, Any] = {}
DISTRICTS_CACHE: List[Dict[str, Any]] = []

# Fallback HCMC Districts (22 official districts of TP.HCM)
FALLBACK_HCMC_DISTRICTS = [
    {"code": 760, "name": "Quận 1"},
    {"code": 769, "name": "Quận 2 (cũ)"},
    {"code": 770, "name": "Quận 3"},
    {"code": 771, "name": "Quận 10"},
    {"code": 772, "name": "Quận 11"},
    {"code": 773, "name": "Quận 4"},
    {"code": 774, "name": "Quận 5"},
    {"code": 775, "name": "Quận 6"},
    {"code": 776, "name": "Quận 8"},
    {"code": 777, "name": "Quận Bình Thạnh"},
    {"code": 778, "name": "Quận Gò Vấp"},
    {"code": 764, "name": "Quận 9 (cũ)"},
    {"code": 765, "name": "Quận Phú Nhuận"},
    {"code": 766, "name": "Quận Tân Bình"},
    {"code": 767, "name": "Quận Tân Phú"},
    {"code": 768, "name": "Quận Bình Tân"},
    {"code": 769, "name": "TP. Thủ Đức"},
    {"code": 783, "name": "Huyện Củ Chi"},
    {"code": 784, "name": "Huyện Hóc Môn"},
    {"code": 785, "name": "Huyện Bình Chánh"},
    {"code": 786, "name": "Huyện Nhà Bè"},
    {"code": 787, "name": "Huyện Cần Giờ"}
]


def load_raw_json(instance_name: str) -> Dict[str, Any]:
    """Load raw JSON file to preserve district and ward attributes."""
    if instance_name in RAW_JSON_CACHE:
        return RAW_JSON_CACHE[instance_name]

    if instance_name == "HCMC-District-20":
        path = DATA_DIR / "hcmc_district_20.json"
    elif instance_name == "HCMC-FPT-40":
        path = DATA_DIR / "hcmc_fpt_40.json"
    else:
        path = DATA_DIR / "hcmc_district_20.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    RAW_JSON_CACHE[instance_name] = data
    return data


def get_instance(instance_name: str) -> CVRPInstance:
    """Load or retrieve instance from cache."""
    if instance_name in INSTANCES_CACHE:
        return INSTANCES_CACHE[instance_name]

    if instance_name == "HCMC-District-20":
        path = DATA_DIR / "hcmc_district_20.json"
        inst = CVRPInstance.from_json(path)
    elif instance_name == "HCMC-FPT-40":
        path = DATA_DIR / "hcmc_fpt_40.json"
        inst = CVRPInstance.from_json(path)
    elif instance_name == "Synthetic-24":
        inst = CVRPInstance.generate_synthetic(name="Synthetic-24", num_customers=24, seed=42)
    else:
        path = DATA_DIR / "hcmc_district_20.json"
        inst = CVRPInstance.from_json(path)

    INSTANCES_CACHE[instance_name] = inst
    return inst


def get_agent(algorithm: str) -> Any:
    """Retrieve or initialize policy agent."""
    algo = algorithm.lower()
    if algo not in AGENTS_CACHE:
        if algo == "dqn":
            agent = DQNAgent(algorithm="dqn")
        elif algo == "ddqn":
            agent = DQNAgent(algorithm="ddqn")
        elif algo == "greedy":
            agent = GreedyPolicy(probe_candidates=32)
        elif algo == "cyclic":
            agent = CyclicPolicy()
        elif algo == "random":
            agent = RandomPolicy()
        else:
            agent = DQNAgent(algorithm="ddqn")
        AGENTS_CACHE[algo] = agent
    return AGENTS_CACHE[algo]


# --- Request Models ---
class InitRequest(BaseModel):
    instance_name: str = "HCMC-District-20"


class StepRequest(BaseModel):
    instance_name: str = "HCMC-District-20"
    routes: List[List[int]]
    algorithm: str = "ddqn"
    evals_used: int = 0
    step_count: int = 0
    initial_cost: float
    feature_patience: float = 30.0
    history_streaks: Optional[List[int]] = None
    history_calls: Optional[List[int]] = None
    history_successes: Optional[List[int]] = None


class RunAllRequest(BaseModel):
    instance_name: str = "HCMC-District-20"
    algorithm: str = "ddqn"
    max_steps: int = 35


class CustomInstanceRequest(BaseModel):
    name: Optional[str] = None
    num_customers: int = 15
    capacity: int = 30
    max_vehicles: int = 5
    district: str = "all"
    min_demand: int = 1
    max_demand: int = 6
    seed: Optional[int] = None


class BenchmarkRequest(BaseModel):
    instance_name: str = "HCMC-District-20"
    budget_steps: int = 30
    algorithms: Optional[List[str]] = None
    seed: int = 42


class CompareDuelRequest(BaseModel):
    instance_name: str = "HCMC-District-20"
    model_a: str = "ddqn"
    model_b: str = "greedy"
    budget_steps: int = 30


class FilterDistrictRequest(BaseModel):
    instance_name: str = "HCMC-District-20"
    district: str = "all"
    algorithm: Optional[str] = "ddqn"


class NodeDemandConfig(BaseModel):
    node_id: int
    demand: int


class InitComparisonRequest(BaseModel):
    base_instance: str = "HCMC-District-20"
    district: Optional[str] = "all"
    selected_nodes: List[NodeDemandConfig]
    capacity: int = 30
    model_a: str = "ddqn"
    model_b: str = "greedy"


class CompareStepRequest(BaseModel):
    instance_name: str
    routes_a: List[List[int]]
    routes_b: List[List[int]]
    model_a: str = "ddqn"
    model_b: str = "greedy"
    step_count: int = 0
    evals_used_a: int = 0
    evals_used_b: int = 0
    initial_cost: float
    feature_patience: float = 30.0
    history_calls_a: Optional[List[int]] = None
    history_streaks_a: Optional[List[int]] = None
    history_successes_a: Optional[List[int]] = None
    history_calls_b: Optional[List[int]] = None
    history_streaks_b: Optional[List[int]] = None
    history_successes_b: Optional[List[int]] = None


class DynamicNode(BaseModel):
    id: Optional[int] = None
    name: str
    lat: float
    lng: float
    demand: int = 1
    district: Optional[str] = "TP.HCM"
    address: Optional[str] = ""


class CreateDynamicScenarioRequest(BaseModel):
    depot: DynamicNode
    customers: List[DynamicNode]
    max_vehicles: int = 5
    capacity: int = 30
    metric: str = "distance"  # "distance" (km) or "duration" (min)
    scenario_name: Optional[str] = None


def normalize_district(raw_text: Optional[str]) -> str:
    """Normalize HCMC district names to canonical IDs.
    Guarantees strict separation between Quận 1, Quận 10, Quận 11, Quận 12.
    """
    if not raw_text:
        return ""
    text = raw_text.strip().lower()

    # Check numbered districts with word boundary
    m = re.search(r'(?:quận|q\.?)\s*(\d+)(?:\s*\(cũ\))?(?=\s*,|\s*$|\s*-|\s*\))', text)
    if not m:
        m = re.search(r'(?:quận|q\.?)\s*(\d+)', text)
    if m:
        num = int(m.group(1))
        return f"quan_{num}"

    # Named districts & suburban counties
    if "bình thạnh" in text or "binh thanh" in text:
        return "quan_binh_thanh"
    if "gò vấp" in text or "go vap" in text:
        return "quan_go_vap"
    if "phú nhuận" in text or "phu nhuan" in text:
        return "quan_phu_nhuan"
    if "tân bình" in text or "tan binh" in text:
        return "quan_tan_binh"
    if "tân phú" in text or "tan phu" in text:
        return "quan_tan_phu"
    if "bình tân" in text or "binh tan" in text:
        return "quan_binh_tan"
    if "thủ đức" in text or "thu duc" in text:
        return "tp_thu_duc"
    if "củ chi" in text or "cu chi" in text:
        return "huyen_cu_chi"
    if "hóc môn" in text or "hoc mon" in text:
        return "huyen_hoc_mon"
    if "bình chánh" in text or "binh chanh" in text:
        return "huyen_binh_chanh"
    if "nhà bè" in text or "nha be" in text:
        return "huyen_nha_be"
    if "cần giờ" in text or "can gio" in text:
        return "huyen_can_gio"

    return re.sub(r'[^a-z0-9]+', '_', text).strip('_')


@app.get("/")
def root():
    """Serve the interactive map application."""
    index_file = STATIC_DIR / "index.html"
    return FileResponse(
        index_file,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/api/provinces/hcmc_districts")
def get_hcmc_districts():
    """Fetch administrative districts of Ho Chi Minh City from provinces.open-api.vn
    with reliable fallback.
    """
    global DISTRICTS_CACHE
    if DISTRICTS_CACHE:
        return {"source": "cache", "districts": DISTRICTS_CACHE}

    url = "https://provinces.open-api.vn/api/p/79?depth=2"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            districts = [{"code": d["code"], "name": d["name"]} for d in data.get("districts", [])]
            DISTRICTS_CACHE = districts
            return {"source": "provinces.open-api.vn", "districts": districts}
    except Exception as e:
        DISTRICTS_CACHE = FALLBACK_HCMC_DISTRICTS
        return {"source": "fallback", "error": str(e), "districts": FALLBACK_HCMC_DISTRICTS}


@app.get("/api/instances")
def list_instances():
    """List available instances and metadata."""
    instances = [
        {
            "id": "HCMC-District-20",
            "name": "TP.HCM Nội thành (20 Điểm Giao Hàng)",
            "comment": "Depot tại Đại học FPT TP.HCM (Khu CNC TP.Thủ Đức) phục vụ các quận nội thành",
            "customers": 20,
            "capacity": 30,
            "max_vehicles": 6,
            "depot_name": "Kho Trung Tâm - Đại học FPT TP.HCM",
            "is_geo": True
        },
        {
            "id": "HCMC-FPT-40",
            "name": "TP.HCM Đô thị Mở rộng (40 Điểm Giao Hàng)",
            "comment": "Phủ rộng toàn thành phố và các khu đô thị vệ tinh",
            "customers": 40,
            "capacity": 35,
            "max_vehicles": 8,
            "depot_name": "Kho Trung Tâm - Đại học FPT TP.HCM",
            "is_geo": True
        },
        {
            "id": "Synthetic-24",
            "name": "Benchmark Tổng hợp (Synthetic-24)",
            "comment": "Dữ liệu 24 điểm thử nghiệm đối sánh nghiên cứu học thuật",
            "customers": 24,
            "capacity": 30,
            "max_vehicles": 5,
            "depot_name": "Depot 0 (50, 50)",
            "is_geo": False
        }
    ]
    return {"instances": instances}


@app.post("/api/init")
def initialize_instance(req: InitRequest):
    """Load instance with administrative metadata and run Clarke-Wright Savings heuristic."""
    inst = get_instance(req.instance_name)
    routes = clarke_wright_savings(inst)
    initial_cost = inst.solution_cost(routes)

    # Load administrative details if available
    raw_data = load_raw_json(req.instance_name) if inst.is_geo else {}
    depot_raw = raw_data.get("depot", {})
    cust_raw_list = raw_data.get("customers", [])
    cust_map = {c["id"]: c for c in cust_raw_list}

    # Format nodes for map
    nodes = []
    for i in range(len(inst.coords)):
        is_depot = (i == 0)
        if is_depot:
            district = depot_raw.get("district", "TP. Thủ Đức")
            ward = depot_raw.get("ward", "Phường Long Thạnh Mỹ")
            address = depot_raw.get("address", "Lô E2a-7, Đường D1, Khu CNC, TP. Thủ Đức")
        else:
            c_info = cust_map.get(i, {})
            district = c_info.get("district", "TP. Hồ Chí Minh")
            ward = c_info.get("ward", "")
            address = c_info.get("address", "")

        nodes.append({
            "id": i,
            "name": inst.node_names[i] if i < len(inst.node_names) else f"Node {i}",
            "lat": float(inst.coords[i, 0]),
            "lng": float(inst.coords[i, 1]),
            "demand": int(inst.demands[i]),
            "district": district,
            "ward": ward,
            "address": address,
            "is_depot": is_depot
        })

    validation = IndependentValidator.validate(inst, routes)

    # Compute real initial Q-values using pre-trained DDQN agent for Step 0
    extractor = StateExtractorF28(feature_patience=30.0)
    init_state = extractor.extract(routes, inst, evals_used=0, max_evals=2560, initial_cost=initial_cost)
    agent = get_agent("ddqn")
    active_routes_count = len([r for r in routes if len(r) > 2])
    action_mask = [True, True, False, False] if active_routes_count <= 1 else [True, True, True, True]
    action, q_vals = agent.act(init_state, action_mask=action_mask)
    q_dict = {OPERATOR_NAMES[op]: float(q_vals[op]) for op in range(4)}

    # Compute street-level road geometries and fleet details
    road_geometries = []
    fleet_active = []
    active_routes = [r for r in routes if len(r) > 2]
    if inst.is_geo:
        for v_idx, r in enumerate(active_routes):
            r_coords = [[float(inst.coords[node, 0]), float(inst.coords[node, 1])] for node in r]
            road_geometries.append(osrm_service.get_route_geometry(r_coords))
            d_sum = sum(inst.demands[node] for node in r)
            fleet_active.append({
                "vehicle_id": v_idx + 1,
                "status": "Đang vận hành",
                "demand": int(d_sum),
                "capacity": inst.capacity,
                "pct": round((d_sum / inst.capacity) * 100.0, 1),
                "stops": len(r) - 2,
                "route": r
            })

    fleet_idle = []
    for v_idx in range(len(active_routes), inst.max_vehicles):
        fleet_idle.append({
            "vehicle_id": v_idx + 1,
            "status": "Chờ lệnh tại kho",
            "demand": 0,
            "capacity": inst.capacity,
            "pct": 0.0,
            "stops": 0,
            "route": [0, 0]
        })

    return {
        "instance_name": inst.name,
        "is_geo": inst.is_geo,
        "capacity": inst.capacity,
        "max_vehicles": inst.max_vehicles,
        "num_customers": inst.num_customers,
        "nodes": nodes,
        "routes": routes,
        "fleet_active": fleet_active,
        "fleet_idle": fleet_idle,
        "road_geometries": road_geometries if inst.is_geo else None,
        "initial_cost": round(initial_cost, 2),
        "current_cost": round(initial_cost, 2),
        "num_active_routes": len(routes),
        "is_feasible": validation["is_valid"],
        "validation": validation,
        "action_mask": action_mask,
        "chosen_action": int(action),
        "operator_name": OPERATOR_NAMES[action],
        "q_values": q_dict,
        "f28_summary": {
            "mean_load_ratio": round(float(init_state[0]), 3),
            "max_load_ratio": round(float(init_state[2]), 3),
            "vehicles_used_ratio": round(float(init_state[4]), 3),
            "budget_progress": round(float(init_state[10]), 3),
            "cumulative_improvement": round(float(init_state[11]), 3)
        }
    }


@app.post("/api/filter_district")
def filter_district(req: FilterDistrictRequest):
    """Filter CVRP instance by exact normalized district and re-solve with minimal K."""
    if not req.district or req.district.lower() == "all":
        return initialize_instance(InitRequest(instance_name=req.instance_name))

    norm_target = normalize_district(req.district)
    raw = load_raw_json(req.instance_name)
    depot = raw.get("depot", {
        "id": 0,
        "name": "Kho Trung Tâm - Đại học FPT TP.HCM",
        "lat": 10.8415,
        "lng": 106.8098,
        "demand": 0,
        "district": "TP. Thủ Đức",
        "ward": "Phường Long Thạnh Mỹ",
        "address": "Lô E2a-7, Đường D1, Khu CNC, TP. Thủ Đức",
        "is_depot": True
    })
    all_customers = raw.get("customers", [])

    # Strict normalization match:
    matching_customers = [
        c for c in all_customers
        if normalize_district(c.get("district", "")) == norm_target
    ]

    if len(matching_customers) == 0:
        # Strict Reality: District has 0 delivery points in this scenario
        return {
            "instance_name": req.instance_name,
            "district": req.district,
            "normalized_district": norm_target,
            "num_customers": 0,
            "k_min": 0,
            "capacity": 30,
            "max_vehicles": 6,
            "nodes": [dict(depot, id=0, is_depot=True)],
            "routes": [],
            "initial_cost": 0.0,
            "current_cost": 0.0,
            "num_active_routes": 0,
            "is_feasible": True,
            "action_mask": [False, False, False, False],
            "chosen_action": 0,
            "operator_name": "2-opt",
            "q_values": {
                "2-opt": 0.0,
                "intra-relocate": 0.0,
                "inter-relocate": 0.0,
                "inter-swap": 0.0
            },
            "message": f"Địa bàn [{req.district}] hiện không có đơn hàng nào cần giao trong kịch bản này."
        }

    # District has >= 1 customer! Re-solve CVRP with optimal minimal fleet size K_min
    total_demand = sum(c.get("demand", 1) for c in matching_customers)
    capacity = 30
    k_min = max(1, int(np.ceil(total_demand / capacity)))
    max_vehicles = max(k_min, 6)

    # Re-index nodes: Depot is 0, customers are 1 .. len(matching_customers)
    coords_list = [[float(depot["lat"]), float(depot["lng"])]]
    demands_list = [0]
    node_names = [depot.get("name", "Kho Trung Tâm - Đại học FPT TP.HCM")]
    sub_nodes_info = [dict(depot, id=0, is_depot=True)]

    for new_id, c in enumerate(matching_customers, start=1):
        c_copy = dict(c)
        c_copy["id"] = new_id
        c_copy["is_depot"] = False
        coords_list.append([float(c["lat"]), float(c["lng"])])
        demands_list.append(int(c.get("demand", 1)))
        node_names.append(c.get("name", f"Điểm #{new_id}"))
        sub_nodes_info.append(c_copy)

    coords = np.array(coords_list, dtype=np.float64)
    demands = np.array(demands_list, dtype=np.int32)
    n_nodes = len(coords)

    dist_mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dist_mat[i, j] = haversine_distance(
                    coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]
                )

    sub_inst_name = f"{req.instance_name}_{norm_target}"
    sub_inst = CVRPInstance(
        name=sub_inst_name,
        num_customers=len(matching_customers),
        capacity=capacity,
        max_vehicles=max_vehicles,
        coords=coords,
        demands=demands,
        distance_matrix=dist_mat,
        node_names=node_names,
        is_geo=True
    )

    INSTANCES_CACHE[sub_inst_name] = sub_inst
    RAW_JSON_CACHE[sub_inst_name] = {
        "depot": depot,
        "customers": sub_nodes_info[1:]
    }

    # Re-run Clarke-Wright on active subset
    raw_routes = clarke_wright_savings(sub_inst)
    active_routes = [r for r in raw_routes if len(r) > 2]

    initial_cost = sub_inst.solution_cost(active_routes)
    validation = IndependentValidator.validate(sub_inst, active_routes)

    # Action Masking based on active routes count
    action_mask = [True, True, False, False] if len(active_routes) <= 1 else [True, True, True, True]

    extractor = StateExtractorF28(feature_patience=30.0)
    init_state = extractor.extract(
        active_routes, sub_inst, evals_used=0, max_evals=2560, initial_cost=initial_cost
    )
    agent = get_agent(req.algorithm)
    if req.algorithm.lower() == "greedy":
        action, q_vals = agent.select_operator(active_routes, sub_inst, action_mask=action_mask)
    else:
        action, q_vals = agent.act(init_state, action_mask=action_mask)

    q_dict = {OPERATOR_NAMES[op]: float(q_vals[op]) for op in range(4)}

    return {
        "instance_name": sub_inst_name,
        "base_instance": req.instance_name,
        "district": req.district,
        "normalized_district": norm_target,
        "num_customers": len(matching_customers),
        "k_min": k_min,
        "capacity": capacity,
        "max_vehicles": max_vehicles,
        "nodes": sub_nodes_info,
        "routes": active_routes,
        "initial_cost": round(float(initial_cost), 2),
        "current_cost": round(float(initial_cost), 2),
        "num_active_routes": len(active_routes),
        "is_feasible": validation["is_valid"],
        "validation": validation,
        "action_mask": action_mask,
        "chosen_action": int(action),
        "operator_name": OPERATOR_NAMES[action],
        "q_values": q_dict,
        "f28_summary": {
            "mean_load_ratio": round(float(init_state[0]), 3),
            "max_load_ratio": round(float(init_state[2]), 3),
            "vehicles_used_ratio": round(float(init_state[4]), 3),
            "budget_progress": round(float(init_state[10]), 3),
            "cumulative_improvement": round(float(init_state[11]), 3)
        }
    }


@app.post("/api/step")
def execute_step(req: StepRequest):
    """Execute one local search operator step guided by the selected policy."""
    inst = get_instance(req.instance_name)
    agent = get_agent(req.algorithm)

    extractor = StateExtractorF28(feature_patience=req.feature_patience)
    if req.history_calls and req.history_streaks and req.history_successes:
        for op in range(4):
            extractor.operator_stats[op].calls = req.history_calls[op]
            extractor.operator_stats[op].streak = req.history_streaks[op]
            extractor.operator_stats[op].successes = req.history_successes[op]
        extractor.total_steps = req.step_count

    state = extractor.extract(
        req.routes,
        inst,
        evals_used=req.evals_used,
        max_evals=2560,
        initial_cost=req.initial_cost
    )

    active_routes_count = len([r for r in req.routes if len(r) > 2])
    action_mask = [True, True, False, False] if active_routes_count <= 1 else [True, True, True, True]

    if req.algorithm.lower() == "greedy":
        action, q_vals = agent.select_operator(req.routes, inst, action_mask=action_mask)
    else:
        action, q_vals = agent.act(state, action_mask=action_mask)

    new_routes, delta, cand_count, improved = best_move(
        req.routes, action, inst, max_candidates=128
    )

    new_cost = inst.solution_cost(new_routes)
    new_evals_used = req.evals_used + cand_count

    extractor.record_step(action, improved, delta)
    validation = IndependentValidator.validate(inst, new_routes)

    impr_pct = ((req.initial_cost - new_cost) / max(1e-6, req.initial_cost)) * 100.0

    q_dict = {
        OPERATOR_NAMES[op]: float(q_vals[op]) for op in range(4)
    }

    # Compute updated road geometries and fleet details if geo-instance
    road_geometries = []
    fleet_active = []
    active_new_routes = [r for r in new_routes if len(r) > 2]
    if inst.is_geo:
        for v_idx, r in enumerate(active_new_routes):
            r_coords = [[float(inst.coords[node, 0]), float(inst.coords[node, 1])] for node in r]
            road_geometries.append(osrm_service.get_route_geometry(r_coords))
            d_sum = sum(inst.demands[node] for node in r)
            fleet_active.append({
                "vehicle_id": v_idx + 1,
                "status": "Đang vận hành",
                "demand": int(d_sum),
                "capacity": inst.capacity,
                "pct": round((d_sum / inst.capacity) * 100.0, 1),
                "stops": len(r) - 2,
                "route": r
            })

    return {
        "step": req.step_count + 1,
        "chosen_action": action,
        "operator_name": OPERATOR_NAMES[action],
        "action_mask": action_mask,
        "improved": improved,
        "delta_cost": round(float(delta), 2),
        "new_cost": round(float(new_cost), 2),
        "improvement_pct": round(float(impr_pct), 2),
        "evals_used": new_evals_used,
        "candidates_evaluated": cand_count,
        "q_values": q_dict,
        "routes": new_routes,
        "road_geometries": road_geometries if inst.is_geo else None,
        "fleet_active": fleet_active if inst.is_geo else None,
        "is_feasible": validation["is_valid"],
        "history_calls": [extractor.operator_stats[op].calls for op in range(4)],
        "history_streaks": [extractor.operator_stats[op].streak for op in range(4)],
        "history_successes": [extractor.operator_stats[op].successes for op in range(4)],
        "f28_summary": {
            "mean_load_ratio": round(float(state[0]), 3),
            "max_load_ratio": round(float(state[2]), 3),
            "vehicles_used_ratio": round(float(state[4]), 3),
            "budget_progress": round(float(state[10]), 3),
            "cumulative_improvement": round(float(state[11]), 3)
        }
    }


@app.post("/api/run_all")
def run_all_steps(req: RunAllRequest):
    """Run full optimization process up to max_steps."""
    inst = get_instance(req.instance_name)
    agent = get_agent(req.algorithm)
    routes = clarke_wright_savings(inst)
    initial_cost = inst.solution_cost(routes)

    extractor = StateExtractorF28(feature_patience=30.0)
    evals_used = 0
    history = []

    for step_idx in range(req.max_steps):
        state = extractor.extract(
            routes, inst, evals_used=evals_used, max_evals=2560, initial_cost=initial_cost
        )
        active_routes_count = len([r for r in routes if len(r) > 2])
        action_mask = [True, True, False, False] if active_routes_count <= 1 else [True, True, True, True]
        if req.algorithm.lower() == "greedy":
            action, q_vals = agent.select_operator(routes, inst, action_mask=action_mask)
        else:
            action, q_vals = agent.act(state, action_mask=action_mask)

        new_routes, delta, cand_count, improved = best_move(
            routes, action, inst, max_candidates=128
        )
        evals_used += cand_count

        if improved:
            routes = new_routes

        extractor.record_step(action, improved, delta)
        cur_cost = inst.solution_cost(routes)

        history.append({
            "step": step_idx + 1,
            "operator": OPERATOR_NAMES[action],
            "improved": improved,
            "cost": round(float(cur_cost), 2),
            "delta": round(float(delta), 2)
        })

        if evals_used >= 2560:
            break

    final_cost = inst.solution_cost(routes)
    impr_pct = ((initial_cost - final_cost) / max(1e-6, initial_cost)) * 100.0
    validation = IndependentValidator.validate(inst, routes)

    return {
        "instance_name": inst.name,
        "algorithm": req.algorithm,
        "initial_cost": round(initial_cost, 2),
        "final_cost": round(final_cost, 2),
        "improvement_pct": round(float(impr_pct), 2),
        "total_evals": evals_used,
        "total_steps": len(history),
        "routes": routes,
        "history": history,
        "is_feasible": validation["is_valid"]
    }


DISTRICT_CENTROIDS = {
    "Quận 1": (10.7769, 106.7009),
    "Quận 2 (cũ)": (10.7872, 106.7498),
    "Quận 3": (10.7844, 106.6844),
    "Quận 4": (10.7580, 106.7030),
    "Quận 5": (10.7554, 106.6628),
    "Quận 6": (10.7470, 106.6350),
    "Quận 7": (10.7340, 106.7218),
    "Quận 8": (10.7240, 106.6280),
    "Quận 9 (cũ)": (10.8428, 106.8286),
    "Quận 10": (10.7719, 106.6678),
    "Quận 11": (10.7630, 106.6490),
    "Quận 12": (10.8672, 106.6414),
    "Quận Bình Thạnh": (10.8030, 106.7000),
    "Quận Gò Vấp": (10.8388, 106.6657),
    "Quận Phú Nhuận": (10.7992, 106.6800),
    "Quận Tân Bình": (10.8015, 106.6538),
    "Quận Tân Phú": (10.7900, 106.6280),
    "Quận Bình Tân": (10.7654, 106.6047),
    "TP. Thủ Đức": (10.8497, 106.7725),
    "Huyện Củ Chi": (11.0067, 106.5132),
    "Huyện Hóc Môn": (10.8842, 106.5925),
    "Huyện Bình Chánh": (10.6874, 106.5938),
    "Huyện Nhà Bè": (10.6953, 106.7324),
    "Huyện Cần Giờ": (10.4114, 106.9547),
}


@app.post("/api/custom_instance")
def create_custom_instance(req: CustomInstanceRequest):
    """Generate dynamic custom CVRP instance within HCMC with custom K, Q, N."""
    rng = np.random.RandomState(req.seed if req.seed is not None else int(time.time() % 100000))
    inst_name = req.name or f"Custom-{req.num_customers}pts-{int(time.time() % 10000)}"

    # Depot at FPT University HCMC (Khu CNC TP. Thủ Đức)
    depot_lat, depot_lng = 10.8415, 106.8098

    coords_list = [[depot_lat, depot_lng]]
    demands_list = [0]
    node_names = ["Kho Trung Tâm - Đại học FPT TP.HCM"]
    nodes_info = [{
        "id": 0,
        "name": "Kho Trung Tâm - Đại học FPT TP.HCM",
        "lat": depot_lat,
        "lng": depot_lng,
        "demand": 0,
        "district": "TP. Thủ Đức",
        "ward": "Phường Long Thạnh Mỹ",
        "address": "Lô E2a-7, Đường D1, Khu CNC, TP. Thủ Đức",
        "is_depot": True
    }]

    target_district = req.district if req.district in DISTRICT_CENTROIDS else None

    for i in range(1, req.num_customers + 1):
        if target_district:
            c_lat, c_lng = DISTRICT_CENTROIDS[target_district]
            lat = float(c_lat + rng.normal(0, 0.012))
            lng = float(c_lng + rng.normal(0, 0.014))
            dist_name = target_district
        else:
            picked_dist = rng.choice(list(DISTRICT_CENTROIDS.keys()))
            c_lat, c_lng = DISTRICT_CENTROIDS[picked_dist]
            lat = float(c_lat + rng.normal(0, 0.015))
            lng = float(c_lng + rng.normal(0, 0.018))
            dist_name = picked_dist

        demand = int(rng.randint(req.min_demand, req.max_demand + 1))
        coords_list.append([lat, lng])
        demands_list.append(demand)
        node_name = f"Điểm #{i} ({dist_name})"
        node_names.append(node_name)
        nodes_info.append({
            "id": i,
            "name": node_name,
            "lat": round(lat, 5),
            "lng": round(lng, 5),
            "demand": demand,
            "district": dist_name,
            "ward": "Địa bàn giao nhận",
            "address": f"Khu vực {dist_name}, TP.HCM",
            "is_depot": False
        })

    coords = np.array(coords_list, dtype=np.float64)
    demands = np.array(demands_list, dtype=np.int32)
    n_nodes = len(coords)

    dist_mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dist_mat[i, j] = haversine_distance(
                    coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]
                )

    inst = CVRPInstance(
        name=inst_name,
        num_customers=req.num_customers,
        capacity=req.capacity,
        max_vehicles=req.max_vehicles,
        coords=coords,
        demands=demands,
        distance_matrix=dist_mat,
        node_names=node_names,
        is_geo=True
    )

    INSTANCES_CACHE[inst_name] = inst
    RAW_JSON_CACHE[inst_name] = {
        "depot": nodes_info[0],
        "customers": nodes_info[1:]
    }

    routes = clarke_wright_savings(inst)
    initial_cost = inst.solution_cost(routes)
    validation = IndependentValidator.validate(inst, routes)

    extractor = StateExtractorF28(feature_patience=30.0)
    init_state = extractor.extract(routes, inst, evals_used=0, max_evals=2560, initial_cost=initial_cost)
    agent = get_agent("ddqn")
    action, q_vals = agent.act(init_state)
    q_dict = {OPERATOR_NAMES[op]: float(q_vals[op]) for op in range(4)}

    return {
        "instance_name": inst_name,
        "is_geo": True,
        "capacity": inst.capacity,
        "max_vehicles": inst.max_vehicles,
        "num_customers": inst.num_customers,
        "nodes": nodes_info,
        "routes": routes,
        "initial_cost": round(float(initial_cost), 2),
        "current_cost": round(float(initial_cost), 2),
        "num_active_routes": len(routes),
        "is_feasible": validation["is_valid"],
        "validation": validation,
        "chosen_action": int(action),
        "operator_name": OPERATOR_NAMES[action],
        "q_values": q_dict,
        "f28_summary": {
            "mean_load_ratio": round(float(init_state[0]), 3),
            "max_load_ratio": round(float(init_state[2]), 3),
            "vehicles_used_ratio": round(float(init_state[4]), 3),
            "budget_progress": round(float(init_state[10]), 3),
            "cumulative_improvement": round(float(init_state[11]), 3)
        }
    }


@app.post("/api/benchmark")
def run_benchmark(req: BenchmarkRequest):
    """Run head-to-head comparison across 5 algorithms on identical starting solution."""
    inst = get_instance(req.instance_name)
    initial_routes = clarke_wright_savings(inst)
    initial_cost = float(inst.solution_cost(initial_routes))

    algos = req.algorithms or ["ddqn", "dqn", "greedy", "cyclic", "random"]
    algo_display_names = {
        "ddqn": "Double DQN Agent",
        "dqn": "Standard DQN Agent",
        "greedy": "Greedy Policy (Tham lam)",
        "cyclic": "Cyclic Policy (Xoay vòng)",
        "random": "Random Walk (Ngẫu nhiên)"
    }

    results = []
    for algo in algos:
        agent = get_agent(algo)
        routes = [r.copy() for r in initial_routes]
        extractor = StateExtractorF28(feature_patience=30.0)
        evals_used = 0
        improved_steps = 0

        t0 = time.perf_counter()
        for step_idx in range(req.budget_steps):
            state = extractor.extract(
                routes, inst, evals_used=evals_used, max_evals=2560, initial_cost=initial_cost
            )
            if algo.lower() == "greedy":
                action, _ = agent.select_operator(routes, inst)
            else:
                action, _ = agent.act(state)

            new_routes, delta, cand_count, improved = best_move(
                routes, action, inst, max_candidates=128
            )
            evals_used += cand_count
            if improved:
                routes = new_routes
                improved_steps += 1

            extractor.record_step(action, improved, delta)
            if evals_used >= 2560:
                break

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        final_cost = float(inst.solution_cost(routes))
        delta_cost = round(initial_cost - final_cost, 2)
        impr_pct = round(max(0.0, (delta_cost / max(1e-6, initial_cost)) * 100.0), 2)
        active_veh = len([r for r in routes if len(r) > 2])
        valid = IndependentValidator.validate(inst, routes)

        results.append({
            "algorithm": algo,
            "display_name": algo_display_names.get(algo, algo.upper()),
            "initial_cost": round(initial_cost, 2),
            "final_cost": round(final_cost, 2),
            "delta_cost": delta_cost,
            "savings_pct": impr_pct,
            "steps_run": req.budget_steps,
            "improved_steps": improved_steps,
            "evals_used": evals_used,
            "active_vehicles": active_veh,
            "cpu_time_ms": elapsed_ms,
            "is_feasible": valid["is_valid"],
            "routes": routes
        })

    results.sort(key=lambda x: (x["final_cost"], x["cpu_time_ms"]))
    medals = ["🥇", "🥈", "🥉", "#4", "#5", "#6"]
    for idx, r in enumerate(results):
        r["rank"] = idx + 1
        r["medal"] = medals[idx] if idx < len(medals) else f"#{idx + 1}"

    return {
        "instance_name": inst.name,
        "initial_cost": round(initial_cost, 2),
        "budget_steps": req.budget_steps,
        "leaderboard": results,
        "winner": results[0]
    }


@app.post("/api/compare_duel")
def compare_duel(req: CompareDuelRequest):
    """Run head-to-head duel comparison between 2 specific models on identical starting solution."""
    inst = get_instance(req.instance_name)
    initial_routes = clarke_wright_savings(inst)
    initial_cost = float(inst.solution_cost(initial_routes))

    algo_display_names = {
        "ddqn": "Double DQN Agent",
        "dqn": "Standard DQN Agent",
        "greedy": "Greedy Policy (Tham lam)",
        "cyclic": "Cyclic Policy (Xoay vòng)",
        "random": "Random Walk (Ngẫu nhiên)"
    }

    def solve_model(algo_name: str):
        agent = get_agent(algo_name)
        routes = [r.copy() for r in initial_routes]
        extractor = StateExtractorF28(feature_patience=30.0)
        evals_used = 0
        improved_steps = 0

        t0 = time.perf_counter()
        for step_idx in range(req.budget_steps):
            state = extractor.extract(
                routes, inst, evals_used=evals_used, max_evals=2560, initial_cost=initial_cost
            )
            if algo_name.lower() == "greedy":
                action, _ = agent.select_operator(routes, inst)
            else:
                action, _ = agent.act(state)

            new_routes, delta, cand_count, improved = best_move(
                routes, action, inst, max_candidates=128
            )
            evals_used += cand_count
            if improved:
                routes = new_routes
                improved_steps += 1

            extractor.record_step(action, improved, delta)
            if evals_used >= 2560:
                break

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        final_cost = float(inst.solution_cost(routes))
        delta_cost = round(initial_cost - final_cost, 2)
        impr_pct = round(max(0.0, (delta_cost / max(1e-6, initial_cost)) * 100.0), 2)
        active_veh = len([r for r in routes if len(r) > 2])
        valid = IndependentValidator.validate(inst, routes)

        return {
            "algorithm": algo_name,
            "display_name": algo_display_names.get(algo_name, algo_name.upper()),
            "initial_cost": round(initial_cost, 2),
            "final_cost": round(final_cost, 2),
            "delta_cost": delta_cost,
            "savings_pct": impr_pct,
            "steps_run": req.budget_steps,
            "improved_steps": improved_steps,
            "evals_used": evals_used,
            "active_vehicles": active_veh,
            "cpu_time_ms": elapsed_ms,
            "is_feasible": valid["is_valid"],
            "routes": routes
        }

    res_a = solve_model(req.model_a)
    res_b = solve_model(req.model_b)

    diff_km = round(abs(res_a["final_cost"] - res_b["final_cost"]), 2)
    winner_key = "model_a" if res_a["final_cost"] < res_b["final_cost"] else ("model_b" if res_b["final_cost"] < res_a["final_cost"] else "tie")

    return {
        "instance_name": inst.name,
        "initial_cost": round(initial_cost, 2),
        "budget_steps": req.budget_steps,
        "model_a": res_a,
        "model_b": res_b,
        "diff_km": diff_km,
        "winner_key": winner_key,
        "winner_name": res_a["display_name"] if winner_key == "model_a" else (res_b["display_name"] if winner_key == "model_b" else "Hòa nhau")
    }


@app.post("/api/init_comparison")
def init_comparison(req: InitComparisonRequest):
    """Initialize a synchronized dual comparison problem instance with locked configuration."""
    if len(req.selected_nodes) < 2:
        raise HTTPException(
            status_code=400,
            detail="Bắt buộc phải chọn ít nhất 2 điểm giao hàng để khởi tạo so sánh."
        )

    for item in req.selected_nodes:
        if item.demand <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"Điểm #{item.node_id} có số kiện {item.demand} <= 0. Không được tồn tại điểm giao 0 kiện."
            )
        if item.demand > req.capacity:
            raise HTTPException(
                status_code=400,
                detail=f"Điểm #{item.node_id} có số kiện {item.demand} vượt quá sức chứa xe ({req.capacity} kiện)."
            )

    base_inst = get_instance(req.base_instance)
    base_raw = load_raw_json(req.base_instance) if base_inst.is_geo else {}
    depot_raw = base_raw.get("depot", {})
    cust_raw_list = base_raw.get("customers", [])
    cust_map = {c["id"]: c for c in cust_raw_list}

    # Depot info
    depot_lat = float(base_inst.coords[0, 0])
    depot_lng = float(base_inst.coords[0, 1])
    depot_name = base_inst.node_names[0] if base_inst.node_names else "Kho Trung Tâm - Đại học FPT TP.HCM"
    depot_dict = {
        "id": 0,
        "name": depot_name,
        "lat": depot_lat,
        "lng": depot_lng,
        "demand": 0,
        "district": depot_raw.get("district", "TP. Thủ Đức"),
        "ward": depot_raw.get("ward", "Phường Long Thạnh Mỹ"),
        "address": depot_raw.get("address", "Lô E2a-7, Đường D1, Khu CNC, TP. Thủ Đức"),
        "is_depot": True
    }

    # Build customer nodes
    coords_list = [[depot_lat, depot_lng]]
    demands_list = [0]
    node_names = [depot_name]
    sub_nodes_info = [depot_dict]

    for new_id, item in enumerate(req.selected_nodes, start=1):
        orig_id = item.node_id
        if orig_id < len(base_inst.coords):
            lat = float(base_inst.coords[orig_id, 0])
            lng = float(base_inst.coords[orig_id, 1])
        else:
            lat = depot_lat + 0.01 * new_id
            lng = depot_lng + 0.01 * new_id

        c_info = cust_map.get(orig_id, {})
        name = base_inst.node_names[orig_id] if orig_id < len(base_inst.node_names) else f"Điểm #{new_id}"

        coords_list.append([lat, lng])
        demands_list.append(int(item.demand))
        node_names.append(name)
        sub_nodes_info.append({
            "id": new_id,
            "orig_id": orig_id,
            "name": name,
            "lat": lat,
            "lng": lng,
            "demand": int(item.demand),
            "district": c_info.get("district", "TP. Hồ Chí Minh"),
            "ward": c_info.get("ward", ""),
            "address": c_info.get("address", ""),
            "is_depot": False
        })

    coords = np.array(coords_list, dtype=np.float64)
    demands = np.array(demands_list, dtype=np.int32)
    n_nodes = len(coords)

    # Calculate optimal minimal fleet size K_min
    total_demand = sum(item.demand for item in req.selected_nodes)
    capacity = int(req.capacity)
    k_min = max(1, int(np.ceil(total_demand / capacity)))
    max_vehicles = k_min

    # Compute Haversine distance matrix
    dist_mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dist_mat[i, j] = haversine_distance(
                    coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]
                )

    sub_inst_name = f"Duel_{req.base_instance}_{int(time.time()*1000)}"
    sub_inst = CVRPInstance(
        name=sub_inst_name,
        num_customers=len(req.selected_nodes),
        capacity=capacity,
        max_vehicles=max_vehicles,
        coords=coords,
        demands=demands,
        distance_matrix=dist_mat,
        node_names=node_names,
        is_geo=True
    )

    INSTANCES_CACHE[sub_inst_name] = sub_inst
    RAW_JSON_CACHE[sub_inst_name] = {
        "depot": depot_dict,
        "customers": sub_nodes_info[1:]
    }

    # Generate identical initial solution S_0 using Clarke-Wright
    raw_routes = clarke_wright_savings(sub_inst)
    active_routes = [r for r in raw_routes if len(r) > 2]
    initial_cost = sub_inst.solution_cost(active_routes)
    validation = IndependentValidator.validate(sub_inst, active_routes)

    # Action Masking: if K_min <= 1, mask cross-route operators
    action_mask = [True, True, False, False] if len(active_routes) <= 1 else [True, True, True, True]

    # Pre-compute Model A Q-values at Step 0
    extractor_a = StateExtractorF28(feature_patience=30.0)
    init_state_a = extractor_a.extract(
        active_routes, sub_inst, evals_used=0, max_evals=2560, initial_cost=initial_cost
    )
    agent_a = get_agent(req.model_a)
    if req.model_a.lower() == "greedy":
        action_a, q_vals_a = agent_a.select_operator(active_routes, sub_inst, action_mask=action_mask)
    else:
        action_a, q_vals_a = agent_a.act(init_state_a, action_mask=action_mask)

    q_dict_a = {OPERATOR_NAMES[op]: float(q_vals_a[op]) for op in range(4)}

    return {
        "instance_name": sub_inst_name,
        "base_instance": req.base_instance,
        "district": req.district,
        "capacity": capacity,
        "k_min": k_min,
        "max_vehicles": max_vehicles,
        "total_demand": int(total_demand),
        "num_customers": len(req.selected_nodes),
        "nodes": sub_nodes_info,
        "routes": active_routes,
        "initial_cost": round(float(initial_cost), 2),
        "current_cost": round(float(initial_cost), 2),
        "num_active_routes": len(active_routes),
        "is_feasible": validation["is_valid"],
        "validation": validation,
        "action_mask": action_mask,
        "model_a": req.model_a,
        "model_b": req.model_b,
        "chosen_action_a": int(action_a),
        "operator_name_a": OPERATOR_NAMES[action_a],
        "q_values_a": q_dict_a,
        "f28_summary_a": {
            "mean_load_ratio": round(float(init_state_a[0]), 3),
            "max_load_ratio": round(float(init_state_a[2]), 3),
            "vehicles_used_ratio": round(float(init_state_a[4]), 3),
            "budget_progress": round(float(init_state_a[10]), 3),
            "cumulative_improvement": round(float(init_state_a[11]), 3)
        }
    }


@app.post("/api/compare_step")
def compare_step(req: CompareStepRequest):
    """Execute a single synchronized step for both Model A and Model B on their respective route copies."""
    inst = get_instance(req.instance_name)

    # 1. Step Model A
    active_count_a = len([r for r in req.routes_a if len(r) > 2])
    mask_a = [True, True, False, False] if active_count_a <= 1 else [True, True, True, True]

    extractor_a = StateExtractorF28(feature_patience=req.feature_patience)
    if req.history_calls_a and req.history_streaks_a and req.history_successes_a:
        for op in range(4):
            extractor_a.operator_stats[op].calls = req.history_calls_a[op]
            extractor_a.operator_stats[op].streak = req.history_streaks_a[op]
            extractor_a.operator_stats[op].successes = req.history_successes_a[op]

    state_a = extractor_a.extract(
        req.routes_a, inst, evals_used=req.evals_used_a, max_evals=2560, initial_cost=req.initial_cost
    )
    agent_a = get_agent(req.model_a)
    if req.model_a.lower() == "greedy":
        action_a, q_vals_a = agent_a.select_operator(req.routes_a, inst, action_mask=mask_a)
    else:
        action_a, q_vals_a = agent_a.act(state_a, action_mask=mask_a)

    new_routes_a, delta_a, cand_count_a, improved_a = best_move(
        req.routes_a, action_a, inst, max_candidates=128
    )
    cost_a = inst.solution_cost(new_routes_a)
    evals_a = req.evals_used_a + cand_count_a
    extractor_a.record_step(action_a, improved_a, delta_a)
    valid_a = IndependentValidator.validate(inst, new_routes_a)
    savings_a = round(max(0.0, ((req.initial_cost - cost_a) / max(1e-6, req.initial_cost)) * 100.0), 2)

    # 2. Step Model B
    active_count_b = len([r for r in req.routes_b if len(r) > 2])
    mask_b = [True, True, False, False] if active_count_b <= 1 else [True, True, True, True]

    extractor_b = StateExtractorF28(feature_patience=req.feature_patience)
    if req.history_calls_b and req.history_streaks_b and req.history_successes_b:
        for op in range(4):
            extractor_b.operator_stats[op].calls = req.history_calls_b[op]
            extractor_b.operator_stats[op].streak = req.history_streaks_b[op]
            extractor_b.operator_stats[op].successes = req.history_successes_b[op]

    state_b = extractor_b.extract(
        req.routes_b, inst, evals_used=req.evals_used_b, max_evals=2560, initial_cost=req.initial_cost
    )
    agent_b = get_agent(req.model_b)
    if req.model_b.lower() == "greedy":
        action_b, q_vals_b = agent_b.select_operator(req.routes_b, inst, action_mask=mask_b)
    else:
        action_b, q_vals_b = agent_b.act(state_b, action_mask=mask_b)

    new_routes_b, delta_b, cand_count_b, improved_b = best_move(
        req.routes_b, action_b, inst, max_candidates=128
    )
    cost_b = inst.solution_cost(new_routes_b)
    evals_b = req.evals_used_b + cand_count_b
    extractor_b.record_step(action_b, improved_b, delta_b)
    valid_b = IndependentValidator.validate(inst, new_routes_b)
    savings_b = round(max(0.0, ((req.initial_cost - cost_b) / max(1e-6, req.initial_cost)) * 100.0), 2)

    diff_km = round(abs(cost_a - cost_b), 2)
    winner_key = "model_a" if cost_a < cost_b else ("model_b" if cost_b < cost_a else "tie")
    q_dict_a = {OPERATOR_NAMES[op]: float(q_vals_a[op]) for op in range(4)}

    return {
        "step": req.step_count + 1,
        "instance_name": req.instance_name,
        "initial_cost": req.initial_cost,
        "diff_km": diff_km,
        "winner_key": winner_key,
        "model_a": {
            "algorithm": req.model_a,
            "routes": new_routes_a,
            "cost": round(float(cost_a), 2),
            "delta_cost": round(float(delta_a), 2),
            "improved": improved_a,
            "action": int(action_a),
            "operator_name": OPERATOR_NAMES[action_a],
            "action_mask": mask_a,
            "q_values": q_dict_a,
            "savings_pct": savings_a,
            "evals_used": evals_a,
            "is_feasible": valid_a["is_valid"],
            "history_calls": [extractor_a.operator_stats[op].calls for op in range(4)],
            "history_streaks": [extractor_a.operator_stats[op].streak for op in range(4)],
            "history_successes": [extractor_a.operator_stats[op].successes for op in range(4)],
            "f28_summary": {
                "mean_load_ratio": round(float(state_a[0]), 3),
                "max_load_ratio": round(float(state_a[2]), 3),
                "vehicles_used_ratio": round(float(state_a[4]), 3),
                "budget_progress": round(float(state_a[10]), 3),
                "cumulative_improvement": round(float(state_a[11]), 3)
            }
        },
        "model_b": {
            "algorithm": req.model_b,
            "routes": new_routes_b,
            "cost": round(float(cost_b), 2),
            "delta_cost": round(float(delta_b), 2),
            "improved": improved_b,
            "action": int(action_b),
            "operator_name": OPERATOR_NAMES[action_b],
            "action_mask": mask_b,
            "savings_pct": savings_b,
            "evals_used": evals_b,
            "is_feasible": valid_b["is_valid"],
            "history_calls": [extractor_b.operator_stats[op].calls for op in range(4)],
            "history_streaks": [extractor_b.operator_stats[op].streak for op in range(4)],
            "history_successes": [extractor_b.operator_stats[op].successes for op in range(4)]
        }
    }


@app.get("/api/geocode")
def geocode_location(q: str):
    """Geocode an address query using OpenStreetMap Nominatim with VN priority."""
    if not q or len(q.strip()) < 2:
        return {"results": []}

    clean_q = q.strip()
    landmarks = [
        {"name": "ĐH FPT TP.HCM (Khu CNC TP.Thủ Đức)", "lat": 10.8415, "lng": 106.8098, "district": "TP. Thủ Đức", "display_name": "Đại học FPT TP.HCM, Đường D1, Khu Công nghệ cao, TP. Thủ Đức, TP. Hồ Chí Minh"},
        {"name": "Chợ Bến Thành (Quận 1)", "lat": 10.7725, "lng": 106.6980, "district": "Quận 1", "display_name": "Chợ Bến Thành, Lê Lợi, Phường Bến Thành, Quận 1, TP. Hồ Chí Minh"},
        {"name": "Landmark 81 (Bình Thạnh)", "lat": 10.7951, "lng": 106.7218, "district": "Quận Bình Thạnh", "display_name": "Landmark 81, Vinhomes Central Park, Bình Thạnh, TP. Hồ Chí Minh"},
        {"name": "Sân bay Tân Sơn Nhất (Tân Bình)", "lat": 10.8185, "lng": 106.6588, "district": "Quận Tân Bình", "display_name": "Cảng hàng không quốc tế Tân Sơn Nhất, Quận Tân Bình, TP. Hồ Chí Minh"},
        {"name": "Khu chế xuất Tân Thuận (Quận 7)", "lat": 10.7562, "lng": 106.7285, "district": "Quận 7", "display_name": "Khu Chế Xuất Tân Thuận, Tân Thuận Đông, Quận 7, TP. Hồ Chí Minh"},
        {"name": "Bến xe Miền Đông Mới (TP. Thủ Đức)", "lat": 10.8778, "lng": 106.8322, "district": "TP. Thủ Đức", "display_name": "Bến xe Miền Đông Mới, Long Bình, TP. Thủ Đức, TP. Hồ Chí Minh"},
        {"name": "Aeon Mall Bình Tân", "lat": 10.7423, "lng": 106.6128, "district": "Quận Bình Tân", "display_name": "AEON Mall Bình Tân, Bình Trị Đông B, Quận Bình Tân, TP. Hồ Chí Minh"},
        {"name": "Công viên Phần mềm Quang Trung (Quận 12)", "lat": 10.8540, "lng": 106.6270, "district": "Quận 12", "display_name": "Công viên phần mềm Quang Trung, Tân Chánh Hiệp, Quận 12, TP. Hồ Chí Minh"}
    ]

    results = []
    try:
        encoded = urllib.parse.quote(f"{clean_q}, TP. Hồ Chí Minh, Vietnam")
        url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&addressdetails=1&limit=6&countrycodes=vn"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DQN-CVRP-Dispatcher/2.0 (Google Antigravity Agent; contact: tanphat@example.com)"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for item in data:
                lat = float(item["lat"])
                lng = float(item["lon"])
                addr = item.get("address", {})
                dist_name = addr.get("city_district") or addr.get("suburb") or addr.get("district") or addr.get("county") or "TP.HCM"
                results.append({
                    "name": item.get("name") or item.get("display_name", "").split(",")[0],
                    "display_name": item.get("display_name", ""),
                    "lat": lat,
                    "lng": lng,
                    "district": dist_name
                })
    except Exception:
        pass

    if len(results) < 2:
        q_lower = clean_q.lower()
        for lm in landmarks:
            if q_lower in lm["name"].lower() or q_lower in lm["display_name"].lower() or q_lower in lm["district"].lower():
                if not any(abs(r["lat"] - lm["lat"]) < 1e-4 and abs(r["lng"] - lm["lng"]) < 1e-4 for r in results):
                    results.append(lm)

    return {"results": results}


@app.post("/api/create_dynamic_scenario")
def create_dynamic_scenario(req: CreateDynamicScenarioRequest):
    """Dynamic CVRP Dispatch Endpoint:
    Accepts arbitrary depot, dynamic delivery points with demands di > 0, fleet constraints,
    computes real-road OSRM distance & duration matrices, solves Clarke-Wright S0,
    validates 100% feasibility, extracts F28 state, evaluates DDQN Q-values, and caches instance.
    """
    if not req.depot:
        raise HTTPException(status_code=400, detail="Thiếu thông tin Điểm xuất phát (Depot).")

    if not req.customers or len(req.customers) < 2:
        raise HTTPException(
            status_code=400,
            detail="Cần tối thiểu ít nhất 2 điểm giao hàng để tối ưu hóa lộ trình CVRP."
        )

    if req.capacity <= 0:
        raise HTTPException(status_code=400, detail="Sức chứa xe (Capacity Q) phải lớn hơn 0.")

    # Validate individual demands
    total_demand = 0
    for idx, c in enumerate(req.customers):
        if c.demand <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"Điểm giao hàng #{idx+1} '{c.name}' có số kiện hàng <= 0 ({c.demand} kiện). Bắt buộc di > 0."
            )
        if c.demand > req.capacity:
            raise HTTPException(
                status_code=400,
                detail=f"Điểm giao hàng '{c.name}' có nhu cầu {c.demand} kiện, vượt quá tải trọng xe ({req.capacity} kiện). Hãy tách đơn hàng."
            )
        total_demand += c.demand

    # Validate total fleet capacity
    total_capacity = req.max_vehicles * req.capacity
    if total_demand > total_capacity:
        missing_vehicles = math.ceil((total_demand - total_capacity) / req.capacity)
        raise HTTPException(
            status_code=400,
            detail=f"Tổng số kiện hàng ({total_demand} kiện) vượt quá tổng tải trọng tối đa của đội xe ({total_capacity} kiện cho {req.max_vehicles} xe). Cần bổ sung ít nhất {missing_vehicles} xe nữa hoặc giảm tải đơn hàng."
        )

    # Minimal vehicles required
    k_min = max(1, math.ceil(total_demand / req.capacity))
    fleet_max = max(k_min, req.max_vehicles)

    # Coordinates & demands array: Node 0 = Depot, Nodes 1..N = Customers
    coords = [[req.depot.lat, req.depot.lng]]
    demands = [0]
    node_names = [req.depot.name or "Kho trung tâm (Depot)"]
    districts = [req.depot.district or "TP.HCM"]

    for c in req.customers:
        coords.append([c.lat, c.lng])
        demands.append(int(c.demand))
        node_names.append(c.name)
        districts.append(c.district or "TP.HCM")

    # Compute OSRM distance and duration matrices
    dist_mat, duration_mat = osrm_service.compute_matrices(coords, metric=req.metric)
    used_mat = duration_mat if req.metric == "duration" else dist_mat

    # Build CVRPInstance
    scenario_id = req.scenario_name or f"Dynamic-{len(req.customers)}pts-{int(time.time()) % 10000}"
    instance = CVRPInstance(
        name=scenario_id,
        num_customers=len(req.customers),
        capacity=req.capacity,
        max_vehicles=fleet_max,
        coords=np.array(coords, dtype=np.float64),
        demands=np.array(demands, dtype=np.int32),
        distance_matrix=used_mat,
        node_names=node_names,
        is_geo=True
    )

    # Solve Clarke-Wright S0
    init_routes = clarke_wright_savings(instance)

    # Independent Constraint Validation
    valid_res = IndependentValidator.validate(instance, init_routes)
    if not valid_res["is_valid"]:
        raise HTTPException(
            status_code=500,
            detail=f"Nghiệm ban đầu không thỏa mãn ràng buộc CVRP: {valid_res['errors']}"
        )

    # Initial cost calculation
    cost = float(instance.solution_cost(init_routes))

    # Action Masking
    active_routes = [r for r in init_routes if len(r) > 2]
    if len(active_routes) <= 1:
        action_mask = [True, True, False, False]
    else:
        action_mask = [True, True, True, True]

    # F28 state extraction
    extractor = StateExtractorF28(feature_patience=30.0)
    s_0 = extractor.extract(init_routes, instance, evals_used=0, max_evals=2560, initial_cost=cost)

    # Preloaded DDQN initial Q-values
    q_dict = {"2-opt": 0.25, "intra-relocate": 0.25, "inter-relocate": 0.25, "inter-swap": 0.25}
    try:
        agent = get_agent("ddqn")
        q_vals = agent.predict_q_values(s_0)
        q_dict = {
            OPERATOR_NAMES[i]: round(float(q_vals[i]), 4) if action_mask[i] else 0.0
            for i in range(4)
        }
    except Exception:
        pass

    # Compute street-level road geometries via OSRM Route API
    road_geometries = []
    for r in active_routes:
        r_coords = [[float(coords[node][0]), float(coords[node][1])] for node in r]
        road_geometries.append(osrm_service.get_route_geometry(r_coords))

    # Compute active vs idle fleet details
    fleet_active = []
    for v_idx, r in enumerate(active_routes):
        d_sum = sum(demands[node] for node in r)
        fleet_active.append({
            "vehicle_id": v_idx + 1,
            "status": "Đang vận hành",
            "demand": int(d_sum),
            "capacity": req.capacity,
            "pct": round((d_sum / req.capacity) * 100.0, 1),
            "stops": len(r) - 2,
            "route": r
        })

    fleet_idle = []
    for v_idx in range(len(active_routes), fleet_max):
        fleet_idle.append({
            "vehicle_id": v_idx + 1,
            "status": "Chờ lệnh tại kho",
            "demand": 0,
            "capacity": req.capacity,
            "pct": 0.0,
            "stops": 0,
            "route": [0, 0]
        })

    # Cache instance
    INSTANCES_CACHE[scenario_id] = instance
    raw_nodes = [
        {
            "id": 0,
            "name": req.depot.name or "Kho trung tâm (Depot)",
            "lat": req.depot.lat,
            "lng": req.depot.lng,
            "demand": 0,
            "district": req.depot.district or "TP.HCM",
            "address": req.depot.address or "",
            "is_depot": True
        }
    ]
    for idx, c in enumerate(req.customers):
        raw_nodes.append({
            "id": idx + 1,
            "name": c.name,
            "lat": c.lat,
            "lng": c.lng,
            "demand": c.demand,
            "district": c.district or "TP.HCM",
            "address": c.address or "",
            "is_depot": False
        })

    RAW_JSON_CACHE[scenario_id] = {
        "name": scenario_id,
        "depot": raw_nodes[0],
        "customers": raw_nodes[1:],
        "capacity": req.capacity,
        "max_vehicles": fleet_max
    }

    return {
        "status": "success",
        "scenario_name": scenario_id,
        "num_customers": len(req.customers),
        "total_demand": total_demand,
        "k_min": k_min,
        "max_vehicles": fleet_max,
        "capacity": req.capacity,
        "metric": req.metric,
        "initial_cost": round(float(cost), 2),
        "current_cost": round(float(cost), 2),
        "routes": init_routes,
        "action_mask": action_mask,
        "q_values": q_dict,
        "nodes": raw_nodes,
        "fleet_active": fleet_active,
        "fleet_idle": fleet_idle,
        "road_geometries": road_geometries,
        "duration_matrix": duration_mat.tolist() if duration_mat is not None else None,
        "distance_matrix": dist_mat.tolist() if dist_mat is not None else None,
        "is_feasible": valid_res["is_valid"]
    }


if __name__ == "__main__":
    try:
        uvicorn.run(
            "src.web.server:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            reload_dirs=["src/web"],
            reload_excludes=[
                "*.log", "*.pt", "*.npy", "*.graphml", "*.json",
                "*.tmp", "*.pyc", "__pycache__", ".pytest_cache",
                "data/*", "models/*", "tests/*", "scripts/*", "docs/*"
            ],
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n[INFO] Server gracefully stopped by user.")
        import sys
        sys.exit(0)
