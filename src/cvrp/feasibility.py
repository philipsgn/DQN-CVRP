"""
MP-MOTO-CVRP (V9.2) — Multi-Tier Feasibility Gate
Enforces 4 strict feasibility tiers before calling any CVRP solver:
  Tier 1: Overweight item check (d_i > Q_kg)
  Tier 2: Total demand check (sum(d_i) > K * Q_kg)
  Tier 3: Bin-packing feasibility (FFD fast heuristic + OR-Tools CP-SAT exact solver)
  Tier 4: Graph reachability check (all nodes must be in the strongly connected component)
"""

import math
from typing import List, Dict, Any, Tuple, Optional
from ortools.sat.python import cp_model

class FeasibilityResult:
    def __init__(self, is_feasible: bool, reason: str = "", code: str = "OK", metadata: Optional[Dict[str, Any]] = None):
        self.is_feasible = is_feasible
        self.reason = reason
        self.code = code
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_feasible": self.is_feasible,
            "code": self.code,
            "reason": self.reason,
            "metadata": self.metadata
        }

def first_fit_decreasing_bins(demands: List[float], capacity: float) -> int:
    """Fast First-Fit Decreasing bin packing."""
    sorted_demands = sorted(demands, reverse=True)
    bins: List[float] = []

    for d in sorted_demands:
        placed = False
        for b_idx in range(len(bins)):
            if bins[b_idx] + d <= capacity + 1e-5:
                bins[b_idx] += d
                placed = True
                break
        if not placed:
            bins.append(d)
    return len(bins)

def solve_exact_bin_packing(demands: List[float], capacity: float, max_bins: int, time_limit_s: float = 1.5) -> Tuple[bool, int]:
    """
    Exact Bin Packing problem using OR-Tools CP-SAT solver.
    Returns: (is_feasible_with_max_bins, min_bins_needed)
    """
    n_items = len(demands)
    if n_items == 0:
        return True, 0

    # Upper bound on bins
    upper_bound = min(n_items, max_bins + 4)
    model = cp_model.CpModel()

    # x[i, j] = 1 if item i is placed in bin j
    x = {}
    for i in range(n_items):
        for j in range(upper_bound):
            x[i, j] = model.NewBoolVar(f"x_{i}_{j}")

    # y[j] = 1 if bin j is used
    y = [model.NewBoolVar(f"y_{j}") for j in range(upper_bound)]

    # Each item must be in exactly one bin
    for i in range(n_items):
        model.Add(sum(x[i, j] for j in range(upper_bound)) == 1)

    # Bin capacity constraint & linking x with y
    for j in range(upper_bound):
        model.Add(sum(int(round(demands[i] * 100)) * x[i, j] for i in range(n_items)) <= int(round(capacity * 100)) * y[j])

    # Symmetry breaking: y[j] >= y[j+1]
    for j in range(upper_bound - 1):
        model.Add(y[j] >= y[j + 1])

    # Minimize number of bins used
    model.Minimize(sum(y))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        bins_used = sum(int(solver.Value(y[j])) for j in range(upper_bound))
        return (bins_used <= max_bins), bins_used

    # If timeout or undetermined, use FFD estimate
    ffd_estimate = first_fit_decreasing_bins(demands, capacity)
    return (ffd_estimate <= max_bins), ffd_estimate

def check_feasibility(
    demands: List[float],
    k_vehicles: int,
    capacity_kg: float,
    node_ids: Optional[List[int]] = None,
    valid_scc_nodes: Optional[set] = None,
    store_names: Optional[List[str]] = None
) -> FeasibilityResult:
    """
    Executes the 4-tier feasibility gate for Motorbike CVRP.
    """
    n_stores = len(demands)
    if n_stores == 0:
        return FeasibilityResult(False, "Danh sách cửa hàng giao hàng đang trống.", "EMPTY_STORES")

    if k_vehicles < 1:
        return FeasibilityResult(False, "Số xe máy sẵn sàng phải ít nhất là 1.", "INVALID_FLEET")

    if capacity_kg <= 0:
        return FeasibilityResult(False, "Tải trọng xe máy phải lớn hơn 0.", "INVALID_CAPACITY")

    # -------------------------------------------------------------
    # TIER 1: Individual Item Overweight Check
    # -------------------------------------------------------------
    for i, d in enumerate(demands):
        if d > capacity_kg + 1e-5:
            st_name = store_names[i] if store_names and i < len(store_names) else f"Cửa hàng #{i+1}"
            splits_needed = math.ceil(d / capacity_kg)
            return FeasibilityResult(
                is_feasible=False,
                code="ORDER_EXCEEDS_Q",
                reason=(
                    f"Đơn hàng tại '{st_name}' ({d:.1f} kg) vượt quá tải trọng xe ({capacity_kg:.1f} kg). "
                    f"Cần tách thành ít nhất {splits_needed} chuyến hoặc nâng tải trọng xe."
                ),
                metadata={
                    "violating_index": i,
                    "store_name": st_name,
                    "demand_kg": d,
                    "capacity_kg": capacity_kg,
                    "splits_suggested": splits_needed
                }
            )

    # -------------------------------------------------------------
    # TIER 2: Total Demand Capacity Check
    # -------------------------------------------------------------
    total_demand = sum(demands)
    max_fleet_capacity = k_vehicles * capacity_kg
    k_min_by_weight = math.ceil(total_demand / capacity_kg)

    if total_demand > max_fleet_capacity + 1e-5:
        return FeasibilityResult(
            is_feasible=False,
            code="TOTAL_EXCEEDS_CAPACITY",
            reason=(
                f"Tổng khối lượng ({total_demand:.1f} kg) vượt quá tổng tải trọng của {k_vehicles} xe "
                f"({max_fleet_capacity:.1f} kg). Cần điều động ít nhất {k_min_by_weight} xe máy hoặc giảm số lượng đơn."
            ),
            metadata={
                "total_demand_kg": total_demand,
                "max_fleet_capacity_kg": max_fleet_capacity,
                "k_available": k_vehicles,
                "k_min_by_weight": k_min_by_weight
            }
        )

    # -------------------------------------------------------------
    # TIER 3: Bin-Packing Feasibility Check (FFD + CP-SAT)
    # -------------------------------------------------------------
    ffd_bins = first_fit_decreasing_bins(demands, capacity_kg)
    if ffd_bins > k_vehicles:
        # FFD failed to fit into k_vehicles. Run exact CP-SAT solver to verify if a valid packing exists.
        # Large batches get a tighter budget to keep the dispatch API responsive.
        cp_time_limit = 0.25 if n_stores > 200 else 1.5
        is_exact_fit, exact_min_bins = solve_exact_bin_packing(
            demands, capacity_kg, k_vehicles, time_limit_s=cp_time_limit
        )
        if not is_exact_fit:
            return FeasibilityResult(
                is_feasible=False,
                code="BINPACKING_INFEASIBLE",
                reason=(
                    f"Dù tổng tải trọng ({total_demand:.1f} kg) nhỏ hơn tổng sức chứa ({max_fleet_capacity:.1f} kg), "
                    f"các kiện hàng không thể xếp vừa vào {k_vehicles} xe do giới hạn kích thước kiện (Bin-Packing). "
                    f"Cần ít nhất {exact_min_bins} xe máy để chở hết hàng."
                ),
                metadata={
                    "total_demand_kg": total_demand,
                    "k_available": k_vehicles,
                    "k_min_binpacking": exact_min_bins
                }
            )

    # -------------------------------------------------------------
    # TIER 4: Graph Reachability Check (SCC)
    # -------------------------------------------------------------
    if node_ids is not None and valid_scc_nodes is not None:
        for idx, nid in enumerate(node_ids):
            if nid not in valid_scc_nodes:
                st_name = store_names[idx] if store_names and idx < len(store_names) else f"Điểm #{idx}"
                return FeasibilityResult(
                    is_feasible=False,
                    code="UNREACHABLE_NODE",
                    reason=f"Điểm '{st_name}' (Node {nid}) nằm ngoài vùng liên thông của mạng lưới xe máy.",
                    metadata={"unreachable_node_id": nid}
                )

    return FeasibilityResult(
        is_feasible=True,
        code="OK",
        reason="Bài toán hoàn toàn khả thi qua 4 tầng kiểm tra.",
        metadata={
            "total_demand_kg": total_demand,
            "k_available": k_vehicles,
            "k_min_by_weight": k_min_by_weight
        }
    )

