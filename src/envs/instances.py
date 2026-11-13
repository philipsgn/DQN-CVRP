"""CVRPLIB benchmark instance pipeline.

Downloads instances and best-known solutions from the official CVRPLIB
mirror (github.com/vidalt/CVRPLIB), caches them locally, and exposes a
parsed representation with the TSPLIB EUC_2D rounded distance matrix so
optimality gaps are computed on the same convention as published
best-known costs.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MIRROR = "https://raw.githubusercontent.com/vidalt/CVRPLIB/main"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "instances"

# name -> CVRPLIB set directory. 4 small (n<=25), 4 medium (n<=50).
INSTANCES: dict[str, str] = {
    "P-n16-k8": "P",
    "P-n20-k2": "P",
    "E-n22-k4": "E",
    "E-n23-k3": "E",
    "A-n32-k5": "A",
    "A-n33-k5": "A",
    "B-n31-k5": "B",
    "A-n45-k6": "A",
}

SMALL = ["P-n16-k8", "P-n20-k2", "E-n22-k4", "E-n23-k3"]
MEDIUM = ["A-n32-k5", "A-n33-k5", "B-n31-k5", "A-n45-k6"]


@dataclass
class CVRPInstance:
    name: str
    coords: np.ndarray          # (n+1, 2) — index 0 is the depot
    demands: np.ndarray         # (n+1,)   — demand[0] == 0
    capacity: int
    dist: np.ndarray            # (n+1, n+1) rounded EUC_2D matrix
    best_known_cost: float
    best_known_routes: list[list[int]] = field(default_factory=list)

    @property
    def n_customers(self) -> int:
        return len(self.demands) - 1


def _fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as resp:
        dest.write_bytes(resp.read())


def download(name: str) -> tuple[Path, Path]:
    """Download (with caching) the .vrp and .sol files for an instance."""
    group = INSTANCES[name]
    vrp = DATA_DIR / f"{name}.vrp"
    sol = DATA_DIR / f"{name}.sol"
    _fetch(f"{MIRROR}/{group}/Instances/{name}.vrp", vrp)
    _fetch(f"{MIRROR}/{group}/Solutions/{name}.sol", sol)
    return vrp, sol


def euc_2d_matrix(coords: np.ndarray) -> np.ndarray:
    """TSPLIB EUC_2D: Euclidean distance rounded to nearest integer."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.round(np.sqrt((diff**2).sum(-1)))


def route_cost(routes: list[list[int]], dist: np.ndarray) -> float:
    """Cost of a set of depot-to-depot routes (customer indices, 1-based)."""
    total = 0.0
    for route in routes:
        if not route:
            continue
        path = [0, *route, 0]
        total += sum(dist[a, b] for a, b in zip(path, path[1:]))
    return total


_ROUTE_RE = re.compile(r"^Route\s*#\s*\d+\s*:\s*(.*)$", re.IGNORECASE)
_COST_RE = re.compile(r"^Cost\s+([0-9.]+)", re.IGNORECASE)


def _read_vrp(path: Path) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    """Parse the TSPLIB/CVRPLIB sections needed by this project."""
    headers: dict[str, str] = {}
    coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, int] = {}
    section = ""

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF"):
            section = line
            if line == "EOF":
                break
            continue
        if section == "NODE_COORD_SECTION":
            node, x, y = line.split()[:3]
            coords[int(node)] = (float(x), float(y))
        elif section == "DEMAND_SECTION":
            node, demand = line.split()[:2]
            demands[int(node)] = int(demand)
        elif section == "DEPOT_SECTION":
            continue
        elif ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().upper()] = value.strip()

    dimension = int(headers["DIMENSION"])
    node_ids = range(1, dimension + 1)
    coords_arr = np.asarray([coords[i] for i in node_ids], dtype=float)
    demands_arr = np.asarray([demands.get(i, 0) for i in node_ids], dtype=int)
    return headers, coords_arr, demands_arr


def _read_solution(path: Path) -> tuple[list[list[int]], float]:
    routes: list[list[int]] = []
    cost: float | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        route_match = _ROUTE_RE.match(line)
        if route_match:
            routes.append([int(node) for node in route_match.group(1).split()])
            continue
        cost_match = _COST_RE.match(line)
        if cost_match:
            cost = float(cost_match.group(1))
    if cost is None:
        raise ValueError(f"No 'Cost' line found in {path}")
    return routes, cost


def load(name: str) -> CVRPInstance:
    vrp_path, sol_path = download(name)
    headers, coords, demands = _read_vrp(vrp_path)
    routes, best_known_cost = _read_solution(sol_path)

    return CVRPInstance(
        name=name,
        coords=coords,
        demands=demands,
        capacity=int(headers["CAPACITY"]),
        dist=euc_2d_matrix(coords),
        best_known_cost=best_known_cost,
        best_known_routes=routes,
    )


def load_all() -> dict[str, CVRPInstance]:
    return {name: load(name) for name in INSTANCES}


if __name__ == "__main__":
    print(f"{'instance':<12} {'n':>3} {'cap':>5} {'BKS':>7} {'reproduced':>10} {'match':>6}")
    for name in INSTANCES:
        inst = load(name)
        reproduced = route_cost(inst.best_known_routes, inst.dist)
        ok = "OK" if reproduced == inst.best_known_cost else "FAIL"
        print(
            f"{inst.name:<12} {inst.n_customers:>3} {inst.capacity:>5} "
            f"{inst.best_known_cost:>7.0f} {reproduced:>10.0f} {ok:>6}"
        )
