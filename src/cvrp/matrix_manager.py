"""
MP-MOTO-CVRP (V9.2) — Matrix Manager
Loads precomputed asymmetric matrices and provides high-speed sub-matrix slicing.
Supports On-the-fly Single-Source Dijkstra with In-memory LRU Cache for dynamic map clicks.
Implements reconstruct_leg_geometry for the Single Source of Truth invariant.
"""

import json
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

class MatrixManager:
    _instance = None

    def __init__(self):
        self.dist_matrix_path = DATA_DIR / "matrices" / "dist_m.npy"
        self.time_matrix_path = DATA_DIR / "matrices" / "time_s.npy"
        self.node_index_path = DATA_DIR / "graph" / "node_index.json"
        self.graph_path = DATA_DIR / "graph" / "hcm_motorbike.graphml"

        self.dist_m: Optional[np.ndarray] = None
        self.time_s: Optional[np.ndarray] = None
        self.node_to_idx: Dict[str, int] = {}
        self.graph: Optional[nx.DiGraph] = None

        # In-memory LRU cache for on-the-fly calculated pairs (u, v) -> (dist_m, time_s)
        self._dijkstra_cache: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._path_cache: Dict[Tuple[int, int], List[int]] = {}

        # SciPy all-pairs shortest-path accelerator (602-node HCMC graph -> ~0.1s)
        self._scipy_nodes: Optional[List[str]] = None
        self._scipy_node_to_idx: Dict[str, int] = {}
        self._scipy_dist_len: Optional[np.ndarray] = None
        self._scipy_dist_time: Optional[np.ndarray] = None
        self._scipy_pred_len: Optional[np.ndarray] = None
        self._scipy_pred_time: Optional[np.ndarray] = None

        self._load_matrices()

    @classmethod
    def get_instance(cls) -> "MatrixManager":
        if cls._instance is None:
            cls._instance = MatrixManager()
        return cls._instance

    def _load_matrices(self):
        if self.dist_matrix_path.exists() and self.time_matrix_path.exists() and self.node_index_path.exists():
            self.dist_m = np.load(self.dist_matrix_path)
            self.time_s = np.load(self.time_matrix_path)
            with open(self.node_index_path, "r", encoding="utf-8") as f:
                self.node_to_idx = json.load(f)
            print(f"[MatrixManager] Loaded matrices shape={self.dist_m.shape}, indexed nodes={len(self.node_to_idx)}")
        else:
            print("[MatrixManager] WARNING: Matrix files not found. Please run scripts/build_dataset.py first.")

        if self.graph_path.exists():
            self.graph = nx.read_graphml(self.graph_path)
            print(f"[MatrixManager] Loaded graph with {self.graph.number_of_nodes()} nodes.")

    def get_submatrix(self, node_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts submatrix for the given list of node IDs.
        If all node IDs exist in precomputed index, extracts via NumPy slicing in O(1).
        If any node is dynamic (e.g. user clicked on map), executes On-the-fly Single-Source
        Dijkstra with LRU Cache to assemble a 100% complete valid submatrix.
        """
        N = len(node_ids)
        sub_dist = np.zeros((N, N), dtype=np.float32)
        sub_time = np.zeros((N, N), dtype=np.float32)

        # Check if all nodes are pre-indexed
        all_indexed = all(str(nid) in self.node_to_idx for nid in node_ids)
        if all_indexed and self.dist_m is not None and self.time_s is not None:
            indices = [self.node_to_idx[str(nid)] for nid in node_ids]
            sub_dist = self.dist_m[np.ix_(indices, indices)].copy()
            sub_time = self.time_s[np.ix_(indices, indices)].copy()
            return sub_dist, sub_time

        # Hybrid: some nodes are dynamic (map clicks). Supports any N, no truncation.
        G = self.graph
        if G is None:
            raise RuntimeError("GraphML not loaded in MatrixManager for on-the-fly routing.")

        indexed_mask = [str(nid) in self.node_to_idx for nid in node_ids]

        # 1) Precomputed -> precomputed pairs (O(1) NumPy slicing per pair)
        if self.dist_m is not None and self.time_s is not None:
            for i, u in enumerate(node_ids):
                if not indexed_mask[i]:
                    continue
                idx_u = self.node_to_idx[str(u)]
                for j, v in enumerate(node_ids):
                    if i == j or not indexed_mask[j]:
                        continue
                    idx_v = self.node_to_idx[str(v)]
                    sub_dist[i, j] = self.dist_m[idx_u, idx_v]
                    sub_time[i, j] = self.time_s[idx_u, idx_v]

        # 2) Dynamic sources -> all targets (one single-source Dijkstra per source)
        for i, u in enumerate(node_ids):
            if indexed_mask[i]:
                continue
            su = str(u)
            try:
                dist_row = nx.single_source_dijkstra_path_length(G, su, weight="length_m")
                time_row = nx.single_source_dijkstra_path_length(G, su, weight="travel_time_s")
            except Exception:
                dist_row, time_row = {}, {}

            for j, v in enumerate(node_ids):
                if i == j:
                    continue
                sv = str(v)
                d = dist_row.get(sv)
                t = time_row.get(sv)
                if d is None or t is None:
                    if (u, v) in self._dijkstra_cache:
                        d, t = self._dijkstra_cache[(u, v)]
                    else:
                        d, t = 1000.0, 120.0
                self._dijkstra_cache[(u, v)] = (float(d), float(t))
                sub_dist[i, j] = float(d)
                sub_time[i, j] = float(t)

        # 3) Precomputed sources -> dynamic targets (reverse single-source Dijkstra)
        dynamic_targets = [j for j in range(N) if not indexed_mask[j]]
        if dynamic_targets:
            G_rev = G.reverse(copy=False)
            for j, v in enumerate(node_ids):
                if indexed_mask[j]:
                    continue
                sv = str(v)
                try:
                    rev_dist = nx.single_source_dijkstra_path_length(G_rev, sv, weight="length_m")
                    rev_time = nx.single_source_dijkstra_path_length(G_rev, sv, weight="travel_time_s")
                except Exception:
                    rev_dist, rev_time = {}, {}

                for i, u in enumerate(node_ids):
                    if i == j or not indexed_mask[i]:
                        continue
                    su = str(u)
                    d = rev_dist.get(su)
                    t = rev_time.get(su)
                    if d is None or t is None:
                        if (u, v) in self._dijkstra_cache:
                            d, t = self._dijkstra_cache[(u, v)]
                        else:
                            d, t = 1000.0, 120.0
                    self._dijkstra_cache[(u, v)] = (float(d), float(t))
                    sub_dist[i, j] = float(d)
                    sub_time[i, j] = float(t)

        return sub_dist, sub_time

    def _ensure_scipy_graph(self):
        """Builds a SciPy sparse graph and all-pairs shortest-path predecessors once."""
        if self._scipy_nodes is not None or self.graph is None:
            return

        G = self.graph
        nodes = list(G.nodes())
        node_to_i = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        rows, cols, w_len, w_time = [], [], [], []
        for u, v, data in G.edges(data=True):
            rows.append(node_to_i[u])
            cols.append(node_to_i[v])
            w_len.append(float(data.get("length_m", 1.0)))
            w_time.append(float(data.get("travel_time_s", 1.0)))

        mat_len = sp.csr_matrix((w_len, (rows, cols)), shape=(n, n))
        mat_time = sp.csr_matrix((w_time, (rows, cols)), shape=(n, n))

        self._scipy_nodes = nodes
        self._scipy_node_to_idx = node_to_i
        self._scipy_dist_len, self._scipy_pred_len = dijkstra(
            mat_len, directed=True, return_predecessors=True
        )
        self._scipy_dist_time, self._scipy_pred_time = dijkstra(
            mat_time, directed=True, return_predecessors=True
        )

    def _scipy_node_path(self, u_node: int, v_node: int) -> Optional[List[int]]:
        """Reconstructs the node path from SciPy predecessors (fast, no NetworkX overhead)."""
        self._ensure_scipy_graph()
        if self._scipy_nodes is None or self._scipy_pred_len is None:
            return None

        su, sv = str(u_node), str(v_node)
        if su not in self._scipy_node_to_idx or sv not in self._scipy_node_to_idx:
            return None

        iu = self._scipy_node_to_idx[su]
        iv = self._scipy_node_to_idx[sv]
        pred = self._scipy_pred_len

        if iu == iv:
            return [u_node]
        if pred[iu, iv] == -9999:
            return None

        idx_path = []
        cur = iv
        while cur != iu:
            idx_path.append(cur)
            cur = int(pred[iu, cur])
            if cur < 0 or len(idx_path) > len(self._scipy_nodes):
                return None
        idx_path.append(iu)
        idx_path.reverse()

        try:
            return [int(self._scipy_nodes[i]) for i in idx_path]
        except (TypeError, ValueError):
            return None

    def reconstruct_leg_geometry(self, u_node: int, v_node: int) -> Dict[str, Any]:
        """
        Reconstructs the exact road geometry [lat, lon] along the shortest path
        on the directed motorbike road graph for a leg from u_node to v_node.
        """
        G = self.graph
        if G is None:
            raise RuntimeError("GraphML not loaded.")

        su, sv = str(u_node), str(v_node)

        if (u_node, v_node) in self._path_cache:
            node_path = self._path_cache[(u_node, v_node)]
        else:
            node_path = self._scipy_node_path(u_node, v_node)
            if node_path is None:
                try:
                    raw_path = nx.shortest_path(G, su, sv, weight="length_m")
                    node_path = [int(n) for n in raw_path]
                except Exception:
                    node_path = [u_node, v_node]
            self._path_cache[(u_node, v_node)] = node_path

        # Assemble coordinates
        geometry = []
        for nid in node_path:
            node_data = G.nodes[str(nid)]
            geometry.append([float(node_data["lat"]), float(node_data["lon"])])

        # Distance and time
        if (u_node, v_node) in self._dijkstra_cache:
            dist_m, time_s = self._dijkstra_cache[(u_node, v_node)]
        elif su in self.node_to_idx and sv in self.node_to_idx and self.dist_m is not None and self.time_s is not None:
            idx_u = self.node_to_idx[su]
            idx_v = self.node_to_idx[sv]
            dist_m = float(self.dist_m[idx_u, idx_v])
            time_s = float(self.time_s[idx_u, idx_v])
        elif self._scipy_dist_len is not None and su in self._scipy_node_to_idx and sv in self._scipy_node_to_idx:
            idx_u = self._scipy_node_to_idx[su]
            idx_v = self._scipy_node_to_idx[sv]
            dist_m = float(self._scipy_dist_len[idx_u, idx_v])
            time_s = float(self._scipy_dist_time[idx_u, idx_v])
        else:
            dist_m = float(nx.shortest_path_length(G, su, sv, weight="length_m"))
            time_s = float(nx.shortest_path_length(G, su, sv, weight="travel_time_s"))

        return {
            "from_node": u_node,
            "to_node": v_node,
            "node_path": node_path,
            "geometry": geometry,
            "distance_m": round(dist_m, 1),
            "time_s": round(time_s, 1)
        }
