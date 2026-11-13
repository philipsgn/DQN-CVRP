"""OSRM Routing & Distance/Duration Matrix Service with Local Caching and Robust Fallback.
Computes real-world road network distance matrix (km) and travel duration matrix (minutes)
using the Open Source Routing Machine (OSRM) Table API.
"""
from __future__ import annotations
import json
import math
import time
import urllib.request
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from src.cvrp.instance import haversine_distance


class DynamicOSRMService:
    """Service to compute real-road distance and duration matrices with smart caching."""

    def __init__(self, base_url: str = "https://router.project-osrm.org"):
        self.base_url = base_url.rstrip("/")
        # Smart local cache: key is "lat1,lng1_lat2,lng2" -> (dist_km, time_min)
        self._pair_cache: Dict[str, Tuple[float, float]] = {}
        # Geometry cache: key is route signature -> list of [lat, lng]
        self._route_cache: Dict[str, List[List[float]]] = {}

    def get_route_geometry(self, route_coords: List[List[float]]) -> List[List[float]]:
        """Fetch street-level road polyline coordinates via OSRM Route API.
        route_coords is a list of [lat, lng] in traveling order.
        Returns detailed list of [lat, lng] waypoints for map polyline.
        """
        if len(route_coords) < 2:
            return route_coords

        cache_key = "_".join([f"{c[0]:.4f},{c[1]:.4f}" for c in route_coords])
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]

        try:
            # OSRM expects coordinates in lng,lat format!
            coords_str = ";".join([f"{c[1]:.5f},{c[0]:.5f}" for c in route_coords])
            url = f"{self.base_url}/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "DQN-CVRP-Dispatcher/2.0 (Google Antigravity Agent)"}
            )
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("code") == "Ok" and data.get("routes"):
                    geom = data["routes"][0].get("geometry", {})
                    coordinates = geom.get("coordinates", [])
                    if coordinates:
                        # Convert [lng, lat] back to [lat, lng]
                        lat_lng_polyline = [[round(pt[1], 5), round(pt[0], 5)] for pt in coordinates]
                        self._route_cache[cache_key] = lat_lng_polyline
                        return lat_lng_polyline
        except Exception:
            pass

        # Fallback to direct waypoints
        return route_coords

    def compute_matrices(
        self,
        coords: List[List[float]],
        metric: str = "distance"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (N+1) x (N+1) distance (km) and duration (minutes) matrices.
        coords[0] is Depot, coords[1..N] are delivery locations.
        coords format: [[lat0, lng0], [lat1, lng1], ..., [latN, lngN]]
        """
        n = len(coords)
        dist_mat = np.zeros((n, n), dtype=np.float64)
        time_mat = np.zeros((n, n), dtype=np.float64)

        if n <= 1:
            return dist_mat, time_mat

        # Helper: Haversine fallback with urban detour factor (1.35x) and 28 km/h city speed
        def apply_haversine_fallback():
            for i in range(n):
                for j in range(n):
                    if i != j:
                        d = haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1]) * 1.35
                        t = (d / 28.0) * 60.0  # minutes
                        dist_mat[i, j] = round(d, 3)
                        time_mat[i, j] = round(t, 2)

        # First, check if all pairs are cached
        all_cached = True
        for i in range(n):
            for j in range(n):
                if i != j:
                    key = f"{coords[i][0]:.5f},{coords[i][1]:.5f}_{coords[j][0]:.5f},{coords[j][1]:.5f}"
                    if key in self._pair_cache:
                        dist_mat[i, j], time_mat[i, j] = self._pair_cache[key]
                    else:
                        all_cached = False

        if all_cached:
            return dist_mat, time_mat

        # Try OSRM Table API
        try:
            # OSRM expects coordinates in lng,lat format!
            coords_str = ";".join([f"{c[1]:.5f},{c[0]:.5f}" for c in coords])
            url = f"{self.base_url}/table/v1/driving/{coords_str}?annotations=distance,duration"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "DQN-CVRP-Dispatcher/2.0 (Google Antigravity Agent)"}
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("code") == "Ok":
                    distances = data.get("distances", [])
                    durations = data.get("durations", [])
                    if len(distances) == n and len(durations) == n:
                        for i in range(n):
                            for j in range(n):
                                if i != j:
                                    raw_d = distances[i][j]
                                    raw_t = durations[i][j]
                                    d_km = round(raw_d / 1000.0, 3) if raw_d is not None else haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1]) * 1.35
                                    t_min = round(raw_t / 60.0, 2) if raw_t is not None else (d_km / 28.0) * 60.0
                                    dist_mat[i, j] = d_km
                                    time_mat[i, j] = t_min
                                    # Cache the pair
                                    key = f"{coords[i][0]:.5f},{coords[i][1]:.5f}_{coords[j][0]:.5f},{coords[j][1]:.5f}"
                                    self._pair_cache[key] = (d_km, t_min)
                        return dist_mat, time_mat
        except Exception:
            # Fall back safely
            pass

        apply_haversine_fallback()
        return dist_mat, time_mat
