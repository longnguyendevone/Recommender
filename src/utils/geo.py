"""Vectorized Haversine distance calculations and BallTree spatial queries."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Compute great-circle distance in kilometers between WGS84 coordinates."""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))
    return EARTH_RADIUS_KM * c


def build_ball_tree(coords_deg: np.ndarray) -> BallTree:
    """Build a Haversine BallTree from an (N, 2) array of [lat, lon] in degrees."""
    coords_arr = np.asarray(coords_deg, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        raise ValueError(f"Expected shape (N, 2) for coords_deg, got {coords_arr.shape}")
    return BallTree(np.radians(coords_arr), metric="haversine")


def query_radius_km(
    tree: BallTree,
    lat: float,
    lon: float,
    radius_km: float,
    sort_results: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Query POI indices and distances (km) within radius_km of (lat, lon)."""
    query_rad = np.radians(np.array([[lat, lon]], dtype=np.float64))
    radius_rad = float(radius_km) / EARTH_RADIUS_KM
    indices_arr, dist_rad_arr = tree.query_radius(
        query_rad,
        r=radius_rad,
        return_distance=True,
        sort_results=sort_results,
    )
    return indices_arr[0].astype(np.int64), (dist_rad_arr[0] * EARTH_RADIUS_KM).astype(
        np.float64
    )
