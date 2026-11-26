"""Global and distance-aware local popularity recommender (CPU)."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseRecommender
from src.utils.geo import build_ball_tree, haversine_km, query_radius_km
from src.utils.io import ensure_dir


class PopularityRecommender(BaseRecommender):
    """Combines normalized log-popularity with exponential spatial decay within a local radius."""

    name = "popularity"

    def __init__(self, local_radius_km: float = 3.0, local_weight: float = 0.6) -> None:
        self.local_radius_km = float(local_radius_km)
        self.local_weight = float(local_weight)
        self.poi_indices: np.ndarray | None = None
        self.global_pop: np.ndarray | None = None
        self.coords_deg: np.ndarray | None = None
        self.ball_tree: Any = None
        self.top_global_pos: np.ndarray | None = None

    def fit(self, train_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        poi_meta = train_data["poi_metadata"]
        self.poi_indices = poi_meta["poi_idx"].to_numpy(dtype=np.int64)
        raw_pop = poi_meta["train_popularity"].to_numpy(dtype=np.float64)
        log_pop = np.log1p(raw_pop)
        max_pop = float(log_pop.max()) if len(log_pop) > 0 else 1.0
        self.global_pop = (log_pop / max(max_pop, 1e-12)).astype(np.float32)

        self.coords_deg = poi_meta[["lat", "lon"]].to_numpy(dtype=np.float64)
        self.ball_tree = train_data.get("ball_tree") or build_ball_tree(self.coords_deg)
        self.top_global_pos = np.argsort(-self.global_pop)[:100].astype(np.int64)

        elapsed = time.perf_counter() - t0
        return {"model": self.name, "fit_time_s": round(elapsed, 4), "num_pois": len(self.poi_indices)}

    def _score_positions(self, lat: float, lon: float, positions: np.ndarray) -> np.ndarray:
        assert self.global_pop is not None and self.coords_deg is not None
        g = self.global_pop[positions].astype(np.float64)
        dists = haversine_km(
            lat,
            lon,
            self.coords_deg[positions, 0],
            self.coords_deg[positions, 1],
        )
        within = dists <= self.local_radius_km
        local_boost = np.where(within, np.exp(-dists / max(self.local_radius_km, 1e-6)) * g, 0.0)
        return ((1.0 - self.local_weight) * g + self.local_weight * local_boost).astype(np.float32)

    def score_candidates(
        self, query: dict[str, Any], candidate_poi_indices: np.ndarray
    ) -> np.ndarray:
        cand = np.asarray(candidate_poi_indices, dtype=np.int64)
        positions = cand - 2
        return self._score_positions(float(query["last_lat"]), float(query["last_lon"]), positions)

    def predict_topk(
        self, queries: list[dict[str, Any]], k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        assert (
            self.poi_indices is not None
            and self.ball_tree is not None
            and self.top_global_pos is not None
        )
        n = len(queries)
        out_idx = np.zeros((n, k), dtype=np.int64)
        out_scores = np.zeros((n, k), dtype=np.float32)
        global_pool = self.top_global_pos[: max(k, 50)]

        for i, q in enumerate(queries):
            lat = float(q["last_lat"])
            lon = float(q["last_lon"])
            local_pos, _ = query_radius_km(
                self.ball_tree, lat, lon, self.local_radius_km, sort_results=False
            )
            if len(local_pos) > 0:
                cand_pos = np.unique(np.concatenate([local_pos, global_pool]))
            else:
                cand_pos = global_pool

            scores = self._score_positions(lat, lon, cand_pos)
            k_eff = min(k, len(scores))
            top_rel = np.argpartition(-scores, k_eff - 1)[:k_eff]
            top_rel = top_rel[np.argsort(-scores[top_rel])]

            chosen_pos = cand_pos[top_rel]
            out_idx[i, :k_eff] = self.poi_indices[chosen_pos]
            out_scores[i, :k_eff] = scores[top_rel]

        return out_idx, out_scores

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        ensure_dir(out_path.parent)
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "local_radius_km": self.local_radius_km,
                    "local_weight": self.local_weight,
                    "poi_indices": self.poi_indices,
                    "global_pop": self.global_pop,
                    "coords_deg": self.coords_deg,
                    "top_global_pos": self.top_global_pos,
                },
                f,
            )
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "PopularityRecommender":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            local_radius_km=state["local_radius_km"],
            local_weight=state["local_weight"],
        )
        obj.poi_indices = state["poi_indices"]
        obj.global_pop = state["global_pop"]
        obj.coords_deg = state["coords_deg"]
        obj.top_global_pos = state["top_global_pos"]
        obj.ball_tree = build_ball_tree(obj.coords_deg)
        return obj
