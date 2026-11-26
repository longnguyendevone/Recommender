"""CPU-only inference service loading trained NYC Next-POI artifacts."""

from __future__ import annotations

import math
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.backend.schemas import (
    CategoryCount,
    NearbyPOI,
    POIRecommendation,
    POISummary,
    RecommendRequest,
    RecommendResponse,
    SampleTrajectory,
    ScoreBreakdown,
)
from src.models.cf_bpr import CFBPRRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender, _minmax_norm
from src.models.popularity import PopularityRecommender
from src.models.sequential import SequentialRecommender
from src.models.sequential_cl import CL4SRecRecommender
from src.utils.geo import haversine_km, query_radius_km
from src.utils.io import load_json, load_parquet

MIDTOWN_LAT = 40.7580
MIDTOWN_LON = -73.9855


class RecommenderService:
    """Loads metadata, BallTree, and all 6 models on CPU to serve low-latency Top-K."""

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        models_dir: str | Path = "artifacts/models",
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.device = "cpu"

        self.mappings = load_json(self.processed_dir / "mappings.json")
        self.user2idx = {int(k): int(v) for k, v in self.mappings["user2idx"].items()}
        self.poi2idx = {int(k): int(v) for k, v in self.mappings["poi2idx"].items()}

        self.poi_meta = load_parquet(self.processed_dir / "poi_metadata.parquet")
        with open(self.processed_dir / "poi_balltree.pkl", "rb") as f:
            bt_payload = pickle.load(f)
        self.ball_tree = bt_payload["tree"]

        self.poi_indices = self.poi_meta["poi_idx"].to_numpy(dtype=np.int64)
        self.venue_ids = self.poi_meta["venue_id"].to_numpy(dtype=np.int64)
        self.names = self.poi_meta["name"].astype(str).to_numpy()
        self.categories_arr = self.poi_meta["category"].astype(str).to_numpy()
        self.cat_indices = self.poi_meta["cat_idx"].to_numpy(dtype=np.int64)
        self.lats = self.poi_meta["lat"].to_numpy(dtype=np.float64)
        self.lons = self.poi_meta["lon"].to_numpy(dtype=np.float64)
        self.addresses = self.poi_meta["address"].astype(str).to_numpy()
        self.pops = self.poi_meta["train_popularity"].to_numpy(dtype=np.int64)
        self.categories_lower = [str(c).lower() for c in self.categories_arr]
        self.search_text = [
            f"{n} {c}".lower() for n, c in zip(self.names, self.categories_arr)
        ]

        self.cat_counts = self._build_category_counts()
        self.models: dict[str, Any] = {
            "popularity": PopularityRecommender.load(self.models_dir / "popularity.pkl"),
            "cf_bpr": CFBPRRecommender.load(self.models_dir / "cf_bpr.pkl"),
            "content_based": ContentBasedRecommender.load(self.models_dir / "content_based.pkl"),
            "sequential": SequentialRecommender.load(
                self.models_dir / "sequential.pt", device="cpu"
            ),
            "sequential_cl": CL4SRecRecommender.load(
                self.models_dir / "sequential_cl.pt", device="cpu"
            ),
            "hybrid": HybridRecommender.load(
                self.models_dir / "hybrid.pkl", ball_tree=self.ball_tree
            ),
        }
        self.sample_trajectories = self._build_sample_trajectories(limit=20)
        if self.sample_trajectories:
            first = self.sample_trajectories[0]
            self.recommend(
                RecommendRequest(
                    user_id=first.user_id,
                    trajectory=[s.poi_id for s in first.stops],
                    model="hybrid",
                    top_k=5,
                )
            )

    def _build_category_counts(self) -> list[CategoryCount]:
        vc = self.poi_meta["category"].astype(str).value_counts()
        return [CategoryCount(category=str(cat), count=int(cnt)) for cat, cnt in vc.items()]

    def _poi_summary_from_pos(self, pos: int) -> POISummary:
        return POISummary(
            poi_id=int(self.venue_ids[pos]),
            name=str(self.names[pos]),
            category=str(self.categories_arr[pos]),
            lat=float(self.lats[pos]),
            lon=float(self.lons[pos]),
            address=str(self.addresses[pos]),
            train_popularity=int(self.pops[pos]),
        )

    def _build_sample_trajectories(self, limit: int = 20) -> list[SampleTrajectory]:
        seq_path = self.processed_dir / "test_sequences.parquet"
        if not seq_path.exists():
            return []
        df = load_parquet(seq_path)
        samples: list[SampleTrajectory] = []
        seen_users: set[int] = set()

        for row in df.itertuples(index=False):
            p_seq = [int(x) for x in row.poi_seq]
            if len(p_seq) < 3 or len(p_seq) > 8:
                continue
            if any(p < 2 for p in p_seq):
                continue
            uid = int(row.user_id)
            if uid in seen_users:
                continue
            last_lat = float(row.lat_seq[-2])
            last_lon = float(row.lon_seq[-2])
            if not (40.68 <= last_lat <= 40.82 and -74.02 <= last_lon <= -73.92):
                continue

            stops = [self._poi_summary_from_pos(p - 2) for p in p_seq[:-1]]
            gt = self._poi_summary_from_pos(p_seq[-1] - 2)
            samples.append(
                SampleTrajectory(
                    trail_id=str(row.trail_id),
                    user_id=uid,
                    stops=stops,
                    ground_truth=gt,
                )
            )
            seen_users.add(uid)
            if len(samples) >= limit:
                break
        return samples

    def search_pois(self, q: str = "", category: str = "", limit: int = 20) -> list[POISummary]:
        mask = np.ones(len(self.poi_indices), dtype=bool)
        if category.strip():
            cat_lower = category.strip().lower()
            mask &= np.array([c == cat_lower for c in self.categories_lower], dtype=bool)
        if q.strip():
            q_lower = q.strip().lower()
            mask &= np.array([q_lower in s for s in self.search_text], dtype=bool)

        matched = np.flatnonzero(mask)
        if len(matched) == 0:
            return []
        order = matched[np.argsort(-self.pops[matched])][:limit]
        return [self._poi_summary_from_pos(int(pos)) for pos in order]

    def get_nearby_pois(
        self, lat: float, lon: float, radius_km: float = 0.8, limit: int = 8
    ) -> list[NearbyPOI]:
        cand_pos, dists = query_radius_km(
            self.ball_tree, lat, lon, radius_km, sort_results=True
        )
        if len(cand_pos) < limit:
            cand_pos, dists = query_radius_km(
                self.ball_tree, lat, lon, max(radius_km, 3.0), sort_results=True
            )

        results: list[NearbyPOI] = []
        seen_ids: set[int] = set()
        for pos, dist_val in zip(cand_pos, dists):
            pid = int(self.venue_ids[pos])
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            results.append(
                NearbyPOI(
                    poi_id=pid,
                    name=str(self.names[pos]),
                    category=str(self.categories_arr[pos]),
                    lat=float(self.lats[pos]),
                    lon=float(self.lons[pos]),
                    address=str(self.addresses[pos]),
                    train_popularity=int(self.pops[pos]),
                    distance_km=round(float(dist_val), 3),
                )
            )
            if len(results) >= limit:
                break
        return results

    def get_categories(self) -> list[CategoryCount]:
        return self.cat_counts

    def get_sample_trajectories(self, limit: int = 20) -> list[SampleTrajectory]:
        return self.sample_trajectories[:limit]

    def _build_explanation(
        self,
        model_name: str,
        trail_cats: list[str],
        dist_km: float,
        breakdown: dict[str, float],
        is_cold_start: bool,
    ) -> str:
        parts: list[str] = []
        unique_cats: list[str] = []
        for c in trail_cats[-3:]:
            if c not in unique_cats:
                unique_cats.append(c)
        if unique_cats:
            parts.append(f"Matches your {' & '.join(unique_cats[-2:])} trail")
        elif is_cold_start:
            parts.append("Cold-start NYC recommendation")

        parts.append(f"{dist_km:.1f} km away")

        if model_name == "popularity":
            parts.append("Popular nearby venue")
        elif model_name == "cf_bpr":
            parts.append("Strong collaborative filtering affinity")
        elif model_name == "content_based":
            parts.append("High category & name similarity")
        else:
            best_comp = max(
                [
                    ("sequential", breakdown["sequential"]),
                    ("content", breakdown["content"]),
                    ("cf", breakdown["cf"]),
                    ("spatial", breakdown["spatial"]),
                ],
                key=lambda x: x[1],
            )[0]
            label_map = {
                "sequential": "Top sequential pick",
                "content": "Strong category match",
                "cf": "High user-venue affinity",
                "spatial": "Very close to current stop",
            }
            parts.append(label_map[best_comp])

        return " | ".join(parts)

    @torch.no_grad()
    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        t0 = time.perf_counter()
        model_name = req.model if req.model in self.models else "hybrid"
        model = self.models[model_name]

        user_idx = self.user2idx.get(req.user_id, 1) if req.user_id is not None else 1
        is_cold_start = user_idx < 2

        poi_seq_idx: list[int] = []
        cat_seq_idx: list[int] = []
        trail_cats: list[str] = []
        for vid in req.trajectory:
            p_idx = self.poi2idx.get(int(vid), 1)
            poi_seq_idx.append(p_idx)
            if p_idx >= 2:
                pos = p_idx - 2
                cat_seq_idx.append(int(self.cat_indices[pos]))
                trail_cats.append(str(self.categories_arr[pos]))
            else:
                cat_seq_idx.append(1)

        if req.current_lat is not None and req.current_lon is not None:
            anchor_lat = float(req.current_lat)
            anchor_lon = float(req.current_lon)
        else:
            known_idx = [p for p in poi_seq_idx if p >= 2]
            if known_idx:
                last_pos = known_idx[-1] - 2
                anchor_lat = float(self.lats[last_pos])
                anchor_lon = float(self.lons[last_pos])
            else:
                anchor_lat = MIDTOWN_LAT
                anchor_lon = MIDTOWN_LON

        hour = req.hour_of_day if req.hour_of_day is not None else 14
        hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
        hour_cos = math.cos(2.0 * math.pi * hour / 24.0)

        query = {
            "user_idx": user_idx,
            "poi_seq": poi_seq_idx,
            "cat_seq": cat_seq_idx,
            "last_lat": anchor_lat,
            "last_lon": anchor_lon,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": 0.0,
            "dow_cos": 1.0,
        }

        cand_pos, dists = query_radius_km(
            self.ball_tree, anchor_lat, anchor_lon, req.max_distance_km, sort_results=True
        )
        if len(cand_pos) == 0:
            # Expand gracefully to 15 km if user's tiny radius has zero venues.
            cand_pos, dists = query_radius_km(
                self.ball_tree, anchor_lat, anchor_lon, 15.0, sort_results=True
            )

        if req.categories:
            allowed = {c.strip().lower() for c in req.categories if c.strip()}
            if allowed:
                cat_mask = np.array(
                    [self.categories_arr[p].lower() in allowed for p in cand_pos], dtype=bool
                )
                cand_pos = cand_pos[cat_mask]
                dists = dists[cat_mask]

        # Exclude exact last visited POI from immediate next-stop recommendations.
        if poi_seq_idx and poi_seq_idx[-1] >= 2 and len(cand_pos) > 0:
            last_pos = poi_seq_idx[-1] - 2
            keep_mask = cand_pos != last_pos
            cand_pos = cand_pos[keep_mask]
            dists = dists[keep_mask]

        if len(cand_pos) > 800:
            top_local = np.argpartition(-self.pops[cand_pos], 799)[:800]
            cand_pos = cand_pos[top_local]
            dists = dists[top_local]

        if len(cand_pos) == 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return RecommendResponse(
                model_used=model_name,
                inference_time_ms=round(elapsed_ms, 2),
                is_cold_start=is_cold_start,
                recommendations=[],
            )

        cand_pois = self.poi_indices[cand_pos]
        hybrid_model: HybridRecommender = self.models["hybrid"]
        s_seq_raw = hybrid_model.seq_model.score_candidates(query, cand_pois)
        s_cf_raw = hybrid_model.cf_model.score_candidates(query, cand_pois)
        profile = hybrid_model.content_model._build_profile_row(poi_seq_idx)
        s_content_raw = (
            (profile @ hybrid_model.content_model.item_tfidf[cand_pos].T).toarray().ravel()
        )
        s_spatial = np.exp(-dists / max(hybrid_model.spatial_sigma_km, 1e-6)).astype(np.float32)

        s_seq_n = _minmax_norm(s_seq_raw)
        s_cf_n = _minmax_norm(s_cf_raw)
        s_content_n = _minmax_norm(s_content_raw)

        if model_name == "hybrid":
            w = hybrid_model.weights
            scores = (
                w["sequential"] * s_seq_n
                + w["content"] * s_content_n
                + w["cf"] * s_cf_n
                + w["spatial"] * s_spatial
            )
        elif model_name == "sequential_cl":
            scores = s_seq_n
        elif model_name == "cf_bpr":
            scores = s_cf_n
        else:
            scores = _minmax_norm(model.score_candidates(query, cand_pois))

        k_eff = min(req.top_k, len(scores))
        top_rel = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top_rel = top_rel[np.argsort(-scores[top_rel])]

        recs: list[POIRecommendation] = []
        for rank_idx, rel in enumerate(top_rel, start=1):
            pos = int(cand_pos[rel])
            breakdown = {
                "sequential": round(float(s_seq_n[rel]), 4),
                "content": round(float(s_content_n[rel]), 4),
                "cf": round(float(s_cf_n[rel]), 4),
                "spatial": round(float(s_spatial[rel]), 4),
            }
            dist_val = float(dists[rel])
            recs.append(
                POIRecommendation(
                    rank=rank_idx,
                    poi_id=int(self.venue_ids[pos]),
                    name=str(self.names[pos]),
                    category=str(self.categories_arr[pos]),
                    lat=float(self.lats[pos]),
                    lon=float(self.lons[pos]),
                    distance_km=round(dist_val, 3),
                    total_score=round(float(scores[rel]), 4),
                    score_breakdown=ScoreBreakdown(**breakdown),
                    explanation=self._build_explanation(
                        model_name, trail_cats, dist_val, breakdown, is_cold_start
                    ),
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RecommendResponse(
            model_used=model_name,
            inference_time_ms=round(elapsed_ms, 2),
            is_cold_start=is_cold_start,
            recommendations=recs,
        )
