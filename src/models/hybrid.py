"""Context-Aware Hybrid Reranker + Knowledge-Based Constraints (CPU)."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.data.dataset import build_eval_queries
from src.evaluation.metrics import compute_ranking_metrics
from src.models.base import BaseRecommender
from src.models.cf_bpr import CFBPRRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.sequential import SequentialRecommender
from src.models.sequential_cl import CL4SRecRecommender
from src.utils.geo import build_ball_tree, haversine_km, query_radius_km
from src.utils.io import ensure_dir


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


class HybridRecommender(BaseRecommender):
    """Two-stage BallTree retrieve-and-rerank model fusing sequential, content, CF, and spatial signals."""

    name = "hybrid"

    def __init__(
        self,
        candidate_radius_km: float = 15.0,
        candidate_limit: int = 500,
        default_max_distance_km: float = 5.0,
        spatial_sigma_km: float = 2.0,
        weights: dict[str, float] | None = None,
        seq_model_path: str = "artifacts/models/sequential_cl.pt",
        cf_model_path: str = "artifacts/models/cf_bpr.pkl",
        content_model_path: str = "artifacts/models/content_based.pkl",
    ) -> None:
        self.candidate_radius_km = float(candidate_radius_km)
        self.candidate_limit = int(candidate_limit)
        self.default_max_distance_km = float(default_max_distance_km)
        self.spatial_sigma_km = float(spatial_sigma_km)
        self.weights = dict(
            weights
            or {"sequential": 0.45, "content": 0.20, "cf": 0.15, "spatial": 0.20}
        )
        self.seq_model_path = str(seq_model_path)
        self.cf_model_path = str(cf_model_path)
        self.content_model_path = str(content_model_path)

        self.seq_model: SequentialRecommender | None = None
        self.cf_model: CFBPRRecommender | None = None
        self.content_model: ContentBasedRecommender | None = None
        self.poi_indices: np.ndarray | None = None
        self.coords_deg: np.ndarray | None = None
        self.categories: np.ndarray | None = None
        self.global_pop: np.ndarray | None = None
        self.ball_tree: Any = None

    def _load_submodels(self) -> None:
        seq_p = Path(self.seq_model_path)
        if "sequential_cl" in seq_p.name and seq_p.exists():
            self.seq_model = CL4SRecRecommender.load(seq_p, device="cpu")
        else:
            fallback_p = Path("artifacts/models/sequential.pt")
            target_p = seq_p if seq_p.exists() else fallback_p
            self.seq_model = SequentialRecommender.load(target_p, device="cpu")

        self.cf_model = CFBPRRecommender.load(self.cf_model_path)
        self.content_model = ContentBasedRecommender.load(self.content_model_path)

    def _attach_metadata(self, poi_meta: Any, ball_tree: Any = None) -> None:
        self.poi_indices = poi_meta["poi_idx"].to_numpy(dtype=np.int64)
        self.coords_deg = poi_meta[["lat", "lon"]].to_numpy(dtype=np.float64)
        self.categories = poi_meta["category"].astype(str).to_numpy()
        raw_pop = poi_meta["train_popularity"].to_numpy(dtype=np.float64)
        log_pop = np.log1p(raw_pop)
        self.global_pop = (log_pop / max(float(log_pop.max()), 1e-12)).astype(np.float32)
        self.ball_tree = ball_tree or build_ball_tree(self.coords_deg)

    def _retrieve_candidate_positions(
        self,
        lat: float,
        lon: float,
        seq_top_pos: np.ndarray,
        cf_top_pos: np.ndarray,
        max_distance_km: float | None = None,
        categories: list[str] | None = None,
        strict_constraints: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return candidate 0-based positions and distances in km."""
        assert (
            self.ball_tree is not None
            and self.coords_deg is not None
            and self.global_pop is not None
            and self.categories is not None
        )
        radius = float(max_distance_km) if strict_constraints and max_distance_km else self.candidate_radius_km
        nearby_pos, _ = query_radius_km(self.ball_tree, lat, lon, radius, sort_results=False)
        if len(nearby_pos) > self.candidate_limit:
            top_local = np.argpartition(-self.global_pop[nearby_pos], self.candidate_limit - 1)[
                : self.candidate_limit
            ]
            nearby_pos = nearby_pos[top_local]

        if strict_constraints:
            cand_pos = nearby_pos
            if len(cand_pos) > 0 and len(seq_top_pos) > 0:
                extra = np.unique(np.concatenate([seq_top_pos, cf_top_pos]))
                extra_dists = haversine_km(
                    lat, lon, self.coords_deg[extra, 0], self.coords_deg[extra, 1]
                )
                extra_ok = extra[extra_dists <= radius]
                cand_pos = np.unique(np.concatenate([cand_pos, extra_ok]))
        else:
            cand_pos = np.unique(np.concatenate([nearby_pos, seq_top_pos, cf_top_pos]))

        if len(cand_pos) == 0:
            cand_pos = np.unique(np.concatenate([seq_top_pos, cf_top_pos]))

        if categories:
            allowed = {c.strip().lower() for c in categories if c.strip()}
            if allowed:
                cat_mask = np.array(
                    [self.categories[p].lower() in allowed for p in cand_pos], dtype=bool
                )
                if cat_mask.any():
                    cand_pos = cand_pos[cat_mask]
                elif strict_constraints:
                    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        dists = haversine_km(
            lat, lon, self.coords_deg[cand_pos, 0], self.coords_deg[cand_pos, 1]
        )
        return cand_pos.astype(np.int64), np.asarray(dists, dtype=np.float64)

    def _precompute_batch_signals(
        self, queries: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.seq_model is not None and self.cf_model is not None
        assert self.seq_model.net is not None and self.cf_model.valid_item_factors is not None

        seq_vecs = self.seq_model.encode_queries(queries, batch_size=256)
        valid_seq_ids = self.seq_model.poi_indices
        seq_item_emb = (
            self.seq_model.net.poi_emb.weight[valid_seq_ids].detach().cpu().numpy().astype(np.float32)
        )
        cf_vecs = self.cf_model._build_query_vectors(queries)
        return seq_vecs, seq_item_emb, cf_vecs, self.cf_model.valid_item_factors

    def fit(self, train_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        self._load_submodels()
        self._attach_metadata(train_data["poi_metadata"], train_data.get("ball_tree"))

        assert (
            self.seq_model is not None
            and self.cf_model is not None
            and self.content_model is not None
            and self.poi_indices is not None
        )

        val_df = train_data.get("val_sequences")
        best_weights = dict(self.weights)
        best_ndcg = -1.0

        if val_df is not None and len(val_df) > 0:
            tune_limit = min(2000, len(val_df))
            val_queries = build_eval_queries(val_df.iloc[:tune_limit])
            targets = np.array([q["target_poi_idx"] for q in val_queries], dtype=np.int64)

            seq_vecs, seq_item_emb, cf_vecs, cf_item_emb = self._precompute_batch_signals(
                val_queries
            )
            seq_top_idx, _ = self.seq_model.predict_topk(val_queries, k=100, batch_size=256)
            cf_top_idx, _ = self.cf_model.predict_topk(val_queries, k=50, batch_size=512)

            cached: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            for i, q in enumerate(val_queries):
                cand_pos, dists = self._retrieve_candidate_positions(
                    float(q["last_lat"]),
                    float(q["last_lon"]),
                    seq_top_idx[i] - 2,
                    cf_top_idx[i] - 2,
                )
                s_seq = _minmax_norm(seq_item_emb[cand_pos] @ seq_vecs[i])
                s_cf = _minmax_norm(cf_item_emb[cand_pos] @ cf_vecs[i])
                profile = self.content_model._build_profile_row(q.get("poi_seq", []))
                s_content = _minmax_norm(
                    (profile @ self.content_model.item_tfidf[cand_pos].T).toarray().ravel()
                )
                s_spatial = np.exp(-dists / max(self.spatial_sigma_km, 1e-6)).astype(np.float32)
                cached.append((self.poi_indices[cand_pos], s_seq, s_content, s_cf, s_spatial))

            weight_grid = [
                (0.55, 0.10, 0.15, 0.20),
                (0.50, 0.15, 0.15, 0.20),
                (0.45, 0.20, 0.15, 0.20),
                (0.45, 0.10, 0.20, 0.25),
                (0.60, 0.10, 0.10, 0.20),
                (0.50, 0.05, 0.20, 0.25),
                (0.65, 0.05, 0.15, 0.15),
                (0.40, 0.15, 0.20, 0.25),
            ]
            for ws, wc, wcf, wsp in weight_grid:
                preds = np.zeros((len(val_queries), 10), dtype=np.int64)
                for i, (c_poi, s_seq, s_content, s_cf, s_spatial) in enumerate(cached):
                    score = ws * s_seq + wc * s_content + wcf * s_cf + wsp * s_spatial
                    k_eff = min(10, len(score))
                    top_rel = np.argpartition(-score, k_eff - 1)[:k_eff]
                    top_rel = top_rel[np.argsort(-score[top_rel])]
                    preds[i, :k_eff] = c_poi[top_rel]
                m = compute_ranking_metrics(preds, targets, len(self.poi_indices))
                if m["NDCG@10"] > best_ndcg:
                    best_ndcg = m["NDCG@10"]
                    best_weights = {
                        "sequential": ws,
                        "content": wc,
                        "cf": wcf,
                        "spatial": wsp,
                    }

        self.weights = best_weights
        elapsed = time.perf_counter() - t0
        return {
            "model": self.name,
            "tuned_weights": self.weights,
            "val_tune_NDCG@10": round(best_ndcg, 5),
            "fit_time_s": round(elapsed, 2),
        }

    def score_candidates(
        self, query: dict[str, Any], candidate_poi_indices: np.ndarray
    ) -> np.ndarray:
        assert (
            self.seq_model is not None
            and self.cf_model is not None
            and self.content_model is not None
            and self.coords_deg is not None
        )
        cand = np.asarray(candidate_poi_indices, dtype=np.int64)
        positions = cand - 2

        s_seq = _minmax_norm(self.seq_model.score_candidates(query, cand))
        s_cf = _minmax_norm(self.cf_model.score_candidates(query, cand))
        profile = self.content_model._build_profile_row(query.get("poi_seq", []))
        s_content = _minmax_norm(
            (profile @ self.content_model.item_tfidf[positions].T).toarray().ravel()
        )
        dists = haversine_km(
            float(query["last_lat"]),
            float(query["last_lon"]),
            self.coords_deg[positions, 0],
            self.coords_deg[positions, 1],
        )
        s_spatial = np.exp(-dists / max(self.spatial_sigma_km, 1e-6)).astype(np.float32)

        w = self.weights
        return (
            w["sequential"] * s_seq
            + w["content"] * s_content
            + w["cf"] * s_cf
            + w["spatial"] * s_spatial
        ).astype(np.float32)

    def recommend_with_breakdown(
        self,
        query: dict[str, Any],
        k: int = 10,
        max_distance_km: float | None = None,
        categories: list[str] | None = None,
        strict_constraints: bool = True,
    ) -> list[dict[str, Any]]:
        """Single-query recommendation with per-component score breakdown for the API."""
        assert (
            self.seq_model is not None
            and self.cf_model is not None
            and self.content_model is not None
            and self.poi_indices is not None
        )
        seq_top_idx, _ = self.seq_model.predict_topk([query], k=100, batch_size=1)
        cf_top_idx, _ = self.cf_model.predict_topk([query], k=50, batch_size=1)

        cand_pos, dists = self._retrieve_candidate_positions(
            float(query["last_lat"]),
            float(query["last_lon"]),
            seq_top_idx[0] - 2,
            cf_top_idx[0] - 2,
            max_distance_km=max_distance_km,
            categories=categories,
            strict_constraints=strict_constraints,
        )
        if len(cand_pos) == 0:
            return []

        cand_pois = self.poi_indices[cand_pos]
        s_seq = _minmax_norm(self.seq_model.score_candidates(query, cand_pois))
        s_cf = _minmax_norm(self.cf_model.score_candidates(query, cand_pois))
        profile = self.content_model._build_profile_row(query.get("poi_seq", []))
        s_content = _minmax_norm(
            (profile @ self.content_model.item_tfidf[cand_pos].T).toarray().ravel()
        )
        s_spatial = np.exp(-dists / max(self.spatial_sigma_km, 1e-6)).astype(np.float32)

        w = self.weights
        total = (
            w["sequential"] * s_seq
            + w["content"] * s_content
            + w["cf"] * s_cf
            + w["spatial"] * s_spatial
        )
        k_eff = min(k, len(total))
        top_rel = np.argpartition(-total, k_eff - 1)[:k_eff]
        top_rel = top_rel[np.argsort(-total[top_rel])]

        results: list[dict[str, Any]] = []
        for idx in top_rel:
            results.append(
                {
                    "poi_idx": int(cand_pois[idx]),
                    "total_score": float(total[idx]),
                    "distance_km": float(dists[idx]),
                    "score_breakdown": {
                        "sequential": float(s_seq[idx]),
                        "content": float(s_content[idx]),
                        "cf": float(s_cf[idx]),
                        "spatial": float(s_spatial[idx]),
                    },
                }
            )
        return results

    def predict_topk(
        self, queries: list[dict[str, Any]], k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        assert (
            self.seq_model is not None
            and self.cf_model is not None
            and self.content_model is not None
            and self.poi_indices is not None
        )
        n = len(queries)
        out_idx = np.zeros((n, k), dtype=np.int64)
        out_scores = np.zeros((n, k), dtype=np.float32)
        if n == 0:
            return out_idx, out_scores

        seq_vecs, seq_item_emb, cf_vecs, cf_item_emb = self._precompute_batch_signals(queries)
        seq_top_idx, _ = self.seq_model.predict_topk(queries, k=100, batch_size=256)
        cf_top_idx, _ = self.cf_model.predict_topk(queries, k=50, batch_size=512)
        w = self.weights

        for i, q in enumerate(queries):
            cand_pos, dists = self._retrieve_candidate_positions(
                float(q["last_lat"]),
                float(q["last_lon"]),
                seq_top_idx[i] - 2,
                cf_top_idx[i] - 2,
            )
            s_seq = _minmax_norm(seq_item_emb[cand_pos] @ seq_vecs[i])
            s_cf = _minmax_norm(cf_item_emb[cand_pos] @ cf_vecs[i])
            profile = self.content_model._build_profile_row(q.get("poi_seq", []))
            s_content = _minmax_norm(
                (profile @ self.content_model.item_tfidf[cand_pos].T).toarray().ravel()
            )
            s_spatial = np.exp(-dists / max(self.spatial_sigma_km, 1e-6)).astype(np.float32)

            score = (
                w["sequential"] * s_seq
                + w["content"] * s_content
                + w["cf"] * s_cf
                + w["spatial"] * s_spatial
            )
            k_eff = min(k, len(score))
            top_rel = np.argpartition(-score, k_eff - 1)[:k_eff]
            top_rel = top_rel[np.argsort(-score[top_rel])]
            out_idx[i, :k_eff] = self.poi_indices[cand_pos[top_rel]]
            out_scores[i, :k_eff] = score[top_rel]

        return out_idx, out_scores

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        ensure_dir(out_path.parent)
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "candidate_radius_km": self.candidate_radius_km,
                    "candidate_limit": self.candidate_limit,
                    "default_max_distance_km": self.default_max_distance_km,
                    "spatial_sigma_km": self.spatial_sigma_km,
                    "weights": self.weights,
                    "seq_model_path": self.seq_model_path,
                    "cf_model_path": self.cf_model_path,
                    "content_model_path": self.content_model_path,
                    "poi_indices": self.poi_indices,
                    "coords_deg": self.coords_deg,
                    "categories": self.categories,
                    "global_pop": self.global_pop,
                },
                f,
            )
        return out_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        ball_tree: Any = None,
    ) -> "HybridRecommender":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            candidate_radius_km=state["candidate_radius_km"],
            candidate_limit=state["candidate_limit"],
            default_max_distance_km=state["default_max_distance_km"],
            spatial_sigma_km=state["spatial_sigma_km"],
            weights=state["weights"],
            seq_model_path=state["seq_model_path"],
            cf_model_path=state["cf_model_path"],
            content_model_path=state["content_model_path"],
        )
        obj.poi_indices = state["poi_indices"]
        obj.coords_deg = state["coords_deg"]
        obj.categories = state["categories"]
        obj.global_pop = state["global_pop"]
        obj.ball_tree = ball_tree or build_ball_tree(obj.coords_deg)
        obj._load_submodels()
        return obj
