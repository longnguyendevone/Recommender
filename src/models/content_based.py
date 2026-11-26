"""Content-Based POI recommender using Category/Name TF-IDF + Spatial Proximity (CPU)."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.base import BaseRecommender
from src.utils.geo import haversine_km
from src.utils.io import ensure_dir


class ContentBasedRecommender(BaseRecommender):
    """Matches recency-weighted TF-IDF POI profiles with exponential spatial proximity."""

    name = "content_based"

    def __init__(
        self,
        spatial_sigma_km: float = 2.0,
        category_weight: float = 0.6,
        spatial_weight: float = 0.4,
        max_features: int = 4096,
        recency_decay: float = 0.7,
    ) -> None:
        self.spatial_sigma_km = float(spatial_sigma_km)
        self.category_weight = float(category_weight)
        self.spatial_weight = float(spatial_weight)
        self.max_features = int(max_features)
        self.recency_decay = float(recency_decay)

        self.vectorizer: TfidfVectorizer | None = None
        self.item_tfidf: csr_matrix | None = None
        self.poi_indices: np.ndarray | None = None
        self.coords_deg: np.ndarray | None = None
        self.num_pois: int = 0

    def fit(self, train_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        poi_meta = train_data["poi_metadata"]
        self.num_pois = int(train_data["num_pois"])
        self.poi_indices = poi_meta["poi_idx"].to_numpy(dtype=np.int64)
        self.coords_deg = poi_meta[["lat", "lon"]].to_numpy(dtype=np.float64)

        texts = (
            poi_meta["category"].astype(str) + " " + poi_meta["name"].astype(str)
        ).tolist()
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.item_tfidf = self.vectorizer.fit_transform(texts).tocsr()

        elapsed = time.perf_counter() - t0
        return {
            "model": self.name,
            "fit_time_s": round(elapsed, 4),
            "vocab_size": int(self.item_tfidf.shape[1]),
        }

    def _build_profile_row(self, poi_seq: list[int]) -> csr_matrix:
        assert self.item_tfidf is not None
        known = [int(p) for p in poi_seq if 2 <= int(p) < self.num_pois]
        if not known:
            return csr_matrix((1, self.item_tfidf.shape[1]), dtype=np.float32)

        L = len(known)
        weights = np.array(
            [self.recency_decay ** (L - 1 - i) for i in range(L)], dtype=np.float32
        )
        weights /= weights.sum()
        rows = self.item_tfidf[np.asarray(known, dtype=np.int64) - 2]
        profile = rows.multiply(weights[:, None]).sum(axis=0)
        profile_csr = csr_matrix(profile, dtype=np.float32)
        norm = float(np.sqrt(profile_csr.multiply(profile_csr).sum()))
        if norm > 1e-8:
            profile_csr = profile_csr / norm
        return profile_csr

    def score_candidates(
        self, query: dict[str, Any], candidate_poi_indices: np.ndarray
    ) -> np.ndarray:
        assert self.item_tfidf is not None and self.coords_deg is not None
        cand = np.asarray(candidate_poi_indices, dtype=np.int64)
        positions = cand - 2

        profile = self._build_profile_row(query.get("poi_seq", []))
        content_sim = (profile @ self.item_tfidf[positions].T).toarray().ravel()

        dists = haversine_km(
            float(query["last_lat"]),
            float(query["last_lon"]),
            self.coords_deg[positions, 0],
            self.coords_deg[positions, 1],
        )
        spatial_prox = np.exp(-dists / max(self.spatial_sigma_km, 1e-6))
        return (self.category_weight * content_sim + self.spatial_weight * spatial_prox).astype(
            np.float32
        )

    def predict_topk(
        self, queries: list[dict[str, Any]], k: int = 10, batch_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        assert (
            self.item_tfidf is not None
            and self.coords_deg is not None
            and self.poi_indices is not None
        )
        n = len(queries)
        out_idx = np.zeros((n, k), dtype=np.int64)
        out_scores = np.zeros((n, k), dtype=np.float32)
        if n == 0:
            return out_idx, out_scores

        item_t = self.item_tfidf.T
        all_lats = self.coords_deg[:, 0][None, :]
        all_lons = self.coords_deg[:, 1][None, :]
        k_eff = min(k, len(self.poi_indices))

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_q = queries[start:end]
            profiles = vstack([self._build_profile_row(q.get("poi_seq", [])) for q in batch_q])
            content_scores = (profiles @ item_t).toarray().astype(np.float32)

            b_lats = np.array([float(q["last_lat"]) for q in batch_q], dtype=np.float64)[:, None]
            b_lons = np.array([float(q["last_lon"]) for q in batch_q], dtype=np.float64)[:, None]
            dists = haversine_km(b_lats, b_lons, all_lats, all_lons)
            spatial_scores = np.exp(-dists / max(self.spatial_sigma_km, 1e-6)).astype(np.float32)

            total = self.category_weight * content_scores + self.spatial_weight * spatial_scores
            top_rel = np.argpartition(-total, k_eff - 1, axis=1)[:, :k_eff]
            row_idx = np.arange(end - start)[:, None]
            top_scores = total[row_idx, top_rel]
            order = np.argsort(-top_scores, axis=1)
            sorted_pos = top_rel[row_idx, order]

            out_idx[start:end, :k_eff] = self.poi_indices[sorted_pos]
            out_scores[start:end, :k_eff] = top_scores[row_idx, order]

        return out_idx, out_scores

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        ensure_dir(out_path.parent)
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "spatial_sigma_km": self.spatial_sigma_km,
                    "category_weight": self.category_weight,
                    "spatial_weight": self.spatial_weight,
                    "max_features": self.max_features,
                    "recency_decay": self.recency_decay,
                    "vectorizer": self.vectorizer,
                    "item_tfidf": self.item_tfidf,
                    "poi_indices": self.poi_indices,
                    "coords_deg": self.coords_deg,
                    "num_pois": self.num_pois,
                },
                f,
            )
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "ContentBasedRecommender":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            spatial_sigma_km=state["spatial_sigma_km"],
            category_weight=state["category_weight"],
            spatial_weight=state["spatial_weight"],
            max_features=state["max_features"],
            recency_decay=state["recency_decay"],
        )
        obj.vectorizer = state["vectorizer"]
        obj.item_tfidf = state["item_tfidf"]
        obj.poi_indices = state["poi_indices"]
        obj.coords_deg = state["coords_deg"]
        obj.num_pois = state["num_pois"]
        return obj
