"""Collaborative Filtering baseline via BPR-MF (implicit CPU)."""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from implicit.cpu.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix

from src.models.base import BaseRecommender
from src.utils.io import ensure_dir


class CFBPRRecommender(BaseRecommender):
    """Bayesian Personalized Ranking Matrix Factorization trained on CPU."""

    name = "cf_bpr"

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        iterations: int = 50,
        num_threads: int = 4,
        random_state: int = 42,
    ) -> None:
        self.factors = int(factors)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.num_threads = int(num_threads)
        self.random_state = int(random_state)

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.valid_item_factors: np.ndarray | None = None
        self.poi_indices: np.ndarray | None = None
        self.num_users: int = 0
        self.num_pois: int = 0
        self.default_user_vec: np.ndarray | None = None

    def fit(self, train_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        checkins = train_data["train_checkins"]
        poi_meta = train_data["poi_metadata"]
        self.num_users = int(train_data["num_users"])
        self.num_pois = int(train_data["num_pois"])
        self.poi_indices = poi_meta["poi_idx"].to_numpy(dtype=np.int64)

        valid = checkins[(checkins["user_idx"] >= 2) & (checkins["poi_idx"] >= 2)]
        rows = valid["user_idx"].to_numpy(dtype=np.int64)
        cols = valid["poi_idx"].to_numpy(dtype=np.int64)
        vals = np.ones(len(rows), dtype=np.float32)

        user_items = csr_matrix((vals, (rows, cols)), shape=(self.num_users, self.num_pois))
        user_items.sum_duplicates()
        user_items.data[:] = 1.0

        bpr = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            num_threads=self.num_threads,
            random_state=self.random_state,
            verify_negative_samples=True,
        )
        bpr.fit(user_items, show_progress=False)

        self.user_factors = np.asarray(bpr.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(bpr.item_factors, dtype=np.float32)
        self.valid_item_factors = self.item_factors[self.poi_indices]

        if self.num_users > 2:
            self.default_user_vec = self.user_factors[2:].mean(axis=0).astype(np.float32)
        else:
            self.default_user_vec = np.zeros(self.user_factors.shape[1], dtype=np.float32)

        elapsed = time.perf_counter() - t0
        return {
            "model": self.name,
            "fit_time_s": round(elapsed, 4),
            "nnz_interactions": int(user_items.nnz),
        }

    def _build_query_vectors(self, queries: list[dict[str, Any]]) -> np.ndarray:
        assert (
            self.user_factors is not None
            and self.item_factors is not None
            and self.default_user_vec is not None
        )
        dim = self.user_factors.shape[1]
        vecs = np.empty((len(queries), dim), dtype=np.float32)

        for i, q in enumerate(queries):
            uid = int(q.get("user_idx", 1))
            if 2 <= uid < self.num_users:
                vecs[i] = self.user_factors[uid]
            else:
                known_pois = [int(p) for p in q.get("poi_seq", []) if 2 <= int(p) < self.num_pois]
                if known_pois:
                    vecs[i] = self.item_factors[known_pois].mean(axis=0)
                else:
                    vecs[i] = self.default_user_vec
        return vecs

    def score_candidates(
        self, query: dict[str, Any], candidate_poi_indices: np.ndarray
    ) -> np.ndarray:
        assert self.item_factors is not None
        cand = np.asarray(candidate_poi_indices, dtype=np.int64)
        u_vec = self._build_query_vectors([query])[0]
        return (self.item_factors[cand] @ u_vec).astype(np.float32)

    def predict_topk(
        self, queries: list[dict[str, Any]], k: int = 10, batch_size: int = 512
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.valid_item_factors is not None and self.poi_indices is not None
        n = len(queries)
        out_idx = np.zeros((n, k), dtype=np.int64)
        out_scores = np.zeros((n, k), dtype=np.float32)
        if n == 0:
            return out_idx, out_scores

        q_vecs = self._build_query_vectors(queries)
        item_t = self.valid_item_factors.T
        k_eff = min(k, len(self.poi_indices))

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            scores = q_vecs[start:end] @ item_t
            top_rel = np.argpartition(-scores, k_eff - 1, axis=1)[:, :k_eff]
            row_idx = np.arange(end - start)[:, None]
            top_scores = scores[row_idx, top_rel]
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
                    "factors": self.factors,
                    "learning_rate": self.learning_rate,
                    "regularization": self.regularization,
                    "iterations": self.iterations,
                    "num_threads": self.num_threads,
                    "random_state": self.random_state,
                    "user_factors": self.user_factors,
                    "item_factors": self.item_factors,
                    "poi_indices": self.poi_indices,
                    "num_users": self.num_users,
                    "num_pois": self.num_pois,
                    "default_user_vec": self.default_user_vec,
                },
                f,
            )
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "CFBPRRecommender":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            factors=state["factors"],
            learning_rate=state["learning_rate"],
            regularization=state["regularization"],
            iterations=state["iterations"],
            num_threads=state["num_threads"],
            random_state=state["random_state"],
        )
        obj.user_factors = state["user_factors"]
        obj.item_factors = state["item_factors"]
        obj.poi_indices = state["poi_indices"]
        obj.valid_item_factors = obj.item_factors[obj.poi_indices]
        obj.num_users = state["num_users"]
        obj.num_pois = state["num_pois"]
        obj.default_user_vec = state["default_user_vec"]
        return obj
