"""Abstract base interface for all NYC Next-POI recommender models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseRecommender(ABC):
    """Common interface shared by popularity, CF, content, sequential, and hybrid models."""

    name: str = "base"

    @abstractmethod
    def fit(self, train_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Train the model on training data and return summary metrics/timings."""

    @abstractmethod
    def predict_topk(
        self, queries: list[dict[str, Any]], k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (top_k_poi_indices, top_k_scores) with shapes (N, k)."""

    @abstractmethod
    def score_candidates(
        self, query: dict[str, Any], candidate_poi_indices: np.ndarray
    ) -> np.ndarray:
        """Score a 1D array of valid candidate poi_idx values for a single query."""

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Persist trained model state to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseRecommender":
        """Restore trained model state from disk."""
