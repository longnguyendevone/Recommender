"""Vectorized offline ranking metrics for Next-POI recommendation."""

from __future__ import annotations

import numpy as np


def compute_ranking_metrics(
    pred_indices: np.ndarray,
    targets: np.ndarray,
    num_catalog_items: int,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall@K, HitRate@K, NDCG@K, MRR@10, Coverage@10, and in-vocab breakdown."""
    preds = np.asarray(pred_indices, dtype=np.int64)
    tgts = np.asarray(targets, dtype=np.int64)
    if preds.ndim != 2:
        raise ValueError(f"Expected 2D pred_indices, got shape {preds.shape}")
    if len(preds) != len(tgts):
        raise ValueError("pred_indices and targets must have matching batch length")

    n = len(tgts)
    max_k = preds.shape[1]
    metrics: dict[str, float] = {}

    for k in ks:
        k_eff = min(k, max_k)
        hits_at_k = preds[:, :k_eff] == tgts[:, None]
        has_hit = hits_at_k.any(axis=1)
        recall_k = float(has_hit.mean()) if n > 0 else 0.0
        metrics[f"Recall@{k}"] = recall_k
        metrics[f"HitRate@{k}"] = recall_k

        # Single ground-truth target => IDCG = 1.0, DCG = 1 / log2(rank + 2)
        pos = np.argmax(hits_at_k, axis=1)
        dcg = np.where(has_hit, 1.0 / np.log2(pos + 2.0), 0.0)
        metrics[f"NDCG@{k}"] = float(dcg.mean()) if n > 0 else 0.0

    k_mrr = min(10, max_k)
    hits_10 = preds[:, :k_mrr] == tgts[:, None]
    has_hit_10 = hits_10.any(axis=1)
    pos_10 = np.argmax(hits_10, axis=1)
    rr_10 = np.where(has_hit_10, 1.0 / (pos_10 + 1.0), 0.0)
    metrics["MRR@10"] = float(rr_10.mean()) if n > 0 else 0.0

    unique_recs = np.unique(preds[:, :k_mrr])
    metrics["Coverage@10"] = (
        float(len(unique_recs) / num_catalog_items) if num_catalog_items > 0 else 0.0
    )

    in_vocab_mask = tgts >= 2
    oov_rate = float((~in_vocab_mask).mean()) if n > 0 else 0.0
    metrics["oov_target_rate"] = oov_rate
    if in_vocab_mask.any():
        metrics["in_vocab_Recall@10"] = float(has_hit_10[in_vocab_mask].mean())
        dcg_10 = np.where(has_hit_10, 1.0 / np.log2(pos_10 + 2.0), 0.0)
        metrics["in_vocab_NDCG@10"] = float(dcg_10[in_vocab_mask].mean())
    else:
        metrics["in_vocab_Recall@10"] = 0.0
        metrics["in_vocab_NDCG@10"] = 0.0

    return metrics
