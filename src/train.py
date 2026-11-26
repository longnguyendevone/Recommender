"""Unified training CLI for all NYC Next-POI recommender models."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

from src.models.cf_bpr import CFBPRRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.models.popularity import PopularityRecommender
from src.models.sequential import SequentialRecommender
from src.models.sequential_cl import CL4SRecRecommender
from src.utils.io import load_json, load_parquet, load_yaml, save_json
from src.utils.seed import set_seed

DEFAULT_CONFIGS = {
    "popularity": "configs/baselines.yaml",
    "cf_bpr": "configs/baselines.yaml",
    "content_based": "configs/baselines.yaml",
    "sequential": "configs/sequential.yaml",
    "sequential_cl": "configs/sequential.yaml",
    "hybrid": "configs/hybrid.yaml",
}


def _load_train_bundle(processed_dir: Path) -> dict[str, Any]:
    mappings = load_json(processed_dir / "mappings.json")
    poi_meta = load_parquet(processed_dir / "poi_metadata.parquet")
    train_checkins = load_parquet(processed_dir / "train_checkins.parquet")
    val_sequences = load_parquet(processed_dir / "validation_sequences.parquet")
    with open(processed_dir / "poi_balltree.pkl", "rb") as f:
        ball_tree = pickle.load(f)["tree"]

    return {
        "poi_metadata": poi_meta,
        "train_checkins": train_checkins,
        "train_sequences_path": processed_dir / "train_sequences.parquet",
        "val_sequences": val_sequences,
        "ball_tree": ball_tree,
        "num_users": mappings["num_users"],
        "num_pois": mappings["num_pois"],
        "num_cats": mappings["num_cats"],
    }


def train_model(model_name: str, config_path: str | None = None) -> dict[str, Any]:
    cfg_path = config_path or DEFAULT_CONFIGS[model_name]
    cfg = load_yaml(cfg_path)
    set_seed(int(cfg.get("seed", 42)))

    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts/models"))
    train_data = _load_train_bundle(processed_dir)

    if model_name == "popularity":
        model = PopularityRecommender(**cfg["popularity"])
        ckpt_path = artifacts_dir / "popularity.pkl"
    elif model_name == "cf_bpr":
        model = CFBPRRecommender(**cfg["cf_bpr"], random_state=int(cfg.get("seed", 42)))
        ckpt_path = artifacts_dir / "cf_bpr.pkl"
    elif model_name == "content_based":
        model = ContentBasedRecommender(**cfg["content_based"])
        ckpt_path = artifacts_dir / "content_based.pkl"
    elif model_name == "sequential":
        model = SequentialRecommender(
            **cfg["model"],
            **cfg["train"],
            device=cfg.get("device", "cuda"),
            use_amp=cfg.get("use_amp", True),
            batch_size=cfg.get("batch_size", 128),
            grad_accum_steps=cfg.get("grad_accum_steps", 2),
            seed=int(cfg.get("seed", 42)),
        )
        ckpt_path = artifacts_dir / "sequential.pt"
    elif model_name == "sequential_cl":
        model = CL4SRecRecommender(
            **cfg["contrastive"],
            **cfg["model"],
            **cfg["train"],
            device=cfg.get("device", "cuda"),
            use_amp=cfg.get("use_amp", True),
            batch_size=cfg.get("batch_size", 128),
            grad_accum_steps=cfg.get("grad_accum_steps", 2),
            seed=int(cfg.get("seed", 42)),
        )
        ckpt_path = artifacts_dir / "sequential_cl.pt"
    elif model_name == "hybrid":
        model = HybridRecommender(
            candidate_radius_km=cfg.get("candidate_radius_km", 15.0),
            candidate_limit=cfg.get("candidate_limit", 500),
            default_max_distance_km=cfg.get("default_max_distance_km", 5.0),
            spatial_sigma_km=cfg.get("spatial_sigma_km", 2.0),
            weights=cfg.get("weights"),
        )
        ckpt_path = artifacts_dir / "hybrid.pkl"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    summary = model.fit(train_data)
    saved_to = model.save(ckpt_path)
    summary["checkpoint"] = str(saved_to)
    save_json(summary, artifacts_dir / f"{model_name}_train_summary.json")
    print(f"Saved {model_name} -> {saved_to} | summary={summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Next-POI recommender model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["popularity", "cf_bpr", "content_based", "sequential", "sequential_cl", "hybrid", "all"],
    )
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.model == "all":
        order = ["popularity", "cf_bpr", "content_based", "sequential", "sequential_cl", "hybrid"]
        for m in order:
            train_model(m, args.config)
    else:
        train_model(args.model, args.config)


if __name__ == "__main__":
    main()
