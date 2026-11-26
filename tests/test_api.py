"""FastAPI endpoint and CPU latency tests for NYC Next-POI Recommender."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.backend.main import app


def test_health_and_catalog_endpoints() -> None:
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert body["device"] == "cpu"
        assert body["total_pois_indexed"] == 39113
        assert set(body["loaded_models"]) == {
            "cf_bpr",
            "content_based",
            "hybrid",
            "popularity",
            "sequential",
            "sequential_cl",
        }

        search_res = client.get("/pois/search?q=coffee&limit=5")
        assert search_res.status_code == 200
        assert len(search_res.json()) > 0

        cats_res = client.get("/pois/categories")
        assert cats_res.status_code == 200
        assert len(cats_res.json()) > 100

        samples_res = client.get("/users/sample?limit=20")
        assert samples_res.status_code == 200
        samples = samples_res.json()
        assert len(samples) == 20
        assert len(samples[0]["stops"]) >= 2


def test_recommend_known_and_cold_start_under_150ms() -> None:
    with TestClient(app) as client:
        samples = client.get("/users/sample?limit=5").json()
        sample = samples[0]
        known_traj = [s["poi_id"] for s in sample["stops"]]

        # Warm up JIT/BLAS caches once, then assert latency across all 6 models.
        for model_name in [
            "popularity",
            "cf_bpr",
            "content_based",
            "sequential",
            "sequential_cl",
            "hybrid",
        ]:
            t0 = time.perf_counter()
            res = client.post(
                "/recommend",
                json={
                    "user_id": sample["user_id"],
                    "trajectory": known_traj,
                    "max_distance_km": 5.0,
                    "model": model_name,
                    "top_k": 10,
                },
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            assert res.status_code == 200
            data = res.json()
            assert data["model_used"] == model_name
            assert data["is_cold_start"] is False
            assert len(data["recommendations"]) > 0
            assert data["inference_time_ms"] < 150.0
            assert wall_ms < 250.0

        # Cold-start tourist with explicit category filter & coordinates.
        cold_res = client.post(
            "/recommend",
            json={
                "user_id": None,
                "trajectory": known_traj[:2],
                "current_lat": 40.7580,
                "current_lon": -73.9855,
                "hour_of_day": 18,
                "max_distance_km": 3.0,
                "categories": ["Coffee Shop", "Italian Restaurant"],
                "model": "hybrid",
                "top_k": 5,
            },
        )
        assert cold_res.status_code == 200
        cold_data = cold_res.json()
        assert cold_data["is_cold_start"] is True
        assert cold_data["inference_time_ms"] < 150.0
        for rec in cold_data["recommendations"]:
            assert rec["distance_km"] <= 3.01
            assert rec["category"] in {"Coffee Shop", "Italian Restaurant"}
