"""FastAPI application serving CPU-only NYC Next-POI recommendations."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.backend.recommender_service import RecommenderService
from app.backend.schemas import (
    CategoryCount,
    HealthResponse,
    POISummary,
    RecommendRequest,
    RecommendResponse,
    SampleTrajectory,
)

service_holder: dict[str, RecommenderService] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service_holder["service"] = RecommenderService()
    yield
    service_holder.clear()


app = FastAPI(
    title="NYC Context-Aware Next-POI Recommender API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_service() -> RecommenderService:
    return service_holder["service"]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    svc = _get_service()
    return HealthResponse(
        status="ok",
        device=svc.device,
        total_pois_indexed=len(svc.poi_indices),
        loaded_models=sorted(svc.models.keys()),
    )


@app.get("/pois/search", response_model=list[POISummary])
def search_pois(
    q: str = Query(default=""),
    category: str = Query(default=""),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[POISummary]:
    return _get_service().search_pois(q=q, category=category, limit=limit)


@app.get("/pois/categories", response_model=list[CategoryCount])
def get_categories() -> list[CategoryCount]:
    return _get_service().get_categories()


@app.get("/users/sample", response_model=list[SampleTrajectory])
def get_sample_users(limit: int = Query(default=20, ge=1, le=50)) -> list[SampleTrajectory]:
    return _get_service().get_sample_trajectories(limit=limit)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    return _get_service().recommend(req)
