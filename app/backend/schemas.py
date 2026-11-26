"""Pydantic v2 request and response schemas for the NYC Next-POI API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int | None = None
    trajectory: list[int] = Field(default_factory=list)
    current_lat: float | None = None
    current_lon: float | None = None
    hour_of_day: int | None = Field(default=None, ge=0, le=23)
    max_distance_km: float = Field(default=5.0, gt=0.0, le=50.0)
    categories: list[str] = Field(default_factory=list)
    model: str = Field(default="hybrid")
    top_k: int = Field(default=10, ge=1, le=30)


class ScoreBreakdown(BaseModel):
    sequential: float
    content: float
    cf: float
    spatial: float


class POIRecommendation(BaseModel):
    rank: int
    poi_id: int
    name: str
    category: str
    lat: float
    lon: float
    distance_km: float
    total_score: float
    score_breakdown: ScoreBreakdown
    explanation: str


class RecommendResponse(BaseModel):
    model_used: str
    inference_time_ms: float
    is_cold_start: bool
    recommendations: list[POIRecommendation]


class HealthResponse(BaseModel):
    status: str
    device: str
    total_pois_indexed: int
    loaded_models: list[str]


class POISummary(BaseModel):
    poi_id: int
    name: str
    category: str
    lat: float
    lon: float
    address: str
    train_popularity: int


class NearbyPOI(POISummary):
    distance_km: float


class CategoryCount(BaseModel):
    category: str
    count: int


class SampleTrajectory(BaseModel):
    trail_id: str
    user_id: int
    stops: list[POISummary]
    ground_truth: POISummary
