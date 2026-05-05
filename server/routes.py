import os

from fastapi import APIRouter

from server.models import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
)
from server.scheduler import Scheduler

router = APIRouter(tags=["api"])


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


scheduler = Scheduler(
    batch_size=_env_int("BATCH_SIZE", 4),
    batch_timeout=_env_float("BATCH_TIMEOUT", 0.05),
)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    """Synchronous stub endpoint; async scheduling will be introduced later."""
    print(f"Received request from {payload.user_id}")
    result = scheduler.submit_request(
        prompt=payload.prompt,
        user_id=payload.user_id,
        request_id=payload.request_id,
    )
    # Keep external response shape stable: one object with `response` per request.
    return GenerateResponse(response=result)


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(**scheduler.get_metrics())
