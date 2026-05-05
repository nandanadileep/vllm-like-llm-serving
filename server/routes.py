from fastapi import APIRouter

from server.models import GenerateRequest, GenerateResponse, HealthResponse

router = APIRouter(tags=["api"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    """Synchronous stub endpoint; async scheduling will be introduced later."""
    print(f"Received request from {payload.user_id}")
    # Stub: wire payload.prompt/user_id/request_id to batching + model logic later.
    return GenerateResponse(response="stub response")
