from fastapi import APIRouter

from server.models import GenerateRequest, GenerateResponse, HealthResponse

router = APIRouter(tags=["api"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    """Synchronous stub endpoint; async scheduling will be introduced later."""
    # Stub: wire payload.prompt / payload.user_id to the model when ready.
    return GenerateResponse(response="stub response")
