from uuid import uuid4

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt text")
    user_id: str = Field(..., min_length=1, description="Requesting user identifier")
    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description="Client request identifier used for response correlation",
    )


class GenerateResponse(BaseModel):
    # Keep external API shape stable as a single object per request.
    # Internal batching should still map each request to one response payload.
    response: str = Field(
        ...,
        description="Single response string for one request (stable external shape)",
        examples=["stub response"],
    )


class MetricsResponse(BaseModel):
    total_batches: int
    avg_wait_time: float
    max_queue_length: int
    kv_total_blocks: int = 0
    kv_free_blocks: int = 0
    kv_active_sequences: int = 0
