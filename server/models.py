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
    response: str
