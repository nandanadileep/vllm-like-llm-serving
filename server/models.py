from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt text")
    user_id: str = Field(..., min_length=1, description="Requesting user identifier")


class GenerateResponse(BaseModel):
    response: str
