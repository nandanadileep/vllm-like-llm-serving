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
    max_tokens: int = Field(default=200, ge=1, le=2048, description="Maximum tokens to generate")
    streaming: bool = Field(
        default=False,
        description="Hint for clients; use POST /generate/stream for SSE (this field is ignored by POST /generate)",
    )


class GenerateResponse(BaseModel):
    # Keep external API shape stable as a single object per request.
    # Internal batching should still map each request to one response payload.
    response: str = Field(
        ...,
        description="Single response string for one request (stable external shape)",
        examples=["stub response"],
    )


class ChatCompletionMessage(BaseModel):
    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., min_length=1)
    messages: list[ChatCompletionMessage] = Field(..., min_length=1)
    max_tokens: int = Field(default=200, ge=1, le=2048)
    stream: bool = False


class ChatCompletionResponseMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionResponseMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    completion_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class MetricsResponse(BaseModel):
    total_batches: int
    total_processed: int = 0
    avg_wait_time: float
    avg_ttft: float = 0.0
    max_queue_length: int
    total_preemptions: int = 0
    total_prefill_chunks: int = 0
    total_blocks: int = 0
    free_blocks: int = 0
    used_physical_blocks: int = 0
    peak_used_physical_blocks: int = 0
    active_sequences: int = 0
    committed_kv_tokens: int = 0
    reserved_token_slots: int = 0
    unused_reserved_slots: int = 0
    pool_slot_utilization: float = 0.0
    total_tokens_generated: int = 0
    memory_mb: float = 0.0
    prefix_cache_hits: int = 0
    prefix_cache_shared_hits: int = 0
    prefix_cache_misses: int = 0
    prefix_cache_entries: int = 0
    prefix_cache_matched_prefix_tokens: int = 0
    global_batch_steps: int = 0
    speculative_generations: int = 0
