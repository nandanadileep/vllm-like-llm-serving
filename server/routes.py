import json
import os
from uuid import uuid4

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from server.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionUsage,
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
        max_tokens=payload.max_tokens,
    )
    # Keep external response shape stable: one object with `response` per request.
    return GenerateResponse(response=result)

def _sse_generate(payload: GenerateRequest):
    print(f"[stream] request from {payload.user_id}")
    for token in scheduler.submit_request_stream_tokens(
        prompt=payload.prompt,
        user_id=payload.user_id,
        request_id=payload.request_id,
        max_tokens=payload.max_tokens,
    ):
        chunk = json.dumps({"token": token, "done": False})
        yield f"data: {chunk}\n\n"
    final = json.dumps({"token": "", "done": True})
    yield f"data: {final}\n\n"


@router.post("/generate/stream")
def generate_stream_post(payload: GenerateRequest) -> StreamingResponse:
    return StreamingResponse(
        _sse_generate(payload),
        media_type="text/event-stream",
    )


@router.get("/generate/stream")
def generate_stream_get(
    prompt: str = Query(..., min_length=1),
    user_id: str = Query(..., min_length=1),
    request_id: str | None = Query(
        default=None,
        min_length=1,
        description="Optional; generated if omitted",
    ),
) -> StreamingResponse:
    rid = request_id or str(uuid4())
    payload = GenerateRequest(
        prompt=prompt,
        user_id=user_id,
        request_id=rid,
    )
    return StreamingResponse(
        _sse_generate(payload),
        media_type="text/event-stream",
    )


def _chat_prompt(payload: ChatCompletionRequest) -> str:
    messages = [
        {"role": message.role, "content": message.content}
        for message in payload.messages
    ]
    return scheduler.format_chat_prompt(messages)


def _sse_chat_completion(payload: ChatCompletionRequest, request_id: str):
    prompt = _chat_prompt(payload)
    for token in scheduler.submit_request_stream_tokens(
        prompt=prompt,
        user_id="openai-chat",
        request_id=request_id,
        max_tokens=payload.max_tokens,
    ):
        chunk = json.dumps({"choices": [{"delta": {"content": token}}]})
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions", response_model=None)
def chat_completions(
    payload: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    request_id = f"chatcmpl-{uuid4()}"
    if payload.stream:
        return StreamingResponse(
            _sse_chat_completion(payload, request_id),
            media_type="text/event-stream",
        )

    prompt = _chat_prompt(payload)
    content = scheduler.submit_request(
        prompt=prompt,
        user_id="openai-chat",
        request_id=request_id,
        max_tokens=payload.max_tokens,
    )
    return ChatCompletionResponse(
        id=request_id,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionResponseMessage(content=content),
            )
        ],
        usage=ChatCompletionUsage(
            completion_tokens=scheduler.count_completion_tokens(content),
        ),
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(**scheduler.get_metrics())
