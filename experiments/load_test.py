import json
import os
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE_URL = "http://127.0.0.1:8000"
GENERATE_URL = f"{BASE_URL}/generate"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"
HEALTH_URL = f"{BASE_URL}/health"
METRICS_URL = f"{BASE_URL}/metrics"
CONCURRENCY_LEVELS = [1, 2, 4, 8]
REQUESTS_PER_LEVEL = 10
REQUEST_TIMEOUT_SECONDS = 300
MAX_TOKENS = 50
OUTPUT_QUALITY_MAX_TOKENS = 120
LOCAL_CHAT_MODEL = os.getenv("MODEL_PATH", "mlx-community/Llama-3.2-1B-Instruct-4bit")
RESULTS_JSON = os.getenv("LOAD_TEST_RESULTS_JSON", "experiments/latest_results.json")

SHORT_PROMPTS = [
    "What is a transformer?",
    "Define tokenization.",
    "What does RLHF stand for?",
    "What is a context window?",
    "What is quantization in ML?",
]

MEDIUM_PROMPTS = [
    "Explain how attention mechanisms work in neural networks.",
    "What is the difference between supervised and unsupervised learning?",
    "Describe the encoder-decoder architecture used in seq2seq models.",
    "How does continuous batching improve LLM serving throughput?",
    "What is PagedAttention and why was it introduced in vLLM?",
    "Explain the difference between the prefill and decode phases in LLM inference.",
    "What is speculative decoding and how does it reduce latency?",
    "Describe how key-value caching works during autoregressive generation.",
    "What is the role of positional encoding in transformer models?",
    "How does beam search differ from greedy decoding?",
]

LONG_PROMPTS = [
    (
        "You are an expert in distributed systems. Explain in detail how a "
        "production LLM serving system handles thousands of concurrent requests, "
        "covering request queuing, memory management, batching strategies, and "
        "how it avoids out-of-memory errors when many long sequences arrive at once."
    ),
    (
        "Compare and contrast vLLM, TensorRT-LLM, and llama.cpp as inference "
        "engines. Cover: hardware requirements, memory management approach, "
        "batching strategy, streaming support, ease of deployment, and which use "
        "cases each is best suited for."
    ),
    (
        "Walk through the full lifecycle of a single inference request in a "
        "system like vLLM: from the moment the HTTP request arrives, through "
        "scheduling, prefill, decode, KV cache management, to streaming the "
        "response back to the client. Be specific about what happens at each step."
    ),
]

SHARED_PREFIX = "You are a concise technical assistant. Answer in 2-3 sentences only.\n\n"
PREFIX_PROMPTS = [
    SHARED_PREFIX + "What is gradient descent?",
    SHARED_PREFIX + "What is layer normalization?",
    SHARED_PREFIX + "What is the softmax function?",
    SHARED_PREFIX + "What is a residual connection?",
    SHARED_PREFIX + "What is temperature sampling?",
]

PROMPTS = SHORT_PROMPTS + MEDIUM_PROMPTS + LONG_PROMPTS + PREFIX_PROMPTS
OUTPUT_QUALITY_PROMPTS = MEDIUM_PROMPTS[:5]


def send_request(i: int) -> float:
    prompt = PROMPTS[i % len(PROMPTS)]
    payload = {"prompt": prompt, "user_id": f"user-{i}", "max_tokens": MAX_TOKENS}
    start = time.time()
    response = requests.post(
        GENERATE_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    end = time.time()
    response.raise_for_status()
    return end - start


def ensure_server_ready() -> bool:
    try:
        response = requests.get(HEALTH_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        print("Server is not reachable at http://127.0.0.1:8000")
        print("Start it with: uvicorn server.app:app\n")
        return False
    return payload.get("status") == "ok"


def fetch_metrics() -> dict:
    try:
        response = requests.get(METRICS_URL, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {}


def run_level(concurrency: int) -> dict:
    m_before = fetch_metrics()
    tokens_before = m_before.get("total_tokens_generated", 0)
    processed_before = m_before.get("total_processed", 0)
    ttft_sum_before = m_before.get("avg_ttft", 0.0) * processed_before

    latencies = []
    start_wall = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i) for i in range(REQUESTS_PER_LEVEL)]
        for future in futures:
            latencies.append(future.result())
    elapsed = time.time() - start_wall

    m_after = fetch_metrics()
    tokens_after = m_after.get("total_tokens_generated", 0)
    tokens_this_level = tokens_after - tokens_before
    processed_after = m_after.get("total_processed", 0)
    ttft_sum_after = m_after.get("avg_ttft", 0.0) * processed_after
    processed_this_level = processed_after - processed_before
    avg_ttft_this_level = (
        (ttft_sum_after - ttft_sum_before) / processed_this_level
        if processed_this_level > 0
        else 0.0
    )

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    result = {
        "concurrency": concurrency,
        "avg_latency": sum(latencies) / n,
        "p50": latencies_sorted[int(n * 0.50)],
        "p90": latencies_sorted[int(n * 0.90)],
        "p99": latencies_sorted[max(0, int(n * 0.99) - 1)],
        "avg_ttft": avg_ttft_this_level,
        "processed": processed_this_level,
        "throughput_req_s": REQUESTS_PER_LEVEL / elapsed,
        "throughput_tok_s": tokens_this_level / elapsed if elapsed > 0 else 0,
        "elapsed": elapsed,
    }

    print(f"Concurrency: {concurrency}")
    print(f"  Avg latency  : {result['avg_latency']:.3f}s")
    print(f"  P50 latency  : {result['p50']:.3f}s")
    print(f"  P90 latency  : {result['p90']:.3f}s")
    print(f"  P99 latency  : {result['p99']:.3f}s")
    print(f"  Avg TTFT     : {result['avg_ttft']:.3f}s")
    print(f"  Req/s        : {result['throughput_req_s']:.3f}")
    print(f"  Tok/s        : {result['throughput_tok_s']:.1f}")
    print()
    return result


def chat_completion_payload(url: str, model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": OUTPUT_QUALITY_MAX_TOKENS,
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def chat_completion(url: str, model: str, prompt: str) -> str:
    return chat_completion_payload(url, model, prompt)["choices"][0]["message"]["content"]


def send_vllm_request(i: int, vllm_chat_url: str, vllm_model: str) -> tuple[float, int]:
    prompt = PROMPTS[i % len(PROMPTS)]
    start = time.time()
    payload = chat_completion_payload(vllm_chat_url, vllm_model, prompt)
    elapsed = time.time() - start
    completion_tokens = payload.get("usage", {}).get("completion_tokens", 0)
    return elapsed, completion_tokens


def run_level_vllm(concurrency: int, vllm_chat_url: str, vllm_model: str) -> dict:
    latencies = []
    completion_tokens = 0
    start_wall = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_vllm_request, i, vllm_chat_url, vllm_model)
            for i in range(REQUESTS_PER_LEVEL)
        ]
        for future in futures:
            latency, tokens = future.result()
            latencies.append(latency)
            completion_tokens += tokens
    elapsed = time.time() - start_wall
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    return {
        "concurrency": concurrency,
        "avg_latency": sum(latencies) / n,
        "p50": latencies_sorted[int(n * 0.50)],
        "p90": latencies_sorted[int(n * 0.90)],
        "p99": latencies_sorted[max(0, int(n * 0.99) - 1)],
        "throughput_req_s": REQUESTS_PER_LEVEL / elapsed,
        "throughput_tok_s": completion_tokens / elapsed if elapsed > 0 else 0,
        "elapsed": elapsed,
    }


def request_error_summary(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    ngrok_code = response.headers.get("ngrok-error-code")
    if ngrok_code:
        return f"{response.status_code} {response.reason} ({ngrok_code})"
    return f"{response.status_code} {response.reason}"


def get_vllm_model_or_error(vllm_url: str) -> tuple[str | None, str | None]:
    models_url = f"{vllm_url.rstrip('/')}/models"
    try:
        response = requests.get(models_url, timeout=30)
        response.raise_for_status()
        models = response.json().get("data", [])
    except (ValueError, requests.RequestException) as exc:
        if isinstance(exc, requests.RequestException):
            return None, request_error_summary(exc)
        return None, str(exc)
    if os.getenv("VLLM_MODEL"):
        return os.environ["VLLM_MODEL"], None
    if models:
        return models[0].get("id", "default"), None
    return "default", None


def print_side_by_side(left: str, right: str, width: int = 39) -> None:
    left_lines = textwrap.wrap(left, width=width) or [""]
    right_lines = textwrap.wrap(right, width=width) or [""]
    rows = max(len(left_lines), len(right_lines))
    for i in range(rows):
        lhs = left_lines[i] if i < len(left_lines) else ""
        rhs = right_lines[i] if i < len(right_lines) else ""
        print(f"  {lhs:<{width}} | {rhs:<{width}}")


def print_output_quality_comparison() -> None:
    print("Section D - Output quality")
    print("-" * 84)
    vllm_url = os.getenv("VLLM_URL")
    if not vllm_url:
        print("Set VLLM_URL=http://<host>/v1 to enable output comparison")
        return

    vllm_url = vllm_url.rstrip("/")
    vllm_chat_url = f"{vllm_url}/chat/completions"
    vllm_model, vllm_error = get_vllm_model_or_error(vllm_url)
    print(f"Local chat URL : {CHAT_URL}")
    print(f"vLLM chat URL  : {vllm_chat_url}")
    if vllm_error is not None:
        print(f"vLLM unavailable: {vllm_error}")
        print("Skipping output comparison until VLLM_URL /models returns JSON.")
        return
    print(f"vLLM model     : {vllm_model}")
    print()

    for idx, prompt in enumerate(OUTPUT_QUALITY_PROMPTS, start=1):
        print(f"Prompt {idx}: {prompt}")
        try:
            yours = chat_completion(CHAT_URL, LOCAL_CHAT_MODEL, prompt)
        except requests.RequestException as exc:
            yours = f"[local request failed: {exc}]"
        try:
            vllm = chat_completion(vllm_chat_url, vllm_model, prompt)
        except requests.RequestException as exc:
            vllm = f"[vLLM request failed: {request_error_summary(exc)}]"
        print(f"  {'Yours':<39} | {'vLLM':<39}")
        print_side_by_side(yours, vllm)
        print()


def value_for(results: list[dict], concurrency: int, key: str, default: float = 0.0) -> float:
    return next((r[key] for r in results if r["concurrency"] == concurrency), default)


def format_number(value: float, suffix: str = "", precision: int = 1) -> str:
    if value <= 0:
        return "n/a"
    return f"{value:.{precision}f}{suffix}"


def print_comparison_table(results: list[dict], vllm_results: list[dict]) -> None:
    print("=" * 84)
    print("COMPARISON TABLE - Llama-3.2-1B serving stack")
    print("=" * 84)
    print("  Fair section  : Section B compares scheduling ratios, not raw hardware speed")
    print("  Raw section   : Section C labels hardware explicitly")
    print(f"  Max tokens    : {MAX_TOKENS}")
    print()

    print("Section A - Feature parity checklist")
    print("-" * 84)
    print(f"{'':<4} {'Feature':<28} {'vLLM':<24} {'Yours':<24}")
    print(f"{'[x]':<4} {'Continuous batching':<28} {'BatchGenerator':<24} {'MLX_GLOBAL_BATCH_GENERATOR':<24}")
    print(f"{'[x]':<4} {'Paged KV memory manager':<28} {'GPU block tables':<24} {'Python block pool + optional mx.take':<24}")
    print(f"{'[x]':<4} {'Prefix KV cache':<28} {'Radix cache':<24} {'Radix trie':<24}")
    print(f"{'[x]':<4} {'Speculative decoding':<28} {'draft model':<24} {'mlx_lm draft model':<24}")
    print(f"{'[x]':<4} {'Preemption':<28} {'swap/requeue':<24} {'requeue':<24}")
    print(f"{'[x]':<4} {'Chunked prefill (sched)':<28} {'model-level':<24} {'model-level (real)':<24}")
    print(f"{'[x]':<4} {'OpenAI API':<28} {'yes':<24} {'/v1/chat/completions':<24}")
    print(f"{'[ ]':<4} {'Fused block attention':<28} {'CUDA kernel':<24} {'N/A (Metal/MLX limitation)':<24}")
    print(f"{'[ ]':<4} {'Tensor parallelism':<28} {'yes':<24} {'N/A':<24}")
    print()

    print("Section B - Scheduling efficiency")
    print("-" * 84)
    print(f"{'Metric':<44} {'Yours':>14} {'vLLM (e2e)':>14}")
    print("-" * 84)

    # Throughput
    tok_s_1 = value_for(results, 1, "throughput_tok_s")
    tok_s_8 = value_for(results, 8, "throughput_tok_s")
    vllm_tok_s_1 = value_for(vllm_results, 1, "throughput_tok_s")
    vllm_tok_s_8 = value_for(vllm_results, 8, "throughput_tok_s")
    batching_gain = tok_s_8 / tok_s_1 if tok_s_1 > 0 else 0
    vllm_batching_gain = vllm_tok_s_8 / vllm_tok_s_1 if vllm_tok_s_1 > 0 else 0
    print(
        f"{'Tok/s @ C1':<44} {format_number(tok_s_1):>14} "
        f"{format_number(vllm_tok_s_1):>14}"
    )
    print(
        f"{'Tok/s @ C8':<44} {format_number(tok_s_8):>14} "
        f"{format_number(vllm_tok_s_8):>14}"
    )
    print(
        f"{'Batching gain (C8/C1 tok/s)':<44} {format_number(batching_gain, 'x'):>14} "
        f"{format_number(vllm_batching_gain, 'x'):>14}"
    )

    # TTFT
    ttft_1 = value_for(results, 1, "avg_ttft")
    ttft_4 = value_for(results, 4, "avg_ttft")
    p50_1 = value_for(results, 1, "p50")
    p50_8 = value_for(results, 8, "p50")
    vllm_p50_1 = value_for(vllm_results, 1, "p50")
    vllm_p50_8 = value_for(vllm_results, 8, "p50")
    ttft_ratio = ttft_4 / ttft_1 if ttft_1 > 0 else 0
    vllm_p50_4 = value_for(vllm_results, 4, "p50")
    vllm_e2e_ratio = vllm_p50_4 / vllm_p50_1 if vllm_p50_1 > 0 else 0
    print(
        f"{'P50 latency @ C1':<44} {format_number(p50_1, 's', 2):>14} "
        f"{format_number(vllm_p50_1, 's', 2):>14}"
    )
    print(
        f"{'P50 latency @ C8':<44} {format_number(p50_8, 's', 2):>14} "
        f"{format_number(vllm_p50_8, 's', 2):>14}"
    )
    print(
        f"{'TTFT ratio C4/C1 (vLLM: e2e p50 ratio)':<44} {format_number(ttft_ratio, 'x'):>14} "
        f"{format_number(vllm_e2e_ratio, 'x'):>14}"
    )

    print()
    print("Section C - Raw throughput (hardware-labeled)")
    print("-" * 84)
    print(f"  Yours (M1 Air, Llama-3.2-1B 4-bit) @ C1: {tok_s_1:.1f} tok/s")
    print(f"  Yours (M1 Air, Llama-3.2-1B 4-bit) @ C8: {tok_s_8:.1f} tok/s")
    print(f"  Yours TTFT @ C1: {ttft_1:.2f}s")
    if vllm_results:
        print(f"  vLLM @ C1: {vllm_tok_s_1:.1f} tok/s")
        print(f"  vLLM @ C8: {vllm_tok_s_8:.1f} tok/s")
    else:
        vllm_url = os.getenv("VLLM_URL")
        if not vllm_url:
            print("  vLLM reference: set VLLM_URL=http://<host>/v1 to collect live numbers")
        else:
            print("  vLLM throughput skipped because VLLM_URL is not reachable")

    print()
    print_output_quality_comparison()


def save_results(results: list[dict], vllm_results: list[dict]) -> None:
    payload = {
        "config": {
            "concurrency_levels": CONCURRENCY_LEVELS,
            "requests_per_level": REQUESTS_PER_LEVEL,
            "max_tokens": MAX_TOKENS,
            "local_model": LOCAL_CHAT_MODEL,
            "vllm_url": os.getenv("VLLM_URL"),
        },
        "local": results,
        "vllm": vllm_results,
    }
    os.makedirs(os.path.dirname(RESULTS_JSON) or ".", exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results JSON: {RESULTS_JSON}")


def main() -> None:
    print("Running load test against /generate")
    print(f"Target       : {GENERATE_URL}")
    print(f"Req/level    : {REQUESTS_PER_LEVEL}")
    print(f"Max tokens   : {MAX_TOKENS}\n")

    if not ensure_server_ready():
        return

    results = []
    for concurrency in CONCURRENCY_LEVELS:
        results.append(run_level(concurrency))

    vllm_results = []
    vllm_url = os.getenv("VLLM_URL")
    if vllm_url:
        vllm_url = vllm_url.rstrip("/")
        vllm_chat_url = f"{vllm_url}/chat/completions"
        vllm_model, vllm_error = get_vllm_model_or_error(vllm_url)
        if vllm_error is not None:
            print(f"\nSkipping vLLM sweep: {vllm_error}")
        else:
            print(f"\nRunning same sweep against vLLM at {vllm_chat_url}")
            for concurrency in CONCURRENCY_LEVELS:
                result = run_level_vllm(concurrency, vllm_chat_url, vllm_model)
                vllm_results.append(result)
                print(f"vLLM concurrency: {concurrency}")
                print(f"  Avg latency  : {result['avg_latency']:.3f}s")
                print(f"  P50 latency  : {result['p50']:.3f}s")
                print(f"  P90 latency  : {result['p90']:.3f}s")
                print(f"  P99 latency  : {result['p99']:.3f}s")
                print(f"  Req/s        : {result['throughput_req_s']:.3f}")
                print(f"  Tok/s        : {result['throughput_tok_s']:.1f}")
                print()

    print_comparison_table(results, vllm_results)
    save_results(results, vllm_results)


if __name__ == "__main__":
    main()
