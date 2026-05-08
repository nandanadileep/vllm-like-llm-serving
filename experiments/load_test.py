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

PROMPTS = [
    "Explain what a transformer model is in one paragraph.",
    "What is the difference between supervised and unsupervised learning?",
    "Describe how attention mechanisms work in neural networks.",
    "What are the main components of a large language model?",
    "Explain the concept of tokenization in NLP.",
    "What is gradient descent and how does it work?",
    "Describe the encoder-decoder architecture.",
    "What is the role of embeddings in language models?",
    "Explain what fine-tuning a language model means.",
    "What is the difference between GPT and BERT?",
    "How does beam search work in text generation?",
    "What is temperature in the context of language model sampling?",
    "Explain what RLHF means and why it is used.",
    "What is a context window in a language model?",
    "How does quantization reduce model size?",
    "What is the purpose of layer normalization?",
    "Explain what key-value cache does during inference.",
    "What is the difference between prefill and decode phases?",
    "How does continuous batching improve LLM throughput?",
    "What is PagedAttention and why was it introduced?",
]

OUTPUT_QUALITY_PROMPTS = PROMPTS[:5]


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


def chat_completion(url: str, model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": OUTPUT_QUALITY_MAX_TOKENS,
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def infer_vllm_model(vllm_url: str) -> str:
    if os.getenv("VLLM_MODEL"):
        return os.environ["VLLM_MODEL"]
    try:
        response = requests.get(f"{vllm_url.rstrip('/')}/models", timeout=30)
        response.raise_for_status()
        models = response.json().get("data", [])
        if models:
            return models[0]["id"]
    except (KeyError, requests.RequestException):
        pass
    return "default"


def print_side_by_side(left: str, right: str, width: int = 39) -> None:
    left_lines = textwrap.wrap(left, width=width) or [""]
    right_lines = textwrap.wrap(right, width=width) or [""]
    rows = max(len(left_lines), len(right_lines))
    for i in range(rows):
        lhs = left_lines[i] if i < len(left_lines) else ""
        rhs = right_lines[i] if i < len(right_lines) else ""
        print(f"  {lhs:<{width}} | {rhs:<{width}}")


def print_output_quality_comparison() -> None:
    print("Section D — Output quality")
    print("-" * 84)
    vllm_url = os.getenv("VLLM_URL")
    if not vllm_url:
        print("Set VLLM_URL=http://<host>/v1 to enable output comparison")
        return

    vllm_url = vllm_url.rstrip("/")
    vllm_chat_url = f"{vllm_url}/chat/completions"
    vllm_model = infer_vllm_model(vllm_url)
    print(f"Local chat URL : {CHAT_URL}")
    print(f"vLLM chat URL  : {vllm_chat_url}")
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
            vllm = f"[vLLM request failed: {exc}]"
        print(f"  {'Yours':<39} | {'vLLM':<39}")
        print_side_by_side(yours, vllm)
        print()


def print_comparison_table(results: list[dict]) -> None:
    print("=" * 84)
    print("COMPARISON TABLE — Llama-3.2-1B serving stack")
    print("=" * 84)
    print("  Fair section  : Section B compares scheduling ratios, not raw hardware speed")
    print("  Raw section   : Section C labels hardware explicitly")
    print(f"  Max tokens    : {MAX_TOKENS}")
    print()

    print("Section A — Feature parity checklist")
    print("-" * 84)
    print(f"{'':<4} {'Feature':<28} {'vLLM':<24} {'Yours':<24}")
    print(f"{'[x]':<4} {'Continuous batching':<28} {'BatchGenerator':<24} {'MLX_GLOBAL_BATCH_GENERATOR':<24}")
    print(f"{'[x]':<4} {'Paged KV memory manager':<28} {'GPU block tables':<24} {'Python block pool + optional mx.take':<24}")
    print(f"{'[x]':<4} {'Prefix KV cache':<28} {'Radix cache':<24} {'Exact-match + shared prefix':<24}")
    print(f"{'[x]':<4} {'Speculative decoding':<28} {'draft model':<24} {'mlx_lm draft model':<24}")
    print(f"{'[x]':<4} {'Preemption':<28} {'swap/requeue':<24} {'requeue':<24}")
    print(f"{'[x]':<4} {'Chunked prefill (sched)':<28} {'model-level':<24} {'cursor-tracked (scheduler only)':<24}")
    print(f"{'[x]':<4} {'OpenAI API':<28} {'yes':<24} {'/v1/chat/completions':<24}")
    print(f"{'[ ]':<4} {'Fused block attention':<28} {'CUDA kernel':<24} {'N/A (Metal/MLX limitation)':<24}")
    print(f"{'[ ]':<4} {'Tensor parallelism':<28} {'yes':<24} {'N/A':<24}")
    print()

    print("Section B — Scheduling efficiency")
    print("-" * 84)
    print(f"{'Metric':<44} {'Yours':>14}")
    print("-" * 84)

    # Throughput
    tok_s_1 = next((r["throughput_tok_s"] for r in results if r["concurrency"] == 1), 0)
    tok_s_8 = next((r["throughput_tok_s"] for r in results if r["concurrency"] == 8), 0)
    batching_gain = tok_s_8 / tok_s_1 if tok_s_1 > 0 else 0
    print(f"{'Tok/s @ concurrency 1':<44} {tok_s_1:>14.1f}")
    print(f"{'Tok/s @ concurrency 8':<44} {tok_s_8:>14.1f}")
    print(f"{'Batching gain (c8 tok/s / c1 tok/s)':<44} {batching_gain:>13.1f}x")

    # TTFT
    ttft_1 = next((r["avg_ttft"] for r in results if r["concurrency"] == 1), 0)
    ttft_4 = next((r["avg_ttft"] for r in results if r["concurrency"] == 4), 0)
    ttft_ratio = ttft_4 / ttft_1 if ttft_1 > 0 else 0
    print(f"{'TTFT ratio (c4 avg_ttft / c1 avg_ttft)':<44} {ttft_ratio:>13.1f}x")

    print()
    print("Section C — Raw throughput (hardware-labeled)")
    print("-" * 84)
    print(f"  Yours (M1 Air, Llama-3.2-1B 4-bit) @ C1: {tok_s_1:.1f} tok/s")
    print(f"  Yours (M1 Air, Llama-3.2-1B 4-bit) @ C8: {tok_s_8:.1f} tok/s")
    print(f"  Yours TTFT @ C1: {ttft_1:.2f}s")
    vllm_url = os.getenv("VLLM_URL")
    if not vllm_url:
        print("  vLLM reference: set VLLM_URL=http://<host>/v1 to collect live numbers")
    else:
        print("  (vLLM live numbers collected in Section D)")

    print()
    print_output_quality_comparison()


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

    print_comparison_table(results)


if __name__ == "__main__":
    main()
