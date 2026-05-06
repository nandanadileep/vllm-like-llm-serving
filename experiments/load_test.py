import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE_URL = "http://127.0.0.1:8000"
GENERATE_URL = f"{BASE_URL}/generate"
HEALTH_URL = f"{BASE_URL}/health"
METRICS_URL = f"{BASE_URL}/metrics"
CONCURRENCY_LEVELS = [1, 2, 4, 8]
REQUESTS_PER_LEVEL = 10
REQUEST_TIMEOUT_SECONDS = 300
MAX_TOKENS = 50

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

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    result = {
        "concurrency": concurrency,
        "avg_latency": sum(latencies) / n,
        "p50": latencies_sorted[int(n * 0.50)],
        "p90": latencies_sorted[int(n * 0.90)],
        "p99": latencies_sorted[max(0, int(n * 0.99) - 1)],
        "throughput_req_s": REQUESTS_PER_LEVEL / elapsed,
        "throughput_tok_s": tokens_this_level / elapsed if elapsed > 0 else 0,
        "elapsed": elapsed,
    }

    print(f"Concurrency: {concurrency}")
    print(f"  Avg latency  : {result['avg_latency']:.3f}s")
    print(f"  P50 latency  : {result['p50']:.3f}s")
    print(f"  P90 latency  : {result['p90']:.3f}s")
    print(f"  P99 latency  : {result['p99']:.3f}s")
    print(f"  Req/s        : {result['throughput_req_s']:.3f}")
    print(f"  Tok/s        : {result['throughput_tok_s']:.1f}")
    print()
    return result


def print_comparison_table(results: list[dict], final_metrics: dict) -> None:
    avg_ttft = final_metrics.get("avg_ttft", 0.0)
    memory_mb = final_metrics.get("memory_mb", 0.0)

    print("=" * 65)
    print("COMPARISON TABLE — Your impl (M1 Air 8GB) vs vLLM (RTX 4090)")
    print("=" * 65)
    print(f"  Model (yours) : Llama-3.2-3B 4-bit")
    print(f"  Model (vLLM)  : Llama-3.1-8B FP16")
    print(f"  Max tokens    : {MAX_TOKENS} (yours) / 256 (vLLM)")
    print()

    print(f"{'Metric':<30} {'Yours':>12} {'vLLM':>12}")
    print("-" * 56)

    # Throughput
    tok_s_1 = next((r["throughput_tok_s"] for r in results if r["concurrency"] == 1), 0)
    tok_s_8 = next((r["throughput_tok_s"] for r in results if r["concurrency"] == 8), 0)
    print(f"{'Tok/s @ concurrency 1':<30} {tok_s_1:>11.1f} {'71':>12}")
    print(f"{'Tok/s @ concurrency 8':<30} {tok_s_8:>11.1f} {'~485 (@ 10)':>12}")

    # Latency
    p50_1 = next((r["p50"] for r in results if r["concurrency"] == 1), 0)
    p99_1 = next((r["p99"] for r in results if r["concurrency"] == 1), 0)
    p50_8 = next((r["p50"] for r in results if r["concurrency"] == 8), 0)
    p99_8 = next((r["p99"] for r in results if r["concurrency"] == 8), 0)
    print(f"{'P50 latency @ concurrency 1':<30} {p50_1:>10.2f}s {'3.5s':>12}")
    print(f"{'P99 latency @ concurrency 1':<30} {p99_1:>10.2f}s {'3.8s':>12}")
    print(f"{'P50 latency @ concurrency 8':<30} {p50_8:>10.2f}s {'~1.8s (@ 50)':>12}")

    # TTFT
    print(f"{'Avg TTFT':<30} {avg_ttft:>10.2f}s {'0.082s':>12}")

    # Memory
    print(f"{'Server memory':<30} {memory_mb:>9.0f}MB {'~16,000MB':>12}")

    # Batching efficiency
    tok_s_base = next((r["throughput_tok_s"] for r in results if r["concurrency"] == 1), 1)
    tok_s_peak = max(r["throughput_tok_s"] for r in results)
    batching_gain = tok_s_peak / tok_s_base if tok_s_base > 0 else 0
    print(f"{'Batching efficiency gain':<30} {batching_gain:>10.1f}x {'~13x (@ 50)':>12}")

    print()
    print("Scheduler metrics:")
    print(f"  total_batches        : {final_metrics.get('total_batches', 'n/a')}")
    print(f"  avg_wait_time        : {final_metrics.get('avg_wait_time', 0):.3f}s")
    print(f"  total_preemptions    : {final_metrics.get('total_preemptions', 'n/a')}")
    print(f"  total_prefill_chunks : {final_metrics.get('total_prefill_chunks', 'n/a')}")
    print(f"  max_queue_length     : {final_metrics.get('max_queue_length', 'n/a')}")
    print(f"  kv_free_blocks       : {final_metrics.get('free_blocks', 'n/a')} / {final_metrics.get('total_blocks', 'n/a')}")


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

    final_metrics = fetch_metrics()
    print_comparison_table(results, final_metrics)


if __name__ == "__main__":
    main()
