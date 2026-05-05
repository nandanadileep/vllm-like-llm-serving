import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE_URL = "http://127.0.0.1:8000"
GENERATE_URL = f"{BASE_URL}/generate"
METRICS_URL = f"{BASE_URL}/metrics"
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
REQUESTS_PER_LEVEL = 64
REQUEST_TIMEOUT_SECONDS = 10


def send_request(i: int) -> float:
    payload = {"prompt": f"hello {i}", "user_id": f"user-{i}"}
    start = time.time()
    response = requests.post(
        GENERATE_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    end = time.time()
    response.raise_for_status()
    return end - start


def fetch_metrics() -> None:
    try:
        response = requests.get(METRICS_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        metrics = response.json()
    except requests.RequestException:
        print("Metrics: /metrics endpoint not available\n")
        return

    print("Metrics:")
    print(f"  total_batches: {metrics.get('total_batches', 'n/a')}")
    print(f"  avg_wait_time: {metrics.get('avg_wait_time', 'n/a')}")
    print(f"  max_queue_length: {metrics.get('max_queue_length', 'n/a')}\n")


def run_level(concurrency: int) -> None:
    latencies = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i) for i in range(REQUESTS_PER_LEVEL)]
        for future in futures:
            latencies.append(future.result())

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)

    print(f"Concurrency: {concurrency}")
    print(f"Avg latency: {avg_latency:.4f}s")
    print(f"Max latency: {max_latency:.4f}s")
    print(f"Min latency: {min_latency:.4f}s\n")


def main() -> None:
    print("Running load test against /generate")
    print(f"Target: {GENERATE_URL}")
    print(f"Requests per level: {REQUESTS_PER_LEVEL}\n")

    for concurrency in CONCURRENCY_LEVELS:
        run_level(concurrency)

    fetch_metrics()


if __name__ == "__main__":
    main()
