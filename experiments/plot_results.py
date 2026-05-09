import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt


def series(results: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    return (
        [int(r["concurrency"]) for r in results],
        [float(r.get(key, 0.0)) for r in results],
    )


def plot_metric(
    local: list[dict[str, Any]],
    vllm: list[dict[str, Any]],
    key: str,
    ylabel: str,
    title: str,
    output_path: str,
) -> None:
    plt.figure(figsize=(7, 4.5))
    x, y = series(local, key)
    plt.plot(x, y, marker="o", label="Yours (M1 Air)")
    if vllm:
        x_vllm, y_vllm = series(vllm, key)
        plt.plot(x_vllm, y_vllm, marker="o", label="vLLM (e2e)")
    plt.xlabel("Concurrency")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_batching_gain(
    local: list[dict[str, Any]],
    vllm: list[dict[str, Any]],
    output_path: str,
) -> None:
    def gain(results: list[dict[str, Any]]) -> float:
        by_c = {int(r["concurrency"]): float(r.get("throughput_tok_s", 0.0)) for r in results}
        return by_c.get(8, 0.0) / by_c.get(1, 1.0) if by_c.get(1, 0.0) > 0 else 0.0

    labels = ["Yours"]
    values = [gain(local)]
    if vllm:
        labels.append("vLLM")
        values.append(gain(vllm))

    plt.figure(figsize=(5.5, 4))
    plt.bar(labels, values)
    plt.ylabel("C8 tok/s / C1 tok/s")
    plt.title("Batching Gain")
    plt.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot load test comparison results.")
    parser.add_argument(
        "results_json",
        nargs="?",
        default="experiments/latest_results.json",
        help="Path to JSON produced by experiments/load_test.py",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/plots",
        help="Directory for generated PNG files",
    )
    args = parser.parse_args()

    with open(args.results_json, encoding="utf-8") as f:
        data = json.load(f)

    local = data.get("local", [])
    vllm = data.get("vllm", [])
    os.makedirs(args.out_dir, exist_ok=True)

    plot_metric(
        local,
        vllm,
        "throughput_tok_s",
        "Tokens / second",
        "Decode Throughput vs Concurrency",
        os.path.join(args.out_dir, "throughput_tok_s.png"),
    )
    plot_metric(
        local,
        vllm,
        "p50",
        "P50 latency (s)",
        "P50 End-to-End Latency vs Concurrency",
        os.path.join(args.out_dir, "p50_latency.png"),
    )
    plot_metric(
        local,
        vllm,
        "p99",
        "P99 latency (s)",
        "P99 End-to-End Latency vs Concurrency",
        os.path.join(args.out_dir, "p99_latency.png"),
    )
    if local and "avg_ttft" in local[0]:
        plot_metric(
            local,
            [],
            "avg_ttft",
            "TTFT (s)",
            "Local TTFT vs Concurrency",
            os.path.join(args.out_dir, "local_ttft.png"),
        )
    plot_batching_gain(
        local,
        vllm,
        os.path.join(args.out_dir, "batching_gain.png"),
    )

    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
