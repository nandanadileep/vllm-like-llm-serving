# vLLM-like LLM Serving

MLX-based LLM serving stack with continuous batching, prefix KV cache, chunked prefill, OpenAI-compatible chat completions, and benchmark comparison against vLLM and llama.cpp CPU.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run Local Server

```bash
. .venv/bin/activate
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

The local server exposes:

```text
http://127.0.0.1:8000/generate
http://127.0.0.1:8000/v1/chat/completions
http://127.0.0.1:8000/metrics
```

## Optional vLLM Baseline

Use the Colab guide in `docs/colab/vllm_llama32_server.md` to start vLLM and get an ngrok URL like:

```text
https://<your-ngrok-host>/v1
```

## Optional llama.cpp CPU Baseline

Install and download the GGUF model:

```bash
brew install llama.cpp
mkdir -p ~/models
hf download bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --local-dir ~/models
```

Start the CPU-only server in a separate terminal:

```bash
llama-server \
  --model ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --port 8001 \
  --n-gpu-layers 0 \
  --ctx-size 2048
```

## Run Benchmarks

Local server only:

```bash
. .venv/bin/activate
python experiments/load_test.py
```

Local server plus vLLM and llama.cpp CPU:

```bash
. .venv/bin/activate
VLLM_URL=https://<your-ngrok-host>/v1 CPU_URL=http://127.0.0.1:8001/v1 python experiments/load_test.py
python experiments/plot_results.py
```

Results are written to:

```text
experiments/latest_results.json
experiments/plots/
```

If `VLLM_URL` or `CPU_URL` is missing or unreachable, that baseline is skipped.
