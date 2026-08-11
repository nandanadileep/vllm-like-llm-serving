# Contributing

Thanks for helping improve this educational LLM serving project. Focused bug fixes, benchmark improvements, documentation, tests, and experiments are welcome.

## Before opening an issue

- Search existing issues to avoid duplicates.
- Include your macOS version, Apple chip, Python version, `mlx-lm` version, model, and the exact command you ran.
- Remove API keys, Hugging Face tokens, ngrok tokens, and private model paths from logs.
- For performance reports, include concurrency, prompt shape, `max_tokens`, configuration variables, and whether the model was already warm.

## Local development

```bash
git clone https://github.com/nandanadileep/vllm-like-llm-serving.git
cd vllm-like-llm-serving
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Check the server before and after your change:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

## Pull requests

1. Keep the change focused and explain the serving behavior it affects.
2. Preserve the existing API response shapes unless the change is intentionally breaking.
3. Document new environment variables in the README.
4. Include reproducible before/after results for performance changes.
5. Do not commit model weights, generated virtual environments, credentials, or private benchmark URLs.

Benchmark results vary by hardware and configuration. Avoid general performance claims based on a single machine or run.
