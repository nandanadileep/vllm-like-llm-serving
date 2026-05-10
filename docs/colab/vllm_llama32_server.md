# vLLM Llama 3.2 Colab Server

Run these cells in Google Colab to expose a vLLM OpenAI-compatible server for
`meta-llama/Llama-3.2-1B-Instruct`.

Do not commit Hugging Face or ngrok tokens. Paste them only into the Colab
runtime prompts.

## 1. Install dependencies

```python
!pip install -q "vllm" "pyngrok" "huggingface_hub"
```

## 2. Authenticate Hugging Face and ngrok

You need approved Hugging Face access for
`meta-llama/Llama-3.2-1B-Instruct` and an ngrok authtoken.

```python
from getpass import getpass

from huggingface_hub import login
from pyngrok import ngrok

hf_token = getpass("Hugging Face token: ")
login(token=hf_token)

ngrok_token = getpass("ngrok authtoken: ")
ngrok.set_auth_token(ngrok_token)
```

## 3. Verify gated model access

```python
from huggingface_hub import model_info

model_info("meta-llama/Llama-3.2-1B-Instruct")
print("Hugging Face model access OK")
```

## 4. Start vLLM

If vLLM fails with GPU memory errors, lower `GPU_MEMORY_UTILIZATION` to
`0.50` or reduce `MAX_MODEL_LEN`.

```python
import os
import signal
import subprocess
import time

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PORT = 8000
MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.65
LOG_FILE = "vllm.log"

try:
    if "vllm_proc" in globals() and vllm_proc.poll() is None:
        os.killpg(os.getpgid(vllm_proc.pid), signal.SIGTERM)
except Exception as exc:
    print(f"Previous server cleanup skipped: {exc}")

cmd = [
    "python",
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    MODEL,
    "--host",
    "0.0.0.0",
    "--port",
    str(PORT),
    "--max-model-len",
    str(MAX_MODEL_LEN),
    "--gpu-memory-utilization",
    str(GPU_MEMORY_UTILIZATION),
]

log = open(LOG_FILE, "w")
vllm_proc = subprocess.Popen(
    cmd,
    stdout=log,
    stderr=subprocess.STDOUT,
    preexec_fn=os.setsid,
)
print(f"Started vLLM pid={vllm_proc.pid}. Logs: {LOG_FILE}")
time.sleep(5)
```

## 5. Wait for local readiness

Model loading can take several minutes.

```python
import requests
import time

models_url = f"http://localhost:{PORT}/v1/models"
deadline = time.time() + 600
last_error = None

while time.time() < deadline:
    try:
        response = requests.get(models_url, timeout=10)
        if response.ok:
            print("vLLM is ready")
            print(response.json())
            break
        last_error = f"{response.status_code} {response.text[:200]}"
    except requests.RequestException as exc:
        last_error = str(exc)
    time.sleep(10)
else:
    raise RuntimeError(f"vLLM did not become ready. Last error: {last_error}")
```

## 6. Inspect logs if readiness fails

```python
!tail -100 vllm.log
```

## 7. Expose with ngrok

```python
from pyngrok import ngrok

ngrok.kill()
public_url = ngrok.connect(PORT, "http")
VLLM_URL = f"{public_url.public_url}/v1"
print(f'Ngrok tunnel: "{public_url.public_url}" -> "http://localhost:{PORT}"')
print(f"Use this as VLLM_URL: {VLLM_URL}")
```

## 8. Test `/v1/models`

```python
response = requests.get(f"{VLLM_URL}/models", timeout=30)
print(response.status_code)
print(response.text[:1000])
```

## 9. Test `/v1/chat/completions`

```python
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 20,
    "stream": False,
}

response = requests.post(f"{VLLM_URL}/chat/completions", json=payload, timeout=120)
print(response.status_code)
print(response.text[:2000])
```

## 10. Run benchmark from your Mac

Use the printed ngrok URL as `VLLM_URL`:

```bash
. .venv/bin/activate
VLLM_URL=https://<your-ngrok-host>/v1 python experiments/load_test.py
python experiments/plot_results.py
```
