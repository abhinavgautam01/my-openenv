# Quick Start

## Local Validation

```bash
cd /Users/gautam/Documents/scaler/round1
python3 -m pytest -q
openenv validate
docker build -t email-triage-openenv .
docker run --rm -p 8000:8000 email-triage-openenv
curl http://localhost:8000/health
```

## Hugging Face Space

1. Create a new Docker Space.
2. Upload the repo with the root `Dockerfile`.
3. Set:
   - `HF_TOKEN`
   - `API_BASE_URL`
   - `MODEL_NAME`
4. Verify:

```bash
curl https://YOUR_SPACE.hf.space/health
curl -X POST https://YOUR_SPACE.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Inference

```bash
export HF_TOKEN=...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export LOCAL_IMAGE_NAME=email-triage-openenv:latest
python3 inference.py
```
