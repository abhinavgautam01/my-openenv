# Submission Status

This repo should not be treated as “automatically ready” without rerunning validation.

Before submission, verify all of the following on the current commit:

```bash
python3 -m pytest -q
openenv validate
docker build -t email-triage-openenv .
docker run --rm -p 8000:8000 email-triage-openenv
curl http://localhost:8000/health
```

Then validate the deployed Hugging Face Space:

```bash
curl -X POST https://YOUR_SPACE.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```

Baseline scores should be regenerated with your actual model endpoint and credentials.
