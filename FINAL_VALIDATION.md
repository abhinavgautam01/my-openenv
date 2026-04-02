# Validation Notes

This file is no longer a frozen “ready to submit” report.

Use the current local checks instead:

```bash
python3 -m pytest -q
openenv validate
docker build -t email-triage-openenv .
docker run --rm -p 8000:8000 email-triage-openenv
curl http://localhost:8000/health
```

External validation still needs to be reproduced in your own environment:

- Hugging Face Space build
- live `/reset` ping on the deployed Space
- baseline inference with your own model credentials
