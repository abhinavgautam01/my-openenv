# Email Triage OpenEnv Submission Spec

This document records the concrete submission contract for this repository and the implementation choices that satisfy it.

## 1. Submission Contract

The repository must provide:

- A real-world environment, not a game or toy.
- Typed OpenEnv-compatible action and observation models.
- Standard `reset()`, `step()`, and `state()` interaction.
- A valid `openenv.yaml`.
- At least 3 deterministic tasks with graders that score in `[0.0, 1.0]`.
- Meaningful reward shaping with partial-progress signal.
- A root-level `inference.py` demo script using the OpenAI client with:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`
- Structured stdout logs using `[START]`, `[STEP]`, and `[END]`.
- A working Docker build and a deployed Hugging Face Space.
- A README covering environment motivation, spaces, setup, usage, and validation.

## 2. Implemented Environment

Domain:

- Email triage for knowledge workers and operations teams.

Tasks:

- `classification`: 1 email
- `ranking`: 8 emails
- `full_triage`: 25 emails

Core interface:

- Typed models in `models.py` and `server/models.py`
- API endpoints in `server/app.py`
- Scenario generation in `server/scenarios/generator.py`
- Graders in `server/graders.py`

## 3. Task Expectations

### `classification`

- Agent classifies one email into one of five categories.
- Exact match grader with score `0.0` or `1.0`.

### `ranking`

- Agent submits one `ranking=[...]` list containing all 8 email IDs exactly once.
- Graded with normalized Kendall Tau score in `[0.0, 1.0]`.

### `full_triage`

- Agent processes 25 emails with priority, category, disposition, and optional response draft.
- Hard mode includes thread dependencies, duplicate handling, time pressure, and an 8-response budget.
- Final grader scores priority, category, disposition, response quality, response-budget efficiency, thread awareness, and completion rate.

## 4. Runtime and Packaging

- Root Docker image is used for local validation and Hugging Face Docker Spaces.
- `requirements.txt` exists at repo root.
- `inference.py` exists at repo root and acts as the demo script.
- `openenv.yaml` exists at repo root.

## 5. Validation Checklist

The current repo is intended to be validated with:

```bash
python3 -m pytest -q
openenv validate
docker build -t email-triage-openenv .
docker run --rm -p 8000:8000 email-triage-openenv
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
```

## 6. Deployment Targets

- GitHub repository: `https://github.com/abhinavgautam01/my-openenv`
- Hugging Face Space repository: `https://huggingface.co/spaces/abhinavgautam01/my-env`
- Hugging Face app URL: `https://abhinavgautam01-my-env.hf.space`
