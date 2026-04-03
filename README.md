---
title: Email Triage OpenEnv
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
---

# Email Triage OpenEnv Environment

Email triage is a real workplace task: people classify incoming mail, decide what matters now, and manage limited attention. This environment turns that workflow into a reproducible OpenEnv benchmark with deterministic tasks, typed models, programmatic graders, and a containerized API.

## Submission Links

- GitHub repository: `https://github.com/abhinavgautam01/my-openenv`
- Hugging Face Space: `https://huggingface.co/spaces/abhinavgautam01/my-env`
- Space app URL: `https://abhinavgautam01-my-env.hf.space`

## Environment Summary

- Domain: inbox triage for knowledge workers
- Benchmark id: `email_triage`
- API: `reset()`, `step()`, `state()`
- Models: typed Pydantic `EmailTriageAction` and `EmailTriageObservation`
- Transport: FastAPI server with `/reset`, `/step`, `/state`, `/health`, `/tasks`

## Tasks

| Task | Difficulty | Emails | Objective |
|------|------------|--------|-----------|
| `classification` | Easy | 1 | Classify a single email into one of 5 categories |
| `ranking` | Medium | 8 | Rank a batch of emails from highest to lowest priority |
| `full_triage` | Hard | 25 | Triage a full inbox with priority, category, disposition, optional response drafts, thread dependencies, and a fixed response budget |

## Action Space

```python
class EmailTriageAction:
    email_id: str
    priority: Literal["HIGH", "MEDIUM", "LOW"] | None
    category: Literal["URGENT", "ACTION_REQUIRED", "INFO", "SPAM", "PERSONAL"] | None
    disposition: Literal["RESPOND", "DELEGATE", "ARCHIVE", "DEFER"] | None
    response_draft: str | None
    ranking: list[str] | None
```

Task-specific expectations:

- `classification`: submit `email_id` and `category`
- `ranking`: submit one action with `ranking=[...]` containing all email IDs exactly once
- `full_triage`: submit `email_id`, `priority`, and `category`; `disposition` and `response_draft` are optional but affect reward and final grade

## Observation Space

```python
class EmailTriageObservation:
    task_id: str
    task_type: Literal["classification", "ranking", "full_triage"]
    emails: list[Email]
    current_time: datetime
    emails_processed: int
    emails_remaining: int
    time_budget_remaining: float | None
    last_action_result: str | None
    done: bool
    reward: float              # immediate reward from the most recent step
    cumulative_reward: float   # total reward across the episode
```

## Reward Design

- `classification`: exact category match yields `1.0`, otherwise `0.0`
- `ranking`: reward is normalized Kendall Tau score in `[0.0, 1.0]`
- `full_triage`: dense per-email reward
  - priority correctness: `0.4`
  - category correctness: `0.3`
  - disposition correctness: `0.2`
  - response quality: `0.1`
- invalid actions receive a penalty
- hard mode also has a response budget, time budget, duplicate-thread penalties, and bonuses for spending responses on the highest-value emails

## Graders

- `classification`: deterministic exact-match grader
- `ranking`: deterministic normalized Kendall Tau grader
- `full_triage`: weighted deterministic grader over priority, category, disposition, response quality, response-budget efficiency, thread awareness, and completion rate

All final task scores are normalized to `[0.0, 1.0]`.

## Hard Mode Mechanics

The hard task is intentionally more than repeated classification. It includes:

- Multi-email threads with `root`, `followup`, and `duplicate` messages
- Emails that should usually be handled after another message in the same thread
- A fixed response budget, so low-value replies consume scarce capacity
- Duplicate or visibility-only follow-ups that should usually be archived or deferred
- High-impact business contexts such as incidents, renewals, approvals, and executive meeting prep

## Local Setup

```bash
pip install -e ".[server,dev,inference]"
python3 -m pytest -q
openenv validate
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Docker

Root Docker build:

```bash
docker build -t email-triage-openenv .
docker run --rm -p 8000:8000 email-triage-openenv
curl http://localhost:8000/health
```

Validator-compatible server-only build:

```bash
docker build -t email-triage-openenv ./server
```

## API Examples

Reset:

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type":"ranking","seed":42}'
```

Step for ranking:

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"email_id":"e1","ranking":["e1","e3","e2","e4","e5","e6","e7","e8"]}'
```

## Inference

The root `inference.py` is the demo script and baseline inference runner. It uses the OpenAI client and reads:

- `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME`
- `LOCAL_IMAGE_NAME`

Optional task selectors:

- `EMAIL_TRIAGE_TASK=ranking`
- `EMAIL_TRIAGE_TASKS=all`

Example:

```bash
export HF_TOKEN=...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export LOCAL_IMAGE_NAME=email-triage-openenv:latest
python3 inference.py
```

Stdout uses the required structured format:

```text
[START] task=classification env=email_triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=classify(e1,URGENT) reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00
```

## Validation

Validated locally with:

- `python3 -m pytest -q`
- `openenv validate`
- `docker build -t email-triage-openenv-final-check .`
- fresh container smoke checks for `/health`, `/tasks`, and `/reset`

The deployed Space is available at `https://abhinavgautam01-my-env.hf.space`.
