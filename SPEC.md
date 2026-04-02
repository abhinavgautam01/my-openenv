# Email Triage OpenEnv Submission Spec

This document is the working submission spec for this repo. It is intentionally strict and repo-specific: it captures what the competition requires, what this implementation currently does, and what still blocks a strong submission.

## 1. Competition Contract

The submission must provide all of the following:

- A real-world environment, not a toy or game.
- Typed OpenEnv models for observation, action, and reward/state payloads.
- Standard environment interaction via `reset()`, `step()`, and `state()`.
- A valid `openenv.yaml`.
- At least 3 tasks with deterministic programmatic graders scoring in `[0.0, 1.0]`.
- Meaningful reward shaping with partial-progress signal across the trajectory.
- A root-level `inference.py` that uses the OpenAI client and reads:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`
- Structured stdout logs in the required `[START]`, `[STEP]`, `[END]` format.
- Containerized deployment with a working Docker build.
- Hugging Face Spaces deployment as an `openenv`-tagged Docker Space.
- A README covering environment motivation, action/observation spaces, tasks, setup, usage, and baseline scores.

## 2. Judging Priorities

Weights from the prompt:

- Real-world utility: 30%
- Task and grader quality: 25%
- Environment design: 20%
- Code quality and spec compliance: 15%
- Creativity and novelty: 10%

Winning entries will usually need more than validator compliance. They need strong domain modeling, clean grader design, a hard task that actually pressures frontier models, and no obvious deployment or reproducibility risks.

## 3. Current Repo Snapshot

Domain:

- Email triage is a valid real-world domain and is acceptable for the challenge.

Implemented tasks:

- `classification`
- `ranking`
- `full_triage`

Actual task sizes in code:

- `classification`: 1 email
- `ranking`: 8 emails
- `full_triage`: 25 emails

The primary docs and metadata are now aligned with the code at 1 / 8 / 25.

## 4. Requirement Status

### Pass

- Real-world domain exists.
- Typed Pydantic models exist in `models.py` and `server/models.py`.
- `openenv validate` passes in the current workspace.
- Three tasks exist.
- Deterministic scenario generation exists via seed-based templates.
- Tests currently pass locally.
- There is a Dockerfile in `server/`.
- There is a repo-root Dockerfile for Docker Space style deployment.
- Reward semantics are now incremental per step, with cumulative reward tracked separately.
- Ranking actions now receive shaped reward directly from the environment.

### Partial

- Graders exist, but the hard task still behaves mostly like repeated independent labeling rather than a truly rich inbox-management problem.
- Inference script exists and uses the OpenAI client, but there are compatibility and reproducibility risks.

### Fail / High Risk

- Hugging Face Space deployment readiness is still not demonstrated against a live Space URL.
- External baseline reproduction still depends on your model credentials and deployment environment.

## 5. High-Priority Gaps

### Gap A: The hard task is not yet strong enough to feel frontier-hard

Current behavior:

- `full_triage` is mainly per-email prediction with a response cap.
- There are limited cross-email dependencies, limited thread memory, and no inbox-level tradeoff reasoning beyond simple prioritization.

Why it matters:

- Good models will likely perform well once they learn the template distribution.
- Judges looking for a winner will want deeper mechanics than repeated template classification.

### Gap B: Inference still needs live reproduction

Current behavior:

- `inference.py` now supports `LOCAL_IMAGE_NAME`, normalizes ranking outputs, and the client forwards `ranking`.
- Actual reproducibility against the evaluator model/router still depends on external credentials and runtime conditions.

Why it matters:

- This can still break evaluator runs if the deployed image or model endpoint differs from local assumptions.

### Gap C: Hugging Face Space deployment still needs proof

Current behavior:

- There is now a repo-root Dockerfile.
- A real HF Space build and live `/reset` ping have not yet been re-run on this commit.

Why it matters:

- This remains a possible disqualification path until the deployed Space is verified.

## 6. Competitive Assessment

As of this snapshot:

- The project is a credible submission candidate.
- It is not yet a strong “likely winner” candidate.

Current estimated scoring band if submitted as-is:

- Real-world utility: 20-24 / 30
- Task and grader quality: 15-19 / 25
- Environment design: 11-15 / 20
- Code quality and spec compliance: 9-12 / 15
- Creativity and novelty: 4-6 / 10

Estimated total:

- 59-76 / 100

That is good enough to be taken seriously, but not good enough to assume a top finish.

## 7. What Must Be Fixed Before Calling It “Submission Ready”

1. Rebuild and verify the root Docker image and live HTTP endpoints.
2. Deploy the current commit to a Hugging Face Docker Space and verify `/reset`.
3. Re-run `inference.py` with real evaluator-compatible credentials and record baseline scores from this commit.
4. Make the hard task more inbox-native:
   - thread continuity
   - conflicting deadlines
   - sender-history context
   - limited response budget tied to actual tradeoffs
   - actions whose value depends on what was already done elsewhere in the inbox

## 8. Recommendation

Use this repo as the base, but do not submit it unchanged.

The core idea is good. The current implementation is closer to “solid prototype that passes basic validation” than “competition-winning benchmark.” The next phase should focus on consistency, reward correctness, evaluator compatibility, and making the hard task genuinely difficult in a realistic way.
