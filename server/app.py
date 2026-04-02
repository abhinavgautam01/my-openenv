"""
FastAPI application for Email Triage OpenEnv Environment.

Provides HTTP endpoints:
- POST /reset - Reset environment and get initial observation
- POST /step - Execute action and get result
- GET /state - Get current environment state
- GET /health - Health check
"""

import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
from uuid import uuid4
import logging

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import EmailTriageAction, EmailTriageObservation, TaskType
from server.environment import EmailTriageEnvironment
from server.graders import grade_episode
from server.scenarios.generator import TASK_CONFIG

logger = logging.getLogger(__name__)

# Constants
MAX_SESSIONS = 100  # Limit concurrent sessions
SESSION_TTL_SECONDS = 3600  # 1 hour session timeout
SESSION_COOKIE_NAME = "email_triage_session"


# Request/Response models
class ResetRequest(BaseModel):
    """Request body for /reset endpoint."""
    task_type: Optional[TaskType] = None
    seed: Optional[int] = Field(default=None, ge=0, le=2**31-1)


class StepRequest(BaseModel):
    """Request body for /step endpoint."""
    email_id: str = Field(..., min_length=1, max_length=100)
    priority: Optional[str] = None
    category: Optional[str] = None
    disposition: Optional[str] = None
    response_draft: Optional[str] = Field(default=None, max_length=10000)
    ranking: Optional[List[str]] = None


class StepResponse(BaseModel):
    """Response from /step endpoint."""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    """Response from /state endpoint."""
    episode_id: str
    step_count: int
    cumulative_reward: float
    last_step_reward: float
    emails_processed: int
    emails_remaining: int
    task_type: Optional[str] = None
    done: bool


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: str
    version: str
    timestamp: str
    active_sessions: int = 0


# Create FastAPI app
app = FastAPI(
    title="Email Triage OpenEnv",
    description="OpenEnv environment for training AI agents to prioritize and manage emails",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restricted in HF Spaces by infrastructure
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread-safe session management
_environments: Dict[str, Tuple[EmailTriageEnvironment, float]] = {}  # session_id -> (env, last_access_time)
_env_lock = threading.Lock()


def cleanup_expired_sessions():
    """Remove sessions that have exceeded TTL."""
    current_time = time.time()
    with _env_lock:
        expired = [
            sid for sid, (_, last_access) in _environments.items()
            if current_time - last_access > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del _environments[sid]
            logger.info(f"Cleaned up expired session: {sid}")


def get_or_create_env(session_id: str = "default") -> EmailTriageEnvironment:
    """Get or create environment for session (thread-safe)."""
    cleanup_expired_sessions()  # Opportunistic cleanup
    
    with _env_lock:
        if session_id in _environments:
            env, _ = _environments[session_id]
            _environments[session_id] = (env, time.time())  # Update access time
            return env
        
        # Check session limit
        if len(_environments) >= MAX_SESSIONS:
            # Remove oldest session
            oldest_sid = min(_environments.keys(), key=lambda s: _environments[s][1])
            del _environments[oldest_sid]
            logger.warning(f"Session limit reached, removed oldest session: {oldest_sid}")
        
        env = EmailTriageEnvironment()
        _environments[session_id] = (env, time.time())
        return env


def get_existing_env(session_id: Optional[str]) -> Optional[EmailTriageEnvironment]:
    """Return an existing environment for a session, if present."""
    if not session_id:
        return None

    cleanup_expired_sessions()

    with _env_lock:
        if session_id not in _environments:
            return None
        env, _ = _environments[session_id]
        _environments[session_id] = (env, time.time())
        return env


def resolve_session_id(request: Request) -> Optional[str]:
    """Resolve session ID from header or cookie."""
    return request.headers.get("x-session-id") or request.cookies.get(SESSION_COOKIE_NAME)


def build_empty_state() -> StateResponse:
    """Return the zero-state payload for users without an active session."""
    return StateResponse(
        episode_id="",
        step_count=0,
        cumulative_reward=0.0,
        last_step_reward=0.0,
        emails_processed=0,
        emails_remaining=0,
        task_type=None,
        done=True,
    )


@app.get("/", response_class=HTMLResponse)
async def home():
    """Human-friendly landing page for the public Space root."""
    tasks_markup = "".join(
        f"""
        <article class="task-card">
          <div class="task-meta">
            <span class="pill">{config["difficulty"]}</span>
            <span class="count">{config["email_count"]} emails</span>
          </div>
          <h3>{task_name}</h3>
          <p>{config["description"]}</p>
        </article>
        """
        for task_name, config in TASK_CONFIG.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Email Triage OpenEnv</title>
    <style>
      :root {{
        color-scheme: dark;
        --bg: #08111f;
        --panel: rgba(10, 20, 38, 0.88);
        --panel-strong: rgba(15, 28, 50, 0.98);
        --border: rgba(131, 170, 255, 0.22);
        --text: #eef3ff;
        --muted: #aebcd8;
        --accent: #61d3ff;
        --accent-2: #8ef7c0;
        --shadow: 0 28px 80px rgba(0, 0, 0, 0.36);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        color: var(--text);
        background:
          radial-gradient(circle at top left, rgba(97, 211, 255, 0.18), transparent 30%),
          radial-gradient(circle at top right, rgba(142, 247, 192, 0.14), transparent 28%),
          linear-gradient(160deg, #06101d 0%, #08111f 48%, #0d1730 100%);
        min-height: 100vh;
      }}
      .shell {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 48px 24px 56px;
      }}
      .hero {{
        display: grid;
        gap: 22px;
        grid-template-columns: 1.5fr 1fr;
        align-items: stretch;
      }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 24px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(14px);
      }}
      .hero-copy {{
        padding: 30px;
      }}
      .eyebrow {{
        display: inline-flex;
        gap: 10px;
        align-items: center;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(97, 211, 255, 0.12);
        color: var(--accent);
        font-size: 13px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }}
      h1 {{
        margin: 18px 0 12px;
        font-size: clamp(2.2rem, 5vw, 4rem);
        line-height: 0.95;
      }}
      .lede {{
        margin: 0;
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 56ch;
      }}
      .hero-actions {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 22px;
      }}
      .button {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 12px 16px;
        border-radius: 14px;
        text-decoration: none;
        color: var(--text);
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.04);
      }}
      .button.primary {{
        background: linear-gradient(135deg, rgba(97, 211, 255, 0.22), rgba(142, 247, 192, 0.18));
        border-color: rgba(142, 247, 192, 0.28);
      }}
      .status-card {{
        padding: 24px;
        display: grid;
        gap: 18px;
        background: var(--panel-strong);
      }}
      .status-row {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: center;
      }}
      .status-row strong {{
        color: var(--muted);
        font-weight: 600;
      }}
      .status-value {{
        font-family: "IBM Plex Mono", monospace;
        color: var(--accent-2);
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 18px;
        margin-top: 20px;
      }}
      .task-card, .endpoint-card {{
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 20px;
      }}
      .task-meta {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 12px;
        align-items: center;
      }}
      .pill {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(97, 211, 255, 0.12);
        color: var(--accent);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }}
      .count {{
        color: var(--muted);
        font-size: 13px;
      }}
      h2 {{
        margin: 30px 0 14px;
        font-size: 1.3rem;
      }}
      h3 {{
        margin: 0 0 10px;
        font-size: 1.1rem;
      }}
      p {{
        margin: 0;
        color: var(--muted);
        line-height: 1.65;
      }}
      pre {{
        margin: 0;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-word;
        background: #020815;
        border: 1px solid rgba(131, 170, 255, 0.18);
        border-radius: 16px;
        padding: 18px;
        color: #d6e6ff;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.92rem;
        line-height: 1.6;
      }}
      .two-up {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 18px;
        margin-top: 20px;
      }}
      .note {{
        margin-top: 18px;
        padding: 16px 18px;
        border-radius: 16px;
        border: 1px solid rgba(142, 247, 192, 0.24);
        background: rgba(142, 247, 192, 0.08);
        color: #d9ffef;
      }}
      @media (max-width: 900px) {{
        .hero, .grid, .two-up {{
          grid-template-columns: 1fr;
        }}
        .shell {{
          padding: 28px 16px 36px;
        }}
        .hero-copy, .status-card {{
          padding: 22px;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <div class="panel hero-copy">
          <div class="eyebrow">OpenEnv Benchmark · Email Triage</div>
          <h1>Email Triage OpenEnv</h1>
          <p class="lede">
            A thread-aware inbox management environment for agent evaluation. The API supports
            classification, ranking, and full inbox triage with response-budget constraints,
            duplicate handling, and deterministic graders.
          </p>
          <div class="hero-actions">
            <a class="button primary" href="/health">Health</a>
            <a class="button" href="/tasks">Tasks</a>
            <a class="button" href="https://huggingface.co/spaces/abhinavgautam01/my-env" target="_blank" rel="noreferrer">Space Repo</a>
          </div>
        </div>
        <aside class="panel status-card">
          <div class="status-row"><strong>Status</strong><span class="status-value">healthy</span></div>
          <div class="status-row"><strong>Version</strong><span class="status-value">0.1.0</span></div>
          <div class="status-row"><strong>SDK</strong><span class="status-value">docker</span></div>
          <div class="status-row"><strong>App Port</strong><span class="status-value">8000</span></div>
          <div class="status-row"><strong>Sessions</strong><span class="status-value">cookie-backed</span></div>
          <div class="note">
            Public users are isolated by per-client session cookies. Call <code>/reset</code> first,
            then reuse the same client for <code>/step</code> and <code>/state</code>.
          </div>
        </aside>
      </section>

      <h2>Tasks</h2>
      <section class="grid">{tasks_markup}</section>

      <h2>Endpoints</h2>
      <section class="grid">
        <article class="endpoint-card"><h3>GET /health</h3><p>Runtime heartbeat and active session count.</p></article>
        <article class="endpoint-card"><h3>GET /tasks</h3><p>Available benchmark tasks with difficulty and email count.</p></article>
        <article class="endpoint-card"><h3>GET /state</h3><p>Current episode state for your session cookie.</p></article>
        <article class="endpoint-card"><h3>POST /reset</h3><p>Start a new episode and attach a session cookie.</p></article>
        <article class="endpoint-card"><h3>POST /step</h3><p>Apply one action and receive observation, reward, and grade details.</p></article>
        <article class="endpoint-card"><h3>openenv.yaml</h3><p>Typed action and observation contract for validation.</p></article>
      </section>

      <h2>Quick Start</h2>
      <section class="two-up">
        <pre>curl -X POST /reset \\
  -H "Content-Type: application/json" \\
  -d '{{"task_type":"ranking","seed":42}}'</pre>
        <pre>curl -X POST /step \\
  -H "Content-Type: application/json" \\
  -d '{{"email_id":"e1","ranking":["e1","e2","e3","e4","e5","e6","e7","e8"]}}'</pre>
      </section>
    </main>
  </body>
</html>"""


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.now().isoformat(),
        active_sessions=len(_environments),
    )


@app.post("/reset")
async def reset(request: Request, reset_request: Optional[ResetRequest] = None):
    """
    Reset environment and return initial observation.
    
    Args:
        task_type: Task type (classification/ranking/full_triage)
        seed: Random seed for reproducibility
        
    Returns:
        Initial observation with emails to process
    """
    session_id = resolve_session_id(request)
    if not session_id:
        session_id = f"sess_{uuid4().hex}"

    env = get_or_create_env(session_id)
    
    task_type = reset_request.task_type if reset_request else None
    seed = reset_request.seed if reset_request else None
    
    observation = env.reset(task_type=task_type, seed=seed)

    response = JSONResponse(
        content={
            "observation": observation.model_dump(mode="json"),
            "reward": 0.0,
            "done": False,
            "info": {
                "episode_id": env.state.episode_id,
                "task_type": observation.task_type,
                "session_id": session_id,
            }
        }
    )
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=True,
    )
    return response


@app.post("/step", response_model=StepResponse)
async def step(step_request: StepRequest, request: Request):
    """
    Execute action and return result.
    
    Args:
        email_id: ID of email to act on
        priority: Priority assignment (HIGH/MEDIUM/LOW)
        category: Category (URGENT/ACTION_REQUIRED/INFO/SPAM/PERSONAL)
        disposition: Action to take (RESPOND/DELEGATE/ARCHIVE/DEFER)
        response_draft: Draft response text
        
    Returns:
        Updated observation, reward, done flag, and info
    """
    session_id = resolve_session_id(request)
    env = get_existing_env(session_id)
    
    if env is None or env.state.episode_id == "":
        raise HTTPException(
            status_code=400,
            detail="No episode in progress. Call /reset first."
        )
    
    # Create action from request
    action = EmailTriageAction(
        email_id=step_request.email_id,
        priority=step_request.priority,
        category=step_request.category,
        disposition=step_request.disposition,
        response_draft=step_request.response_draft,
        ranking=step_request.ranking,
    )
    
    try:
        observation = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Get grading info if episode is done
    info = {
        "last_action_result": observation.last_action_result,
        "step_count": env.state.step_count,
    }
    
    if observation.done:
        # Calculate final grade
        scenario = env.get_ground_truth()
        actions = env.get_actions_taken()
        if scenario and actions:
            result = grade_episode(scenario, actions)
            info["final_score"] = result.score
            info["passed"] = result.passed
            info["grade_details"] = result.details
    
    return StepResponse(
        observation=observation.model_dump(mode="json"),
        reward=observation.reward,
        done=observation.done,
        info=info,
    )


@app.get("/state", response_model=StateResponse)
async def get_state(request: Request):
    """
    Get current environment state.
    
    Returns:
        Current episode state information
    """
    session_id = resolve_session_id(request)
    env = get_existing_env(session_id)
    
    if env is None or env.state.episode_id == "":
        return build_empty_state()
    
    scenario = env.get_ground_truth()
    
    return StateResponse(
        episode_id=env.state.episode_id,
        step_count=env.state.step_count,
        cumulative_reward=env._cumulative_reward,
        last_step_reward=env._last_step_reward,
        emails_processed=len(env._processed_emails),
        emails_remaining=len(scenario.emails) - len(env._processed_emails) if scenario else 0,
        task_type=scenario.task_type if scenario else None,
        done=env._done,
    )


@app.get("/tasks")
async def list_tasks():
    """
    List available tasks with descriptions.
    
    Returns:
        List of task definitions
    """
    return {
        "tasks": [
            {
                "name": task_name,
                "description": config["description"],
                "difficulty": config["difficulty"],
                "email_count": config["email_count"],
            }
            for task_name, config in TASK_CONFIG.items()
        ]
    }


def main():
    """Entry point for the server script."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


# For running directly
if __name__ == "__main__":
    main()
