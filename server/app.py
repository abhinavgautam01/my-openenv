"""
FastAPI application for Email Triage OpenEnv Environment.

Provides HTTP endpoints:
- POST /reset - Reset environment and get initial observation
- POST /step - Execute action and get result
- GET /state - Get current environment state
- GET /health - Health check
"""

import json
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
from uuid import uuid4
import logging

from fastapi import FastAPI, HTTPException, Request
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
          <div class="task-meta"><span class="pill">{config["difficulty"]}</span><span class="count">{config["email_count"]} emails</span></div>
          <h3>{task_name}</h3>
          <p>{config["description"]}</p>
        </article>
        """
        for task_name, config in TASK_CONFIG.items()
    )
    tasks_json = json.dumps(TASK_CONFIG)

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Email Triage OpenEnv</title>
    <style>
      :root {{
        color-scheme: dark;
        --bg: #06070b;
        --bg-soft: #0f1524;
        --panel: rgba(10, 16, 30, 0.9);
        --panel-strong: rgba(16, 22, 40, 0.96);
        --panel-alt: rgba(18, 26, 48, 0.92);
        --border: rgba(122, 150, 255, 0.18);
        --text: #f4f7ff;
        --muted: #9aa8c9;
        --accent: #83a9ff;
        --accent-2: #67ebc7;
        --accent-3: #ffd166;
        --danger: #ff7b93;
        --ok: #5de1b8;
        --warning: #ffbf69;
        --shadow: 0 30px 80px rgba(0, 0, 0, 0.4);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        color: var(--text);
        background:
          radial-gradient(circle at 10% 0%, rgba(131, 169, 255, 0.16), transparent 26%),
          radial-gradient(circle at 92% 8%, rgba(103, 235, 199, 0.1), transparent 22%),
          radial-gradient(circle at 50% 100%, rgba(255, 209, 102, 0.08), transparent 22%),
          linear-gradient(180deg, #05070b 0%, #09101d 48%, #10192d 100%);
        min-height: 100vh;
      }}
      .shell {{
        max-width: 1260px;
        margin: 0 auto;
        padding: 32px 24px 56px;
      }}
      .topbar {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: center;
        margin-bottom: 18px;
      }}
      .brand {{
        display: flex;
        align-items: center;
        gap: 12px;
      }}
      .mark {{
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background:
          linear-gradient(145deg, rgba(128, 169, 255, 0.24), rgba(89, 241, 200, 0.16)),
          rgba(255, 255, 255, 0.04);
        border: 1px solid var(--border);
        position: relative;
        box-shadow: var(--shadow);
      }}
      .mark::before,
      .mark::after {{
        content: "";
        position: absolute;
        border-radius: 999px;
      }}
      .mark::before {{
        width: 20px;
        height: 4px;
        background: var(--accent);
        left: 10px;
        top: 11px;
        box-shadow: 0 8px 0 var(--accent), 0 16px 0 var(--accent-2);
      }}
      .mark::after {{
        width: 4px;
        height: 20px;
        background: var(--accent-3);
        right: 11px;
        top: 11px;
      }}
      .brand-copy strong {{
        display: block;
        font-size: 0.98rem;
        letter-spacing: 0.02em;
      }}
      .brand-copy span {{
        color: var(--muted);
        font-size: 0.88rem;
      }}
      .nav-links {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }}
      .nav-links a,
      .hero-actions a,
      button,
      select,
      textarea {{
        font: inherit;
      }}
      .nav-links a,
      .hero-actions a,
      .button {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 11px 14px;
        border-radius: 14px;
        text-decoration: none;
        color: var(--text);
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.04);
      }}
      .nav-links a.primary,
      .hero-actions a.primary {{
        background: linear-gradient(135deg, rgba(128, 169, 255, 0.20), rgba(89, 241, 200, 0.14));
      }}
      .metrics {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 20px;
      }}
      .metric {{
        padding: 14px 16px;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.035);
        box-shadow: var(--shadow);
      }}
      .metric label {{
        display: block;
        margin-bottom: 8px;
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .metric span {{
        font-family: "IBM Plex Mono", monospace;
        color: var(--accent-2);
        font-size: 0.98rem;
      }}
      .hero {{
        display: grid;
        gap: 22px;
        grid-template-columns: 1.35fr 0.95fr;
        align-items: start;
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
        background: rgba(128, 169, 255, 0.12);
        color: var(--accent);
        font-size: 13px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }}
      h1 {{
        margin: 18px 0 12px;
        font-size: clamp(2.5rem, 5.5vw, 4.4rem);
        line-height: 0.92;
        max-width: 10ch;
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
      .hero-actions a.primary {{
        background: linear-gradient(135deg, rgba(128, 169, 255, 0.20), rgba(255, 200, 87, 0.16));
        border-color: rgba(255, 200, 87, 0.20);
      }}
      .aside-stack {{
        display: grid;
        gap: 18px;
      }}
      .status-card, .request-card {{
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
      .request-card {{
        background:
          linear-gradient(180deg, rgba(128, 169, 255, 0.08), rgba(255, 255, 255, 0.02)),
          var(--panel-alt);
      }}
      .request-card h3 {{
        margin-bottom: 6px;
      }}
      .request-card p {{
        margin-bottom: 10px;
      }}
      .request-card code {{
        color: var(--accent-3);
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 18px;
        margin-top: 20px;
      }}
      .task-card, .endpoint-card, .state-card, .demo-card, .result-card, .email-card {{
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
        background: rgba(128, 169, 255, 0.12);
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
        background: #060a16;
        border: 1px solid rgba(126, 146, 255, 0.18);
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
      .workspace {{
        display: grid;
        grid-template-columns: 0.92fr 1.08fr;
        gap: 18px;
        margin-top: 22px;
      }}
      .demo-card,
      .state-card,
      .result-card {{
        background: var(--panel-strong);
      }}
      .controls {{
        display: grid;
        gap: 14px;
      }}
      .field {{
        display: grid;
        gap: 8px;
      }}
      .field label,
      .meta-label {{
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      select,
      textarea {{
        width: 100%;
        border-radius: 14px;
        border: 1px solid rgba(126, 146, 255, 0.2);
        background: rgba(4, 8, 18, 0.9);
        color: var(--text);
        padding: 12px 14px;
        outline: none;
      }}
      textarea {{
        min-height: 112px;
        resize: vertical;
        font-family: "IBM Plex Mono", monospace;
        line-height: 1.5;
      }}
      .button-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }}
      button,
      .button {{
        cursor: pointer;
        border-radius: 14px;
        border: 1px solid var(--border);
        padding: 11px 14px;
        color: var(--text);
        background: rgba(255, 255, 255, 0.04);
        text-decoration: none;
      }}
      button.primary {{
        background: linear-gradient(135deg, rgba(128, 169, 255, 0.2), rgba(89, 241, 200, 0.14));
      }}
      button.warn {{
        background: linear-gradient(135deg, rgba(255, 209, 102, 0.15), rgba(255, 123, 147, 0.08));
      }}
      .explainer {{
        margin-top: 16px;
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid rgba(103, 235, 199, 0.18);
        background: rgba(103, 235, 199, 0.06);
        color: #c9fff0;
      }}
      .state-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }}
      .state-chip {{
        border-radius: 14px;
        padding: 12px 14px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.025);
      }}
      .state-chip strong {{
        display: block;
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .state-chip span {{
        font-family: "IBM Plex Mono", monospace;
      }}
      .email-list {{
        display: grid;
        gap: 12px;
      }}
      .email-card {{
        background: rgba(255, 255, 255, 0.03);
      }}
      .email-head {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: start;
        margin-bottom: 10px;
      }}
      .email-head h3 {{
        margin: 0;
        font-size: 1rem;
      }}
      .email-sub {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 10px;
      }}
      .badge {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 5px 10px;
        background: rgba(128, 169, 255, 0.12);
        color: var(--accent);
        font-size: 12px;
      }}
      .email-body {{
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.6;
      }}
      .mono {{
        font-family: "IBM Plex Mono", monospace;
      }}
      .empty {{
        color: var(--muted);
        padding: 18px;
        border: 1px dashed rgba(126, 146, 255, 0.22);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.02);
      }}
      .result-top {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        margin-bottom: 14px;
      }}
      .result-value {{
        font-size: 1.6rem;
        font-weight: 700;
      }}
      .good {{ color: var(--ok); }}
      .bad {{ color: var(--danger); }}
      .neutral {{ color: var(--accent-3); }}
      .json-box {{
        margin-top: 14px;
      }}
      .json-box pre {{
        min-height: 160px;
      }}
      .hidden {{
        display: none;
      }}
      .note {{
        margin-top: 18px;
        padding: 16px 18px;
        border-radius: 16px;
        border: 1px solid rgba(255, 122, 144, 0.22);
        background: rgba(255, 122, 144, 0.08);
        color: #ffd4dc;
      }}
      .section-head {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: end;
        margin-top: 32px;
      }}
      .section-head p {{
        max-width: 56ch;
      }}
      .footer-note {{
        margin-top: 22px;
        color: var(--muted);
        font-size: 0.92rem;
      }}
      @media (max-width: 900px) {{
        .topbar,
        .metrics,
        .hero, .grid, .two-up, .workspace, .state-grid {{
          grid-template-columns: 1fr;
        }}
        .topbar {{
          align-items: flex-start;
          flex-direction: column;
        }}
        .shell {{
          padding: 28px 16px 36px;
        }}
        .hero-copy, .status-card, .request-card {{
          padding: 22px;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="shell">
      <header class="topbar">
        <div class="brand">
          <div class="mark" aria-hidden="true"></div>
          <div class="brand-copy">
            <strong>Email Triage OpenEnv</strong>
            <span>Docs-first benchmark runtime</span>
          </div>
        </div>
        <nav class="nav-links">
          <a class="primary" href="/docs">Open API Docs</a>
          <a href="/tasks">Task Catalog</a>
          <a href="/health">Health</a>
        </nav>
      </header>

      <section class="metrics">
        <div class="metric"><label>Mode</label><span>Multi-task benchmark</span></div>
        <div class="metric"><label>Hard Mode</label><span>25 emails / 8 replies</span></div>
        <div class="metric"><label>Transport</label><span>Cookie or x-session-id</span></div>
        <div class="metric"><label>Grading</label><span>Deterministic, 0.0-1.0</span></div>
      </section>

      <section class="hero">
        <div class="panel hero-copy">
          <div class="eyebrow">Dark API Portal · OpenEnv</div>
          <h1>Email Triage OpenEnv</h1>
          <p class="lede">
            A thread-aware benchmark for inbox decision-making under real workplace pressure. It combines incident
            response, executive coordination, budget approvals, duplicate threads, and reply-budget tradeoffs
            behind a single OpenEnv-compatible API.
          </p>
          <div class="hero-actions">
            <a class="primary" href="/docs">Launch Docs</a>
            <a href="/tasks">Browse Tasks</a>
            <a href="https://huggingface.co/spaces/abhinavgautam01/my-env" target="_blank" rel="noreferrer">Space Repo</a>
          </div>
          <p class="footer-note">Use <code>/docs</code> for interactive API exploration, or call <code>/reset</code>, <code>/step</code>, and <code>/state</code> directly.</p>
        </div>
        <aside class="aside-stack">
          <section class="panel status-card">
            <div class="status-row"><strong>Status</strong><span class="status-value">healthy</span></div>
            <div class="status-row"><strong>Version</strong><span class="status-value">0.1.0</span></div>
            <div class="status-row"><strong>SDK</strong><span class="status-value">docker / FastAPI</span></div>
            <div class="status-row"><strong>Sessions</strong><span class="status-value">isolated per client</span></div>
            <div class="status-row"><strong>Spec</strong><span class="status-value">OpenEnv</span></div>
          </section>
          <section class="panel request-card">
            <h3>Session Model</h3>
            <p>Start with <code>/reset</code>. Browsers reuse the secure cookie automatically. Programmatic clients can also pass the returned <code>session_id</code> as <code>x-session-id</code>.</p>
            <pre>POST /reset
GET  /state
POST /step</pre>
            <div class="note">Public traffic is isolated. One client cannot read or mutate another client’s episode state. This homepage only explores the environment. It does not call an LLM or depend on any inference provider.</div>
          </section>
        </aside>
      </section>

      <div class="section-head">
        <div>
          <h2>Manual Explorer</h2>
          <p>Play all three tasks directly from the browser. Reset an episode, inspect the observation, submit one action, and inspect reward, done state, and grading details after each step.</p>
        </div>
      </div>
      <section class="workspace">
        <div class="demo-card">
          <div class="controls">
            <div class="field">
              <label for="task-select">Task</label>
              <select id="task-select">
                <option value="classification">classification</option>
                <option value="ranking">ranking</option>
                <option value="full_triage">full_triage</option>
              </select>
            </div>
            <div class="field">
              <label for="seed-input">Seed</label>
              <select id="seed-input">
                <option value="42">42</option>
                <option value="7">7</option>
                <option value="123">123</option>
                <option value="2026">2026</option>
              </select>
            </div>
            <div class="button-row">
              <button class="primary" id="reset-btn" type="button">Reset Task</button>
              <button id="state-btn" type="button">Refresh State</button>
            </div>
            <div class="explainer">
              Browser actions call your live environment only. No agent is invoked here. Use this to inspect task dynamics, edge cases, and session behavior manually.
            </div>
            <div class="field">
              <label for="email-select">Email Target</label>
              <select id="email-select"></select>
            </div>
            <div class="field" id="classification-field">
              <label for="classification-category">Category</label>
              <select id="classification-category">
                <option value="URGENT">URGENT</option>
                <option value="ACTION_REQUIRED">ACTION_REQUIRED</option>
                <option value="INFO">INFO</option>
                <option value="SPAM">SPAM</option>
                <option value="PERSONAL">PERSONAL</option>
              </select>
            </div>
            <div class="field hidden" id="ranking-field">
              <label for="ranking-order">Ranking Order</label>
              <textarea id="ranking-order" placeholder="One email id per line"></textarea>
            </div>
            <div class="field hidden" id="priority-field">
              <label for="priority-select">Priority</label>
              <select id="priority-select">
                <option value="HIGH">HIGH</option>
                <option value="MEDIUM">MEDIUM</option>
                <option value="LOW">LOW</option>
              </select>
            </div>
            <div class="field hidden" id="triage-category-field">
              <label for="triage-category">Category</label>
              <select id="triage-category">
                <option value="URGENT">URGENT</option>
                <option value="ACTION_REQUIRED">ACTION_REQUIRED</option>
                <option value="INFO">INFO</option>
                <option value="SPAM">SPAM</option>
                <option value="PERSONAL">PERSONAL</option>
              </select>
            </div>
            <div class="field hidden" id="disposition-field">
              <label for="disposition-select">Disposition</label>
              <select id="disposition-select">
                <option value="RESPOND">RESPOND</option>
                <option value="DELEGATE">DELEGATE</option>
                <option value="ARCHIVE">ARCHIVE</option>
                <option value="DEFER">DEFER</option>
              </select>
            </div>
            <div class="field hidden" id="response-field">
              <label for="response-draft">Response Draft</label>
              <textarea id="response-draft" placeholder="Optional response draft for RESPOND actions"></textarea>
            </div>
            <div class="button-row">
              <button class="primary" id="step-btn" type="button">Submit Step</button>
              <button class="warn" id="prefill-btn" type="button">Prefill Ranking</button>
            </div>
          </div>
        </div>
        <div class="state-card">
          <div class="section-head">
            <div>
              <h2>Live Session</h2>
              <p>Observation and compact episode state for the current browser session.</p>
            </div>
          </div>
          <div class="state-grid">
            <div class="state-chip"><strong>Episode</strong><span id="state-episode" class="mono">none</span></div>
            <div class="state-chip"><strong>Task</strong><span id="state-task" class="mono">none</span></div>
            <div class="state-chip"><strong>Processed</strong><span id="state-processed" class="mono">0</span></div>
            <div class="state-chip"><strong>Remaining</strong><span id="state-remaining" class="mono">0</span></div>
            <div class="state-chip"><strong>Cumulative Reward</strong><span id="state-cumulative" class="mono">0.00</span></div>
            <div class="state-chip"><strong>Done</strong><span id="state-done" class="mono">true</span></div>
          </div>
          <div class="section-head">
            <div>
              <h2>Observation Emails</h2>
              <p>The current observation is rendered below so you can inspect inbox state before each step.</p>
            </div>
          </div>
          <div id="email-list" class="email-list">
            <div class="empty">Reset a task to load observation emails.</div>
          </div>
        </div>
      </section>

      <section class="result-card" style="margin-top: 18px;">
        <div class="section-head">
          <div>
            <h2>Step Result</h2>
            <p>Reward, completion state, and raw grading details from the last action.</p>
          </div>
        </div>
        <div class="result-top">
          <span class="result-value neutral" id="result-reward">0.00</span>
          <span class="badge" id="result-done">done: true</span>
          <span class="badge" id="result-action">last_action_result: none</span>
        </div>
        <div class="json-box">
          <pre id="result-json">{{"reward": 0.0, "done": true}}</pre>
        </div>
      </section>

      <div class="section-head">
        <div>
          <h2>Tasks</h2>
          <p>Three deterministic tasks cover classification, ranking, and high-pressure inbox triage with thread-level dependencies.</p>
        </div>
      </div>
      <section class="grid">{tasks_markup}</section>

      <div class="section-head">
        <div>
          <h2>Endpoints</h2>
          <p>The landing page stays minimal, but the API surface is complete and interactive docs are available at <code>/docs</code>.</p>
        </div>
      </div>
      <section class="grid">
        <article class="endpoint-card"><h3>GET /health</h3><p>Runtime heartbeat and active session count.</p></article>
        <article class="endpoint-card"><h3>GET /tasks</h3><p>Available benchmark tasks with difficulty and email count.</p></article>
        <article class="endpoint-card"><h3>GET /state</h3><p>Current episode state for your session cookie.</p></article>
        <article class="endpoint-card"><h3>POST /reset</h3><p>Start a new episode and attach a session cookie.</p></article>
        <article class="endpoint-card"><h3>POST /step</h3><p>Apply one action and receive observation, reward, and final grade details.</p></article>
        <article class="endpoint-card"><h3>openenv.yaml</h3><p>Typed action and observation contract for validation.</p></article>
      </section>

      <div class="section-head">
        <div>
          <h2>Quick Start</h2>
          <p>Minimal examples for calling the live environment directly from curl or any HTTP client.</p>
        </div>
      </div>
      <section class="two-up">
        <pre>curl -X POST /reset \\
  -H "Content-Type: application/json" \\
  -d '{{"task_type":"ranking","seed":42}}'</pre>
        <pre>curl -X POST /step \\
  -H "Content-Type: application/json" \\
  -d '{{"email_id":"e1","ranking":["e1","e2","e3","e4","e5","e6","e7","e8"]}}'</pre>
      </section>
    </main>
    <script>
      const taskConfig = {tasks_json};
      let currentObservation = null;
      let currentTaskType = "classification";

      const taskSelect = document.getElementById("task-select");
      const seedInput = document.getElementById("seed-input");
      const emailSelect = document.getElementById("email-select");
      const classificationField = document.getElementById("classification-field");
      const rankingField = document.getElementById("ranking-field");
      const priorityField = document.getElementById("priority-field");
      const triageCategoryField = document.getElementById("triage-category-field");
      const dispositionField = document.getElementById("disposition-field");
      const responseField = document.getElementById("response-field");
      const rankingOrder = document.getElementById("ranking-order");
      const responseDraft = document.getElementById("response-draft");

      function setResult(payload) {{
        const reward = Number(payload.reward ?? 0);
        const done = Boolean(payload.done);
        const lastAction = payload.observation?.last_action_result ?? payload.info?.last_action_result ?? "none";
        const rewardEl = document.getElementById("result-reward");
        rewardEl.textContent = reward.toFixed(2);
        rewardEl.className = "result-value " + (reward > 0.75 ? "good" : reward < 0 ? "bad" : "neutral");
        document.getElementById("result-done").textContent = "done: " + String(done);
        document.getElementById("result-action").textContent = "last_action_result: " + lastAction;
        document.getElementById("result-json").textContent = JSON.stringify(payload, null, 2);
      }}

      function renderStatePayload(state) {{
        document.getElementById("state-episode").textContent = state.episode_id || "none";
        document.getElementById("state-task").textContent = state.task_type || "none";
        document.getElementById("state-processed").textContent = String(state.emails_processed ?? 0);
        document.getElementById("state-remaining").textContent = String(state.emails_remaining ?? 0);
        document.getElementById("state-cumulative").textContent = Number(state.cumulative_reward ?? 0).toFixed(2);
        document.getElementById("state-done").textContent = String(Boolean(state.done));
      }}

      function renderEmails(emails) {{
        const list = document.getElementById("email-list");
        if (!emails || emails.length === 0) {{
          list.innerHTML = '<div class="empty">No visible emails remain in the current observation.</div>';
          return;
        }}
        list.innerHTML = emails.map((email) => `
          <article class="email-card">
            <div class="email-head">
              <div>
                <h3>${{email.subject}}</h3>
                <div class="email-sub">
                  <span class="badge mono">${{email.id}}</span>
                  <span class="badge">${{email.sender_importance}}</span>
                  <span class="badge">${{email.sender_name}}</span>
                  <span class="badge">${{email.thread_id || "single"}}</span>
                </div>
              </div>
              <span class="count mono">${{new Date(email.timestamp).toLocaleString()}}</span>
            </div>
            <p class="email-body">${{email.body}}</p>
          </article>
        `).join("");
      }}

      function updateEmailSelector(emails) {{
        emailSelect.innerHTML = "";
        (emails || []).forEach((email) => {{
          const option = document.createElement("option");
          option.value = email.id;
          option.textContent = `${{email.id}} - ${{email.subject}}`;
          emailSelect.appendChild(option);
        }});
      }}

      function syncControls() {{
        currentTaskType = taskSelect.value;
        classificationField.classList.toggle("hidden", currentTaskType !== "classification");
        rankingField.classList.toggle("hidden", currentTaskType !== "ranking");
        priorityField.classList.toggle("hidden", currentTaskType !== "full_triage");
        triageCategoryField.classList.toggle("hidden", currentTaskType !== "full_triage");
        dispositionField.classList.toggle("hidden", currentTaskType !== "full_triage");
        responseField.classList.toggle("hidden", currentTaskType !== "full_triage");
        document.getElementById("prefill-btn").classList.toggle("hidden", currentTaskType !== "ranking");
      }}

      async function refreshState() {{
        const response = await fetch("/state", {{ credentials: "same-origin" }});
        const data = await response.json();
        renderStatePayload(data);
      }}

      function renderObservation(observation) {{
        currentObservation = observation;
        updateEmailSelector(observation.emails || []);
        renderEmails(observation.emails || []);
        if (currentTaskType === "ranking") {{
          rankingOrder.value = (observation.emails || []).map((email) => email.id).join("\\n");
        }}
      }}

      async function resetTask() {{
        syncControls();
        const body = {{
          task_type: taskSelect.value,
          seed: Number(seedInput.value),
        }};
        const response = await fetch("/reset", {{
          method: "POST",
          credentials: "same-origin",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(body),
        }});
        const data = await response.json();
        renderObservation(data.observation);
        setResult(data);
        await refreshState();
      }}

      function prefillRanking() {{
        if (!currentObservation?.emails) {{
          return;
        }}
        rankingOrder.value = currentObservation.emails.map((email) => email.id).join("\\n");
      }}

      async function submitStep() {{
        const payload = {{
          email_id: emailSelect.value || currentObservation?.emails?.[0]?.id || "e1",
        }};

        if (currentTaskType === "classification") {{
          payload.category = document.getElementById("classification-category").value;
        }} else if (currentTaskType === "ranking") {{
          payload.ranking = rankingOrder.value
            .split(/[,\\n]+/)
            .map((item) => item.trim())
            .filter(Boolean);
        }} else {{
          payload.priority = document.getElementById("priority-select").value;
          payload.category = document.getElementById("triage-category").value;
          payload.disposition = document.getElementById("disposition-select").value;
          const draft = responseDraft.value.trim();
          if (draft) {{
            payload.response_draft = draft;
          }}
        }}

        const response = await fetch("/step", {{
          method: "POST",
          credentials: "same-origin",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload),
        }});

        const text = await response.text();
        let data = {{}};
        try {{
          data = JSON.parse(text);
        }} catch (error) {{
          data = {{ detail: text, reward: 0.0, done: false }};
        }}

        if (!response.ok) {{
          setResult(data);
          return;
        }}

        renderObservation(data.observation);
        setResult(data);
        await refreshState();
      }}

      taskSelect.addEventListener("change", syncControls);
      document.getElementById("reset-btn").addEventListener("click", () => void resetTask());
      document.getElementById("state-btn").addEventListener("click", () => void refreshState());
      document.getElementById("step-btn").addEventListener("click", () => void submitStep());
      document.getElementById("prefill-btn").addEventListener("click", prefillRanking);
      document.getElementById("disposition-select").addEventListener("change", (event) => {{
        responseField.classList.toggle("hidden", currentTaskType !== "full_triage" || event.target.value !== "RESPOND");
      }});

      syncControls();
      refreshState();
    </script>
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
        secure=request.url.scheme == "https",
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
