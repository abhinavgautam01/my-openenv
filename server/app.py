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
        transition: transform 180ms ease, border-color 180ms ease, background 180ms ease, box-shadow 180ms ease;
      }}
      .task-card:hover, .endpoint-card:hover, .email-card:hover {{
        transform: translateY(-2px);
        border-color: rgba(131, 169, 255, 0.28);
        background: rgba(255, 255, 255, 0.05);
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
      .controls-top {{
        display: grid;
        gap: 14px;
        grid-template-columns: 1.15fr 0.85fr;
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
        border-radius: 18px;
        border: 1px solid rgba(126, 146, 255, 0.22);
        background: linear-gradient(180deg, rgba(8, 12, 24, 0.96), rgba(10, 16, 30, 0.92));
        color: var(--text);
        padding: 14px 16px;
        outline: none;
        transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
      }}
      select {{
        appearance: none;
        -webkit-appearance: none;
        -moz-appearance: none;
        padding-right: 54px;
        background-image:
          linear-gradient(45deg, transparent 50%, rgba(131, 169, 255, 0.92) 50%),
          linear-gradient(135deg, rgba(131, 169, 255, 0.92) 50%, transparent 50%);
        background-position:
          calc(100% - 24px) calc(50% - 2px),
          calc(100% - 18px) calc(50% - 2px);
        background-size: 7px 7px, 7px 7px;
        background-repeat: no-repeat;
      }}
      select:focus,
      textarea:focus {{
        border-color: rgba(131, 169, 255, 0.42);
        box-shadow: 0 0 0 4px rgba(131, 169, 255, 0.08);
      }}
      textarea {{
        min-height: 112px;
        resize: vertical;
        font-family: "IBM Plex Mono", monospace;
        line-height: 1.5;
      }}
      #ranking-order {{
        min-height: 196px;
      }}
      .button-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }}
      .helper {{
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.5;
      }}
      .helper strong {{
        color: var(--text);
      }}
      .session-banner {{
        display: grid;
        gap: 10px;
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid rgba(126, 146, 255, 0.18);
        background: rgba(255, 255, 255, 0.03);
      }}
      .status-pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }}
      .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 10px;
        border-radius: 999px;
        border: 1px solid rgba(126, 146, 255, 0.18);
        background: rgba(255, 255, 255, 0.025);
        color: var(--text);
        font-size: 0.88rem;
      }}
      .status-pill strong {{
        color: var(--muted);
        font-weight: 600;
      }}
      button,
      .button {{
        cursor: pointer;
        border-radius: 16px;
        border: 1px solid var(--border);
        padding: 12px 16px;
        color: var(--text);
        background: rgba(255, 255, 255, 0.04);
        text-decoration: none;
        transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
      }}
      button.primary {{
        background: linear-gradient(135deg, rgba(128, 169, 255, 0.24), rgba(89, 241, 200, 0.16));
        box-shadow: 0 12px 26px rgba(98, 151, 255, 0.14);
      }}
      button.warn {{
        background: linear-gradient(135deg, rgba(255, 209, 102, 0.15), rgba(255, 123, 147, 0.08));
      }}
      button:hover,
      .button:hover {{
        transform: translateY(-1px);
        border-color: rgba(131, 169, 255, 0.34);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
      }}
      button:active,
      .button:active {{
        transform: translateY(0);
      }}
      button:disabled {{
        cursor: not-allowed;
        opacity: 0.48;
        filter: saturate(0.7);
      }}
      .button-row button.primary.loading {{
        position: relative;
        color: rgba(244, 247, 255, 0.92);
      }}
      .button-row button.primary.loading::after {{
        content: "";
        width: 14px;
        height: 14px;
        margin-left: 10px;
        border-radius: 999px;
        border: 2px solid rgba(255, 255, 255, 0.28);
        border-top-color: rgba(255, 255, 255, 0.94);
        display: inline-block;
        animation: spin 700ms linear infinite;
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
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.025));
        padding: 22px;
      }}
      .email-head {{
        display: flex;
        justify-content: space-between;
        gap: 18px;
        align-items: start;
        margin-bottom: 14px;
      }}
      .email-head h3 {{
        margin: 0;
        font-size: 1.14rem;
        line-height: 1.15;
      }}
      .email-sub {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
      }}
      .email-meta {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 124px;
        text-align: right;
        padding: 8px 10px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(126, 146, 255, 0.14);
        color: var(--muted);
        line-height: 1.3;
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
      @keyframes spin {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
      }}
      @media (max-width: 900px) {{
        .topbar,
        .metrics,
        .hero, .grid, .two-up, .workspace, .state-grid, .controls-top {{
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
        .email-head {{
          flex-direction: column;
        }}
        .email-meta {{
          min-width: 0;
          text-align: left;
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
            <div class="controls-top">
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
            </div>
            <div class="button-row">
              <button class="primary" id="reset-btn" type="button">Load Fresh Task</button>
              <button id="state-btn" type="button">Refresh State</button>
            </div>
            <div class="explainer">
              Browser actions call your live environment only. No agent is invoked here. Use this to inspect task dynamics, edge cases, and session behavior manually.
            </div>
            <div class="session-banner">
              <div class="status-pills">
                <span class="status-pill"><strong>Session</strong><span id="session-id" class="mono">none</span></span>
                <span class="status-pill"><strong>Transport</strong><span id="session-transport">cookie + x-session-id</span></span>
                <span class="status-pill"><strong>Auto Load</strong><span id="auto-load-state">on task or seed change</span></span>
              </div>
              <div class="helper" id="session-helper">Choose a task and seed, then a fresh episode loads automatically. Ranking submit stays disabled until all visible ids are present exactly once.</div>
            </div>
            <div class="field" id="email-target-field">
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
              <div class="helper" id="ranking-helper">Ranking requires every visible email ID exactly once. Use <strong>Prefill Ranking</strong>, then reorder the lines.</div>
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
            <div class="empty">A fresh task loads automatically when the page opens or when task and seed change.</div>
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
      const emailTargetField = document.getElementById("email-target-field");
      const classificationField = document.getElementById("classification-field");
      const rankingField = document.getElementById("ranking-field");
      const priorityField = document.getElementById("priority-field");
      const triageCategoryField = document.getElementById("triage-category-field");
      const dispositionField = document.getElementById("disposition-field");
      const responseField = document.getElementById("response-field");
      const rankingOrder = document.getElementById("ranking-order");
      const responseDraft = document.getElementById("response-draft");
      const rankingHelper = document.getElementById("ranking-helper");
      const stepButton = document.getElementById("step-btn");
      const resetButton = document.getElementById("reset-btn");
      const stateButton = document.getElementById("state-btn");
      const sessionIdEl = document.getElementById("session-id");
      const sessionHelper = document.getElementById("session-helper");
      let currentSessionId = null;
      let resetSerial = 0;

      function buildHeaders() {{
        const headers = {{ "Content-Type": "application/json" }};
        if (currentSessionId) {{
          headers["x-session-id"] = currentSessionId;
        }}
        return headers;
      }}

      function setResult(payload) {{
        const reward = Number(payload.reward ?? 0);
        const done = Boolean(payload.done);
        const lastAction = payload.observation?.last_action_result ?? payload.info?.last_action_result ?? payload.detail ?? "none";
        const rewardEl = document.getElementById("result-reward");
        rewardEl.textContent = reward.toFixed(2);
        rewardEl.className = "result-value " + (reward > 0.75 ? "good" : reward < 0 ? "bad" : "neutral");
        document.getElementById("result-done").textContent = "done: " + String(done);
        document.getElementById("result-action").textContent = "last_action_result: " + lastAction;
        document.getElementById("result-json").textContent = JSON.stringify(payload, null, 2);
      }}

      function updateSessionBanner() {{
        sessionIdEl.textContent = currentSessionId || "none";
        if (!currentSessionId) {{
          sessionHelper.textContent = "Choose a task and seed, then a fresh episode loads automatically. Ranking submit stays disabled until all visible ids are present exactly once.";
          return;
        }}
        if (currentObservation?.done) {{
          sessionHelper.textContent = "Episode complete. Change task or seed, or use Load Fresh Task to start another episode.";
          return;
        }}
        if (currentTaskType === "ranking") {{
          sessionHelper.textContent = "Ranking is a single-shot task. Keep one visible email id per line, with no duplicates or omissions, then submit once.";
          return;
        }}
        sessionHelper.textContent = "Session is active. You can inspect state, submit actions, and watch reward and grade details update after each step.";
      }}

      function setBusy(button, busyLabel) {{
        if (!button) {{
          return;
        }}
        if (busyLabel) {{
          button.dataset.originalLabel = button.textContent;
          button.textContent = busyLabel;
          button.classList.add("loading");
          button.disabled = true;
          return;
        }}
        button.textContent = button.dataset.originalLabel || button.textContent;
        button.classList.remove("loading");
        button.disabled = false;
        updateStepButtonState();
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
              <span class="email-meta mono">${{new Date(email.timestamp).toLocaleString()}}</span>
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
        emailTargetField.classList.toggle("hidden", currentTaskType === "ranking");
        priorityField.classList.toggle("hidden", currentTaskType !== "full_triage");
        triageCategoryField.classList.toggle("hidden", currentTaskType !== "full_triage");
        dispositionField.classList.toggle("hidden", currentTaskType !== "full_triage");
        responseField.classList.toggle("hidden", currentTaskType !== "full_triage");
        document.getElementById("prefill-btn").classList.toggle("hidden", currentTaskType !== "ranking");
        updateSessionBanner();
        updateRankingHelper();
        updateStepButtonState();
      }}

      async function refreshState() {{
        stateButton.disabled = true;
        const response = await fetch("/state", {{ credentials: "same-origin", headers: currentSessionId ? {{ "x-session-id": currentSessionId }} : undefined }});
        const data = await response.json();
        renderStatePayload(data);
        stateButton.disabled = false;
      }}

      function renderObservation(observation) {{
        currentObservation = observation;
        updateEmailSelector(observation.emails || []);
        renderEmails(observation.emails || []);
        if (currentTaskType === "ranking") {{
          rankingOrder.value = (observation.emails || []).map((email) => email.id).join("\\n");
        }}
        updateRankingHelper();
        updateStepButtonState();
      }}

      async function resetTask() {{
        const serial = ++resetSerial;
        syncControls();
        setBusy(resetButton, "Loading Task");
        const body = {{
          task_type: taskSelect.value,
          seed: Number(seedInput.value),
        }};
        const response = await fetch("/reset", {{
          method: "POST",
          credentials: "same-origin",
          headers: buildHeaders(),
          body: JSON.stringify(body),
        }});
        const data = await response.json();
        if (serial !== resetSerial) {{
          setBusy(resetButton, null);
          return;
        }}
        currentSessionId = data.info?.session_id || currentSessionId;
        renderObservation(data.observation);
        setResult(data);
        updateSessionBanner();
        await refreshState();
        setBusy(resetButton, null);
      }}

      function prefillRanking() {{
        if (!currentObservation?.emails) {{
          return;
        }}
        rankingOrder.value = currentObservation.emails.map((email) => email.id).join("\\n");
        updateRankingHelper();
      }}

      function normalizeRankingInput() {{
        return rankingOrder.value
          .split(/[,\\n]+/)
          .map((item) => item.trim())
          .filter(Boolean);
      }}

      function updateRankingHelper() {{
        if (currentTaskType !== "ranking") {{
          rankingHelper.innerHTML = "Ranking helper is only active for the ranking task.";
          return;
        }}
        const expected = currentObservation?.emails?.map((email) => email.id) || [];
        const submitted = normalizeRankingInput();
        const unique = [...new Set(submitted)];
        if (!expected.length) {{
          rankingHelper.innerHTML = 'Load a ranking task to populate all visible email IDs.';
          return;
        }}
        const isExact = submitted.length === expected.length && unique.length === expected.length && expected.every((id) => submitted.includes(id));
        rankingHelper.innerHTML = isExact
          ? `Ranking ready: <strong>${{submitted.length}}</strong> ids captured.`
          : `Ranking requires <strong>${{expected.length}}</strong> ids exactly once. Current input has <strong>${{submitted.length}}</strong> lines and <strong>${{unique.length}}</strong> unique ids.`;
      }}

      function canSubmitCurrentAction() {{
        if (!currentSessionId || !currentObservation || currentObservation.done) {{
          return false;
        }}
        if (currentTaskType !== "ranking") {{
          return true;
        }}
        const expectedIds = currentObservation?.emails?.map((email) => email.id) || [];
        const submitted = normalizeRankingInput();
        const uniqueIds = [...new Set(submitted)];
        return expectedIds.length > 0
          && submitted.length === expectedIds.length
          && uniqueIds.length === expectedIds.length
          && expectedIds.every((id) => submitted.includes(id));
      }}

      function updateStepButtonState() {{
        stepButton.disabled = !canSubmitCurrentAction();
      }}

      async function submitStep() {{
        const payload = {{
          email_id: emailSelect.value || currentObservation?.emails?.[0]?.id || "e1",
        }};

        if (currentTaskType === "classification") {{
          payload.category = document.getElementById("classification-category").value;
        }} else if (currentTaskType === "ranking") {{
          payload.ranking = normalizeRankingInput();
          const expectedIds = currentObservation?.emails?.map((email) => email.id) || [];
          const uniqueIds = [...new Set(payload.ranking)];
          const isExact = payload.ranking.length === expectedIds.length
            && uniqueIds.length === expectedIds.length
            && expectedIds.every((id) => payload.ranking.includes(id));
          if (!isExact) {{
            setResult({{
              reward: -0.05,
              done: false,
              detail: `Ranking must include all ${{expectedIds.length}} visible email IDs exactly once before submit.`,
            }});
            updateRankingHelper();
            updateStepButtonState();
            return;
          }}
          payload.email_id = payload.ranking[0];
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
          headers: buildHeaders(),
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
          if (response.status === 400 && data.detail && String(data.detail).includes("No episode in progress")) {{
            data = {{
              reward: 0.0,
              done: false,
              detail: "No active browser session was found. Reset the task once, then submit again from the same page.",
            }};
          }}
          setResult(data);
          updateSessionBanner();
          updateStepButtonState();
          return;
        }}

        renderObservation(data.observation);
        setResult(data);
        updateRankingHelper();
        updateSessionBanner();
        await refreshState();
      }}

      taskSelect.addEventListener("change", () => void resetTask());
      seedInput.addEventListener("change", () => void resetTask());
      rankingOrder.addEventListener("input", updateRankingHelper);
      rankingOrder.addEventListener("input", updateStepButtonState);
      document.getElementById("reset-btn").addEventListener("click", () => void resetTask());
      document.getElementById("state-btn").addEventListener("click", () => void refreshState());
      document.getElementById("step-btn").addEventListener("click", () => void submitStep());
      document.getElementById("prefill-btn").addEventListener("click", prefillRanking);
      document.getElementById("disposition-select").addEventListener("change", (event) => {{
        responseField.classList.toggle("hidden", currentTaskType !== "full_triage" || event.target.value !== "RESPOND");
      }});

      syncControls();
      refreshState();
      updateRankingHelper();
      void resetTask();
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
