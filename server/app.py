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
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
async def reset(request: ResetRequest = None):
    """
    Reset environment and return initial observation.
    
    Args:
        task_type: Task type (classification/ranking/full_triage)
        seed: Random seed for reproducibility
        
    Returns:
        Initial observation with emails to process
    """
    env = get_or_create_env()
    
    task_type = request.task_type if request else None
    seed = request.seed if request else None
    
    observation = env.reset(task_type=task_type, seed=seed)
    
    return {
        "observation": observation.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {
            "episode_id": env.state.episode_id,
            "task_type": observation.task_type,
        }
    }


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
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
    env = get_or_create_env()
    
    if env.state.episode_id == "":
        raise HTTPException(
            status_code=400,
            detail="No episode in progress. Call /reset first."
        )
    
    # Create action from request
    action = EmailTriageAction(
        email_id=request.email_id,
        priority=request.priority,
        category=request.category,
        disposition=request.disposition,
        response_draft=request.response_draft,
        ranking=request.ranking,  # Include ranking field for explicit ranking task
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
        observation=observation.model_dump(),
        reward=observation.reward,
        done=observation.done,
        info=info,
    )


@app.get("/state", response_model=StateResponse)
async def get_state():
    """
    Get current environment state.
    
    Returns:
        Current episode state information
    """
    env = get_or_create_env()
    
    if env.state.episode_id == "":
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
