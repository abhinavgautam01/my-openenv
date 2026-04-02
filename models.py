"""
Pydantic models for Email Triage OpenEnv Environment.

Defines typed models for:
- Email: Single email in the inbox
- EmailTriageAction: Agent's action on an email  
- EmailTriageObservation: Environment state observation
"""

from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    # Fallback for local development without openenv-core
    class Action(BaseModel):
        """Base action class."""
        pass
    
    class Observation(BaseModel):
        """Base observation class."""
        done: bool = Field(default=False, description="Whether episode is complete")
        reward: float = Field(default=0.0, description="Reward from the most recent step")


# Type definitions
SenderImportance = Literal["VIP", "KNOWN", "UNKNOWN", "EXTERNAL"]
Priority = Literal["HIGH", "MEDIUM", "LOW"]
Category = Literal["URGENT", "ACTION_REQUIRED", "INFO", "SPAM", "PERSONAL"]
Disposition = Literal["RESPOND", "DELEGATE", "ARCHIVE", "DEFER"]
TaskType = Literal["classification", "ranking", "full_triage"]


class Email(BaseModel):
    """Single email in the inbox."""
    
    id: str = Field(..., description="Unique email identifier")
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Sender email address")
    sender_name: str = Field(..., description="Sender display name")
    sender_importance: SenderImportance = Field(
        ..., 
        description="Sender reputation: VIP (boss/client), KNOWN (colleague), UNKNOWN (first contact), EXTERNAL (outside org)"
    )
    body: str = Field(..., description="Email body text (plain text)")
    timestamp: datetime = Field(..., description="When email was received")
    has_attachment: bool = Field(default=False, description="Whether email has attachments")
    thread_id: Optional[str] = Field(default=None, description="Thread ID if part of conversation")
    is_reply: bool = Field(default=False, description="Whether this is a reply to another email")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "e1",
                "subject": "URGENT: Production server down",
                "sender": "ops-alerts@company.com",
                "sender_name": "Operations Team",
                "sender_importance": "VIP",
                "body": "Alert: Production database server is unreachable. Immediate action required.",
                "timestamp": "2024-01-15T09:00:00Z",
                "has_attachment": False,
                "thread_id": None,
                "is_reply": False
            }
        }
    }


class EmailTriageAction(Action):
    """
    Action taken by the agent on an email.
    
    Different tasks require different fields:
    - classification: category required
    - ranking: ranking required  
    - full_triage: priority, category, disposition required; response_draft optional
    """
    
    email_id: str = Field(
        ..., 
        description="ID of email being acted upon",
        min_length=1,
        max_length=100
    )
    priority: Optional[Priority] = Field(
        default=None, 
        description="Priority assignment: HIGH (urgent), MEDIUM (important), LOW (can wait)"
    )
    category: Optional[Category] = Field(
        default=None,
        description="Category: URGENT (immediate), ACTION_REQUIRED (needs response), INFO (FYI), SPAM, PERSONAL"
    )
    disposition: Optional[Disposition] = Field(
        default=None,
        description="Action: RESPOND (reply), DELEGATE (forward), ARCHIVE (done), DEFER (later)"
    )
    response_draft: Optional[str] = Field(
        default=None,
        description="Draft response text (only if disposition=RESPOND)",
        max_length=10000
    )
    ranking: Optional[List[str]] = Field(
        default=None,
        description="For ranking task: explicit ordering of email IDs by priority (highest to lowest)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "email_id": "e1",
                "priority": "HIGH",
                "category": "URGENT",
                "disposition": "RESPOND",
                "response_draft": "I'll look into this immediately and get back to you within 30 minutes."
            }
        }
    }


class EmailTriageObservation(Observation):
    """
    Observation of the current environment state.
    
    Contains the emails to process and episode metadata.
    """
    
    task_id: str = Field(..., description="Unique identifier for this task instance")
    task_type: TaskType = Field(..., description="Task type: classification, ranking, or full_triage")
    emails: List[Email] = Field(..., description="Emails to process in this episode")
    current_time: datetime = Field(..., description="Simulated current timestamp")
    emails_processed: int = Field(default=0, description="Number of emails already processed")
    emails_remaining: int = Field(..., description="Number of emails left to process")
    time_budget_remaining: Optional[float] = Field(
        default=None, 
        description="Seconds remaining for time-pressured tasks (hard mode only)"
    )
    last_action_result: Optional[str] = Field(
        default=None,
        description="Result of last action: 'correct', 'partial', 'incorrect', or None"
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Cumulative reward across the episode"
    )
    
    # Inherited from Observation
    # done: bool - Whether episode is complete
    # reward: float - Most recent step reward
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "task_abc123",
                "task_type": "ranking",
                "emails": [],
                "current_time": "2024-01-15T09:00:00Z",
                "emails_processed": 0,
                "emails_remaining": 8,
                "time_budget_remaining": None,
                "last_action_result": None,
                "done": False,
                "reward": 0.0,
                "cumulative_reward": 0.0
            }
        }
    }


class GroundTruth(BaseModel):
    """
    Ground truth labels for scenario validation.
    Hidden from agent, used by graders.
    """
    
    email_id: str = Field(..., description="Email ID")
    correct_priority: Priority = Field(..., description="Correct priority assignment")
    correct_category: Category = Field(..., description="Correct category")
    correct_disposition: Disposition = Field(..., description="Correct disposition")
    priority_rank: int = Field(..., description="Position in priority order (1=highest)")
    urgency_score: float = Field(..., ge=0.0, le=1.0, description="Urgency score 0-1")
    needs_response: bool = Field(default=False, description="Whether email needs a response")
    response_context: Optional[str] = Field(
        default=None, 
        description="Context for evaluating response quality"
    )
    expected_response: Optional[str] = Field(
        default=None,
        description="Reference response text for higher-quality reply scoring"
    )
    thread_id: Optional[str] = Field(default=None, description="Thread group identifier for related emails")
    thread_role: Optional[Literal["root", "followup", "duplicate"]] = Field(
        default=None,
        description="Role this email plays within its thread"
    )
    duplicate_of: Optional[str] = Field(
        default=None,
        description="Email ID of the primary message if this email is a duplicate or low-value follow-up"
    )
    recommended_after: Optional[str] = Field(
        default=None,
        description="Email ID that should usually be handled before this one"
    )
    response_value: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How valuable it is to spend one response slot on this email"
    )
    business_impact: Optional[str] = Field(
        default=None,
        description="Business impact category for harder inbox-level tradeoffs"
    )


class ScenarioData(BaseModel):
    """
    Complete scenario with emails and ground truth.
    Used by scenario generator and graders.
    """
    
    scenario_id: str = Field(..., description="Unique scenario identifier")
    task_type: TaskType = Field(..., description="Task type for this scenario")
    seed: int = Field(..., description="Random seed for reproducibility")
    emails: List[Email] = Field(..., description="Emails in this scenario")
    ground_truth: List[GroundTruth] = Field(..., description="Ground truth for each email")
    priority_order: List[str] = Field(..., description="Correct email IDs in priority order")
    difficulty_factors: dict = Field(
        default_factory=dict,
        description="Factors affecting difficulty: ambiguity, sender_mix, etc."
    )
    config: dict = Field(
        default_factory=dict,
        description="Task configuration settings (time_budget, max_responses, etc.)"
    )
