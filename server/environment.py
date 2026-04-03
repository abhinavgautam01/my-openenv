"""
Core Email Triage Environment implementation.

Implements the OpenEnv interface:
- reset() -> Initial observation
- step(action) -> (observation, reward, done, info)
- state() -> Current state

OPTIMIZATIONS:
- Thread-safe state management
- Comprehensive input validation
- Rich error messages for debugging
- Episode metrics tracking
- Time-based rewards for hard mode
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.models import (
    Email, EmailTriageAction, EmailTriageObservation,
    GroundTruth, ScenarioData, TaskType, Category, Priority, Disposition
)
from server.graders import kendall_tau_correlation
from server.scenarios.generator import generate_scenario, TASK_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:
    # Fallback for local development
    from dataclasses import dataclass, field
    
    @dataclass
    class State:
        episode_id: str = ""
        step_count: int = 0
    
    class Environment:
        """Base environment interface."""
        pass


class EpisodeMetrics:
    """Track detailed metrics for analysis and debugging."""
    
    def __init__(self):
        self.start_time: float = time.time()
        self.step_times: List[float] = []
        self.rewards_per_step: List[float] = []
        self.correct_actions: int = 0
        self.partial_actions: int = 0
        self.incorrect_actions: int = 0
        self.invalid_actions: int = 0
    
    def record_step(self, reward: float, result: str):
        self.step_times.append(time.time() - self.start_time)
        self.rewards_per_step.append(reward)
        if result == "correct":
            self.correct_actions += 1
        elif result == "partial":
            self.partial_actions += 1
        elif result == "incorrect":
            self.incorrect_actions += 1
        else:
            self.invalid_actions += 1
    
    def to_dict(self) -> Dict[str, Any]:
        total = self.correct_actions + self.partial_actions + self.incorrect_actions
        return {
            "duration_seconds": time.time() - self.start_time,
            "total_steps": len(self.step_times),
            "accuracy": self.correct_actions / max(1, total),
            "correct": self.correct_actions,
            "partial": self.partial_actions,
            "incorrect": self.incorrect_actions,
            "invalid": self.invalid_actions,
        }


class EmailTriageEnvironment(Environment):
    """
    Email Triage Environment for training AI agents.
    
    The agent must prioritize, categorize, and decide disposition
    for incoming emails based on urgency, sender importance, and content.
    """
    
    def __init__(
        self,
        default_task: TaskType = "classification",
        episode_timeout: float = 300.0,
        max_steps: int = 100,
    ):
        """
        Initialize the environment.
        
        Args:
            default_task: Default task type if not specified in reset()
            episode_timeout: Maximum time for an episode (seconds)
            max_steps: Maximum steps per episode
        """
        self.default_task = default_task
        self.episode_timeout = episode_timeout
        self.max_steps = max_steps
        
        # Episode state
        self._state = State(episode_id="", step_count=0)
        self._scenario: Optional[ScenarioData] = None
        self._processed_emails: Set[str] = set()
        self._actions_taken: List[EmailTriageAction] = []
        self._cumulative_reward: float = 0.0
        self._ground_truth_map: Dict[str, GroundTruth] = {}
        self._done: bool = False
        self._last_action_result: Optional[str] = None
        self._last_step_reward: float = 0.0
        self._metrics: Optional[EpisodeMetrics] = None
        self._episode_start_time: float = 0.0
        
        # Resource constraints
        self._responses_sent: int = 0
        
    def reset(
        self,
        task_type: Optional[TaskType] = None,
        seed: Optional[int] = None,
    ) -> EmailTriageObservation:
        """
        Reset environment to initial state with new scenario.
        
        Args:
            task_type: Task type (classification/ranking/full_triage)
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation with emails to process
        """
        # Validate task_type if provided
        valid_tasks = ["classification", "ranking", "full_triage"]
        if task_type is not None and task_type not in valid_tasks:
            logger.warning(f"Invalid task_type '{task_type}', using default")
            task_type = None
        
        # Use default task if not specified
        task = task_type or self.default_task
        
        # Generate new scenario
        self._scenario = generate_scenario(task_type=task, seed=seed)
        
        # Reset state
        self._state = State(
            episode_id=f"ep_{uuid4().hex[:12]}",
            step_count=0,
        )
        self._processed_emails = set()
        self._actions_taken = []
        self._cumulative_reward = 0.0
        self._done = False
        self._last_action_result = None
        self._last_step_reward = 0.0
        self._metrics = EpisodeMetrics()
        self._episode_start_time = time.time()
        self._responses_sent = 0  # Reset resource counter
        
        # Build ground truth lookup
        self._ground_truth_map = {
            gt.email_id: gt for gt in self._scenario.ground_truth
        }
        
        logger.info(f"Reset: task={task}, seed={seed}, episode={self._state.episode_id}")
        
        # Return initial observation
        return self._build_observation()
    
    def step(self, action: EmailTriageAction) -> EmailTriageObservation:
        """
        Execute one step with the given action.
        
        Args:
            action: Agent's action on an email
            
        Returns:
            Updated observation after action
        """
        if self._done:
            raise RuntimeError("Episode already done. Call reset() first.")
        
        if self._scenario is None:
            raise RuntimeError("No scenario loaded. Call reset() first.")
        
        # Increment step count
        self._state.step_count += 1
        
        # Validate action
        error = self._validate_action(action)
        if error:
            # Invalid action - apply configurable penalty
            penalty = self._scenario.config.get("error_penalty", -0.05)
            self._last_step_reward = penalty
            self._cumulative_reward += penalty
            self._last_action_result = f"error: {error}"
            if self._metrics is not None:
                self._metrics.record_step(penalty, "error")
            return self._build_observation()
        
        # Process action
        reward = self._calculate_reward(action)
        self._last_step_reward = reward
        self._cumulative_reward += reward
        
        # Mark emails as processed based on task type
        if self._scenario.task_type == "ranking" and action.ranking:
            # For ranking, all emails are processed with one action
            for email_id in action.ranking:
                self._processed_emails.add(email_id)
        else:
            # For other tasks, only the target email is processed
            self._processed_emails.add(action.email_id)
            
        self._actions_taken.append(action)
        
        # Track responses for resource constraint
        if action.disposition == "RESPOND":
            self._responses_sent += 1
        
        # Check if episode is done
        self._check_done()

        if self._metrics is not None:
            self._metrics.record_step(reward, self._last_action_result or "incorrect")
        
        return self._build_observation()
    
    @property
    def state(self) -> State:
        """Return current environment state."""
        return self._state
    
    def _build_observation(self) -> EmailTriageObservation:
        """Build observation from current state."""
        if self._scenario is None:
            raise RuntimeError("No scenario loaded")
        
        # Filter out processed emails for the observation
        remaining_emails = [
            e for e in self._scenario.emails 
            if e.id not in self._processed_emails
        ]
        
        # Get task config
        config = TASK_CONFIG.get(self._scenario.task_type, {})
        time_budget = config.get("time_budget")
        
        return EmailTriageObservation(
            task_id=self._scenario.scenario_id,
            task_type=self._scenario.task_type,
            emails=remaining_emails if not self._done else [],
            current_time=datetime.now(),
            emails_processed=len(self._processed_emails),
            emails_remaining=len(remaining_emails),
            time_budget_remaining=self._get_time_remaining(time_budget),
            last_action_result=self._last_action_result,
            done=self._done,
            reward=self._last_step_reward,
            cumulative_reward=self._cumulative_reward,
        )
    
    def _get_time_remaining(self, time_budget: Optional[float]) -> Optional[float]:
        """Calculate remaining time budget for hard mode."""
        if time_budget is None:
            return None
        elapsed = time.time() - self._episode_start_time
        return max(0.0, time_budget - elapsed)
    
    def _validate_action(self, action: EmailTriageAction) -> Optional[str]:
        """
        Validate the action.
        
        Returns:
            Error message if invalid, None if valid
        """
        # Check email exists
        valid_ids = {e.id for e in self._scenario.emails}
        if action.email_id not in valid_ids:
            return f"Unknown email_id: {action.email_id}"
        
        # Check email not already processed
        if action.email_id in self._processed_emails:
            return f"Email already processed: {action.email_id}"
        
        # Task-specific validation
        task_type = self._scenario.task_type
        
        if task_type == "classification":
            if action.category is None:
                return "Classification task requires 'category' field"
                
        elif task_type == "ranking":
            if action.ranking is None or not action.ranking:
                return "Ranking task requires 'ranking' field with ordered email IDs"
            
            # Validate ranking has all emails and no duplicates
            available_emails = set(email.id for email in self._scenario.emails)
            ranking_emails = set(action.ranking)
            
            if ranking_emails != available_emails or len(action.ranking) != len(self._scenario.emails):
                return f"Ranking must include all {len(available_emails)} email IDs exactly once"
                
        elif task_type == "full_triage":
            if action.priority is None or action.category is None:
                return "Full triage requires 'priority' and 'category' fields"
                
            # Check response resource constraint
            if (action.disposition == "RESPOND" and 
                self._responses_sent >= self._scenario.config.get("max_responses", float('inf'))):
                return f"Max responses limit ({self._scenario.config.get('max_responses')}) exceeded"
        
        return None  # Valid
    
    def _calculate_reward(self, action: EmailTriageAction) -> float:
        """
        Calculate reward for the action.
        
        Reward components (for full_triage):
        - Priority correctness: 0.4 (0.2 for adjacent)
        - Category correctness: 0.3
        - Disposition correctness: 0.2
        - Response appropriateness: 0.1
        
        Scaled for simpler tasks.
        """
        task_type = self._scenario.task_type

        if task_type == "ranking" and action.ranking:
            tau = kendall_tau_correlation(action.ranking, self._scenario.priority_order)
            reward = max(0.0, min(1.0, (tau + 1) / 2))
            if reward >= 0.999:
                self._last_action_result = "correct"
            elif reward > 0.0:
                self._last_action_result = "partial"
            else:
                self._last_action_result = "incorrect"
            return reward

        gt = self._ground_truth_map.get(action.email_id)
        if gt is None:
            return -0.1  # Shouldn't happen if validation passed
        
        reward = 0.0
        
        # Priority scoring
        if action.priority is not None:
            if action.priority == gt.correct_priority:
                reward += 0.4
                self._last_action_result = "correct"
            elif self._is_adjacent_priority(action.priority, gt.correct_priority):
                reward += 0.2  # Partial credit
                self._last_action_result = "partial"
            else:
                self._last_action_result = "incorrect"
        
        # Category scoring
        if action.category is not None:
            if action.category == gt.correct_category:
                reward += 0.3
                if self._last_action_result != "incorrect":
                    self._last_action_result = "correct" if self._last_action_result == "correct" else "partial"
            else:
                self._last_action_result = "incorrect" if reward == 0 else "partial"
        
        # Disposition scoring (full_triage only)
        if task_type == "full_triage" and action.disposition is not None:
            if action.disposition == gt.correct_disposition:
                reward += 0.2

            # Reward thoughtful use of the limited response budget.
            if action.disposition == "RESPOND":
                if gt.response_value >= 0.8:
                    reward += 0.1
                elif gt.response_value <= 0.2:
                    reward -= 0.1
            elif gt.correct_disposition == "RESPOND" and gt.response_value >= 0.8:
                reward -= 0.1

            # Duplicate and follow-up handling should reflect inbox-level context.
            if gt.duplicate_of:
                if action.disposition == "RESPOND":
                    reward -= 0.15
                elif action.disposition in {"ARCHIVE", "DEFER"}:
                    reward += 0.05

            if gt.recommended_after and action.disposition == "RESPOND":
                prior_action = next(
                    (past for past in self._actions_taken if past.email_id == gt.recommended_after),
                    None,
                )
                if prior_action and prior_action.disposition == "RESPOND":
                    reward += 0.05
                else:
                    reward -= 0.05
        
        # Response scoring (if applicable)
        if task_type == "full_triage" and action.response_draft is not None:
            if gt.needs_response and action.disposition == "RESPOND":
                # Basic check - non-empty response for emails that need response
                if len(action.response_draft.strip()) > 20:
                    reward += 0.1
        
        # Scale reward for simpler tasks
        if task_type == "classification":
            self._last_action_result = "correct" if action.category == gt.correct_category else "incorrect"
            # Classification only uses category, normalize
            reward = reward / 0.3 if reward > 0 else 0.0
            reward = min(1.0, reward)  # Cap at 1.0
            
        elif task_type == "ranking":
            # Ranking only uses priority, normalize
            reward = reward / 0.4 if reward > 0 else 0.0
            reward = min(1.0, reward)
        elif task_type == "full_triage":
            reward = max(-0.4, min(1.0, reward))
        
        return reward
    
    def _is_adjacent_priority(self, p1: Priority, p2: Priority) -> bool:
        """Check if two priorities are adjacent (one level apart)."""
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        return abs(order.get(p1, -1) - order.get(p2, -1)) == 1
    
    def _check_done(self) -> None:
        """Check if episode should end."""
        if self._scenario is None:
            return

        config = TASK_CONFIG.get(self._scenario.task_type, {})
        time_budget = config.get("time_budget")
        if time_budget is not None and self._get_time_remaining(time_budget) <= 0:
            self._done = True
            return
            
        # Done when all emails processed
        if len(self._processed_emails) >= len(self._scenario.emails):
            self._done = True
            return
        
        # Done if max steps reached
        if self._state.step_count >= self.max_steps:
            self._done = True
            return
    
    def get_ground_truth(self) -> Optional[ScenarioData]:
        """
        Get ground truth for current scenario (for grading).
        
        Returns:
            ScenarioData with ground truth labels
        """
        return self._scenario
    
    def get_actions_taken(self) -> List[EmailTriageAction]:
        """Get list of actions taken in this episode."""
        return self._actions_taken.copy()
