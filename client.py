"""
Python client for Email Triage OpenEnv Environment.

Provides a simple interface for agents to interact with the environment.
"""

import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass

import httpx

from models import (
    Email, EmailTriageAction, EmailTriageObservation, TaskType
)


@dataclass
class StepResult:
    """Result from environment step."""
    observation: EmailTriageObservation
    reward: float
    done: bool
    info: Dict[str, Any]


class EmailTriageEnv:
    """
    Client for Email Triage OpenEnv Environment.
    
    Usage:
        async with EmailTriageEnv("http://localhost:8000") as env:
            result = await env.reset(task_type="ranking")
            while not result.done:
                action = my_agent.decide(result.observation)
                result = await env.step(action)
            print(f"Final score: {result.info.get('final_score')}")
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        """
        Initialize client.
        
        Args:
            base_url: URL of the environment server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self) -> "EmailTriageEnv":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def reset(
        self,
        task_type: Optional[TaskType] = None,
        seed: Optional[int] = None,
    ) -> StepResult:
        """
        Reset environment and get initial observation.
        
        Args:
            task_type: Task type (classification/ranking/full_triage)
            seed: Random seed for reproducibility
            
        Returns:
            StepResult with initial observation
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        payload = {}
        if task_type:
            payload["task_type"] = task_type
        if seed is not None:
            payload["seed"] = seed
        
        response = await self._client.post("/reset", json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        return StepResult(
            observation=EmailTriageObservation(**data["observation"]),
            reward=data["reward"],
            done=data["done"],
            info=data["info"],
        )
    
    async def step(self, action: EmailTriageAction) -> StepResult:
        """
        Execute action and get result.
        
        Args:
            action: Action to execute
            
        Returns:
            StepResult with updated observation
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        payload = {
            "email_id": action.email_id,
        }
        if action.priority is not None:
            payload["priority"] = action.priority
        if action.category is not None:
            payload["category"] = action.category
        if action.disposition is not None:
            payload["disposition"] = action.disposition
        if action.response_draft is not None:
            payload["response_draft"] = action.response_draft
        if action.ranking is not None:
            payload["ranking"] = action.ranking
        
        response = await self._client.post("/step", json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        return StepResult(
            observation=EmailTriageObservation(**data["observation"]),
            reward=data["reward"],
            done=data["done"],
            info=data["info"],
        )
    
    async def state(self) -> Dict[str, Any]:
        """
        Get current environment state.
        
        Returns:
            State information dict
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        response = await self._client.get("/state")
        response.raise_for_status()
        
        return response.json()
    
    async def health(self) -> Dict[str, Any]:
        """
        Check environment health.
        
        Returns:
            Health status dict
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        response = await self._client.get("/health")
        response.raise_for_status()
        
        return response.json()
    
    async def close(self):
        """Close the client connection."""
        if self._client:
            await self._client.aclose()
            self._client = None


class EmailTriageEnvSync:
    """
    Synchronous wrapper for EmailTriageEnv.
    
    For use in non-async code (like inference scripts).
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
        )
    
    def reset(
        self,
        task_type: Optional[TaskType] = None,
        seed: Optional[int] = None,
    ) -> StepResult:
        """Reset environment and get initial observation."""
        payload = {}
        if task_type:
            payload["task_type"] = task_type
        if seed is not None:
            payload["seed"] = seed
        
        response = self._client.post("/reset", json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        return StepResult(
            observation=EmailTriageObservation(**data["observation"]),
            reward=data["reward"],
            done=data["done"],
            info=data["info"],
        )
    
    def step(self, action: EmailTriageAction) -> StepResult:
        """Execute action and get result."""
        payload = {
            "email_id": action.email_id,
        }
        if action.priority is not None:
            payload["priority"] = action.priority
        if action.category is not None:
            payload["category"] = action.category
        if action.disposition is not None:
            payload["disposition"] = action.disposition
        if action.response_draft is not None:
            payload["response_draft"] = action.response_draft
        if action.ranking is not None:
            payload["ranking"] = action.ranking
        
        response = self._client.post("/step", json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        return StepResult(
            observation=EmailTriageObservation(**data["observation"]),
            reward=data["reward"],
            done=data["done"],
            info=data["info"],
        )
    
    def state(self) -> Dict[str, Any]:
        """Get current environment state."""
        response = self._client.get("/state")
        response.raise_for_status()
        return response.json()
    
    def health(self) -> Dict[str, Any]:
        """Check environment health."""
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the client connection."""
        self._client.close()
    
    def __enter__(self) -> "EmailTriageEnvSync":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage
if __name__ == "__main__":
    async def main():
        async with EmailTriageEnv("http://localhost:8000") as env:
            # Check health
            health = await env.health()
            print(f"Health: {health}")
            
            # Run classification task
            result = await env.reset(task_type="classification", seed=42)
            print(f"\nTask: {result.observation.task_type}")
            print(f"Emails: {len(result.observation.emails)}")
            
            if result.observation.emails:
                email = result.observation.emails[0]
                print(f"\nEmail: {email.subject}")
                print(f"From: {email.sender_name} ({email.sender_importance})")
                
                # Take action
                action = EmailTriageAction(
                    email_id=email.id,
                    category="URGENT",  # Guess
                )
                result = await env.step(action)
                
                print(f"\nResult: {result.info.get('last_action_result')}")
                print(f"Done: {result.done}")
                if result.done:
                    print(f"Final score: {result.info.get('final_score', 'N/A')}")
    
    asyncio.run(main())
