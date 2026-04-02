"""
Inference Script for Email Triage OpenEnv Environment
======================================================

MANDATORY - Before running, ensure the following environment variables are set:
- API_BASE_URL: The API endpoint for the LLM (default: https://router.huggingface.co/v1)
- MODEL_NAME: The model identifier to use (default: Qwen/Qwen2.5-72B-Instruct)
- HF_TOKEN: Your Hugging Face / API key
- LOCAL_IMAGE_NAME: Docker image name for the environment

STDOUT FORMAT:
The script emits exactly three line types:
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import sys
import json
import textwrap
from typing import List, Optional, Dict, Any

from openai import OpenAI

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.models import EmailTriageAction, Email, TaskType


# Configuration from environment
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME", "email-triage-openenv:latest")

# Task configuration
TASK_NAME = os.getenv("EMAIL_TRIAGE_TASK", "classification")
BENCHMARK = "email_triage"
MAX_STEPS = 20
TEMPERATURE = 0.3
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.5


# Environment class that spawns Docker containers
class EmailTriageEnv:
    """Email Triage Environment that manages Docker containers."""
    
    def __init__(self, container=None):
        self.container = container
        self.base_url = None
    
    @classmethod
    async def from_docker_image(cls, image_name: str):
        """Create environment from Docker image."""
        import subprocess
        import time
        import httpx
        
        # Start container
        try:
            # Run container in detached mode with random port
            cmd = [
                "docker", "run", "-d", "--rm", 
                "-p", "0:8000",  # Random host port
                image_name
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            
            # Get the actual port
            cmd = ["docker", "port", container_id, "8000"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            port_info = result.stdout.strip()  # e.g., "0.0.0.0:32768"
            host_port = port_info.split(":")[-1]
            
            # Wait for container to be ready
            base_url = f"http://localhost:{host_port}"
            async with httpx.AsyncClient(timeout=2.0) as client:
                for _ in range(30):  # Wait up to 30 seconds
                    try:
                        response = await client.get(f"{base_url}/health")
                        if response.status_code == 200:
                            break
                    except Exception:
                        pass
                    time.sleep(1)
                else:
                    raise Exception("Container failed to start within 30 seconds")
            
            env = cls(container=container_id)
            env.base_url = base_url
            return env
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to start container: {e}", file=sys.stderr)
            raise
    
    async def reset(self, task_type: str, seed: int = 42):
        """Reset environment."""
        import httpx
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/reset",
                json={"task_type": task_type, "seed": seed}
            )
            response.raise_for_status()
            data = response.json()
            
            from models import EmailTriageObservation
            return StepResult(
                observation=EmailTriageObservation(**data["observation"]),
                reward=data["reward"],
                done=data["done"],
                info=data["info"]
            )
    
    async def step(self, action: EmailTriageAction):
        """Take environment step."""
        import httpx
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/step",
                json=action.model_dump()
            )
            response.raise_for_status()
            data = response.json()
            
            from models import EmailTriageObservation
            return StepResult(
                observation=EmailTriageObservation(**data["observation"]),
                reward=data["reward"],
                done=data["done"],
                info=data["info"]
            )
    
    async def close(self):
        """Stop container."""
        if self.container:
            import subprocess
            try:
                subprocess.run(["docker", "stop", self.container], 
                             capture_output=True, check=True)
            except subprocess.CalledProcessError:
                pass  # Container may have already stopped


# Result dataclass
class StepResult:
    """Result from environment step."""
    def __init__(self, observation, reward, done, info):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


# System prompts for each task type
SYSTEM_PROMPTS = {
    "classification": textwrap.dedent("""
        You are an email triage assistant. Your task is to classify emails into one of 5 categories:
        
        - URGENT: Requires immediate action (server outages, security issues, critical deadlines)
        - ACTION_REQUIRED: Needs response but not immediate (review requests, approvals, follow-ups)
        - INFO: Informational only, no action needed (FYIs, status updates, announcements)
        - SPAM: Marketing, promotions, unsolicited content
        - PERSONAL: Personal/social messages (lunch invites, birthday wishes)
        
        Analyze the email subject, sender, and body to determine the correct category.
        
        Respond with ONLY a JSON object in this format:
        {"category": "CATEGORY_NAME"}
    """).strip(),
    
    "ranking": textwrap.dedent("""
        You are an email triage assistant. Your task is to rank ALL emails in order of priority.
        
        Priority criteria:
        - HIGH: Urgent, time-sensitive, from VIP senders, critical issues
        - MEDIUM: Important but not urgent, requires attention today  
        - LOW: Can wait, informational, newsletters, personal
        
        Consider:
        1. Sender importance (VIP > KNOWN > UNKNOWN > EXTERNAL)
        2. Subject urgency indicators (URGENT, ASAP, deadline mentions)
        3. Content criticality (production issues > routine requests > FYI)
        
        You will receive multiple emails. You must provide a complete ranking.
        
        Respond with ONLY a JSON object in this format:
        {"ranking": ["email_id_1", "email_id_2", "..."]}
        
        Order from HIGHEST priority (first) to LOWEST priority (last).
        Include ALL email IDs exactly once.
    """).strip(),
    
    "full_triage": textwrap.dedent("""
        You are an email triage assistant. For each email, provide:
        
        1. Priority: HIGH (urgent), MEDIUM (important), LOW (can wait)
        2. Category: URGENT, ACTION_REQUIRED, INFO, SPAM, or PERSONAL
        3. Disposition: RESPOND (reply needed), DELEGATE (forward to someone), 
                       ARCHIVE (no action), DEFER (handle later)
        4. Response draft: If disposition is RESPOND, provide a brief response
        
        Consider sender importance, urgency signals, and content to make decisions.
        
        Respond with ONLY a JSON object in this format:
        {
            "priority": "HIGH|MEDIUM|LOW",
            "category": "CATEGORY_NAME",
            "disposition": "RESPOND|DELEGATE|ARCHIVE|DEFER",
            "response_draft": "Your response text here (only if disposition is RESPOND)"
        }
    """).strip(),
}


def log_start(task: str, env: str, model: str) -> None:
    """Log episode start."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log step result."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def format_email_for_prompt(email: Email) -> str:
    """Format email for LLM prompt."""
    return textwrap.dedent(f"""
        Email ID: {email.id}
        From: {email.sender_name} <{email.sender}> [{email.sender_importance}]
        Subject: {email.subject}
        Time: {email.timestamp}
        Attachments: {"Yes" if email.has_attachment else "No"}
        
        Body:
        {email.body}
    """).strip()


def format_emails_for_ranking_prompt(emails: List[Email]) -> str:
    """Format all emails for ranking prompt."""
    email_texts = []
    for i, email in enumerate(emails, 1):
        email_text = textwrap.dedent(f"""
            Email {i} (ID: {email.id}):
            From: {email.sender_name} <{email.sender}> [{email.sender_importance}]
            Subject: {email.subject}
            Time: {email.timestamp}
            Attachments: {"Yes" if email.has_attachment else "No"}
            
            Body:
            {email.body}
            
            ---
        """).strip()
        email_texts.append(email_text)
    
    return "\n\n".join(email_texts) + "\n\nRank these emails from HIGHEST to LOWEST priority."


def get_llm_ranking_decision(
    client: OpenAI,
    emails: List[Email],
    task_type: TaskType,
    user_prompt: str,
) -> Dict[str, Any]:
    """Get LLM ranking decision for all emails."""
    system_prompt = SYSTEM_PROMPTS[task_type]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        
        content = response.choices[0].message.content or ""
        decision = parse_llm_response(content, task_type)
        return normalize_ranking_decision(decision, emails)
        
    except Exception as e:
        # Silent fallback - no debug output to stdout
        return normalize_ranking_decision(parse_llm_response("", task_type), emails)


def create_ranking_action(decision: Dict[str, Any]) -> EmailTriageAction:
    """Create ranking action from LLM decision."""
    ranking = decision.get("ranking", [])
    
    # Use first email ID as placeholder (required field)
    email_id = ranking[0] if ranking else "e1"
    
    return EmailTriageAction(
        email_id=email_id,
        ranking=ranking,
    )


def normalize_ranking_decision(decision: Dict[str, Any], emails: List[Email]) -> Dict[str, Any]:
    """Normalize ranking output so it always contains each email exactly once."""
    valid_ids = [email.id for email in emails]
    proposed = decision.get("ranking")

    normalized: List[str] = []
    seen = set()

    if isinstance(proposed, list):
        for email_id in proposed:
            if email_id in valid_ids and email_id not in seen:
                normalized.append(email_id)
                seen.add(email_id)

    for email_id in valid_ids:
        if email_id not in seen:
            normalized.append(email_id)

    decision["ranking"] = normalized
    return decision


def parse_llm_response(response_text: str, task_type: TaskType) -> Dict[str, Any]:
    """Parse LLM response JSON."""
    # Try to extract JSON from response
    text = response_text.strip()
    
    # Handle markdown code blocks
    if "```json" in text:
        parts = text.split("```json")
        if len(parts) > 1:
            text = parts[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1].split("```")[0].strip()
    
    # Try to find JSON object - validate braces exist
    if "{" in text and "}" in text:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            text = text[start:end]
        except ValueError:
            pass  # Fall through to defaults
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback defaults based on task type
        if task_type == "classification":
            return {"category": "INFO"}
        elif task_type == "ranking":
            return {"ranking": []}
        else:
            return {
                "priority": "MEDIUM",
                "category": "INFO",
                "disposition": "ARCHIVE",
            }


def get_llm_decision(
    client: OpenAI,
    email: Email,
    task_type: TaskType,
) -> Dict[str, Any]:
    """Get LLM decision for an email."""
    system_prompt = SYSTEM_PROMPTS[task_type]
    user_prompt = format_email_for_prompt(email)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        
        content = response.choices[0].message.content or ""
        return parse_llm_response(content, task_type)
        
    except Exception as e:
        # Silent fallback - no debug output to stdout
        return parse_llm_response("", task_type)


def create_action(
    email_id: str,
    decision: Dict[str, Any],
    task_type: TaskType,
) -> EmailTriageAction:
    """Create action from LLM decision."""
    return EmailTriageAction(
        email_id=email_id,
        priority=decision.get("priority"),
        category=decision.get("category"),
        disposition=decision.get("disposition"),
        response_draft=decision.get("response_draft"),
    )


def format_action_for_log(action: EmailTriageAction, task_type: TaskType) -> str:
    """Format action for logging."""
    if task_type == "classification":
        return f"classify({action.email_id},{action.category})"
    elif task_type == "ranking":
        return f"prioritize({action.email_id},{action.priority})"
    else:
        return f"triage({action.email_id},{action.priority},{action.category},{action.disposition})"


async def run_task(
    client: OpenAI,
    env: EmailTriageEnv,
    task_type: TaskType,
    seed: int = 42,
) -> float:
    """
    Run a single task and return the score.
    
    Args:
        client: OpenAI client
        env: Environment client
        task_type: Task to run
        seed: Random seed for reproducibility
        
    Returns:
        Final score (0.0 to 1.0)
    """
    log_start(task=task_type, env=BENCHMARK, model=MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        # Reset environment
        result = await env.reset(task_type=task_type, seed=seed)
        
        # Process emails based on task type
        if task_type == "ranking":
            # For ranking task, process all emails in one action
            if not result.observation.emails:
                # No emails to rank - score is 0
                score = 0.0
                success = False
            else:
                # Get all email IDs and create prompt with all emails
                all_emails = result.observation.emails
                user_prompt = format_emails_for_ranking_prompt(all_emails)
                
                # Get LLM decision for ranking
                decision = get_llm_ranking_decision(client, all_emails, task_type, user_prompt)
                
                # Create and execute action
                action = create_ranking_action(decision)
                result = await env.step(action)
                
                # Log step
                reward = result.reward
                rewards.append(reward)
                steps_taken = 1
                
                error = None
                last_action_result = result.info.get("last_action_result") or ""
                if last_action_result.startswith("error"):
                    error = last_action_result
                
                action_str = f"rank({len(decision.get('ranking', []))} emails)"
                log_step(
                    step=1,
                    action=action_str,
                    reward=reward,
                    done=result.done,
                    error=error,
                )
                
                # Get final score
                score = result.info.get("final_score", 0.0)
                success = score >= SUCCESS_SCORE_THRESHOLD
            
        else:
            # For other tasks, process emails one by one
            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break
                
                if not result.observation.emails:
                    break
                
                # Get first unprocessed email
                email = result.observation.emails[0]
                
                # Get LLM decision
                decision = get_llm_decision(client, email, task_type)
                
                # Create and execute action
                action = create_action(email.id, decision, task_type)
                result = await env.step(action)
                
                # Log step
                reward = result.reward
                rewards.append(reward)
                steps_taken = step
                
                error = None
                if result.info.get("last_action_result", "").startswith("error"):
                    error = result.info.get("last_action_result", "unknown_error")
                
                action_str = format_action_for_log(action, task_type)
                log_step(
                    step=step,
                    action=action_str,
                    reward=reward,
                    done=result.done,
                    error=error,
                )
                
                if result.done:
                    break
        
        # Get final score
        score = result.info.get("final_score", 0.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as e:
        # Silent failure - log to stderr if needed
        import sys
        print(f"Task failed: {e}", file=sys.stderr, flush=True)
        score = 0.0
        success = False
    
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return score


async def main():
    """Main entry point - emits ONLY [START]/[STEP]/[END] lines to stdout."""
    import sys
    
    # Check for API key
    if not API_KEY:
        print("No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY.", file=sys.stderr, flush=True)
        sys.exit(1)
    
    # Initialize clients
    openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Determine which tasks to run
    tasks_to_run = os.getenv("EMAIL_TRIAGE_TASKS") or os.getenv("EMAIL_TRIAGE_TASK") or "all"
    
    if tasks_to_run == "all":
        task_list = ["classification", "ranking", "full_triage"]
    else:
        task_list = [tasks_to_run]
    
    # Run each task
    scores = {}
    
    # Create environment from Docker image
    env = await EmailTriageEnv.from_docker_image(LOCAL_IMAGE_NAME)
    try:
        print(f"Container started successfully", file=sys.stderr, flush=True)
        
        # Run tasks - NO separators or headers to stdout
        for task_type in task_list:
            score = await run_task(
                client=openai_client,
                env=env,
                task_type=task_type,
                seed=42,  # Fixed seed for reproducibility
            )
            scores[task_type] = score
    
    finally:
        # Always close the environment
        await env.close()
        print("Container stopped", file=sys.stderr, flush=True)
    
    # No summary to stdout - all output is [START]/[STEP]/[END] only
    # Summary can go to stderr if needed
    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print("SUMMARY", file=sys.stderr, flush=True)
    print(f"{'='*60}", file=sys.stderr, flush=True)
    for task, score in scores.items():
        status = "PASS" if score >= SUCCESS_SCORE_THRESHOLD else "FAIL"
        print(f"  {task}: {score:.3f} [{status}]", file=sys.stderr, flush=True)
    
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  Average: {avg_score:.3f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
