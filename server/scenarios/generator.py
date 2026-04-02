"""
Scenario generator for Email Triage environment.

Generates reproducible email batches for each task type:
- classification: 1 email
- ranking: 8 emails
- full_triage: 25 emails
"""

import random
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from server.models import (
    Email, GroundTruth, ScenarioData, 
    TaskType, Category, Priority, Disposition
)
from server.scenarios.templates import (
    ALL_TEMPLATES, TOPICS, SENDERS,
    get_random_sender, fill_template, EmailTemplate
)


# Task configurations
TASK_CONFIG = {
    "classification": {
        "email_count": 1,
        "time_budget": None,
        "difficulty_multiplier": 1.0,
        "difficulty": "easy",
        "description": "Classify a single email into one of 5 categories",
    },
    "ranking": {
        "email_count": 8,  # Increased from 5 to 8
        "time_budget": None,
        "difficulty_multiplier": 1.5,
        "difficulty": "medium",
        "description": "Rank 8 emails by priority order",
    },
    "full_triage": {
        "email_count": 25,  # Increased from 15 to 25
        "time_budget": 180.0,  # Reduced from 300s (5min) to 180s (3min)
        "difficulty_multiplier": 2.5,  # Increased from 2.0 to 2.5
        "max_responses": 8,  # New constraint: can only respond to 8 emails max
        "error_penalty": -0.3,  # New: harsh penalty for wrong actions
        "difficulty": "hard",
        "description": "Full inbox triage with 25 emails, threaded dependencies, duplicate handling, and a fixed response budget",
    },
}


# Category distribution by difficulty
CATEGORY_DISTRIBUTIONS = {
    "classification": {
        "URGENT": 0.20,
        "ACTION_REQUIRED": 0.25,
        "INFO": 0.25,
        "SPAM": 0.15,
        "PERSONAL": 0.15,
    },
    "ranking": {
        "URGENT": 0.15,  # Reduced for more conflicts
        "ACTION_REQUIRED": 0.35,  # Increased ambiguous middle category
        "INFO": 0.20,
        "SPAM": 0.15,
        "PERSONAL": 0.15,
    },
    "full_triage": {
        "URGENT": 0.25,  # More urgent items for conflicts
        "ACTION_REQUIRED": 0.35,  # More action items
        "INFO": 0.20,
        "SPAM": 0.10,
        "PERSONAL": 0.10,
    },
}


def generate_email(
    email_id: str,
    template: EmailTemplate,
    rng: random.Random,
    base_time: datetime,
    time_offset_hours: float = 0,
) -> tuple[Email, GroundTruth]:
    """Generate a single email with ground truth from a template."""
    
    # Select sender
    sender_type = rng.choice(template.sender_patterns)
    sender_info = get_random_sender(sender_type, rng)
    
    # Generate subject and body
    topic_category = {
        "URGENT": "urgent",
        "ACTION_REQUIRED": "action", 
        "INFO": "info",
        "SPAM": "products",
        "PERSONAL": "events",
    }.get(template.category, "info")
    
    topic = rng.choice(TOPICS.get(topic_category, TOPICS["info"]))
    context = {"topic": topic}
    
    subject = fill_template(rng.choice(template.subject_templates), rng, context)
    body = fill_template(rng.choice(template.body_templates), rng, context)
    
    # Calculate timestamp (offset from base time)
    timestamp = base_time - timedelta(hours=time_offset_hours)
    
    # Calculate urgency score within template range
    urgency_score = rng.uniform(*template.urgency_score_range)
    
    # Create email
    email = Email(
        id=email_id,
        subject=subject,
        sender=sender_info["email"],
        sender_name=sender_info["name"],
        sender_importance=sender_info["importance"],
        body=body,
        timestamp=timestamp,
        has_attachment=rng.random() < 0.2,  # 20% have attachments
        thread_id=None,
        is_reply=rng.random() < 0.3,  # 30% are replies
    )
    
    # Create ground truth
    ground_truth = GroundTruth(
        email_id=email_id,
        correct_priority=template.priority,
        correct_category=template.category,
        correct_disposition=template.disposition,
        priority_rank=0,  # Will be set later
        urgency_score=urgency_score,
        needs_response=template.needs_response,
        response_context=body[:200] if template.needs_response else None,
        response_value=0.5,
    )
    
    return email, ground_truth


def select_categories(
    count: int,
    task_type: TaskType,
    rng: random.Random,
) -> List[Category]:
    """Select categories for emails based on distribution."""
    distribution = CATEGORY_DISTRIBUTIONS[task_type]
    categories = list(distribution.keys())
    weights = list(distribution.values())
    
    # For classification, pick one random category
    if count == 1:
        return [rng.choices(categories, weights=weights)[0]]
    
    # For multiple emails, ensure variety
    selected = []
    
    # Guarantee at least one high-priority email for ranking/triage
    if task_type in ["ranking", "full_triage"]:
        selected.append(rng.choice(["URGENT", "ACTION_REQUIRED"]))
        count -= 1
    
    # Fill remaining with weighted random selection
    selected.extend(rng.choices(categories, weights=weights, k=count))
    
    # Shuffle to avoid predictable patterns
    rng.shuffle(selected)
    
    return selected


def _default_response_value(ground_truth: GroundTruth) -> float:
    """Assign a default response-slot value before hard-task enrichment."""
    if ground_truth.correct_disposition == "RESPOND":
        return {"HIGH": 0.78, "MEDIUM": 0.58, "LOW": 0.35}[ground_truth.correct_priority]
    if ground_truth.correct_disposition == "DELEGATE":
        return {"HIGH": 0.52, "MEDIUM": 0.38, "LOW": 0.24}[ground_truth.correct_priority]
    if ground_truth.correct_disposition == "DEFER":
        return {"HIGH": 0.34, "MEDIUM": 0.22, "LOW": 0.16}[ground_truth.correct_priority]
    return 0.08


def _recompute_priority_order(ground_truths: List[GroundTruth]) -> List[str]:
    """Recompute priority order after scenario enrichment."""
    sorted_gts = sorted(
        ground_truths,
        key=lambda gt: (-gt.urgency_score, -gt.response_value, gt.email_id),
    )
    for rank, gt in enumerate(sorted_gts, start=1):
        gt.priority_rank = rank
    return [gt.email_id for gt in sorted_gts]


def _rewrite_email_and_truth(
    emails: List[Email],
    ground_truths: List[GroundTruth],
    index: int,
    rng: random.Random,
    base_time: datetime,
    *,
    sender_type: str,
    subject: str,
    body: str,
    hours_ago: float,
    priority: Priority,
    category: Category,
    disposition: Disposition,
    urgency_score: float,
    needs_response: bool,
    response_value: float,
    thread_id: Optional[str],
    thread_role: Optional[str],
    business_impact: str,
    expected_response: Optional[str] = None,
    duplicate_of: Optional[str] = None,
    recommended_after: Optional[str] = None,
    has_attachment: Optional[bool] = None,
) -> None:
    """Overwrite a generated email with deterministic hard-mode story content."""
    sender_info = get_random_sender(sender_type, rng)
    existing_email = emails[index]
    emails[index] = existing_email.model_copy(
        update={
            "subject": subject,
            "sender": sender_info["email"],
            "sender_name": sender_info["name"],
            "sender_importance": sender_info["importance"],
            "body": body,
            "timestamp": base_time - timedelta(hours=hours_ago),
            "thread_id": thread_id,
            "is_reply": thread_role in {"followup", "duplicate"},
            "has_attachment": existing_email.has_attachment if has_attachment is None else has_attachment,
        }
    )

    existing_truth = ground_truths[index]
    ground_truths[index] = existing_truth.model_copy(
        update={
            "correct_priority": priority,
            "correct_category": category,
            "correct_disposition": disposition,
            "urgency_score": urgency_score,
            "needs_response": needs_response,
            "response_context": body[:200] if needs_response else None,
            "expected_response": expected_response,
            "thread_id": thread_id,
            "thread_role": thread_role,
            "duplicate_of": duplicate_of,
            "recommended_after": recommended_after,
            "response_value": response_value,
            "business_impact": business_impact,
        }
    )


def _enrich_full_triage_scenario(
    emails: List[Email],
    ground_truths: List[GroundTruth],
    rng: random.Random,
    base_time: datetime,
    config: dict,
) -> tuple[List[Email], List[GroundTruth], dict, dict]:
    """Turn hard mode into an inbox-management task with thread state and tradeoffs."""
    for gt in ground_truths:
        gt.response_value = _default_response_value(gt)
        if gt.business_impact is None:
            gt.business_impact = "coordination"

    # Bundle 1: active P1 incident thread.
    thread_incident = "thread_incident_bridge"
    _rewrite_email_and_truth(
        emails,
        ground_truths,
        0,
        rng,
        base_time,
        sender_type="ops_alerts",
        subject="P1 incident: checkout API outage affecting enterprise customers",
        body=(
            "We have an active production incident. Checkout requests are failing for multiple enterprise tenants and "
            "support is escalating. Please acknowledge in the next 10 minutes, join the incident bridge, and send an "
            "ETA for mitigation."
        ),
        hours_ago=0.08,
        priority="HIGH",
        category="URGENT",
        disposition="RESPOND",
        urgency_score=0.995,
        needs_response=True,
        response_value=1.00,
        thread_id=thread_incident,
        thread_role="root",
        business_impact="revenue",
        expected_response="Acknowledged. I am joining the incident bridge now and will send an ETA in 10 minutes.",
    )
    _rewrite_email_and_truth(
        emails,
        ground_truths,
        1,
        rng,
        base_time,
        sender_type="client",
        subject="Re: P1 incident: executive update needed before 09:15",
        body=(
            "Our VP is about to brief leadership and needs a customer-facing ETA plus a short mitigation summary. "
            "Please reply in-thread once you have acknowledged the outage internally."
        ),
        hours_ago=0.05,
        priority="HIGH",
        category="URGENT",
        disposition="RESPOND",
        urgency_score=0.982,
        needs_response=True,
        response_value=0.96,
        thread_id=thread_incident,
        thread_role="followup",
        business_impact="customer_trust",
        expected_response="Acknowledged. We are actively mitigating the outage and I will send a customer-safe ETA shortly.",
        recommended_after="e1",
    )
    _rewrite_email_and_truth(
        emails,
        ground_truths,
        2,
        rng,
        base_time,
        sender_type="colleague",
        subject="Re: P1 incident: duplicate request for the same status update",
        body=(
            "Can you also send me the same status note you are already preparing for the incident bridge? "
            "No new information here, I just want to stay in the loop."
        ),
        hours_ago=0.03,
        priority="MEDIUM",
        category="INFO",
        disposition="DEFER",
        urgency_score=0.36,
        needs_response=False,
        response_value=0.10,
        thread_id=thread_incident,
        thread_role="duplicate",
        business_impact="coordination",
        duplicate_of="e1",
    )

    # Bundle 2: budget approval thread with reminder and low-value duplicate.
    thread_budget = "thread_budget_freeze"
    _rewrite_email_and_truth(
        emails,
        ground_truths,
        3,
        rng,
        base_time,
        sender_type="boss",
        subject="Urgent approval needed: hiring plan locked unless budget is approved by noon",
        body=(
            "Finance will freeze headcount planning at noon if we do not confirm the revised engineering budget. "
            "I need your approval or edits on the attached plan before the leadership review starts."
        ),
        hours_ago=0.65,
        priority="HIGH",
        category="ACTION_REQUIRED",
        disposition="RESPOND",
        urgency_score=0.93,
        needs_response=True,
        response_value=0.91,
        thread_id=thread_budget,
        thread_role="root",
        business_impact="hiring",
        expected_response="Reviewed. I approve the revised budget with minor edits and will add comments before the leadership review.",
        has_attachment=True,
    )
    _rewrite_email_and_truth(
        emails,
        ground_truths,
        4,
        rng,
        base_time,
        sender_type="hr",
        subject="Re: budget approval reminder before leadership review",
        body=(
            "Following up because finance needs a yes or no before the leadership packet is finalized. "
            "Please reply after you have looked at the original approval request."
        ),
        hours_ago=0.42,
        priority="HIGH",
        category="ACTION_REQUIRED",
        disposition="RESPOND",
        urgency_score=0.88,
        needs_response=True,
        response_value=0.84,
        thread_id=thread_budget,
        thread_role="followup",
        business_impact="hiring",
        expected_response="I reviewed the original request and am sending approval now so the leadership packet can proceed.",
        recommended_after="e4",
    )
    _rewrite_email_and_truth(
        emails,
        ground_truths,
        5,
        rng,
        base_time,
        sender_type="colleague",
        subject="Re: budget approval spreadsheet copied for visibility",
        body=(
            "Sharing the same spreadsheet for visibility. This does not need a separate reply if you already answer the main thread."
        ),
        hours_ago=0.35,
        priority="LOW",
        category="INFO",
        disposition="ARCHIVE",
        urgency_score=0.20,
        needs_response=False,
        response_value=0.05,
        thread_id=thread_budget,
        thread_role="duplicate",
        business_impact="coordination",
        duplicate_of="e4",
        has_attachment=True,
    )

    # Bundle 3: client security questionnaire with delegate step.
    thread_security = "thread_security_questionnaire"
    _rewrite_email_and_truth(
        emails,
        ground_truths,
        6,
        rng,
        base_time,
        sender_type="client",
        subject="Security questionnaire due today for enterprise renewal",
        body=(
            "We cannot finalize the renewal without answers to the attached security questionnaire by end of day. "
            "Please confirm ownership and send back the completed response package."
        ),
        hours_ago=1.6,
        priority="HIGH",
        category="ACTION_REQUIRED",
        disposition="RESPOND",
        urgency_score=0.90,
        needs_response=True,
        response_value=0.89,
        thread_id=thread_security,
        thread_role="root",
        business_impact="renewal",
        expected_response="Acknowledged. I am coordinating the questionnaire now and will send the completed package before end of day.",
        has_attachment=True,
    )
    _rewrite_email_and_truth(
        emails,
        ground_truths,
        7,
        rng,
        base_time,
        sender_type="colleague",
        subject="Re: security questionnaire should be routed to security engineering",
        body=(
            "This needs the security engineering team for the technical controls section. "
            "Please forward it to them instead of drafting a duplicate answer yourself."
        ),
        hours_ago=1.2,
        priority="MEDIUM",
        category="ACTION_REQUIRED",
        disposition="DELEGATE",
        urgency_score=0.67,
        needs_response=False,
        response_value=0.46,
        thread_id=thread_security,
        thread_role="followup",
        business_impact="renewal",
        recommended_after="e7",
    )
    _rewrite_email_and_truth(
        emails,
        ground_truths,
        8,
        rng,
        base_time,
        sender_type="unknown_internal",
        subject="Re: automated reminder from vendor portal",
        body=(
            "Automated reminder: the same security questionnaire is still waiting in the vendor portal. "
            "No separate reply needed if the primary owner has already acknowledged it."
        ),
        hours_ago=0.9,
        priority="LOW",
        category="INFO",
        disposition="ARCHIVE",
        urgency_score=0.18,
        needs_response=False,
        response_value=0.04,
        thread_id=thread_security,
        thread_role="duplicate",
        business_impact="coordination",
        duplicate_of="e7",
    )

    # Bundle 4: executive meeting prep thread.
    thread_exec = "thread_exec_customer_meeting"
    _rewrite_email_and_truth(
        emails,
        ground_truths,
        9,
        rng,
        base_time,
        sender_type="boss",
        subject="Need final customer deck sign-off before 15:00 meeting",
        body=(
            "I need you to confirm the talking points and approve the final customer deck before the executive review at 15:00. "
            "A short response with approval or blockers is enough."
        ),
        hours_ago=2.4,
        priority="HIGH",
        category="ACTION_REQUIRED",
        disposition="RESPOND",
        urgency_score=0.86,
        needs_response=True,
        response_value=0.83,
        thread_id=thread_exec,
        thread_role="root",
        business_impact="executive_visibility",
        expected_response="Reviewed. The deck is approved for the 15:00 meeting with one minor note on the customer timeline slide.",
        has_attachment=True,
    )
    _rewrite_email_and_truth(
        emails,
        ground_truths,
        10,
        rng,
        base_time,
        sender_type="colleague",
        subject="Re: formatting note on the customer deck",
        body=(
            "Small formatting note on slide seven. This can wait until after the main sign-off unless you have spare time."
        ),
        hours_ago=2.1,
        priority="LOW",
        category="INFO",
        disposition="DEFER",
        urgency_score=0.16,
        needs_response=False,
        response_value=0.07,
        thread_id=thread_exec,
        thread_role="duplicate",
        business_impact="coordination",
        duplicate_of="e10",
    )
    _rewrite_email_and_truth(
        emails,
        ground_truths,
        11,
        rng,
        base_time,
        sender_type="client",
        subject="Re: can we shift the customer meeting 30 minutes earlier?",
        body=(
            "Our COO can only attend if the meeting starts 30 minutes earlier. Please confirm after you have checked the approved deck and speaker availability."
        ),
        hours_ago=1.9,
        priority="HIGH",
        category="ACTION_REQUIRED",
        disposition="RESPOND",
        urgency_score=0.84,
        needs_response=True,
        response_value=0.80,
        thread_id=thread_exec,
        thread_role="followup",
        business_impact="executive_visibility",
        expected_response="Acknowledged. I am confirming the deck and speaker availability now and will reply with the updated meeting time shortly.",
        recommended_after="e10",
    )

    used_indices = set(range(12))
    extra_critical = [
        idx for idx, gt in enumerate(ground_truths)
        if idx not in used_indices and gt.correct_disposition == "RESPOND"
    ]
    for idx, value in zip(extra_critical[:2], [0.81, 0.79]):
        ground_truths[idx].response_value = value
        ground_truths[idx].business_impact = "delivery"

    ideal_response_ids = [
        gt.email_id
        for gt in sorted(
            [gt for gt in ground_truths if gt.correct_disposition == "RESPOND"],
            key=lambda gt: (-gt.response_value, -gt.urgency_score, gt.email_id),
        )[: config["max_responses"]]
    ]
    thread_count = len({gt.thread_id for gt in ground_truths if gt.thread_id})
    dependency_count = sum(
        1 for gt in ground_truths if gt.duplicate_of is not None or gt.recommended_after is not None
    )
    config["ideal_response_ids"] = ideal_response_ids
    config["thread_count"] = thread_count
    config["dependency_count"] = dependency_count

    difficulty_factors = {
        "email_count": len(emails),
        "category_variety": len({gt.correct_category for gt in ground_truths}),
        "ambiguity_level": config["difficulty_multiplier"],
        "time_pressured": config["time_budget"] is not None,
        "thread_count": thread_count,
        "dependency_count": dependency_count,
        "response_budget": config["max_responses"],
    }
    return emails, ground_truths, config, difficulty_factors


def generate_scenario(
    task_type: TaskType = "classification",
    seed: Optional[int] = None,
) -> ScenarioData:
    """
    Generate a complete scenario for the given task type.
    
    Args:
        task_type: "classification", "ranking", or "full_triage"
        seed: Random seed for reproducibility (optional)
        
    Returns:
        ScenarioData with emails and ground truth
    """
    # Initialize RNG
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    rng = random.Random(seed)
    
    # Get task config
    config = dict(TASK_CONFIG[task_type])
    email_count = config["email_count"]
    
    # Generate scenario ID
    scenario_id = f"scenario_{task_type}_{seed}"
    
    # Base time (simulated "now")
    base_time = datetime(2024, 1, 15, 9, 0, 0)
    
    # Select categories for each email
    categories = select_categories(email_count, task_type, rng)
    
    # Generate emails
    emails = []
    ground_truths = []
    
    for i, category in enumerate(categories):
        email_id = f"e{i+1}"
        template = ALL_TEMPLATES[category]
        
        # Vary arrival time (more recent = potentially more urgent context)
        time_offset = rng.uniform(0, 24) * (i + 1) / email_count
        
        email, gt = generate_email(
            email_id=email_id,
            template=template,
            rng=rng,
            base_time=base_time,
            time_offset_hours=time_offset,
        )
        
        emails.append(email)
        ground_truths.append(gt)
    
    if task_type == "full_triage":
        emails, ground_truths, config, difficulty_factors = _enrich_full_triage_scenario(
            emails=emails,
            ground_truths=ground_truths,
            rng=rng,
            base_time=base_time,
            config=config,
        )
    else:
        difficulty_factors = {
            "email_count": email_count,
            "category_variety": len(set(categories)),
            "ambiguity_level": config["difficulty_multiplier"],
            "time_pressured": config["time_budget"] is not None,
        }

    priority_order = _recompute_priority_order(ground_truths)
    
    return ScenarioData(
        scenario_id=scenario_id,
        task_type=task_type,
        seed=seed,
        emails=emails,
        ground_truth=ground_truths,
        priority_order=priority_order,
        config=config,  # Include task configuration
        difficulty_factors=difficulty_factors,
    )


def generate_batch_scenarios(
    task_type: TaskType,
    count: int = 10,
    base_seed: int = 42,
) -> List[ScenarioData]:
    """Generate multiple scenarios for testing."""
    return [
        generate_scenario(task_type, seed=base_seed + i)
        for i in range(count)
    ]


# Pre-generate some scenarios for quick loading
def create_seed_scenarios():
    """Create seed scenarios for each difficulty level."""
    scenarios = {
        "easy": [generate_scenario("classification", seed=i) for i in range(10)],
        "medium": [generate_scenario("ranking", seed=i) for i in range(10)],
        "hard": [generate_scenario("full_triage", seed=i) for i in range(10)],
    }
    return scenarios


if __name__ == "__main__":
    # Test scenario generation
    print("Generating test scenarios...")
    
    for task_type in ["classification", "ranking", "full_triage"]:
        scenario = generate_scenario(task_type, seed=42)
        print(f"\n{'='*60}")
        print(f"Task: {task_type}")
        print(f"Scenario ID: {scenario.scenario_id}")
        print(f"Email count: {len(scenario.emails)}")
        print(f"Priority order: {scenario.priority_order}")
        
        for email in scenario.emails:
            print(f"\n  [{email.id}] {email.subject[:50]}...")
            print(f"       From: {email.sender_name} ({email.sender_importance})")
            
        for gt in scenario.ground_truth:
            print(f"  GT[{gt.email_id}]: {gt.correct_category}, {gt.correct_priority}, rank={gt.priority_rank}")
