"""
Email templates for scenario generation.

Contains realistic email patterns across 5 categories:
- URGENT: Immediate action required
- ACTION_REQUIRED: Needs response but not immediate
- INFO: Informational/FYI
- SPAM: Marketing/promotional
- PERSONAL: Personal/social
"""

from dataclasses import dataclass
from typing import List, Dict
import random


@dataclass
class EmailTemplate:
    """Template for generating synthetic emails."""
    category: str
    priority: str
    disposition: str
    subject_templates: List[str]
    body_templates: List[str]
    sender_patterns: List[Dict]
    urgency_score_range: tuple  # (min, max)
    needs_response: bool = False


# Sender profile templates
SENDERS = {
    "boss": {
        "names": ["Sarah Chen", "Michael Roberts", "Jennifer Walsh", "David Kim"],
        "domains": ["company.com"],
        "importance": "VIP",
        "titles": ["CEO", "VP Engineering", "Director", "CTO"]
    },
    "client": {
        "names": ["John Smith", "Emily Davis", "Robert Wilson", "Lisa Anderson"],
        "domains": ["clientcorp.com", "bigclient.io", "enterprise.net"],
        "importance": "VIP",
        "titles": ["Account Manager", "Project Lead", "Senior Director"]
    },
    "colleague": {
        "names": ["Alex Johnson", "Sam Taylor", "Chris Brown", "Jordan Lee", "Casey Martinez"],
        "domains": ["company.com"],
        "importance": "KNOWN",
        "titles": ["Engineer", "Designer", "PM", "Analyst"]
    },
    "ops_alerts": {
        "names": ["System Monitor", "Ops Alert", "Infrastructure Bot", "Monitoring"],
        "domains": ["alerts.company.com", "monitoring.company.com"],
        "importance": "VIP",
        "titles": ["Automated System"]
    },
    "hr": {
        "names": ["HR Team", "People Ops", "Benefits Admin"],
        "domains": ["company.com"],
        "importance": "KNOWN",
        "titles": ["HR Manager", "People Partner"]
    },
    "newsletter": {
        "names": ["TechCrunch", "Morning Brew", "The Hustle", "Product Hunt"],
        "domains": ["newsletter.techcrunch.com", "morningbrew.com", "thehustle.co"],
        "importance": "EXTERNAL",
        "titles": ["Newsletter"]
    },
    "marketing": {
        "names": ["Special Offers", "Sales Team", "Promotions"],
        "domains": ["marketing.randomstore.com", "offers.saascompany.io"],
        "importance": "EXTERNAL",
        "titles": ["Marketing"]
    },
    "unknown_internal": {
        "names": ["New Hire", "Intern", "Contractor"],
        "domains": ["company.com"],
        "importance": "UNKNOWN",
        "titles": ["New Team Member"]
    },
    "friend": {
        "names": ["Mike", "Jessica", "Tom", "Rachel", "Kevin"],
        "domains": ["gmail.com", "yahoo.com", "outlook.com"],
        "importance": "EXTERNAL",
        "titles": ["Friend"]
    }
}


# Email templates by category
URGENT_TEMPLATES = EmailTemplate(
    category="URGENT",
    priority="HIGH",
    disposition="RESPOND",
    subject_templates=[
        "URGENT: {topic}",
        "CRITICAL: {topic}",
        "IMMEDIATE ACTION REQUIRED: {topic}",
        "EMERGENCY: {topic}",
        "[P0] {topic}",
        "ASAP: {topic}",
        "Time-sensitive: {topic} - Due in {hours} hours",
    ],
    body_templates=[
        "This requires your immediate attention.\n\n{details}\n\nPlease respond ASAP.",
        "CRITICAL ALERT:\n\n{details}\n\nImmediate action required. This is blocking {affected}.",
        "Hi,\n\nThis is extremely urgent. {details}\n\nWe need your input within the next {hours} hours.\n\nThanks",
        "Priority escalation:\n\n{details}\n\nThe deadline is {deadline}. Please acknowledge receipt.",
        "[ALERT] {details}\n\nThis is impacting production/customers. Need immediate response.",
    ],
    sender_patterns=["boss", "client", "ops_alerts", "colleague"],
    urgency_score_range=(0.85, 1.0),
    needs_response=True
)

ACTION_REQUIRED_TEMPLATES = EmailTemplate(
    category="ACTION_REQUIRED",
    priority="MEDIUM",
    disposition="RESPOND",
    subject_templates=[
        "Action needed: {topic}",
        "Please review: {topic}",
        "Approval required: {topic}",
        "Your input needed on {topic}",
        "RE: {topic} - waiting on your response",
        "Follow-up: {topic}",
        "Request: {topic}",
        "Reminder: {topic} due {date}",
    ],
    body_templates=[
        "Hi,\n\nCould you please {action}?\n\n{details}\n\nLet me know if you have any questions.\n\nThanks!",
        "Hey,\n\nJust following up on {topic}. We need your {deliverable} by {date}.\n\n{details}\n\nBest",
        "Hi team,\n\n{details}\n\nPlease review and approve when you get a chance.\n\nDeadline: {date}",
        "Hello,\n\nI need your feedback on {topic}.\n\n{details}\n\nCan you take a look this week?",
        "Quick request: {details}\n\nNo rush, but hoping to wrap this up by {date}.",
    ],
    sender_patterns=["colleague", "boss", "client", "hr"],
    urgency_score_range=(0.5, 0.84),
    needs_response=True
)

INFO_TEMPLATES = EmailTemplate(
    category="INFO",
    priority="LOW",
    disposition="ARCHIVE",
    subject_templates=[
        "FYI: {topic}",
        "Update: {topic}",
        "[Info] {topic}",
        "Weekly update: {topic}",
        "Status report: {topic}",
        "Heads up: {topic}",
        "News: {topic}",
        "Announcement: {topic}",
    ],
    body_templates=[
        "Hi all,\n\nJust wanted to share an update on {topic}.\n\n{details}\n\nNo action needed.\n\nBest",
        "FYI - {details}\n\nLet me know if you have questions, but no response required.",
        "Team,\n\nHere's the weekly summary:\n\n{details}\n\nHave a great week!",
        "Quick update:\n\n{details}\n\nThis is just for your awareness.",
        "Hi,\n\nSharing this for visibility: {details}\n\nNo action required on your end.",
    ],
    sender_patterns=["colleague", "hr", "boss"],
    urgency_score_range=(0.1, 0.4),
    needs_response=False
)

SPAM_TEMPLATES = EmailTemplate(
    category="SPAM",
    priority="LOW",
    disposition="ARCHIVE",
    subject_templates=[
        "[SALE] EXCLUSIVE OFFER: {product}",
        "You won't believe this deal on {product}!",
        "Limited time: {discount}% off {product}",
        "Don't miss out: {product}",
        "Act now: {product} sale ends today!",
        "Free trial: {product}",
        "Unlock your potential with {product}",
        "{name}, we miss you! Here's {discount}% off",
    ],
    body_templates=[
        "Hi {name},\n\nWe have an incredible offer just for you!\n\n{product} is now {discount}% off!\n\n[SHOP NOW]\n\nUnsubscribe",
        "LIMITED TIME OFFER!\n\n{details}\n\nUse code SAVE{discount} at checkout.\n\nClick here to claim your discount.",
        "Dear valued customer,\n\n{details}\n\nThis offer expires in 24 hours!\n\n[BUY NOW]",
        "You're missing out!\n\n{details}\n\nJoin thousands of satisfied customers.\n\n[GET STARTED FREE]",
        "[GIFT] Special gift inside!\n\n{details}\n\nLimit one per customer. Act fast!",
    ],
    sender_patterns=["marketing", "newsletter"],
    urgency_score_range=(0.0, 0.15),
    needs_response=False
)

PERSONAL_TEMPLATES = EmailTemplate(
    category="PERSONAL",
    priority="LOW",
    disposition="DEFER",
    subject_templates=[
        "Lunch today?",
        "Weekend plans?",
        "Happy Birthday! [CAKE]",
        "Catching up",
        "Coffee sometime?",
        "Long time no talk!",
        "Quick question (personal)",
        "Re: Plans for {event}",
    ],
    body_templates=[
        "Hey!\n\nWanna grab lunch today? I'm free around noon.\n\nLet me know!",
        "Hi!\n\n{details}\n\nNo rush to respond - just whenever you have a moment.\n\nTake care!",
        "Hey {name}!\n\nHappy birthday! Hope you have an amazing day! [SALE]\n\nLet's celebrate soon!",
        "Hey,\n\nIt's been a while! {details}\n\nWould love to catch up when you're free.",
        "Quick personal note: {details}\n\nText me when you get a chance!",
    ],
    sender_patterns=["friend", "colleague"],
    urgency_score_range=(0.15, 0.35),
    needs_response=False
)


# All templates grouped by category
ALL_TEMPLATES = {
    "URGENT": URGENT_TEMPLATES,
    "ACTION_REQUIRED": ACTION_REQUIRED_TEMPLATES,
    "INFO": INFO_TEMPLATES,
    "SPAM": SPAM_TEMPLATES,
    "PERSONAL": PERSONAL_TEMPLATES,
}


# Topic/variable pools for template filling
TOPICS = {
    "urgent": [
        "Production server down",
        "Security breach detected",
        "Customer escalation from {client}",
        "Critical bug in release",
        "Database outage",
        "Payment system failure",
        "API rate limit exceeded",
        "DDoS attack detected",
        "Data loss incident",
        "Compliance violation",
    ],
    "action": [
        "Q4 budget approval",
        "Code review for PR #{number}",
        "Design review meeting",
        "Contract renewal",
        "Performance review feedback",
        "Project timeline update",
        "Resource allocation request",
        "Vendor selection decision",
        "Feature prioritization",
        "Hiring decision for {role}",
    ],
    "info": [
        "Team offsite next month",
        "New company policy",
        "Office closure for holiday",
        "Product launch update",
        "Quarterly results",
        "New team member introduction",
        "Infrastructure upgrade",
        "Process improvement update",
        "Training resources available",
        "Company all-hands recap",
        "Parking lot maintenance notice",
        "Updated expense policy",
        "New Slack channels available",
        "Building security update",
        "IT system maintenance window",
    ],
    "products": [
        "Cloud Storage Pro",
        "AI Assistant Plus",
        "Project Management Suite",
        "CRM Enterprise",
        "Analytics Dashboard",
        "Security Scanner",
        "DevOps Toolkit",
        "Design System",
        "Workflow Automation",
        "Data Pipeline Manager",
    ],
    "events": [
        "the conference",
        "Sarah's wedding",
        "team dinner",
        "holiday party",
        "book club",
        "hiking trip",
        "the offsite",
        "the workshop",
    ],
}


# More realistic body content for better training
REALISTIC_BODIES = {
    "urgent": [
        "Hi team,\n\nWe're seeing a critical issue in production. The main API is returning 500 errors for approximately 30% of requests. This started at 2:45 PM and is affecting customer-facing services.\n\nI need someone to jump on this immediately. Current impact:\n- Payment processing is failing\n- Users cannot complete checkouts\n- We're getting escalations from support\n\nPlease acknowledge and let me know your ETA.",
        "ALERT: Our monitoring detected unusual activity on the production database. We're seeing:\n- 10x normal query volume\n- CPU at 95%\n- Memory pressure alerts\n\nThis needs immediate investigation. I've already paged the on-call but need backup. Can you join the incident bridge?",
        "The client demo scheduled for 3 PM today is in jeopardy. The staging environment is completely down and we cannot present the new features. This is a $2M deal on the line.\n\nI need:\n1. Someone to investigate staging ASAP\n2. Backup plan if we can't fix in 1 hour\n3. Comms to the client if needed\n\nPlease respond immediately.",
    ],
    "action": [
        "Hi,\n\nI've put together the Q4 planning document and need your input before the leadership review on Friday.\n\nSpecifically, I need:\n- Your team's capacity estimates\n- Top 3 priorities for next quarter\n- Any dependencies on other teams\n\nCan you review and add your section by Thursday EOD?\n\nThanks!",
        "Hey,\n\nThe PR for the authentication refactor is ready for review: github.com/org/repo/pull/1234\n\nThis is a significant change that touches:\n- Login flow\n- Session management\n- Token refresh logic\n\nWould appreciate a thorough review given the security implications. No rush, but hoping to merge by end of week.\n\nThanks!",
        "Following up on our conversation about the new hire. HR needs the final decision by Wednesday to extend the offer.\n\nMy recommendation is to proceed - the candidate showed strong technical skills and good culture fit. However, I want your sign-off given the budget implications.\n\nLet me know your thoughts.",
    ],
    "info": [
        "Hi all,\n\nJust wanted to share that we successfully completed the database migration over the weekend. All systems are running normally.\n\nKey changes:\n- 40% improvement in query performance\n- New backup system active\n- Monitoring dashboards updated\n\nNo action needed on your end. Let me know if you notice anything unusual.\n\nBest,",
        "Team,\n\nHere's the weekly engineering update:\n\n[DONE] Completed: Payment v2 launch, Mobile app 3.2 release\n[IN_PROGRESS] In Progress: Search improvements, Analytics dashboard\n[UPCOMING] Upcoming: Q4 planning, Security audit\n\nGreat work everyone! No blockers to report.\n\nHave a good weekend!",
        "FYI - The new coffee machine has been installed in the 3rd floor kitchen. It's the fancy one with oat milk options.\n\nAlso, reminder that the office will be closed next Monday for the holiday. Make sure to take your laptops home if you need them.\n\nCheers!",
    ],
}


def get_random_sender(sender_type: str, rng: random.Random) -> dict:
    """Generate a random sender from the given type."""
    profile = SENDERS[sender_type]
    name = rng.choice(profile["names"])
    domain = rng.choice(profile["domains"])
    
    # Create email address
    email_name = name.lower().replace(" ", ".") if " " in name else name.lower()
    email = f"{email_name}@{domain}"
    
    return {
        "name": name,
        "email": email,
        "importance": profile["importance"],
        "title": rng.choice(profile["titles"]),
    }


def fill_template(template: str, rng: random.Random, context: dict = None) -> str:
    """Fill in template variables with random or provided values."""
    context = context or {}
    
    # Default substitutions
    defaults = {
        "topic": rng.choice(TOPICS["action"]),
        "product": rng.choice(TOPICS["products"]),
        "details": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "hours": str(rng.randint(1, 4)),
        "date": f"2024-01-{rng.randint(15, 31)}",
        "deadline": f"EOD today",
        "affected": "multiple customers",
        "discount": str(rng.choice([10, 15, 20, 25, 30, 40, 50])),
        "name": "there",
        "number": str(rng.randint(100, 999)),
        "role": rng.choice(["Senior Engineer", "Designer", "PM", "Analyst"]),
        "client": rng.choice(["Acme Corp", "TechGiant", "StartupXYZ"]),
        "event": rng.choice(TOPICS["events"]),
        "action": "review the attached document",
        "deliverable": "feedback",
    }
    
    # Merge with provided context
    substitutions = {**defaults, **context}
    
    # Fill in all placeholders
    result = template
    for key, value in substitutions.items():
        result = result.replace(f"{{{key}}}", str(value))
    
    return result
