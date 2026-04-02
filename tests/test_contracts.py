from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from client import EmailTriageEnvSync
from inference import normalize_ranking_decision
from server.app import app
from server.environment import EmailTriageEnvironment
from server.models import Email, EmailTriageAction
from server.graders import grade_episode
from server.scenarios.generator import TASK_CONFIG
from server.scenarios.generator import generate_scenario


def _build_full_triage_action(gt):
    return EmailTriageAction(
        email_id=gt.email_id,
        priority=gt.correct_priority,
        category=gt.correct_category,
        disposition=gt.correct_disposition,
        response_draft="I'll handle this right away." if gt.needs_response else None,
    )


class StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class StubHTTPClient:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def post(self, path, json):
        self.requests.append((path, json))
        return StubResponse(self.payload)

    def close(self):
        return None


def test_ranking_step_returns_immediate_reward():
    env = EmailTriageEnvironment()
    env.reset(task_type="ranking", seed=42)
    scenario = env.get_ground_truth()

    action = EmailTriageAction(
        email_id=scenario.priority_order[0],
        ranking=scenario.priority_order,
    )
    result = env.step(action)

    assert result.done is True
    assert result.reward == pytest.approx(1.0)
    assert result.cumulative_reward == pytest.approx(1.0)
    assert result.last_action_result == "correct"


def test_full_triage_observation_tracks_incremental_and_cumulative_reward():
    env = EmailTriageEnvironment()
    env.reset(task_type="full_triage", seed=42)
    scenario = env.get_ground_truth()

    first = env.step(_build_full_triage_action(scenario.ground_truth[0]))
    second = env.step(_build_full_triage_action(scenario.ground_truth[1]))

    assert first.reward > 0.0
    assert first.cumulative_reward == pytest.approx(first.reward)
    assert second.reward > 0.0
    assert second.cumulative_reward == pytest.approx(first.reward + second.reward)


def test_tasks_endpoint_matches_task_config():
    client = TestClient(app)
    response = client.get("/tasks")
    response.raise_for_status()

    data = response.json()["tasks"]
    task_map = {task["name"]: task for task in data}

    assert set(task_map.keys()) == set(TASK_CONFIG.keys())
    for task_name, config in TASK_CONFIG.items():
        assert task_map[task_name]["email_count"] == config["email_count"]
        assert task_map[task_name]["difficulty"] == config["difficulty"]


def test_sync_client_forwards_ranking_field():
    payload = {
        "observation": {
            "task_id": "task_1",
            "task_type": "ranking",
            "emails": [],
            "current_time": "2024-01-15T09:00:00",
            "emails_processed": 8,
            "emails_remaining": 0,
            "time_budget_remaining": None,
            "last_action_result": "correct",
            "done": True,
            "reward": 1.0,
            "cumulative_reward": 1.0,
        },
        "reward": 1.0,
        "done": True,
        "info": {},
    }
    stub_client = StubHTTPClient(payload)

    env_client = EmailTriageEnvSync("http://localhost:8000")
    env_client._client = stub_client

    action = EmailTriageAction(email_id="e1", ranking=["e1", "e2", "e3"])
    env_client.step(action)

    assert stub_client.requests[0][1]["ranking"] == ["e1", "e2", "e3"]


def test_normalize_ranking_decision_fills_missing_ids_and_removes_duplicates():
    emails = [
        Email(
            id="e1",
            subject="A",
            sender="a@example.com",
            sender_name="A",
            sender_importance="KNOWN",
            body="a",
            timestamp=datetime(2024, 1, 15, 9, 0, 0),
        ),
        Email(
            id="e2",
            subject="B",
            sender="b@example.com",
            sender_name="B",
            sender_importance="KNOWN",
            body="b",
            timestamp=datetime(2024, 1, 15, 9, 0, 0),
        ),
        Email(
            id="e3",
            subject="C",
            sender="c@example.com",
            sender_name="C",
            sender_importance="KNOWN",
            body="c",
            timestamp=datetime(2024, 1, 15, 9, 0, 0),
        ),
    ]

    decision = normalize_ranking_decision({"ranking": ["e2", "e2", "bad-id"]}, emails)

    assert decision["ranking"] == ["e2", "e1", "e3"]


def test_full_triage_scenario_contains_threaded_dependencies():
    scenario = generate_scenario("full_triage", seed=42)

    duplicates = [gt for gt in scenario.ground_truth if gt.duplicate_of]
    followups = [gt for gt in scenario.ground_truth if gt.recommended_after]

    assert len(duplicates) >= 3
    assert len(followups) >= 3
    assert scenario.config["thread_count"] >= 4


def test_full_triage_grader_reports_budget_and_thread_metrics():
    scenario = generate_scenario("full_triage", seed=42)
    actions = []
    for gt in scenario.ground_truth:
        actions.append(
            EmailTriageAction(
                email_id=gt.email_id,
                priority=gt.correct_priority,
                category=gt.correct_category,
                disposition=gt.correct_disposition,
                response_draft="I have this and will follow up shortly." if gt.needs_response else None,
            )
        )

    result = grade_episode(scenario, actions)

    assert "response_budget_efficiency" in result.details
    assert "thread_awareness" in result.details
    assert 0.0 <= result.details["response_budget_efficiency"] <= 1.0
    assert 0.0 <= result.details["thread_awareness"] <= 1.0


def test_duplicate_thread_reply_is_penalized_in_full_triage_reward():
    env = EmailTriageEnvironment()
    obs = env.reset(task_type="full_triage", seed=42)
    scenario = env.get_ground_truth()
    duplicate_gt = next(gt for gt in scenario.ground_truth if gt.duplicate_of)

    duplicate_action = EmailTriageAction(
        email_id=duplicate_gt.email_id,
        priority="HIGH",
        category=duplicate_gt.correct_category,
        disposition="RESPOND",
        response_draft="Replying separately even though this is redundant.",
    )
    result = env.step(duplicate_action)

    assert result.reward < 0.3
