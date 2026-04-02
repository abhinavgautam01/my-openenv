import pytest
from pathlib import Path
import sys

# Add the parent directory to the path to import from the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.models import EmailTriageObservation, EmailTriageAction
from server.environment import EmailTriageEnvironment
from server.scenarios.generator import generate_scenario
from server.graders import TaskGrader, grade_episode


class TestScenarioGenerator:
    def test_classification_scenario(self):
        """Test classification scenario generation."""
        scenario = generate_scenario("classification", seed=42)
        
        assert scenario.task_type == "classification"
        assert len(scenario.emails) == 1
        assert scenario.ground_truth[0].email_id == scenario.emails[0].id
    
    def test_ranking_scenario(self):
        """Test ranking scenario generation."""
        scenario = generate_scenario("ranking", seed=42)
        
        assert scenario.task_type == "ranking"
        assert len(scenario.emails) == 8  # Updated from 5 to 8 (Phase 2)
        assert len(scenario.ground_truth) == 8
        assert len(scenario.priority_order) == 8
    
    def test_full_triage_scenario(self):
        """Test full triage scenario generation."""
        scenario = generate_scenario("full_triage", seed=42)
        
        assert scenario.task_type == "full_triage"
        assert len(scenario.emails) == 25  # Updated from 15 to 25 (Phase 2)
        assert scenario.config["time_budget"] == 180.0
        assert scenario.config["thread_count"] >= 4
        assert len(scenario.config["ideal_response_ids"]) == scenario.config["max_responses"]


class TestEnvironment:
    def test_reset_returns_observation(self):
        """Test that reset returns valid observation."""
        env = EmailTriageEnvironment()
        obs = env.reset(task_type="classification", seed=42)
        
        assert isinstance(obs, EmailTriageObservation)
        assert obs.emails_remaining > 0
        assert obs.done == False
        assert obs.reward == 0.0
    
    def test_step_processes_action(self):
        """Test that step processes action correctly."""
        env = EmailTriageEnvironment()
        obs = env.reset(task_type="classification", seed=42)
        
        email = obs.emails[0]
        action = EmailTriageAction(email_id=email.id, category="INFO")
        result = env.step(action)
        
        assert isinstance(result, EmailTriageObservation)
        assert result.emails_processed == 1
        assert result.done == True  # Classification task ends after 1 email
    
    def test_episode_ends_correctly(self):
        """Test that episode ends when all emails processed."""
        env = EmailTriageEnvironment()
        obs = env.reset(task_type="classification", seed=42)
        
        email = obs.emails[0]
        action = EmailTriageAction(email_id=email.id, category="INFO")
        final_obs = env.step(action)
        
        assert final_obs.done == True
        assert final_obs.emails_processed == len(obs.emails)


class TestGraders:
    def test_classification_grader_correct(self):
        """Test classification grader with correct action."""
        scenario = generate_scenario("classification", seed=42)
        gt = scenario.ground_truth[0]
        
        action = EmailTriageAction(
            email_id=gt.email_id,
            category=gt.correct_category,
        )
        
        result = grade_episode(scenario, [action])
        assert result.score == 1.0
    
    def test_classification_grader_wrong(self):
        """Test classification grader with wrong action."""
        scenario = generate_scenario("classification", seed=42)
        gt = scenario.ground_truth[0]
        
        # Use wrong category
        action = EmailTriageAction(
            email_id=gt.email_id,
            category="SPAM" if gt.correct_category != "SPAM" else "INFO",
        )
        
        result = grade_episode(scenario, [action])
        assert result.score == 0.0
    
    def test_ranking_grader_perfect(self):
        """Test ranking grader with perfect order."""
        scenario = generate_scenario("ranking", seed=42)
        
        # Use explicit ranking instead of individual priority actions
        actions = [EmailTriageAction(
            email_id=scenario.priority_order[0],
            ranking=scenario.priority_order
        )]
        
        result = grade_episode(scenario, actions)
        
        # Perfect ranking should score high
        assert result.score >= 0.8
    
    def test_full_triage_grader(self):
        """Test full triage grader."""
        scenario = generate_scenario("full_triage", seed=42)
        
        # Create reasonable actions
        actions = []
        for i, email in enumerate(scenario.emails[:3]):  # Only process first 3
            actions.append(EmailTriageAction(
                email_id=email.id,
                priority="HIGH",
                category="ACTION_REQUIRED",
                disposition="RESPOND"
            ))
        
        result = grade_episode(scenario, actions)
        assert 0.0 <= result.score <= 1.0


class TestIntegration:
    def test_full_episode_classification(self):
        """Test complete classification episode."""
        env = EmailTriageEnvironment()
        obs = env.reset(task_type="classification", seed=42)
        
        scenario = env.get_ground_truth()
        gt = scenario.ground_truth[0]
        
        action = EmailTriageAction(
            email_id=gt.email_id,
            category=gt.correct_category,
        )
        result = env.step(action)
        
        assert result.done == True
        assert result.reward > 0  # Should get positive reward for correct action
    
    def test_full_episode_ranking(self):
        """Test complete ranking episode."""
        env = EmailTriageEnvironment()
        obs = env.reset(task_type="ranking", seed=42)
        
        scenario = env.get_ground_truth()
        
        # Use single ranking action instead of processing each email
        all_email_ids = [email.id for email in obs.emails]
        ranking_action = EmailTriageAction(
            email_id=all_email_ids[0],
            ranking=scenario.priority_order
        )
        result = env.step(ranking_action)
        
        assert result.done == True
