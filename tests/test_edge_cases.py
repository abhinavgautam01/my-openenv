"""
Additional edge case tests for Email Triage Environment.
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import EmailTriageAction, Category, Priority
from server.environment import EmailTriageEnvironment
from server.scenarios.generator import generate_scenario
from server.graders import TaskGrader, kendall_tau_distance


class TestEdgeCases:
    """Test edge cases not covered in main test suite."""
    
    def test_empty_response_draft_counts_as_none(self):
        """Empty response draft should be treated as None."""
        env = EmailTriageEnvironment()
        obs = env.reset(task_type="full_triage", seed=42)
        
        email_id = obs.emails[0].id
        action = EmailTriageAction(
            email_id=email_id,
            priority="HIGH",
            category="URGENT",
            disposition="RESPOND",
            response_draft=""  # Empty string
        )
        
        result = env.step(action)
        # Should not crash
        assert result.done == False or result.done == True
    
    # DISABLED: test_duplicate_email_processing_rejected - not applicable with new task structure
    # def test_duplicate_email_processing_rejected(self):
    #     pass

    def test_action_after_episode_done_rejected(self):
        """Actions after episode ends should raise error."""
        env = EmailTriageEnvironment()
        obs = env.reset(task_type="classification", seed=42)
        
        # Complete the episode
        email_id = obs.emails[0].id
        action = EmailTriageAction(email_id=email_id, category="INFO")
        env.step(action)
        
        # Try to act after done
        with pytest.raises(RuntimeError, match="already done"):
            env.step(action)
    
    def test_time_budget_decreases_over_time(self):
        """Time budget should decrease in full_triage mode."""
        import time
        env = EmailTriageEnvironment()
        obs = env.reset(task_type="full_triage", seed=42)
        
        initial_time = obs.time_budget_remaining
        
        # Wait a bit
        time.sleep(0.5)
        
        # Take an action
        email_id = obs.emails[0].id
        action = EmailTriageAction(
            email_id=email_id,
            priority="HIGH",
            category="URGENT",
            disposition="RESPOND"
        )
        obs2 = env.step(action)
        
        # Time should have decreased
        assert obs2.time_budget_remaining < initial_time
    
    def test_very_long_response_draft_accepted_within_limit(self):
        """Response drafts up to 10k chars should be accepted."""
        env = EmailTriageEnvironment()
        obs = env.reset(task_type="full_triage", seed=42)
        
        email_id = obs.emails[0].id
        long_response = "A" * 9999  # Just under limit
        
        action = EmailTriageAction(
            email_id=email_id,
            priority="HIGH",
            category="URGENT",
            disposition="RESPOND",
            response_draft=long_response
        )
        
        # Should not crash
        result = env.step(action)
        assert result is not None
    
    def test_response_draft_exceeding_limit_rejected(self):
        """Response drafts over 10k chars should be rejected by Pydantic."""
        with pytest.raises(Exception):  # ValidationError
            EmailTriageAction(
                email_id="e1",
                priority="HIGH",
                category="URGENT",
                disposition="RESPOND",
                response_draft="A" * 10001
            )
    
    def test_invalid_email_id_format_rejected(self):
        """Email IDs over 100 chars should be rejected."""
        with pytest.raises(Exception):  # ValidationError
            EmailTriageAction(
                email_id="e" * 101,  # Over limit
                category="INFO"
            )


class TestGraderEdgeCases:
    """Test grader edge cases."""
    
    def test_ranking_with_incomplete_actions(self):
        """Grader should handle when agent submits incomplete ranking."""
        scenario = generate_scenario("ranking", seed=42)
        
        # Submit incomplete ranking (missing some email IDs)
        incomplete_ranking = [scenario.emails[i].id for i in range(3)]  # Only 3 out of 8 emails
        
        actions = [
            EmailTriageAction(
                email_id=scenario.emails[0].id, 
                ranking=incomplete_ranking
            )
        ]
        
        grader = TaskGrader(scenario)
        result = grader.grade(actions)
        
        # Should return near-zero score (strict open interval)
        assert 0.0 < result.score < 0.01
        assert "Ranking incomplete" in result.details.get("error", "")
    
    def test_ranking_with_all_same_priority(self):
        """Grader should handle when agent assigns same priority to all."""
        scenario = generate_scenario("ranking", seed=42)
        
        # Assign MEDIUM to everything
        actions = [
            EmailTriageAction(email_id=email.id, priority="MEDIUM")
            for email in scenario.emails
        ]
        
        grader = TaskGrader(scenario)
        result = grader.grade(actions)
        
        # Score should be low but not crash
        assert 0.0 <= result.score <= 1.0
    
    def test_kendall_tau_with_reversed_ranking(self):
        """Completely reversed ranking should give score near 0."""
        ranking1 = ["a", "b", "c", "d", "e"]
        ranking2 = ["e", "d", "c", "b", "a"]
        
        distance = kendall_tau_distance(ranking1, ranking2)
        # Fully reversed should be distance = 1.0
        assert distance == 1.0
    
    def test_kendall_tau_with_mismatched_items_raises_error(self):
        """Rankings with different items should raise ValueError."""
        ranking1 = ["a", "b", "c"]
        ranking2 = ["d", "e", "f"]
        
        with pytest.raises(ValueError, match="same items"):
            kendall_tau_distance(ranking1, ranking2)
    
    def test_classification_with_wrong_action_count(self):
        """Classification with multiple actions should fail."""
        scenario = generate_scenario("classification", seed=42)
        
        # Submit 2 actions instead of 1
        actions = [
            EmailTriageAction(email_id=scenario.emails[0].id, category="INFO"),
            EmailTriageAction(email_id=scenario.emails[0].id, category="URGENT"),
        ]
        
        grader = TaskGrader(scenario)
        result = grader.grade(actions)
        
        assert 0.0 < result.score < 0.01
        assert "Expected 1 action" in result.details.get("error", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
