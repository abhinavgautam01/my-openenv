"""
Graders for Email Triage tasks.

Each grader scores agent performance on a strict-open scale:
- final task score is always in [0.10, 0.99]
- Classification: Exact category match
- Ranking: Kendall Tau rank correlation
- Full Triage: Multi-objective weighted score
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.models import (
    EmailTriageAction, GroundTruth, ScenarioData,
    TaskType, Category, Priority
)

logger = logging.getLogger(__name__)
_MIN_TASK_SCORE = 0.10
_MAX_TASK_SCORE = 0.99


def kendall_tau_distance(ranking1: List[str], ranking2: List[str]) -> float:
    """
    Calculate Kendall Tau distance between two rankings.
    
    Returns value in [0, 1] where 0 = identical, 1 = reversed.
    
    Raises:
        ValueError: If rankings have different lengths or different items
    """
    if len(ranking1) != len(ranking2):
        raise ValueError("Rankings must have same length")
    
    if len(ranking1) == 0:
        return 0.0
    
    # Validate same items in both rankings
    set1, set2 = set(ranking1), set(ranking2)
    if set1 != set2:
        missing_in_2 = set1 - set2
        missing_in_1 = set2 - set1
        raise ValueError(
            f"Rankings must contain same items. "
            f"Missing in ranking2: {missing_in_2}, Missing in ranking1: {missing_in_1}"
        )
    
    n = len(ranking1)
    
    # Create position maps
    pos1 = {item: i for i, item in enumerate(ranking1)}
    pos2 = {item: i for i, item in enumerate(ranking2)}
    
    # Count discordant pairs
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            item_i = ranking1[i]
            item_j = ranking1[j]
            
            # Check if items exist in both rankings
            if item_i not in pos2 or item_j not in pos2:
                continue
                
            # Compare relative order
            order1 = pos1[item_i] < pos1[item_j]
            order2 = pos2[item_i] < pos2[item_j]
            
            if order1 != order2:
                discordant += 1
    
    # Maximum possible discordant pairs
    max_discordant = n * (n - 1) / 2
    
    if max_discordant == 0:
        return 0.0
    
    return discordant / max_discordant


def kendall_tau_correlation(ranking1: List[str], ranking2: List[str]) -> float:
    """
    Calculate Kendall Tau correlation between two rankings.
    
    Returns value in [-1, 1] where 1 = identical, -1 = reversed.
    """
    distance = kendall_tau_distance(ranking1, ranking2)
    # Convert distance [0, 1] to correlation [-1, 1]
    return 1 - 2 * distance


@dataclass
class GradeResult:
    """Result of grading an episode."""
    score: float  # strict range [0.10, 0.99]
    task_type: TaskType
    details: Dict  # Breakdown of score components
    passed: bool  # True if score >= threshold


class TaskGrader:
    """
    Grader for Email Triage tasks.
    
    Scores agent performance based on task type:
    - classification: Exact match accuracy
    - ranking: Kendall Tau correlation
    - full_triage: Multi-objective weighted score
    """
    
    def __init__(
        self,
        scenario: ScenarioData,
        success_threshold: float = 0.5,
    ):
        """
        Initialize grader with scenario ground truth.
        
        Args:
            scenario: Scenario with ground truth labels
            success_threshold: Minimum score to pass
        """
        self.scenario = scenario
        self.task_type = scenario.task_type
        self.success_threshold = success_threshold
        
        # Build ground truth lookup
        self._gt_map = {gt.email_id: gt for gt in scenario.ground_truth}
    
    def grade(self, actions: List[EmailTriageAction]) -> GradeResult:
        """
        Grade agent's actions against ground truth.
        
        Args:
            actions: List of actions taken by agent
            
        Returns:
            GradeResult with score and details
        """
        # Handle edge case: no actions
        if not actions:
            score = self._strict_open_score(0.0)
            return GradeResult(
                score=score,
                task_type=self.task_type,
                details={"error": "No actions provided"},
                passed=score >= self.success_threshold,
            )
        
        # Filter out invalid actions (unknown email_ids)
        valid_actions = [a for a in actions if a.email_id in self._gt_map]
        if len(valid_actions) < len(actions):
            logger.warning(f"Filtered {len(actions) - len(valid_actions)} actions with unknown email_ids")
        
        if self.task_type == "classification":
            return self._grade_classification(valid_actions)
        elif self.task_type == "ranking":
            return self._grade_ranking(valid_actions)
        elif self.task_type == "full_triage":
            return self._grade_full_triage(valid_actions)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _grade_classification(self, actions: List[EmailTriageAction]) -> GradeResult:
        """
        Grade classification task.
        
        Score = 1.0 if correct category, 0.0 otherwise.
        """
        if len(actions) != 1:
            score = self._strict_open_score(0.0)
            return GradeResult(
                score=score,
                task_type=self.task_type,
                details={"error": f"Expected 1 action, got {len(actions)}"},
                passed=score >= self.success_threshold,
            )
        
        action = actions[0]
        gt = self._gt_map.get(action.email_id)
        
        if gt is None:
            score = self._strict_open_score(0.0)
            return GradeResult(
                score=score,
                task_type=self.task_type,
                details={"error": f"Unknown email_id: {action.email_id}"},
                passed=score >= self.success_threshold,
            )
        
        # Check category match
        correct = action.category == gt.correct_category
        score = self._strict_open_score(1.0 if correct else 0.0)
        
        return GradeResult(
            score=score,
            task_type=self.task_type,
            details={
                "email_id": action.email_id,
                "agent_category": action.category,
                "correct_category": gt.correct_category,
                "correct": correct,
            },
            passed=score >= self.success_threshold,
        )
    
    def _grade_ranking(self, actions: List[EmailTriageAction]) -> GradeResult:
        """
        Grade ranking task using explicit ranking order.
        
        Expects a single action with the 'ranking' field containing ordered email IDs.
        Score = (kendall_tau + 1) / 2, normalized to [0, 1].
        """
        if len(actions) != 1:
            score = self._strict_open_score(0.0)
            return GradeResult(
                score=score,
                task_type=self.task_type,
                details={"error": f"Ranking task expects exactly 1 action with ranking field, got {len(actions)} actions"},
                passed=score >= self.success_threshold,
            )
        
        action = actions[0]
        
        # Validate explicit ranking provided
        if not action.ranking:
            score = self._strict_open_score(0.0)
            return GradeResult(
                score=score,
                task_type=self.task_type,
                details={"error": "No ranking provided in action"},
                passed=score >= self.success_threshold,
            )
        
        # Check all email IDs are included exactly once
        expected_ids = set(email.id for email in self.scenario.emails)
        provided_ids = set(action.ranking)
        
        if expected_ids != provided_ids:
            missing = expected_ids - provided_ids
            extra = provided_ids - expected_ids
            score = self._strict_open_score(0.0)
            return GradeResult(
                score=score,
                task_type=self.task_type,
                details={
                    "error": "Ranking incomplete",
                    "missing_emails": list(missing),
                    "extra_emails": list(extra),
                    "expected_count": len(expected_ids),
                    "provided_count": len(provided_ids)
                },
                passed=score >= self.success_threshold,
            )
        
        # Ground truth order (sorted by priority)
        gt_order = self.scenario.priority_order
        
        # Agent's explicit order
        agent_order = action.ranking
        
        # Calculate Kendall Tau correlation
        try:
            tau = kendall_tau_correlation(agent_order, gt_order)
        except ValueError as e:
            score = self._strict_open_score(0.0)
            return GradeResult(
                score=score,
                task_type=self.task_type,
                details={"error": f"Kendall tau calculation failed: {e}"},
                passed=score >= self.success_threshold,
            )
        
        # Normalize to [0, 1]
        score = (tau + 1) / 2
        score = self._strict_open_score(score)
        
        return GradeResult(
            score=score,
            task_type=self.task_type,
            details={
                "agent_ranking": agent_order,
                "ground_truth_ranking": gt_order,
                "kendall_tau": tau,
                "normalized_score": score,
            },
            passed=score >= self.success_threshold,
        )
    
    def _grade_full_triage(self, actions: List[EmailTriageAction]) -> GradeResult:
        """
        Grade full triage task with multi-objective scoring.
        
        Weights:
        - Priority correctness: 30%
        - Category correctness: 20%
        - Disposition correctness: 20%
        - Response quality: 10%
        - Response budget efficiency: 10%
        - Thread awareness: 10%
        """
        expected_count = len(self.scenario.emails)
        
        if len(actions) < expected_count:
            # Partial completion penalty
            completion_penalty = len(actions) / expected_count
        else:
            completion_penalty = 1.0
        
        # Score each component
        priority_scores = []
        category_scores = []
        disposition_scores = []
        response_scores = []
        action_map = {action.email_id: action for action in actions}
        
        for action in actions:
            gt = self._gt_map.get(action.email_id)
            if gt is None:
                continue
            
            # Priority
            if action.priority == gt.correct_priority:
                priority_scores.append(1.0)
            elif self._is_adjacent_priority(action.priority, gt.correct_priority):
                priority_scores.append(0.5)
            else:
                priority_scores.append(0.0)
            
            # Category
            if action.category == gt.correct_category:
                category_scores.append(1.0)
            else:
                category_scores.append(0.0)
            
            # Disposition
            if action.disposition == gt.correct_disposition:
                disposition_scores.append(1.0)
            elif action.disposition is not None:
                disposition_scores.append(0.0)
            
            # Response quality - enhanced scoring
            if gt.needs_response and action.disposition == "RESPOND":
                response_score = self._score_response(action.response_draft, gt)
                response_scores.append(response_score)
        
        # Calculate averages (or 0 if empty)
        def safe_avg(scores):
            return sum(scores) / len(scores) if scores else 0.0
        
        priority_avg = safe_avg(priority_scores)
        category_avg = safe_avg(category_scores)
        disposition_avg = safe_avg(disposition_scores)
        response_avg = safe_avg(response_scores) if response_scores else 1.0  # No penalty if no responses needed
        budget_efficiency = self._score_response_budget_efficiency(action_map)
        thread_awareness = self._score_thread_awareness(action_map)
        
        # Weighted score
        score = (
            0.30 * priority_avg +
            0.20 * category_avg +
            0.20 * disposition_avg +
            0.10 * response_avg +
            0.10 * budget_efficiency +
            0.10 * thread_awareness
        ) * completion_penalty
        score = self._strict_open_score(score)
        
        return GradeResult(
            score=score,
            task_type=self.task_type,
            details={
                "emails_processed": len(actions),
                "emails_expected": expected_count,
                "completion_rate": completion_penalty,
                "priority_accuracy": priority_avg,
                "category_accuracy": category_avg,
                "disposition_accuracy": disposition_avg,
                "response_quality": response_avg,
                "response_budget_efficiency": budget_efficiency,
                "thread_awareness": thread_awareness,
                "weights": {
                    "priority": 0.30,
                    "category": 0.20,
                    "disposition": 0.20,
                    "response": 0.10,
                    "response_budget": 0.10,
                    "thread_awareness": 0.10,
                },
            },
            passed=score >= self.success_threshold,
        )

    @staticmethod
    def _strict_open_score(score: float) -> float:
        """Clamp task scores to validator-safe range [0.10, 0.99]."""
        if score <= _MIN_TASK_SCORE:
            return _MIN_TASK_SCORE
        if score >= _MAX_TASK_SCORE:
            return _MAX_TASK_SCORE
        return score

    def _score_response_budget_efficiency(self, action_map: Dict[str, EmailTriageAction]) -> float:
        """Score whether the limited response budget was spent on the right emails."""
        max_responses = self.scenario.config.get("max_responses")
        ideal_ids = self.scenario.config.get("ideal_response_ids", [])
        if not max_responses or not ideal_ids:
            return 1.0

        responded_ids = {
            email_id
            for email_id, action in action_map.items()
            if action.disposition == "RESPOND"
        }
        ideal_set = set(ideal_ids)

        precision = len(responded_ids & ideal_set) / len(responded_ids) if responded_ids else 0.0
        recall = len(responded_ids & ideal_set) / len(ideal_set) if ideal_set else 1.0

        if precision + recall == 0:
            score = 0.0
        else:
            score = 2 * precision * recall / (precision + recall)

        overflow_penalty = max(0, len(responded_ids) - max_responses) * 0.1
        return max(0.0, min(1.0, score - overflow_penalty))

    def _score_thread_awareness(self, action_map: Dict[str, EmailTriageAction]) -> float:
        """Score whether related emails were handled with thread-level context."""
        scores = []

        for gt in self.scenario.ground_truth:
            action = action_map.get(gt.email_id)
            if action is None:
                continue

            if gt.duplicate_of:
                score = 1.0 if action.disposition in {"ARCHIVE", "DEFER"} else 0.0
                if action.priority == "HIGH":
                    score = min(score, 0.25)
                scores.append(score)
            elif gt.recommended_after:
                dependency_action = action_map.get(gt.recommended_after)
                if action.disposition != gt.correct_disposition:
                    scores.append(0.0)
                elif dependency_action and dependency_action.disposition == self._gt_map[gt.recommended_after].correct_disposition:
                    scores.append(1.0)
                elif dependency_action is not None:
                    scores.append(0.4)
                else:
                    scores.append(0.2)

        return sum(scores) / len(scores) if scores else 1.0
    
    def _is_adjacent_priority(self, p1: Optional[Priority], p2: Priority) -> bool:
        """Check if two priorities are adjacent."""
        if p1 is None:
            return False
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        return abs(order.get(p1, -1) - order.get(p2, -1)) == 1
    
    def _score_response(self, response: Optional[str], gt: GroundTruth) -> float:
        """
        Score the quality of a response draft using semantic analysis.
        
        Returns score in [0, 1] based on:
        - Semantic similarity to ground truth response (if available)
        - Professional language quality
        - Action consistency with disposition
        - Email-specific relevance
        """
        if not response or not response.strip():
            return 0.0
        
        response = response.strip()
        
        # Minimum length check
        if len(response) < 10:
            return 0.1
        
        score = 0.0
        
        # 1. Semantic similarity (40% weight)
        if hasattr(gt, 'expected_response') and gt.expected_response:
            semantic_score = self._compute_semantic_similarity(response, gt.expected_response)
            score += semantic_score * 0.4
        else:
            # Fallback to heuristics if no ground truth response
            score += self._score_response_heuristics(response) * 0.4
        
        # 2. Action consistency (30% weight)
        consistency_score = self._score_action_consistency(response, gt)
        score += consistency_score * 0.3
        
        # 3. Email relevance (20% weight)
        relevance_score = self._score_email_relevance(response, gt)
        score += relevance_score * 0.2
        
        # 4. Professional quality (10% weight)
        professional_score = self._score_professional_quality(response)
        score += professional_score * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _compute_semantic_similarity(self, response: str, expected: str) -> float:
        """Compute semantic similarity between response and expected response."""
        try:
            import sentence_transformers
            
            # Use a lightweight sentence transformer for similarity
            if not hasattr(self, '_similarity_model'):
                self._similarity_model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
            
            # Compute embeddings
            response_embedding = self._similarity_model.encode([response])
            expected_embedding = self._similarity_model.encode([expected])
            
            # Compute cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(response_embedding, expected_embedding)[0][0]
            
            # Map from [-1, 1] to [0, 1]
            return (similarity + 1) / 2
            
        except ImportError:
            # Fall back to simple text matching if libraries not available
            return self._simple_text_similarity(response, expected)
    
    def _simple_text_similarity(self, response: str, expected: str) -> float:
        """Simple text similarity as fallback."""
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.5
        
        intersection = response_words.intersection(expected_words)
        union = response_words.union(expected_words)
        
        # Jaccard similarity
        return len(intersection) / len(union) if union else 0.0
    
    def _score_action_consistency(self, response: str, gt: GroundTruth) -> float:
        """Score whether response is consistent with the disposition action."""
        if not hasattr(gt, 'correct_disposition'):
            return 0.5  # Neutral if no disposition available
        
        response_lower = response.lower()
        disposition = gt.correct_disposition
        
        if disposition == "RESPOND":
            # Should be a proper response
            return 1.0  # Already responding, so consistent
        elif disposition == "DELEGATE":
            # Should mention forwarding/delegation
            delegate_terms = ["forward", "delegate", "send to", "refer to", "pass to"]
            return 1.0 if any(term in response_lower for term in delegate_terms) else 0.3
        elif disposition == "ARCHIVE":
            # Shouldn't be a lengthy response
            return 0.5 if len(response) < 50 else 0.2
        elif disposition == "DEFER":
            # Should mention timing/delay
            defer_terms = ["later", "tomorrow", "next", "when", "after", "schedule"]
            return 1.0 if any(term in response_lower for term in defer_terms) else 0.3
        
        return 0.5
    
    def _score_email_relevance(self, response: str, gt: GroundTruth) -> float:
        """Score whether response references email-specific details."""
        # This would ideally check if response mentions sender, subject, or key facts
        # For now, use heuristics
        response_lower = response.lower()
        
        score = 0.5  # Base score
        
        # Check for personal references
        personal_refs = ["your", "you", "thanks for", "regarding", "about"]
        if any(ref in response_lower for ref in personal_refs):
            score += 0.3
        
        # Check for specificity (not too generic)
        if len(response.split()) > 15:  # More detailed responses
            score += 0.2
        
        return min(1.0, score)
    
    def _score_professional_quality(self, response: str) -> float:
        """Score professional language quality (legacy heuristics)."""
        response_lower = response.lower()
        
        # Professional markers
        professional_markers = [
            "thank", "please", "will", "can", "would", "hi", "hello",
            "regards", "best", "sincerely", "let me", "i'll", "happy to",
            "looking forward", "appreciate", "understand"
        ]
        marker_count = sum(1 for m in professional_markers if m in response_lower)
        prof_score = min(1.0, marker_count * 0.2)
        
        # Penalty for generic responses
        generic_markers = ["lorem ipsum", "placeholder", "todo", "tbd", "xxx"]
        if any(m in response_lower for m in generic_markers):
            prof_score -= 0.5
        
        return max(0.0, prof_score)
    
    def _score_response_heuristics(self, response: str) -> float:
        """Fallback heuristic scoring when no ground truth available."""
        score = 0.0
        response_lower = response.lower()
        
        # Length score (up to 0.4)
        if len(response) >= 20:
            score += 0.2
        if len(response) >= 50:
            score += 0.1
        if len(response) >= 100:
            score += 0.1
        
        # Professional markers (up to 0.3)
        professional_markers = [
            "thank", "please", "will", "can", "would", "hi", "hello",
            "regards", "best", "sincerely", "let me", "i'll", "happy to",
            "looking forward", "appreciate", "understand"
        ]
        marker_count = sum(1 for m in professional_markers if m in response_lower)
        score += min(0.3, marker_count * 0.1)
        
        # Action words bonus (up to 0.3)
        action_markers = [
            "schedule", "meet", "discuss", "review", "follow up",
            "investigate", "fix", "resolve", "address", "handle",
            "confirm", "approve", "send", "share", "update"
        ]
        action_count = sum(1 for m in action_markers if m in response_lower)
        score += min(0.3, action_count * 0.1)
        
        # Penalty for generic responses
        generic_markers = ["lorem ipsum", "placeholder", "todo", "tbd", "xxx"]
        if any(m in response_lower for m in generic_markers):
            score -= 0.2
        
        return score


def grade_episode(
    scenario: ScenarioData,
    actions: List[EmailTriageAction],
    success_threshold: float = 0.5,
) -> GradeResult:
    """
    Convenience function to grade an episode.
    
    Args:
        scenario: Scenario with ground truth
        actions: Agent's actions
        success_threshold: Minimum score to pass
        
    Returns:
        GradeResult with score and details
    """
    grader = TaskGrader(scenario, success_threshold)
    return grader.grade(actions)


if __name__ == "__main__":
    # Test graders
    from server.scenarios.generator import generate_scenario
    
    print("Testing graders...")
    
    # Test classification
    scenario = generate_scenario("classification", seed=42)
    email = scenario.emails[0]
    gt = scenario.ground_truth[0]
    
    # Perfect action
    perfect_action = EmailTriageAction(
        email_id=email.id,
        category=gt.correct_category,
    )
    result = grade_episode(scenario, [perfect_action])
    print(f"\nClassification (correct): score={result.score:.2f}, passed={result.passed}")
    
    # Wrong action
    wrong_action = EmailTriageAction(
        email_id=email.id,
        category="SPAM" if gt.correct_category != "SPAM" else "INFO",
    )
    result = grade_episode(scenario, [wrong_action])
    print(f"Classification (wrong): score={result.score:.2f}, passed={result.passed}")
    
    # Test ranking
    scenario = generate_scenario("ranking", seed=42)
    
    # Perfect ranking
    actions = [
        EmailTriageAction(
            email_id=eid,
            priority=scenario.ground_truth[i].correct_priority,
        )
        for i, eid in enumerate(scenario.priority_order)
    ]
    result = grade_episode(scenario, actions)
    print(f"\nRanking (correct): score={result.score:.2f}, tau={result.details['kendall_tau']:.2f}")
    
    # Reversed ranking
    reversed_actions = actions[::-1]
    result = grade_episode(scenario, reversed_actions)
    print(f"Ranking (reversed): score={result.score:.2f}, tau={result.details['kendall_tau']:.2f}")
    
    # Test full triage
    scenario = generate_scenario("full_triage", seed=42)
    
    # Perfect triage
    actions = [
        EmailTriageAction(
            email_id=gt.email_id,
            priority=gt.correct_priority,
            category=gt.correct_category,
            disposition=gt.correct_disposition,
            response_draft="I'll handle this right away." if gt.needs_response else None,
        )
        for gt in scenario.ground_truth
    ]
    result = grade_episode(scenario, actions)
    print(f"\nFull triage (correct): score={result.score:.2f}")
    print(f"  Details: {result.details}")
