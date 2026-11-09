"""Breaks caught: self-confirmed success, test-set writeback and harmful-skill retention."""
import pytest


def store(tmp_path):
    from pd_vqa.memory import SkillMemory
    return SkillMemory(tmp_path / "skills.sqlite")


def test_unverified_candidate_cannot_be_retrieved(tmp_path):
    memory = store(tmp_path)
    sid = memory.propose("gait", "low_quality", "Select an unobstructed walking interval", ["trace1"])
    assert sid
    assert memory.retrieve("gait", "low_quality") == []


def test_external_verified_success_promotes_but_failure_lowers_utility(tmp_path):
    memory = store(tmp_path)
    sid = memory.propose("gait", "low_quality", "Select an unobstructed walking interval", ["trace1"])
    memory.feedback(sid, success=True, source="clinician_review", reference_id="review1", split="train")
    assert memory.retrieve("gait", "low_quality")[0]["skill_id"] == sid
    memory.feedback(sid, success=False, source="clinician_review", reference_id="review2", split="train")
    memory.feedback(sid, success=False, source="clinician_review", reference_id="review3", split="train")
    assert memory.retrieve("gait", "low_quality") == []


def test_consensus_is_not_external_ground_truth(tmp_path):
    memory = store(tmp_path)
    sid = memory.propose("gait", "low_quality", "Inspect a different segment", ["t"])
    with pytest.raises(ValueError):
        memory.feedback(sid, success=True, source="agent_consensus", reference_id="c", split="train")


def test_test_split_cannot_mutate_memory(tmp_path):
    memory = store(tmp_path)
    sid = memory.propose("gait", "low_quality", "Inspect a different segment", ["t"])
    with pytest.raises(ValueError):
        memory.feedback(sid, success=True, source="benchmark_label", reference_id="y1", split="test")


def test_duplicate_feedback_cannot_inflate_utility(tmp_path):
    memory = store(tmp_path)
    sid = memory.propose("gait", "low_quality", "Inspect a different segment", ["t"])
    memory.feedback(sid,success=True,source="tool_validation",reference_id="v1",split="train")
    with pytest.raises(ValueError):
        memory.feedback(sid,success=True,source="tool_validation",reference_id="v1",split="train")


def test_measurement_quality_feedback_does_not_become_diagnostic_skill(tmp_path):
    memory = store(tmp_path)
    sid = memory.propose("gait", "low_quality", "Recompute keypoints", ["t"], scope="diagnostic")
    with pytest.raises(ValueError):
        memory.feedback(sid,success=True,source="tool_validation",reference_id="v1",split="train")
