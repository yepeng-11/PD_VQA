"""Breaks caught: forced consensus, irrelevant experts, unbounded retries and evidence leakage."""
import json
import math

import pytest


def components():
    # A missing implementation is reported as the first explicit failing requirement.
    import importlib.util
    assert importlib.util.find_spec("pd_vqa") is not None, "The new workflow package is missing"
    from pd_vqa.contracts import Case, Record, Task, Evidence, ExpertReport, Prediction
    from pd_vqa.workflow import Workflow, EvidenceCritic, WeightedEvidenceFusion
    return Case, Record, Task, Evidence, ExpertReport, Prediction, Workflow, EvidenceCritic, WeightedEvidenceFusion


def record(modality="gait", **kw):
    _, Record, *_ = components()
    return Record(subject_id="P1", record_id=modality+"1", modality=modality, **kw)


def evidence(modality="gait", quality=1., **kw):
    _, _, _, Evidence, *_ = components()
    suffix = kw.pop("evidence_id_suffix", "obs")
    return Evidence(evidence_id=modality+"-"+suffix, subject_id="P1", record_id=modality+"1",
                    modality=modality, kind="measurement", name="observed_motion", value=.2,
                    unit="ratio", quality=quality, source="actual_tool:v1", **kw)


def report(modality="gait", target="pd_presence", p=(.1, .9), quality=1., evidence_items=None):
    _, _, _, _, Report, Prediction, *_ = components()
    items = evidence_items if evidence_items is not None else [evidence(modality, quality)]
    pred = Prediction(target=target, labels=["healthy", "pd"], probabilities=list(p),
                      model_id="fixture_model", trained=True, evidence_ids=[x.evidence_id for x in items])
    return Report(subject_id="P1", modality=modality, record_id=modality+"1", target=target,
                  evidence=items, prediction=pred, summary="Test observation", quality=quality)


class RecordingTool:
    """Only the external perception boundary is replaced; the controller and critic remain real."""
    def __init__(self, outputs, actions=None):
        self.outputs, self.actions = outputs, actions or {}
        self.calls = []

    def analyze(self, rec, task, action=None):
        self.calls.append((rec.modality, action))
        values = self.outputs[rec.modality]
        return values[min(sum(m == rec.modality for m, _ in self.calls)-1, len(values)-1)]

    def review_options(self, rec, task, current):
        return self.actions.get(rec.modality, [])


def test_non_synchronous_different_symptoms_do_not_become_conflicts():
    *_, Critic, _ = components()
    r = Critic().assess([report("gait", "arm_swing"), report("face", "facial_motion", (.9,.1))])
    assert not any(x.kind == "semantic" for x in r.issues)


def test_same_target_disagreement_is_preserved_before_fusion():
    *_, Critic, Fusion = components()
    reports = [report("gait"), report("speech", p=(.9,.1))]
    review = Critic().assess(reports)
    assert review.status == "review_required"
    fused = Fusion().fuse(reports)
    assert len(fused.reports) == 2
    assert fused.groups["pd_presence"]["probabilities"] == pytest.approx([.5,.5])
    assert sum(fused.groups["pd_presence"]["weights"].values()) == pytest.approx(1.)


def test_quality_problem_can_trigger_review_without_disagreement():
    *_, Critic, _ = components()
    result = Critic().assess([report(quality=.1)])
    assert result.status == "review_required"
    assert any(i.kind == "reliability" for i in result.issues)


def test_invalid_probability_and_nonfinite_measurements_are_rejected():
    _, _, _, Evidence, _, Prediction, *_ = components()
    with pytest.raises(ValueError):
        Prediction("pd_presence", ["healthy","pd"], [.2,.2], "bad", True, ["e"])
    with pytest.raises(ValueError):
        Evidence("e","P1","r1","gait","measurement","v", math.nan, "ratio",1.,"tool")


def test_reassessment_uses_changed_action_and_keeps_original_evidence():
    Case, _, Task, _, _, _, Workflow, *_ = components()
    initial = report(quality=.1)
    updated = report(evidence_items=[evidence(evidence_id_suffix="new")])
    tools = RecordingTool({"gait":[initial,updated]}, {"gait":[{"operation":"window", "start":2., "end":5.}]})
    result = Workflow(tools, max_reviews=2).run(Case("P1", [record()]), Task("证据是否充分？", "pd_presence", ["gait"]))
    assert result.review_rounds == 1
    assert len(result.history) == 2
    assert result.history[0][0].quality == .1
    assert result.history[1][0].quality == 1.
    assert tools.calls[1][1]["start"] == 2.


def test_repeated_identical_recheck_is_not_executed_forever():
    Case, _, Task, _, _, _, Workflow, *_ = components()
    tools = RecordingTool({"gait":[report(quality=.1)]}, {"gait":[{"operation":"window","start":1.,"end":3.}]})
    result = Workflow(tools, max_reviews=5).run(Case("P1",[record()]), Task("判断", "pd_presence", ["gait"]))
    assert len(tools.calls) == 2
    assert result.answer.status == "insufficient_evidence"
    assert result.stop_reason == "no_new_action"


def test_unresolved_disagreement_cannot_be_overwritten_by_majority():
    Case, _, Task, _, _, _, Workflow, *_ = components()
    tools = RecordingTool({"gait":[report()], "face":[report("face")], "speech":[report("speech",p=(.9,.1))]})
    result = Workflow(tools).run(Case("P1",[record(m) for m in ["gait","face","speech"]]),
                               Task("是否患病", "pd_presence", ["gait","face","speech"]))
    assert result.answer.status == "unresolved_conflict"
    assert result.answer.candidate is None
    assert len(result.fusion.reports) == 3


def test_only_question_relevant_available_experts_are_called():
    Case, _, Task, _, _, _, Workflow, *_ = components()
    tools = RecordingTool({"gait":[report()]})
    result = Workflow(tools).run(Case("P1",[record(),record("face")]), Task("步态", "pd_presence", ["gait"]))
    assert len(tools.calls) == 1
    assert result.activated_modalities == ["gait"]


def test_missing_required_modality_is_visible_and_prevents_complete_answer():
    Case, _, Task, _, _, _, Workflow, *_ = components()
    tools = RecordingTool({"gait":[report()]})
    result = Workflow(tools).run(Case("P1",[record()]), Task("综合判断", "pd_presence", ["gait","speech"]))
    assert result.missing_modalities == ["speech"]
    assert result.answer.status == "insufficient_evidence"


def test_subject_or_record_mixup_is_rejected():
    Case, _, Task, _, _, _, Workflow, *_ = components()
    wrong = report()
    wrong.subject_id = "P2"
    tools = RecordingTool({"gait":[wrong]})
    with pytest.raises(ValueError, match="subject|record"):
        Workflow(tools).run(Case("P1",[record()]), Task("判断", "pd_presence", ["gait"]))


def test_source_ids_cannot_reference_other_report_or_missing_evidence():
    *_, Critic, _ = components()
    wrong = report()
    wrong.prediction.evidence_ids = ["invented"]
    with pytest.raises(ValueError, match="evidence"):
        Critic().assess([wrong])


def test_serialized_api_context_excludes_paths_and_labels():
    Case, _, Task, _, _, _, Workflow, *_ = components()
    r = record(path="/secret/PD_patient_label_3.mp4")
    tools = RecordingTool({"gait":[report()]})
    result = Workflow(tools).run(Case("P1",[r]), Task("判断", "pd_presence", ["gait"]))
    payload = json.dumps(result.fusion.api_payload(),ensure_ascii=False)
    assert "/secret" not in payload
    assert "PD_patient_label_3" not in payload
    assert "gait-obs" in payload


def test_tool_failure_yields_explicit_insufficient_evidence():
    Case, _, Task, _, _, _, Workflow, *_ = components()
    class BrokenTool(RecordingTool):
        def analyze(self, rec, task, action=None):
            raise FileNotFoundError("private recording path")
    result = Workflow(BrokenTool({})).run(Case("P1",[record()]),Task("判断","pd_presence",["gait"]))
    assert result.answer.status == "insufficient_evidence"
    assert "private recording path" not in json.dumps(result.to_dict())


def test_same_target_different_class_definitions_are_not_silently_fused():
    _, _, _, _, _, Prediction, _, Critic, _ = components()
    a,b=report(),report("speech")
    b.prediction=Prediction("pd_presence",["control","patient"],[.1,.9],"other",True,["speech-obs"])
    review=Critic().assess([a,b])
    assert any(i.kind=="comparability" for i in review.issues)


def test_orchestrator_identifies_observation_question_without_involving_other_modalities():
    from pd_vqa.workflow import Orchestrator
    task=Orchestrator().identify("请分析步态视频中的摆臂测量",["gait","face","speech"])
    assert task.target == "gait_observation"
    assert task.modalities == ["gait"]


def test_invalid_tool_media_returns_insufficient_rather_than_crashing():
    Case, _, Task, _, _, _, Workflow, *_ = components()
    class InvalidTool(RecordingTool):
        def analyze(self,rec,task,action=None):
            raise ValueError("invalid video frame rate")
    result=Workflow(InvalidTool({})).run(Case("P1",[record()]),Task("观察","gait_observation",["gait"]))
    assert result.answer.status == "insufficient_evidence"
