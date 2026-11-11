"""Read-only reviewer reproductions, fixed with behavior-level regression tests."""
import json

import pytest


def test_manifest_cannot_relabel_another_subject(tmp_path):
    from pd_vqa.io import read_case
    p=tmp_path/"case.json"
    p.write_text(json.dumps({"subject_id":"P1","records":[{"subject_id":"P2","record_id":"r","modality":"gait"}]}))
    with pytest.raises(ValueError,match="subject"):
        read_case(p)


def test_cached_features_with_unknown_quality_are_not_fully_trusted(tmp_path):
    import torch
    from pd_vqa.contracts import Record,Task
    from pd_vqa.tools import MediaTools
    p=tmp_path/"x.pt"
    torch.save({"subject_id":"p","record_id":"r","modality":"gait","features":torch.zeros(3,4),
                "timestamps":torch.arange(3)},p)
    report=MediaTools(workspace=tmp_path/"out").analyze(Record("p","r","gait",feature_cache=str(p)),
                                                     Task("观察","gait_observation",["gait"]))
    assert report.quality < .6
    assert all(e.quality < .6 for e in report.evidence)


def test_api_cannot_replace_fixed_candidate_with_conflicting_text():
    from test_workflow import report
    from pd_vqa.contracts import Task
    from pd_vqa.workflow import DecisionAgent,EvidenceCritic,WeightedEvidenceFusion
    class API:
        def complete_json(self,messages):
            return {"text":"Confirmed healthy; measured step length 42 cm.","evidence_ids":["gait-obs"]}
    reports=[report()]
    result=DecisionAgent(API()).answer(Task("诊断","pd_presence",["gait"]),WeightedEvidenceFusion().fuse(reports),
                                      EvidenceCritic().assess(reports),[],"parkinson",[])
    assert result.candidate == "pd"
    assert "Confirmed healthy" not in result.text
    assert "42 cm" not in result.text
    assert "pd" in result.text


def test_identification_preserves_explicit_missing_modality():
    from pd_vqa.workflow import Orchestrator
    task=Orchestrator().identify("Summarize gait and speech measurements",["gait"])
    assert set(task.modalities)=={"gait","speech"}


def test_diagnosis_identification_keeps_named_missing_modalities():
    from pd_vqa.workflow import Orchestrator
    task=Orchestrator().identify("Use gait and speech to assess Parkinson disease",["gait"])
    assert task.target == "pd_presence"
    assert set(task.modalities)=={"gait","speech"}


def test_online_router_cannot_discard_explicitly_requested_modalities():
    from pd_vqa.workflow import Orchestrator
    class API:
        def complete_json(self,messages):
            return {"target":"pd_presence","modalities":["gait"]}
    task=Orchestrator().identify("Use gait and speech to assess Parkinson disease",["gait"],API())
    assert set(task.modalities)=={"gait","speech"}


def test_checkpoint_rejects_same_width_different_encoder_contract(tmp_path):
    import torch
    from pd_vqa.models import TemporalClassifier,load_predictor
    model=TemporalClassifier(4,hidden_dim=8)
    a={"encoder":"encoder_a","revision":"r1","preprocessing":"v1","feature_dim":4}
    b={**a,"encoder":"encoder_b"}
    p=tmp_path/"m.pt"
    torch.save({"state_dict":model.state_dict(),"model_config":{"input_dim":4,"hidden_dim":8},
                "target":"pd_presence","labels":["healthy","pd"],"modality":"gait","trained":True,
                "train_subjects":["train"],"validation_subjects":["val"],"feature_contract":a},p)
    with pytest.raises(ValueError,match="contract|encoder"):
        load_predictor(p,target="pd_presence",modality="gait",feature_contract=b)


def test_severity_definition_is_bound_to_checkpoint(tmp_path):
    import torch
    import hashlib
    from pd_vqa.models import TemporalClassifier,load_predictor
    model=TemporalClassifier(4,hidden_dim=8,num_classes=4)
    contract={"encoder":"a","revision":"r1","preprocessing":"v1","feature_dim":4}
    p=tmp_path/"m.pt"
    torch.save({"state_dict":model.state_dict(),"model_config":{"input_dim":4,"hidden_dim":8,"num_classes":4},
                "target":"severity","labels":["0","1","2","3"],"modality":"gait","trained":True,
                "train_subjects":["train"],"validation_subjects":["val"],"feature_contract":contract,
                "label_definition_sha256":hashlib.sha256(b"definition A").hexdigest()},p)
    with pytest.raises(ValueError,match="definition"):
        load_predictor(p,target="severity",modality="gait",feature_contract=contract,label_definition="definition B")


def test_diagnostic_request_does_not_hide_a_missing_specialist_prediction():
    from test_workflow import report
    from pd_vqa.contracts import Task
    from pd_vqa.workflow import DecisionAgent,EvidenceCritic,WeightedEvidenceFusion
    reports=[report(),report("speech")]
    reports[1].prediction=None
    answer=DecisionAgent().answer(Task("结合步态和语音诊断","pd_presence",["gait","speech"]),
        WeightedEvidenceFusion().fuse(reports),EvidenceCritic().assess(reports),[],"parkinson",[])
    assert answer.status == "insufficient_evidence"


@pytest.mark.parametrize("scenario", ["missing_classifier", "missing_modality", "empty_case"])
def test_decision_abstention_does_not_record_evidence_sufficient_stop(scenario):
    from test_workflow import RecordingTool, record, report
    from pd_vqa.contracts import Case, Task
    from pd_vqa.workflow import Workflow
    observation = report("gait", target="gait_observation")
    observation.prediction = None
    tools = RecordingTool({"gait": [observation]})
    records = [record()]
    target, modalities = "gait_observation", ["gait"]
    if scenario == "missing_classifier":
        target = "pd_presence"
        observation.target = target
    elif scenario == "missing_modality":
        modalities.append("speech")
    else:
        records = []
    result = Workflow(tools, max_reviews=0).run(Case("P1", records), Task("检查证据", target, modalities))
    assert result.answer.status == "insufficient_evidence"
    assert result.stop_reason == "decision_insufficient_evidence"
    assert result.review_rounds == 0
    assert result.events[-1]["stop_reason"] == result.stop_reason


def test_cache_cannot_inherit_quality_from_a_different_raw_file(tmp_path):
    import torch
    from test_tools_models import wav
    from pd_vqa.contracts import Record,Task
    from pd_vqa.tools import MediaTools
    raw=tmp_path/"actual.wav"
    wav(raw)
    cache=tmp_path/"old.pt"
    torch.save({"subject_id":"p","record_id":"r","modality":"speech", "features":torch.zeros(3,4),
                "timestamps":torch.arange(3)/2,"raw_source_sha256":"different-recording",
                "source_interval":[0.,2.],"timestamp_origin_seconds":0.},cache)
    with pytest.raises(ValueError,match="source|recording"):
        MediaTools(workspace=tmp_path/"out").analyze(Record("p","r","speech",path=str(raw),
                  feature_cache=str(cache)),Task("测量","speech_observation",["speech"]))
