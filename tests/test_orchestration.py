"""Task recognition, expert contracts and dispatch boundaries using the real controller."""
from dataclasses import asdict
import json
import subprocess
import sys

import pytest

from pd_vqa.contracts import Case,Record,Task
from pd_vqa.workflow import Orchestrator,Workflow


@pytest.mark.parametrize("question,target,modalities",[
    ("请描述这位帕金森患者的步态表现","gait_observation",["gait"]),
    ("不要分类，只计算这段语音的声学测量","speech_observation",["speech"]),
    ("Describe the facial movements of this Parkinson patient","face_observation",["face"]),
    ("结合步态和语音判断是否患有帕金森病","pd_presence",["gait","speech"]),
    ("评估严重程度分组并解释依据","severity",["gait","face","speech"]),
    ("只描述步态测量，不做诊断","gait_observation",["gait"]),
    ("不进行诊断，只分析语音的声学特征","speech_observation",["speech"]),
    ("Please describe speech without making a diagnosis","speech_observation",["speech"]),
    ("No diagnosis, just detect pauses in speech","speech_observation",["speech"]),
    ("Detect pauses in this speech recording and measure their duration","speech_observation",["speech"]),
    ("Predict the next facial movement","face_observation",["face"]),
    ("Does this patient have Parkinson disease based on the gait?","pd_presence",["gait"]),
    ("Is this gait consistent with Parkinson disease?","pd_presence",["gait"]),
    ("Estimate Parkinson severity from speech","severity",["speech"]),
    ("Can you describe the gait of this Parkinson patient?","gait_observation",["gait"]),
    ("Could you measure speech pauses in this Parkinson patient?","speech_observation",["speech"]),
    ("Does this Parkinson patient have reduced arm swing?","gait_observation",["gait"]),
    ("请描述语音，引用测量。不要推断疾病、严重程度或输出分类概率。","speech_observation",["speech"]),
    ("不要只描述语音，请进行帕金森分类。","pd_presence",["speech"]),
    ("不要只描述语音而要进行帕金森分类。","pd_presence",["speech"]),
    ("不需要报告全部声学参数但请进行帕金森分类。","pd_presence",["speech"]),
])
def test_recognition_separates_disease_context_from_requested_classification(question,target,modalities):
    task=Orchestrator().identify(question,["gait","face","speech"])
    assert task.target==target
    assert task.modalities==modalities


def test_task_recognition_records_multiple_goals():
    task=Orchestrator().identify("结合步态和语音进行疾病分类，报告测量并解释依据",["gait","speech"])
    assert {"classification","measurement","explanation"} <= set(task.intents)


def test_api_cannot_turn_a_description_only_query_into_classification():
    class API:
        def complete_json(self,messages):
            return {"target":"pd_presence","modalities":["gait"],"intents":["classification"]}
    task=Orchestrator().identify("不要诊断，只描述步态视频中的动作",["gait"],API())
    assert task.target=="gait_observation"
    assert "classification" not in task.intents
    assert task.routing_notes


def case():
    return Case("P1",[Record("P1","G1","gait",path="/private/gait.mp4"),
                      Record("P1","S1","speech",path="/private/speech.wav")],"parkinson")


def capabilities():
    return {
        "gait":{"operations":["inspect_media","measure","encode","classify"],
                "classification_targets":["pd_presence"]},
        "speech":{"operations":["inspect_media","measure"],"classification_targets":[]}
    }


def test_plan_assigns_bound_subtasks_and_classification_only_to_capable_experts():
    plan=Orchestrator().build_plan(case(),Task("结合步态和语音分类并解释","pd_presence",["gait","speech"]),
                                   capabilities=capabilities())
    gait,speech=plan.assignments
    assert gait.task.record_id=="G1"
    assert gait.task.subject_id=="P1"
    assert gait.task.modalities==["gait"]
    assert gait.task.classification_requested is True
    assert gait.task.classification_enabled is True
    assert speech.task.classification_requested is True
    assert speech.task.classification_enabled is False
    assert "classification_model_unavailable" in speech.blocked_requirements
    assert gait.task.instructions
    assert "evidence_ids" in gait.task.output_requirements
    assert "/private/" not in json.dumps(asdict(plan))


def test_observation_plan_excludes_classifier_even_when_registered():
    plan=Orchestrator().build_plan(case(),Task("描述步态","gait_observation",["gait"]),
                                   capabilities=capabilities())
    assignment=plan.assignments[0]
    assert assignment.task.classification_enabled is False
    assert "classify" not in assignment.task.allowed_operations


def test_missing_requested_agent_is_recorded_in_plan():
    plan=Orchestrator().build_plan(case(),Task("结合步态和面部分析","evidence_summary",["gait","face"]),
                                   capabilities=capabilities())
    assert plan.missing_modalities==["face"]
    assert plan.assignments[1].status=="missing_data"
    assert plan.assignments[1].task is None


def test_severity_model_is_not_enabled_without_group_definition():
    caps=capabilities()
    caps["gait"]["classification_targets"]=["severity"]
    plan=Orchestrator().build_plan(case(),Task("评估分组","severity",["gait"]),capabilities=caps)
    assert plan.assignments[0].task.classification_enabled is False
    assert "severity_definition_missing" in plan.assignments[0].blocked_requirements


def test_workflow_dispatches_and_persists_the_expert_task_not_the_unsplit_question():
    from test_workflow import RecordingTool,record,report
    class Specialist(RecordingTool):
        def analyze(self,rec,task,action=None):
            self.subtask=task
            return super().analyze(rec,task,action)
    tools=Specialist({"gait":[report()]})
    result=Workflow(tools).run(Case("P1",[record()]),Task("判断疾病类别","pd_presence",["gait"]))
    assert tools.subtask.modality=="gait"
    assert tools.subtask.record_id=="gait1"
    assert result.execution_plan.assignments[0].task.task_id==tools.subtask.task_id
    assert any(e.get("phase")=="dispatch" and e.get("record_id")=="gait1" for e in result.events)
    assert any(e.get("phase")=="completed" for e in result.events)


def test_actual_media_observation_with_checkpoint_never_calls_classifier(tmp_path,monkeypatch):
    torch=pytest.importorskip("torch")
    from test_tools_models import wav
    from pd_vqa.tools import MediaTools,source_digest
    import pd_vqa.models
    def unexpected(*args,**kw):
        raise RuntimeError("classifier should not be invoked")
    monkeypatch.setattr(pd_vqa.models,"load_predictor",unexpected)
    raw=tmp_path/"s.wav"
    wav(raw)
    cache=tmp_path/"features.pt"
    torch.save({"subject_id":"p","record_id":"s","modality":"speech","features":torch.zeros(4,3),
                "timestamps":torch.tensor([0.,.5,1.,1.5]),"raw_source_sha256":source_digest(raw),
                "source_interval":[0.,2.],"timestamp_origin_seconds":0.},cache)
    rec=Record("p","s","speech",path=str(raw),feature_cache=str(cache),checkpoint=str(tmp_path/"unused.pt"))
    report=MediaTools(workspace=tmp_path/"work").analyze(rec,Task("描述语音","speech_observation",["speech"]))
    assert report.prediction is None
    assert report.evidence


def test_plan_cli_does_not_execute_media_analysis(tmp_path):
    from test_tools_models import wav
    raw=tmp_path/"audio.wav"
    wav(raw)
    manifest=tmp_path/"case.json"
    manifest.write_text(json.dumps({"subject_id":"p","records":[{"record_id":"r","modality":"speech",
                                         "path":"audio.wav"}]}),encoding="utf-8")
    out=tmp_path/"plan.json"
    proc=subprocess.run([sys.executable,"-m","pd_vqa","plan","--case",str(manifest),
                         "--question","Describe the speech measurements","--out",str(out)],
                         capture_output=True,text=True,encoding="utf-8")
    assert proc.returncode==0,proc.stderr
    assert proc.stdout and json.loads(proc.stdout)["subject_id"]=="p"
    plan=json.loads(out.read_text(encoding="utf-8"))
    assert plan["assignments"][0]["task"]["classification_enabled"] is False
    assert not (tmp_path/"cache").exists()


def test_reflection_dispatch_returns_through_orchestrator_and_keeps_task_binding():
    from test_workflow import RecordingTool,record,report,evidence
    class Specialist(RecordingTool):
        def analyze(self,rec,task,action=None):
            if action:
                self.review_task=task
            return super().analyze(rec,task,action)
    tools=Specialist({"gait":[report(quality=.1),report(evidence_items=[evidence(evidence_id_suffix="new")])]},
                         {"gait":[{"operation":"window","start":1.,"end":3.}]})
    result=Workflow(tools).run(Case("P1",[record()]),Task("判断","pd_presence",["gait"]))
    assert tools.review_task.record_id=="gait1"
    assert any(e.get("agent")=="orchestrator" and e.get("phase")=="review_dispatch" for e in result.events)
    assert len(result.history)==2


def test_corrupt_checkpoint_preserves_measurement_capability(tmp_path):
    from test_tools_models import wav
    from pd_vqa.tools import MediaTools
    import torch
    raw,cache,head=tmp_path/"s.wav",tmp_path/"features.pt",tmp_path/"bad.pt"
    wav(raw)
    torch.save({},cache)
    head.write_bytes(b"invalid checkpoint")
    tools=MediaTools(workspace=tmp_path/"work")
    rec=Record("p","r","speech",path=str(raw),feature_cache=str(cache),checkpoint=str(head))
    caps=tools.capabilities(rec,Task("诊断","pd_presence",["speech"]))
    assert "classify" not in caps["operations"]
    assert "inspect_media" in caps["operations"]
    assert caps["limitations"]
    assert not (tmp_path/"work").exists()


def test_real_classifier_runs_only_when_enabled_and_plan_is_saved(tmp_path):
    import torch
    from pd_vqa.models import TemporalClassifier
    from pd_vqa.tools import MediaTools
    from pd_vqa.io import save_run
    contract={"encoder":"test_encoder","revision":"fixture","preprocessing":"fixture","feature_dim":4}
    cache,head=tmp_path/"features.pt",tmp_path/"head.pt"
    torch.save({"subject_id":"p","record_id":"r","modality":"speech","features":torch.zeros(3,4),
                "timestamps":torch.arange(3)/2,"feature_contract":contract,
                "quality_metadata":{"score":1.,"method":"fixture_quality"}},cache)
    config={"input_dim":4,"hidden_dim":8}
    torch.save({"model_config":config,"state_dict":TemporalClassifier(**config).state_dict(),
                "target":"pd_presence","modality":"speech","trained":True,"labels":["healthy","pd"],
                "train_subjects":["fixture_train"],"validation_subjects":["fixture_val"],
                "feature_contract":contract},head)
    records=Case("p",[Record("p","r","speech",feature_cache=str(cache),checkpoint=str(head))],"engineering")
    workflow=Workflow(MediaTools(workspace=tmp_path/"work"),max_reviews=0)
    classified=workflow.run(records,Task("分类语音","pd_presence",["speech"]))
    observed=workflow.run(records,Task("描述语音","speech_observation",["speech"]))
    assert classified.execution_plan.assignments[0].task.classification_enabled is True
    assert classified.fusion.reports[0].prediction is not None
    assert observed.execution_plan.assignments[0].task.classification_enabled is False
    assert observed.fusion.reports[0].prediction is None
    save_run(tmp_path/"result",observed)
    persisted=json.loads((tmp_path/"result"/"plan.json").read_text(encoding="utf-8"))
    assert persisted==asdict(observed.execution_plan)


def test_reasoner_receives_the_same_expert_task_when_querying_review_tools():
    from test_workflow import RecordingTool,record,report,evidence
    class BoundTool(RecordingTool):
        def review_options(self,rec,task,current):
            assert task.record_id==rec.record_id
            assert task.modalities==[rec.modality]
            return super().review_options(rec,task,current)
    tools=BoundTool({"gait":[report(quality=.1),report(evidence_items=[evidence(evidence_id_suffix="new")])]},
                    {"gait":[{"operation":"window","start":1.,"end":3.}]})
    result=Workflow(tools).run(Case("P1",[record()]),Task("分类","pd_presence",["gait"]))
    assert result.review_rounds==1


def test_no_backend_capability_is_not_reported_as_an_activated_expert():
    from test_workflow import RecordingTool,record,report
    class Unavailable(RecordingTool):
        def capabilities(self,rec,task):
            return {"operations":[],"classification_targets":[]}
    tools=Unavailable({"gait":[report()]})
    result=Workflow(tools).run(Case("P1",[record()]),Task("分类","pd_presence",["gait"]))
    assert result.activated_modalities==[]
    assert tools.calls==[]
    assert result.answer.status=="insufficient_evidence"


@pytest.mark.parametrize("malicious",["unrelated","undeclared","duplicate","repeat","binding"])
def test_review_dispatch_validates_entire_batch_before_any_tool_execution(malicious):
    from test_workflow import RecordingTool,record,report
    from pd_vqa.workflow import ExpertAgent,EvidenceCritic
    action={"operation":"window","start":1.,"end":3.}
    records=Case("P1",[record(),record("speech")])
    reports=[report(quality=.1),report("speech")]
    tools=RecordingTool({"gait":[reports[0]],"speech":[reports[1]]},{"gait":[action],"speech":[action]})
    orch=Orchestrator()
    plan=orch.build_plan(records,Task("判断","pd_presence",["gait","speech"]),
                         capabilities={r.modality:tools.capabilities(r,None) for r in records.records})
    actions=[{"modality":"gait","action":action}]
    attempted=set()
    if malicious=="unrelated":
        actions.append({"modality":"speech","action":action})
    elif malicious=="undeclared":
        actions[0]["action"]={**action,"end":4.}
    elif malicious=="duplicate":
        actions+=actions[:]
    elif malicious=="repeat":
        attempted.add(("gait",json.dumps(action,sort_keys=True)))
    else:
        records.records[0].record_id="different"
    with pytest.raises(ValueError):
        orch.coordinate_review(records,plan,ExpertAgent(tools),actions,reports,
                               EvidenceCritic().assess(reports),attempted,[])
    assert tools.calls==[]


def test_observation_capability_probe_never_loads_checkpoint(tmp_path,monkeypatch):
    from pd_vqa.tools import MediaTools
    import torch
    def unexpected(*args,**kwargs):
        raise AssertionError("Observation planner must not load weights")
    monkeypatch.setattr(torch,"load",unexpected)
    cache=tmp_path/"features.pt"
    cache.touch()
    rec=Record("p","r","face",feature_cache=str(cache),checkpoint="unused.pt")
    caps=MediaTools(workspace=tmp_path/"work").capabilities(rec,Task("描述面部","face_observation",["face"]))
    assert caps["operations"]==["read_features"]
    assert not (tmp_path/"work").exists()
