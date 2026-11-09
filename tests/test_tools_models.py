"""Real signal, tensor and CLI integration tests (not diagnostic performance tests)."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import wave

import pytest


def wav(path):
    values = [32767]*8000 + [int(2000*((i%20)-10)/10) for i in range(8000)]
    with wave.open(str(path),"wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(8000)
        out.writeframes(struct.pack("<"+"h"*len(values),*values))


def test_real_audio_window_reanalysis_changes_measured_evidence(tmp_path):
    from pd_vqa.contracts import Case,Record,Task
    from pd_vqa.tools import MediaTools
    from pd_vqa.workflow import Workflow
    path = tmp_path/"test.wav"
    wav(path)
    rec = Record("p","a1","speech",path=str(path),windows=[{"start":1.,"end":2.}])
    result = Workflow(MediaTools(workspace=tmp_path/"work")).run(
        Case("p",[rec],"engineering"),Task("总结语音测量", "speech_observation",["speech"]))
    assert result.review_rounds == 1
    assert result.history[0][0].quality < .6
    assert result.history[1][0].quality >= .6
    assert result.answer.status == "answered"
    assert result.history[0][0].evidence[0].evidence_id != result.history[1][0].evidence[0].evidence_id
    assert all(e.interval == [1.,2.] for e in result.history[1][0].evidence)


def test_no_checkpoint_never_produces_disease_probabilities(tmp_path):
    from pd_vqa.contracts import Record,Task
    from pd_vqa.tools import MediaTools
    path=tmp_path/"test.wav"
    wav(path)
    report = MediaTools(workspace=tmp_path).analyze(Record("p","r","speech",path=str(path)),
                                                  Task("是否患病","pd_presence",["speech"]))
    assert report.prediction is None
    assert any("checkpoint" in s for s in report.limitations)


def test_review_options_never_invent_unavailable_segments(tmp_path):
    from pd_vqa.contracts import Record,Task
    from pd_vqa.tools import MediaTools
    tools=MediaTools(workspace=tmp_path)
    assert tools.review_options(Record("p","r","gait"),Task("观察","gait_observation",["gait"]),None)==[]


def test_temporal_classifier_ignores_padding_and_supports_backprop():
    import torch
    from pd_vqa.models import TemporalClassifier
    torch.manual_seed(5)
    model=TemporalClassifier(6,hidden_dim=8,num_classes=2,dropout=0.).eval()
    x=torch.randn(2,7,6)
    mask=torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,1]],dtype=torch.bool)
    a=model(x,mask)
    x[0,4:]=1000
    b=model(x,mask)
    assert torch.allclose(a,b,atol=1e-6)
    a.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_all_padding_is_rejected():
    import torch
    from pd_vqa.models import TemporalClassifier
    with pytest.raises(ValueError):
        TemporalClassifier(3,hidden_dim=8)(torch.zeros(1,2,3),torch.zeros(1,2,dtype=torch.bool))


def test_checkpoint_cannot_answer_a_different_task(tmp_path):
    import torch
    from pd_vqa.models import TemporalClassifier,load_predictor
    model=TemporalClassifier(3,hidden_dim=8)
    path=tmp_path/"m.pt"
    torch.save({"state_dict":model.state_dict(),"model_config":{"input_dim":3,"hidden_dim":8},
                "target":"pd_presence","labels":["healthy","pd"],"trained":True,
                "modality":"gait","train_subjects":["train1"],"validation_subjects":["val1"]},path)
    with pytest.raises(ValueError,match="target"):
        load_predictor(path,target="gait_observation",modality="gait")


def test_subject_leakage_between_training_and_validation_is_rejected():
    from pd_vqa.training import validate_split
    with pytest.raises(ValueError,match="subject"):
        validate_split([{"subject_id":"p1","split":"train"},{"subject_id":"p1","split":"validation"}])


def test_cli_demo_exercises_bounded_real_signal_review_and_writes_artifacts(tmp_path):
    run=subprocess.run([sys.executable,"-m","pd_vqa","demo","--out",str(tmp_path/"demo")],
                       capture_output=True,text=True,encoding="utf-8")
    assert run.returncode == 0, run.stderr
    result=json.loads((tmp_path/"demo"/"result.json").read_text(encoding="utf-8"))
    assert result["review_rounds"] == 1
    assert result["answer"]["candidate"] is None
    assert (tmp_path/"demo"/"trace.jsonl").is_file()
    assert (tmp_path/"demo"/"answer.md").is_file()


def test_cli_validate_resolves_paths_relative_to_case_not_working_directory(tmp_path):
    from pd_vqa.io import read_case
    p=tmp_path/"case.json"
    p.write_text(json.dumps({"subject_id":"p","records":[{"record_id":"r","modality":"speech",
                                                            "path":"a.wav"}]}),encoding="utf-8")
    assert Path(read_case(p).records[0].path) == tmp_path/"a.wav"


def test_training_checkpoint_is_selected_without_loading_test_cache(tmp_path):
    import torch
    from pd_vqa.training import train
    from pd_vqa.models import load_predictor
    torch.set_num_threads(1)
    rows=[]
    for i,split in enumerate(["train","train","validation","validation","test"]):
        cache=tmp_path/f"{i}.pt"
        row={"subject_id":f"s{i}","record_id":f"r{i}","modality":"gait", "split":split,
             "label":["healthy","pd"][i%2],"feature_cache":str(cache)}
        if split != "test":
            torch.save({"subject_id":f"s{i}","record_id":f"r{i}","modality":"gait",
                        "features":torch.randn(5,3),"timestamps":torch.arange(5),
                        "feature_contract":{"encoder":"synthetic_test","revision":"r1",
                                            "preprocessing":"synthetic_v1","feature_dim":3}},cache)
        rows.append(row)
    manifest=tmp_path/"train.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    ck=train(manifest,tmp_path/"experiment",target="pd_presence",modality="gait",
             labels=["healthy","pd"],epochs=2)
    model,labels,_=load_predictor(ck,target="pd_presence",modality="gait",feature_contract={
        "encoder":"synthetic_test","revision":"r1","preprocessing":"synthetic_v1","feature_dim":3})
    assert labels==["healthy","pd"]
    assert model(torch.zeros(1,5,3),torch.ones(1,5,dtype=torch.bool)).shape==(1,2)


def test_real_three_modality_decoding_and_targeted_review(tmp_path):
    cv2=pytest.importorskip("cv2")
    import numpy as np
    from pd_vqa.contracts import Case,Record,Task
    from pd_vqa.tools import MediaTools
    from pd_vqa.workflow import Workflow
    from pd_vqa.io import save_run
    records=[]
    for modality in ["gait","face"]:
        path=tmp_path/(modality+".avi")
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),10.,(64,64))
        assert writer.isOpened()
        for i in range(20):
            image=np.zeros((64,64,3),dtype=np.uint8)
            cv2.rectangle(image,(i,20),(i+12,40),(255,255,255),-1)
            writer.write(image)
        writer.release()
        records.append(Record("engineering","r_"+modality,modality,path=str(path)))
    audio=tmp_path/"speech.wav"
    wav(audio)
    records.append(Record("engineering","r_speech","speech",path=str(audio),windows=[{"start":1.,"end":2.}]))
    result=Workflow(MediaTools(workspace=tmp_path/"cache")).run(Case("engineering",records,"engineering"),
            Task("Summarize the available measurements","evidence_summary",["gait","face","speech"]))
    save_run(tmp_path/"result",result)
    assert result.activated_modalities==["gait","face","speech"]
    assert result.review_rounds==1
    assert result.answer.status=="answered"
    assert result.answer.candidate is None
    assert result.history[0][0].evidence[0].interval==[0.,2.]
