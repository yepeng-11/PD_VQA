"""Research CLI. Offline by default, online API calls require an explicit flag."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import struct
import sys
import wave

from .contracts import MODALITIES,TARGETS,Task,Case,Record
from .io import read_case,save_run,save_json


def demo(out):
    from .tools import MediaTools
    from .workflow import Workflow
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"result.json").exists():
        raise FileExistsError("Use a new demo output directory")
    path = out/"engineering_signal.wav"
    # Purposefully clipped first half; the second half is a quiet synthetic tone.
    y = [32767]*8000+[int(2000*math.sin(2*math.pi*160*i/8000)) for i in range(8000)]
    with wave.open(str(path),"wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(struct.pack("<"+"h"*len(y),*y))
    case = Case("engineering_subject",[Record("engineering_subject","audio1","speech",path=str(path),
                    windows=[{"start":1.,"end":2.}])],"engineering")
    result = Workflow(MediaTools(workspace=out/"cache")).run(case,
                 Task("这段音频质量如何？请报告可核查的测量。","speech_observation",["speech"]))
    save_run(out,result)
    save_json(out/"demo_metadata.json",{"synthetic":True,"clinical_performance":False,"api_called":False,
              "purpose":"A real signal is measured twice to exercise bounded review; not a diagnosis benchmark"})
    return result


def main(argv=None):
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="PD_VQA: question-driven multimodal evidence agents")
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("demo",help="Offline real-signal engineering review loop")
    p.add_argument("--out",required=True)
    p = sub.add_parser("validate",help="Check case/subject bindings and available files")
    p.add_argument("--case",required=True)
    for command in ("run","plan"):
        p = sub.add_parser(command,help="Execute analysis" if command == "run" else "Preview routing without analysis")
        p.add_argument("--case",required=True)
        p.add_argument("--question",required=True)
        p.add_argument("--target",choices=TARGETS,help="Explicit task override")
        p.add_argument("--modalities",nargs="+",choices=MODALITIES)
        p.add_argument("--label-definition",help="Path to clinician's research group definition")
        p.add_argument("--out",required=True)
        p.add_argument("--online",action="store_true",help="Enable configured DeepSeek JSON calls")
        p.add_argument("--expert-backend",choices=["local","qwen"],default="local",
                       help="qwen uploads selected source media for multimodal expert interpretation")
        if command == "run":
            p.add_argument("--max-reviews",type=int,default=2)
            p.add_argument("--device",default="cpu")
            p.add_argument("--knowledge",help="Source-linked medical knowledge cards JSON")
            p.add_argument("--memory",help="Existing frozen skill database")
    p = sub.add_parser("download-models",help="Explicitly download selected frozen encoders")
    p.add_argument("--root",required=True)
    p.add_argument("--modalities",nargs="+",choices=MODALITIES,default=list(MODALITIES))
    p = sub.add_parser("extract",help="Cache real sequences from one explicit case")
    p.add_argument("--case",required=True)
    p.add_argument("--model-root",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--device",default="cpu")
    p = sub.add_parser("train",help="Train one specialist with subject-disjoint validation")
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--target",choices=TARGETS,required=True)
    p.add_argument("--modality",choices=MODALITIES,required=True)
    p.add_argument("--labels",nargs="+",required=True)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--device",default="cpu")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--label-definition",help="Required versioned group definition for severity training")
    p = sub.add_parser("evaluate",help="Aggregate held-out recordings by subject before scoring")
    p.add_argument("--manifest",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--split",choices=["validation","test"],default="test")
    p.add_argument("--device",default="cpu")
    p.add_argument("--out",required=True)
    p = sub.add_parser("model-smoke",help="Synthetic temporal model forward/backward, no clinical scores")
    p.add_argument("--device",default="cpu")
    p.add_argument("--out",required=True)
    p = sub.add_parser("reflect",help="Propose unverified skills from a completed development run")
    p.add_argument("--run",required=True)
    p.add_argument("--memory",required=True)
    p.add_argument("--split",choices=["train","development"],required=True)
    p = sub.add_parser("feedback",help="Attach externally obtained feedback to a candidate skill")
    p.add_argument("--memory",required=True)
    p.add_argument("--skill-id",required=True)
    p.add_argument("--source",choices=["clinician_review","benchmark_label","tool_validation"],required=True)
    p.add_argument("--reference-id",required=True)
    p.add_argument("--split",choices=["train","development"],required=True)
    p.add_argument("--success",choices=["true","false"],required=True)
    args = parser.parse_args(argv)
    if args.command == "demo":
        result = demo(args.out)
        print(json.dumps({"synthetic":True,"review_rounds":result.review_rounds,
                          "status":result.answer.status},ensure_ascii=False))
    elif args.command == "validate":
        case = read_case(args.case)
        items = [{"modality":r.modality,"record_id":r.record_id,
                  "raw_available":bool(r.path and Path(r.path).is_file()),
                  "cache_available":bool(r.feature_cache and Path(r.feature_cache).is_file()),
                  "checkpoint_available":bool(r.checkpoint and Path(r.checkpoint).is_file())} for r in case.records]
        print(json.dumps({"subject_id":case.subject_id,"records":items},ensure_ascii=False,indent=2))
    elif args.command in {"run","plan"}:
        from .tools import MediaTools
        from .workflow import Workflow,Orchestrator
        from .memory import SkillMemory
        from .clinical import ClinicalAgent
        from .api import ChatAPI
        case = read_case(args.case)
        api = ChatAPI.from_env() if args.online else None
        inferred = Orchestrator().identify(args.question,[r.modality for r in case.records],api)
        target = args.target or inferred.target
        intents = [i for i in inferred.intents if i != "classification"]
        if target in {"pd_presence","severity"}:
            intents.append("classification")
        task = Task(args.question,target,args.modalities or inferred.modalities,
                    Path(args.label_definition).read_text(encoding="utf-8") if args.label_definition else None,
                    intents,"explicit_override" if args.target or args.modalities else inferred.identification_source,
                    inferred.routing_notes)
        tools = MediaTools(workspace=Path(args.out)/"cache",device=getattr(args,"device","cpu"))
        if args.expert_backend == "qwen":
            from .qwen import QwenAPI
            from .experts import MultimodalExpertTools
            tools = MultimodalExpertTools(tools,QwenAPI.from_env())
        if args.command == "plan":
            execution_plan = Orchestrator().build_plan(case,task,capabilities={
                r.modality:tools.capabilities(r,task) for r in case.records if r.modality in task.modalities})
            save_json(args.out,asdict(execution_plan))
            print(json.dumps(asdict(execution_plan),ensure_ascii=False))
            return
        memory = SkillMemory(args.memory,frozen=True) if args.memory else None
        clinical = ClinicalAgent(args.knowledge) if args.knowledge else None
        result = Workflow(tools,api=api,memory=memory,
                          clinical=clinical,max_reviews=args.max_reviews).run(case,task)
        save_run(args.out,result)
        print(json.dumps(asdict(result.answer),ensure_ascii=False))
    elif args.command == "download-models":
        from .backends.encoders import download_models
        print(json.dumps(download_models(args.root,args.modalities)))
    elif args.command == "extract":
        import torch
        from dataclasses import replace
        from .backends.encoders import FrozenEncoders
        from .models import contract_from_source
        from .tools import MediaTools,source_digest
        case = read_case(args.case)
        out = Path(args.out)
        out.mkdir(parents=True,exist_ok=True)
        cache_manifest = []
        for i,record in enumerate(case.records):
            target = out/f"record_{i:03d}.pt"
            if target.exists():
                raise FileExistsError("Use a new feature output directory")
            encoder = FrozenEncoders(args.model_root,args.device,record.face_is_cropped)
            x,times,source = encoder.encode(record.modality,record.path)
            probe = MediaTools().analyze(replace(record,checkpoint=None,model_root=None,
                    feature_cache=None,landmark_model=None,capture_task="unspecified"),
                    Task("Record extraction quality","evidence_summary",[record.modality]))
            interval = probe.evidence[0].interval if probe.evidence else None
            torch.save({"subject_id":record.subject_id,"record_id":record.record_id,"modality":record.modality,
                        "features":x,"timestamps":times,"source":source,
                        "feature_contract":contract_from_source(source,x.shape[-1]),
                        "quality_metadata":{"score":probe.quality,"method":"record_tool_quality_v1"},
                        "raw_source_sha256":source_digest(record.path),"source_interval":interval,
                        "timestamp_origin_seconds":0.},target)
            cache_manifest.append({**asdict(record),"feature_cache":str(target.resolve())})
        save_json(out/"case.json",{"subject_id":case.subject_id,"dataset_domain":case.dataset_domain,
                                   "records":cache_manifest})
    elif args.command == "train":
        from .training import train
        print(train(args.manifest,args.out,target=args.target,modality=args.modality,labels=args.labels,
                    epochs=args.epochs,device=args.device,seed=args.seed,
                    label_definition=Path(args.label_definition).read_text(encoding="utf-8") if args.label_definition else None))
    elif args.command == "model-smoke":
        import torch
        from .models import TemporalClassifier
        torch.manual_seed(42)
        model = TemporalClassifier(8,hidden_dim=16).to(args.device)
        x = torch.randn(2,10,8,device=args.device)
        logits = model(x,torch.ones(2,10,dtype=torch.bool,device=args.device))
        loss = torch.nn.functional.cross_entropy(logits,torch.tensor([0,1],device=args.device))
        loss.backward()
        if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError("Nonfinite model smoke result")
        save_json(args.out,{"synthetic":True,"device":args.device,"loss":loss.item(),
                            "clinical_performance":False,"finite_gradients":True})
        print("Temporal classifier forward/backward passed")
    elif args.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(args.manifest,args.checkpoint,split=args.split,device=args.device)
        save_json(args.out,result)
        print(json.dumps(result["metrics"],ensure_ascii=False))
    elif args.command == "reflect":
        from types import SimpleNamespace
        from .memory import ReflectionAgent,SkillMemory
        raw = json.loads(Path(args.run).read_text(encoding="utf-8"))
        skills = ReflectionAgent().propose_from_run(SimpleNamespace(events=raw["events"]),SkillMemory(args.memory),
                                                    str(Path(args.run).resolve()),split=args.split)
        print(json.dumps({"candidate_skill_ids":skills,"validated":False}))
    elif args.command == "feedback":
        from .memory import SkillMemory
        SkillMemory(args.memory).feedback(args.skill_id,success=args.success=="true",source=args.source,
                  reference_id=args.reference_id,split=args.split)
        print("External feedback recorded")


if __name__ == "__main__":
    main()
