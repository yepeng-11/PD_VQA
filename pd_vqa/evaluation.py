"""Subject-level held-out evaluation. Repeated recordings are aggregated before scoring."""
import json
from pathlib import Path

from .contracts import Prediction


def classification_metrics(rows,labels):
    if not rows:
        raise ValueError("No predictions to score")
    subjects = {}
    for row in rows:
        p,y,s = row["probabilities"],row["label"],row["subject_id"]
        Prediction("evaluation",labels,p,"evaluation",True,["prediction"])
        if type(y) is not int or not 0 <= y < len(labels):
            raise ValueError("Invalid ground truth class index")
        g = subjects.setdefault(s,{"label":y,"rows":[]})
        if g["label"] != y:
            raise ValueError("Inconsistent ground truth for one subject")
        g["rows"].append(p)
    observations = []
    for g in subjects.values():
        p = [sum(r[i] for r in g["rows"])/len(g["rows"]) for i in range(len(labels))]
        observations.append((g["label"],p))
    matrix = [[0]*len(labels) for _ in labels]
    for y,p in observations:
        matrix[y][max(range(len(p)),key=p.__getitem__)] += 1
    f1,recall = [],[]
    for k in range(len(labels)):
        tp,actual,predicted = matrix[k][k],sum(matrix[k]),sum(r[k] for r in matrix)
        f1.append(2*tp/(actual+predicted) if actual+predicted else 0.)
        if actual:
            recall.append(tp/actual)
    aucs = []
    for k in ([1] if len(labels)==2 else range(len(labels))):
        positive = [p[k] for y,p in observations if y==k]
        negative = [p[k] for y,p in observations if y!=k]
        if not positive or not negative:
            aucs.append(None)
        else:
            aucs.append(sum(1. if a>b else .5 if a==b else 0. for a in positive for b in negative)/
                        (len(positive)*len(negative)))
    return {"subjects":len(subjects),"records":len(rows),"labels":labels,"confusion_matrix":matrix,
            "accuracy":sum(matrix[i][i] for i in range(len(labels)))/len(subjects),
            "macro_f1":sum(f1)/len(f1),"balanced_accuracy":sum(recall)/len(recall),
            "auroc":sum(aucs)/len(aucs) if all(a is not None for a in aucs) else None,
            "brier":sum(sum((p[i]-int(i==y))**2 for i in range(len(labels))) for y,p in observations)/len(subjects)}


def evaluate(manifest,checkpoint,*,split="test",device="cpu"):
    import torch
    from .models import load_predictor,validate_feature_contract
    from .training import validate_split
    path = Path(manifest).resolve()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_split(rows)
    state = torch.load(checkpoint,map_location="cpu",weights_only=True)
    model,labels,digest = load_predictor(checkpoint,target=state["target"],modality=state["modality"],device=device,
                        feature_contract=state.get("feature_contract"),label_definition=state.get("label_definition_text"))
    selected = [r for r in rows if r["split"] == split and r["modality"] == state["modality"]]
    if split == "test" and {r["subject_id"] for r in selected}&set(state["train_subjects"]+state["validation_subjects"]):
        raise ValueError("Evaluation subject overlaps model development subjects")
    predictions = []
    with torch.inference_mode():
        for r in selected:
            p = Path(r["feature_cache"])
            cache = torch.load(p if p.is_absolute() else path.parent/p,map_location="cpu",weights_only=True)
            if (cache["subject_id"],cache["record_id"],cache["modality"]) != (r["subject_id"],r["record_id"],r["modality"]):
                raise ValueError("Feature provenance mismatch")
            x = cache["features"].float().unsqueeze(0).to(device)
            if validate_feature_contract(cache.get("feature_contract"),x.shape[-1]) != state["feature_contract"]:
                raise ValueError("Evaluation encoder feature contract differs from training")
            probabilities = model(x,torch.ones(x.shape[:2],dtype=torch.bool,device=device)).softmax(-1)[0].tolist()
            predictions.append({"subject_id":r["subject_id"],"record_id":r["record_id"],
                                "label":labels.index(r["label"]),"probabilities":probabilities})
    return {"checkpoint_sha256":digest,"split":split,"target":state["target"],
            "metrics":classification_metrics(predictions,labels),"predictions":predictions}
