"""Subject-disjoint per-expert training on precomputed sequences; validation selects checkpoints."""
from collections import Counter
import json
from pathlib import Path


def validate_split(rows):
    subject_splits = {}
    subject_labels = {}
    for row in rows:
        sid,split = row["subject_id"],row["split"]
        if split not in {"train","validation","test"}:
            raise ValueError("Explicit train/validation/test split required")
        if sid in subject_splits and subject_splits[sid] != split:
            raise ValueError("A subject occurs in multiple splits")
        subject_splits[sid] = split
        if "label" in row:
            if sid in subject_labels and subject_labels[sid] != row["label"]:
                raise ValueError("Inconsistent label for one subject")
            subject_labels[sid] = row["label"]
    if not rows:
        raise ValueError("Empty training manifest")


def train(manifest,out,*,target,modality,labels,epochs=30,device="cpu",seed=42,label_definition=None):
    import torch
    from .models import TemporalClassifier,validate_feature_contract,definition_hash
    if target == "severity" and not label_definition:
        raise ValueError("Severity training requires the versioned label definition")
    if type(epochs) is not int or epochs < 1 or len(labels)<2 or len(set(labels)) != len(labels):
        raise ValueError("Positive epochs and distinct class labels required")
    path = Path(manifest).resolve()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_split(rows)
    rows = [r for r in rows if r["modality"] == modality]
    if any(r["label"] not in labels for r in rows):
        raise ValueError("Label outside declared target class space")
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    if not train_rows or not val_rows:
        raise ValueError("Training and validation subjects required")
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    checkpoint = out/"best.pt"
    if checkpoint.exists():
        raise FileExistsError("Use a new experiment directory")

    input_contracts = []

    def load(row):
        p = Path(row["feature_cache"])
        data = torch.load(p if p.is_absolute() else path.parent/p,map_location="cpu",weights_only=True)
        if (data["subject_id"],data["record_id"],data["modality"]) != (
                row["subject_id"],row["record_id"],row["modality"]):
            raise ValueError("Feature cache provenance differs from manifest")
        x = data["features"].float()
        contract = validate_feature_contract(data.get("feature_contract"),x.shape[-1])
        if input_contracts and input_contracts[0] != contract:
            raise ValueError("Encoder feature contract differs across development records")
        input_contracts.append(contract)
        return x

    torch.manual_seed(seed)
    # No test feature or test label is loaded for model selection.
    train_data = [(r,load(r)) for r in train_rows]
    val_data = [(r,load(r)) for r in val_rows]
    dim = train_data[0][1].shape[-1]
    config = {"input_dim":dim,"hidden_dim":64,"num_classes":len(labels),"dropout":.1}
    model = TemporalClassifier(**config).to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    counts = Counter(r["subject_id"] for r in train_rows)
    best,stale,log = float("inf"),0,[]
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(train_data),generator=generator):
            r,x = train_data[int(idx)]
            x = x.unsqueeze(0).to(device)
            logits = model(x,torch.ones(x.shape[:2],dtype=torch.bool,device=device))
            loss = torch.nn.functional.cross_entropy(logits,torch.tensor([labels.index(r["label"])],device=device))
            optim.zero_grad()
            (loss/counts[r["subject_id"]]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step()
        model.eval()
        subject_probs,subject_labels = {},{}
        with torch.inference_mode():
            for r,x in val_data:
                p = model(x.unsqueeze(0).to(device),torch.ones(1,len(x),dtype=torch.bool,device=device)).softmax(-1)[0]
                subject_probs.setdefault(r["subject_id"],[]).append(p)
                label = labels.index(r["label"])
                if r["subject_id"] in subject_labels and subject_labels[r["subject_id"]] != label:
                    raise ValueError("Inconsistent subject validation labels")
                subject_labels[r["subject_id"]] = label
        nll = sum(-torch.stack(ps).mean(0)[subject_labels[s]].clamp_min(1e-8).log().item()
                  for s,ps in subject_probs.items())/len(subject_probs)
        log.append({"epoch":epoch+1,"validation_subject_nll":nll})
        if nll < best:
            best,stale = nll,0
            torch.save({"state_dict":model.cpu().state_dict(),"model_config":config,"trained":True,
                        "target":target,"modality":modality,"labels":labels,"seed":seed,
                        "feature_contract":input_contracts[0],
                        "label_definition_text":label_definition,
                        "label_definition_sha256":definition_hash(label_definition) if label_definition else None,
                        "train_subjects":sorted({r["subject_id"] for r in train_rows}),
                        "validation_subjects":sorted(subject_labels),"best_validation_nll":best},checkpoint)
            model.to(device)
        else:
            stale += 1
        if stale >= 10:
            break
    (out/"training.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
    return checkpoint
