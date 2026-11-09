"""Per-modality temporal enhancement and classifier. No cross-expert feature mixing."""
import hashlib
import math
import json

import torch
from torch import nn


class TemporalClassifier(nn.Module):
    def __init__(self,input_dim,hidden_dim=64,num_classes=2,dropout=.1):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even")
        self.projection = nn.Linear(input_dim,hidden_dim)
        self.temporal = nn.Conv1d(hidden_dim,hidden_dim,3,padding=1)
        self.attention = nn.MultiheadAttention(hidden_dim,2,dropout=dropout,batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim,num_classes)

    def forward(self,x,mask):
        if x.ndim != 3 or mask.shape != x.shape[:2] or not mask.any(1).all():
            raise ValueError("Need nonempty sequences and valid masks")
        if not torch.isfinite(x).all():
            raise ValueError("Features must be finite")
        valid = mask.unsqueeze(-1)
        h = self.projection(x.masked_fill(~valid,0)).masked_fill(~valid,0)
        h = h+self.temporal(h.transpose(1,2)).transpose(1,2)
        position = torch.arange(h.shape[1],device=x.device,dtype=h.dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0,h.shape[-1],2,device=x.device,dtype=h.dtype)*(-math.log(10000)/h.shape[-1]))
        encoding = torch.zeros(h.shape[1],h.shape[-1],device=x.device,dtype=h.dtype)
        encoding[:,0::2],encoding[:,1::2] = torch.sin(position*div),torch.cos(position*div)
        h = (h+encoding).masked_fill(~valid,0)
        attended,_ = self.attention(h,h,h,key_padding_mask=~mask,need_weights=False)
        h = self.norm(h+self.dropout(attended)).masked_fill(~valid,0)
        return self.classifier(h.sum(1)/mask.sum(1,keepdim=True))


def definition_hash(text):
    return hashlib.sha256(text.replace("\r\n","\n").strip().encode("utf-8")).hexdigest()


def validate_feature_contract(contract,feature_dim=None):
    if not isinstance(contract,dict) or set(contract) != {"encoder","revision","preprocessing","feature_dim"}:
        raise ValueError("Missing or invalid encoder feature contract")
    if any(not isinstance(contract[k],str) or not contract[k].strip()
           or contract[k] == "unrecorded" for k in ("encoder","revision","preprocessing")):
        raise ValueError("Encoder identity, revision and preprocessing must be recorded")
    if type(contract["feature_dim"]) is not int or contract["feature_dim"] <= 0:
        raise ValueError("Feature contract dimension invalid")
    if feature_dim is not None and contract["feature_dim"] != feature_dim:
        raise ValueError("Feature contract dimension mismatch")
    return contract


def contract_from_source(source,feature_dim):
    return validate_feature_contract({"encoder":source.get("repo"),"revision":source.get("revision"),
                                      "preprocessing":source.get("preprocessing"),"feature_dim":feature_dim})


def load_predictor(path,*,target,modality,device="cpu",feature_contract=None,label_definition=None):
    state = torch.load(path,map_location="cpu",weights_only=True)
    if state.get("target") != target or state.get("modality") != modality:
        raise ValueError("Checkpoint target/modality differs from question")
    stored = validate_feature_contract(state.get("feature_contract"))
    actual = validate_feature_contract(feature_contract)
    if json.dumps(stored,sort_keys=True) != json.dumps(actual,sort_keys=True):
        raise ValueError("Checkpoint encoder feature contract differs from current inputs")
    if target == "severity" and (not label_definition or
            state.get("label_definition_sha256") != definition_hash(label_definition)):
        raise ValueError("Severity definition differs from training definition")
    train,val = state.get("train_subjects",[]),state.get("validation_subjects",[])
    if not state.get("trained") or not train or not val or set(train)&set(val):
        raise ValueError("Checkpoint requires disjoint training/validation provenance")
    model = TemporalClassifier(**state["model_config"])
    model.load_state_dict(state["state_dict"])
    model.eval().to(device)
    with open(path,"rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return model,state["labels"],digest
