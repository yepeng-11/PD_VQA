"""JSON manifests and auditable run artifacts."""
from dataclasses import fields
import json
from pathlib import Path

from .contracts import Case,Record


def read_case(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {f.name for f in fields(Record)}
    records = []
    for raw in data["records"]:
        if set(raw)-allowed:
            raise ValueError("Unknown recording field; training labels do not belong in QA manifests")
        if "subject_id" in raw and raw["subject_id"] != data["subject_id"]:
            raise ValueError("Record subject_id differs from case subject_id")
        row = {**raw,"subject_id":data["subject_id"]}
        for key in ("path","checkpoint","model_root","landmark_model","feature_cache"):
            if row.get(key):
                p = Path(row[key])
                row[key] = str(p if p.is_absolute() else path.parent/p)
        records.append(Record(**row))
    return Case(data["subject_id"],records,data.get("dataset_domain","research"))


def save_run(out,result):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    # Never silently overwrite a previous run or its evidence history.
    path = out/"result.json"
    with path.open("x",encoding="utf-8") as f:
        json.dump(result.to_dict(),f,ensure_ascii=False,indent=2,allow_nan=False)
    (out/"trace.jsonl").write_text("\n".join(json.dumps(e,ensure_ascii=False,allow_nan=False)
                                for e in result.events)+"\n",encoding="utf-8")
    (out/"answer.md").write_text(result.answer.text+"\n\n状态："+result.answer.status+
                              "\n\n证据："+", ".join(result.answer.evidence_ids)+"\n",encoding="utf-8")


def save_json(path,data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
