"""Local, source-linked medical knowledge; never generates patient observations."""
import json
from pathlib import Path


class ClinicalAgent:
    def __init__(self,path):
        self.cards = json.loads(Path(path).read_text(encoding="utf-8"))
        for card in self.cards:
            if not all(card.get(k) for k in ("knowledge_id","statement","source_url","targets")):
                raise ValueError("Knowledge card missing source or task scope")
        if len({c["knowledge_id"] for c in self.cards}) != len(self.cards):
            raise ValueError("Knowledge identifiers must be unique")

    def retrieve(self,question,target,limit=3):
        relevant = [c for c in self.cards if target in c["targets"]]
        relevant.sort(key=lambda c:sum(w in question for w in c.get("keywords",[])),reverse=True)
        return [{"knowledge_id":c["knowledge_id"],"statement":c["statement"],
                 "source_url":c["source_url"],"reviewed":c.get("reviewed",False),
                 "scope":"general_medical_knowledge_not_patient_evidence"} for c in relevant[:limit]]
