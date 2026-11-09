"""SkeMex-inspired, explicitly simplified skill memory. No automatic diagnostic self-labels."""
import hashlib
import json
from pathlib import Path
import sqlite3
from contextlib import closing


class SkillMemory:
    def __init__(self, path, *, frozen=False):
        self.path,self.frozen = Path(path),frozen
        if frozen and not self.path.is_file():
            raise FileNotFoundError("Frozen skill database does not exist")
        if not frozen:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with closing(self._connect()) as c, c:
                c.executescript("""
                CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY, modality TEXT, trigger TEXT, instruction TEXT,
                scope TEXT, traces TEXT, successes INTEGER DEFAULT 0, failures INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS feedback (
                skill_id TEXT, reference_id TEXT, source TEXT, success INTEGER, split TEXT,
                PRIMARY KEY(skill_id, reference_id));
                """)

    def _connect(self):
        if self.frozen:
            c = sqlite3.connect(self.path.resolve().as_uri()+"?mode=ro",uri=True)
        else:
            c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def propose(self, modality, trigger, instruction, trace_ids, *, scope="procedural"):
        if self.frozen:
            raise ValueError("Frozen memory cannot be changed")
        if modality not in {"gait","face","speech"} or not trigger or len(instruction.strip()) < 12 or not trace_ids:
            raise ValueError("A candidate needs a trigger, concrete action and source trace")
        if scope not in {"procedural","diagnostic"}:
            raise ValueError("Invalid memory scope")
        sid = hashlib.sha256(json.dumps([modality,trigger,instruction,scope]).encode()).hexdigest()[:20]
        with closing(self._connect()) as c, c:
            c.execute("INSERT OR IGNORE INTO skills(skill_id,modality,trigger,instruction,scope,traces) "
                      "VALUES(?,?,?,?,?,?)",(sid,modality,trigger,instruction,scope,json.dumps(trace_ids)))
        return sid

    def feedback(self, skill_id, *, success, source, reference_id, split):
        if self.frozen or split not in {"train","development"}:
            raise ValueError("Only development/training feedback may update memory")
        if source not in {"clinician_review","benchmark_label","tool_validation"} or not reference_id:
            raise ValueError("Feedback requires an external source and reference")
        if type(success) is not bool:
            raise ValueError("Feedback success must be a boolean")
        with closing(self._connect()) as c, c:
            row = c.execute("SELECT * FROM skills WHERE skill_id=?",(skill_id,)).fetchone()
            if not row:
                raise ValueError("Unknown skill")
            if row["scope"] == "diagnostic" and source == "tool_validation":
                raise ValueError("Tool quality feedback cannot validate diagnostic knowledge")
            try:
                c.execute("INSERT INTO feedback VALUES(?,?,?,?,?)",(skill_id,reference_id,source,int(success),split))
            except sqlite3.IntegrityError:
                raise ValueError("Duplicate feedback reference") from None
            col = "successes" if success else "failures"
            c.execute(f"UPDATE skills SET {col}={col}+1 WHERE skill_id=?",(skill_id,))

    def retrieve(self, modality, trigger, *, limit=3, scope="procedural"):
        with closing(self._connect()) as c:
            rows = c.execute("SELECT * FROM skills WHERE modality=? AND trigger=? AND scope=? "
                             "AND successes>failures ORDER BY successes-failures DESC LIMIT ?",
                             (modality,trigger,scope,limit)).fetchall()
        return [{**dict(r),"utility":(1+r["successes"])/(2+r["successes"]+r["failures"])} for r in rows]


class ReflectionAgent:
    """Distill executable actions into candidates; correctness requires separate feedback."""
    def propose_from_run(self, result, memory, trace_id, *, split="test"):
        if split not in {"train","development"}:
            raise ValueError("No memory writing from held-out cases")
        proposed = []
        for event in result.events:
            if event["agent"] != "collaborative_reasoner":
                continue
            for item in event["plan"]:
                instruction = "Recheck available source using "+json.dumps(item["action"],sort_keys=True)
                proposed.append(memory.propose(item["modality"],"low_quality",instruction,[trace_id]))
        return list(dict.fromkeys(proposed))
