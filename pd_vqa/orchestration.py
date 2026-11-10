"""Task Identification, Agent Activation and Execution Coordination.

The orchestrator allocates work; it never diagnoses, estimates evidence reliability,
or invents a new tool. The critic and collaborative reasoner retain those roles.
"""
from dataclasses import asdict
import hashlib
import json
import re

from .contracts import (MODALITIES, TARGETS, INTENTS, OPERATIONS, Task, ExpertTask,
                        ExpertAssignment, ExecutionPlan, ExpertReport)

MODALITY_TERMS = {
    "gait": ("步态", "步行", "摆臂", "步幅", "步频", "gait", "walking", "arm swing"),
    "face": ("面部", "表情", "眨眼", "face", "facial"),
    "speech": ("语音", "声音", "声学", "录音", "发音", "speech", "audio", "voice"),
}
FOCUS = {
    "gait": "分析步行与身体运动；运动学指标仅报告当前视角和工具支持的量。",
    "face": "分析面部动作与可用测量，保留表情任务、检测质量和视角局限。",
    "speech": "分析语音及声学测量，区分持续元音与连续讲话任务。",
}


def _classification_request(question):
    q = question.casefold()
    # Contrast clauses end negation even without an explicit comma.
    negative_context = re.sub(r"而要|而应|而是|但是请|但请|但要|但需要", "，", q)
    if re.search(r"(?:不要|不需要|无需|不用|不做|不进行)[^，,。.!！？?;；\n]{0,80}(?:分类|诊断|分级)|"
                 r"\b(?:do not|don't|no|without)\b(?:\s+[\w'-]+){0,4}\s+"
                 r"(?:classif\w*|diagnos\w*)|"
                 r"\b(?:do not|don't)\s+(?:predict|detect|estimate|assess)\s+"
                 r"(?:parkinson|pd\b|severity)", negative_context):
        return False
    return bool(re.search(
        r"分类|诊断|患病|分组|分级|等级|评定|属于.{0,8}(类别|疾病|类型)|"
        r"是否.{0,12}(帕金森|pd)|判断.{0,12}帕金森|"
        r"\bclassif\w*|\bdiagnos\w*|"
        r"\b(?:predict|detect|assess|estimate|screen for)\s+(?:(?:the|this|patient's)\s+)?"
        r"(?:parkinson|pd\b|disease|severity)|"
        r"\b(?:does|do|is|are|could|can|whether)\b.{0,100}"
        r"\b(?:have|has|suffer from|consistent with|suggestive of|evidence of|signs of|risk of)\s+"
        r"(?:(?:a|the)\s+)?(?:parkinson|pd\b)|"
        r"which.{0,20}(class|category|group|stage)|评估.{0,8}严重程度", q))


def _goals(question, classification):
    q = question.casefold()
    intents = []
    if re.search(r"描述|观察|表现|动作|describe|observ|movement|analy", q):
        intents.append("observation")
    if re.search(r"测量|计算|指标|幅度|步频|声学|基频|音高|停顿|measure|comput|metric", q):
        intents.append("measurement")
    if classification:
        intents.append("classification")
    if re.search(r"解释|依据|为什么|explain|evidence|reason", q):
        intents.append("explanation")
    return intents or ["observation"]


class Orchestrator:
    def identify(self, question, available, api=None):
        matches = [m for m, words in MODALITY_TERMS.items() if any(w in question.casefold() for w in words)]
        modalities = matches or list(available) or list(MODALITIES)
        requested = _classification_request(question)
        target = ("severity" if re.search(r"严重程度|分组|分级|等级|severity|stage",question.casefold())
                  else "pd_presence") if requested else (
                      modalities[0]+"_observation" if len(modalities)==1 else "evidence_summary")
        intents = _goals(question, requested)
        source, notes = "rules", []
        if api:
            try:
                response = api.complete_json([
                    {"role":"system","content":"Identify the requested task, not a patient's diagnosis. "
                     "Disease names alone do not request classification. Respect explicit requests not to classify. "
                     "Return JSON {\"target\":\"evidence_summary\",\"modalities\":[\"gait\"],"
                     "\"intents\":[\"observation\"]}. Use only supplied enums."},
                    {"role":"user","content":json.dumps({"question":question,"allowed_targets":TARGETS,
                     "allowed_intents":INTENTS,"allowed_modalities":MODALITIES,
                     "available_modalities":available},ensure_ascii=False)}])
                ms = response.get("modalities")
                proposed = response.get("target")
                if (proposed not in TARGETS or not isinstance(ms,list) or not ms
                        or not all(isinstance(m,str) and m in MODALITIES for m in ms)
                        or len(set(ms)) != len(ms)):
                    raise ValueError("Malformed task identification")
                if (proposed in {"pd_presence","severity"}) != requested:
                    notes.append("API classification switch disagreed with explicit question intent; rule task retained")
                    source = "rules_fallback"
                else:
                    target,modalities = proposed,list(dict.fromkeys(matches+ms))
                    source = "llm"
                extra = response.get("intents",[])
                if isinstance(extra,list) and all(i in INTENTS for i in extra):
                    intents = list(dict.fromkeys(intents+[i for i in extra if i != "classification" or requested]))
            except (RuntimeError, ValueError, TypeError, AttributeError):
                notes.append("Task identification API unavailable or invalid; used explicit rules")
                source = "rules_fallback"
        return Task(question,target,modalities,intents=intents,identification_source=source,routing_notes=notes)

    def route(self, case, task):
        selected = [r for r in case.records if r.modality in task.modalities]
        return selected,[m for m in task.modalities if m not in {r.modality for r in selected}]

    def build_plan(self, case, task, *, capabilities=None):
        # Case/Task can be constructed programmatically; validate their identity bindings again.
        case.__post_init__()
        task.__post_init__()
        if capabilities is None:
            from .tools import MediaTools
            capabilities = {r.modality:MediaTools().capabilities(r,task) for r in case.records}
        by_modality = {r.modality:r for r in case.records}
        assignments,missing = [],[]
        requested = task.target in {"pd_presence","severity"}
        for m in task.modalities:
            name = m.title()+"Expert"
            rec = by_modality.get(m)
            if rec is None:
                missing.append(m)
                assignments.append(ExpertAssignment(name,m,"missing_data",None,["record_missing"]))
                continue
            caps = capabilities.get(m,{})
            ops = caps.get("operations",[])
            targets = caps.get("classification_targets",[])
            if (not isinstance(ops,list) or any(op not in OPERATIONS for op in ops)
                    or not isinstance(targets,list) or any(t not in TARGETS for t in targets)):
                raise ValueError("Invalid expert capability registration")
            blocked = []
            enabled = requested and "classify" in ops and task.target in targets
            if requested and not enabled:
                blocked.append("classification_model_unavailable")
            if requested and task.target == "severity" and not task.label_definition:
                blocked.append("severity_definition_missing")
                enabled = False
            allowed = [op for op in ops if op != "classify" or enabled]
            intents = task.intents or _goals(task.question,requested)
            instructions = [FOCUS[m],"仅分析分配的记录，保留原始证据ID、时间范围、单位、质量和局限。"]
            if "measurement" in intents:
                instructions.append("报告实际工具支持的测量；无法测量时明确缺失，不填充推测数值。")
            if "explanation" in intents:
                instructions.append("区分模型预测、实际测量和解释；每项依据均引用来源。")
            instructions.append("执行匹配目标的已训练分类器，并返回概率及不确定性。" if enabled else
                                "关闭分类流；报告可用观察、测量和无法完成的请求，不由LLM补造分类概率。")
            outputs = ["findings","measurements","evidence_ids","source","quality","limitations"]
            if requested:
                outputs += ["classification_status","prediction","probabilities","uncertainty"]
            identity = json.dumps([case.subject_id,rec.record_id,m,task.question,task.target],ensure_ascii=False)
            subtask = ExpertTask(task.question,task.target,[m],task.label_definition,intents,
                task.identification_source,list(task.routing_notes),
                task_id=hashlib.sha256(identity.encode()).hexdigest()[:20],subject_id=case.subject_id,
                record_id=rec.record_id,modality=m,classification_requested=requested,classification_enabled=enabled,
                allowed_operations=allowed,instructions=instructions,output_requirements=outputs)
            ready = any(op in allowed for op in ("inspect_media","read_features","encode","analyze"))
            if not ready:
                blocked.append("no_executable_input")
            assignments.append(ExpertAssignment(name,m,"ready" if ready else "unavailable",subtask,blocked))
        return ExecutionPlan(case.subject_id,task,assignments,missing)

    def coordinate(self, case, plan, expert, events):
        records = {r.modality:r for r in case.records}
        reports = []
        for assignment in plan.assignments:
            if assignment.task is None:
                continue
            task = assignment.task
            self._bind(case,records[assignment.modality],task)
            events.append({"agent":"orchestrator","phase":"dispatch","expert":assignment.agent_name,
                           "task_id":task.task_id,"record_id":task.record_id,"status":assignment.status,
                           "classification_enabled":task.classification_enabled})
            if assignment.status == "ready":
                report = expert.run(records[assignment.modality],task)
            else:
                report = ExpertReport(case.subject_id,task.modality,task.record_id,task.target,[],
                    limitations=["No executable input/tool capability for this expert"],tool_error="CapabilityUnavailable")
            reports.append(report)
            events.append({"agent":"orchestrator","phase":"completed","task_id":task.task_id,
                           "record_id":task.record_id,"tool_error":report.tool_error,
                           "evidence_ids":[e.evidence_id for e in report.evidence]})
        return reports

    def coordinate_review(self, case, plan, expert, actions, reports, review, attempted, events):
        assignments = {a.modality:a for a in plan.assignments if a.task is not None and a.status == "ready"}
        implicated = {m for i in review.issues for m in i.modalities}
        records = {r.modality:r for r in case.records}
        validated = []
        seen = set()
        # Validate the entire dispatch batch before executing any item.
        for item in actions:
            m,action = item.get("modality"),item.get("action")
            if m not in assignments or m not in implicated or m in seen or not isinstance(action,dict):
                raise ValueError("Review plan exceeds activated/implicated expert scope")
            seen.add(m)
            rec,subtask = records[m],assignments[m].task
            self._bind(case,rec,subtask)
            current = next(r for r in reports if r.modality == m)
            if action not in expert.tools.review_options(rec,subtask,current):
                raise ValueError("Review action is not available for the assigned record")
            fingerprint = (m,json.dumps(action,sort_keys=True,allow_nan=False))
            if fingerprint in attempted:
                raise ValueError("Review action already executed")
            validated.append((rec,subtask,action,fingerprint))
        updated = list(reports)
        for rec,subtask,action,fingerprint in validated:
            attempted.add(fingerprint)
            events.append({"agent":"orchestrator","phase":"review_dispatch","task_id":subtask.task_id,
                           "record_id":rec.record_id,"modality":rec.modality,"action":action})
            report = expert.run(rec,subtask,action)
            updated = [report if r.modality == rec.modality else r for r in updated]
            events.append({"agent":"orchestrator","phase":"review_completed","task_id":subtask.task_id,
                           "record_id":rec.record_id,"tool_error":report.tool_error,
                           "evidence_ids":[e.evidence_id for e in report.evidence]})
        return updated

    @staticmethod
    def _bind(case,record,task):
        if (case.subject_id,record.subject_id,record.record_id,record.modality) != (
                task.subject_id,task.subject_id,task.record_id,task.modality):
            raise ValueError("Execution plan subject/record binding changed")

    @staticmethod
    def event(plan):
        return {"agent":"orchestrator","phase":"plan","execution_plan":asdict(plan)}
