"""Subject-bound evidence contracts; raw paths stay outside language-model payloads."""
from dataclasses import dataclass, field, asdict
import math

MODALITIES = ("gait", "face", "speech")
TARGETS = ("pd_presence", "severity", "gait_observation", "face_observation",
           "speech_observation", "evidence_summary")


def bounded(value, low=0., high=1.):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Expected a finite number")
    if not low <= value <= high:
        raise ValueError("Number outside allowed range")


@dataclass
class Record:
    subject_id: str
    record_id: str
    modality: str
    path: str | None = None
    capture_task: str = "unspecified"
    acquired_at: str | None = None
    session_id: str | None = None
    checkpoint: str | None = None
    model_root: str | None = None
    landmark_model: str | None = None
    feature_cache: str | None = None
    windows: list[dict] = field(default_factory=list)
    face_is_cropped: bool = False

    def __post_init__(self):
        if not self.subject_id or not self.record_id or self.modality not in MODALITIES:
            raise ValueError("Record requires subject, record ID and a supported modality")
        for w in self.windows:
            if set(w) != {"start", "end"}:
                raise ValueError("Review windows require start and end")
            bounded(w["start"], 0, 86400)
            bounded(w["end"], 0, 86400)
            if w["end"] <= w["start"]:
                raise ValueError("Review window must have positive duration")


@dataclass
class Case:
    subject_id: str
    records: list[Record]
    dataset_domain: str = "research"

    def __post_init__(self):
        if any(r.subject_id != self.subject_id for r in self.records):
            raise ValueError("Case contains records from a different subject")
        if len({r.modality for r in self.records}) != len(self.records):
            raise ValueError("Select one explicit visit/record per modality")


@dataclass
class Task:
    question: str
    target: str
    modalities: list[str]
    label_definition: str | None = None

    def __post_init__(self):
        if not self.question.strip() or not self.target or not self.modalities:
            raise ValueError("Question, target and requested modalities are required")
        if len(set(self.modalities)) != len(self.modalities) or any(m not in MODALITIES for m in self.modalities):
            raise ValueError("Invalid requested modalities")


@dataclass
class Evidence:
    evidence_id: str
    subject_id: str
    record_id: str
    modality: str
    kind: str
    name: str
    value: float | str
    unit: str
    quality: float
    source: str
    interval: list[float] | None = None
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not all([self.evidence_id, self.subject_id, self.record_id, self.source, self.name]):
            raise ValueError("Evidence requires identifiers and provenance")
        bounded(self.quality)
        if not isinstance(self.value, (str, int, float)) or isinstance(self.value, bool):
            raise ValueError("Evidence value must be text or a finite scalar")
        if isinstance(self.value, (int, float)) and not math.isfinite(self.value):
            raise ValueError("Nonfinite evidence")
        if self.interval is not None:
            if len(self.interval) != 2:
                raise ValueError("Expected a start/end interval")
            bounded(self.interval[0], 0, 86400)
            bounded(self.interval[1], 0, 86400)
            if self.interval[1] <= self.interval[0]:
                raise ValueError("Invalid evidence interval")


@dataclass
class Prediction:
    target: str
    labels: list[str]
    probabilities: list[float]
    model_id: str
    trained: bool
    evidence_ids: list[str]
    definition_id: str | None = None

    def __post_init__(self):
        if len(self.labels) < 2 or len(self.labels) != len(self.probabilities):
            raise ValueError("Prediction needs matching class labels and probabilities")
        if len(set(self.labels)) != len(self.labels) or not self.model_id:
            raise ValueError("Prediction class space/model ID invalid")
        for p in self.probabilities:
            bounded(p)
        if abs(sum(self.probabilities)-1.) > 1e-5:
            raise ValueError("Prediction probabilities must sum to one")

    @property
    def candidate(self):
        return self.labels[max(range(len(self.labels)), key=self.probabilities.__getitem__)]

    @property
    def uncertainty(self):
        return -sum(p*math.log(p) for p in self.probabilities if p > 0)/math.log(len(self.labels))


@dataclass
class ExpertReport:
    subject_id: str
    modality: str
    record_id: str
    target: str
    evidence: list[Evidence]
    prediction: Prediction | None = None
    summary: str = ""
    quality: float = 0.
    limitations: list[str] = field(default_factory=list)
    action: dict | None = None
    tool_error: str | None = None

    def __post_init__(self):
        bounded(self.quality)

    def api_payload(self):
        # Do not include file paths, training labels, or arbitrary manifest fields.
        return {"modality": self.modality, "record_id": self.record_id, "target": self.target,
                "evidence": [asdict(e) for e in self.evidence],
                "prediction": asdict(self.prediction) if self.prediction else None,
                "predictive_uncertainty": self.prediction.uncertainty if self.prediction else None,
                "confidence_method": "uncalibrated_softmax" if self.prediction else None,
                "summary": self.summary, "quality": self.quality, "limitations": self.limitations}


@dataclass
class Issue:
    kind: str
    modalities: list[str]
    reason: str
    evidence_ids: list[str]
    source_status: str = "unverified"


@dataclass
class Review:
    status: str
    issues: list[Issue]


@dataclass
class Fusion:
    reports: list[ExpertReport]
    groups: dict

    def api_payload(self):
        return {"experts": [r.api_payload() for r in self.reports], "groups": self.groups}


@dataclass
class Answer:
    status: str
    text: str
    candidate: str | None
    evidence_ids: list[str]


@dataclass
class RunResult:
    task: Task
    activated_modalities: list[str]
    missing_modalities: list[str]
    history: list[list[ExpertReport]]
    reviews: list[Review]
    fusion: Fusion
    answer: Answer
    review_rounds: int
    stop_reason: str
    events: list[dict]
    retrieved_skills: list[dict] = field(default_factory=list)
    knowledge: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
