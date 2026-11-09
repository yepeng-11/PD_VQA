"""Unit-preserving observations and a training-only standardizer."""
from dataclasses import asdict, dataclass
import json

import numpy as np


@dataclass
class Measurement:
    name: str
    value: float
    unit: str
    modality: str
    evidence_id: str


def serialize_measurements(items):
    return json.dumps([asdict(item) if isinstance(item, Measurement) else item for item in items],
                      ensure_ascii=False, allow_nan=False)


def describe_matches(weights, names=None, reviewed=False):
    index = int(np.argmax(weights))
    name = (names or {}).get(str(index), f"prototype_{index}") if reviewed else f"prototype_{index}"
    return json.dumps({"prototype": name, "weight": float(weights[index]), "reviewed": bool(reviewed),
                       "interpretation": "similarity weight, not symptom probability"}, ensure_ascii=False)


class MeasurementScaler:
    def fit(self, rows):
        array = np.asarray(rows, dtype=float)
        if array.ndim != 2 or not len(array):
            raise ValueError("Training measurements must be a nonempty matrix")
        valid = np.isfinite(array)
        count = valid.sum(0)
        self.mean = np.where(valid, array, 0).sum(0) / np.maximum(count, 1)
        variance = np.where(valid, (array - self.mean) ** 2, 0).sum(0) / np.maximum(count, 1)
        self.std = np.sqrt(variance)
        self.std[self.std < 1e-8] = 1
        self.observed = count > 0
        return self

    def transform(self, rows):
        array = np.asarray(rows, dtype=float)
        if array.ndim != 2 or array.shape[1] != len(self.mean):
            raise ValueError("Measurement schema differs from training schema")
        mask = np.isfinite(array) & self.observed
        return np.where(mask, (array-self.mean)/self.std, 0).astype(np.float32), mask

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "observed": self.observed.tolist()}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls()
        obj.mean, obj.std = np.array(state["mean"]), np.array(state["std"])
        obj.observed = np.array(state["observed"], dtype=bool)
        return obj
