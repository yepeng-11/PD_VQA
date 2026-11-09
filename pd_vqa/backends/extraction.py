"""Descriptive measurements, not automatic MDS-UPDRS item scoring."""
from pathlib import Path

import numpy as np

from .measurements import Measurement
from .acoustics import acoustic_measurements, pause_statistics  # noqa: F401
from .facial import face_sequence_measurements, lower_lip_position, face_video_measurements  # noqa: F401


def iqr(values):
    a = np.asarray(values, dtype=float)
    return float(np.percentile(a, 75)-np.percentile(a, 25))


def visual_measurements(path, modality, task_path, evidence_prefix=None):
    if modality == "face":
        return face_video_measurements(path, task_path, evidence_prefix or "face")
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from .encoders import sample_video
    prefix = evidence_prefix or modality
    frames, times = sample_video(path, 8.)
    base = python.BaseOptions(model_asset_path=str(task_path))
    if modality == "gait":
        options = vision.PoseLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.VIDEO,
                                               num_poses=1, min_pose_detection_confidence=.5)
        detector = vision.PoseLandmarker.create_from_options(options)
    else:
        raise ValueError("Expected gait or face")
    collected = {}
    try:
        for frame, timestamp in zip(frames, times):
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame),
                                               int(round(timestamp*1000)))
            if modality == "gait" and result.pose_landmarks:
                points = result.pose_landmarks[0]
                needed = [11, 12, 15, 16, 23, 24, 27, 28]
                if min(points[j].visibility for j in needed) < .5:
                    continue
                # Pixel-aspect-corrected 2D geometry. Ratios, not metric lengths.
                xy = np.array([[p.x*frame.shape[1], p.y*frame.shape[0]] for p in points])
                shoulders, hips = xy[[11, 12]].mean(0), xy[[23, 24]].mean(0)
                scale = np.linalg.norm(shoulders-hips)
                if scale < 1e-6:
                    continue
                values = {"arm_swing_left": (xy[15, 0]-xy[11, 0])/scale,
                          "arm_swing_right": (xy[16, 0]-xy[12, 0])/scale,
                          "ankle_separation_iqr": np.linalg.norm(xy[27]-xy[28])/scale,
                          "trunk_tilt_iqr": np.degrees(np.arctan2(shoulders[0]-hips[0], hips[1]-shoulders[1]))}
            else:
                continue
            for name, value in values.items():
                collected.setdefault(name, []).append(float(value))
    finally:
        detector.close()
    items = []
    for name, values in collected.items():
        if len(values) < 3:
            continue
        unit = "degrees" if name in {"head_yaw_iqr", "trunk_tilt_iqr"} else "torso_ratio" if modality == "gait" else "blendshape_score"
        items.append(Measurement(name, iqr(values), unit, modality, f"{prefix}:{name}"))
    return items, {"sampled_frames": len(frames), "valid_counts": {k: len(v) for k, v in collected.items()},
                   "task_model": Path(task_path).name,
                   "note": "2D variability summaries; no symptom thresholds or calibrated MDS-UPDRS scores"}
