"""RGB facial motion proxies; not FACS AU estimates or clinical item scores."""
from pathlib import Path

import numpy as np

from .measurements import Measurement


def lower_lip_position(points):
    """Lower lip (17) in the eye-corner (33,263) 2D similarity coordinate frame."""
    points = np.asarray(points, dtype=float)
    eye_axis = points[263] - points[33]
    scale = np.linalg.norm(eye_axis)
    if not np.isfinite(scale) or scale < 1e-6:
        return np.nan
    vertical = np.array([-eye_axis[1], eye_axis[0]]) / scale
    return float((points[17] - (points[33]+points[263])/2) @ vertical / scale)


def face_sequence_measurements(times, signals, *, fps, evidence_prefix="face"):
    """Do not interpolate detector failures or differentiate across missing frames."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("Face timestamps must be finite and strictly increasing")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Face fps must be positive")
    n = len(times)
    arrays = {k: np.asarray(v, dtype=float) for k, v in signals.items()}
    if any(v.shape != times.shape for v in arrays.values()):
        raise ValueError("Face signals and timestamps must match")
    items, missing = [], {}
    quality = {"sampled_frames": n, "fps": float(fps), "missing": missing,
               "method": "mediapipe_rgb_motion_v2", "valid_counts": {},
               "note": "Task-dependent RGB proxies; no AU conversion or symptom thresholds"}

    def signal(name):
        return arrays.get(name, np.full(n, np.nan))

    def add(name, values, unit="blendshape_score", summary="iqr"):
        values = np.asarray(values)
        valid = values[np.isfinite(values)]
        quality["valid_counts"][name] = len(valid)
        if len(valid) < 5 or len(valid) < .6*n:
            missing[name] = "Insufficient valid frames (need >=5 and >=60% coverage)"
            return
        value = (np.percentile(valid, 75)-np.percentile(valid, 25) if summary == "iqr"
                 else np.mean(valid) if summary == "mean" else np.var(valid))
        items.append(Measurement(name, float(value), unit, "face", f"{evidence_prefix}:{name}"))

    jaw = signal("jawOpen")
    left, right = signal("mouthSmileLeft"), signal("mouthSmileRight")
    add("jaw_open_iqr", jaw)
    add("mouth_smile_iqr", (left+right)/2)
    add("mouth_smile_asymmetry", np.abs(left-right), summary="mean")
    add("lower_lip_motion_iqr", signal("lower_lip_position"), "interocular_ratio")
    add("head_yaw_iqr", signal("head_yaw"), "degrees")  # Legacy descriptive output.
    blink = (signal("eyeBlinkLeft")+signal("eyeBlinkRight"))/2
    add("eye_blink_iqr", blink)  # Kept for legacy 12-feature checkpoints; not a blink rate.
    dt = np.diff(times)
    if len(dt):
        valid = np.isfinite(jaw[:-1]) & np.isfinite(jaw[1:]) & (dt <= 1.5*np.median(dt))
        velocities = np.full(n, np.nan)
        velocities[1:][valid] = np.diff(jaw)[valid]/dt[valid]
        add("jaw_open_velocity_variance", velocities, "blendshape_score^2/s^2", "variance")
    else:
        missing["jaw_open_velocity_variance"] = "No adjacent frames"

    # Count a complete open -> closed -> open cycle with hysteresis.
    observed = np.isfinite(blink)
    observed_seconds = float(observed.sum()/fps)
    effective_fps = 1/np.median(dt) if len(dt) else 0.
    if (fps < 15 or effective_fps < 15 or observed_seconds < 3 or not n
            or observed.mean() < .9 or not (blink[observed] <= .3).any()):
        missing["blink_rate_per_min"] = "Need >=15 fps, >=3s observed, >=90% coverage and open eyes"
    else:
        state, closed_at, events = "unknown", None, 0
        for i, (time, value) in enumerate(zip(times, blink)):
            if not np.isfinite(value) or (i and time-times[i-1] > 1.5/fps):
                state, closed_at = "unknown", None
                continue
            if value <= .3:
                if state == "closed" and .05 <= time-closed_at <= 1.:
                    events += 1
                state, closed_at = "open", None
            elif value >= .6 and state == "open":
                state, closed_at = "closed", time
        quality["blink_events"] = events
        quality["blink_observed_seconds"] = observed_seconds
        items.append(Measurement("blink_rate_per_min", events*60/observed_seconds,
                                 "events/min", "face", f"{evidence_prefix}:blink_rate_per_min"))
    quality["status"] = "measured" if items else "no_valid_detection"
    return items, quality


def face_video_measurements(path, task_path, evidence_prefix="face"):
    """Stream native-rate frames so short eye closures are not lost to 4-fps sampling."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Face video cannot be opened or has no valid frame rate")
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(task_path)),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    names = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight",
             "lower_lip_position", "head_yaw"]
    times, signals, rejected_pose = [], {k: [] for k in names}, 0
    try:
        with vision.FaceLandmarker.create_from_options(options) as detector:
            index, last_ms = 0, -1
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = index/fps
                timestamp_ms = max(last_ms+1, int(round(timestamp*1000)))
                index, last_ms = index+1, timestamp_ms
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=frame), timestamp_ms)
                values = {}
                if result.face_blendshapes and result.face_landmarks and result.facial_transformation_matrixes:
                    matrix = np.asarray(result.facial_transformation_matrixes[0])[:3, :3]
                    yaw = np.degrees(np.arctan2(matrix[0, 2], matrix[2, 2]))
                    pitch = np.degrees(np.arctan2(-matrix[1, 2], np.hypot(matrix[1, 0], matrix[1, 1])))
                    if np.isfinite([yaw, pitch]).all() and abs(yaw) <= 25 and abs(pitch) <= 20:
                        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
                        values = {k: scores.get(k, np.nan) for k in names[:5]}
                        xy = [[p.x*frame.shape[1], p.y*frame.shape[0]] for p in result.face_landmarks[0]]
                        values.update(lower_lip_position=lower_lip_position(xy), head_yaw=yaw)
                    else:
                        rejected_pose += 1
                times.append(timestamp)
                for name in names:
                    signals[name].append(values.get(name, np.nan))
    finally:
        capture.release()
    if not times:
        raise ValueError("Face video has no decodable frames")
    items, quality = face_sequence_measurements(times, signals, fps=fps, evidence_prefix=evidence_prefix)
    quality.update(task_model=Path(task_path).name, rejected_pose_frames=rejected_pose,
                   pose_filter_degrees={"max_abs_yaw": 25, "max_abs_pitch": 20},
                   blink_settings={"closed": .6, "open": .3, "duration_s": [.05, 1.]})
    return items, quality
