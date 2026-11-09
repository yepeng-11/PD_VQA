"""Frozen local-only encoders. Downloading is a separate explicit command."""
import gc
import json
from pathlib import Path

import numpy as np
import torch

MODEL_IDS = {"gait": "MCG-NJU/videomae-base", "face": "facebook/dinov2-small",
             "speech": "microsoft/wavlm-base-plus", "text": "BAAI/bge-small-zh-v1.5"}


def download_models(root, modalities=None):
    from huggingface_hub import HfApi, snapshot_download
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name in modalities or list(MODEL_IDS):
        repo = MODEL_IDS[name]
        revision = HfApi().model_info(repo).sha
        location = root / name
        snapshot_download(repo, revision=revision, local_dir=location,
                          allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "pytorch_model.bin"],
                          ignore_patterns=["tf_model*", "flax_model*"])
        metadata[name] = {"repo": repo, "revision": revision}
        (location / "pd_source.json").write_text(json.dumps(metadata[name], indent=2), encoding="utf-8")
    return metadata


def sample_video(path, target_fps=8., max_frames=4096):
    import cv2
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Video cannot be opened or has no valid frame rate")
    frames, times = [], []
    index, next_time = 0, 0.
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = index/fps
            index += 1
            if timestamp+1e-8 < next_time:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            times.append(timestamp)
            next_time += 1/target_fps
            if len(frames) > max_frames:
                raise ValueError("Video exceeds the sampling limit; split recordings explicitly")
    finally:
        capture.release()
    if not frames:
        raise ValueError("Video has no decodable frames")
    return frames, times


def frame_windows(frames, times, size=16):
    if not frames or len(frames) != len(times):
        raise ValueError("Nonempty matching frames/times required")
    for start in range(0, len(frames), size):
        group, ts = list(frames[start:start+size]), list(times[start:start+size])
        while len(group) < size:
            group.append(group[-1])
            ts.append(ts[-1])
        yield group, ts


def load_audio(path):
    import soundfile as sf
    import librosa
    signal, rate = sf.read(str(path), always_2d=True, dtype="float32")
    signal = signal.mean(-1)
    if len(signal) < rate*.1 or not np.isfinite(signal).all():
        raise ValueError("Audio must contain >=0.1 s of finite samples")
    if rate != 16000:
        signal = librosa.resample(signal, orig_sr=rate, target_sr=16000)
    return signal


def crop_face(frame):
    import cv2
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 1.1, 4)
    if len(faces) != 1:
        return None
    x, y, w, h = faces[0]
    margin = int(max(w, h)*.15)
    return frame[max(0, y-margin):y+h+margin, max(0, x-margin):x+w+margin]


class FrozenEncoders:
    def __init__(self, model_root, device="cpu", face_is_cropped=False):
        self.root, self.device, self.face_is_cropped = Path(model_root), device, face_is_cropped

    @torch.inference_mode()
    def encode(self, modality, path):
        location = self.root / modality
        if not (location / "config.json").is_file():
            raise FileNotFoundError(f"Local model missing: {location}; run download-models first")
        from transformers import AutoImageProcessor, AutoFeatureExtractor, AutoModel
        model = AutoModel.from_pretrained(location, local_files_only=True).eval().to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        values, timestamps = [], []
        try:
            if modality == "gait":
                processor = AutoImageProcessor.from_pretrained(location, local_files_only=True)
                frames, times = sample_video(path, 8.)
                for window, ts in frame_windows(frames, times, model.config.num_frames):
                    batch = processor(window, return_tensors="pt").to(self.device)
                    features = model(**batch).last_hidden_state[0]
                    temporal = model.config.num_frames // model.config.tubelet_size
                    features = features.reshape(temporal, -1, features.shape[-1]).mean(1)
                    values.append(features.cpu())
                    timestamps.extend([float(np.mean(ts[i:i+model.config.tubelet_size]))
                                       for i in range(0, len(ts), model.config.tubelet_size)])
            elif modality == "face":
                processor = AutoImageProcessor.from_pretrained(location, local_files_only=True)
                frames, times = sample_video(path, 4.)
                crops, valid_times = [], []
                for frame, ts in zip(frames, times):
                    crop = frame if self.face_is_cropped else crop_face(frame)
                    if crop is not None:
                        crops.append(crop)
                        valid_times.append(ts)
                if not crops:
                    raise ValueError("No unambiguous face detected; provide a cropped single-person face recording")
                for i in range(0, len(crops), 16):
                    batch = processor(images=crops[i:i+16], return_tensors="pt").to(self.device)
                    values.append(model(**batch).last_hidden_state[:, 0].cpu())
                timestamps = valid_times
            elif modality == "speech":
                processor = AutoFeatureExtractor.from_pretrained(location, local_files_only=True)
                signal = load_audio(path)
                for start in range(0, len(signal), 160000):
                    chunk = signal[start:start+160000]
                    if len(chunk) < 800:
                        continue
                    batch = processor(chunk, sampling_rate=16000, return_tensors="pt").to(self.device)
                    features = model(**batch).last_hidden_state[0]
                    step = len(chunk)/16000/len(features)
                    # Adjacent 5 frames pooled to reduce token count, preserving chronology.
                    pooled = [features[i:i+5].mean(0) for i in range(0, len(features), 5)]
                    values.append(torch.stack(pooled).cpu())
                    timestamps.extend([start/16000 + min(i+2, len(features)-1)*step
                                       for i in range(0, len(features), 5)])
            else:
                raise ValueError("Expected gait, face or speech")
        finally:
            del model
            gc.collect()
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
        if not values:
            raise ValueError("No valid encoded features")
        provenance = location / "pd_source.json"
        source = (json.loads(provenance.read_text()) if provenance.exists()
                  else {"repo": MODEL_IDS[modality], "revision": "unrecorded"})
        source["preprocessing"] = f"pd_vqa_sequence_v1:{modality}:face_cropped={self.face_is_cropped}"
        return torch.cat(values), torch.tensor(timestamps, dtype=torch.float64), source
