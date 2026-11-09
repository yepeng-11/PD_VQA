"""Executable media tools. Measurements are computed before any LLM explanation."""
import array
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import wave

from .contracts import Evidence,ExpertReport,Prediction


def source_digest(path):
    digest = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def audio_window(path,start=0.,end=None):
    """Read PCM WAV without ML dependencies; other formats use optional soundfile."""
    try:
        with wave.open(str(path),"rb") as f:
            rate,channels,width,n = f.getframerate(),f.getnchannels(),f.getsampwidth(),f.getnframes()
            duration = n/rate
            end = duration if end is None else end
            if not 0 <= start < end <= duration+1e-6:
                raise ValueError("Requested window is outside the recording")
            f.setpos(int(start*rate))
            raw = f.readframes(int((end-start)*rate))
        if width != 2:
            raise wave.Error("Use optional decoder for non-16bit PCM")
        samples = array.array("h",raw)
        if sys.byteorder != "little":
            samples.byteswap()
        values = [sum(samples[i:i+channels])/channels/32768 for i in range(0,len(samples),channels)]
        return values,rate,[start,end]
    except (wave.Error,EOFError):
        import soundfile as sf
        info = sf.info(path)
        end = info.duration if end is None else end
        if not 0 <= start < end <= info.duration+1e-6:
            raise ValueError("Requested window is outside the recording")
        y,rate = sf.read(path,start=int(start*info.samplerate),stop=int(end*info.samplerate),always_2d=True)
        return y.mean(1).tolist(),rate,[start,end]


class MediaTools:
    def __init__(self,*,workspace="runs/tool_cache",device="cpu"):
        self.workspace,self.device = Path(workspace),device

    def review_options(self,record,task,current):
        if not record.path or not Path(record.path).is_file():
            return []
        return [{"operation":"window",**w} for w in record.windows]

    def analyze(self,record,task,action=None):
        start,end = 0.,None
        if action:
            if action.get("operation") != "window" or set(action) != {"operation","start","end"}:
                raise ValueError("Only explicitly available window actions are supported")
            if {"start":action["start"],"end":action["end"]} not in record.windows:
                raise ValueError("Undeclared review window")
            start,end = action["start"],action["end"]
        if not record.path and not record.feature_cache:
            raise FileNotFoundError("No recording or feature cache")
        fingerprint = hashlib.sha256(json.dumps([record.record_id,action],sort_keys=True).encode()).hexdigest()[:12]
        prefix = record.modality+":"+record.record_id+":"+fingerprint
        evidence,limitations = [],[]
        quality,interval = 0.,None

        def add(name,value,unit,source="signal_measurement:v1"):
            evidence.append(Evidence(prefix+":"+name,record.subject_id,record.record_id,record.modality,
                                     "measurement",name,float(value),unit,quality,source,interval,
                                     ["Descriptive measurement; no clinical cutoff inferred"]))

        if record.path:
            if record.modality == "speech":
                y,rate,interval = audio_window(record.path,start,end)
                if not y:
                    raise ValueError("Empty audio window")
                rms = (sum(v*v for v in y)/len(y))**.5
                clipping = sum(abs(v)>=.999 for v in y)/len(y)
                quality = max(.05,1.-2.5*clipping) if rms > 1e-6 else .05
                add("rms_digital",rms,"digital_amplitude")
                add("clipping_fraction",clipping,"fraction")
                add("duration_seconds",len(y)/rate,"s")
                limitations.append("RMS is not calibrated sound pressure; quality threshold is an engineering choice")
            else:
                import cv2
                cap = cv2.VideoCapture(str(record.path))
                try:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps if fps > 0 else 0
                    end = duration if end is None else end
                    if not cap.isOpened() or not 0 <= start < end <= duration+1e-6:
                        raise ValueError("Video/window invalid")
                    interval = [start,end]
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(start*fps))
                    count,blur = 0,[]
                    for _ in range(min(int((end-start)*fps),4096)):
                        ok,frame = cap.read()
                        if not ok:
                            break
                        if count % max(1,int(fps/4)) == 0:
                            blur.append(float(cv2.Laplacian(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()))
                        count += 1
                    quality = min(1.,count/max(1,int((end-start)*fps)))
                    add("decoded_frame_fraction",quality,"fraction","opencv_metadata:v1")
                    if blur:
                        add("laplacian_variance",sum(blur)/len(blur),"pixel_intensity_squared","opencv:v1")
                    limitations.append("Decode quality does not establish pose/face detection or clinical validity")
                finally:
                    cap.release()
        prediction = None
        if (record.feature_cache or record.model_root or record.landmark_model or
                (record.modality == "speech" and record.capture_task in {"continuous","sustained_vowel"})):
            self.workspace.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self.workspace) as temp:
                selected = Path(record.path) if record.path else None
                # Tool outputs are calculated from the selected source, not rewritten by the LLM.
                if action and selected:
                    selected = materialize_window(record.path,record.modality,start,end,Path(temp))
                if record.landmark_model and selected and record.modality in {"gait","face"}:
                    from .backends.extraction import visual_measurements
                    measured,q = visual_measurements(selected,record.modality,record.landmark_model,prefix)
                    valid = list(q.get("valid_counts",{}).values())
                    quality = min(quality,min(valid)/max(1,q.get("sampled_frames",1))) if valid else 0.
                    for item in measured:
                        add(item.name,item.value,item.unit,"mediapipe_2d:v2")
                    for e in evidence:
                        e.quality = min(e.quality,quality)
                if selected and record.modality == "speech" and record.capture_task in {"continuous","sustained_vowel"}:
                    try:
                        from .backends.acoustics import acoustic_measurements
                        measured,_ = acoustic_measurements(selected,prefix,task=record.capture_task,return_quality=True)
                        for item in measured:
                            add(item.name,item.value,item.unit,"task_aware_acoustics:v2")
                    except ImportError:
                        limitations.append("Optional task-aware acoustic dependencies unavailable")
                if record.feature_cache or record.model_root:
                    import torch
                    if record.feature_cache:
                        cache = torch.load(record.feature_cache,map_location="cpu",weights_only=True)
                        if (cache.get("subject_id"),cache.get("record_id"),cache.get("modality")) != (
                                record.subject_id,record.record_id,record.modality):
                            raise ValueError("Feature cache subject/record/modality mismatch")
                        if record.path:
                            if cache.get("raw_source_sha256") != source_digest(record.path):
                                raise ValueError("Cached features do not match this raw recording source")
                            coverage = cache.get("source_interval")
                            if (not isinstance(coverage,list) or len(coverage) != 2
                                    or cache.get("timestamp_origin_seconds") != 0.
                                    or not coverage[0] <= interval[0] < interval[1] <= coverage[1]+1e-6):
                                raise ValueError("Cached source interval/origin does not cover the requested recording")
                        x,times = cache["features"],cache["timestamps"]
                        if action:
                            keep = (times>=start)&(times<end)
                            x,times = x[keep],times[keep]
                        if len(x)==0:
                            raise ValueError("No features in requested window")
                        if not record.path:
                            q = cache.get("quality_metadata",{})
                            score = q.get("score")
                            quality = (float(score) if isinstance(score,(int,float)) and not isinstance(score,bool)
                                       and 0 <= score <= 1 and q.get("method") else 0.)
                            interval = None  # Feature coverage is not a raw-data quality measurement.
                            limitations.append("Raw recording unavailable; absent extraction quality is unknown, not perfect")
                        source = cache.get("source",{})
                        contract = cache.get("feature_contract")
                    else:
                        from .backends.encoders import FrozenEncoders
                        x,times,source = FrozenEncoders(record.model_root,self.device,record.face_is_cropped).encode(
                            record.modality,selected)
                        times = times+start
                        from .models import contract_from_source
                        contract = contract_from_source(source,x.shape[-1])
                    if x.ndim != 2 or len(x) != len(times) or not torch.isfinite(x).all():
                        raise ValueError("Invalid encoded sequence")
                    if (times.ndim != 1 or not torch.isfinite(times).all() or len(times)==0
                            or (torch.diff(times)<0).any()):
                        raise ValueError("Encoded timestamps must be finite and chronological")
                    if interval and (times.min()<interval[0]-1e-6 or times.max()>interval[1]+1e-6):
                        raise ValueError("Encoded timestamps outside raw source interval")
                    cache_path = self.workspace/(prefix.replace(":","_")+".pt")
                    torch.save({"subject_id":record.subject_id,"record_id":record.record_id,
                                "modality":record.modality,"features":x,"timestamps":times,"source":source,
                                "feature_contract":contract,"quality_metadata":{"score":quality,
                                "method":"record_tool_quality_v1" if record.path else "copied_cache_quality"},
                                "raw_source_sha256":source_digest(record.path) if record.path else cache.get("raw_source_sha256"),
                                "source_interval":interval if record.path else cache.get("source_interval"),
                                "timestamp_origin_seconds":0.},cache_path)
                    evidence.append(Evidence(prefix+":features",record.subject_id,record.record_id,record.modality,
                        "model_input","encoded_sequence_length",len(x),"tokens",quality,"frozen_encoder:v1",interval,
                        ["Latent features are not directly observed symptoms"]))
                    if record.checkpoint:
                        from .models import load_predictor
                        model,labels,digest = load_predictor(record.checkpoint,target=task.target,
                            modality=record.modality,device=self.device,feature_contract=contract,
                            label_definition=task.label_definition)
                        with torch.inference_mode():
                            probs = model(x.float().unsqueeze(0).to(self.device),
                                          torch.ones(1,len(x),dtype=torch.bool,device=self.device)).softmax(-1)[0].tolist()
                        from .models import definition_hash
                        prediction = Prediction(task.target,labels,probs,digest,True,[prefix+":features"],
                            definition_hash(task.label_definition) if task.target == "severity" else None)
        if not record.checkpoint:
            limitations.append("No trained checkpoint: no Parkinson/severity prediction is produced")
        summary = "；".join(f"{e.name}={e.value:.4g} {e.unit}" for e in evidence if isinstance(e.value,(float,int)))
        return ExpertReport(record.subject_id,record.modality,record.record_id,task.target,evidence,
                            prediction,summary,quality,limitations,action)


def materialize_window(path,modality,start,end,destination):
    if modality == "speech":
        y,rate,_ = audio_window(path,start,end)
        out = destination/"selected.wav"
        values = array.array("h",[int(max(-1.,min(.9999,v))*32768) for v in y])
        if sys.byteorder != "little":
            values.byteswap()
        with wave.open(str(out),"wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(values.tobytes())
        return out
    import cv2
    out = destination/"selected.avi"
    cap = cv2.VideoCapture(str(path))
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(start*fps))
        for _ in range(int((end-start)*fps)):
            ok,frame = cap.read()
            if not ok:
                break
            if writer is None:
                writer = cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"MJPG"),fps,
                                         (frame.shape[1],frame.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("Video window writer unavailable")
            writer.write(frame)
    finally:
        cap.release()
        if writer:
            writer.release()
    if not out.is_file():
        raise RuntimeError("No video window could be decoded")
    return out
