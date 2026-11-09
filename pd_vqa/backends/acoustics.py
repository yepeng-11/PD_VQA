"""Task-aware acoustics. Settings are engineering choices, not clinical cutoffs."""
from importlib.metadata import version

import numpy as np

from .measurements import Measurement

SPEECH_TASKS = ("unspecified", "continuous", "sustained_vowel")


def pause_statistics(flags, frame_seconds=.03, min_pause=.2):
    """Only internal non-speech runs >=min_pause; edges never count as pauses."""
    flags = np.asarray(flags, dtype=bool)
    if flags.ndim != 1 or frame_seconds <= 0 or min_pause <= 0:
        raise ValueError("Invalid VAD sequence or pause settings")
    locations = np.flatnonzero(flags)
    if not len(locations):
        return None
    start, stop = int(locations[0]), int(locations[-1]+1)
    speech = flags[start:stop].copy()
    edges = np.diff(np.r_[False, ~speech, False].astype(int))
    pauses = [(b-a)*frame_seconds for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
              if (b-a)*frame_seconds >= min_pause]
    span = len(speech)*frame_seconds
    return {"start_frame": start, "stop_frame": stop, "span_seconds": span,
            "pause_fraction": float(sum(pauses)/span),
            "pause_duration_median": float(np.median(pauses)) if pauses else None,
            "pause_count": len(pauses), "articulation_seconds": span-sum(pauses)}


def _vad(y):
    import webrtcvad
    # No normalization: gain is not a calibrated sound-pressure measurement.
    width = 480  # 30 ms at 16 kHz; final partial frame is zero padded.
    audio = np.clip(y, -1, 1)
    pcm = (audio*32767).astype("<i2")
    detector = webrtcvad.Vad(2)
    flags = []
    for start in range(0, len(pcm), width):
        frame = np.pad(pcm[start:start+width], (0, max(0, start+width-len(pcm))))
        flags.append(detector.is_speech(frame.tobytes(), 16000))
    return np.asarray(flags, dtype=bool)


def _vowel_quality(y, add, quality):
    import parselmouth
    from parselmouth.praat import call
    names = ["jitter_local", "shimmer_local", "hnr_mean", "cpps"]
    if len(y)/16000 < 1.:
        quality["missing"].update({n: "Need >=1s sustained vowel" for n in names})
        return
    # Exclude the first/last 10%, and use at most the central five seconds.
    length = min(int(len(y)*.8), 80000)
    start = (len(y)-length)//2
    selected = y[start:start+length]
    quality["vowel_window_seconds"] = [start/16000, (start+length)/16000]
    quality["parselmouth_version"] = parselmouth.__version__
    quality["praat_version"] = parselmouth.PRAAT_VERSION
    quality["cpps_settings"] = {"pitch_floor": 60, "time_step": .002, "maximum_frequency": 5000,
                                "pre_emphasis": 50, "subtract_trend": False, "time_smoothing": .01,
                                "quefrency_smoothing": .001, "peak_pitch_range": [60, 330],
                                "tolerance": .05, "interpolation": "Parabolic", "trend_range": [.001, 0],
                                "trend_type": "Straight", "fit": "Robust"}
    if np.max(np.abs(selected)) < 1e-7:
        quality["missing"].update({n: "Silent vowel window" for n in names})
        return
    sound = parselmouth.Sound(selected, sampling_frequency=16000)
    try:
        pulses = call(sound, "To PointProcess (periodic, cc)", 50, 500)
        add("jitter_local", call(pulses, "Get jitter (local)", 0, 0, 1/500, 1/50, 1.3), "fraction")
        add("shimmer_local", call([sound, pulses], "Get shimmer (local)",
                                  0, 0, 1/500, 1/50, 1.3, 1.6), "fraction")
        harmonicity = sound.to_harmonicity_cc(time_step=.01, minimum_pitch=50,
                                             silence_threshold=.1, periods_per_window=4.5)
        valid = harmonicity.values[np.isfinite(harmonicity.values) & (harmonicity.values > -200)]
        add("hnr_mean", np.mean(valid) if len(valid) else np.nan, "dB")
        cepstrogram = call(sound, "To PowerCepstrogram", 60, .002, 5000, 50)
        add("cpps", call(cepstrogram, "Get CPPS", "no", .01, .001, 60, 330, .05,
                         "Parabolic", .001, 0., "Straight", "Robust"), "dB")
    except parselmouth.PraatError as error:
        quality["missing"].update({n: f"Praat extraction failed: {error}" for n in names
                                   if n not in quality["measured_names"]})


def acoustic_measurements(path, evidence_prefix="speech", *, task="unspecified", syllable_count=None,
                          syllable_source=None, return_quality=False):
    import librosa
    from .encoders import load_audio
    if task not in SPEECH_TASKS:
        raise ValueError(f"speech_task must be one of {SPEECH_TASKS}")
    if syllable_count is not None:
        if (task != "continuous" or isinstance(syllable_count, bool) or not isinstance(syllable_count, int)
                or syllable_count < 1 or not isinstance(syllable_source, str) or not syllable_source.strip()):
            raise ValueError("A positive syllable_count needs continuous speech and documented syllable_source")
    y = load_audio(path)
    f0, voiced, _ = librosa.pyin(y, sr=16000, fmin=50, fmax=500, frame_length=1024, hop_length=256)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    out = []
    quality = {"task": task, "status": "measured", "duration_seconds": len(y)/16000,
               "librosa_version": librosa.__version__, "measured_names": [], "missing": {},
               "pitch_settings": {"fmin": 50, "fmax": 500, "frame_length": 1024, "hop_length": 256},
               "note": "Acoustic/temporal proxies, not symptom scores or calibrated loudness"}

    def add(name, value, unit):
        if np.isfinite(value):
            out.append(Measurement(name, float(value), unit, "speech", f"{evidence_prefix}:{name}"))
            quality["measured_names"].append(name)
        else:
            quality["missing"][name] = "No finite estimate"

    # Legacy names/units retained for existing 12-feature checkpoints.
    add("relative_rms_iqr", np.percentile(rms, 75)-np.percentile(rms, 25), "digital_amplitude")
    add("voiced_fraction", voiced.mean(), "fraction")
    valid_f0 = f0[np.isfinite(f0)]
    if len(valid_f0):
        add("f0_median", np.median(valid_f0), "Hz")
        add("f0_iqr", np.percentile(valid_f0, 75)-np.percentile(valid_f0, 25), "Hz")
    if task == "continuous":
        flags = _vad(y)
        stats = pause_statistics(flags)
        quality.update(vad={"implementation": "webrtcvad-wheels", "version": version("webrtcvad-wheels"),
                            "mode": 2, "frame_seconds": .03, "min_pause_seconds": .2},
                       pause_statistics=stats)
        if stats is None or stats["articulation_seconds"] < .5:
            quality["status"] = "no_usable_speech"
            quality["missing"].update({n: "No usable speech span" for n in
                                      ["f0_semitone_iqr", "log_rms_iqr", "pause_fraction",
                                       "pause_duration_median", "articulation_rate"]})
        else:
            add("pause_fraction", stats["pause_fraction"], "fraction")
            if stats["pause_duration_median"] is not None:
                add("pause_duration_median", stats["pause_duration_median"], "s")
            else:
                quality["missing"]["pause_duration_median"] = "No qualifying internal pauses"
            centers = np.arange(len(f0))*256/16000
            vad_mask = flags[np.minimum((centers/.03).astype(int), len(flags)-1)]
            pitches = f0[vad_mask & np.isfinite(f0)]
            if len(pitches) >= 3:
                semitones = 12*np.log2(pitches/np.median(pitches))
                add("f0_semitone_iqr", np.percentile(semitones, 75)-np.percentile(semitones, 25), "semitones")
            else:
                quality["missing"]["f0_semitone_iqr"] = "Too few reliable voiced frames"
            energy = rms[vad_mask & (rms > 1e-7)]
            if len(energy) >= 3:
                log_energy = 20*np.log10(energy)
                add("log_rms_iqr", np.percentile(log_energy, 75)-np.percentile(log_energy, 25), "dB_relative")
            else:
                quality["missing"]["log_rms_iqr"] = "Too few active energy frames"
            if syllable_count is not None:
                add("articulation_rate", syllable_count/stats["articulation_seconds"], "syllables/s")
                quality["syllable_annotation"] = {"count": syllable_count, "source": syllable_source}
            else:
                quality["missing"]["articulation_rate"] = "No documented syllable count; ASR not inferred"
    elif task == "sustained_vowel":
        _vowel_quality(y, add, quality)
    else:
        quality["missing"]["task_dependent_features"] = "Declare speech_task; no automatic task inference"
    if not len(valid_f0) and np.max(np.abs(y)) < 1e-7:
        quality["status"] = "no_usable_speech"
    return (out, quality) if return_quality else out
