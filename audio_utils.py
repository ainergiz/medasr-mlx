#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _to_mono_f32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        return np.mean(audio, axis=1).astype(np.float32, copy=False)
    return np.asarray(audio).reshape(-1).astype(np.float32, copy=False)


def _resample_if_needed(audio: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return audio, sr
    g = math.gcd(int(sr), int(target_sr))
    up = int(target_sr) // g
    down = int(sr) // g
    out = resample_poly(audio, up, down).astype(np.float32, copy=False)
    return out, target_sr


def load_audio_mono(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 with robust fallback when librosa/numba is unavailable."""
    path = Path(path)
    try:
        import librosa

        speech, sr = librosa.load(str(path), sr=target_sr)
        return speech.astype(np.float32, copy=False), int(sr)
    except Exception:
        speech, sr = sf.read(str(path), dtype="float32", always_2d=False)
        speech = _to_mono_f32(np.asarray(speech))
        speech, sr = _resample_if_needed(speech, int(sr), int(target_sr))
        return speech, int(sr)
