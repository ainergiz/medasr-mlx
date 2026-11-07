"""Audio loading and resampling.

``load_audio`` returns 16 kHz mono float32. Decoding uses ``soundfile``
(WAV/FLAC/OGG/MP3) and falls back to ``ffmpeg`` for everything else (m4a,
aac, video containers...). Resampling uses ``soxr`` when installed, otherwise
a built-in Kaiser-windowed polyphase resampler (equivalent to
``scipy.signal.resample_poly``).
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE


def to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio)
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / float(np.iinfo(a.dtype).max + 1)
    if a.ndim == 2:
        # soundfile gives [frames, channels]; accept [channels, frames] for short channel axes.
        axis = 1 if a.shape[1] <= 8 else 0
        a = a.mean(axis=axis)
    elif a.ndim != 1:
        raise ValueError(f"Expected 1-D or 2-D audio, got shape {a.shape}")
    return a.astype(np.float32, copy=False)


def _firwin_kaiser(numtaps: int, cutoff: float, beta: float = 5.0) -> np.ndarray:
    """Low-pass FIR (cutoff relative to Nyquist), like ``scipy.signal.firwin``."""
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    h = cutoff * np.sinc(cutoff * n) * np.kaiser(numtaps, beta)
    return h / h.sum()


def resample_poly(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """Polyphase resampling by ``up / down`` (scipy-compatible filter design)."""
    g = math.gcd(up, down)
    up, down = up // g, down // g
    if up == down == 1:
        return x.astype(np.float32, copy=False)
    max_rate = max(up, down)
    half_len = 10 * max_rate
    h = _firwin_kaiser(2 * half_len + 1, 1.0 / max_rate) * up
    # Pad h so each polyphase branch has the same length.
    taps = -(-h.size // up)
    h = np.pad(h, (0, taps * up - h.size))
    phases = h.reshape(taps, up).T[:, ::-1]  # [up, taps], reversed for correlation
    n_out = -(-x.size * up // down)
    # Output n corresponds to upsampled index n*down; centre the filter delay.
    delay = half_len
    m = np.arange(n_out, dtype=np.int64) * down + delay
    phase = m % up
    base = m // up  # index of the newest input sample touched
    xp = np.concatenate(
        [np.zeros(taps, np.float64), x.astype(np.float64), np.zeros(taps, np.float64)]
    )
    out = np.empty(n_out, np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(xp, taps)
    for p in range(up):
        idx = np.flatnonzero(phase == p)
        if idx.size:
            # window ending at input sample base[idx] (shifted by the leading pad)
            out[idx] = windows[base[idx] + 1] @ phases[p]
    return out.astype(np.float32)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return audio.astype(np.float32, copy=False)
    try:
        import soxr

        return soxr.resample(audio, orig_sr, target_sr, quality="HQ").astype(np.float32, copy=False)
    except ImportError:
        return resample_poly(audio, int(target_sr), int(orig_sr))


def _load_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            f"Cannot decode {path.suffix or 'this'} audio with soundfile and ffmpeg is not "
            "installed (brew install ffmpeg)."
        )
    cmd = [exe, "-nostdin", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
           "-ar", str(sample_rate), "-"]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load any audio file as mono float32 at ``sample_rate``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:  # unsupported container/codec -> ffmpeg
        return _load_ffmpeg(path, sample_rate)
    return resample(to_mono(data), int(sr), sample_rate)


def as_waveform(
    audio: str | Path | np.ndarray, sample_rate: int | None = None, target_sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Accept a path or an array (+ its sample rate) and return 16 kHz mono float32."""
    if isinstance(audio, (str, Path)):
        return load_audio(audio, target_sr)
    wav = to_mono(np.asarray(audio))
    return resample(wav, sample_rate or target_sr, target_sr)
