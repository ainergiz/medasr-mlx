"""Log-mel frontend, a dependency-free port of ``transformers.LasrFeatureExtractor``.

The reference implementation (JAX upstream, ported to torch in transformers) is:

* frames: ``unfold(win_length=400, hop=160)`` with no centring and no padding,
* window: symmetric Hann of length 400, every frame zero-padded to ``n_fft=512``,
* power spectrum ``|rfft|^2`` projected onto a 128-bin Kaldi-mel filterbank
  spanning 125-7500 Hz (DC bin excluded), then ``log(max(x, 1e-5))``.

There is no pre-emphasis, dithering or per-utterance normalisation. Computation
runs in float64 like the reference, block-wise so hour-long inputs stay cheap.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import FrontendConfig


def _hertz_to_mel_kaldi(freq: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def mel_filterbank(
    num_mel_bins: int = 128,
    num_spectrogram_bins: int = 257,
    sample_rate: int = 16_000,
    lower_edge_hz: float = 125.0,
    upper_edge_hz: float = 7500.0,
) -> np.ndarray:
    """``[num_spectrogram_bins, num_mel_bins]`` float64 triangular filterbank."""
    nyquist = sample_rate / 2.0
    # HTK-style: exclude the DC bin, then pad it back as an all-zero row.
    linear_freqs = np.linspace(0.0, nyquist, num_spectrogram_bins, dtype=np.float64)[1:]
    spec_mel = _hertz_to_mel_kaldi(linear_freqs)[:, None]
    edges = np.linspace(
        _hertz_to_mel_kaldi(lower_edge_hz),
        _hertz_to_mel_kaldi(upper_edge_hz),
        num_mel_bins + 2,
        dtype=np.float64,
    )
    lower, center, upper = edges[:-2][None, :], edges[1:-1][None, :], edges[2:][None, :]
    lower_slopes = (spec_mel - lower) / (center - lower)
    upper_slopes = (upper - spec_mel) / (upper - center)
    weights = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))
    return np.pad(weights, [[1, 0], [0, 0]])


@lru_cache(maxsize=4)
def _cached_constants(cfg: FrontendConfig) -> tuple[np.ndarray, np.ndarray]:
    n = cfg.win_length
    # torch.hann_window(n, periodic=False)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / (n - 1))
    filters = mel_filterbank(
        cfg.num_mel_bins,
        cfg.n_fft // 2 + 1,
        cfg.sample_rate,
        cfg.lower_edge_hz,
        cfg.upper_edge_hz,
    )
    window.setflags(write=False)
    filters.setflags(write=False)
    return window, filters


def log_mel_spectrogram(
    audio: np.ndarray,
    config: FrontendConfig | None = None,
    *,
    block_frames: int = 8192,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Compute ``[num_frames, num_mel_bins]`` log-mel features for mono audio.

    ``audio`` is a 1-D float waveform at ``config.sample_rate``. Inputs shorter
    than one analysis window yield an empty ``[0, num_mel_bins]`` array.
    """
    cfg = config or FrontendConfig()
    x = np.asarray(audio)
    if x.ndim != 1:
        raise ValueError(f"Expected mono 1-D audio, got shape {x.shape}")
    # The reference casts to float32 first, then to float64 for the STFT.
    x = x.astype(np.float32, copy=False).astype(np.float64)
    num_frames = cfg.num_frames(x.shape[0])
    out = np.empty((num_frames, cfg.num_mel_bins), dtype=dtype)
    if num_frames == 0:
        return out
    window, filters = _cached_constants(cfg)
    frames = np.lib.stride_tricks.sliding_window_view(x, cfg.win_length)[:: cfg.hop_length]
    for start in range(0, num_frames, block_frames):
        block = frames[start : start + block_frames] * window
        spec = np.fft.rfft(block, n=cfg.n_fft, axis=-1)
        power = spec.real**2 + spec.imag**2
        mel = power @ filters
        np.log(np.maximum(mel, cfg.log_floor), out=mel)
        out[start : start + block.shape[0]] = mel
    return out


def batch_log_mel(
    audios: list[np.ndarray], config: FrontendConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad a batch and return ``(features [B, T, F], frame_mask [B, T])``.

    Padding semantics match the reference processor: audio is zero-padded to
    the longest item before framing, and a frame is valid iff its last sample
    is a real sample.
    """
    cfg = config or FrontendConfig()
    lengths = [int(np.asarray(a).shape[0]) for a in audios]
    max_len = max(lengths, default=0)
    padded = np.zeros((len(audios), max_len), dtype=np.float32)
    for i, a in enumerate(audios):
        padded[i, : lengths[i]] = a
    n_frames = cfg.num_frames(max_len)
    feats = np.stack([log_mel_spectrogram(p, cfg) for p in padded]) if audios else None
    if feats is None:
        return np.zeros((0, 0, cfg.num_mel_bins), np.float32), np.zeros((0, 0), bool)
    valid = np.array([cfg.num_frames(n) for n in lengths])
    mask = np.arange(n_frames)[None, :] < valid[:, None]
    return feats, mask
