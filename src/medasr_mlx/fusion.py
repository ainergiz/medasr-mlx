"""Temporal Posterior Fusion (TPF) - long-form and streaming inference.

Implements the inference recipe from the MedASR technical report
(arXiv:2605.16555, Sections 2.3 and 3.4). The unmodified full-context model is
run on sliding windows of length ``W`` with stride ``S``. Every encoder frame
``t`` covered by windows ``K_t`` gets the fused posterior

    P(y_t) = sum_k  w[rel(t, k)] * softmax(z_{t,k})  /  sum_k w[rel(t, k)]

where ``w`` is a Hann window over the frame's position inside each window, so
frames with little left or right context count less.

* Offline long-form: ``W = 20 s``, ``S = 18 s`` (the report's setting).
* Streaming: a small stride (e.g. 320 ms) and ``W`` seconds of leading zero
  padding, so the first window ends at the start of the audio.

MedASR is fine-tuned on segments of at most 20 s, and a single full-context
pass over long audio (v1's behaviour) degrades badly. On 2.7-minute
LibriSpeech readings, WER was 32.6% with one pass versus 20.6% with TPF 20/18.
On Eka clinical sessions of 28-130 s it was 24.1% versus 22.9%. Hann averaging
and hard "center" selection performed about the same.

Window starts are kept on the 640-sample (40 ms) encoder-frame grid, so frame
``j`` of a window starting at sample ``B`` is exactly global frame ``B / 640 + j``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import ModelConfig

# Paper defaults.
OFFLINE_WINDOW_S = 20.0
OFFLINE_STRIDE_S = 18.0


@lru_cache(maxsize=64)
def hann_weights(num_frames: int) -> np.ndarray:
    """Strictly positive Hann weights (the zero end-points are discarded)."""
    w = np.hanning(num_frames + 2)[1:-1].astype(np.float32)
    w.setflags(write=False)
    return w


@dataclass(frozen=True)
class Window:
    start: int  # sample offset, a multiple of ``samples_per_frame``
    length: int  # samples
    frame_offset: int  # global index of the window's first encoder frame
    num_frames: int


@dataclass(frozen=True)
class WindowPlan:
    windows: tuple[Window, ...]
    num_samples: int
    total_frames: int


def _align(samples: float, quantum: int) -> int:
    return max(quantum, round(samples / quantum) * quantum)


def plan_windows(
    num_samples: int,
    config: ModelConfig,
    *,
    window_s: float | None = OFFLINE_WINDOW_S,
    stride_s: float = OFFLINE_STRIDE_S,
) -> WindowPlan:
    """Plan offline windows covering ``[0, num_samples)``.

    ``window_s=None`` (or audio no longer than one window) gives a single
    full-context pass. The final window is anchored to the end of the audio so
    it never contains synthetic padding.
    """
    spf = config.samples_per_frame
    sr = config.frontend.sample_rate
    total = config.num_output_frames(num_samples)
    if window_s is None or num_samples <= window_s * sr:
        return WindowPlan((Window(0, num_samples, 0, total),), num_samples, total)
    win = _align(window_s * sr, spf)
    stride = min(_align(stride_s * sr, spf), win)
    starts = list(range(0, num_samples - win, stride))
    last = ((num_samples - win) // spf) * spf
    if starts and last <= starts[-1]:
        starts[-1] = last
    else:
        starts.append(last)
    windows = []
    for i, start in enumerate(starts):
        end = num_samples if i == len(starts) - 1 else start + win
        n = config.num_output_frames(end - start)
        windows.append(Window(start, end - start, start // spf, n))
    assert windows[-1].frame_offset + windows[-1].num_frames == total
    return WindowPlan(tuple(windows), num_samples, total)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32, copy=False)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


class PosteriorAccumulator:
    """Running Hann-weighted average of per-window posteriors on a global frame grid."""

    def __init__(self, vocab_size: int, capacity: int = 0) -> None:
        self.vocab_size = vocab_size
        self._num = np.zeros((capacity, vocab_size), np.float32)
        self._den = np.zeros((capacity,), np.float32)
        self.num_frames = 0  # frames that have received at least one observation

    def _ensure(self, n: int) -> None:
        if n <= self._num.shape[0]:
            return
        cap = max(n, int(self._num.shape[0] * 1.5) + 256)
        num = np.zeros((cap, self.vocab_size), np.float32)
        den = np.zeros((cap,), np.float32)
        num[: self._num.shape[0]] = self._num
        den[: self._den.shape[0]] = self._den
        self._num, self._den = num, den

    def add(self, frame_offset: int, log_probs: np.ndarray, *, drop_before: int = 0) -> None:
        """Add a window's ``[n, V]`` log-posteriors starting at global ``frame_offset``.

        Frames with global index ``< drop_before`` (e.g. streaming zero padding)
        are ignored, but keep their Hann weight position.
        """
        n = log_probs.shape[0]
        if n == 0:
            return
        w = hann_weights(n)
        lo = max(0, drop_before - frame_offset)
        if lo >= n:
            return
        g0, g1 = frame_offset + lo - drop_before, frame_offset + n - drop_before
        self._ensure(g1)
        self._num[g0:g1] += w[lo:, None] * np.exp(log_probs[lo:])
        self._den[g0:g1] += w[lo:]
        self.num_frames = max(self.num_frames, g1)

    def log_probs(self, start: int = 0, end: int | None = None) -> np.ndarray:
        end = self.num_frames if end is None else end
        den = np.maximum(self._den[start:end, None], 1e-12)
        return np.log(np.maximum(self._num[start:end] / den, 1e-30))


def fuse_windows(
    plan: WindowPlan,
    window_log_probs: list[np.ndarray],
    vocab_size: int,
    *,
    mode: str = "hann",
) -> np.ndarray:
    """Fuse per-window log-posteriors into ``[total_frames, V]`` log-posteriors.

    ``mode="hann"`` is the paper's weighted posterior average. ``mode="center"``
    instead takes each frame from the window where it is most central (a hard
    cut at overlap midpoints), which avoids averaging misaligned CTC spikes.
    """
    if len(plan.windows) == 1:
        return window_log_probs[0]
    if mode == "center":
        out = np.empty((plan.total_frames, vocab_size), np.float32)
        best = np.full(plan.total_frames, -1.0, np.float32)
        for win, lp in zip(plan.windows, window_log_probs):
            w = hann_weights(win.num_frames)
            sl = slice(win.frame_offset, win.frame_offset + win.num_frames)
            take = w > best[sl]
            out[sl][take] = lp[: win.num_frames][take]
            best[sl] = np.where(take, w, best[sl])
        return out
    if mode != "hann":
        raise ValueError(f"Unknown fusion mode {mode!r}")
    acc = PosteriorAccumulator(vocab_size, plan.total_frames)
    for win, lp in zip(plan.windows, window_log_probs):
        acc.add(win.frame_offset, lp[: win.num_frames])
    out = acc.log_probs(0, plan.total_frames)
    assert out.shape[0] == plan.total_frames
    return out


def run_windows(
    plan: WindowPlan,
    audio: np.ndarray,
    infer: Callable[[list[np.ndarray]], list[np.ndarray]],
    *,
    batch_size: int = 8,
) -> list[np.ndarray]:
    """Run ``infer`` (a batch of waveforms -> list of ``[n, V]`` log-probs) per window."""
    results: list[np.ndarray | None] = [None] * len(plan.windows)
    # Group same-length windows so batches need no padding.
    by_len: dict[int, list[int]] = {}
    for i, w in enumerate(plan.windows):
        by_len.setdefault(w.length, []).append(i)
    for idxs in by_len.values():
        for b in range(0, len(idxs), batch_size):
            chunk = idxs[b : b + batch_size]
            outs = infer(
                [
                    audio[plan.windows[i].start : plan.windows[i].start + plan.windows[i].length]
                    for i in chunk
                ]
            )
            for i, lp in zip(chunk, outs):
                results[i] = lp
    return [r for r in results if r is not None]
