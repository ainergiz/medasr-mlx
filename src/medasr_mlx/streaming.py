"""Streaming recognition with Temporal Posterior Fusion (MedASR paper, Sec. 3.4).

The model itself is unchanged. Every ``stride`` seconds the most recent
``window`` seconds of audio are recognised, and the window's posteriors are
Hann-fused into a running per-frame average. By default windows simply grow
until they are full. ``start_padding=True`` instead left-pads the stream with
``window`` seconds of digital silence so the first window ends at time zero
(the report's recipe); on our evaluation that padding makes the model emit
spurious ``]`` tokens at speech onset, so it is off by default.

Each update reports

* ``committed`` text - frames with at least ``commit_delay`` seconds of right
  context, decoded once and never revised, and
* ``tentative`` text - the newer frames, re-decoded on every update.

:meth:`StreamingTranscriber.finish` flushes the stream and re-decodes the fully
fused posteriors (a second pass), so the final transcript does not depend on
what was committed early.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .decoding.greedy import greedy_decode
from .fusion import PosteriorAccumulator
from .result import TokenSpan, TranscriptionResult

if TYPE_CHECKING:
    from .transcribe import Decoder, MedASR


@dataclass(frozen=True)
class StreamingPreset:
    window_s: float
    stride_s: float
    commit_delay_s: float


PRESETS = {
    # The report's streaming setting: 20 s windows every 320 ms.
    "paper": StreamingPreset(20.0, 0.32, 1.5),
    "balanced": StreamingPreset(10.0, 0.64, 1.5),
    "low-power": StreamingPreset(6.0, 1.28, 1.5),
}


@dataclass
class StreamingUpdate:
    text: str  # committed + tentative
    committed_text: str
    tentative_text: str
    new_committed_text: str
    audio_seconds: float
    committed_seconds: float
    compute_seconds: float  # model time spent in this update


class StreamingTranscriber:
    def __init__(
        self,
        asr: MedASR,
        *,
        preset: str | None = "balanced",
        window_s: float | None = None,
        stride_s: float | None = None,
        commit_delay_s: float | None = None,
        start_padding: bool = False,
    ) -> None:
        base = PRESETS[preset] if preset else PRESETS["balanced"]
        cfg = asr.config
        self.asr = asr
        self.sr = cfg.frontend.sample_rate
        self.spf = cfg.samples_per_frame
        window_s = base.window_s if window_s is None else window_s
        stride_s = base.stride_s if stride_s is None else stride_s
        self.commit_delay_s = base.commit_delay_s if commit_delay_s is None else commit_delay_s
        self.window = max(self.spf, round(window_s * self.sr / self.spf) * self.spf)
        self.stride = max(self.spf, round(stride_s * self.sr / self.spf) * self.spf)
        if self.stride > self.window:
            raise ValueError("stride_s must not exceed window_s")
        self.window = max(self.window, cfg.min_samples + self.spf)
        self.start_padding = start_padding
        self.pad = self.window if start_padding else 0
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._audio = np.zeros(self.pad, np.float32)  # padded-coordinate buffer
        self._buf_start = 0  # padded sample index of _audio[0]
        self.num_samples = 0  # real samples received
        self._next_end = self.stride  # next window end (real samples)
        self._last_end = 0
        self.acc = PosteriorAccumulator(self.asr.config.vocab_size)
        self.committed: list[TokenSpan] = []
        self.commit_frame = 0
        self._prev_frame_id = self.asr.tokenizer.blank_id
        self._committed_text = ""
        self.windows_run = 0
        self.compute_seconds = 0.0

    @property
    def audio_seconds(self) -> float:
        return self.num_samples / self.sr

    # --------------------------------------------------------------- internals
    def _run_window(self, end: int, final: bool = False) -> None:
        """Recognise the window ending at real sample ``end`` (padded coords: end + pad)."""
        pend = end + self.pad
        start = max(0, ((pend - self.window) // self.spf) * self.spf)
        if pend - start < self.asr.config.min_samples and not final:
            return  # growing window still shorter than the model's receptive field
        seg = self._audio[start - self._buf_start : pend - self._buf_start]
        t0 = time.perf_counter()
        lp = self.asr.forward([seg], fixed_shape=seg.shape[0] == self.window)[0]
        self.compute_seconds += time.perf_counter() - t0
        self.acc.add(start // self.spf, lp, drop_before=self.pad // self.spf)
        self.windows_run += 1
        self._last_end = end
        # Later windows (including the final flush) end at or after ``end``, so
        # they start no earlier than this window did.
        keep_from = start
        if keep_from > self._buf_start:
            self._audio = self._audio[keep_from - self._buf_start :]
            self._buf_start = keep_from

    def _commit_upto(self) -> int:
        cfg = self.asr.config
        horizon = self._last_end / self.sr - self.commit_delay_s - cfg.frame_offset
        n = int(np.floor(horizon / cfg.frame_duration)) + 1
        return int(np.clip(n, self.commit_frame, self.acc.num_frames))

    def _decode_range(
        self, start: int, end: int, prev_id: int, *, commit: bool
    ) -> tuple[list[TokenSpan], int]:
        """Greedy decode frames ``[start, end)`` continuing the CTC state ``prev_id``."""
        if end <= start:
            return [], prev_id
        lp = self.acc.log_probs(start, end)
        spans = greedy_decode(lp, self.asr.tokenizer, frame_offset=start)
        ids = lp.argmax(-1)
        if spans and spans[0].start_frame == start and int(ids[0]) == prev_id:
            # The first run continues a token emitted before ``start``.
            first = spans.pop(0)
            if commit and self.committed and self.committed[-1].id == first.id:
                last = self.committed[-1]
                last.end_frame = first.end_frame
                last.confidence = max(last.confidence, first.confidence)
        return spans, int(ids[-1])

    def _update(self, compute_before: float, *, flush: bool = False) -> StreamingUpdate:
        tok = self.asr.tokenizer
        upto = self.acc.num_frames if flush else self._commit_upto()
        new_spans, self._prev_frame_id = self._decode_range(
            self.commit_frame, upto, self._prev_frame_id, commit=True
        )
        self.committed.extend(new_spans)
        self.commit_frame = upto
        tentative, _ = self._decode_range(
            upto, self.acc.num_frames, self._prev_frame_id, commit=False
        )
        committed_text = tok.decode([t.id for t in self.committed])
        full = tok.decode([t.id for t in self.committed + tentative])
        new = committed_text[len(self._committed_text) :]
        self._committed_text = committed_text
        return StreamingUpdate(
            text=full,
            committed_text=committed_text,
            tentative_text=full[len(committed_text) :],
            new_committed_text=new,
            audio_seconds=self.audio_seconds,
            committed_seconds=upto * self.asr.config.frame_duration,
            compute_seconds=self.compute_seconds - compute_before,
        )

    # ------------------------------------------------------------------ public
    def push(self, audio: np.ndarray) -> StreamingUpdate:
        """Add 16 kHz mono samples; runs every window that became complete."""
        before = self.compute_seconds
        chunk = np.asarray(audio, np.float32).reshape(-1)
        self._audio = np.concatenate([self._audio, chunk])
        self.num_samples += chunk.shape[0]
        while self._next_end <= self.num_samples:
            self._run_window(self._next_end)
            self._next_end += self.stride
        return self._update(before)

    def finish(
        self, decoder: Decoder | str | None = None, **decoder_kwargs: Any
    ) -> TranscriptionResult:
        """Flush the stream and return the final (second-pass) transcription."""
        before = self.compute_seconds
        if self.num_samples == 0:
            return self.asr.make_result([], 0.0, {"model": self.asr.name, "mode": "streaming"})
        if self.num_samples > self._last_end or self.windows_run == 0:
            self._run_window(self.num_samples, final=True)
        update = self._update(before, flush=True)  # end of stream: commit everything
        lp = self.acc.log_probs(0, self.acc.num_frames)
        dec = self.asr.resolve_decoder(decoder, **decoder_kwargs)
        tokens = dec(lp, self.asr.tokenizer)
        result = self.asr.make_result(
            tokens,
            self.audio_seconds,
            {
                "model": self.asr.name,
                "mode": "streaming",
                "window_s": self.window / self.sr,
                "stride_s": self.stride / self.sr,
                "commit_delay_s": self.commit_delay_s,
                "start_padding": self.start_padding,
                "windows_run": self.windows_run,
                "compute_s": round(self.compute_seconds, 4),
                "rtf": round(self.compute_seconds / max(self.audio_seconds, 1e-9), 5),
                "streaming_committed_text": update.committed_text,
            },
        )
        return result

    def transcribe(
        self, audio: np.ndarray, chunk_s: float = 0.32
    ) -> tuple[TranscriptionResult, list[StreamingUpdate]]:
        """Simulate live streaming over a whole waveform (for evaluation and demos)."""
        self.reset()
        step = max(1, round(chunk_s * self.sr))
        updates = [self.push(audio[i : i + step]) for i in range(0, audio.shape[0], step)]
        return self.finish(), updates
