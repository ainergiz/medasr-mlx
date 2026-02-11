#!/usr/bin/env python3
from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from scipy.signal import resample_poly

from model import ConformerCache


@dataclass
class StreamingResult:
    text: str
    latency_s: float
    audio_duration_s: float
    rtf: float
    num_chunks: int
    cache_memory_mb: float
    token_count: int


class StreamingLasrTranscriber:
    """Chunked cache-aware streaming runtime for LASR CTC encoder."""

    def __init__(
        self,
        model,
        processor,
        chunk_sec: float = 1.5,
        shift_sec: float | None = None,
        max_left_context_sec: float = 6.0,
        sample_rate: int = 16000,
        encoder_frame_sec: float = 0.04,
    ) -> None:
        if chunk_sec <= 0:
            raise ValueError(f"chunk_sec must be > 0, got {chunk_sec}")
        if shift_sec is None:
            shift_sec = chunk_sec
        if shift_sec <= 0:
            raise ValueError(f"shift_sec must be > 0, got {shift_sec}")
        if shift_sec > chunk_sec:
            raise ValueError(
                f"shift_sec ({shift_sec}) cannot exceed chunk_sec ({chunk_sec}) "
                "for cache-aware streaming."
            )

        self.model = model
        self.processor = processor
        self.chunk_sec = float(chunk_sec)
        self.shift_sec = float(shift_sec)
        self.max_left_context_sec = float(max_left_context_sec)
        self.sample_rate = int(sample_rate)
        self.encoder_frame_sec = float(encoder_frame_sec)
        self.max_left_context_tokens = max(
            1, round(self.max_left_context_sec / self.encoder_frame_sec)
        )

        self.chunk_samples = max(1, round(self.chunk_sec * self.sample_rate))
        self.shift_samples = max(1, round(self.shift_sec * self.sample_rate))
        # Conservative lower bound to keep LASR subsampler conv stack valid on short tail chunks.
        self.min_chunk_samples = 2400
        self.blank_id = 0

        self.reset()

    def reset(self) -> None:
        self.cache: ConformerCache = self.model.init_streaming_cache()
        self.prev_frame_token = self.blank_id
        self.emitted_tokens: list[int] = []
        self.total_latency_s = 0.0
        self.num_chunks = 0
        self.last_cache_memory_mb = 0.0

    def _decode_tokens(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return str(tokenizer.decode(token_ids, skip_special_tokens=True)).strip()
        arr = np.array([token_ids], dtype=np.int64)
        return str(self.processor.batch_decode(arr, skip_special_tokens=True)[0]).strip()

    def _estimate_cache_memory_mb(self) -> float:
        total_bytes = 0
        for layer_cache in self.cache.layers:
            if layer_cache.attn_key is not None:
                total_bytes += np.asarray(layer_cache.attn_key).nbytes
            if layer_cache.attn_value is not None:
                total_bytes += np.asarray(layer_cache.attn_value).nbytes
            if layer_cache.conv_left is not None:
                total_bytes += np.asarray(layer_cache.conv_left).nbytes
        return total_bytes / (1024 * 1024)

    def _collapse_ctc_with_state(self, frame_ids: np.ndarray) -> list[int]:
        new_tokens: list[int] = []
        for tok in frame_ids.tolist():
            tok = int(tok)
            if tok == self.blank_id:
                self.prev_frame_token = self.blank_id
                continue
            if tok != self.prev_frame_token:
                new_tokens.append(tok)
            self.prev_frame_token = tok
        self.emitted_tokens.extend(new_tokens)
        return new_tokens

    def process_chunk(self, audio_chunk: np.ndarray, keep_tail_fraction: float = 1.0) -> dict:
        if audio_chunk.size == 0:
            return {
                "latency_s": 0.0,
                "num_frames": 0,
                "new_tokens": [],
                "partial_text": self._decode_tokens(self.emitted_tokens),
            }
        if audio_chunk.size < self.min_chunk_samples:
            pad = self.min_chunk_samples - int(audio_chunk.size)
            audio_chunk = np.pad(audio_chunk, (0, pad), mode="constant")

        features = self.processor(
            audio_chunk.astype(np.float32),
            sampling_rate=self.sample_rate,
            return_attention_mask=True,
            return_tensors="np",
        )
        input_features = mx.array(features["input_features"])
        attention_mask = mx.array(features["attention_mask"].astype(np.bool_))

        t0 = time.perf_counter()
        logits, self.cache = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            cache=self.cache,
            is_streaming=True,
            max_left_context=self.max_left_context_tokens,
        )
        mx.eval(logits)
        elapsed = time.perf_counter() - t0
        self.total_latency_s += elapsed
        self.num_chunks += 1

        frame_ids = np.asarray(mx.argmax(logits, axis=-1))[0]
        if keep_tail_fraction < 1.0 and frame_ids.size > 0:
            keep = max(1, round(frame_ids.size * keep_tail_fraction))
            frame_ids = frame_ids[-keep:]

        new_tokens = self._collapse_ctc_with_state(frame_ids)
        partial_text = self._decode_tokens(self.emitted_tokens)
        self.last_cache_memory_mb = self._estimate_cache_memory_mb()

        return {
            "latency_s": elapsed,
            "num_frames": int(frame_ids.size),
            "new_tokens": new_tokens,
            "partial_text": partial_text,
            "cache_memory_mb": self.last_cache_memory_mb,
        }

    def transcribe_stream(self, audio_iterable: Iterable[np.ndarray]) -> StreamingResult:
        self.reset()
        total_samples = 0
        for chunk in audio_iterable:
            chunk_np = np.asarray(chunk, dtype=np.float32).reshape(-1)
            total_samples += int(chunk_np.size)
            self.process_chunk(chunk_np, keep_tail_fraction=1.0)

        text = self._decode_tokens(self.emitted_tokens)
        audio_duration_s = total_samples / float(self.sample_rate) if total_samples > 0 else 0.0
        rtf = self.total_latency_s / audio_duration_s if audio_duration_s > 0 else float("nan")
        return StreamingResult(
            text=text,
            latency_s=self.total_latency_s,
            audio_duration_s=audio_duration_s,
            rtf=rtf,
            num_chunks=self.num_chunks,
            cache_memory_mb=self.last_cache_memory_mb,
            token_count=len(self.emitted_tokens),
        )

    def transcribe_audio(self, speech: np.ndarray, sr: int) -> StreamingResult:
        if sr != self.sample_rate:
            import math

            g = math.gcd(int(sr), int(self.sample_rate))
            up = int(self.sample_rate) // g
            down = int(sr) // g
            speech = resample_poly(
                speech.astype(np.float32, copy=False),
                up,
                down,
            ).astype(np.float32, copy=False)
            sr = self.sample_rate
        if sr != self.sample_rate:
            raise ValueError(f"Unexpected sample rate after resample: {sr} != {self.sample_rate}")

        self.reset()
        total_samples = int(speech.shape[0])
        overlap_ratio = self.shift_sec / self.chunk_sec
        overlap_mode = self.shift_samples < self.chunk_samples

        start = 0
        chunk_index = 0
        while start < total_samples:
            end = min(start + self.chunk_samples, total_samples)
            chunk = speech[start:end]
            if chunk.size == 0:
                break
            keep_tail_fraction = 1.0
            if overlap_mode and chunk_index > 0:
                # Approximate boundary handling for overlap mode:
                # only keep the trailing shift-sized fraction from each chunk.
                keep_tail_fraction = overlap_ratio
            self.process_chunk(chunk, keep_tail_fraction=keep_tail_fraction)

            if end >= total_samples:
                break
            start += self.shift_samples
            chunk_index += 1

        text = self._decode_tokens(self.emitted_tokens)
        audio_duration_s = total_samples / float(self.sample_rate) if total_samples > 0 else 0.0
        rtf = self.total_latency_s / audio_duration_s if audio_duration_s > 0 else float("nan")
        return StreamingResult(
            text=text,
            latency_s=self.total_latency_s,
            audio_duration_s=audio_duration_s,
            rtf=rtf,
            num_chunks=self.num_chunks,
            cache_memory_mb=self.last_cache_memory_mb,
            token_count=len(self.emitted_tokens),
        )
