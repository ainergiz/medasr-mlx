"""High-level transcription API."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import mlx.core as mx
import numpy as np

from .audio import as_waveform
from .config import ModelConfig
from .decoding.greedy import greedy_decode
from .features import log_mel_spectrogram
from .fusion import OFFLINE_STRIDE_S, OFFLINE_WINDOW_S, fuse_windows, plan_windows, run_windows
from .model import MedASRModel
from .result import TimeBase, TokenSpan, TranscriptionResult, group_segments, group_words
from .tokenizer import CTCTokenizer
from .weights import DEFAULT_MODEL, find_tokenizer_file, load_model

if TYPE_CHECKING:
    from .streaming import StreamingTranscriber

AudioInput = str | Path | np.ndarray


class Decoder(Protocol):
    def __call__(self, log_probs: np.ndarray, tokenizer: CTCTokenizer) -> list[TokenSpan]: ...


class GreedyDecoder:
    name = "greedy"

    def __call__(self, log_probs: np.ndarray, tokenizer: CTCTokenizer) -> list[TokenSpan]:
        return greedy_decode(log_probs, tokenizer)


class MedASR:
    """MedASR speech recognizer.

    >>> asr = MedASR.from_pretrained()            # ainergiz/medasr-mlx-fp16
    >>> result = asr.transcribe("dictation.wav")
    >>> result.text, result.words[0]
    """

    def __init__(
        self,
        model: MedASRModel,
        tokenizer: CTCTokenizer,
        *,
        name: str | None = None,
        compile: bool | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.config: ModelConfig = model.config
        # mx.compile fuses element-wise ops on Metal; on the CPU backend it measured ~2x
        # slower, and it re-traces for every new shape, so by default it is used only on
        # GPU and only for fixed-shape batches (full TPF windows, steady-state streaming).
        if compile is None:
            compile = mx.default_device() == mx.gpu
        self._compiled = mx.compile(self._forward_unmasked) if compile else None

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path = DEFAULT_MODEL,
        *,
        dtype: mx.Dtype | str | None = None,
        revision: str | None = None,
        compile: bool | None = None,
    ) -> MedASR:
        if isinstance(dtype, str):
            dtype = {"float32": mx.float32, "fp32": mx.float32, "float16": mx.float16,
                     "fp16": mx.float16, "bfloat16": mx.bfloat16, "bf16": mx.bfloat16}[dtype]  # fmt: skip
        model, model_dir = load_model(path_or_repo, dtype=dtype, revision=revision)
        tokenizer = CTCTokenizer.from_file(find_tokenizer_file(model_dir))
        return cls(model, tokenizer, name=str(path_or_repo), compile=compile)

    # ----------------------------------------------------------------- forward
    def _forward_unmasked(self, features: mx.array) -> mx.array:
        logits, _ = self.model(features)
        logits = logits.astype(mx.float32)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def _pad_short(self, wave: np.ndarray) -> np.ndarray:
        need = self.config.min_samples
        return wave if wave.shape[0] >= need else np.pad(wave, (0, need - wave.shape[0]))

    def forward(
        self, waves: Sequence[np.ndarray], *, fixed_shape: bool = False
    ) -> list[np.ndarray]:
        """Log-posteriors ``[frames_i, V]`` for a batch of 16 kHz waveforms.

        ``fixed_shape`` marks calls whose shape recurs (TPF windows), which are
        worth running through the compiled graph.
        """
        waves = [self._pad_short(np.asarray(w, np.float32)) for w in waves]
        feats = [log_mel_spectrogram(w, self.config.frontend) for w in waves]
        lengths = [f.shape[0] for f in feats]
        if len(set(lengths)) == 1:
            fn = self._compiled if (fixed_shape and self._compiled) else self._forward_unmasked
            out = np.asarray(fn(mx.array(np.stack(feats))))
            return [out[i] for i in range(len(waves))]
        max_len = max(lengths)
        batch = np.zeros((len(feats), max_len, feats[0].shape[1]), np.float32)
        mask = np.zeros((len(feats), max_len), bool)
        for i, f in enumerate(feats):
            batch[i, : f.shape[0]] = f
            mask[i, : f.shape[0]] = True
        logits, _frame_mask = self.model(mx.array(batch), mx.array(mask))
        logits = logits.astype(mx.float32)
        lp = np.asarray(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
        n_out = [self.config.encoder.num_output_frames(n) for n in lengths]
        return [lp[i, : n_out[i]] for i in range(len(waves))]

    def log_probs(
        self,
        audio: AudioInput,
        *,
        sample_rate: int | None = None,
        window_s: float | None = OFFLINE_WINDOW_S,
        stride_s: float = OFFLINE_STRIDE_S,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Fused CTC log-posteriors ``[frames, V]`` for one recording of any length."""
        wave = self._pad_short(as_waveform(audio, sample_rate, self.config.frontend.sample_rate))
        plan = plan_windows(wave.shape[0], self.config, window_s=window_s, stride_s=stride_s)
        multi = len(plan.windows) > 1
        outs = run_windows(
            plan, wave, lambda ws: self.forward(ws, fixed_shape=multi), batch_size=batch_size
        )
        return fuse_windows(plan, outs, self.config.vocab_size)

    # -------------------------------------------------------------- transcribe
    def make_result(
        self,
        tokens: list[TokenSpan],
        duration: float,
        metadata: dict[str, Any] | None = None,
    ) -> TranscriptionResult:
        tb = TimeBase(self.config.frame_duration, self.config.frame_offset, duration)
        words = group_words(tokens, self.tokenizer, tb)
        return TranscriptionResult(
            text=self.tokenizer.decode([t.id for t in tokens]),
            duration=round(duration, 3),
            segments=group_segments(words),
            words=words,
            tokens=tokens,
            metadata=metadata or {},
        )

    def transcribe(
        self,
        audio: AudioInput,
        *,
        sample_rate: int | None = None,
        decoder: Decoder | str | None = None,
        window_s: float | None = OFFLINE_WINDOW_S,
        stride_s: float = OFFLINE_STRIDE_S,
        batch_size: int = 8,
        **decoder_kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe a file path or waveform (16 kHz unless ``sample_rate`` is given).

        Long recordings are handled by Temporal Posterior Fusion over 20 s windows
        (18 s stride). Pass ``window_s=None`` to force a single full-context pass.
        """
        t0 = time.perf_counter()
        wave = as_waveform(audio, sample_rate, self.config.frontend.sample_rate)
        duration = wave.shape[0] / self.config.frontend.sample_rate
        t1 = time.perf_counter()
        lp = self.log_probs(wave, window_s=window_s, stride_s=stride_s, batch_size=batch_size)
        t2 = time.perf_counter()
        dec = self.resolve_decoder(decoder, **decoder_kwargs)
        tokens = dec(lp, self.tokenizer)
        t3 = time.perf_counter()
        meta = {
            "model": self.name,
            "dtype": str(self.model.dtype).replace("mlx.core.", ""),
            "decoder": getattr(dec, "name", type(dec).__name__),
            "window_s": window_s,
            "stride_s": stride_s,
            "frames": int(lp.shape[0]),
            "timing_s": {
                "load_audio": round(t1 - t0, 4),
                "acoustic_model": round(t2 - t1, 4),
                "decode": round(t3 - t2, 4),
            },
            "rtf": round((t3 - t1) / duration, 5) if duration > 0 else None,
        }
        return self.make_result(tokens, duration, meta)

    def transcribe_batch(
        self,
        audios: Sequence[AudioInput],
        *,
        sample_rate: int | None = None,
        decoder: Decoder | str | None = None,
        batch_size: int = 8,
        **decoder_kwargs: Any,
    ) -> list[TranscriptionResult]:
        """Transcribe many recordings; clips up to 20 s are batched with padding masks."""
        sr = self.config.frontend.sample_rate
        waves = [as_waveform(a, sample_rate, sr) for a in audios]
        dec = self.resolve_decoder(decoder, **decoder_kwargs)
        results: list[TranscriptionResult | None] = [None] * len(waves)
        short = [i for i, w in enumerate(waves) if w.shape[0] <= OFFLINE_WINDOW_S * sr]
        # Sort by length so padded batches waste little compute.
        short.sort(key=lambda i: waves[i].shape[0])
        for b in range(0, len(short), batch_size):
            idx = short[b : b + batch_size]
            for i, lp in zip(idx, self.forward([waves[i] for i in idx])):
                results[i] = self.make_result(dec(lp, self.tokenizer), waves[i].shape[0] / sr)
        for i, w in enumerate(waves):
            if results[i] is None:
                results[i] = self.transcribe(w, decoder=dec, batch_size=batch_size)
        return [r for r in results if r is not None]

    def resolve_decoder(self, decoder: Decoder | str | None, **kwargs: Any) -> Decoder:
        if decoder is None and kwargs:
            decoder = "beam"  # hotwords / lm / beam_width imply beam search
        if decoder is None or decoder == "greedy":
            if kwargs:
                raise TypeError(f"Greedy decoding takes no options, got {sorted(kwargs)}")
            return GreedyDecoder()
        if decoder == "beam":
            from .decoding.beam import BeamSearchDecoder

            return BeamSearchDecoder(**kwargs)
        if callable(decoder):
            return decoder
        raise ValueError(f"Unknown decoder {decoder!r}")

    def stream(self, **kwargs: Any) -> StreamingTranscriber:
        """Create a :class:`~medasr_mlx.streaming.StreamingTranscriber` for live audio."""
        from .streaming import StreamingTranscriber

        return StreamingTranscriber(self, **kwargs)
