#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import huggingface_hub
import numpy as np
from transformers import AutoProcessor


@dataclass
class DecoderConfig:
    mode: str = "greedy"  # greedy | beam
    hf_model_id: str = "google/medasr"
    kenlm_path: str | None = None
    alpha: float = 0.5
    beta: float = 1.0
    beam_width: int = 128


def resolve_kenlm_path(
    kenlm_path: str | None,
    hf_model_id: str,
) -> str:
    """Return path to KenLM file, downloading from HF repo when needed."""
    if kenlm_path:
        path = Path(kenlm_path)
        if not path.exists():
            raise FileNotFoundError(f"KenLM path not found: {path}")
        return str(path)
    return huggingface_hub.hf_hub_download(hf_model_id, filename="lm_6.kenlm")


def build_ctc_labels(processor: AutoProcessor, vocab_size: int) -> list[str]:
    """Build MedASR-compatible CTC labels for pyctcdecode."""
    tokenizer = processor.tokenizer
    vocab = [None for _ in range(vocab_size)]
    for piece, idx in tokenizer.vocab.items():
        if idx < vocab_size:
            vocab[idx] = piece
    if any(p is None for p in vocab):
        missing = [i for i, p in enumerate(vocab) if p is None][:8]
        raise ValueError(f"Missing tokenizer entries for ids: {missing}")
    # Blank id 0 must map to empty string for pyctcdecode.
    vocab[0] = ""
    # Match MedASR notebook behavior:
    # - prefix each non-special piece with ▁
    # - replace inner ▁ with # so pyctcdecode treats each token as a "word"
    for i in range(1, len(vocab)):
        piece = vocab[i]
        if not piece.startswith("<") and not piece.endswith(">"):
            piece = "▁" + piece.replace("▁", "#")
        vocab[i] = piece
    labels: list[str] = [str(p) for p in vocab]
    return labels


class CTCTextDecoder:
    """Decode CTC logits via greedy argmax or beam search + KenLM."""

    def __init__(self, processor: AutoProcessor, config: DecoderConfig):
        self.processor = processor
        self.config = config
        self._beam_decoder = None
        self._kenlm_path: str | None = None
        self._beam_decoder_vocab_size: int | None = None
        self._build_ctcdecoder = None
        if config.mode == "beam":
            from pyctcdecode import build_ctcdecoder

            self._build_ctcdecoder = build_ctcdecoder
            self._kenlm_path = resolve_kenlm_path(config.kenlm_path, config.hf_model_id)

    def _ensure_beam_decoder(self, vocab_size: int) -> None:
        if self._beam_decoder is not None and self._beam_decoder_vocab_size == vocab_size:
            return
        if self._build_ctcdecoder is None:
            raise RuntimeError("Beam decoder builder not initialized.")
        labels = build_ctc_labels(self.processor, vocab_size=vocab_size)
        self._beam_decoder = self._build_ctcdecoder(
            labels=labels,
            kenlm_model_path=self._kenlm_path,
            alpha=self.config.alpha,
            beta=self.config.beta,
        )
        self._beam_decoder_vocab_size = vocab_size

    @property
    def kenlm_path(self) -> str | None:
        return self._kenlm_path

    @staticmethod
    def _restore_text(text: str) -> str:
        # Mirror MedASR reference notebook's text restoration.
        return text.replace(" ", "").replace("#", " ").replace("</s>", "").strip()

    def decode(
        self,
        logits: np.ndarray,
        pred_ids: np.ndarray | None = None,
    ) -> str:
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError(f"Expected logits shape [1, T, V], got {tuple(logits.shape)}")
        if self.config.mode == "greedy":
            if pred_ids is None:
                pred_ids = np.argmax(logits, axis=-1)
            return self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

        if self._beam_decoder is None or self._beam_decoder_vocab_size != logits.shape[-1]:
            self._ensure_beam_decoder(logits.shape[-1])
        # pyctcdecode expects [T, V] for a single utterance.
        text = self._beam_decoder.decode(logits[0], beam_width=self.config.beam_width)
        return self._restore_text(text)
