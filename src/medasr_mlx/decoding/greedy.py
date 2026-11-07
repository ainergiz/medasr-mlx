"""Greedy (best-path) CTC decoding with frame alignments."""

from __future__ import annotations

import numpy as np

from ..result import TokenSpan
from ..tokenizer import CTCTokenizer


def greedy_decode(
    log_probs: np.ndarray,
    tokenizer: CTCTokenizer,
    *,
    frame_offset: int = 0,
) -> list[TokenSpan]:
    """Best-path decode ``[T, V]`` log-posteriors into token spans.

    A token's span is its run of consecutive identical argmax frames; its
    confidence is the peak posterior over that run.
    """
    if log_probs.shape[0] == 0:
        return []
    ids = log_probs.argmax(axis=-1)
    blank = tokenizer.blank_id
    # Run boundaries of the argmax path.
    change = np.flatnonzero(np.diff(ids)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [ids.shape[0]]]) - 1
    spans: list[TokenSpan] = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        tok = int(ids[s])
        if tok == blank:
            continue
        conf = float(np.exp(log_probs[s : e + 1, tok].max()))
        spans.append(
            TokenSpan(tok, tokenizer.token_text[tok], s + frame_offset, e + frame_offset, conf)
        )
    return spans
