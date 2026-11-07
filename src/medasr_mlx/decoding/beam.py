"""CTC prefix beam search with token-LM shallow fusion and context biasing.

Pure Python + NumPy, replacing v1's dependency on ``pyctcdecode`` (unmaintained
since 2023, pins ``numpy<2``). Two optimisations keep it fast on long audio:

* **blank-frame skipping** - frames whose blank posterior exceeds
  ``blank_skip`` (most CTC frames) only update existing beams in O(beam);
* **O(1) prefix keys** - prefixes are identified by a rolling hash and stored
  as linked lists, so merging and extension never copy token sequences.

Scores are natural-log: ``log P_ctc + alpha * log P_lm + beta * |tokens| + bias``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..result import TokenSpan
from ..tokenizer import CTCTokenizer
from .biasing import BiasState, PhraseBooster
from .lm import TokenLM, load_token_lm

NEG_INF = float("-inf")


def _lse(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class _Hyp:
    __slots__ = (
        "bias_score",
        "bias_state",
        "key",
        "last",
        "lm_score",
        "lm_state",
        "n",
        "node",
        "p_b",
        "p_nb",
    )

    def __init__(self, key, last, node, n, lm_state, lm_score, bias_state, bias_score):
        self.key = key
        self.p_b = NEG_INF
        self.p_nb = NEG_INF
        self.last = last
        self.node = node  # (token, frame, parent_node) or None
        self.n = n
        self.lm_state = lm_state
        self.lm_score = lm_score
        self.bias_state = bias_state
        self.bias_score = bias_score

    def same_prefix(self) -> _Hyp:
        return _Hyp(self.key, self.last, self.node, self.n, self.lm_state, self.lm_score,
                    self.bias_state, self.bias_score)  # fmt: skip


@dataclass
class BeamSearchDecoder:
    """Prefix beam search over ``[T, V]`` CTC log-posteriors.

    Args:
        beam_width: hypotheses kept per frame (the MedASR report uses 8 with its LM).
        lm: a :class:`TokenLM`, or a path to a token-level KenLM/ARPA model such
            as ``lm_6.kenlm`` from ``google/medasr``.
        alpha: LM weight. beta: per-token insertion bonus (only used with an LM).
        hotwords: phrases to boost (list, or mapping phrase -> weight).
        hotword_weight: total log-score bonus for completing a hotword.
        token_prune: ignore tokens whose frame log-prob is below this.
        max_candidates: at most this many non-blank tokens are tried per frame.
        blank_skip: frames with P(blank) above this skip beam extension.
    """

    beam_width: int = 8
    lm: TokenLM | str | Path | None = None
    alpha: float = 0.5
    beta: float = 1.0
    hotwords: Sequence[str] | Mapping[str, float] | None = None
    hotword_weight: float = 5.0
    token_prune: float = -12.0
    max_candidates: int = 16
    blank_skip: float = 0.999
    name: str = "beam"

    def __post_init__(self) -> None:
        self._lm_obj: TokenLM | None = None if isinstance(self.lm, (str, Path)) else self.lm
        self._booster = (
            PhraseBooster(self.hotwords, weight=self.hotword_weight) if self.hotwords else None
        )

    def _get_lm(self, tokenizer: CTCTokenizer) -> TokenLM | None:
        if self._lm_obj is None and self.lm is not None:
            self._lm_obj = load_token_lm(self.lm, tokenizer)
        return self._lm_obj

    def _score(self, h: _Hyp, use_lm: bool) -> float:
        s = _lse(h.p_b, h.p_nb) + h.lm_score + h.bias_score
        return s + self.beta * h.n if use_lm else s

    def __call__(self, log_probs: np.ndarray, tokenizer: CTCTokenizer) -> list[TokenSpan]:
        return self.decode(log_probs, tokenizer)[0]

    def decode(
        self, log_probs: np.ndarray, tokenizer: CTCTokenizer, *, nbest: int = 1
    ) -> list[list[TokenSpan]]:
        lm = self._get_lm(tokenizer)
        booster = self._booster
        use_lm = lm is not None
        blank = tokenizer.blank_id
        special = tokenizer.special_ids
        texts = tokenizer.token_text
        log_skip = math.log(self.blank_skip) if self.blank_skip < 1 else 1.0
        lp_all = np.asarray(log_probs, dtype=np.float32)

        root = _Hyp(0, -1, None, 0, lm.initial_state() if lm else None, 0.0,
                    booster.initial_state() if booster else None, 0.0)  # fmt: skip
        root.p_b = 0.0
        beams: list[_Hyp] = [root]

        for t in range(lp_all.shape[0]):
            row = lp_all[t]
            lp_blank = float(row[blank])
            if lp_blank > log_skip:
                for h in beams:
                    p_nb = h.p_nb + float(row[h.last]) if h.last >= 0 else NEG_INF
                    h.p_b = _lse(h.p_b, h.p_nb) + lp_blank
                    h.p_nb = p_nb
                continue
            cand = np.flatnonzero(row > self.token_prune)
            if cand.size > self.max_candidates + 1:
                cand = cand[
                    np.argpartition(row[cand], -(self.max_candidates + 1))[
                        -(self.max_candidates + 1) :
                    ]
                ]
            cands = [(int(c), float(row[c])) for c in cand if c != blank]
            nxt: dict[int, _Hyp] = {}
            for h in beams:
                total = _lse(h.p_b, h.p_nb)
                same = nxt.get(h.key)
                if same is None:
                    same = nxt[h.key] = h.same_prefix()
                same.p_b = _lse(same.p_b, total + lp_blank)
                if h.last >= 0:
                    same.p_nb = _lse(same.p_nb, h.p_nb + float(row[h.last]))
                for c, lpc in cands:
                    p = (h.p_b if c == h.last else total) + lpc
                    if p == NEG_INF:
                        continue
                    key = hash((h.key, c))
                    child = nxt.get(key)
                    if child is None:
                        lm_state, lm_score = h.lm_state, h.lm_score
                        if lm is not None and c not in special:
                            lp_lm, lm_state = lm.score(h.lm_state, c)
                            lm_score += self.alpha * lp_lm
                        bias_state: BiasState | None = h.bias_state
                        bias_score = h.bias_score
                        if booster is not None and bias_state is not None:
                            bias_state, delta = booster.advance(bias_state, texts[c])
                            bias_score += delta
                        child = _Hyp(key, c, (c, t, lpc, h.node), h.n + 1, lm_state, lm_score,
                                     bias_state, bias_score)  # fmt: skip
                        nxt[key] = child
                    child.p_nb = _lse(child.p_nb, p)
            beams = sorted(nxt.values(), key=lambda x: self._score(x, use_lm), reverse=True)[
                : self.beam_width
            ]

        def final(h: _Hyp) -> float:
            s = self._score(h, use_lm)
            if lm is not None:
                s += self.alpha * lm.final_score(h.lm_state)
            if booster is not None and h.bias_state is not None:
                s += booster.finalize(h.bias_state)
            return s

        ranked = sorted(beams, key=final, reverse=True)[:nbest]
        return [self._spans(h, texts) for h in ranked]

    @staticmethod
    def _spans(h: _Hyp, texts: list[str]) -> list[TokenSpan]:
        out: list[TokenSpan] = []
        node = h.node
        while node is not None:
            tok, frame, lp, node = node
            out.append(TokenSpan(tok, texts[tok], frame, frame, float(math.exp(lp))))
        out.reverse()
        return out
