"""Token-level n-gram language models for CTC shallow fusion.

MedASR's released 6-gram LM (``lm_6.kenlm`` in ``google/medasr``) is trained on
SentencePiece *tokens* rather than words: every vocabulary piece is one LM
"word", with the word-boundary marker ``▁`` written as ``#`` (``▁the`` ->
``#the``). This module scores token ids directly with such an LM.

Two backends:

* :class:`KenLMTokenLM` - wraps a ``kenlm`` binary/ARPA model (optional
  dependency: ``pip install "medasr-mlx[lm]"``).
* :class:`ArpaTokenLM` - a dependency-free ARPA reader for small domain LMs
  (e.g. built with KenLM's ``lmplz`` from your own reports) and for tests.
"""

from __future__ import annotations

import gzip
import lzma
import math
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Protocol

from ..tokenizer import SPACE, CTCTokenizer

LN10 = math.log(10.0)


def medasr_lm_word(piece: str) -> str:
    """LM vocabulary entry for a SentencePiece piece (``▁the`` -> ``#the``)."""
    return piece.replace(SPACE, "#")


class TokenLM(Protocol):
    """Scores token-id sequences; states must be hashable and comparable."""

    def initial_state(self) -> Hashable: ...

    def score(self, state: Hashable, token_id: int) -> tuple[float, Hashable]:
        """Natural-log probability of ``token_id`` after ``state``, and the new state."""
        ...

    def final_score(self, state: Hashable) -> float:
        """Natural-log probability of the end of sentence after ``state``."""
        ...


class KenLMTokenLM:
    def __init__(self, path: str | Path, tokenizer: CTCTokenizer, *, unk_penalty: float = -10.0):
        try:
            import kenlm
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "KenLM is required for LM decoding: pip install 'medasr-mlx[lm]' "
                "(Python >= 3.13: pip install https://github.com/kpu/kenlm/archive/master.zip)"
            ) from exc
        self._kenlm = kenlm
        self.model = kenlm.Model(str(path))
        self.words = [medasr_lm_word(p) for p in tokenizer.pieces]
        self.in_vocab = [w in self.model for w in self.words]
        coverage = sum(self.in_vocab[4:]) / max(1, len(self.words) - 4)
        if coverage < 0.5:
            raise ValueError(
                f"Only {coverage:.0%} of MedASR's tokens are in {path}; this does not look like "
                "a token-level MedASR LM (expected words such as '#the')."
            )
        self.unk_penalty = unk_penalty
        self._cache: dict[tuple[Hashable, int], tuple[float, Hashable]] = {}

    def initial_state(self) -> Hashable:
        state = self._kenlm.State()
        self.model.BeginSentenceWrite(state)
        return _KenState(state)

    def score(self, state: Hashable, token_id: int) -> tuple[float, Hashable]:
        key = (state, token_id)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        assert isinstance(state, _KenState)
        out = self._kenlm.State()
        lp = self.model.BaseScore(state.state, self.words[token_id], out) * LN10
        if not self.in_vocab[token_id]:
            lp += self.unk_penalty
        result = (lp, _KenState(out))
        if len(self._cache) > 2_000_000:
            self._cache.clear()
        self._cache[key] = result
        return result

    def final_score(self, state: Hashable) -> float:
        assert isinstance(state, _KenState)
        out = self._kenlm.State()
        return self.model.BaseScore(state.state, "</s>", out) * LN10


class _KenState:
    __slots__ = ("_hash", "state")

    def __init__(self, state) -> None:
        self.state = state
        self._hash = hash(state)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _KenState) and self.state == other.state


class ArpaTokenLM:
    """Back-off n-gram LM loaded from an ARPA file (plain, .gz or .xz)."""

    def __init__(self, path: str | Path, tokenizer: CTCTokenizer, *, unk_penalty: float = -10.0):
        self.order = 0
        self.probs: dict[tuple[str, ...], tuple[float, float]] = {}
        opener = (
            gzip.open
            if str(path).endswith(".gz")
            else lzma.open
            if str(path).endswith(".xz")
            else open
        )
        section = 0
        with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
            for line in fh:
                line = line.strip()
                if not line or line.startswith("ngram ") or line == "\\data\\":
                    continue
                if line.startswith("\\") and line.endswith("-grams:"):
                    section = int(line[1:].split("-")[0])
                    self.order = max(self.order, section)
                    continue
                if line == "\\end\\":
                    break
                if section == 0:
                    continue
                parts = line.split()
                lp = float(parts[0]) * LN10
                words = tuple(parts[1 : 1 + section])
                bo = float(parts[1 + section]) * LN10 if len(parts) > 1 + section else 0.0
                self.probs[words] = (lp, bo)
        self.words = [medasr_lm_word(p) for p in tokenizer.pieces]
        self.unk = self.probs.get(("<unk>",), (unk_penalty, 0.0))[0]
        self.unk_penalty = unk_penalty

    def initial_state(self) -> Hashable:
        return ("<s>",)

    def _logprob(self, context: tuple[str, ...], word: str) -> float:
        backoff = 0.0
        for start in range(len(context) + 1):
            ctx = context[start:]
            hit = self.probs.get((*ctx, word))
            if hit is not None:
                return backoff + hit[0]
            ctx_entry = self.probs.get(ctx)
            if ctx_entry is not None:
                backoff += ctx_entry[1]
        return backoff + self.unk + self.unk_penalty

    def _advance(self, state: tuple[str, ...], word: str) -> tuple[str, ...]:
        ctx = (*state, word)[-(self.order - 1) :] if self.order > 1 else ()
        # Keep the longest suffix that exists as a context in the model.
        while ctx and ctx not in self.probs:
            ctx = ctx[1:]
        return ctx

    def score(self, state: Hashable, token_id: int) -> tuple[float, Hashable]:
        assert isinstance(state, tuple)
        word = self.words[token_id]
        return self._logprob(state, word), self._advance(state, word)

    def final_score(self, state: Hashable) -> float:
        assert isinstance(state, tuple)
        return self._logprob(state, "</s>")


def load_token_lm(path: str | Path, tokenizer: CTCTokenizer) -> TokenLM:
    """KenLM for binaries / large files when available, the ARPA reader otherwise."""
    p = str(path)
    try:
        return KenLMTokenLM(p, tokenizer)
    except ImportError:
        if p.endswith((".arpa", ".arpa.gz", ".arpa.xz")):
            return ArpaTokenLM(p, tokenizer)
        raise


def write_token_arpa(
    texts: Sequence[str], tokenizer: CTCTokenizer, path: str | Path, *, order: int = 3
) -> Path:
    """Build a tiny add-one-smoothed token ARPA LM from texts (tests/demos only)."""
    from collections import Counter

    counts: list[Counter[tuple[str, ...]]] = [Counter() for _ in range(order + 1)]
    for text in texts:
        toks = [
            "<s>",
            *(medasr_lm_word(tokenizer.pieces[i]) for i in tokenizer.encode(text)),
            "</s>",
        ]
        for n in range(1, order + 1):
            for i in range(len(toks) - n + 1):
                counts[n][tuple(toks[i : i + n])] += 1
    vocab = sorted({w for g in counts[1] for w in g} | {"<unk>"})
    lines = ["\\data\\"] + [
        f"ngram {n}={len(counts[n]) + (1 if n == 1 else 0)}" for n in range(1, order + 1)
    ]
    total = sum(counts[1].values()) + len(vocab)
    # Like lmplz: only n-grams that are extended by a higher order carry a back-off.
    extended = [set()] + [
        {g[:-1] for g in counts[n + 1]} if n < order else set() for n in range(1, order + 1)
    ]
    for n in range(1, order + 1):
        lines += ["", f"\\{n}-grams:"]
        grams = dict(counts[n])
        if n == 1:
            grams[("<unk>",)] = 0
        for gram, c in sorted(grams.items()):
            p = (c + 1) / total if n == 1 else (c + 1) / (counts[n - 1][gram[:-1]] + len(vocab))
            bo = "\t-0.30103" if gram in extended[n] else ("\t0" if n < order else "")
            lines.append(f"{math.log10(p):.6f}\t{' '.join(gram)}{bo}")
    lines += ["", "\\end\\", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return Path(path)
