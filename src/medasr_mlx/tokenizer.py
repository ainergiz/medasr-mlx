"""Pure-Python SentencePiece (Unigram) tokenizer for MedASR's 512-piece vocabulary.

Reads the ``tokenizer.json`` shipped with ``google/medasr`` and reproduces the
behaviour of ``LasrTokenizer.batch_decode(ids, skip_special_tokens=True)``:

* CTC collapse (merge repeats, then drop the blank ``<epsilon>``),
* special tokens are skipped,
* Metaspace decoding: ``▁`` becomes a space, except in the very first token
  where it is dropped (``prepend_scheme="always"``).

``encode`` implements Unigram Viterbi segmentation (used for hotword phrases
and LM utilities); it matches the reference for in-vocabulary text.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

SPACE = "▁"  # ▁


class CTCTokenizer:
    def __init__(
        self,
        pieces: Sequence[str],
        scores: Sequence[float] | None = None,
        *,
        blank_id: int = 0,
        unk_id: int = 3,
        special_ids: Iterable[int] = (),
    ) -> None:
        self.pieces: list[str] = list(pieces)
        self.scores: list[float] = list(scores) if scores is not None else [0.0] * len(pieces)
        self.blank_id = int(blank_id)
        self.unk_id = int(unk_id)
        self.special_ids: frozenset[int] = frozenset(int(i) for i in special_ids) | {self.blank_id}
        self._piece_to_id = {p: i for i, p in enumerate(self.pieces) if i not in self.special_ids}
        self._max_piece_len = max((len(p) for p in self._piece_to_id), default=1)
        min_score = min(
            (s for i, s in enumerate(self.scores) if i not in self.special_ids), default=0
        )
        self._unk_score = min_score - 10.0
        # Text each id contributes when it is *not* the first emitted token.
        self.token_text: list[str] = [
            "" if i in self.special_ids else p.replace(SPACE, " ")
            for i, p in enumerate(self.pieces)
        ]

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_file(cls, path: str | Path) -> CTCTokenizer:
        path = Path(path)
        if path.is_dir():
            for candidate in (path / "tokenizer.json", path / "processor" / "tokenizer.json"):
                if candidate.exists():
                    path = candidate
                    break
            else:
                raise FileNotFoundError(f"No tokenizer.json under {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data["model"]
        if model.get("type") != "Unigram":
            raise ValueError(f"Expected a Unigram tokenizer, got {model.get('type')!r}")
        vocab = model["vocab"]
        pieces = [p for p, _ in vocab]
        scores = [float(s) for _, s in vocab]
        special = {int(t["id"]) for t in data.get("added_tokens", []) if t.get("special")}
        special = {i for i in special if i < len(pieces)}
        blank_id = pieces.index("<epsilon>") if "<epsilon>" in pieces else 0
        unk_id = int(model.get("unk_id", 3) if model.get("unk_id") is not None else 3)
        return cls(pieces, scores, blank_id=blank_id, unk_id=unk_id, special_ids=special)

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    # ----------------------------------------------------------------- decoding
    def ctc_collapse(self, ids: Iterable[int]) -> list[int]:
        out: list[int] = []
        prev = None
        for tok in ids:
            tok = int(tok)
            if tok != prev and tok != self.blank_id:
                out.append(tok)
            prev = tok
        return out

    def decode(self, ids: Iterable[int], *, collapse: bool = False) -> str:
        """Decode token ids to text (``collapse=True`` for raw CTC frame ids)."""
        seq = self.ctc_collapse(ids) if collapse else [int(i) for i in ids]
        parts: list[str] = []
        for tok in seq:
            if tok in self.special_ids or tok < 0 or tok >= len(self.pieces):
                continue
            piece = self.pieces[tok]
            parts.append(piece.replace(SPACE, "") if not parts else piece.replace(SPACE, " "))
        return "".join(parts)

    def is_word_start(self, token_id: int) -> bool:
        return self.pieces[token_id].startswith(SPACE)

    # ----------------------------------------------------------------- encoding
    def encode(self, text: str) -> list[int]:
        """Unigram Viterbi segmentation (WhitespaceSplit + Metaspace pre-tokenizer)."""
        ids: list[int] = []
        for word in text.split():
            ids.extend(self._encode_word(SPACE + word))
        return ids

    @lru_cache(maxsize=4096)  # noqa: B019 - tokenizers are long-lived
    def _encode_word(self, word: str) -> tuple[int, ...]:
        n = len(word)
        best = [float("-inf")] * (n + 1)
        back: list[tuple[int, int]] = [(-1, -1)] * (n + 1)
        best[0] = 0.0
        for end in range(1, n + 1):
            for start in range(max(0, end - self._max_piece_len), end):
                if best[start] == float("-inf"):
                    continue
                tok = self._piece_to_id.get(word[start:end])
                if tok is None:
                    continue
                score = best[start] + self.scores[tok]
                if score > best[end]:
                    best[end], back[end] = score, (start, tok)
            if best[end] == float("-inf") and best[end - 1] != float("-inf"):
                # Unknown character: consume it as <unk> (SentencePiece behaviour).
                best[end], back[end] = best[end - 1] + self._unk_score, (end - 1, self.unk_id)
        out: list[int] = []
        pos = n
        while pos > 0:
            start, tok = back[pos]
            if out and tok == self.unk_id and out[-1] == self.unk_id:
                pos = start  # merge consecutive unknown characters into one <unk>
                continue
            out.append(tok)
            pos = start
        return tuple(reversed(out))
