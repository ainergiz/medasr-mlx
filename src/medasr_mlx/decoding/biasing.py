"""Context biasing ("hotwords") for CTC beam search.

Boosts phrases that matter in a given clinical context: a patient's
medication list, a radiology template's findings, rare drug brands. These are
the terms the MedASR model card calls out as weak spots.

The booster is a character-level prefix trie, so it is independent of how the
model happens to segment a word into SentencePiece pieces. A hypothesis that
is spelling out a phrase receives a bonus per matched character. If the match
breaks before the phrase is complete, the bonus is refunded, so partial
matches can survive in the beam long enough to complete but are never
rewarded on their own. Matching is case-insensitive and starts at word
boundaries.

This follows the prefix-tree "boosting with refund" scheme used by CTC/RNN-T
context-biasing work (e.g. NeMo's phrase-boosting tree, TurboBias).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# State: (node, partial). node == 0 -> at a word start (root), node == -1 -> inside a
# non-matching word, node > 0 -> trie node. ``partial`` is the unconfirmed bonus.
BiasState = tuple[int, float]


class PhraseBooster:
    def __init__(
        self,
        phrases: Iterable[str] | Mapping[str, float],
        *,
        weight: float = 5.0,
    ) -> None:
        """``weight`` is the total log-score bonus for completing a phrase.

        ``phrases`` may map phrase -> weight to override it per phrase.
        """
        items = (
            list(phrases.items())
            if isinstance(phrases, Mapping)
            else [(p, weight) for p in phrases]
        )
        self.children: list[dict[str, int]] = [{}]
        self.bonus: list[float] = [0.0]
        self.terminal: list[bool] = [False]
        self.phrases: list[str] = []
        for phrase, w in items:
            norm = " ".join(phrase.lower().split())
            if not norm:
                continue
            self.phrases.append(norm)
            per_char = float(w) / sum(1 for c in norm if c != " ")
            node = 0
            for ch in norm:
                nxt = self.children[node].get(ch)
                if nxt is None:
                    nxt = len(self.children)
                    self.children[node][ch] = nxt
                    self.children.append({})
                    self.bonus.append(0.0)
                    self.terminal.append(False)
                if ch != " ":
                    self.bonus[nxt] = max(self.bonus[nxt], per_char)
                node = nxt
            self.terminal[node] = True

    def __len__(self) -> int:
        return len(self.phrases)

    def initial_state(self) -> BiasState:
        return (0, 0.0)

    def advance(self, state: BiasState, text: str) -> tuple[BiasState, float]:
        """Feed a token's surface text (leading space = word start)."""
        node, partial = state
        delta = 0.0
        for ch in text.lower():
            if node > 0:
                nxt = self.children[node].get(ch)
                if nxt is not None:
                    node = nxt
                    partial += self.bonus[nxt]
                    delta += self.bonus[nxt]
                    continue
                # The match cannot continue with ``ch``.
                if self.terminal[node] and not ch.isalnum():
                    partial = 0.0  # phrase completed at a word/punctuation boundary
                else:
                    delta -= partial  # refund the unfinished (or overrun) match
                    partial = 0.0
                node = -1
            if ch == " ":
                node, partial = 0, 0.0
            elif node == 0:
                nxt = self.children[0].get(ch)
                if nxt is None:
                    node = -1
                else:
                    node, partial = nxt, self.bonus[nxt]
                    delta += partial
        return (node, partial), delta

    def finalize(self, state: BiasState) -> float:
        """Score adjustment at the end of the utterance (refund unfinished matches)."""
        node, partial = state
        if node > 0 and not self.terminal[node]:
            return -partial
        return 0.0
