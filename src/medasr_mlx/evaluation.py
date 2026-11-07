"""ASR evaluation: text normalisation, WER/CER and medical-term metrics.

The normaliser follows the spirit of the MedASR report's scoring (Sec. 3.1):
lower-case, drop punctuation, fillers ("uh", "um", ...), bracketed tags such as
``[unintelligible]`` and ``{period}``, map number words to digits and a few
units to abbreviations. Apply the same normaliser to references and hypotheses.

Medical-term metrics take reference entity spans (character offsets, e.g. the
``medical_entities`` field of the Eka Medical ASR dataset):

* **term recall** - share of reference entities whose normalised phrase appears
  verbatim in the normalised hypothesis.
* **keyword error rate (kwER)** - substitutions + deletions on entity words,
  divided by the number of entity words, from the WER alignment.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

FILLERS = frozenset(
    [
        "uh",
        "uhh",
        "uhm",
        "um",
        "umm",
        "hmm",
        "hm",
        "mm",
        "mmm",
        "mhm",
        "ah",
        "ahh",
        "er",
        "erm",
        "eh",
        "oh",
        "huh",
    ]
)
_UNITS = {
    "milligram": "mg", "milligrams": "mg", "mgs": "mg",
    "microgram": "mcg", "micrograms": "mcg", "mcgs": "mcg",
    "gram": "g", "grams": "g", "kilogram": "kg", "kilograms": "kg", "kgs": "kg",
    "milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "millilitres": "ml", "mls": "ml",
    "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
    "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
    "%": "percent", "per cent": "percent",
}  # fmt: skip
_ONES = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
_TAG = re.compile(r"\[[^\]]*\]|\{[^}]*\}|<[^>]*>")


def _numbers_to_digits(words: list[str]) -> list[str]:
    """Convert runs of English number words to digits ("twenty five" -> "25")."""
    out: list[str] = []
    i = 0
    while i < len(words):
        if not (words[i] in _ONES and words[i] != "oh") and words[i] not in _TENS:
            out.append(words[i])
            i += 1
            continue
        total, current, j = 0, 0, i  # ``current`` is the part below one thousand
        while j < len(words):
            t = words[j]
            if t in _ONES and t != "oh":
                v = _ONES[t]
                below_100 = current % 100
                if (v < 10 and (below_100 % 10 or 10 <= below_100 < 20)) or (v >= 10 and below_100):
                    break  # "five six" / "twelve three" are separate numbers
                current += v
            elif t in _TENS:
                if current % 100:
                    break
                current += _TENS[t]
            elif t == "hundred" and 0 < current < 100:
                current *= 100
            elif t == "thousand" and 0 < current < 1000:
                total, current = total + current * 1000, 0
            elif (
                t == "and"
                and current >= 100
                and j + 1 < len(words)
                and (words[j + 1] in _ONES or words[j + 1] in _TENS)
            ):
                pass  # "one hundred and five"
            else:
                break
            j += 1
        out.append(str(total + current))
        i = j
    return out


def normalize_text(text: str, *, remove_fillers: bool = True) -> str:
    """Normalise text for WER scoring (apply to both reference and hypothesis)."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.replace("’", "'").replace("‘", "'")
    t = _TAG.sub(" ", t)
    t = t.replace("%", " percent ")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # 1,000 -> 1000
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)  # keep decimal points only
    t = re.sub(r"[^\w\s.]", " ", t)  # punctuation, hyphens, slashes -> space
    t = re.sub(r"_", " ", t)
    t = re.sub(r"(?<=\d)(mg|mcg|ml|g|kg|mm|cm|iu)\b", r" \1", t)  # 500mg -> 500 mg
    words = t.split()
    if remove_fillers:
        words = [w for w in words if w not in FILLERS]
    words = [_UNITS.get(w, w) for w in words]
    words = _numbers_to_digits(words)
    return " ".join(words)


# ------------------------------------------------------------------ alignment
@dataclass
class Alignment:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    ref_len: int = 0
    # per reference word: True if it was substituted or deleted
    ref_errors: list[bool] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def align(ref: Sequence[str], hyp: Sequence[str]) -> Alignment:
    """Levenshtein alignment with per-reference-token error flags."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = cost aligning ref[:i] with hyp[:j]
    prev = list(range(m + 1))
    back: list[bytearray] = [bytearray(m + 1)]
    for j in range(1, m + 1):
        back[0][j] = 2  # insertion
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        row = bytearray(m + 1)
        row[0] = 1  # deletion
        r = ref[i - 1]
        for j in range(1, m + 1):
            sub = prev[j - 1] + (r != hyp[j - 1])
            dele = prev[j] + 1
            ins = cur[j - 1] + 1
            if sub <= dele and sub <= ins:
                cur[j], row[j] = sub, 0
            elif dele <= ins:
                cur[j], row[j] = dele, 1
            else:
                cur[j], row[j] = ins, 2
        back.append(row)
        prev = cur
    a = Alignment(ref_len=n, ref_errors=[False] * n)
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if i > 0 and j > 0 and op == 0:
            if ref[i - 1] != hyp[j - 1]:
                a.substitutions += 1
                a.ref_errors[i - 1] = True
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or op == 1):
            a.deletions += 1
            a.ref_errors[i - 1] = True
            i -= 1
        else:
            a.insertions += 1
            j -= 1
    return a


def wer(refs: Iterable[str], hyps: Iterable[str], *, normalize: bool = True) -> float:
    errors = total = 0
    for r, h in zip(refs, hyps):
        if normalize:
            r, h = normalize_text(r), normalize_text(h)
        a = align(r.split(), h.split())
        errors += a.errors
        total += a.ref_len
    return errors / max(total, 1)


def cer(refs: Iterable[str], hyps: Iterable[str], *, normalize: bool = True) -> float:
    errors = total = 0
    for r, h in zip(refs, hyps):
        if normalize:
            r, h = normalize_text(r), normalize_text(h)
        a = align(list(r.replace(" ", "")), list(h.replace(" ", "")))
        errors += a.errors
        total += a.ref_len
    return errors / max(total, 1)


# ------------------------------------------------------------- entity metrics
def entity_word_mask(text: str, spans: Sequence[tuple[int, int]]) -> tuple[list[str], list[bool]]:
    """Normalise ``text`` piecewise, flagging words that come from entity spans."""
    spans = sorted((int(s), int(e)) for s, e in spans)
    words: list[str] = []
    mask: list[bool] = []
    pos = 0
    for s, e in spans:
        if s < pos:
            continue
        for chunk, flag in ((text[pos:s], False), (text[s:e], True)):
            w = normalize_text(chunk).split()
            words += w
            mask += [flag] * len(w)
        pos = e
    tail = normalize_text(text[pos:]).split()
    return words + tail, mask + [False] * len(tail)


def contains_phrase(hyp_words: Sequence[str], phrase: Sequence[str]) -> bool:
    k = len(phrase)
    if k == 0:
        return True
    return any(list(hyp_words[i : i + k]) == list(phrase) for i in range(len(hyp_words) - k + 1))


@dataclass
class Scores:
    num_utterances: int = 0
    ref_words: int = 0
    word_errors: int = 0
    ref_chars: int = 0
    char_errors: int = 0
    entity_words: int = 0
    entity_word_errors: int = 0
    entities: int = 0
    entities_found: int = 0

    def add(self, ref: str, hyp: str, entities: Sequence[tuple[int, int]] | None = None) -> dict:
        r, h = normalize_text(ref), normalize_text(hyp)
        rw, hw = r.split(), h.split()
        a = align(rw, hw)
        c = align(list(r.replace(" ", "")), list(h.replace(" ", "")))
        self.num_utterances += 1
        self.ref_words += a.ref_len
        self.word_errors += a.errors
        self.ref_chars += c.ref_len
        self.char_errors += c.errors
        item = {"wer": a.errors / max(a.ref_len, 1), "ref_norm": r, "hyp_norm": h}
        if entities:
            ew, emask = entity_word_mask(ref, entities)
            ea = align(ew, hw)
            n_ent = sum(emask)
            err = sum(1 for f, e in zip(emask, ea.ref_errors) if f and e)
            self.entity_words += n_ent
            self.entity_word_errors += err
            found = 0
            for s, e in entities:
                phrase = normalize_text(ref[int(s) : int(e)]).split()
                if phrase:
                    self.entities += 1
                    ok = contains_phrase(hw, phrase)
                    found += ok
                    self.entities_found += ok
            item["entities_found"] = found
        return item

    def summary(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {
            "utterances": self.num_utterances,
            "wer": round(self.word_errors / max(self.ref_words, 1), 4),
            "cer": round(self.char_errors / max(self.ref_chars, 1), 4),
            "ref_words": self.ref_words,
        }
        if self.entities:
            out["kwer"] = round(self.entity_word_errors / max(self.entity_words, 1), 4)
            out["term_recall"] = round(self.entities_found / self.entities, 4)
            out["entities"] = self.entities
        return out
