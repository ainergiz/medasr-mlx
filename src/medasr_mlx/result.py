"""Transcription result types and output formats (txt / json / srt / vtt)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .tokenizer import CTCTokenizer


@dataclass
class TokenSpan:
    id: int
    text: str
    start_frame: int
    end_frame: int  # inclusive
    confidence: float


@dataclass
class Word:
    text: str
    start: float
    end: float
    confidence: float


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    text: str
    duration: float
    segments: list[Segment] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    tokens: list[TokenSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_tokens: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_tokens:
            d.pop("tokens")
        return d

    def to_json(self, *, include_tokens: bool = False, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(include_tokens=include_tokens), indent=indent, ensure_ascii=False
        )

    def to_srt(self) -> str:
        lines = []
        for i, seg in enumerate(self.segments, 1):
            lines += [str(i), f"{_ts(seg.start, ',')} --> {_ts(seg.end, ',')}", seg.text, ""]
        return "\n".join(lines)

    def to_vtt(self) -> str:
        lines = ["WEBVTT", ""]
        for seg in self.segments:
            lines += [f"{_ts(seg.start, '.')} --> {_ts(seg.end, '.')}", seg.text, ""]
        return "\n".join(lines)

    def format(self, fmt: str) -> str:
        if fmt == "txt":
            return self.text + "\n"
        if fmt == "json":
            return self.to_json() + "\n"
        if fmt == "srt":
            return self.to_srt()
        if fmt == "vtt":
            return self.to_vtt()
        raise ValueError(f"Unknown output format {fmt!r}")


def _ts(seconds: float, sep: str) -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


class TimeBase:
    """Maps encoder frame indices to seconds."""

    def __init__(self, frame_duration: float, frame_offset: float, duration: float) -> None:
        self.frame_duration = frame_duration
        # A frame's receptive field is centred ``frame_offset`` after its first sample;
        # report the 40 ms cell around that centre.
        self.offset = frame_offset - frame_duration / 2
        self.duration = duration

    def start(self, frame: int) -> float:
        return min(max(0.0, frame * self.frame_duration + self.offset), self.duration)

    def end(self, frame: int) -> float:
        return min(max(0.0, (frame + 1) * self.frame_duration + self.offset), self.duration)


def group_words(tokens: list[TokenSpan], tokenizer: CTCTokenizer, tb: TimeBase) -> list[Word]:
    words: list[Word] = []
    cur: list[TokenSpan] = []

    def flush() -> None:
        text = "".join(t.text for t in cur).strip()
        if text:
            words.append(
                Word(
                    text=text,
                    start=round(tb.start(cur[0].start_frame), 3),
                    end=round(tb.end(cur[-1].end_frame), 3),
                    confidence=round(min(t.confidence for t in cur), 4),
                )
            )

    for tok in tokens:
        if tok.id in tokenizer.special_ids:
            continue
        if cur and tokenizer.is_word_start(tok.id):
            flush()
            cur = []
        cur.append(tok)
    if cur:
        flush()
    return words


def group_segments(
    words: list[Word],
    *,
    max_gap: float = 1.0,
    max_duration: float = 20.0,
    min_duration: float = 1.0,
) -> list[Segment]:
    """Split words into subtitle-sized segments at sentence ends, pauses or length."""
    segments: list[Segment] = []
    cur: list[Word] = []

    def flush() -> None:
        if cur:
            segments.append(
                Segment(" ".join(w.text for w in cur), cur[0].start, cur[-1].end, list(cur))
            )

    for w in words:
        if cur:
            gap = w.start - cur[-1].end
            dur = cur[-1].end - cur[0].start
            ends_sentence = cur[-1].text[-1:] in ".?!" and dur >= min_duration
            if gap > max_gap or dur + (w.end - cur[-1].end) > max_duration or ends_sentence:
                flush()
                cur = []
        cur.append(w)
    flush()
    return segments
