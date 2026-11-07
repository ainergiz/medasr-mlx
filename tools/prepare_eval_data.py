#!/usr/bin/env python3
"""Build evaluation manifests from public datasets (no credentials needed).

Writes 16 kHz WAV files and JSONL manifests consumable by ``medasr eval``:

* ``eka-short.jsonl`` - Eka Medical ASR Evaluation Dataset (English, MIT
  license; drug brands, clinical findings, conversations) with
  ``medical_entities`` spans for medical-term metrics. A stratified subset by
  default (``--eka-limit``), or everything with ``--eka-limit 0``.
* ``eka-longform.jsonl`` - Eka conversation segments concatenated per session
  (28-130 s each) for long-form / streaming evaluation.
* ``librispeech-longform.jsonl`` - contiguous LibriSpeech dev-clean readings
  (``hf-internal-testing/librispeech_asr_dummy``) concatenated per chapter.

Usage::

    uv run --extra eval python tools/prepare_eval_data.py --out eval-data
    uv run medasr eval eval-data/eka-short.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

EKA = "ekacare/eka-medical-asr-evaluation-dataset"
LIBRI = "hf-internal-testing/librispeech_asr_dummy"
SR = 16_000


def _download(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


def _decode(audio_bytes: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        from medasr_mlx.audio import resample

        audio = resample(audio, sr, SR)
    return audio.astype(np.float32)


def _entities(raw: str | None) -> list[list[int]]:
    spans: list[list[int]] = []
    for item in json.loads(raw or "[]"):
        for s, e in item[2]:
            spans.append([int(s), int(e)])
    return spans


def _join(texts: list[str], spans: list[list[list[int]]]) -> tuple[str, list[list[int]]]:
    out, out_spans, offset = [], [], 0
    for t, sp in zip(texts, spans):
        t = t.strip()
        out_spans += [[s + offset, e + offset] for s, e in sp]
        out.append(t)
        offset += len(t) + 1
    return " ".join(out), out_spans


def prepare_eka(out: Path, limit: int, seed: int) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import list_repo_files

    files = sorted(f for f in list_repo_files(EKA, repo_type="dataset") if f.startswith("en/"))
    rows = []
    for f in files:
        rows += pq.read_table(_download(EKA, f)).to_pylist()
    print(f"eka: {len(rows)} rows")
    (out / "eka").mkdir(parents=True, exist_ok=True)

    # Short-form subset, stratified by recording context.
    rng = random.Random(seed)
    by_ctx: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ctx[r["recording_context"]].append(r)
    picked: list[dict] = []
    for ctx, items in sorted(by_ctx.items()):
        items = sorted(items, key=lambda r: r["md5_audio"])
        rng.shuffle(items)
        n = len(items) if limit <= 0 or ctx == "conversation" else min(len(items), limit)
        picked += items[:n]
    with (out / "eka-short.jsonl").open("w") as fh:
        for r in picked:
            uid = r["md5_audio"][:12]
            path = out / "eka" / f"{uid}.wav"
            if not path.exists():
                sf.write(path, _decode(r["audio"]), SR)
            rec = {"id": uid, "audio": f"eka/{uid}.wav", "text": r["text"].strip(),
                   "entities": _entities(r["medical_entities"]),
                   "context": r["recording_context"], "type": r["type_concept"]}  # fmt: skip
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"eka-short: {len(picked)} clips")

    # Long-form: concatenate a session's conversation segments in order.
    sessions: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for r in by_ctx.get("conversation", []):
        m = re.search(r"/([^/]+)/(\d+)\.\w+$", r["file_name"])
        if m:
            sessions[m.group(1)].append((int(m.group(2)), r))
    n_long = 0
    with (out / "eka-longform.jsonl").open("w") as fh:
        for sid, segs in sorted(sessions.items()):
            segs.sort(key=lambda x: x[0])
            audio = np.concatenate([_decode(r["audio"]) for _, r in segs])
            if audio.shape[0] < 25 * SR:
                continue
            text, spans = _join(
                [r["text"] for _, r in segs], [_entities(r["medical_entities"]) for _, r in segs]
            )
            path = out / "eka" / f"session-{sid[:12]}.wav"
            sf.write(path, audio, SR)
            fh.write(json.dumps({"id": f"session-{sid[:12]}", "audio": f"eka/{path.name}", "text": text,
                                 "entities": spans, "duration": round(audio.shape[0] / SR, 2)}) + "\n")  # fmt: skip
            n_long += 1
    print(f"eka-longform: {n_long} sessions")


def prepare_librispeech(out: Path) -> None:
    import pyarrow.parquet as pq

    rows = pq.read_table(_download(LIBRI, "clean/validation-00000-of-00001.parquet")).to_pylist()
    chapters: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        chapters[(r["speaker_id"], r["chapter_id"])].append(r)
    (out / "librispeech").mkdir(parents=True, exist_ok=True)
    with (out / "librispeech-longform.jsonl").open("w") as fh:
        for (spk, ch), items in sorted(chapters.items()):
            items.sort(key=lambda r: r["id"])
            audio = np.concatenate([_decode(r["audio"]["bytes"]) for r in items])
            text = " ".join(r["text"].lower() for r in items)
            name = f"{spk}-{ch}.wav"
            sf.write(out / "librispeech" / name, audio, SR)
            fh.write(json.dumps({"id": f"{spk}-{ch}", "audio": f"librispeech/{name}", "text": text,
                                 "duration": round(audio.shape[0] / SR, 2)}) + "\n")  # fmt: skip
    print(f"librispeech-longform: {len(chapters)} chapters")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("eval-data"))
    ap.add_argument(
        "--eka-limit", type=int, default=300, help="clips per recording context (0 = all)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-eka", action="store_true")
    ap.add_argument("--skip-librispeech", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_eka:
        prepare_eka(args.out, args.eka_limit, args.seed)
    if not args.skip_librispeech:
        prepare_librispeech(args.out)


if __name__ == "__main__":
    main()
