"""``medasr`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .weights import DEFAULT_MODEL

FORMATS = ("txt", "json", "srt", "vtt")


# ----------------------------------------------------------------- helpers
def _load_asr(args: argparse.Namespace):
    from .transcribe import MedASR

    return MedASR.from_pretrained(args.model, dtype=args.dtype)


def _read_hotwords(args: argparse.Namespace) -> dict[str, float] | None:
    phrases: dict[str, float] = {}
    for p in (args.hotwords or "").split(","):
        if p.strip():
            phrases[p.strip()] = args.hotword_weight
    if args.hotwords_file:
        for line in Path(args.hotwords_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            phrase, _, weight = line.partition("\t")
            phrases[phrase.strip()] = float(weight) if weight else args.hotword_weight
    return phrases or None


def _decoder_kwargs(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    hotwords = _read_hotwords(args)
    if args.decoder == "greedy" and (hotwords or args.lm):
        args.decoder = "beam"  # LM and hotwords need beam search
    if args.decoder == "greedy":
        return "greedy", {}
    kw: dict[str, Any] = {"beam_width": args.beam_width, "hotword_weight": args.hotword_weight}
    if args.lm:
        kw.update(lm=args.lm, alpha=args.alpha, beta=args.beta)
    if hotwords:
        kw["hotwords"] = hotwords
    return "beam", kw


def _window(args: argparse.Namespace) -> dict[str, Any]:
    return {"window_s": None if args.no_window else args.window, "stride_s": args.stride}


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL, help="local dir or Hugging Face repo id")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None,
                   help="compute dtype (default: float16 on GPU, float32 on CPU)")  # fmt: skip


def _add_decoder_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("decoding")
    g.add_argument("--decoder", choices=["greedy", "beam"], default="greedy")
    g.add_argument("--beam-width", type=int, default=8)
    g.add_argument("--lm", help="token-level KenLM/ARPA LM, e.g. lm_6.kenlm from google/medasr")
    g.add_argument("--alpha", type=float, default=0.5, help="LM weight")
    g.add_argument("--beta", type=float, default=1.0, help="per-token bonus with an LM")
    g.add_argument("--hotwords", help="comma-separated phrases to boost, e.g. 'Xarelto,metoprolol'")
    g.add_argument("--hotwords-file", help="one phrase per line, optional '<TAB>weight'")
    g.add_argument("--hotword-weight", type=float, default=5.0)


def _add_window_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("long-form (Temporal Posterior Fusion)")
    g.add_argument("--window", type=float, default=20.0, help="window length in seconds")
    g.add_argument("--stride", type=float, default=18.0, help="window stride in seconds")
    g.add_argument("--no-window", action="store_true", help="single full-context pass")


# ---------------------------------------------------------------- commands
def cmd_transcribe(args: argparse.Namespace) -> int:
    asr = _load_asr(args)
    decoder, kw = _decoder_kwargs(args)
    formats = FORMATS if args.format == "all" else (args.format,)
    for path in args.audio:
        result = asr.transcribe(path, decoder=decoder, **_window(args), **kw)
        if args.output_dir:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for fmt in formats:
                (out / f"{Path(path).stem}.{fmt}").write_text(result.format(fmt), encoding="utf-8")
        if not args.output_dir or args.verbose:
            if len(args.audio) > 1:
                print(f"==> {path}", file=sys.stderr)
            sys.stdout.write(result.format(formats[0]))
        if args.verbose:
            m = result.metadata
            print(f"[{result.duration:.1f}s audio, rtf={m.get('rtf')}, decoder={m.get('decoder')}]",
                  file=sys.stderr)  # fmt: skip
    return 0


def _print_update(update, final: bool = False) -> None:
    dim, reset = "\033[2m", "\033[0m"
    line = update.committed_text + dim + update.tentative_text + reset
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[K" + line[-4000:] + ("\n" if final else ""))
    else:
        sys.stdout.write(f"[{update.audio_seconds:7.2f}s] {update.text}\n")
    sys.stdout.flush()


def cmd_stream(args: argparse.Namespace) -> int:
    asr = _load_asr(args)
    st = asr.stream(
        preset=args.preset,
        window_s=args.window,
        stride_s=args.stride,
        commit_delay_s=args.commit_delay,
        start_padding=args.start_padding,
    )
    sr = asr.config.frontend.sample_rate
    chunk = max(1, round(args.chunk * sr))
    if args.mic:
        try:
            import sounddevice as sd
        except ImportError:
            print("Microphone input needs: pip install 'medasr-mlx[mic]'", file=sys.stderr)
            return 2
        print("Listening... (Ctrl+C to stop)", file=sys.stderr)
        try:
            with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=chunk) as mic:
                while True:
                    block, _ = mic.read(chunk)
                    _print_update(st.push(block[:, 0]))
        except KeyboardInterrupt:
            pass
    else:
        from .audio import load_audio

        if not args.audio:
            print("Give an audio file or --mic", file=sys.stderr)
            return 2
        wave = load_audio(args.audio)
        t_start = time.perf_counter()
        for i in range(0, wave.shape[0], chunk):
            if args.realtime:
                lag = (i / sr) - (time.perf_counter() - t_start)
                if lag > 0:
                    time.sleep(lag)
            _print_update(st.push(wave[i : i + chunk]))
    result = st.finish()
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[K")
    print(result.text)
    m = result.metadata
    print(f"[{result.duration:.1f}s audio, {m['windows_run']} windows, compute rtf={m['rtf']}]",
          file=sys.stderr)  # fmt: skip
    return 0


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        items = []
        for f in sorted(path.iterdir()):
            ref = f.with_suffix(".txt")
            if f.suffix.lower() in exts and ref.exists():
                items.append({"id": f.stem, "audio": str(f), "text": ref.read_text().strip()})
        return items
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            audio = Path(rec["audio"])
            rec["audio"] = str(audio if audio.is_absolute() else path.parent / audio)
            items.append(rec)
    return items


def cmd_eval(args: argparse.Namespace) -> int:
    from .audio import load_audio
    from .evaluation import Scores

    items = _load_manifest(Path(args.data))
    if args.limit:
        items = items[: args.limit]
    asr = _load_asr(args)
    decoder, kw = _decoder_kwargs(args)
    scores, rows = Scores(), []
    by_group: dict[str, Scores] = {}
    audio_s = compute_s = 0.0
    for i, item in enumerate(items):
        wave = load_audio(item["audio"])
        t0 = time.perf_counter()
        if args.streaming:
            st = asr.stream(preset=args.preset)
            st.push(wave)
            hyp = st.finish(decoder=decoder, **kw).text
        else:
            hyp = asr.transcribe(wave, decoder=decoder, **_window(args), **kw).text
        compute_s += time.perf_counter() - t0
        audio_s += wave.shape[0] / asr.config.frontend.sample_rate
        ents = [tuple(e) for e in item.get("entities") or []] or None
        row = scores.add(item["text"], hyp, ents)
        group = item.get("context")
        if group:
            by_group.setdefault(group, Scores()).add(item["text"], hyp, ents)
        rows.append({"id": item.get("id", i), "ref": item["text"], "hyp": hyp, **row})
        if args.verbose:
            print(f"[{i + 1}/{len(items)}] wer={row['wer']:.3f} {hyp[:80]}", file=sys.stderr)
    summary = {**scores.summary(), "audio_hours": round(audio_s / 3600, 3),
               "rtf": round(compute_s / max(audio_s, 1e-9), 5)}  # fmt: skip
    if by_group:
        summary["by_context"] = {k: v.summary() for k, v in sorted(by_group.items())}
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps({"summary": summary, "utterances": rows}, indent=1))
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    from .convert import convert

    out = convert(
        args.source,
        args.output,
        dtype=args.dtype,
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
        quantize_io=args.quantize_io,
    )
    size = sum(f.stat().st_size for f in out.iterdir()) / 2**20
    print(f"Wrote {out} ({size:.1f} MB)")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    import mlx.core as mx

    from .audio import load_audio

    asr = _load_asr(args)
    wave = load_audio(args.audio)
    dur = wave.shape[0] / asr.config.frontend.sample_rate
    asr.transcribe(wave, **_window(args))  # warm-up / compile
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        asr.transcribe(wave, **_window(args))
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]
    peak = mx.get_peak_memory() / 2**20 if hasattr(mx, "get_peak_memory") else float("nan")
    print(json.dumps({"audio_s": round(dur, 2), "device": str(mx.default_device()),
                      "dtype": str(asr.model.dtype), "latency_median_s": round(med, 4),
                      "latency_min_s": round(times[0], 4), "rtf": round(med / dur, 5),
                      "x_realtime": round(dur / med, 1), "peak_memory_mb": round(peak, 1)}, indent=2))  # fmt: skip
    return 0


# -------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="medasr", description="MedASR medical speech recognition on MLX"
    )
    ap.add_argument("--version", action="version", version=f"medasr-mlx {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("transcribe", help="transcribe audio files")
    p.add_argument("audio", nargs="+")
    _add_model_args(p)
    _add_decoder_args(p)
    _add_window_args(p)
    p.add_argument("--format", choices=[*FORMATS, "all"], default="txt")
    p.add_argument("--output-dir", help="write <stem>.<format> files here")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("stream", help="streaming recognition from a file or the microphone")
    p.add_argument("audio", nargs="?")
    p.add_argument("--mic", action="store_true")
    p.add_argument("--realtime", action="store_true", help="feed a file at real-time speed")
    p.add_argument("--preset", choices=["paper", "balanced", "low-power"], default="balanced")
    p.add_argument("--window", type=float, default=None)
    p.add_argument("--stride", type=float, default=None)
    p.add_argument("--commit-delay", type=float, default=None)
    p.add_argument("--start-padding", action="store_true", help="paper-style zero padding")
    p.add_argument("--chunk", type=float, default=0.1, help="audio chunk size in seconds")
    _add_model_args(p)
    p.set_defaults(func=cmd_stream)

    p = sub.add_parser("eval", help="WER / CER / medical-term metrics on a manifest")
    p.add_argument("data", help="JSONL manifest (audio, text[, entities]) or dir of audio + .txt")
    p.add_argument("--output", help="write per-utterance results JSON")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--preset", choices=["paper", "balanced", "low-power"], default="balanced")
    p.add_argument("-v", "--verbose", action="store_true")
    _add_model_args(p)
    _add_decoder_args(p)
    _add_window_args(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser(
        "convert", help="convert google/medasr or v1 weights to v2 (optionally quantized)"
    )
    p.add_argument("--source", default="google/medasr", help="HF repo id or local dir")
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument(
        "--bits", type=int, choices=[3, 4, 5, 6, 8], default=None, help="quantize Linear layers"
    )
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--mode", default="affine", choices=["affine", "mxfp4", "mxfp8", "nvfp4"])
    p.add_argument("--quantize-io", action="store_true", help="also quantize subsampler + CTC head")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("bench", help="latency / real-time factor on one file")
    p.add_argument("audio")
    p.add_argument("--runs", type=int, default=5)
    _add_model_args(p)
    _add_window_args(p)
    p.set_defaults(func=cmd_bench)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
