#!/usr/bin/env python3
"""Three-level benchmark: parity, accuracy corpus, runtime/system metrics.

Usage:
    # Run on default test audio:
    python benchmark.py

    # Run on a folder of WAVs:
    python benchmark.py --audio-dir /path/to/wavs

    # Run with ground-truth transcripts for WER (one .txt per .wav, same stem):
    python benchmark.py --audio-dir /path/to/wavs --with-references

    # Skip HF model (MLX-only benchmarking):
    python benchmark.py --mlx-only

Output: artifacts/benchmark/results.csv + artifacts/benchmark/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jiwer
import mlx.core as mx
import numpy as np
from dotenv import load_dotenv
from transformers import AutoProcessor

from audio_utils import load_audio_mono
from decode import CTCTextDecoder, DecoderConfig
from model import load_mlx_model
from streaming import StreamingLasrTranscriber


@dataclass
class ClipResult:
    filename: str
    audio_duration_s: float
    # MLX metrics
    mlx_latency_s: float = 0.0
    mlx_rtf: float = 0.0
    mlx_transcript: str = ""
    # HF metrics (optional)
    hf_latency_s: float = 0.0
    hf_rtf: float = 0.0
    hf_transcript: str = ""
    # Parity (MLX vs HF)
    parity_wer: float = float("nan")
    parity_token_agreement: float = float("nan")
    parity_max_logit_diff: float = float("nan")
    parity_mean_logit_diff: float = float("nan")
    # Accuracy vs reference (if available)
    mlx_wer: float = float("nan")
    mlx_cer: float = float("nan")
    hf_wer: float = float("nan")
    hf_cer: float = float("nan")


@dataclass
class BenchmarkSummary:
    device: str
    chip: str
    memory_gb: float
    mlx_backend: str
    model_dir: str
    decode_mode: str
    kenlm_path: str | None
    kenlm_alpha: float | None
    kenlm_beta: float | None
    beam_width: int | None
    weights_dtype: str
    weights_size_mb: float
    model_load_time_s: float
    num_clips: int
    total_audio_duration_s: float
    # Latency stats
    mlx_latency_p50_s: float = 0.0
    mlx_latency_p95_s: float = 0.0
    mlx_latency_mean_s: float = 0.0
    mlx_rtf_mean: float = 0.0
    mlx_rtf_p50: float = 0.0
    mlx_rtf_p95: float = 0.0
    # Memory
    gpu_peak_memory_mb: float = 0.0
    gpu_active_memory_mb: float = 0.0
    process_rss_peak_mb: float = 0.0
    # Parity aggregate
    parity_all_wer_zero: bool = False
    parity_mean_token_agreement: float = 0.0
    parity_max_logit_diff: float = 0.0
    # Accuracy aggregate (if references available)
    mlx_corpus_wer: float = float("nan")
    hf_corpus_wer: float = float("nan")
    clips: list[ClipResult] = field(default_factory=list)


def get_chip_name() -> str:
    try:
        import subprocess

        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        if out:
            return out
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True)
        for line in out.splitlines():
            if "Chip" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return "unknown"


def get_system_memory_gb() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)


def get_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def find_audio_files(audio_dir: Path) -> list[Path]:
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = sorted(f for f in audio_dir.iterdir() if f.suffix.lower() in exts)
    return files


def load_reference(audio_path: Path) -> str | None:
    txt_path = audio_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text().strip()
    return None


def parse_chunk_seconds(value: str) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        sec = float(raw)
        if sec <= 0:
            raise ValueError(f"Chunk size must be > 0, got {sec}")
        if sec in seen:
            continue
        seen.add(sec)
        out.append(sec)
    if not out:
        raise ValueError("No valid chunk sizes provided for --chunk-sec.")
    return out


def run_mlx_inference(
    model, processor, decoder: CTCTextDecoder, speech: np.ndarray, sr: int
) -> tuple[mx.array, np.ndarray, str, float]:
    features = processor(speech, sampling_rate=sr, return_attention_mask=True, return_tensors="np")
    input_features = mx.array(features["input_features"])
    attention_mask = mx.array(features["attention_mask"].astype(np.bool_))

    t0 = time.perf_counter()
    logits = model(input_features=input_features, attention_mask=attention_mask)
    mx.eval(logits)
    elapsed = time.perf_counter() - t0

    logits_np = np.asarray(logits)
    pred_ids = np.asarray(mx.argmax(logits, axis=-1))
    text = decoder.decode(logits_np, pred_ids=pred_ids)
    return logits, pred_ids, text, elapsed


def run_mlx_streaming_inference(
    model,
    processor,
    speech: np.ndarray,
    sr: int,
    chunk_sec: float,
    shift_sec: float,
    left_context_sec: float,
) -> dict[str, float | int | str]:
    transcriber = StreamingLasrTranscriber(
        model=model,
        processor=processor,
        chunk_sec=chunk_sec,
        shift_sec=shift_sec,
        max_left_context_sec=left_context_sec,
        sample_rate=sr,
    )
    result = transcriber.transcribe_audio(speech=speech, sr=sr)
    return {
        "text": result.text,
        "latency_s": result.latency_s,
        "rtf": result.rtf,
        "num_chunks": result.num_chunks,
        "cache_memory_mb": result.cache_memory_mb,
        "token_count": result.token_count,
    }


def run_hf_inference(hf_model, processor, decoder: CTCTextDecoder, speech: np.ndarray, sr: int):
    import torch

    features = processor(speech, sampling_rate=sr, return_attention_mask=True, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        hf_logits = hf_model(**features).logits
    elapsed = time.perf_counter() - t0

    hf_ids = torch.argmax(hf_logits, dim=-1).cpu().numpy()
    text = decoder.decode(hf_logits.detach().cpu().numpy(), pred_ids=hf_ids)
    return hf_logits, hf_ids, text, elapsed


def compute_parity(
    mlx_logits: mx.array,
    mlx_ids: np.ndarray,
    hf_logits,
    hf_ids: np.ndarray,
) -> tuple[float, float, float, float]:
    import torch

    if hf_ids.shape == mlx_ids.shape:
        token_agreement = float((hf_ids == mlx_ids).mean())
    else:
        token_agreement = float("nan")

    hf_np = hf_logits.cpu().numpy() if isinstance(hf_logits, torch.Tensor) else np.array(hf_logits)
    mlx_np = np.asarray(mlx_logits)

    if hf_np.shape == mlx_np.shape:
        diff = np.abs(hf_np - mlx_np)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
    else:
        max_diff = float("nan")
        mean_diff = float("nan")

    wer = 0.0 if token_agreement == 1.0 else float("nan")
    return wer, token_agreement, max_diff, mean_diff


def benchmark_clip(
    audio_path: Path,
    mlx_model,
    processor,
    decoder: CTCTextDecoder,
    hf_model=None,
    with_references: bool = False,
    warmup_done: bool = False,
) -> ClipResult:
    speech, sr = load_audio_mono(audio_path, target_sr=16000)
    audio_duration = len(speech) / sr

    result = ClipResult(filename=audio_path.name, audio_duration_s=audio_duration)

    # Warmup if first clip
    if not warmup_done:
        run_mlx_inference(mlx_model, processor, decoder, speech, sr)

    # MLX: run 3 times, take median
    mlx_latencies = []
    for _ in range(3):
        mlx_logits, mlx_ids, mlx_text, mlx_elapsed = run_mlx_inference(
            mlx_model, processor, decoder, speech, sr
        )
        mlx_latencies.append(mlx_elapsed)

    result.mlx_latency_s = float(np.median(mlx_latencies))
    result.mlx_rtf = result.mlx_latency_s / audio_duration
    result.mlx_transcript = mlx_text

    # HF (if available)
    if hf_model is not None:
        hf_logits, hf_ids, hf_text, hf_elapsed = run_hf_inference(
            hf_model, processor, decoder, speech, sr
        )
        result.hf_latency_s = hf_elapsed
        result.hf_rtf = hf_elapsed / audio_duration
        result.hf_transcript = hf_text

        # Parity
        wer, tok_agree, max_diff, mean_diff = compute_parity(mlx_logits, mlx_ids, hf_logits, hf_ids)
        result.parity_wer = wer
        result.parity_token_agreement = tok_agree
        result.parity_max_logit_diff = max_diff
        result.parity_mean_logit_diff = mean_diff

    # Reference WER (if .txt file exists)
    if with_references:
        ref = load_reference(audio_path)
        if ref:
            result.mlx_wer = jiwer.wer(ref, mlx_text)
            result.mlx_cer = jiwer.cer(ref, mlx_text)
            if hf_model is not None:
                result.hf_wer = jiwer.wer(ref, hf_text)
                result.hf_cer = jiwer.cer(ref, hf_text)

    return result


def benchmark_clip_streaming(
    audio_path: Path,
    mlx_model,
    processor,
    with_references: bool,
    chunk_sec: float,
    shift_sec: float,
    left_context_sec: float,
    warmup_done: bool = False,
) -> tuple[ClipResult, int, float]:
    speech, sr = load_audio_mono(audio_path, target_sr=16000)
    audio_duration = len(speech) / sr

    result = ClipResult(filename=audio_path.name, audio_duration_s=audio_duration)

    if not warmup_done:
        _ = run_mlx_streaming_inference(
            mlx_model,
            processor,
            speech,
            sr,
            chunk_sec=chunk_sec,
            shift_sec=shift_sec,
            left_context_sec=left_context_sec,
        )

    run_rows: list[dict[str, float | int | str]] = []
    for _ in range(3):
        row = run_mlx_streaming_inference(
            mlx_model,
            processor,
            speech,
            sr,
            chunk_sec=chunk_sec,
            shift_sec=shift_sec,
            left_context_sec=left_context_sec,
        )
        run_rows.append(row)

    latencies = np.array([float(r["latency_s"]) for r in run_rows], dtype=np.float64)
    median_index = int(np.argsort(latencies)[len(latencies) // 2])
    median_run = run_rows[median_index]

    result.mlx_latency_s = float(median_run["latency_s"])
    result.mlx_rtf = result.mlx_latency_s / audio_duration
    result.mlx_transcript = str(median_run["text"])

    if with_references:
        ref = load_reference(audio_path)
        if ref:
            result.mlx_wer = jiwer.wer(ref, result.mlx_transcript)
            result.mlx_cer = jiwer.cer(ref, result.mlx_transcript)

    return (
        result,
        int(median_run["num_chunks"]),
        float(median_run["cache_memory_mb"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3-level MedASR benchmark.")
    parser.add_argument(
        "--model-dir",
        default="artifacts/medasr-mlx-fp16",
        help="MLX model directory.",
    )
    parser.add_argument(
        "--hf-model-id",
        default="google/medasr",
        help="HF model ID for parity comparison.",
    )
    parser.add_argument(
        "--audio-dir",
        default="audio",
        help="Directory of audio files to benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmark",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--mlx-only",
        action="store_true",
        help="Skip HF model loading (MLX-only benchmark).",
    )
    parser.add_argument(
        "--with-references",
        action="store_true",
        help="Compute WER against .txt reference files (same stem as audio).",
    )
    parser.add_argument(
        "--decode-mode",
        default="greedy",
        choices=["greedy", "beam"],
        help="CTC decode mode used for transcript text.",
    )
    parser.add_argument(
        "--kenlm-path",
        default=None,
        help="Path to KenLM file for beam decoding. If omitted, downloads lm_6.kenlm from HF repo.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=128,
        help="Beam width used when --decode-mode beam.",
    )
    parser.add_argument(
        "--kenlm-alpha",
        type=float,
        default=0.5,
        help="KenLM alpha weight used when --decode-mode beam.",
    )
    parser.add_argument(
        "--kenlm-beta",
        type=float,
        default=1.0,
        help="KenLM beta weight used when --decode-mode beam.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Run cache-aware streaming benchmark matrix instead of offline full-context only.",
    )
    parser.add_argument(
        "--chunk-sec",
        default="1.5",
        help="Chunk size(s) in seconds for --streaming. Accepts comma list, e.g. 1.0,1.5,3.0.",
    )
    parser.add_argument(
        "--shift-sec",
        type=float,
        default=None,
        help="Chunk shift in seconds for --streaming. Defaults to chunk size.",
    )
    parser.add_argument(
        "--left-context-sec",
        type=float,
        default=6.0,
        help="Maximum cached left context in seconds for streaming attention KV cache.",
    )
    parser.add_argument(
        "--skip-offline-baseline",
        action="store_true",
        help=(
            "In --streaming mode, skip full-context offline baseline. "
            "Useful for very long clips where offline inference may OOM."
        ),
    )
    return parser.parse_args()


def run_streaming_matrix(
    *,
    args: argparse.Namespace,
    audio_files: list[Path],
    output_dir: Path,
    mlx_model,
    processor,
    decoder: CTCTextDecoder,
    weights_dtype: str,
    weights_size_mb: float,
    model_load_time: float,
) -> None:
    if args.decode_mode != "greedy":
        raise SystemExit("Streaming benchmark currently supports only --decode-mode greedy.")

    chunk_sizes = parse_chunk_seconds(args.chunk_sec)
    if args.shift_sec is not None and args.shift_sec <= 0:
        raise SystemExit(f"--shift-sec must be > 0, got {args.shift_sec}")
    if args.left_context_sec <= 0:
        raise SystemExit(f"--left-context-sec must be > 0, got {args.left_context_sec}")

    offline_results: list[ClipResult] = []
    offline_failure_reason: str | None = None
    if args.skip_offline_baseline:
        print()
        print("Skipping offline baseline (--skip-offline-baseline).")
    else:
        print()
        print("Running offline baseline for streaming comparison...")
        try:
            for i, audio_path in enumerate(audio_files):
                print(
                    f"  [offline {i + 1}/{len(audio_files)}] {audio_path.name}...",
                    end=" ",
                    flush=True,
                )
                clip = benchmark_clip(
                    audio_path=audio_path,
                    mlx_model=mlx_model,
                    processor=processor,
                    decoder=decoder,
                    hf_model=None,
                    with_references=args.with_references,
                    warmup_done=(i > 0),
                )
                offline_results.append(clip)
                print(
                    f"dur={clip.audio_duration_s:.1f}s mlx_lat={clip.mlx_latency_s:.3f}s "
                    f"rtf={clip.mlx_rtf:.4f}"
                )
        except RuntimeError as exc:
            err = str(exc)
            if "metal::malloc" in err or "maximum allowed buffer size" in err:
                offline_failure_reason = err
                offline_results = []
                print()
                print("Offline baseline failed with OOM; continuing with streaming-only scenarios.")
            else:
                raise

    if offline_results:
        offline_lat = np.array([r.mlx_latency_s for r in offline_results], dtype=np.float64)
        offline_rtf = np.array([r.mlx_rtf for r in offline_results], dtype=np.float64)
        offline_refs: list[str] = []
        offline_hyps: list[str] = []
        if args.with_references:
            for r in offline_results:
                ref = load_reference(Path(args.audio_dir) / r.filename)
                if ref:
                    offline_refs.append(ref)
                    offline_hyps.append(r.mlx_transcript)
        offline_corpus_wer = jiwer.wer(offline_refs, offline_hyps) if offline_refs else float("nan")

        offline_summary: dict[str, Any] = {
            "num_clips": len(offline_results),
            "total_audio_duration_s": round(
                float(sum(r.audio_duration_s for r in offline_results)), 1
            ),
            "mlx_latency_mean_s": round(float(np.mean(offline_lat)), 4),
            "mlx_latency_p50_s": round(float(np.percentile(offline_lat, 50)), 4),
            "mlx_latency_p95_s": round(float(np.percentile(offline_lat, 95)), 4),
            "mlx_rtf_mean": round(float(np.mean(offline_rtf)), 4),
            "mlx_rtf_p50": round(float(np.percentile(offline_rtf, 50)), 4),
            "mlx_rtf_p95": round(float(np.percentile(offline_rtf, 95)), 4),
            "mlx_corpus_wer": offline_corpus_wer,
            "clips": [
                {
                    "filename": r.filename,
                    "audio_duration_s": r.audio_duration_s,
                    "mlx_latency_s": r.mlx_latency_s,
                    "mlx_rtf": r.mlx_rtf,
                    "mlx_wer": r.mlx_wer,
                }
                for r in offline_results
            ],
        }
    else:
        offline_corpus_wer = float("nan")
        offline_summary = {
            "num_clips": 0,
            "total_audio_duration_s": 0.0,
            "mlx_latency_mean_s": float("nan"),
            "mlx_latency_p50_s": float("nan"),
            "mlx_latency_p95_s": float("nan"),
            "mlx_rtf_mean": float("nan"),
            "mlx_rtf_p50": float("nan"),
            "mlx_rtf_p95": float("nan"),
            "mlx_corpus_wer": float("nan"),
            "clips": [],
        }

    scenario_summaries: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for chunk_sec in chunk_sizes:
        shift_sec = args.shift_sec if args.shift_sec is not None else chunk_sec
        print()
        print(
            "Running streaming setting "
            f"(chunk={chunk_sec:.2f}s, shift={shift_sec:.2f}s, left_context={args.left_context_sec:.2f}s)..."
        )

        stream_results: list[ClipResult] = []
        chunk_counts: list[int] = []
        cache_mbs: list[float] = []

        for i, audio_path in enumerate(audio_files):
            print(
                f"  [stream {i + 1}/{len(audio_files)}] {audio_path.name}...", end=" ", flush=True
            )
            clip, num_chunks, cache_mb = benchmark_clip_streaming(
                audio_path=audio_path,
                mlx_model=mlx_model,
                processor=processor,
                with_references=args.with_references,
                chunk_sec=chunk_sec,
                shift_sec=shift_sec,
                left_context_sec=args.left_context_sec,
                warmup_done=(i > 0),
            )
            stream_results.append(clip)
            chunk_counts.append(num_chunks)
            cache_mbs.append(cache_mb)
            print(
                f"dur={clip.audio_duration_s:.1f}s mlx_lat={clip.mlx_latency_s:.3f}s "
                f"rtf={clip.mlx_rtf:.4f} chunks={num_chunks} cache={cache_mb:.2f}MB"
            )

            matrix_rows.append(
                {
                    "chunk_sec": chunk_sec,
                    "shift_sec": shift_sec,
                    "left_context_sec": args.left_context_sec,
                    "filename": clip.filename,
                    "audio_duration_s": clip.audio_duration_s,
                    "mlx_latency_s": clip.mlx_latency_s,
                    "mlx_rtf": clip.mlx_rtf,
                    "num_chunks": num_chunks,
                    "cache_memory_mb": cache_mb,
                    "mlx_wer": clip.mlx_wer,
                    "mlx_cer": clip.mlx_cer,
                }
            )

        lat = np.array([r.mlx_latency_s for r in stream_results], dtype=np.float64)
        rtf = np.array([r.mlx_rtf for r in stream_results], dtype=np.float64)
        refs: list[str] = []
        hyps: list[str] = []
        if args.with_references:
            for r in stream_results:
                ref = load_reference(Path(args.audio_dir) / r.filename)
                if ref:
                    refs.append(ref)
                    hyps.append(r.mlx_transcript)
        stream_corpus_wer = jiwer.wer(refs, hyps) if refs else float("nan")

        wer_delta = (
            float(stream_corpus_wer - offline_corpus_wer)
            if (not np.isnan(stream_corpus_wer) and not np.isnan(offline_corpus_wer))
            else float("nan")
        )

        scenario_summary = {
            "chunk_sec": chunk_sec,
            "shift_sec": shift_sec,
            "left_context_sec": args.left_context_sec,
            "num_clips": len(stream_results),
            "mlx_latency_mean_s": round(float(np.mean(lat)), 4),
            "mlx_latency_p50_s": round(float(np.percentile(lat, 50)), 4),
            "mlx_latency_p95_s": round(float(np.percentile(lat, 95)), 4),
            "mlx_rtf_mean": round(float(np.mean(rtf)), 4),
            "mlx_rtf_p50": round(float(np.percentile(rtf, 50)), 4),
            "mlx_rtf_p95": round(float(np.percentile(rtf, 95)), 4),
            "mlx_corpus_wer": stream_corpus_wer,
            "wer_delta_vs_offline": wer_delta,
            "num_chunks_mean": round(float(np.mean(np.array(chunk_counts, dtype=np.float64))), 2),
            "cache_memory_mb_mean": round(float(np.mean(np.array(cache_mbs, dtype=np.float64))), 3),
            "cache_memory_mb_p95": round(
                float(np.percentile(np.array(cache_mbs, dtype=np.float64), 95)), 3
            ),
            "clips": [
                {
                    "filename": r.filename,
                    "audio_duration_s": r.audio_duration_s,
                    "mlx_latency_s": r.mlx_latency_s,
                    "mlx_rtf": r.mlx_rtf,
                    "mlx_wer": r.mlx_wer,
                    "num_chunks": n,
                    "cache_memory_mb": c,
                }
                for r, n, c in zip(stream_results, chunk_counts, cache_mbs)
            ],
        }
        scenario_summaries.append(scenario_summary)

    csv_path = output_dir / "results_streaming_matrix.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "chunk_sec",
            "shift_sec",
            "left_context_sec",
            "filename",
            "audio_duration_s",
            "mlx_latency_s",
            "mlx_rtf",
            "num_chunks",
            "cache_memory_mb",
            "mlx_wer",
            "mlx_cer",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix_rows)

    notes = [
        "Streaming mode keeps offline path untouched and is enabled only with --streaming.",
        "MedASR is full-context trained, so chunk boundary WER degradation is expected at smaller chunks.",
        "Subsampler/front-end state is not yet cached in this v1 implementation.",
    ]
    if args.skip_offline_baseline:
        notes.append("Offline baseline skipped via --skip-offline-baseline.")
    if offline_failure_reason:
        notes.append("Offline baseline failed due memory limits on long full-context audio.")

    summary = {
        "mode": "streaming-matrix",
        "device": "Mac",
        "chip": get_chip_name(),
        "memory_gb": round(get_system_memory_gb(), 1),
        "mlx_backend": str(mx.default_device()),
        "model_dir": args.model_dir,
        "decode_mode": args.decode_mode,
        "weights_dtype": weights_dtype,
        "weights_size_mb": round(weights_size_mb, 1),
        "model_load_time_s": round(model_load_time, 3),
        "offline_baseline": offline_summary,
        "streaming_scenarios": scenario_summaries,
        "notes": notes,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print("=" * 60)
    print("STREAMING MATRIX SUMMARY")
    print("=" * 60)
    print(f"Model:                   {args.model_dir} ({weights_dtype})")
    print(f"Output dir:              {output_dir}")
    if np.isnan(offline_summary["mlx_latency_p50_s"]):
        print("Offline baseline:        skipped/unavailable")
    else:
        print(
            "Offline baseline:        "
            f"lat_p50={offline_summary['mlx_latency_p50_s']:.4f}s "
            f"rtf_p50={offline_summary['mlx_rtf_p50']:.4f} "
            f"wer={offline_summary['mlx_corpus_wer'] if not np.isnan(offline_summary['mlx_corpus_wer']) else 'n/a'}"
        )
    print()
    for scenario in scenario_summaries:
        wer = scenario["mlx_corpus_wer"]
        delta = scenario["wer_delta_vs_offline"]
        print(
            f"chunk={scenario['chunk_sec']:.2f}s shift={scenario['shift_sec']:.2f}s "
            f"lat_p50={scenario['mlx_latency_p50_s']:.4f}s rtf_p50={scenario['mlx_rtf_p50']:.4f} "
            f"wer={(f'{wer:.4f}' if not np.isnan(wer) else 'n/a')} "
            f"delta={(f'{delta:+.4f}' if not np.isnan(delta) else 'n/a')} "
            f"cache_mb_mean={scenario['cache_memory_mb_mean']:.3f}"
        )
    print()
    print(f"Results: {csv_path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    load_dotenv(".env.local", override=False)
    args = parse_args()

    audio_dir = Path(args.audio_dir)
    if args.streaming and args.output_dir == "artifacts/benchmark":
        output_dir = Path("artifacts/benchmark-streaming")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        raise SystemExit(1)
    print(f"Found {len(audio_files)} audio file(s) in {audio_dir}")

    # Load MLX model
    print(f"Loading MLX model from {args.model_dir}...")
    t0 = time.perf_counter()
    mlx_model = load_mlx_model(args.model_dir)
    model_load_time = time.perf_counter() - t0

    model_dir = Path(args.model_dir)
    processor_path = model_dir / "processor"
    if processor_path.exists():
        processor = AutoProcessor.from_pretrained(str(processor_path))
    else:
        processor = AutoProcessor.from_pretrained(args.hf_model_id)
    decoder = CTCTextDecoder(
        processor=processor,
        config=DecoderConfig(
            mode=args.decode_mode,
            hf_model_id=args.hf_model_id,
            kenlm_path=args.kenlm_path,
            alpha=args.kenlm_alpha,
            beta=args.kenlm_beta,
            beam_width=args.beam_width,
        ),
    )

    if args.streaming and not args.mlx_only:
        print("Streaming mode uses MLX-only path. Ignoring HF parity loading.")
        args.mlx_only = True

    # Load HF model (optional)
    hf_model = None
    if not args.mlx_only:
        print(f"Loading HF model {args.hf_model_id}...")
        from transformers import AutoModelForCTC

        hf_model = AutoModelForCTC.from_pretrained(args.hf_model_id)
        hf_model.eval()

    # Weights info
    weights_path = model_dir / "weights.npz"
    weights_size_mb = weights_path.stat().st_size / (1024 * 1024) if weights_path.exists() else 0.0
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    qcfg = config.get("_quantization", {}) if isinstance(config, dict) else {}
    if isinstance(qcfg, dict) and qcfg.get("enabled", False):
        q_bits = qcfg.get("bits", "?")
        q_group = qcfg.get("group_size", "?")
        q_mode = qcfg.get("mode", "affine")
        weights_dtype = f"{q_mode}-int{q_bits}-g{q_group}"
    else:
        weights_dtype = config.get("_conversion", {}).get("weights_dtype", "unknown")

    if args.streaming:
        mx.get_peak_memory()  # read to reset
        run_streaming_matrix(
            args=args,
            audio_files=audio_files,
            output_dir=output_dir,
            mlx_model=mlx_model,
            processor=processor,
            decoder=decoder,
            weights_dtype=weights_dtype,
            weights_size_mb=weights_size_mb,
            model_load_time=model_load_time,
        )
        return

    # Reset memory tracking
    mx.get_peak_memory()  # read to reset

    # Benchmark each clip
    results: list[ClipResult] = []
    for i, audio_path in enumerate(audio_files):
        print(f"  [{i + 1}/{len(audio_files)}] {audio_path.name}...", end=" ", flush=True)
        clip = benchmark_clip(
            audio_path=audio_path,
            mlx_model=mlx_model,
            processor=processor,
            decoder=decoder,
            hf_model=hf_model,
            with_references=args.with_references,
            warmup_done=(i > 0),
        )
        results.append(clip)
        print(
            f"dur={clip.audio_duration_s:.1f}s "
            f"mlx_lat={clip.mlx_latency_s:.3f}s "
            f"rtf={clip.mlx_rtf:.4f}"
            + (f" parity_tok={clip.parity_token_agreement:.4f}" if hf_model else "")
        )

    # Aggregate stats
    mlx_latencies = np.array([r.mlx_latency_s for r in results])
    mlx_rtfs = np.array([r.mlx_rtf for r in results])
    total_audio = sum(r.audio_duration_s for r in results)

    gpu_peak = mx.get_peak_memory() / (1024 * 1024)
    gpu_active = mx.get_active_memory() / (1024 * 1024)
    rss_peak = get_rss_mb()

    parity_agreements = [
        r.parity_token_agreement for r in results if not np.isnan(r.parity_token_agreement)
    ]
    parity_max_diffs = [
        r.parity_max_logit_diff for r in results if not np.isnan(r.parity_max_logit_diff)
    ]

    # Corpus-level WER (concatenate all transcripts)
    mlx_corpus_wer = float("nan")
    hf_corpus_wer = float("nan")
    if args.with_references:
        refs = []
        mlx_hyps = []
        hf_hyps = []
        for r in results:
            ref = load_reference(Path(args.audio_dir) / r.filename)
            if ref:
                refs.append(ref)
                mlx_hyps.append(r.mlx_transcript)
                if hf_model:
                    hf_hyps.append(r.hf_transcript)
        if refs:
            mlx_corpus_wer = jiwer.wer(refs, mlx_hyps)
            if hf_hyps:
                hf_corpus_wer = jiwer.wer(refs, hf_hyps)

    summary = BenchmarkSummary(
        device="Mac",
        chip=get_chip_name(),
        memory_gb=round(get_system_memory_gb(), 1),
        mlx_backend=str(mx.default_device()),
        model_dir=args.model_dir,
        decode_mode=args.decode_mode,
        kenlm_path=decoder.kenlm_path if args.decode_mode == "beam" else None,
        kenlm_alpha=args.kenlm_alpha if args.decode_mode == "beam" else None,
        kenlm_beta=args.kenlm_beta if args.decode_mode == "beam" else None,
        beam_width=args.beam_width if args.decode_mode == "beam" else None,
        weights_dtype=weights_dtype,
        weights_size_mb=round(weights_size_mb, 1),
        model_load_time_s=round(model_load_time, 3),
        num_clips=len(results),
        total_audio_duration_s=round(total_audio, 1),
        mlx_latency_mean_s=round(float(np.mean(mlx_latencies)), 4),
        mlx_latency_p50_s=round(float(np.percentile(mlx_latencies, 50)), 4),
        mlx_latency_p95_s=round(float(np.percentile(mlx_latencies, 95)), 4),
        mlx_rtf_mean=round(float(np.mean(mlx_rtfs)), 4),
        mlx_rtf_p50=round(float(np.percentile(mlx_rtfs, 50)), 4),
        mlx_rtf_p95=round(float(np.percentile(mlx_rtfs, 95)), 4),
        gpu_peak_memory_mb=round(gpu_peak, 1),
        gpu_active_memory_mb=round(gpu_active, 1),
        process_rss_peak_mb=round(rss_peak, 1),
        parity_all_wer_zero=all(r.parity_wer == 0.0 for r in results)
        if parity_agreements
        else False,
        parity_mean_token_agreement=round(float(np.mean(parity_agreements)), 6)
        if parity_agreements
        else 0.0,
        parity_max_logit_diff=round(float(np.max(parity_max_diffs)), 6)
        if parity_max_diffs
        else 0.0,
        mlx_corpus_wer=mlx_corpus_wer,
        hf_corpus_wer=hf_corpus_wer,
        clips=results,
    )

    # Write CSV
    csv_path = output_dir / "results.csv"
    fieldnames = [
        "filename",
        "audio_duration_s",
        "mlx_latency_s",
        "mlx_rtf",
        "mlx_transcript",
        "hf_latency_s",
        "hf_rtf",
        "hf_transcript",
        "parity_wer",
        "parity_token_agreement",
        "parity_max_logit_diff",
        "parity_mean_logit_diff",
        "mlx_wer",
        "mlx_cer",
        "hf_wer",
        "hf_cer",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            writer.writerow({k: row[k] for k in fieldnames})

    # Write JSON summary
    summary_path = output_dir / "summary.json"
    summary_dict = asdict(summary)
    # Clean up clips for JSON (remove long transcript strings from summary level)
    for clip in summary_dict["clips"]:
        clip["mlx_transcript"] = (
            clip["mlx_transcript"][:200] + "..."
            if len(clip["mlx_transcript"]) > 200
            else clip["mlx_transcript"]
        )
        clip["hf_transcript"] = (
            clip["hf_transcript"][:200] + "..."
            if len(clip["hf_transcript"]) > 200
            else clip["hf_transcript"]
        )
    summary_path.write_text(json.dumps(summary_dict, indent=2) + "\n")

    # Print summary
    print()
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Device:                  {summary.chip} ({summary.memory_gb} GB)")
    print(f"MLX backend:             {summary.mlx_backend}")
    print(f"Model:                   {summary.model_dir} ({summary.weights_dtype})")
    print(f"Decode mode:             {summary.decode_mode}")
    if summary.decode_mode == "beam":
        print(f"KenLM path:              {summary.kenlm_path}")
        print(
            "KenLM params:            "
            f"beam={summary.beam_width}, alpha={summary.kenlm_alpha}, beta={summary.kenlm_beta}"
        )
    print(f"Weights size:            {summary.weights_size_mb:.1f} MB")
    print(f"Model load time:         {summary.model_load_time_s:.3f}s")
    print(f"Clips:                   {summary.num_clips}")
    print(f"Total audio:             {summary.total_audio_duration_s:.1f}s")
    print()
    print("--- Runtime (MLX) ---")
    print(f"Latency p50:             {summary.mlx_latency_p50_s:.4f}s")
    print(f"Latency p95:             {summary.mlx_latency_p95_s:.4f}s")
    print(f"Latency mean:            {summary.mlx_latency_mean_s:.4f}s")
    print(f"RTF p50:                 {summary.mlx_rtf_p50:.4f}")
    print(f"RTF p95:                 {summary.mlx_rtf_p95:.4f}")
    print(f"RTF mean:                {summary.mlx_rtf_mean:.4f}")
    print()
    print("--- Memory ---")
    print(f"GPU peak memory:         {summary.gpu_peak_memory_mb:.1f} MB")
    print(f"GPU active memory:       {summary.gpu_active_memory_mb:.1f} MB")
    print(f"Process RSS peak:        {summary.process_rss_peak_mb:.1f} MB")
    if parity_agreements:
        print()
        print("--- Parity (MLX vs HF) ---")
        print(f"All WER zero:            {summary.parity_all_wer_zero}")
        print(f"Mean token agreement:    {summary.parity_mean_token_agreement:.6f}")
        print(f"Max logit diff:          {summary.parity_max_logit_diff:.6f}")
    if not np.isnan(mlx_corpus_wer):
        print()
        print("--- Accuracy (vs reference) ---")
        print(f"MLX corpus WER:          {mlx_corpus_wer:.4f}")
        if not np.isnan(hf_corpus_wer):
            print(f"HF corpus WER:           {hf_corpus_wer:.4f}")
    print()
    print(f"Results: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
