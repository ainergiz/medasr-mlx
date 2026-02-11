#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import huggingface_hub
import mlx.core as mx
import numpy as np
from dotenv import load_dotenv
from transformers import AutoProcessor

from audio_utils import load_audio_mono
from decode import CTCTextDecoder, DecoderConfig
from model import load_mlx_model
from streaming import StreamingLasrTranscriber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedASR MLX transcription on local Mac.")
    parser.add_argument(
        "--model-dir",
        default="artifacts/medasr-mlx-fp16",
        help="Directory containing weights.npz/config.json/processor/",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to audio file. If omitted, uses google/medasr test_audio.wav.",
    )
    parser.add_argument(
        "--hf-model-id",
        default="google/medasr",
        help="HF model id only used for default test audio download fallback.",
    )
    parser.add_argument(
        "--decode-mode",
        default="greedy",
        choices=["greedy", "beam"],
        help="CTC decode mode: greedy argmax or beam search.",
    )
    parser.add_argument(
        "--kenlm-path",
        default=None,
        help="Path to KenLM file for beam decoding. If omitted, downloads lm_6.kenlm from HF model repo.",
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
        help="Enable cache-aware chunked streaming inference.",
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=1.5,
        help="Chunk length in seconds for --streaming mode.",
    )
    parser.add_argument(
        "--shift-sec",
        type=float,
        default=None,
        help="Chunk shift in seconds for --streaming mode. Defaults to --chunk-sec.",
    )
    parser.add_argument(
        "--left-context-sec",
        type=float,
        default=6.0,
        help="Maximum cached left context (seconds) for streaming attention KV cache.",
    )
    return parser.parse_args()


def load_audio(audio_path: str | None, hf_model_id: str) -> tuple[np.ndarray, int, str]:
    if audio_path:
        path = Path(audio_path)
    else:
        path = Path(huggingface_hub.hf_hub_download(hf_model_id, filename="test_audio.wav"))
    speech, sr = load_audio_mono(path, target_sr=16000)
    return speech.astype(np.float32, copy=False), sr, str(path)


def main() -> None:
    load_dotenv(".env.local", override=False)
    args = parse_args()

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

    model = load_mlx_model(model_dir)
    speech, sr, resolved_audio_path = load_audio(args.audio, args.hf_model_id)

    if args.streaming:
        if args.decode_mode != "greedy":
            raise SystemExit("Streaming v1 currently supports only --decode-mode greedy.")
        transcriber = StreamingLasrTranscriber(
            model=model,
            processor=processor,
            chunk_sec=args.chunk_sec,
            shift_sec=args.shift_sec,
            max_left_context_sec=args.left_context_sec,
            sample_rate=sr,
        )
        result = transcriber.transcribe_audio(speech=speech, sr=sr)
        print(f"audio: {resolved_audio_path}")
        print("streaming: enabled")
        print(f"chunk_sec: {args.chunk_sec}")
        print(f"shift_sec: {args.shift_sec if args.shift_sec is not None else args.chunk_sec}")
        print(f"left_context_sec: {args.left_context_sec}")
        print(f"chunks_processed: {result.num_chunks}")
        print(f"latency_s: {result.latency_s:.4f}")
        print(f"rtf: {result.rtf:.4f}")
        print(f"cache_memory_mb: {result.cache_memory_mb:.2f}")
        print(f"transcript: {result.text}")
        return

    features = processor(
        speech,
        sampling_rate=sr,
        return_attention_mask=True,
        return_tensors="np",
    )
    input_features = mx.array(features["input_features"])
    attention_mask = mx.array(features["attention_mask"].astype(np.bool_))

    logits = model(input_features=input_features, attention_mask=attention_mask)
    logits_np = np.asarray(logits)
    pred_ids = np.asarray(mx.argmax(logits, axis=-1))
    text = decoder.decode(logits_np, pred_ids=pred_ids)

    print(f"audio: {resolved_audio_path}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"decode mode: {args.decode_mode}")
    if args.decode_mode == "beam":
        print(f"kenlm path: {decoder.kenlm_path}")
        print(f"beam width: {args.beam_width}")
    print(f"transcript: {text}")


if __name__ == "__main__":
    main()
