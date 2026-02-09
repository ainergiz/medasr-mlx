#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import huggingface_hub
import librosa
import mlx.core as mx
import numpy as np
from dotenv import load_dotenv
from transformers import AutoProcessor

from model import load_mlx_model


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
    return parser.parse_args()


def load_audio(audio_path: str | None, hf_model_id: str) -> tuple[np.ndarray, int, str]:
    if audio_path:
        path = Path(audio_path)
    else:
        path = Path(huggingface_hub.hf_hub_download(hf_model_id, filename="test_audio.wav"))
    speech, sr = librosa.load(str(path), sr=16000)
    return speech.astype(np.float32), sr, str(path)


def main() -> None:
    load_dotenv(".env.local", override=False)
    args = parse_args()

    model_dir = Path(args.model_dir)
    processor_path = model_dir / "processor"
    if processor_path.exists():
        processor = AutoProcessor.from_pretrained(str(processor_path))
    else:
        processor = AutoProcessor.from_pretrained(args.hf_model_id)

    model = load_mlx_model(model_dir)
    speech, sr, resolved_audio_path = load_audio(args.audio, args.hf_model_id)

    features = processor(
        speech,
        sampling_rate=sr,
        return_attention_mask=True,
        return_tensors="np",
    )
    input_features = mx.array(features["input_features"])
    attention_mask = mx.array(features["attention_mask"].astype(np.bool_))

    logits = model(input_features=input_features, attention_mask=attention_mask)
    pred_ids = np.asarray(mx.argmax(logits, axis=-1))
    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

    print(f"audio: {resolved_audio_path}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"transcript: {text}")


if __name__ == "__main__":
    main()
