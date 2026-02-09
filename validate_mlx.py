#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import huggingface_hub
import jiwer
import librosa
import mlx.core as mx
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCTC, AutoProcessor

from model import load_mlx_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HF MedASR logits/text against MLX port.")
    parser.add_argument("--model-dir", default="artifacts/medasr-mlx-fp16")
    parser.add_argument("--hf-model-id", default="google/medasr")
    parser.add_argument("--audio", default=None)
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

    speech, sr, resolved_audio_path = load_audio(args.audio, args.hf_model_id)
    processor = AutoProcessor.from_pretrained(args.hf_model_id)

    hf_inputs = processor(
        speech,
        sampling_rate=sr,
        return_attention_mask=True,
        return_tensors="pt",
    )
    np_inputs = processor(
        speech,
        sampling_rate=sr,
        return_attention_mask=True,
        return_tensors="np",
    )

    hf_model = AutoModelForCTC.from_pretrained(args.hf_model_id)
    hf_model.eval()
    with torch.no_grad():
        hf_logits = hf_model(**hf_inputs).logits
    hf_ids = torch.argmax(hf_logits, dim=-1).cpu().numpy()
    hf_text = processor.batch_decode(hf_ids, skip_special_tokens=True)[0]

    mlx_model = load_mlx_model(args.model_dir)
    mlx_logits = mlx_model(
        input_features=mx.array(np_inputs["input_features"]),
        attention_mask=mx.array(np_inputs["attention_mask"].astype(np.bool_)),
    )
    mlx_logits_np = np.asarray(mlx_logits)
    mlx_ids = np.argmax(mlx_logits_np, axis=-1)
    mlx_text = processor.batch_decode(mlx_ids, skip_special_tokens=True)[0]

    if hf_ids.shape == mlx_ids.shape:
        token_agreement = float((hf_ids == mlx_ids).mean())
    else:
        token_agreement = float("nan")

    if hf_logits.shape == mlx_logits_np.shape:
        max_abs_diff = float(np.max(np.abs(hf_logits.cpu().numpy() - mlx_logits_np)))
        mean_abs_diff = float(np.mean(np.abs(hf_logits.cpu().numpy() - mlx_logits_np)))
    else:
        max_abs_diff = float("nan")
        mean_abs_diff = float("nan")

    wer = jiwer.wer(hf_text, mlx_text)

    print(f"audio: {resolved_audio_path}")
    print(f"hf logits shape: {tuple(hf_logits.shape)}")
    print(f"mlx logits shape: {tuple(mlx_logits.shape)}")
    print(f"token agreement: {token_agreement:.6f}")
    print(f"max abs logit diff: {max_abs_diff:.6f}")
    print(f"mean abs logit diff: {mean_abs_diff:.6f}")
    print(f"text WER (hf vs mlx): {wer:.6f}")
    print("-----")
    print(f"HF : {hf_text}")
    print(f"MLX: {mlx_text}")


if __name__ == "__main__":
    main()
