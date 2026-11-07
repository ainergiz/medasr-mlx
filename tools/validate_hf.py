#!/usr/bin/env python3
"""Parity check: MedASR-MLX vs the reference PyTorch implementation (transformers).

Needs access to the gated ``google/medasr`` repo (accept the HAI-DEF terms on
Hugging Face, then ``huggingface-cli login`` or set ``HF_TOKEN``)::

    uv run --extra reference python tools/validate_hf.py               # test_audio.wav
    uv run --extra reference python tools/validate_hf.py --audio a.wav --mlx-model ainergiz/medasr-mlx-fp16

By default the MLX side loads the *same* original safetensors directly, so this
also exercises the v2 loader's PyTorch-layout path.
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="google/medasr")
    ap.add_argument("--mlx-model", default=None, help="defaults to --hf-model")
    ap.add_argument("--audio", default=None, help="defaults to test_audio.wav from --hf-model")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = ap.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCTC, AutoProcessor

    import medasr_mlx
    from medasr_mlx.audio import load_audio
    from medasr_mlx.evaluation import wer

    audio_path = args.audio or hf_hub_download(args.hf_model, "test_audio.wav")
    wave = load_audio(audio_path)
    processor = AutoProcessor.from_pretrained(args.hf_model)
    hf = AutoModelForCTC.from_pretrained(args.hf_model).eval()
    inputs = processor(wave, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    with torch.no_grad():
        ref_logits = hf(**inputs).logits[0].float().numpy()
    ref_text = processor.batch_decode(ref_logits.argmax(-1)[None], skip_special_tokens=True)[0]

    asr = medasr_mlx.load(args.mlx_model or args.hf_model, dtype=args.dtype)
    feats = mx.array(inputs["input_features"].numpy())
    logits, _ = asr.model(feats)
    ours = np.asarray(logits.astype(mx.float32))[0]
    text = asr.transcribe(wave, window_s=None).text

    agree = float((ours.argmax(-1) == ref_logits.argmax(-1)).mean())
    print(f"audio:              {audio_path} ({wave.size / 16000:.1f}s)")
    print(f"frames:             {ours.shape[0]} (reference {ref_logits.shape[0]})")
    print(f"argmax agreement:   {agree:.6f}")
    print(f"max |logit diff|:   {np.abs(ours - ref_logits).max():.5f}")
    print(f"text WER vs ref:    {wer([ref_text], [text], normalize=False):.4f}")
    print(f"reference: {ref_text}")
    print(f"mlx:       {text}")
    tpf = asr.transcribe(wave).text
    if tpf != text:
        print(f"mlx (20 s TPF windows): {tpf}")


if __name__ == "__main__":
    main()
