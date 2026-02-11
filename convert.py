#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from dotenv import load_dotenv
from huggingface_hub.utils import HfHubHTTPError
from transformers import AutoModelForCTC, AutoProcessor


def load_local_env() -> None:
    for path in [".env.local", "env.local", "env.lcoal", ".env"]:
        dotenv_path = Path(path)
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)


def resolve_hf_token() -> str | None:
    keys = [
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
    ]
    for key in keys:
        value = os.getenv(key)
        if value:
            os.environ.setdefault("HF_TOKEN", value)
            return value
    return None


def parse_dtype(dtype_str: str) -> tuple[mx.Dtype, np.dtype]:
    normalized = dtype_str.lower()
    if normalized in ("float32", "fp32"):
        return mx.float32, np.float32
    if normalized in ("float16", "fp16"):
        return mx.float16, np.float16
    if normalized in ("bfloat16", "bf16"):
        return mx.bfloat16, np.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _infer_attention_info(model: Any) -> dict[str, Any]:
    for layer in getattr(model.encoder, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        q_proj = getattr(attn, "q_proj", None)
        if q_proj is None:
            continue
        hidden_size = int(q_proj.out_features)
        head_dim = int(getattr(attn, "head_dim", 0)) or None
        num_heads = getattr(attn, "num_heads", None)
        if num_heads is None and head_dim:
            num_heads = hidden_size // head_dim
        if head_dim is None and num_heads:
            head_dim = hidden_size // int(num_heads)
        return {
            "hidden_size": hidden_size,
            "num_heads": int(num_heads) if num_heads is not None else None,
            "head_dim": head_dim,
            "is_causal": bool(getattr(attn, "is_causal", False)),
            "scaling": float(getattr(attn, "scaling", 0.0))
            if getattr(attn, "scaling", None) is not None
            else None,
        }
    return {}


def _infer_conv_cache_info(model: Any) -> dict[str, Any]:
    depthwise = []
    for layer_idx, layer in enumerate(getattr(model.encoder, "layers", [])):
        conv = getattr(layer, "conv", None)
        if conv is None or not hasattr(conv, "depthwise_conv"):
            continue
        depthwise_conv = conv.depthwise_conv
        kernel_size = int(depthwise_conv.kernel_size[0])
        padding = depthwise_conv.padding
        if isinstance(padding, str):
            padding_repr: str | int | list[int] = padding
        elif isinstance(padding, tuple):
            padding_repr = [int(v) for v in padding]
        else:
            padding_repr = int(padding)
        depthwise.append(
            {
                "layer": layer_idx,
                "name": f"encoder.layers.{layer_idx}.conv.depthwise_conv",
                "kernel_size": kernel_size,
                "cache_size_frames": kernel_size - 1,
                "padding": padding_repr,
                "appears_causal": padding_repr == 0,
            }
        )
    return {
        "num_layers": len(getattr(model.encoder, "layers", [])),
        "depthwise": depthwise,
    }


def _infer_subsampling_info(model: Any) -> dict[str, Any]:
    subsampler = getattr(model.encoder, "subsampler", None)
    if subsampler is None:
        return {}

    conv_names = ["conv_0", "conv_1"]
    convs = []
    factor = 1
    for name in conv_names:
        module = getattr(subsampler, name, None)
        if module is None:
            continue
        stride = int(module.stride[0] if isinstance(module.stride, tuple) else module.stride)
        factor *= stride
        convs.append(
            {
                "name": f"encoder.subsampler.{name}",
                "stride": stride,
                "kernel_size": int(module.kernel_size[0]),
            }
        )
    return {"convs": convs, "estimated_factor": factor if convs else None}


def _infer_positional_info(model: Any) -> dict[str, Any]:
    rotary = getattr(model.encoder, "rotary_emb", None)
    if rotary is None:
        return {}
    info: dict[str, Any] = {"type": rotary.__class__.__name__}
    if hasattr(rotary, "max_seq_len_cached"):
        info["max_seq_len_cached"] = int(rotary.max_seq_len_cached)
    if hasattr(rotary, "inv_freq") and hasattr(rotary.inv_freq, "shape"):
        info["inv_freq_shape"] = [int(v) for v in rotary.inv_freq.shape]
    if hasattr(rotary, "base"):
        info["base"] = rotary.base
    return info


def build_config(model: Any, processor: Any, model_id: str, dtype_str: str) -> dict[str, Any]:
    config = model.config.to_dict()
    config["_conversion"] = {
        "source_model_id": model_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "weights_dtype": dtype_str,
        "converter": "convert.py",
        "weight_layout_notes": {
            "linear": "unchanged (out_features, in_features)",
            "conv1d": "transposed from torch [out, in, kernel] to mlx [out, kernel, in]",
            "skipped": ["*.num_batches_tracked"],
        },
    }
    config["_cache"] = {
        "attention": _infer_attention_info(model),
        "convolution": _infer_conv_cache_info(model),
        "subsampling": _infer_subsampling_info(model),
        "positional_encoding": _infer_positional_info(model),
    }

    feature_extractor = getattr(processor, "feature_extractor", None)
    if feature_extractor is not None and hasattr(feature_extractor, "to_dict"):
        config["_audio"] = feature_extractor.to_dict()
    return config


def should_skip(name: str) -> bool:
    return name.endswith("num_batches_tracked")


def needs_conv_transpose(name: str, array: np.ndarray) -> bool:
    return name.endswith(".weight") and array.ndim == 3


def convert_state_dict_to_mlx(
    state_dict: dict[str, Any], mx_dtype: mx.Dtype, np_dtype: np.dtype
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    converted: dict[str, mx.array] = {}
    skipped: list[str] = []
    transposed: list[str] = []

    for name, tensor in state_dict.items():
        if should_skip(name):
            skipped.append(name)
            continue

        arr = tensor.detach().cpu().numpy()
        if needs_conv_transpose(name, arr):
            arr = np.transpose(arr, (0, 2, 1))
            transposed.append(name)

        # Keep non-floating arrays as-is. Cast float arrays to target dtype.
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np_dtype, copy=False)

        if np.issubdtype(arr.dtype, np.floating):
            converted[name] = mx.array(arr, dtype=mx_dtype)
        else:
            converted[name] = mx.array(arr)

    report = {
        "num_input_tensors": len(state_dict),
        "num_output_tensors": len(converted),
        "num_skipped_tensors": len(skipped),
        "skipped_tensors": skipped,
        "num_transposed_conv_tensors": len(transposed),
        "transposed_conv_tensors": transposed,
    }
    return converted, report


def write_outputs(
    output_dir: Path,
    weights: dict[str, mx.array],
    config: dict[str, Any],
    report: dict[str, Any],
    processor: Any,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mx.savez(str(output_dir / "weights.npz"), **weights)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    processor.save_pretrained(str(output_dir / "processor"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face MedASR weights to MLX format."
    )
    parser.add_argument("--model-id", default="google/medasr", help="Hugging Face model ID.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/medasr-mlx",
        help="Directory to write MLX weights/config artifacts.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "fp16", "float32", "fp32", "bfloat16", "bf16"],
        help="Output weight dtype.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_local_env()
    token = resolve_hf_token()
    mx_dtype, np_dtype = parse_dtype(args.dtype)

    print(f"[convert] Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, token=token)

    print(f"[convert] Loading model: {args.model_id}")
    model = AutoModelForCTC.from_pretrained(args.model_id, token=token)
    state_dict = model.state_dict()

    print(f"[convert] Converting tensors to MLX ({args.dtype})")
    weights, report = convert_state_dict_to_mlx(state_dict, mx_dtype=mx_dtype, np_dtype=np_dtype)

    config = build_config(
        model=model, processor=processor, model_id=args.model_id, dtype_str=args.dtype
    )
    output_dir = Path(args.output_dir)
    write_outputs(
        output_dir=output_dir, weights=weights, config=config, report=report, processor=processor
    )

    print("[convert] Done")
    print(
        f"[convert] Tensors: {report['num_output_tensors']} (skipped {report['num_skipped_tensors']})"
    )
    print(f"[convert] Output: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except HfHubHTTPError as exc:
        print("[convert] Hugging Face access error.")
        print(
            "[convert] Ensure you accepted terms for google/medasr and set HF_TOKEN in .env.local."
        )
        print(f"[convert] Details: {exc}")
        raise SystemExit(2) from exc
