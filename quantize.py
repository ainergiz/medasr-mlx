#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.nn as nn

from model import load_mlx_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize MedASR MLX checkpoints (int8/int4).")
    parser.add_argument(
        "--input-model-dir",
        default="artifacts/medasr-mlx-fp16",
        help="Source MLX model directory containing weights.npz/config.json/processor/.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output model directory. Defaults to artifacts/medasr-mlx-int{bits}.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=8,
        help="Quantization bit width.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        choices=[32, 64, 128],
        default=64,
        help="Quantization group size (mlx affine mode supports 32/64/128).",
    )
    parser.add_argument(
        "--mode",
        choices=["affine"],
        default="affine",
        help="Quantization mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--skip-verify-load",
        action="store_true",
        help="Skip loading the quantized checkpoint for post-write verification.",
    )
    return parser.parse_args()


def _ensure_dir_state(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already exists and is not empty. "
            "Use --overwrite to reuse it or choose a new --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _build_quantized_config(
    source_config: dict[str, Any],
    *,
    input_model_dir: Path,
    bits: int,
    group_size: int,
    mode: str,
) -> dict[str, Any]:
    cfg = deepcopy(source_config)
    conversion = cfg.setdefault("_conversion", {})
    source_dtype = conversion.get("weights_dtype")
    if source_dtype is not None:
        conversion.setdefault("source_weights_dtype", source_dtype)
    conversion["weights_dtype"] = f"{mode}-int{bits}"

    cfg["_quantization"] = {
        "enabled": True,
        "bits": bits,
        "group_size": group_size,
        "mode": mode,
        "target_modules": "mlx.nn.quantize default predicate (Linear/Embedding layers)",
        "source_model_dir": str(input_model_dir),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "notes": [
            "Convolution layers remain non-quantized with this path.",
            "Model loader applies nn.quantize() before loading quantized weights.",
        ],
    }
    return cfg


def _copy_processor(input_dir: Path, output_dir: Path) -> None:
    src = input_dir / "processor"
    if not src.exists():
        return
    dst = output_dir / "processor"
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _write_report(
    output_dir: Path,
    *,
    input_model_dir: Path,
    bits: int,
    group_size: int,
    mode: str,
    quantization_time_s: float,
    source_size_mb: float,
    output_size_mb: float,
) -> None:
    size_reduction_pct = 0.0
    compression_ratio = 0.0
    if source_size_mb > 0:
        size_reduction_pct = (1.0 - (output_size_mb / source_size_mb)) * 100.0
        compression_ratio = source_size_mb / output_size_mb if output_size_mb > 0 else 0.0

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_model_dir": str(input_model_dir),
        "output_model_dir": str(output_dir),
        "quantization": {
            "bits": bits,
            "group_size": group_size,
            "mode": mode,
            "target_modules": "mlx.nn.quantize default predicate (Linear/Embedding layers)",
        },
        "timing": {
            "quantization_time_s": round(quantization_time_s, 4),
        },
        "size_mb": {
            "source_weights": round(source_size_mb, 3),
            "output_weights": round(output_size_mb, 3),
            "compression_ratio_x": round(compression_ratio, 3),
            "reduction_percent": round(size_reduction_pct, 2),
        },
    }
    (output_dir / "quantization_report.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_model_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(f"artifacts/medasr-mlx-int{args.bits}")
    )
    input_config_path = input_dir / "config.json"
    input_weights_path = input_dir / "weights.npz"

    if not input_config_path.exists():
        raise FileNotFoundError(f"Missing input config: {input_config_path}")
    if not input_weights_path.exists():
        raise FileNotFoundError(f"Missing input weights: {input_weights_path}")

    _ensure_dir_state(output_dir, overwrite=args.overwrite)

    print(f"[quantize] Loading source model from {input_dir}")
    model = load_mlx_model(input_dir)

    print(
        "[quantize] Applying quantization "
        f"(bits={args.bits}, group_size={args.group_size}, mode={args.mode})"
    )
    t0 = time.perf_counter()
    nn.quantize(model, bits=args.bits, group_size=args.group_size, mode=args.mode)
    quantization_time_s = time.perf_counter() - t0

    output_weights_path = output_dir / "weights.npz"
    print(f"[quantize] Saving quantized weights to {output_weights_path}")
    model.save_weights(str(output_weights_path))

    source_config = _load_json(input_config_path)
    quantized_config = _build_quantized_config(
        source_config,
        input_model_dir=input_dir,
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
    )
    (output_dir / "config.json").write_text(json.dumps(quantized_config, indent=2) + "\n")
    _copy_processor(input_dir, output_dir)

    source_size_mb = input_weights_path.stat().st_size / (1024 * 1024)
    output_size_mb = output_weights_path.stat().st_size / (1024 * 1024)
    _write_report(
        output_dir,
        input_model_dir=input_dir,
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
        quantization_time_s=quantization_time_s,
        source_size_mb=source_size_mb,
        output_size_mb=output_size_mb,
    )

    if not args.skip_verify_load:
        print("[quantize] Verifying quantized checkpoint load...")
        _ = load_mlx_model(output_dir)

    print("[quantize] Done.")
    print(
        "[quantize] Size: "
        f"{source_size_mb:.1f} MB -> {output_size_mb:.1f} MB "
        f"({(1.0 - output_size_mb / source_size_mb) * 100.0:.1f}% reduction)"
    )
    print(f"[quantize] Output dir: {output_dir}")


if __name__ == "__main__":
    main()
