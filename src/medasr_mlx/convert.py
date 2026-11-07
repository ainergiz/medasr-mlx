"""Convert ``google/medasr`` (or a v1 MedASR-MLX checkpoint) to the v2 layout.

No PyTorch needed: the original safetensors are read directly by MLX. Output
is ``model.safetensors`` + ``config.json`` + ``tokenizer.json``, optionally
quantized. Saved weights are never fp16-rescaled (that happens at load time).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .weights import find_tokenizer_file, load_model, quantize_model, save_model

_DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}


def keep_io_layers(path: str, module: nn.Module) -> bool:
    """Quantization filter that keeps the subsampler and CTC head in floating point."""
    return not (path.startswith("encoder.subsampler") or path == "ctc_head")


def convert(
    source: str | Path,
    output: str | Path,
    *,
    dtype: str = "float16",
    bits: int | None = None,
    group_size: int = 64,
    mode: str = "affine",
    quantize_io: bool = False,
    revision: str | None = None,
) -> Path:
    model, model_dir = load_model(source, dtype=mx.float32, revision=revision)
    if bits is not None:
        quantize_model(
            model,
            bits=bits,
            group_size=group_size,
            mode=mode,
            include=None if quantize_io else keep_io_layers,
        )
    target = _DTYPES[dtype]
    params = [
        (k, v.astype(target) if mx.issubdtype(v.dtype, mx.floating) else v)
        for k, v in tree_flatten(model.parameters())
    ]
    model.update(tree_unflatten(params))
    mx.eval(model.parameters())
    extra = {"quantized_io_layers": quantize_io} if bits is not None else {}
    return save_model(
        model,
        output,
        tokenizer_file=find_tokenizer_file(model_dir),
        source=str(source),
        extra_metadata=extra,
    )
