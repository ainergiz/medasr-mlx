"""Loading, sanitising and saving MedASR weights.

Three on-disk layouts are understood and auto-detected per tensor:

1. ``google/medasr`` - PyTorch safetensors (conv weights ``[out, in, k]``,
   BatchNorm buffers, ``num_batches_tracked``), fp32.
2. MedASR-MLX v1 - ``weights.npz`` with MLX conv layout ``[out, k, in]`` and
   unfused BatchNorm, optionally int8/int4 quantized Linear layers.
3. MedASR-MLX v2 - ``model.safetensors`` holding the runtime layout written by
   :func:`save_model` (pointwise convs as Linear, BatchNorm folded).
"""

from __future__ import annotations

import json
import math
import shutil
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import ModelConfig, QuantizationConfig
from .model import MedASRModel

FORMAT_VERSION = 2
DEFAULT_MODEL = "ainergiz/medasr-mlx-fp16"
_WEIGHT_FILES = ("model.safetensors", "weights.safetensors", "weights.npz")
_TOKENIZER_FILES = ("tokenizer.json", "processor/tokenizer.json")


# --------------------------------------------------------------------------- paths
def resolve_model_path(path_or_repo: str | Path, revision: str | None = None) -> Path:
    """Return a local directory for ``path_or_repo``, downloading from the Hub if needed."""
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return path
    repo_id = str(path_or_repo)
    from huggingface_hub import list_repo_files, snapshot_download

    files = set(list_repo_files(repo_id, revision=revision))
    weight = next((f for f in _WEIGHT_FILES if f in files), None)
    if weight is None:
        raise FileNotFoundError(f"{repo_id} has none of {_WEIGHT_FILES}")
    patterns = ["config.json", weight, *[t for t in _TOKENIZER_FILES if t in files]]
    return Path(snapshot_download(repo_id, revision=revision, allow_patterns=patterns))


def find_weight_file(model_dir: Path) -> Path:
    for name in _WEIGHT_FILES:
        if (model_dir / name).exists():
            return model_dir / name
    raise FileNotFoundError(f"No weights ({', '.join(_WEIGHT_FILES)}) in {model_dir}")


def find_tokenizer_file(model_dir: Path) -> Path:
    for name in _TOKENIZER_FILES:
        if (model_dir / name).exists():
            return model_dir / name
    raise FileNotFoundError(f"No tokenizer.json in {model_dir}")


def default_dtype() -> mx.Dtype:
    """fp16 on GPU; fp32 on CPU, where MLX emulates half precision slowly."""
    return mx.float16 if mx.default_device() == mx.gpu else mx.float32


# ----------------------------------------------------------------------- sanitize
def _fold_batchnorm(weights: dict[str, mx.array], prefix: str, eps: float) -> None:
    """Fold ``{prefix}.norm.*`` into ``{prefix}.depthwise_conv`` in place (MLX layout)."""
    gamma = weights.pop(f"{prefix}.norm.weight").astype(mx.float32)
    beta = weights.pop(f"{prefix}.norm.bias").astype(mx.float32)
    mean = weights.pop(f"{prefix}.norm.running_mean").astype(mx.float32)
    var = weights.pop(f"{prefix}.norm.running_var").astype(mx.float32)
    weights.pop(f"{prefix}.norm.num_batches_tracked", None)
    scale = gamma * mx.rsqrt(var + eps)
    w_key, b_key = f"{prefix}.depthwise_conv.weight", f"{prefix}.depthwise_conv.bias"
    w = weights[w_key].astype(mx.float32)
    bias = weights[b_key].astype(mx.float32) if b_key in weights else mx.zeros_like(mean)
    weights[w_key] = w * scale[:, None, None]
    weights[b_key] = (bias - mean) * scale + beta


def sanitize_weights(
    weights: dict[str, mx.array],
    expected_shapes: dict[str, tuple[int, ...]],
    *,
    batch_norm_eps: float = 1e-5,
) -> dict[str, mx.array]:
    """Map any supported checkpoint layout onto the runtime parameter layout."""
    weights = {k: v for k, v in weights.items() if not k.endswith("num_batches_tracked")}
    # 1) conv layouts: torch [out, in, k] -> mlx [out, k, in]; pointwise convs -> Linear.
    for key in list(weights):
        if key not in expected_shapes:
            continue
        value, target = weights[key], expected_shapes[key]
        if tuple(value.shape) == target:
            continue
        if value.ndim == 3 and len(target) == 3 and tuple(value.transpose(0, 2, 1).shape) == target:
            weights[key] = value.transpose(0, 2, 1)
        elif value.ndim == 3 and len(target) == 2 and 1 in value.shape[1:]:
            axis = 1 if value.shape[1] == 1 else 2
            weights[key] = value.squeeze(axis)
    # 2) BatchNorm folding (the depthwise conv is now in MLX layout).
    for key in [k for k in weights if k.endswith(".norm.running_mean")]:
        prefix = key[: -len(".norm.running_mean")]
        dw = weights.get(f"{prefix}.depthwise_conv.weight")
        if dw is not None and dw.ndim == 3 and dw.shape[-1] != 1:
            weights[f"{prefix}.depthwise_conv.weight"] = dw.transpose(0, 2, 1)
        _fold_batchnorm(weights, prefix, batch_norm_eps)
    # 3) validate
    missing = sorted(set(expected_shapes) - set(weights))
    unexpected = sorted(set(weights) - set(expected_shapes))
    bad = [
        (k, tuple(weights[k].shape), expected_shapes[k])
        for k in expected_shapes
        if k in weights and tuple(weights[k].shape) != expected_shapes[k]
    ]
    if missing or unexpected or bad:
        raise ValueError(
            "Checkpoint does not match the MedASR architecture: "
            f"missing={missing[:5]} unexpected={unexpected[:5]} shape_mismatch={bad[:5]}"
        )
    return weights


FP16_RESIDUAL_SCALE = 2.0**-8
FP16_SUBSAMPLER_SCALE = 2.0**-3


def stabilize_for_fp16(
    weights: dict[str, mx.array],
    num_layers: int,
    *,
    alpha: float = FP16_RESIDUAL_SCALE,
    beta: float = FP16_SUBSAMPLER_SCALE,
    quant_mode: str = "affine",
) -> dict[str, mx.array]:
    """Rescale weights so float16 activations cannot overflow. Exact up to LayerNorm eps.

    MedASR's residual stream reaches ~2e6 inside blocks (fp16 max is 65504).
    Every sub-layer reads a LayerNorm of the stream and every block ends in a
    LayerNorm, so multiplying the stream *and* each sub-layer's output
    projection by ``alpha`` leaves all normalised quantities unchanged. The
    subsampler's ReLU chain is rescaled the same way (``relu(b*z) = b*relu(z)``).
    Powers of two keep the rescaling bit-exact for the weights themselves.

    Quantized layers are rescaled through their scales: affine scales/biases are
    multiplied, and MX (``mxfp4``/``mxfp8``) power-of-two E8M0 scale bytes are
    shifted by ``log2(factor)``. ``nvfp4`` (E4M3 scales) is not supported.
    """
    w = dict(weights)

    def scale(key: str, factor: float) -> None:
        base = key.rsplit(".", 1)[0]
        if key.endswith(".weight") and f"{base}.scales" in w:
            scales = w[f"{base}.scales"]
            if mx.issubdtype(scales.dtype, mx.floating):
                # affine: W = scales * q + biases -> scale scales and biases, keep q.
                w[f"{base}.scales"] = scales * factor
                if f"{base}.biases" in w:
                    w[f"{base}.biases"] = w[f"{base}.biases"] * factor
            elif quant_mode in ("mxfp4", "mxfp8"):
                shift = round(math.log2(factor))
                exps = scales.astype(mx.int32) + shift
                if int(exps.min()) < 1:
                    raise ValueError(f"{base}: MX scale exponent underflow during fp16 rescaling")
                w[f"{base}.scales"] = exps.astype(mx.uint8)
            else:
                raise ValueError(
                    f"{base}: float16 compute is not supported for {quant_mode} checkpoints; "
                    "use dtype=bfloat16 or float32."
                )
        elif key in w:
            w[key] = w[key] * factor

    sub = "encoder.subsampler"
    scale(f"{sub}.conv_1.weight", beta)
    scale(f"{sub}.conv_1.bias", beta)
    scale(f"{sub}.dense_1.weight", alpha / beta)
    scale(f"{sub}.dense_1.bias", alpha)
    for i in range(num_layers):
        p = f"encoder.layers.{i}"
        for name in (
            "feed_forward1.linear2",
            "self_attn.o_proj",
            "conv.pointwise_conv2",
            "feed_forward2.linear2",
        ):
            scale(f"{p}.{name}.weight", alpha)
            scale(f"{p}.{name}.bias", alpha)
        if i < num_layers - 1:  # feeds the next block's (scaled) residual stream
            scale(f"{p}.norm_out.weight", alpha)
    return w


def _cast_floats(weights: dict[str, mx.array], dtype: mx.Dtype) -> dict[str, mx.array]:
    return {
        k: (v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v)
        for k, v in weights.items()
    }


def quantization_predicate(weights: dict[str, Any]) -> Callable[[str, nn.Module], bool]:
    """Quantize exactly the layers whose packed weights are present (mlx-lm convention)."""
    return lambda path, module: hasattr(module, "to_quantized") and f"{path}.scales" in weights


# --------------------------------------------------------------------------- load
def load_config(model_dir: Path) -> tuple[ModelConfig, dict[str, Any]]:
    raw = json.loads((model_dir / "config.json").read_text())
    return ModelConfig.from_dict(raw), raw


def build_model(
    config: ModelConfig,
    weights: dict[str, mx.array],
    *,
    dtype: mx.Dtype | None = None,
) -> MedASRModel:
    """Instantiate the runtime model and load (raw, unsanitised) ``weights`` into it."""
    dtype = dtype or default_dtype()
    q = config.quantization
    if dtype == mx.float16 and q is not None and q.mode == "nvfp4":
        warnings.warn("nvfp4 scales cannot be fp16-rescaled; computing in bfloat16", stacklevel=2)
        dtype = mx.bfloat16
    alpha = FP16_RESIDUAL_SCALE if dtype == mx.float16 else 1.0
    model = MedASRModel(config, residual_scale=alpha)
    if q is not None:
        nn.quantize(
            model,
            group_size=q.group_size,
            bits=q.bits,
            mode=q.mode,
            class_predicate=quantization_predicate(weights),
        )
    expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    weights = sanitize_weights(weights, expected, batch_norm_eps=config.encoder.batch_norm_eps)
    if alpha != 1.0:
        weights = stabilize_for_fp16(
            weights,
            config.encoder.num_hidden_layers,
            alpha=alpha,
            quant_mode=q.mode if q is not None else "affine",
        )
    weights = _cast_floats(weights, dtype)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def load_model(
    path_or_repo: str | Path = DEFAULT_MODEL,
    *,
    dtype: mx.Dtype | None = None,
    revision: str | None = None,
) -> tuple[MedASRModel, Path]:
    """Load a MedASR checkpoint (local dir or Hub repo) and return ``(model, model_dir)``."""
    model_dir = resolve_model_path(path_or_repo, revision=revision)
    config, _ = load_config(model_dir)
    weights = mx.load(str(find_weight_file(model_dir)))
    assert isinstance(weights, dict)
    return build_model(config, weights, dtype=dtype), model_dir


# --------------------------------------------------------------------------- save
def save_model(
    model: MedASRModel,
    out_dir: str | Path,
    *,
    tokenizer_file: str | Path,
    source: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write ``model.safetensors`` + ``config.json`` + ``tokenizer.json`` (v2 layout)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out_dir / "model.safetensors"), weights, metadata={"format": "mlx"})
    config = model.config.to_dict()
    config["medasr_mlx"] = {
        "format_version": FORMAT_VERSION,
        "layout": "runtime (pointwise convs as Linear, BatchNorm folded into depthwise conv)",
        "source": source,
        "dtype": str(model.dtype).replace("mlx.core.", ""),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra_metadata or {}),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copyfile(tokenizer_file, out_dir / "tokenizer.json")
    return out_dir


def quantize_model(
    model: MedASRModel,
    *,
    bits: int = 8,
    group_size: int = 64,
    mode: str = "affine",
    include: Callable[[str, nn.Module], bool] | None = None,
) -> MedASRModel:
    """Quantize Linear layers in place (convolutions stay in floating point).

    ``include(path, module)`` can further restrict which Linear layers are
    quantized (e.g. keep the CTC head in fp16).
    """

    def predicate(path: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if module.weight.shape[-1] % group_size != 0:
            return False
        return include(path, module) if include is not None else True

    nn.quantize(model, group_size=group_size, bits=bits, mode=mode, class_predicate=predicate)
    object.__setattr__(
        model, "config", _replace_quant(model.config, QuantizationConfig(bits, group_size, mode))
    )
    mx.eval(model.parameters())
    return model


def _replace_quant(config: ModelConfig, q: QuantizationConfig) -> ModelConfig:
    from dataclasses import replace

    return replace(config, quantization=q)
