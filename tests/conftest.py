"""Shared fixtures: a tiny synthetic tokenizer and random tiny MedASR checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from medasr_mlx.config import EncoderConfig, ModelConfig
from medasr_mlx.tokenizer import CTCTokenizer

PIECES = [
    "<epsilon>", "<s>", "</s>", "<unk>", "▁", "▁the", "▁pa", "tient", "▁has", "▁fever",
    "▁a", "b", "c", "▁mg", ".", ",", "▁met", "o", "pro", "lol", "▁dolo", "▁650",
]  # fmt: skip
TINY_ENCODER = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 64,
    "conv_kernel_size": 8,
    "subsampling_conv_channels": 16,
    "num_mel_bins": 128,
}


def tokenizer_json(pieces: list[str] = PIECES) -> dict:
    vocab = [[p, 0.0 if p.startswith("<") else -float(i) / 4] for i, p in enumerate(pieces)]
    return {
        "version": "1.0",
        "added_tokens": [
            {"id": i, "content": p, "special": True} for i, p in enumerate(pieces[:4])
        ],
        "normalizer": None,
        "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"},
        "model": {"type": "Unigram", "unk_id": 3, "vocab": vocab, "byte_fallback": False},
    }


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory) -> CTCTokenizer:
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    path.write_text(json.dumps(tokenizer_json()), encoding="utf-8")
    return CTCTokenizer.from_file(path)


def tiny_config_dict(vocab_size: int = len(PIECES)) -> dict:
    return {"model_type": "lasr_ctc", "vocab_size": vocab_size, "pad_token_id": 0,
            "encoder_config": dict(TINY_ENCODER)}  # fmt: skip


def random_hf_weights(cfg: ModelConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """Random weights in the *PyTorch* ``google/medasr`` layout (incl. BatchNorm)."""
    rng = np.random.default_rng(seed)
    e: EncoderConfig = cfg.encoder
    h, k = e.hidden_size, e.subsampling_conv_kernel_size

    def w(*shape, scale=None):
        fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
        return (rng.standard_normal(shape) * (scale or 1 / np.sqrt(fan_in))).astype(np.float32)

    out = {
        "encoder.subsampler.dense_0.weight": w(h, e.num_mel_bins),
        "encoder.subsampler.dense_0.bias": w(h, scale=0.1),
        "encoder.subsampler.conv_0.weight": w(h, h, k),
        "encoder.subsampler.conv_0.bias": w(h, scale=0.1),
        "encoder.subsampler.conv_1.weight": w(e.subsampling_conv_channels, h, k),
        "encoder.subsampler.conv_1.bias": w(e.subsampling_conv_channels, scale=0.1),
        "encoder.subsampler.dense_1.weight": w(h, e.subsampling_conv_channels),
        "encoder.subsampler.dense_1.bias": w(h, scale=0.1),
        "encoder.out_norm.weight": 1 + w(h, scale=0.1),
        "ctc_head.weight": w(cfg.vocab_size, h, 1),
        "ctc_head.bias": w(cfg.vocab_size, scale=0.1),
    }
    for i in range(e.num_hidden_layers):
        p = f"encoder.layers.{i}"
        for ff in ("feed_forward1", "feed_forward2"):
            out[f"{p}.{ff}.linear1.weight"] = w(e.intermediate_size, h)
            out[f"{p}.{ff}.linear2.weight"] = w(h, e.intermediate_size)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[f"{p}.self_attn.{proj}.weight"] = w(h, h)
        out[f"{p}.conv.pointwise_conv1.weight"] = w(2 * h, h, 1)
        out[f"{p}.conv.depthwise_conv.weight"] = w(h, 1, e.conv_kernel_size)
        out[f"{p}.conv.norm.weight"] = 1 + w(h, scale=0.1)
        out[f"{p}.conv.norm.bias"] = w(h, scale=0.1)
        out[f"{p}.conv.norm.running_mean"] = w(h, scale=0.1)
        out[f"{p}.conv.norm.running_var"] = (0.5 + rng.random(h)).astype(np.float32)
        out[f"{p}.conv.norm.num_batches_tracked"] = np.array(7, dtype=np.int64)
        out[f"{p}.conv.pointwise_conv2.weight"] = w(h, h, 1)
        for n in (
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ):
            out[f"{p}.{n}.weight"] = 1 + w(h, scale=0.1)
    return out


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """A google/medasr-style directory (PyTorch layout safetensors) with random weights."""
    d = tmp_path_factory.mktemp("tiny-medasr")
    cfg_dict = tiny_config_dict()
    (d / "config.json").write_text(json.dumps(cfg_dict))
    (d / "tokenizer.json").write_text(json.dumps(tokenizer_json()), encoding="utf-8")
    weights = random_hf_weights(ModelConfig.from_dict(cfg_dict))
    mx.save_safetensors(str(d / "model.safetensors"), {k: mx.array(v) for k, v in weights.items()})
    return d


@pytest.fixture
def speechlike() -> np.ndarray:
    """3.3 s of deterministic band-limited noise bursts (no real speech needed)."""
    r = np.random.default_rng(1)
    t = np.arange(int(3.3 * 16000)) / 16000
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t)
    return (0.1 * env * r.standard_normal(t.size) + 0.05 * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32
    )


def real_model_dir() -> str | None:
    """Real MedASR weights for slow tests: $MEDASR_TEST_MODEL (dir or repo id)."""
    return os.environ.get("MEDASR_TEST_MODEL")
