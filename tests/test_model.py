import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from medasr_mlx.config import ModelConfig
from medasr_mlx.model import MedASRModel
from medasr_mlx.weights import (
    FP16_RESIDUAL_SCALE,
    build_model,
    load_model,
    quantize_model,
    sanitize_weights,
    save_model,
    stabilize_for_fp16,
)

from .conftest import random_hf_weights, tiny_config_dict


def _tiny(dtype=mx.float32, seed=0):
    cfg = ModelConfig.from_dict(tiny_config_dict())
    w = {k: mx.array(v) for k, v in random_hf_weights(cfg, seed).items()}
    return build_model(cfg, w, dtype=dtype), cfg


def _feats(t=203, b=1, seed=0):
    return mx.array(np.random.default_rng(seed).standard_normal((b, t, 128)).astype(np.float32))


def test_output_shape_and_lengths():
    model, cfg = _tiny()
    logits, mask = model(_feats(203))
    assert logits.shape == (1, cfg.encoder.num_output_frames(203), cfg.vocab_size)
    assert mask is None


def test_padding_mask_is_exact():
    """A padded batch must match running each item alone (attention + conv masking)."""
    model, _ = _tiny()
    a, b = _feats(203, seed=1)[0], _feats(150, seed=2)[0]
    batch = mx.zeros((2, 203, 128))
    batch[0] = a
    batch[1, :150] = b
    mask = mx.array(np.arange(203)[None, :] < np.array([[203], [150]]))
    logits, frame_mask = model(batch, mask)
    solo_a, _ = model(a[None])
    solo_b, _ = model(b[None])
    nb = solo_b.shape[1]
    np.testing.assert_allclose(np.asarray(logits[0]), np.asarray(solo_a[0]), atol=1e-4)
    np.testing.assert_allclose(np.asarray(logits[1, :nb]), np.asarray(solo_b[0]), atol=1e-4)
    assert int(np.asarray(frame_mask[1]).sum()) == nb


def test_fp16_stabilisation_is_exact_in_fp32():
    cfg = ModelConfig.from_dict(tiny_config_dict())
    raw = {k: mx.array(v) for k, v in random_hf_weights(cfg).items()}
    ref = build_model(cfg, dict(raw), dtype=mx.float32)
    w = dict(tree_flatten(ref.parameters()))
    m = MedASRModel(cfg, residual_scale=FP16_RESIDUAL_SCALE)
    m.load_weights(list(stabilize_for_fp16(w, cfg.encoder.num_hidden_layers).items()))
    x = _feats()
    np.testing.assert_allclose(np.asarray(m(x)[0]), np.asarray(ref(x)[0]), atol=1e-4)


def test_fp16_compute_close_to_fp32():
    m32, _ = _tiny(mx.float32)
    m16, _ = _tiny(mx.float16)
    x = _feats()
    a = np.asarray(m32(x)[0])
    b = np.asarray(m16(x)[0].astype(mx.float32))
    assert np.isfinite(b).all()
    assert (a.argmax(-1) == b.argmax(-1)).mean() > 0.95


def test_matches_reference_pytorch_implementation():
    """Architecture parity with transformers' LasrForCTC on a random tiny config."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg_dict = tiny_config_dict()
    hf_cfg = transformers.LasrCTCConfig(vocab_size=cfg_dict["vocab_size"],
                                        encoder_config=cfg_dict["encoder_config"])  # fmt: skip
    torch.manual_seed(0)
    hf = transformers.LasrForCTC(hf_cfg).eval()
    cfg = ModelConfig.from_dict(cfg_dict)
    weights = random_hf_weights(cfg, seed=3)
    hf.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()}, strict=True)
    x = np.random.default_rng(5).standard_normal((2, 211, 128)).astype(np.float32)
    mask = np.ones((2, 211), bool)
    mask[1, 170:] = False
    with torch.no_grad():
        ref = hf(
            input_features=torch.from_numpy(x), attention_mask=torch.from_numpy(mask)
        ).logits.numpy()
    model = build_model(cfg, {k: mx.array(v) for k, v in weights.items()}, dtype=mx.float32)
    ours, frame_mask = model(mx.array(x), mx.array(mask))
    valid = np.asarray(frame_mask)
    np.testing.assert_allclose(np.asarray(ours)[valid], ref[valid], atol=2e-4, rtol=1e-4)


def test_sanitize_handles_all_layouts():
    cfg = ModelConfig.from_dict(tiny_config_dict())
    hf = {k: mx.array(v) for k, v in random_hf_weights(cfg).items()}
    ref_model = build_model(cfg, dict(hf), dtype=mx.float32)
    x = _feats()
    ref = np.asarray(ref_model(x)[0])
    # v1 layout: MLX conv layout, BatchNorm unfused, pointwise convs as Conv1d.
    v1 = {
        k: (v.transpose(0, 2, 1) if v.ndim == 3 else v)
        for k, v in hf.items()
        if "num_batches" not in k
    }
    m1 = build_model(cfg, v1, dtype=mx.float32)
    np.testing.assert_allclose(np.asarray(m1(x)[0]), ref, atol=1e-5)
    # v2 layout: the runtime parameters themselves.
    v2 = dict(tree_flatten(ref_model.parameters()))
    m2 = build_model(cfg, v2, dtype=mx.float32)
    np.testing.assert_allclose(np.asarray(m2(x)[0]), ref, atol=1e-6)
    with pytest.raises(ValueError, match="does not match"):
        expected = {k: tuple(v.shape) for k, v in tree_flatten(ref_model.parameters())}
        sanitize_weights({"bogus.weight": mx.zeros((1,))}, expected)


def test_save_load_roundtrip_and_v1_npz(tmp_path, tiny_checkpoint):
    model, model_dir = load_model(tiny_checkpoint, dtype=mx.float32)
    x = _feats()
    ref = np.asarray(model(x)[0])
    out = save_model(model, tmp_path / "v2", tokenizer_file=model_dir / "tokenizer.json")
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["medasr_mlx"]["format_version"] == 2
    m2, _ = load_model(out, dtype=mx.float32)
    np.testing.assert_allclose(np.asarray(m2(x)[0]), ref, atol=1e-6)
    # v1-style directory: weights.npz + processor/tokenizer.json
    v1 = tmp_path / "v1"
    (v1 / "processor").mkdir(parents=True)
    (v1 / "config.json").write_text((tiny_checkpoint / "config.json").read_text())
    (v1 / "processor" / "tokenizer.json").write_text(
        (tiny_checkpoint / "tokenizer.json").read_text()
    )
    raw = mx.load(str(tiny_checkpoint / "model.safetensors"))
    mx.savez(
        str(v1 / "weights.npz"),
        **{k: (v.transpose(0, 2, 1) if v.ndim == 3 else v) for k, v in raw.items()},
    )
    m1, _ = load_model(v1, dtype=mx.float32)
    np.testing.assert_allclose(np.asarray(m1(x)[0]), ref, atol=1e-5)


@pytest.mark.parametrize("bits", [8, 4])
def test_quantize_save_load(tmp_path, tiny_checkpoint, bits):
    model, model_dir = load_model(tiny_checkpoint, dtype=mx.float32)
    x = _feats()
    ref = np.asarray(model(x)[0])
    quantize_model(model, bits=bits, group_size=32)
    q = np.asarray(model(x)[0])
    assert (q.argmax(-1) == ref.argmax(-1)).mean() > (0.9 if bits == 8 else 0.6)
    out = save_model(model, tmp_path / f"q{bits}", tokenizer_file=model_dir / "tokenizer.json")
    m2, _ = load_model(out, dtype=mx.float32)
    np.testing.assert_allclose(np.asarray(m2(x)[0]), q, atol=1e-5)
    m3, _ = load_model(out, dtype=mx.float16)  # fp16 stabilisation of quantized layers
    assert np.isfinite(np.asarray(m3(x)[0].astype(mx.float32))).all()


@pytest.mark.parametrize("mode", ["mxfp8", "mxfp4"])
def test_mx_quantized_fp16_rescaling_is_exact(tmp_path, tiny_checkpoint, mode):
    model, model_dir = load_model(tiny_checkpoint, dtype=mx.float32)
    quantize_model(model, bits=8 if mode == "mxfp8" else 4, group_size=32, mode=mode)
    out = save_model(model, tmp_path / mode, tokenizer_file=model_dir / "tokenizer.json")
    x = _feats()
    ref = np.asarray(load_model(out, dtype=mx.float32)[0](x)[0])
    # fp32 compute with the fp16 rescaling applied must be (numerically) identical
    cfg = ModelConfig.from_dict(json.loads((out / "config.json").read_text()))
    raw = mx.load(str(out / "model.safetensors"))
    m = MedASRModel(cfg, residual_scale=FP16_RESIDUAL_SCALE)
    nn.quantize(m, group_size=32, bits=cfg.quantization.bits, mode=mode,
                class_predicate=lambda p, mod: hasattr(mod, "to_quantized") and f"{p}.scales" in raw)  # fmt: skip
    m.load_weights(
        list(stabilize_for_fp16(raw, cfg.encoder.num_hidden_layers, quant_mode=mode).items())
    )
    np.testing.assert_allclose(np.asarray(m(x)[0]), ref, atol=1e-4)
    assert np.isfinite(
        np.asarray(load_model(out, dtype=mx.float16)[0](x)[0].astype(mx.float32))
    ).all()
