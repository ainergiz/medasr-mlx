"""Model and frontend configuration.

The config format is the Hugging Face ``LasrCTCConfig`` layout used by
``google/medasr`` (``config.json`` with a nested ``encoder_config``). MedASR-MLX
adds a few private keys on top of it when writing converted checkpoints:

* ``quantization`` - mlx-lm style ``{"bits", "group_size", "mode"}`` block.
* ``medasr_mlx`` - ``{"format_version": 2, ...}`` metadata about the checkpoint.

v1 checkpoints (``weights.npz`` + ``_quantization``) are still understood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SAMPLE_RATE = 16_000


def _get(d: dict[str, Any], key: str, default: Any) -> Any:
    value = d.get(key)
    return default if value is None else value


@dataclass(frozen=True)
class FrontendConfig:
    """Log-mel frontend matching ``transformers.LasrFeatureExtractor``."""

    sample_rate: int = SAMPLE_RATE
    num_mel_bins: int = 128
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    lower_edge_hz: float = 125.0
    upper_edge_hz: float = 7500.0
    log_floor: float = 1e-5

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FrontendConfig:
        d = d or {}
        return cls(
            sample_rate=int(_get(d, "sampling_rate", SAMPLE_RATE)),
            num_mel_bins=int(_get(d, "feature_size", 128)),
            n_fft=int(_get(d, "n_fft", 512)),
            win_length=int(_get(d, "win_length", 400)),
            hop_length=int(_get(d, "hop_length", 160)),
        )

    def num_frames(self, num_samples: int) -> int:
        """Number of mel frames for ``num_samples`` (no centering, no padding)."""
        if num_samples < self.win_length:
            return 0
        return (num_samples - self.win_length) // self.hop_length + 1


@dataclass(frozen=True)
class EncoderConfig:
    hidden_size: int = 512
    num_hidden_layers: int = 17
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    intermediate_size: int = 2048
    hidden_act: str = "silu"
    attention_bias: bool = False
    convolution_bias: bool = False
    conv_kernel_size: int = 32
    subsampling_conv_kernel_size: int = 5
    subsampling_conv_stride: int = 2
    subsampling_conv_channels: int = 256
    num_mel_bins: int = 128
    layer_norm_eps: float = 1e-6
    batch_norm_eps: float = 1e-5
    feed_forward_residual_weights: tuple[float, float] = (1.5, 0.5)
    conv_residual_weights: tuple[float, float] = (2.0, 1.0)
    rope_theta: float = 10_000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def subsampling_factor(self) -> int:
        return self.subsampling_conv_stride**2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EncoderConfig:
        rope = d.get("rope_parameters") or {}
        rope_type = rope.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(f"Unsupported rope_type {rope_type!r}; only 'default' RoPE is ported.")
        ff = _get(d, "feed_forward_residual_weights", [1.5, 0.5])
        conv = _get(d, "conv_residual_weights", [2.0, 1.0])
        heads = int(_get(d, "num_attention_heads", 8))
        return cls(
            hidden_size=int(_get(d, "hidden_size", 512)),
            num_hidden_layers=int(_get(d, "num_hidden_layers", 17)),
            num_attention_heads=heads,
            num_key_value_heads=int(_get(d, "num_key_value_heads", heads)),
            intermediate_size=int(_get(d, "intermediate_size", 2048)),
            hidden_act=str(_get(d, "hidden_act", "silu")),
            attention_bias=bool(_get(d, "attention_bias", False)),
            convolution_bias=bool(_get(d, "convolution_bias", False)),
            conv_kernel_size=int(_get(d, "conv_kernel_size", 32)),
            subsampling_conv_kernel_size=int(_get(d, "subsampling_conv_kernel_size", 5)),
            subsampling_conv_stride=int(_get(d, "subsampling_conv_stride", 2)),
            subsampling_conv_channels=int(_get(d, "subsampling_conv_channels", 256)),
            num_mel_bins=int(_get(d, "num_mel_bins", 128)),
            layer_norm_eps=float(_get(d, "layer_norm_eps", 1e-6)),
            feed_forward_residual_weights=(float(ff[0]), float(ff[1])),
            conv_residual_weights=(float(conv[0]), float(conv[1])),
            rope_theta=float(_get(rope, "rope_theta", 10_000.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "lasr_encoder",
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "attention_bias": self.attention_bias,
            "convolution_bias": self.convolution_bias,
            "conv_kernel_size": self.conv_kernel_size,
            "subsampling_conv_kernel_size": self.subsampling_conv_kernel_size,
            "subsampling_conv_stride": self.subsampling_conv_stride,
            "subsampling_conv_channels": self.subsampling_conv_channels,
            "num_mel_bins": self.num_mel_bins,
            "layer_norm_eps": self.layer_norm_eps,
            "feed_forward_residual_weights": list(self.feed_forward_residual_weights),
            "conv_residual_weights": list(self.conv_residual_weights),
            "rope_parameters": {"rope_theta": self.rope_theta, "rope_type": "default"},
        }

    def num_output_frames(self, num_mel_frames: int) -> int:
        """Encoder frames produced by the (VALID-padded) subsampling stack."""
        n = num_mel_frames
        for _ in range(2):
            n = (n - self.subsampling_conv_kernel_size) // self.subsampling_conv_stride + 1
        return max(n, 0)


@dataclass(frozen=True)
class QuantizationConfig:
    bits: int
    group_size: int
    mode: str = "affine"

    @classmethod
    def from_model_config(cls, d: dict[str, Any]) -> QuantizationConfig | None:
        q = d.get("quantization")
        if isinstance(q, dict) and "bits" in q:
            return cls(int(q["bits"]), int(q.get("group_size", 64)), str(q.get("mode", "affine")))
        # v1 checkpoints
        q = d.get("_quantization")
        if isinstance(q, dict) and q.get("enabled", False):
            return cls(
                int(q.get("bits", 8)), int(q.get("group_size", 64)), str(q.get("mode", "affine"))
            )
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"bits": self.bits, "group_size": self.group_size, "mode": self.mode}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 512
    blank_id: int = 0
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    quantization: QuantizationConfig | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        encoder = d.get("encoder_config", d)
        frontend = d.get("_audio") or d.get("feature_extractor") or {}
        return cls(
            vocab_size=int(_get(d, "vocab_size", 512)),
            blank_id=int(_get(d, "pad_token_id", 0)),
            encoder=EncoderConfig.from_dict(encoder),
            frontend=FrontendConfig.from_dict(frontend),
            quantization=QuantizationConfig.from_model_config(d),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "architectures": ["LasrForCTC"],
            "model_type": "lasr_ctc",
            "vocab_size": self.vocab_size,
            "pad_token_id": self.blank_id,
            "encoder_config": self.encoder.to_dict(),
            "_audio": {
                "feature_extractor_type": "LasrFeatureExtractor",
                "sampling_rate": self.frontend.sample_rate,
                "feature_size": self.frontend.num_mel_bins,
                "n_fft": self.frontend.n_fft,
                "win_length": self.frontend.win_length,
                "hop_length": self.frontend.hop_length,
            },
        }
        if self.quantization is not None:
            out["quantization"] = self.quantization.to_dict()
        return out

    @property
    def samples_per_frame(self) -> int:
        """Audio samples per encoder output frame (640 = 40 ms for MedASR)."""
        return self.frontend.hop_length * self.encoder.subsampling_factor

    @property
    def frame_duration(self) -> float:
        return self.samples_per_frame / self.frontend.sample_rate

    def num_output_frames(self, num_samples: int) -> int:
        return self.encoder.num_output_frames(self.frontend.num_frames(num_samples))

    @property
    def min_samples(self) -> int:
        """Shortest input that yields at least one encoder frame."""
        k = self.encoder.subsampling_conv_kernel_size
        s = self.encoder.subsampling_conv_stride
        mel_needed = (k - 1) * s + k  # 13 mel frames for k=5, s=2
        return self.frontend.win_length + (mel_needed - 1) * self.frontend.hop_length

    @property
    def frame_offset(self) -> float:
        """Seconds from a frame's first input sample to its receptive-field centre."""
        k = self.encoder.subsampling_conv_kernel_size
        s = self.encoder.subsampling_conv_stride
        mel_span = (k - 1) * s + k
        span = self.frontend.win_length + (mel_span - 1) * self.frontend.hop_length
        return 0.5 * span / self.frontend.sample_rate
