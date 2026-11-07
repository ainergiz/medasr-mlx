"""MedASR (LASR Conformer-CTC) in MLX, inference-optimised.

Compared with the v1 port this module:

* uses ``mx.fast.scaled_dot_product_attention`` and ``mx.fast.rope``,
* expresses the kernel-1 "pointwise" convolutions and the CTC head as
  ``nn.Linear`` (identical maths, and quantizable by ``nn.quantize``),
* folds the conv-module BatchNorm into the depthwise convolution at load time,
* drops the v1 KV-cache streaming path. MedASR is trained full-context, and
  the paper's streaming recipe (Temporal Posterior Fusion, see ``fusion.py``)
  runs the unmodified model on sliding windows instead.

Parameter names follow the Hugging Face checkpoint so weights map 1:1.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import EncoderConfig, ModelConfig

_ACTIVATIONS = {"silu": nn.silu, "swish": nn.silu, "relu": nn.relu, "gelu": nn.gelu}


class Subsampling(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        k, s = cfg.subsampling_conv_kernel_size, cfg.subsampling_conv_stride
        self.dense_0 = nn.Linear(cfg.num_mel_bins, cfg.hidden_size)
        self.conv_0 = nn.Conv1d(cfg.hidden_size, cfg.hidden_size, kernel_size=k, stride=s)
        self.conv_1 = nn.Conv1d(
            cfg.hidden_size, cfg.subsampling_conv_channels, kernel_size=k, stride=s
        )
        self.dense_1 = nn.Linear(cfg.subsampling_conv_channels, cfg.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.dense_0(x))
        x = nn.relu(self.conv_0(x))
        x = nn.relu(self.conv_1(x))
        return self.dense_1(x)


class Attention(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        self.rope_theta = cfg.rope_theta
        bias = cfg.attention_bias
        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=bias)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, t, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, t, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        # LASR uses "rotate_half" (GPT-NeoX style) RoPE == traditional=False.
        q = mx.fast.rope(
            q, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=0
        )
        k = mx.fast.rope(
            k, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=0
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, t, -1))


class ConvModule(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        c = cfg.hidden_size
        self.kernel_size = cfg.conv_kernel_size
        self.activation = _ACTIVATIONS[cfg.hidden_act]
        self.pointwise_conv1 = nn.Linear(c, 2 * c, bias=cfg.convolution_bias)
        # BatchNorm is folded into this convolution's weight and bias at load time.
        self.depthwise_conv = nn.Conv1d(c, c, kernel_size=self.kernel_size, groups=c, bias=True)
        self.pointwise_conv2 = nn.Linear(c, c, bias=cfg.convolution_bias)

    def __call__(self, x: mx.array, frame_mask: mx.array | None) -> mx.array:
        x = self.pointwise_conv1(x)
        a, g = mx.split(x, 2, axis=-1)
        x = a * mx.sigmoid(g)
        if frame_mask is not None:
            x = mx.where(frame_mask[..., None], x, mx.zeros((), dtype=x.dtype))
        # torch padding="same" for an even kernel pads (k-1)//2 left, k//2 right.
        left = (self.kernel_size - 1) // 2
        x = mx.pad(x, [(0, 0), (left, self.kernel_size - 1 - left), (0, 0)])
        x = self.activation(self.depthwise_conv(x))
        return self.pointwise_conv2(x)


class FeedForward(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.activation = _ACTIVATIONS[cfg.hidden_act]
        self.linear1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.attention_bias)
        self.linear2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.attention_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.activation(self.linear1(x)))


class ConformerBlock(nn.Module):
    def __init__(self, cfg: EncoderConfig, residual_scale: float = 1.0):
        super().__init__()
        # With fp16 stabilisation the residual stream is scaled by ``residual_scale``;
        # scaling eps by its square keeps every LayerNorm output exactly unchanged.
        h, eps = cfg.hidden_size, cfg.layer_norm_eps * residual_scale**2
        self.feed_forward1 = FeedForward(cfg)
        self.self_attn = Attention(cfg)
        self.conv = ConvModule(cfg)
        self.feed_forward2 = FeedForward(cfg)
        self.norm_feed_forward1 = nn.LayerNorm(h, eps=eps, bias=False)
        self.norm_self_att = nn.LayerNorm(h, eps=eps, bias=False)
        self.norm_conv = nn.LayerNorm(h, eps=eps, bias=False)
        self.norm_feed_forward2 = nn.LayerNorm(h, eps=eps, bias=False)
        self.norm_out = nn.LayerNorm(h, eps=eps, bias=False)
        self.ff_w = cfg.feed_forward_residual_weights
        self.conv_w = cfg.conv_residual_weights

    def __call__(
        self, x: mx.array, attn_mask: mx.array | None, frame_mask: mx.array | None
    ) -> mx.array:
        ff0, ff1 = self.ff_w
        x = ff0 * x + ff1 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), attn_mask)
        c0, c1 = self.conv_w
        x = c0 * x + c1 * self.conv(self.norm_conv(x), frame_mask)
        x = ff0 * x + ff1 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, residual_scale: float = 1.0):
        super().__init__()
        self.config = cfg
        self.subsampler = Subsampling(cfg)
        self.layers = [ConformerBlock(cfg, residual_scale) for _ in range(cfg.num_hidden_layers)]
        self.out_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps, bias=False)

    def output_lengths(self, mel_lengths: mx.array) -> mx.array:
        k, s = self.config.subsampling_conv_kernel_size, self.config.subsampling_conv_stride
        n = mel_lengths
        for _ in range(2):
            n = (n - k) // s + 1
        return mx.maximum(n, 0)

    def __call__(
        self, features: mx.array, mel_mask: mx.array | None = None
    ) -> tuple[mx.array, mx.array | None]:
        # Convolutions are never quantized, so their dtype is the compute dtype.
        x = self.subsampler(features.astype(self.subsampler.conv_0.weight.dtype))
        frame_mask = attn_mask = None
        if mel_mask is not None:
            lengths = self.output_lengths(mel_mask.astype(mx.int32).sum(axis=-1))
            frame_mask = mx.arange(x.shape[1])[None, :] < lengths[:, None]
            attn_mask = frame_mask[:, None, None, :]  # mask padded keys only
        for layer in self.layers:
            x = layer(x, attn_mask, frame_mask)
        return self.out_norm(x), frame_mask


class MedASRModel(nn.Module):
    """``LasrForCTC``: Conformer encoder + linear CTC head (logits over 512 pieces)."""

    def __init__(self, config: ModelConfig, residual_scale: float = 1.0):
        super().__init__()
        self.config = config
        self.residual_scale = residual_scale
        self.encoder = Encoder(config.encoder, residual_scale)
        self.ctc_head = nn.Linear(config.encoder.hidden_size, config.vocab_size, bias=True)

    @property
    def dtype(self) -> mx.Dtype:
        w = self.encoder.subsampler.conv_0.weight
        return w.dtype

    def __call__(
        self, features: mx.array, mel_mask: mx.array | None = None
    ) -> tuple[mx.array, mx.array | None]:
        """Return ``(logits [B, T', V], frame_mask [B, T'] or None)``."""
        hidden, frame_mask = self.encoder(features, mel_mask)
        return self.ctc_head(hidden), frame_mask
