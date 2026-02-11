#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


@dataclass
class LasrEncoderConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    attention_bias: bool
    convolution_bias: bool
    conv_kernel_size: int
    subsampling_conv_kernel_size: int
    subsampling_conv_stride: int
    subsampling_conv_channels: int
    num_mel_bins: int
    layer_norm_eps: float
    feed_forward_residual_weights: tuple[float, float]
    conv_residual_weights: tuple[float, float]
    batch_norm_momentum: float
    hidden_act: str
    attention_dropout: float
    dropout: float
    dropout_positions: float
    layerdrop: float
    activation_dropout: float
    max_position_embeddings: int
    rope_theta: float
    rope_type: str

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LasrEncoderConfig:
        rope_parameters = d.get("rope_parameters") or {
            "rope_theta": 10000.0,
            "rope_type": "default",
        }
        ff_weights = d.get("feed_forward_residual_weights", [1.5, 0.5])
        conv_weights = d.get("conv_residual_weights", [2.0, 1.0])
        return cls(
            hidden_size=_as_int(d.get("hidden_size"), 512),
            num_hidden_layers=_as_int(d.get("num_hidden_layers"), 17),
            num_attention_heads=_as_int(d.get("num_attention_heads"), 8),
            num_key_value_heads=_as_int(
                d.get("num_key_value_heads"), _as_int(d.get("num_attention_heads"), 8)
            ),
            intermediate_size=_as_int(d.get("intermediate_size"), 2048),
            attention_bias=bool(d.get("attention_bias", False)),
            convolution_bias=bool(d.get("convolution_bias", False)),
            conv_kernel_size=_as_int(d.get("conv_kernel_size"), 32),
            subsampling_conv_kernel_size=_as_int(d.get("subsampling_conv_kernel_size"), 5),
            subsampling_conv_stride=_as_int(d.get("subsampling_conv_stride"), 2),
            subsampling_conv_channels=_as_int(d.get("subsampling_conv_channels"), 256),
            num_mel_bins=_as_int(d.get("num_mel_bins"), 128),
            layer_norm_eps=_as_float(d.get("layer_norm_eps"), 1e-6),
            feed_forward_residual_weights=(float(ff_weights[0]), float(ff_weights[1])),
            conv_residual_weights=(float(conv_weights[0]), float(conv_weights[1])),
            batch_norm_momentum=_as_float(d.get("batch_norm_momentum"), 0.01),
            hidden_act=str(d.get("hidden_act", "silu")),
            attention_dropout=_as_float(d.get("attention_dropout"), 0.1),
            dropout=_as_float(d.get("dropout"), 0.1),
            dropout_positions=_as_float(d.get("dropout_positions"), 0.0),
            layerdrop=_as_float(d.get("layerdrop"), 0.0),
            activation_dropout=_as_float(d.get("activation_dropout"), 0.1),
            max_position_embeddings=_as_int(d.get("max_position_embeddings"), 10000),
            rope_theta=_as_float(rope_parameters.get("rope_theta"), 10000.0),
            rope_type=str(rope_parameters.get("rope_type", "default")),
        )


@dataclass
class LasrCtcConfig:
    vocab_size: int
    pad_token_id: int
    encoder_config: LasrEncoderConfig

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LasrCtcConfig:
        encoder = d.get("encoder_config", d)
        return cls(
            vocab_size=_as_int(d.get("vocab_size"), 512),
            pad_token_id=_as_int(d.get("pad_token_id"), 0),
            encoder_config=LasrEncoderConfig.from_dict(encoder),
        )


@dataclass
class ConformerLayerCache:
    attn_key: mx.array | None = None
    attn_value: mx.array | None = None
    conv_left: mx.array | None = None


@dataclass
class ConformerCache:
    layers: list[ConformerLayerCache]
    token_offset: int = 0

    @classmethod
    def empty(cls, num_layers: int) -> ConformerCache:
        return cls(layers=[ConformerLayerCache() for _ in range(num_layers)], token_offset=0)


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> tuple[mx.array, mx.array]:
    # q/k: [B, H, T, D], cos/sin: [1, T, D] -> [1, 1, T, D]
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LasrEncoderSubsampling(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.dense_0 = nn.Linear(config.num_mel_bins, config.hidden_size, bias=True)
        self.conv_0 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
            padding=0,
            bias=True,
        )
        self.conv_1 = nn.Conv1d(
            config.hidden_size,
            config.subsampling_conv_channels,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
            padding=0,
            bias=True,
        )
        self.dense_1 = nn.Linear(config.subsampling_conv_channels, config.hidden_size, bias=True)

    def __call__(self, input_features: mx.array) -> mx.array:
        hidden_states = nn.relu(self.dense_0(input_features))
        hidden_states = nn.relu(self.conv_0(hidden_states))
        hidden_states = nn.relu(self.conv_1(hidden_states))
        return self.dense_1(hidden_states)


class LasrEncoderRotaryEmbedding(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_type = config.rope_type
        self.head_dim = config.head_dim

        if self.rope_type != "default":
            raise ValueError(f"Unsupported rope_type for current MLX port: {self.rope_type}")

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # x shape [B, T, C], position_ids [1, T]
        inv_freq = 1.0 / (
            self.rope_theta
            ** (mx.arange(0, self.head_dim, 2, dtype=mx.float32) / float(self.head_dim))
        )
        inv_freq = mx.expand_dims(inv_freq, axis=0)  # [1, D/2]
        freqs = mx.expand_dims(position_ids.astype(mx.float32), axis=-1) * inv_freq  # [1, T, D/2]
        emb = mx.concatenate([freqs, freqs], axis=-1)  # [1, T, D]
        cos = mx.cos(emb).astype(x.dtype)
        sin = mx.sin(emb).astype(x.dtype)
        return cos, sin


class LasrEncoderAttention(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def _repeat_kv(self, hidden_states: mx.array, n_rep: int) -> mx.array:
        # hidden_states [B, KV_H, T, D] -> [B, H, T, D]
        if n_rep == 1:
            return hidden_states
        hidden_states = mx.expand_dims(hidden_states, axis=2)  # [B, KV_H, 1, T, D]
        hidden_states = mx.repeat(hidden_states, repeats=n_rep, axis=2)
        b, kv_h, rep, t, d = hidden_states.shape
        return mx.reshape(hidden_states, (b, kv_h * rep, t, d))

    def _truncate_left_context(
        self,
        k: mx.array,
        v: mx.array,
        max_left_context: int | None,
        q_len: int,
    ) -> tuple[mx.array, mx.array]:
        if max_left_context is None:
            return k, v
        keep = max(int(max_left_context), int(q_len))
        if k.shape[2] <= keep:
            return k, v
        return k[:, :, -keep:, :], v[:, :, -keep:, :]

    def _build_streaming_attn_bias(self, current_mask: mx.array, total_k_len: int) -> mx.array:
        # current_mask: [B, Q] bool
        bsz, q_len = current_mask.shape
        past_len = total_k_len - q_len
        if past_len < 0:
            raise ValueError(
                f"Invalid streaming attention lengths: total_k_len={total_k_len}, q_len={q_len}"
            )
        if past_len > 0:
            past_valid = mx.ones((bsz, past_len), dtype=mx.bool_)
            key_valid = mx.concatenate([past_valid, current_mask], axis=1)
        else:
            key_valid = current_mask
        query_valid = current_mask
        valid = mx.expand_dims(mx.expand_dims(query_valid, axis=1), axis=-1) & mx.expand_dims(
            mx.expand_dims(key_valid, axis=1), axis=1
        )
        zeros = mx.zeros(valid.shape, dtype=mx.float32)
        neg = mx.full(valid.shape, -1e9, dtype=mx.float32)
        return mx.where(valid, zeros, neg)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
        cache: ConformerLayerCache | None = None,
        max_left_context: int | None = None,
        is_streaming: bool = False,
    ) -> tuple[mx.array, ConformerLayerCache | None]:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = mx.transpose(mx.reshape(q, (bsz, q_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = mx.transpose(
            mx.reshape(k, (bsz, q_len, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3)
        )
        v = mx.transpose(
            mx.reshape(v, (bsz, q_len, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3)
        )

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_bias = attention_mask
        layer_cache: ConformerLayerCache | None = None
        if is_streaming:
            if cache is not None and cache.attn_key is not None:
                k = mx.concatenate([cache.attn_key, k], axis=2)
            if cache is not None and cache.attn_value is not None:
                v = mx.concatenate([cache.attn_value, v], axis=2)
            k, v = self._truncate_left_context(
                k,
                v,
                max_left_context=max_left_context,
                q_len=q_len,
            )
            if attention_mask is None:
                current_mask = mx.ones((bsz, q_len), dtype=mx.bool_)
            elif attention_mask.ndim == 2:
                current_mask = attention_mask.astype(mx.bool_)
            else:
                current_mask = mx.ones((bsz, q_len), dtype=mx.bool_)
            attn_bias = self._build_streaming_attn_bias(current_mask, total_k_len=k.shape[2])
            layer_cache = ConformerLayerCache(
                attn_key=k,
                attn_value=v,
                conv_left=cache.conv_left if cache is not None else None,
            )

        k = self._repeat_kv(k, self.num_key_value_groups)
        v = self._repeat_kv(v, self.num_key_value_groups)

        attn_weights = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scaling
        if attn_bias is not None:
            if attn_bias.ndim == 4:
                attn_weights = attn_weights + attn_bias[:, :, :, : k.shape[2]]
            else:
                attn_weights = attn_weights + attn_bias

        attn_probs = mx.softmax(attn_weights.astype(mx.float32), axis=-1).astype(attn_weights.dtype)
        attn_output = mx.matmul(attn_probs, v)
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (bsz, q_len, self.num_heads * self.head_dim))
        return self.o_proj(attn_output), layer_cache


class LasrEncoderConvolutionModule(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        channels = config.hidden_size
        kernel_size = config.conv_kernel_size

        self.kernel_size = kernel_size
        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=config.convolution_bias
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,  # emulate torch padding="same" manually for even kernels
            groups=channels,
            bias=config.convolution_bias,
        )
        self.norm = nn.BatchNorm(channels, momentum=config.batch_norm_momentum)
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=config.convolution_bias
        )

    def _glu(self, x: mx.array) -> mx.array:
        # GLU over channel axis in channel-last layout: split last dim.
        half = x.shape[-1] // 2
        a = x[..., :half]
        b = x[..., half:]
        return a * nn.sigmoid(b)

    def _same_pad(self, x: mx.array) -> mx.array:
        # Match torch Conv1d padding="same" behavior for stride=1.
        total = self.kernel_size - 1
        left = total // 2
        right = total - left
        return mx.pad(x, ((0, 0), (left, right), (0, 0)))

    def _masked_rows(self, attention_mask: mx.array, seq_len: int) -> mx.array:
        # Returns [B, T] bool where True means row should be zeroed.
        if attention_mask.ndim == 2:
            if attention_mask.dtype == mx.bool_:
                return ~attention_mask
            return attention_mask == 0.0

        if attention_mask.ndim == 4:
            if attention_mask.dtype == mx.bool_:
                all_masked_rows = mx.all(~attention_mask, axis=-1)  # [B, 1, T]
            else:
                all_masked_rows = mx.all(attention_mask != 0.0, axis=-1)  # [B, 1, T]
            if all_masked_rows.shape[1] == 1:
                all_masked_rows = mx.squeeze(all_masked_rows, axis=1)
            return all_masked_rows

        bsz = attention_mask.shape[0]
        return mx.zeros((bsz, seq_len), dtype=mx.bool_)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        cache: ConformerLayerCache | None = None,
        is_streaming: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = self._glu(hidden_states)

        if attention_mask is not None:
            all_masked_rows = self._masked_rows(attention_mask, seq_len=hidden_states.shape[1])
            all_masked_rows = mx.expand_dims(all_masked_rows, axis=-1)
            hidden_states = mx.where(all_masked_rows, mx.zeros_like(hidden_states), hidden_states)

        next_conv_left = None
        if is_streaming:
            cache_size = self.kernel_size - 1
            bsz, seq_len, channels = hidden_states.shape
            if cache is not None and cache.conv_left is not None:
                conv_left = cache.conv_left
            else:
                conv_left = mx.zeros((bsz, cache_size, channels), dtype=hidden_states.dtype)
            conv_input = mx.concatenate([conv_left, hidden_states], axis=1)
            if cache_size > 0:
                next_conv_left = conv_input[:, -cache_size:, :]
            else:
                next_conv_left = mx.zeros((bsz, 0, channels), dtype=hidden_states.dtype)
            # Stateful approximation of offline "same" padding:
            # run depthwise conv over [cached_left + current_chunk] and keep only current positions.
            conv_input = self._same_pad(conv_input)
            conv_output_all = self.depthwise_conv(conv_input)
            hidden_states = conv_output_all[:, -seq_len:, :]
        else:
            hidden_states = self._same_pad(hidden_states)
            hidden_states = self.depthwise_conv(hidden_states)

        hidden_states = self.norm(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        return hidden_states, next_conv_left


class LasrEncoderFeedForward(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.linear1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.attention_bias
        )
        self.linear2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.attention_bias
        )
        self.activation_dropout = config.activation_dropout
        self.hidden_act = config.hidden_act

    def _activate(self, x: mx.array) -> mx.array:
        if self.hidden_act == "silu":
            return nn.silu(x)
        if self.hidden_act == "relu":
            return nn.relu(x)
        if self.hidden_act == "gelu":
            return nn.gelu(x)
        raise ValueError(f"Unsupported hidden_act: {self.hidden_act}")

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self._activate(self.linear1(hidden_states))
        if self.training and self.activation_dropout > 0:
            hidden_states = nn.dropout(hidden_states, p=self.activation_dropout)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class LasrEncoderBlock(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.feed_forward1 = LasrEncoderFeedForward(config)
        self.self_attn = LasrEncoderAttention(config)
        self.conv = LasrEncoderConvolutionModule(config)
        self.feed_forward2 = LasrEncoderFeedForward(config)

        # HF parity requirement: LayerNorm with bias=False
        self.norm_feed_forward1 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, affine=True, bias=False
        )
        self.norm_self_att = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, affine=True, bias=False
        )
        self.norm_conv = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, affine=True, bias=False
        )
        self.norm_feed_forward2 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, affine=True, bias=False
        )
        self.norm_out = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, affine=True, bias=False
        )

        self.feed_forward_residual_weights = config.feed_forward_residual_weights
        self.conv_residual_weights = config.conv_residual_weights

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
        cache: ConformerLayerCache | None = None,
        max_left_context: int | None = None,
        is_streaming: bool = False,
    ) -> tuple[mx.array, ConformerLayerCache | None]:
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = (
            self.feed_forward_residual_weights[0] * residual
            + self.feed_forward_residual_weights[1] * hidden_states
        )

        attn_output, attn_layer_cache = self.self_attn(
            hidden_states=self.norm_self_att(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache=cache,
            max_left_context=max_left_context,
            is_streaming=is_streaming,
        )
        hidden_states = hidden_states + attn_output

        conv_output, conv_left = self.conv(
            self.norm_conv(hidden_states),
            attention_mask=attention_mask,
            cache=cache,
            is_streaming=is_streaming,
        )
        hidden_states = (
            self.conv_residual_weights[0] * hidden_states
            + self.conv_residual_weights[1] * conv_output
        )

        residual = hidden_states
        hidden_states = self.feed_forward2(self.norm_feed_forward2(hidden_states))
        hidden_states = (
            self.feed_forward_residual_weights[0] * residual
            + self.feed_forward_residual_weights[1] * hidden_states
        )
        hidden_states = self.norm_out(hidden_states)
        if not is_streaming:
            return hidden_states, None

        new_cache = ConformerLayerCache(
            attn_key=attn_layer_cache.attn_key if attn_layer_cache is not None else None,
            attn_value=attn_layer_cache.attn_value if attn_layer_cache is not None else None,
            conv_left=conv_left,
        )
        return hidden_states, new_cache


class LasrEncoder(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.dropout = config.dropout
        self.dropout_positions = config.dropout_positions
        self.layerdrop = config.layerdrop
        self.subsampler = LasrEncoderSubsampling(config)
        self.rotary_emb = LasrEncoderRotaryEmbedding(config)
        self.layers = [LasrEncoderBlock(config) for _ in range(config.num_hidden_layers)]
        self.out_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, affine=True, bias=False
        )
        self._subsampling_kernel = config.subsampling_conv_kernel_size
        self._subsampling_stride = config.subsampling_conv_stride

    def init_streaming_cache(self) -> ConformerCache:
        return ConformerCache.empty(num_layers=len(self.layers))

    def _get_subsampling_output_length(self, input_lengths: mx.array) -> mx.array:
        kernel = self._subsampling_kernel
        stride = self._subsampling_stride
        for _ in range(2):
            input_lengths = (input_lengths - kernel) // stride + 1
        return input_lengths

    def _get_output_attention_mask(self, attention_mask: mx.array, target_length: int) -> mx.array:
        output_lengths = self._get_subsampling_output_length(mx.sum(attention_mask, axis=-1))
        positions = mx.arange(target_length)[None, :]
        return positions < mx.expand_dims(output_lengths, axis=-1)

    def _create_bidirectional_mask(self, attention_mask: mx.array) -> mx.array:
        # attention_mask: [B, T] bool
        key_mask = mx.expand_dims(mx.expand_dims(attention_mask, axis=1), axis=1)  # [B, 1, 1, T]
        query_mask = mx.expand_dims(mx.expand_dims(attention_mask, axis=1), axis=-1)  # [B, 1, T, 1]
        valid = key_mask & query_mask
        zeros = mx.zeros(valid.shape, dtype=mx.float32)
        neg = mx.full(valid.shape, -1e9, dtype=mx.float32)
        return mx.where(valid, zeros, neg)

    def __call__(
        self,
        input_features: mx.array,
        attention_mask: mx.array | None = None,
        cache: ConformerCache | None = None,
        is_streaming: bool = False,
        max_left_context: int | None = None,
    ) -> mx.array | tuple[mx.array, ConformerCache]:
        hidden_states = self.subsampler(input_features)
        seq_len = hidden_states.shape[1]

        if is_streaming:
            if cache is None:
                cache = self.init_streaming_cache()
            if len(cache.layers) != len(self.layers):
                raise ValueError(
                    f"Invalid cache depth: got {len(cache.layers)} layer caches "
                    f"for {len(self.layers)} encoder layers."
                )
            position_ids = (mx.arange(seq_len, dtype=mx.float32) + float(cache.token_offset))[
                None, :
            ]
        else:
            position_ids = mx.arange(seq_len, dtype=mx.float32)[None, :]
        cos, sin = self.rotary_emb(hidden_states, position_ids=position_ids)

        if self.training and self.dropout > 0:
            hidden_states = nn.dropout(hidden_states, p=self.dropout)
        if self.training and self.dropout_positions > 0:
            cos = nn.dropout(cos, p=self.dropout_positions)
            sin = nn.dropout(sin, p=self.dropout_positions)

        output_mask = None
        if attention_mask is not None:
            output_mask = self._get_output_attention_mask(attention_mask, target_length=seq_len)
        elif is_streaming:
            output_mask = mx.ones((hidden_states.shape[0], seq_len), dtype=mx.bool_)

        layer_attention_mask = None
        if is_streaming:
            layer_attention_mask = output_mask
        elif output_mask is not None:
            layer_attention_mask = self._create_bidirectional_mask(output_mask)

        new_layer_caches: list[ConformerLayerCache] = []
        for idx, layer in enumerate(self.layers):
            layer_cache = cache.layers[idx] if (is_streaming and cache is not None) else None
            hidden_states, new_layer_cache = layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_embeddings=(cos, sin),
                cache=layer_cache,
                max_left_context=max_left_context,
                is_streaming=is_streaming,
            )
            if is_streaming:
                new_layer_caches.append(new_layer_cache or ConformerLayerCache())

        hidden_states = self.out_norm(hidden_states)
        if is_streaming:
            assert cache is not None  # for type checkers
            new_cache = ConformerCache(
                layers=new_layer_caches,
                token_offset=int(cache.token_offset) + int(seq_len),
            )
            return hidden_states, new_cache
        return hidden_states


class LasrForCTC(nn.Module):
    def __init__(self, config: LasrCtcConfig):
        super().__init__()
        self.config = config
        self.encoder = LasrEncoder(config.encoder_config)
        self.ctc_head = nn.Conv1d(
            config.encoder_config.hidden_size, config.vocab_size, kernel_size=1
        )

    def init_streaming_cache(self) -> ConformerCache:
        return self.encoder.init_streaming_cache()

    def __call__(
        self,
        input_features: mx.array,
        attention_mask: mx.array | None = None,
        cache: ConformerCache | None = None,
        is_streaming: bool = False,
        max_left_context: int | None = None,
    ) -> mx.array | tuple[mx.array, ConformerCache]:
        if is_streaming:
            hidden_states, new_cache = self.encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                cache=cache,
                is_streaming=True,
                max_left_context=max_left_context,
            )
            logits = self.ctc_head(hidden_states)
            return logits, new_cache

        hidden_states = self.encoder(input_features=input_features, attention_mask=attention_mask)
        logits = self.ctc_head(hidden_states)
        return logits


def _maybe_apply_quantization(model: LasrForCTC, config_dict: dict[str, Any]) -> None:
    qcfg = config_dict.get("_quantization")
    if not isinstance(qcfg, dict) or not qcfg.get("enabled", False):
        return

    bits = _as_int(qcfg.get("bits"), 4)
    group_size_value = qcfg.get("group_size")
    group_size = _as_int(group_size_value, 64) if group_size_value is not None else None
    mode = str(qcfg.get("mode", "affine"))

    try:
        nn.quantize(model, group_size=group_size, bits=bits, mode=mode)
    except Exception as exc:  # pragma: no cover - defensive error context
        raise ValueError(
            "Failed to initialize quantized MLX model from config "
            f"(bits={bits}, group_size={group_size}, mode={mode})."
        ) from exc


def load_mlx_model(model_dir: str | Path, strict: bool = True) -> LasrForCTC:
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "weights.npz"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    config_dict = json.loads(config_path.read_text())
    config = LasrCtcConfig.from_dict(config_dict)
    model = LasrForCTC(config)
    _maybe_apply_quantization(model, config_dict)
    model.load_weights(str(weights_path), strict=strict)
    model.eval()
    return model
