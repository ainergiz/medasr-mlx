# MedASR-MLX

**Google's [MedASR](https://huggingface.co/google/medasr) (105M Conformer-CTC) ported to [MLX](https://github.com/ml-explore/mlx) for on-device medical speech recognition on Apple Silicon.**

Transcribes 43.8 seconds of medical audio in **0.09 seconds** — 470x real-time, zero cloud calls.

## Models on Hugging Face

| Model | Weights | WER vs PyTorch | Notes |
|-------|---------|----------------|-------|
| [ainergiz/medasr-mlx-fp16](https://huggingface.co/ainergiz/medasr-mlx-fp16) | 201 MB | 0.0% | Full precision, recommended |
| [ainergiz/medasr-mlx-int8](https://huggingface.co/ainergiz/medasr-mlx-int8) | 121 MB | 0.0% | Lossless 8-bit quantized |

## Performance

**Apple M4 Pro, 24 GB — 43.8s medical audio clip:**

| | MedASR MLX (fp16) | MedASR MLX (int8) | HF PyTorch (fp32) |
|---|---|---|---|
| **Latency** | **0.09s** | 0.16s | 0.9–1.6s |
| **Real-Time Factor** | **0.002** | 0.004 | 0.02–0.04 |
| **Speed** | **470x real-time** | 270x real-time | 27–49x real-time |
| **Weights** | 201 MB | 121 MB | ~421 MB |
| **WER parity** | 0.0% | 0.0% | baseline |

**6–17x faster** than HuggingFace PyTorch on Apple Silicon.

## Prerequisites

- Apple Silicon Mac (M1+) or iPhone (A17+)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager

## Quick Start

```bash
git clone https://github.com/ainergiz/medasr-mlx.git
cd medasr-mlx
uv sync
```

### Download weights

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('ainergiz/medasr-mlx-fp16', local_dir='artifacts/medasr-mlx-fp16')
"
```

Or convert from the original yourself (needs HF token with `google/medasr` access):

```bash
uv sync --extra convert
uv run python convert.py      # PyTorch → MLX fp16
uv run python quantize.py     # fp16 → int8
```

### Transcribe

```bash
uv run python transcribe_mlx.py --audio your_audio.wav

# Streaming mode (experimental)
uv run python transcribe_mlx.py --audio your_audio.wav --streaming --chunk-sec 3.0
```

### Benchmark

```bash
uv run python benchmark.py --audio-dir audio/eval-clips --with-references --mlx-only

# Beam search + 6-gram KenLM
uv run python benchmark.py --audio-dir audio/eval-clips --with-references --mlx-only \
  --decode-mode beam --beam-width 32 --kenlm-alpha 0.1 --kenlm-beta 0.0
```

## Architecture

17-layer Conformer encoder with CTC decoding:

```
Audio (16 kHz mono)
  │
  ▼
128-bin Mel Spectrogram (n_fft=512, hop=160, win=400)
  │
  ▼
Subsampling: 2× Conv1d (stride 2, kernel 5) → 4x frame reduction
  │
  ▼
17× Conformer Blocks
  ├── Feed-Forward (½ residual, SiLU, 2048 intermediate)
  ├── Multi-Head Self-Attention (8 heads × 64 dim, RoPE)
  ├── Convolution (depthwise, kernel 32)
  └── Feed-Forward (½ residual)
  │
  ▼
CTC Head → 512-token SentencePiece vocabulary → greedy decode
```

**105M parameters.** Weights converted from PyTorch with Conv1d transposition (`[out, in, kernel]` → `[out, kernel, in]`), BatchNorm buffer handling, and verified at 100% token agreement.

## Conversion Pipeline

1. **`convert.py`** — PyTorch → MLX fp16 (368 parameter tensors + 51 BatchNorm buffers)
2. **`quantize.py`** — fp16 → int8/int4 (affine, group_size=64, Linear/Embedding only)
3. **`validate_mlx.py`** — Bit-exact parity check against HF PyTorch

Key finding: Conv1d layers must stay in fp16 — quantizing them destroys accuracy for this architecture.

## Streaming

Cache-aware streaming is implemented (attention KV cache + convolution left-context) but experimental. MedASR was trained for full-context inference, so streaming degrades WER:

| Mode | Chunk | WER | Delta vs Offline |
|------|-------|-----|------------------|
| Offline (full-context) | — | 0.529 | baseline |
| Streaming | 3.0s | 0.569 | +0.039 |
| Streaming | 1.5s | 0.674 | +0.145 |

3-second chunks are the sweet spot (~4% WER increase).

## File Map

| File | Purpose |
|------|---------|
| `model.py` | MLX Conformer-CTC with streaming cache support |
| `transcribe_mlx.py` | Standalone transcription (offline + streaming) |
| `decode.py` | CTC decoding (greedy + beam search + KenLM) |
| `streaming.py` | Cache-aware streaming transcriber |
| `audio_utils.py` | Audio loading and resampling |
| `convert.py` | PyTorch HF → MLX weight conversion |
| `quantize.py` | MLX fp16 → int8/int4 quantization |
| `validate_mlx.py` | HF vs MLX parity validation |
| `benchmark.py` | WER / latency / RTF / streaming benchmarks |

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.11+ (managed by uv)
- Apple Silicon (M1+ Mac or A17+ iPhone)
- MLX 0.30.6+

## License

Model weights are governed by the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms). Source code is licensed under Apache 2.0.

> HAI-DEF is provided under and subject to the Health AI Developer Foundations Terms of Use.

## Acknowledgments

- [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations) for MedASR
- [Apple MLX](https://github.com/ml-explore/mlx)
