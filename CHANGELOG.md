# Changelog

## 2.0.0 (unreleased)

A rewrite around the MedASR technical report
([arXiv:2605.16555](https://arxiv.org/abs/2605.16555)) and recent work on
context biasing and streaming. Weights are unchanged: every v1 checkpoint
still loads.

### Added
- **Installable package** `medasr_mlx` with a `medasr` CLI (`transcribe`,
  `stream`, `eval`, `convert`, `bench`) and a Python API
  (`medasr_mlx.load(...).transcribe(...)`).
- **Long-form transcription** with Temporal Posterior Fusion (Hann-weighted
  posterior averaging over 20 s windows), the report's long-form recipe.
  Memory stays bounded, so hour-long recordings work.
- **Streaming** with Temporal Posterior Fusion. It runs the *unmodified*
  full-context model on sliding windows and reports committed and tentative
  text, then makes a final second pass. Presets are `paper` (20 s / 320 ms),
  `balanced` and `low-power`. The stream start is padded as in the report, but
  with a -60 dBFS noise floor instead of digital silence, which removes a
  spurious `]` token at speech onset.
- **Word timestamps and confidences**, segments, and `txt` / `json` / `srt` /
  `vtt` output.
- **CTC prefix beam search** in pure Python/NumPy, with blank-frame skipping.
- **Token-level LM shallow fusion** for MedASR's `lm_6.kenlm` through KenLM, or
  a built-in ARPA reader for small domain LMs (it matches KenLM to 1e-6).
- **Context biasing (hotwords)**: a character-trie phrase booster with refunds
  for partial matches, for medication lists, formularies and patient-specific
  vocabulary.
- **Evaluation harness**: a normaliser in the spirit of the report's, WER/CER,
  and medical-term metrics (term recall, keyword error rate).
  `tools/prepare_eval_data.py` builds manifests from the public Eka Medical ASR
  dataset and LibriSpeech.
- **Direct loading of `google/medasr`** (PyTorch safetensors), with no
  conversion step and no torch.
- `medasr convert` with optional quantization: affine int3–8 and MLX's
  `mxfp4` / `mxfp8` / `nvfp4`.
- A test suite of 63 tests, which runs on Linux CPU. It includes architecture
  parity with transformers' `LasrForCTC`. GitHub Actions CI.

### Changed
- **No transformers, torch, pyctcdecode or scipy at inference.** The log-mel
  frontend and tokenizer are dependency-free ports, bit-exact with
  `LasrFeatureExtractor` and `LasrTokenizer`. Core dependencies are `mlx`,
  `numpy`, `huggingface_hub`, `soundfile` and `soxr`.
- **Real fp16 inference.** MedASR's residual stream reaches about 2e6, which
  overflows fp16 (max 65504). v1 never actually ran in fp16: its float32
  features silently promoted every activation to fp32. v2 rescales the
  residual stream and the LayerNorm eps by powers of two, which is exact by
  construction. fp16 then matches fp32 on 99.9% of frames.
  Defaults: fp16 on GPU, fp32 on CPU.
- **Faster kernels.** `mx.fast.scaled_dot_product_attention` and
  `mx.fast.rope`. BatchNorm is folded into the depthwise convolution. The
  kernel-1 convolutions and the CTC head are now `nn.Linear`, so quantization
  covers them too. `mx.compile` is used for fixed-shape batches on GPU.
- Batched inference with exact padding masks (`transcribe_batch`).
- Python `>=3.10` (v1 required `<3.13`), and numpy 2 is supported (v1's
  pyctcdecode pinned `numpy<2`).

### Removed
- The v1 KV-cache streaming path. MedASR is trained full-context, and that
  path cost +3.9 WER points at 3 s chunks and +14.5 at 1.5 s. Streaming now uses the
  report's recipe instead.
- The flat scripts (`transcribe_mlx.py`, `benchmark.py`, `convert.py`,
  `quantize.py`, `validate_mlx.py`, ...). See the migration table in the README.

## 0.1.0

Initial MLX port: Conformer-CTC in MLX, fp16/int8 checkpoints, greedy decoding
and pyctcdecode beam search, and experimental cache-aware streaming.
