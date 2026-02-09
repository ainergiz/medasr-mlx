# MedASR-MLX: Record Your Doctor Visit, Get a Plain-Language Summary

Two HAI-DEF models running **100% on-device** via Apple MLX. Audio never leaves your phone.

1. **MedASR** (105M Conformer-CTC) — medical speech-to-text
2. **MedGemma 1.5 4B** (4-bit) — plain-language visit summary for patients

```
Microphone -> MedASR (0.13s) -> transcript -> MedGemma (~8s) -> patient summary
             100% on-device, zero cloud calls, full privacy
```

## Quick Demo

```bash
# Setup
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-phase0.txt
uv pip install --python .venv/bin/python mlx-lm
source .venv/bin/activate

# Full pipeline (MedASR + MedGemma)
python main.py

# Custom audio
python main.py --audio your_visit.wav

# Transcript only (skip MedGemma)
python main.py --asr-only

# JSON output
python main.py --json
```

Requires HF token in `.env.local` with accepted terms for `google/medasr`.

## Performance (Apple M4 Pro, 24GB)

| Metric | fp16 | int8 | int4 |
|--------|------|------|------|
| Weights on disk | 201 MB | 121 MB | 79 MB |
| Latency (43.8s audio) | 0.093s | 0.165s | 0.162s |
| Real-Time Factor | 0.002 | 0.004 | 0.004 |
| WER vs HF baseline | 0.0% | 0.0% | 1.25% |

MedGemma visit summary: ~8s. Full pipeline: ~8s total.

## File Map

| File | Purpose |
|------|---------|
| `main.py` | Full pipeline: audio -> transcript -> patient visit summary |
| `model.py` | MLX Conformer-CTC encoder implementation |
| `convert.py` | PyTorch HF -> MLX weight conversion |
| `quantize.py` | fp16 -> int8/int4 quantization |
| `transcribe_mlx.py` | Standalone MedASR transcription |
| `validate_mlx.py` | HF vs MLX parity checker |
| `benchmark.py` | Latency / RTF / memory / WER benchmarks |

## Documentation

- `docs/PLAN.md` — Remaining work and submission plan
- `docs/HANDOFF.md` — Current status and reproduction steps
- `docs/WRITEUP.md` — Competition submission writeup draft
- `docs/LEARNINGS.md` — Technical learnings and gotchas
- `docs/lasr-parity-checklist.md` — HF -> MLX parity requirements

## Competition

MedGemma Impact Challenge on Kaggle (deadline: Feb 24, 2026). Targeting the Edge AI prize category.
