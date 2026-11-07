# MedASR-MLX v2

**Google's [MedASR](https://huggingface.co/google/medasr) medical speech recognizer (105M Conformer-CTC) on [MLX](https://github.com/ml-explore/mlx)**, on-device and with no cloud calls. v2 adds long-form transcription, streaming, word timestamps, context biasing for drug names, and an inference stack free of transformers and torch.

```bash
uv tool install "git+https://github.com/ainergiz/medasr-mlx"   # or pip install "git+…"; Linux: "medasr-mlx[cpu] @ git+…"
medasr transcribe dictation.wav --format srt
medasr transcribe consult.m4a --hotwords "Xarelto,metoprolol,Pantop DSR"
medasr stream --mic          # live, with committed + tentative text
```

```python
import medasr_mlx

asr = medasr_mlx.load()                      # ainergiz/medasr-mlx-fp16 (or "google/medasr", a local dir…)
result = asr.transcribe("dictation.wav", decoder="beam", hotwords=["Dolo 650", "azithromycin"])
print(result.text)
for w in result.words:
    print(f"{w.start:6.2f}-{w.end:6.2f}  {w.confidence:.2f}  {w.text}")
```

## What's new in v2

v1 was a faithful MLX port of the model. v2 builds on what has happened since, starting with MedASR's own
[technical report](https://arxiv.org/abs/2605.16555) (May 2026).

| | v1 | v2 |
|---|---|---|
| Long audio | one full-context pass (O(T²) memory, OOM on long files) | **Temporal Posterior Fusion**: the report's 20 s sliding windows with Hann-weighted posterior fusion, bounded memory |
| Streaming | KV-cache chunks, a regime the model was never trained for (+4% WER at 3 s chunks, +15% at 1.5 s) | **TPF streaming** (report §3.4): the unmodified model on sliding windows, with committed/tentative text and a final second pass |
| Medical vocabulary | none | **Context biasing**: boost a medication list, formulary or template terms during beam search |
| Decoding | greedy, or pyctcdecode (unmaintained, pins `numpy<2`) | native CTC prefix beam search with token-level KenLM fusion (`lm_6.kenlm`) and hotwords |
| Output | text | text, **word timestamps and confidences**, segments, JSON / SRT / VTT |
| Inference deps | transformers + torch + pyctcdecode + kenlm + scipy | `mlx`, `numpy`, `huggingface_hub`, `soundfile`, `soxr` |
| Precision | "fp16" weights, but activations silently promoted to fp32 | **true fp16** through exact power-of-two rescaling (MedASR's activations reach about 2e6, far past fp16's 65504) |
| Kernels | hand-written attention and RoPE | `mx.fast` SDPA and RoPE, folded BatchNorm, quantizable pointwise layers, `mx.compile` on GPU |
| Checkpoints | `.npz` from a torch conversion script | loads `google/medasr` directly (no torch), v1 `.npz`, and v2 safetensors; int4/int8/mxfp4/mxfp8 |
| Quality gates | manual scripts | 61 tests (incl. parity with transformers' `LasrForCTC`), CI, eval harness with medical-term metrics |

The wider ASR landscape informed the design. Open leaderboards have moved to streaming-capable and LLM-decoder
models ([Open ASR Leaderboard](https://huggingface.co/blog/open-asr-leaderboard)), and MLX ports now stream
([mlx-audio](https://pypi.org/project/mlx-audio/): Parakeet, Nemotron 3.5 ASR, Voxtral Realtime). Context biasing
has matured for CTC and transducer models ([CTC-WS](https://www.isca-archive.org/interspeech_2024/andrusenko24_interspeech.pdf),
[streaming CTC-WS](https://arxiv.org/abs/2605.18222), [TurboBias 2.0](https://arxiv.org/abs/2608.21343), evaluated
on medical audio), and so has keyword-aware evaluation ([Contextual Earnings-22](https://arxiv.org/abs/2604.07354),
the [Eka medical ASR benchmark](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset)'s kwWER).
MedASR is still a small, fast specialist: the report measures 5.2% WER on Eye Gaze chest X-ray dictation, versus
12.5% for Whisper large-v3 and 5.9% for Gemini 2.5 Pro. v2 makes it practical for real workflows: long recordings,
live dictation, and patient-specific vocabulary.

## Results

RESULTS_PLACEHOLDER

## Install

```bash
git clone https://github.com/ainergiz/medasr-mlx && cd medasr-mlx
uv sync                       # Apple Silicon
uv sync --extra cpu           # Linux (MLX CPU backend); --extra cuda for NVIDIA
uv sync --extra lm            # KenLM for the 6-gram LM (see note below)
uv sync --extra mic           # live microphone streaming
```

Weights download automatically on first use. The default `ainergiz/medasr-mlx-fp16` is public. To use
Google's original checkpoint, accept the [HAI-DEF terms](https://huggingface.co/google/medasr), run
`huggingface-cli login`, and pass `--model google/medasr`; it loads directly with no conversion step.

> **Linux CPU tip:** the `mlx-cpu` wheel bundles netlib *reference* BLAS. Preloading OpenBLAS made matrix
> multiplies about 80× faster in our tests (same results):
> `apt install libopenblas0-pthread` and `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0`.

## Usage

### Transcribe (long-form)

```bash
medasr transcribe visit.wav                          # text to stdout
medasr transcribe *.wav --format all --output-dir out/   # .txt .json .srt .vtt per file
medasr transcribe visit.wav --decoder beam --beam-width 8
medasr transcribe visit.wav --no-window              # v1 behaviour: one full-context pass
```

Recordings longer than 20 s are recognised in 20 s windows with an 18 s stride, and per-frame posteriors are
fused with Hann weights (`--window/--stride` to change it). Windows sit on the model's 40 ms frame grid and
same-length windows are batched, so an hour of audio is a few hundred fixed-shape batches.

```python
result = asr.transcribe("visit.wav")
result.text, result.segments, result.words, result.to_srt(), result.to_json()
asr.transcribe_batch(["a.wav", "b.wav", ...])   # short clips batched with exact padding masks
lp = asr.log_probs("visit.wav")                 # fused [frames, 512] CTC log-posteriors
```

### Context biasing (hotwords)

```bash
medasr transcribe clinic.wav --hotwords "Pantop DSR,Dolo 650,thyroxine"
medasr transcribe clinic.wav --hotwords-file formulary.txt   # one phrase per line, optional "<TAB>weight"
```

```python
asr.transcribe(audio, decoder="beam", hotwords={"Xarelto": 6.0, "metoprolol succinate": 5.0})
```

The booster is a character-level trie, so it does not depend on how the model segments a word into
SentencePiece pieces. Matching is case-insensitive, starts at word boundaries, and supports multi-word phrases.
A hypothesis earns its bonus character by character, and the bonus is refunded if the phrase is not completed,
so partial matches can survive the beam but are never rewarded. A completed phrase earns
`--hotword-weight` (default 5) in log space.

### Language model

MedASR ships a 6-gram LM over SentencePiece *tokens* (`lm_6.kenlm` in `google/medasr`; the report quotes RAD
dictation going from 5.4% to 4.6% WER with it):

```bash
huggingface-cli download google/medasr lm_6.kenlm --local-dir lm
medasr transcribe dictation.wav --lm lm/lm_6.kenlm --alpha 0.5 --beta 1.0
```

`alpha`/`beta` have not been re-tuned for v2's decoder; tune them on your data with `medasr eval`. Small
domain LMs in ARPA format, built with KenLM's `lmplz` on token-piece text, work without the `kenlm` package.
On Python ≥ 3.13 the PyPI `kenlm` sdist does not build; use
`pip install https://github.com/kpu/kenlm/archive/master.zip`.

### Streaming

```bash
medasr stream --mic --preset balanced
medasr stream dictation.wav --realtime          # simulate a live feed from a file
```

```python
stream = asr.stream(preset="balanced")          # "paper" (20 s / 0.32 s), "balanced" (10 s / 0.64 s), "low-power" (6 s / 1.28 s)
for chunk in mic_chunks():                      # any chunk size, 16 kHz float32
    update = stream.push(chunk)
    render(update.committed_text, update.tentative_text)
final = stream.finish()                         # second pass over fully fused posteriors
```

Every *stride* seconds the most recent *window* of audio is recognised and its posteriors are fused into the
running estimate. Frames with at least `commit_delay_s` (default 1.5 s) of right context are committed and never
revised. Newer frames are shown as tentative. Compute per second of audio is about `window / stride` times the
offline cost, so the presets trade accuracy for compute. The report's recipe pads the start with W seconds of
digital silence; in our tests that made the model emit spurious `]` tokens at speech onset, so v2 grows the
first windows instead (`start_padding=True` restores the report's behaviour).

### Evaluate

```bash
uv run --extra eval python tools/prepare_eval_data.py --out eval-data    # public Eka medical + LibriSpeech sets
medasr eval eval-data/eka-short.jsonl --decoder beam -v --output eka.json
medasr eval my_clips/                       # directory of audio + same-stem .txt references
```

Reports WER and CER after a normaliser that follows the report's scoring (fillers, tags, number words,
units). When the manifest has entity spans it also reports **term recall**, the share of reference medical
entities found verbatim, and **kwER**, the error rate on entity words.

### Convert and quantize

```bash
medasr convert --source google/medasr --output medasr-v2-fp16
medasr convert --source google/medasr --output medasr-v2-int8 --bits 8
medasr convert --source ainergiz/medasr-mlx-fp16 --output medasr-v2-mxfp8 --bits 8 --mode mxfp8 --group-size 32
```

All `nn.Linear` layers are quantized, including the pointwise convolutions, which v2 expresses as Linear. The
subsampler and CTC head stay in floating point unless you pass `--quantize-io`. Depthwise and subsampling
convolutions are never quantized.

## Architecture

```
16 kHz audio ─► log-mel (128 bins, 25 ms / 10 ms, Kaldi mel 125-7500 Hz; bit-exact numpy port)
             ─► subsampling: Linear ► 2× Conv1d(k=5, s=2) ► Linear        (4× → 40 ms frames)
             ─► 17 × Conformer block (512-d, 8 heads, RoPE, no biases):
                    ½·FFN ► MHSA (mx.fast SDPA + RoPE) ► Conv (GLU ► depthwise k=32, BN folded ► SiLU) ► ½·FFN ► LN
             ─► CTC head (512 SentencePiece pieces, blank = <epsilon>)
             ─► greedy │ prefix beam search (+ token 6-gram LM, + hotword trie)
             ─► tokens with frames ─► words with timestamps and confidences ─► segments ─► txt/json/srt/vtt
```

Long-form and streaming wrap this in Temporal Posterior Fusion (`fusion.py`, `streaming.py`).

| Module | Purpose |
|---|---|
| `medasr_mlx/model.py` | Conformer-CTC in MLX |
| `medasr_mlx/weights.py` | load/sanitize all checkpoint layouts, BN folding, fp16 stabilisation, save, quantize |
| `medasr_mlx/features.py`, `tokenizer.py`, `audio.py` | frontend, SentencePiece decode/encode, audio I/O and resampling |
| `medasr_mlx/fusion.py`, `streaming.py` | Temporal Posterior Fusion (long-form, streaming) |
| `medasr_mlx/decoding/` | greedy, beam search, token LMs, phrase booster |
| `medasr_mlx/transcribe.py`, `result.py` | `MedASR` API, words/segments, output formats |
| `medasr_mlx/evaluation.py`, `cli.py`, `convert.py` | metrics, `medasr` CLI, conversion |
| `tools/` | eval-data preparation, parity check against PyTorch |

## Migrating from v1

| v1 | v2 |
|---|---|
| `python transcribe_mlx.py --audio a.wav` | `medasr transcribe a.wav` |
| `--streaming --chunk-sec 3.0` | `medasr stream a.wav --preset balanced` |
| `--decode-mode beam --kenlm-path lm.kenlm` | `--decoder beam --lm lm.kenlm` |
| `python convert.py` + `python quantize.py` | `medasr convert --source google/medasr --output DIR [--bits 8]` |
| `python benchmark.py --with-references` | `medasr eval DIR_OR_MANIFEST`, `medasr bench a.wav` |
| `python validate_mlx.py` | `python tools/validate_hf.py` |
| `from model import load_mlx_model` | `import medasr_mlx; medasr_mlx.load(path)` |

v1 model directories (`weights.npz` + `processor/`) load unchanged.

## Development

```bash
uv sync --extra cpu            # (Linux) + dev group: pytest, ruff
uv run pytest -q               # 61 tests; tiny random models, no downloads needed
MEDASR_TEST_MODEL=ainergiz/medasr-mlx-fp16 uv run pytest -m slow   # real-weight regression
uv run --extra reference python tools/validate_hf.py              # parity vs PyTorch (gated weights)
```

## Roadmap

- **Apple Silicon benchmarks of v2**: GPU latency, fp16 vs v1's implicit fp32, and quantized variants. Publish
  v2-format checkpoints.
- **Swift / iOS**: port to mlx-swift, or contribute MedASR to mlx-audio / mlx-audio-swift.
- **Streaming context biasing** with CTC word spotting ([arXiv:2605.18222](https://arxiv.org/abs/2605.18222)) and
  beam search in the streaming path.
- **LM tuning**: an `alpha`/`beta` sweep tool, and a vectorised (MLX) beam search for very long audio.
- **VAD gating** for streaming (skip silent windows) and **diarization** for clinician–patient conversations.
- **Dictation formatting**: spoken punctuation and voice commands ("new paragraph"), and number/date normalisation.
- **LLM post-processing with MedGemma**: structured notes and error correction, as the model card suggests.
- **Fine-tuning in MLX** (CTC / CR-CTC as in the report) for local vocabulary adaptation.
- A local **OpenAI-compatible server** (`/v1/audio/transcriptions`) with WebSocket streaming.

## License

Code: Apache 2.0. Model weights are governed by the
[Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).
MedASR is not a medical device; outputs require independent verification (see the model card).

## Acknowledgments

[Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations) for MedASR and
its technical report, [Apple MLX](https://github.com/ml-explore/mlx), and [Eka Care](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset)
for the open medical ASR evaluation set.
