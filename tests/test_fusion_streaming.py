import itertools

import mlx.core as mx
import numpy as np
import pytest

from medasr_mlx.config import ModelConfig
from medasr_mlx.fusion import (
    PosteriorAccumulator,
    fuse_windows,
    hann_weights,
    log_softmax,
    plan_windows,
    run_windows,
)
from medasr_mlx.transcribe import MedASR
from medasr_mlx.weights import load_model

from .conftest import tiny_config_dict

CFG = ModelConfig.from_dict(tiny_config_dict())


@pytest.fixture(scope="module")
def tiny_asr(tiny_checkpoint, tiny_tokenizer):
    model, _ = load_model(tiny_checkpoint, dtype=mx.float32)
    return MedASR(model, tiny_tokenizer, compile=False)


@pytest.mark.parametrize("seconds", [0.2, 5.0, 20.0, 20.04, 20.9, 37.3, 61.0, 125.17])
def test_plan_covers_audio_on_frame_grid(seconds):
    n = int(seconds * 16000)
    plan = plan_windows(n, CFG, window_s=20.0, stride_s=18.0)
    assert plan.total_frames == CFG.num_output_frames(n)
    covered = np.zeros(plan.total_frames, bool)
    for w in plan.windows:
        assert w.start % CFG.samples_per_frame == 0
        assert w.frame_offset == w.start // CFG.samples_per_frame
        assert w.num_frames == CFG.num_output_frames(w.length)
        assert w.start + w.length <= n
        covered[w.frame_offset : w.frame_offset + w.num_frames] = True
    assert covered.all()
    assert plan.windows[-1].start + plan.windows[-1].length == n
    if n <= 20 * 16000:
        assert len(plan.windows) == 1


def test_window_frames_align_with_full_pass(tiny_asr, speechlike):
    """Frame j of a window starting at sample B is global frame B/640 + j."""
    wave = np.tile(speechlike, 3)
    full = tiny_asr.forward([wave])[0]
    start = 640 * 37
    win = tiny_asr.forward([wave[start:]])[0]
    assert full.shape[0] - win.shape[0] == 37


def test_fusion_identity_and_center_mode():
    v = 6
    rng = np.random.default_rng(0)
    plan = plan_windows(45 * 16000, CFG, window_s=20.0, stride_s=10.0)
    base = log_softmax(rng.standard_normal((plan.total_frames, v)).astype(np.float32))
    outs = [base[w.frame_offset : w.frame_offset + w.num_frames] for w in plan.windows]
    for mode in ("hann", "center"):
        np.testing.assert_allclose(fuse_windows(plan, outs, v, mode=mode), base, atol=1e-5)
    assert (hann_weights(10) > 0).all() and hann_weights(9).argmax() == 4


def test_accumulator_weighted_average():
    acc = PosteriorAccumulator(2)
    a = np.log(np.array([[0.9, 0.1]] * 3, np.float32))
    b = np.log(np.array([[0.1, 0.9]] * 3, np.float32))
    acc.add(0, a)
    acc.add(1, b)
    p = np.exp(acc.log_probs())
    assert p.shape == (4, 2)
    np.testing.assert_allclose(p[0], [0.9, 0.1], atol=1e-6)
    np.testing.assert_allclose(p[3], [0.1, 0.9], atol=1e-6)
    w = hann_weights(3)  # frame 1: a at pos 1 (w[1]), b at pos 0 (w[0])
    expect = (w[1] * 0.9 + w[0] * 0.1) / (w[1] + w[0])
    np.testing.assert_allclose(p[1, 0], expect, atol=1e-6)


def test_long_form_transcribe_runs_windows(tiny_asr, speechlike):
    wave = np.tile(speechlike, 8)  # 26.4 s -> two windows
    r = tiny_asr.transcribe(wave)
    assert r.metadata["frames"] == CFG.num_output_frames(wave.size)
    single = tiny_asr.transcribe(wave, window_s=None)
    assert single.metadata["frames"] == r.metadata["frames"]
    outs = run_windows(plan_windows(wave.size, CFG), wave, tiny_asr.forward)
    assert len(outs) == 2


@pytest.mark.parametrize("start_padding", [False, True])
def test_streaming_is_chunking_invariant_and_matches_offline(tiny_asr, speechlike, start_padding):
    wave = np.tile(speechlike, 3)
    results = []
    for chunk in (1234, 5120, 16000):
        st = tiny_asr.stream(
            window_s=4.0, stride_s=0.64, commit_delay_s=0.5, start_padding=start_padding
        )
        texts = []
        for i in range(0, wave.size, chunk):
            up = st.push(wave[i : i + chunk])
            texts.append(up.committed_text)
            assert up.text.startswith(up.committed_text)
        # committed text only ever grows
        assert all(b.startswith(a) for a, b in itertools.pairwise(texts))
        final = st.finish()
        assert final.metadata["streaming_committed_text"].startswith(texts[-1])
        assert st.commit_frame == st.acc.num_frames  # finish() flushes the tentative tail
        results.append((final, st.acc.log_probs()))
    for res, lp in results[1:]:
        assert res.text == results[0][0].text
        np.testing.assert_allclose(lp, results[0][1], atol=1e-5)
    # Recompute the same windows offline and fuse them: must equal the stream's posteriors.
    st = tiny_asr.stream(window_s=4.0, stride_s=0.64, start_padding=start_padding)
    pad = st.pad
    padded = np.concatenate([np.zeros(pad, np.float32), wave])
    acc = PosteriorAccumulator(CFG.vocab_size)
    ends = list(range(st.stride, wave.size + 1, st.stride))
    if ends[-1] != wave.size:
        ends.append(wave.size)
    for end in ends:
        pend = end + pad
        start = max(0, (pend - st.window) // 640 * 640)
        if pend - start < CFG.min_samples and end != wave.size:
            continue
        acc.add(start // 640, tiny_asr.forward([padded[start:pend]])[0], drop_before=pad // 640)
    np.testing.assert_allclose(results[0][1], acc.log_probs(), atol=1e-5)


def test_streaming_empty_and_tiny(tiny_asr):
    st = tiny_asr.stream()
    assert st.finish().text == ""
    st = tiny_asr.stream()
    st.push(np.zeros(800, np.float32))
    st.finish()  # shorter than the receptive field: must not raise


def test_compiled_path_matches_eager(tiny_checkpoint, tiny_tokenizer, speechlike):
    model, _ = load_model(tiny_checkpoint, dtype=mx.float32)
    eager = MedASR(model, tiny_tokenizer, compile=False)
    compiled = MedASR(model, tiny_tokenizer, compile=True)
    wave = np.tile(speechlike, 8)  # multi-window -> compiled fixed-shape batches
    np.testing.assert_allclose(compiled.log_probs(wave), eager.log_probs(wave), atol=1e-5)
