from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from medasr_mlx.audio import as_waveform, load_audio, resample_poly, to_mono
from medasr_mlx.config import FrontendConfig
from medasr_mlx.features import batch_log_mel, log_mel_spectrogram, mel_filterbank

DATA = Path(__file__).parent / "data"


def test_matches_reference_extractor_golden():
    """Bit-level parity with transformers.LasrFeatureExtractor (golden fixture)."""
    audio = np.load(DATA / "chirp_input.npy")
    ref = np.load(DATA / "chirp_lasr_features.npy")
    ours = log_mel_spectrogram(audio)
    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-5)


def test_matches_reference_extractor_live():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    fe = transformers.LasrFeatureExtractor()
    rng = np.random.default_rng(0)
    audios = [rng.standard_normal(n).astype(np.float32) * 0.1 for n in (16000, 23456)]
    ref = fe(audios, sampling_rate=16000, return_tensors="np", return_attention_mask=True)
    feats, mask = batch_log_mel(audios)
    np.testing.assert_allclose(feats, ref["input_features"], atol=1e-5)
    assert (mask == ref["attention_mask"].astype(bool)).all()


def test_frame_count_and_short_input():
    cfg = FrontendConfig()
    assert cfg.num_frames(399) == 0
    assert cfg.num_frames(400) == 1
    assert cfg.num_frames(16000) == 98
    assert log_mel_spectrogram(np.zeros(100, np.float32)).shape == (0, 128)
    silent = log_mel_spectrogram(np.zeros(1600, np.float32))
    np.testing.assert_allclose(silent, np.log(1e-5), rtol=1e-6)


def test_mel_filterbank_shape():
    fb = mel_filterbank()
    assert fb.shape == (257, 128)
    assert (fb[0] == 0).all()  # DC bin excluded (HTK-style)
    assert (fb >= 0).all() and fb.max() <= 1.0


@pytest.mark.parametrize("sr", [8000, 22050, 44100, 48000])
def test_resample_preserves_tone(sr):
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    y = resample_poly(x, 16000, sr)
    assert abs(y.size - 16000) <= 1
    spectrum = np.abs(np.fft.rfft(y[1000:-1000]))
    peak_hz = np.argmax(spectrum) * 16000 / (y.size - 2000)
    assert abs(peak_hz - 440) < 3


def test_resample_matches_scipy():
    signal = pytest.importorskip("scipy.signal")
    x = np.random.default_rng(0).standard_normal(44100).astype(np.float32)
    np.testing.assert_allclose(
        resample_poly(x, 160, 441), signal.resample_poly(x.astype(np.float64), 160, 441), atol=1e-5
    )


def test_load_audio_roundtrip(tmp_path):
    t = np.arange(48000) / 48000
    stereo = np.stack([np.sin(2 * np.pi * 300 * t), np.sin(2 * np.pi * 300 * t)], 1) * 0.5
    sf.write(tmp_path / "x.wav", stereo, 48000, subtype="PCM_16")
    y = load_audio(tmp_path / "x.wav")
    assert y.dtype == np.float32 and y.ndim == 1
    assert abs(y.size - 16000) <= 1
    assert 0.45 < np.abs(y).max() < 0.55
    np.testing.assert_allclose(as_waveform(y), y)


def test_to_mono_int16():
    x = (np.ones((10, 2)) * 16384).astype(np.int16)
    np.testing.assert_allclose(to_mono(x), 0.5)
