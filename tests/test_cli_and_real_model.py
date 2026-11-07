import json

import numpy as np
import pytest
import soundfile as sf

from medasr_mlx.cli import main

from .conftest import real_model_dir


@pytest.fixture
def wav_file(tmp_path, speechlike):
    path = tmp_path / "clip.wav"
    sf.write(path, speechlike, 16000)
    (tmp_path / "clip.txt").write_text("the patient has fever")
    return path


def test_cli_transcribe_formats(tiny_checkpoint, wav_file, tmp_path, capsys):
    out = tmp_path / "out"
    rc = main(["transcribe", str(wav_file), "--model", str(tiny_checkpoint), "--dtype", "float32",
               "--format", "all", "--output-dir", str(out)])  # fmt: skip
    assert rc == 0
    for ext in ("txt", "json", "srt", "vtt"):
        assert (out / f"clip.{ext}").exists()
    data = json.loads((out / "clip.json").read_text())
    assert {"text", "segments", "words", "duration"} <= set(data)
    assert (out / "clip.vtt").read_text().startswith("WEBVTT")


def test_cli_beam_hotwords_eval_stream(tiny_checkpoint, wav_file, capsys):
    model = ["--model", str(tiny_checkpoint), "--dtype", "float32"]
    assert main(["transcribe", str(wav_file), *model, "--hotwords", "metoprolol,dolo"]) == 0
    capsys.readouterr()
    assert main(["eval", str(wav_file.parent), *model]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["utterances"] == 1 and summary["cer"] >= 0.0
    assert main(["stream", str(wav_file), *model, "--preset", "efficient"]) == 0


def test_cli_convert_quantized(tiny_checkpoint, tmp_path, wav_file):
    out = tmp_path / "q8"
    assert main(["convert", "--source", str(tiny_checkpoint), "--output", str(out),
                 "--bits", "8", "--group-size", "32"]) == 0  # fmt: skip
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["quantization"]["bits"] == 8
    assert main(["transcribe", str(wav_file), "--model", str(out), "--dtype", "float32"]) == 0


# ----------------------------------------------------------- real weights (opt-in)
@pytest.mark.slow
def test_real_model_librispeech_sample():
    """Set MEDASR_TEST_MODEL=ainergiz/medasr-mlx-fp16 (or a local dir) to run."""
    model_id = real_model_dir()
    if not model_id:
        pytest.skip("MEDASR_TEST_MODEL not set")
    pq = pytest.importorskip("pyarrow.parquet")
    from huggingface_hub import hf_hub_download

    import medasr_mlx
    from medasr_mlx.evaluation import wer

    path = hf_hub_download("hf-internal-testing/librispeech_asr_dummy",
                           "clean/validation-00000-of-00001.parquet", repo_type="dataset")  # fmt: skip
    rows = pq.read_table(path).to_pylist()[:5]
    asr = medasr_mlx.load(model_id)
    import io

    hyps, refs = [], []
    for r in rows:
        audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        res = asr.transcribe(audio, sample_rate=sr)
        assert res.words and res.words[0].start >= 0
        hyps.append(res.text)
        refs.append(r["text"])
    # Out-of-domain audiobook speech: MedASR is tuned for medical dictation.
    assert wer(refs, hyps) < 0.35
    # v1-parity anchor (identical across v1 and v2, fp32 and fp16 compute).
    assert hyps[1] == "nor is Mr. Chter's manner less interesting than his matter."
    assert np.isfinite(asr.log_probs(audio, sample_rate=sr)).all()
