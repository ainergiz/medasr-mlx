import numpy as np
import pytest

from medasr_mlx.evaluation import Scores, align, cer, entity_word_mask, normalize_text, wer


def test_ctc_collapse_and_decode(tiny_tokenizer):
    tok = tiny_tokenizer
    ids = [0, 5, 5, 0, 6, 7, 7, 0, 0, 8, 9, 14]  # ▁the ▁pa tient ▁has ▁fever .
    assert tok.decode(ids, collapse=True) == "the patient has fever."
    # repeats separated by blank are distinct tokens
    assert tok.decode([12, 0, 12], collapse=True) == "cc"
    assert tok.decode([12, 12], collapse=True) == "c"
    # special tokens are skipped; the first ▁ is dropped
    assert tok.decode([1, 2, 3, 4, 10]) == " a"


def test_encode_viterbi(tiny_tokenizer):
    tok = tiny_tokenizer
    assert tok.decode(tok.encode("the patient has fever")) == "the patient has fever"
    ids = tok.encode("metoprolol")
    assert [tok.pieces[i] for i in ids] == ["▁met", "o", "pro", "lol"]
    assert tok.unk_id in tok.encode("zz")


def test_parity_with_reference_tokenizer():
    """Decode parity with transformers' LasrTokenizer on random CTC paths."""
    pytest.importorskip("transformers")
    hf_hub = pytest.importorskip("huggingface_hub")
    from transformers import AutoTokenizer

    from medasr_mlx.tokenizer import CTCTokenizer

    try:
        root = hf_hub.snapshot_download("ainergiz/medasr-mlx-fp16", allow_patterns=["processor/*"])
    except Exception as exc:  # offline
        pytest.skip(f"tokenizer download failed: {exc}")
    # tokenizer_config.json is needed so AutoTokenizer resolves LasrTokenizer (CTC grouping)
    hf = AutoTokenizer.from_pretrained(f"{root}/processor")
    tok = CTCTokenizer.from_file(f"{root}/processor")
    rng = np.random.default_rng(0)
    for _ in range(300):
        ids = rng.integers(0, 512, size=int(rng.integers(0, 50)))
        ids[rng.random(ids.size) < 0.5] = 0
        assert (
            tok.decode(ids, collapse=True)
            == hf.batch_decode(ids[None], skip_special_tokens=True)[0]
        )
    for text in ("Metoprolol 25 mg twice daily.", "pt c/o SOB x3 days; 98.6°F"):
        assert tok.encode(text) == hf(text, add_special_tokens=False)["input_ids"]


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("Take Dolo 650, twice a day for six days.", "take dolo 650 twice a day for 6 days"),
        ("twenty five milligrams", "25 mg"),
        ("one hundred and twenty five", "125"),
        ("five six seven", "5 6 7"),
        ("Uh, the patient [unintelligible] has 2% ESR {period}", "the patient has 2 percent esr"),
        ("L5-S1 micro-lumbar", "l5 s1 micro lumbar"),
        ("12.5 mg/5 mg tablet.", "12.5 mg 5 mg tablet"),
        ("500mg", "500 mg"),
        ("Hmm, hmm. Okay okay.", "okay okay"),
    ],
)
def test_normalize_text(raw, norm):
    assert normalize_text(raw) == norm


def test_wer_cer_alignment():
    assert wer(["the cat sat on the mat"], ["the cat sit on mat"]) == pytest.approx(2 / 6)
    a = align(["a", "b", "c", "d"], ["a", "x", "c"])
    assert (a.substitutions, a.deletions, a.insertions) == (1, 1, 0)
    assert a.ref_errors == [False, True, False, True]
    assert cer(["abc"], ["abd"]) == pytest.approx(1 / 3)


def test_entity_metrics():
    ref = "Patient has fever, prescribe azithromycin 500 mg."
    span = (ref.index("azithromycin"), ref.index("azithromycin") + len("azithromycin"))
    words, mask = entity_word_mask(ref, [span])
    assert [w for w, m in zip(words, mask) if m] == ["azithromycin"]
    s = Scores()
    s.add(ref, "patient has fever prescribe azithromycin 500 milligrams", [span])
    s.add(ref, "patient has fever prescribe a zithromycin 500 mg", [span])
    summary = s.summary()
    assert summary["term_recall"] == 0.5
    assert summary["kwer"] == 0.5
