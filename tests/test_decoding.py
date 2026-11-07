import itertools
import math

import numpy as np
import pytest

from medasr_mlx.decoding.beam import BeamSearchDecoder
from medasr_mlx.decoding.biasing import PhraseBooster
from medasr_mlx.decoding.greedy import greedy_decode
from medasr_mlx.decoding.lm import ArpaTokenLM, write_token_arpa
from medasr_mlx.result import TimeBase, group_segments, group_words

V = 22  # tiny tokenizer size


def peaky(path, v=V, p=0.98):
    """Log-posteriors putting probability ``p`` on each frame's token."""
    lp = np.full((len(path), v), math.log((1 - p) / (v - 1)), np.float32)
    lp[np.arange(len(path)), path] = math.log(p)
    return lp


def test_greedy_spans(tiny_tokenizer):
    path = [0, 5, 5, 0, 6, 7, 0, 0, 8, 0]
    spans = greedy_decode(peaky(path), tiny_tokenizer)
    assert [(s.id, s.start_frame, s.end_frame) for s in spans] == [
        (5, 1, 2),
        (6, 4, 4),
        (7, 5, 5),
        (8, 8, 8),
    ]
    assert all(0.97 < s.confidence < 0.99 for s in spans)


def test_words_and_timestamps(tiny_tokenizer):
    spans = greedy_decode(peaky([0, 5, 0, 6, 7, 0, 8, 9, 14, 0]), tiny_tokenizer)
    tb = TimeBase(0.04, 0.0725, 10.0)
    words = group_words(spans, tiny_tokenizer, tb)
    assert [w.text for w in words] == ["the", "patient", "has", "fever."]
    assert words[1].start == pytest.approx(3 * 0.04 + 0.0525, abs=1e-3)
    assert all(a.end <= b.start + 1e-9 for a, b in itertools.pairwise(words))
    segs = group_segments(words)
    assert " ".join(s.text for s in segs) == "the patient has fever."


def test_beam_matches_greedy_on_peaky_input(tiny_tokenizer):
    lp = peaky([0, 5, 5, 0, 6, 7, 0, 0, 8, 0, 9, 14, 0])
    g = [s.id for s in greedy_decode(lp, tiny_tokenizer)]
    b = [s.id for s in BeamSearchDecoder(beam_width=4)(lp, tiny_tokenizer)]
    assert g == b


def test_beam_beats_best_path(tiny_tokenizer):
    """Classic case: best path is all-blank, but the label 'a' is more probable."""
    lp = np.full((2, V), -30.0, np.float32)
    lp[:, 0] = math.log(0.6)
    lp[:, 10] = math.log(0.4)
    assert greedy_decode(lp, tiny_tokenizer) == []
    out = BeamSearchDecoder(beam_width=4, blank_skip=1.0)(lp, tiny_tokenizer)
    assert [s.id for s in out] == [10]  # P('a') = 0.64 > P('') = 0.36


def test_phrase_booster_refunds_partial_matches():
    b = PhraseBooster(["metoprolol", "pantop dsr"], weight=10.0)
    state, total = b.initial_state(), 0.0
    for piece in [" met", "o", "pro", "lol", ","]:
        state, d = b.advance(state, piece)
        total += d
    assert total == pytest.approx(10.0)
    state, total = b.initial_state(), 0.0
    for piece in [" met", "o", "pro", "lols"]:  # overruns the phrase -> refunded
        state, d = b.advance(state, piece)
        total += d
    assert total + b.finalize(state) == pytest.approx(0.0)
    state, total = b.initial_state(), 0.0
    for piece in [" pan", "top", " dsr"]:  # multi-word phrase across a word boundary
        state, d = b.advance(state, piece)
        total += d
    assert total + b.finalize(state) == pytest.approx(10.0)
    state, d = b.advance(b.initial_state(), " xmetoprolol")  # must start at a word boundary
    assert d == 0


def test_hotword_flips_ambiguous_decision(tiny_tokenizer):
    # Frames are ambiguous between "▁dolo" (20) and "▁the" (5); acoustics prefer "▁the".
    lp = np.full((5, V), -20.0, np.float32)
    lp[[0, 2, 4], 0] = 0.0
    lp[1, 5], lp[1, 20] = math.log(0.6), math.log(0.4)
    lp[3, 21] = 0.0  # "▁650"
    plain = BeamSearchDecoder(beam_width=4)(lp, tiny_tokenizer)
    assert tiny_tokenizer.decode([s.id for s in plain]) == "the 650"
    biased = BeamSearchDecoder(beam_width=4, hotwords=["dolo"], hotword_weight=2.0)(
        lp, tiny_tokenizer
    )
    assert tiny_tokenizer.decode([s.id for s in biased]) == "dolo 650"


def test_token_lm_changes_ranking(tiny_tokenizer, tmp_path):
    arpa = write_token_arpa(
        ["the patient has fever"] * 20, tiny_tokenizer, tmp_path / "lm.arpa", order=3
    )
    lm = ArpaTokenLM(arpa, tiny_tokenizer)
    # acoustically "▁the ▁pa tient ▁a" (a: 0.55) vs "... ▁has" (0.45)
    path = [0, 5, 0, 6, 7, 0, 8, 0]
    lp = peaky(path)
    lp[6, 8], lp[6, 10] = math.log(0.45), math.log(0.55)
    no_lm = BeamSearchDecoder(beam_width=8)(lp, tiny_tokenizer)
    assert tiny_tokenizer.decode([s.id for s in no_lm]) == "the patient a"
    with_lm = BeamSearchDecoder(beam_width=8, lm=lm, alpha=1.0, beta=0.0)(lp, tiny_tokenizer)
    assert tiny_tokenizer.decode([s.id for s in with_lm]) == "the patient has"


def test_arpa_reader_matches_kenlm(tiny_tokenizer, tmp_path):
    pytest.importorskip("kenlm")
    from medasr_mlx.decoding.lm import KenLMTokenLM

    texts = ["the patient has fever", "a patient has a fever", "metoprolol dolo 650 mg"]
    arpa = write_token_arpa(texts * 3, tiny_tokenizer, tmp_path / "lm.arpa", order=3)
    a = ArpaTokenLM(arpa, tiny_tokenizer)
    try:
        k = KenLMTokenLM(arpa, tiny_tokenizer)
    except ValueError:
        pytest.skip("tiny vocab coverage check")
    rng = np.random.default_rng(0)
    for _ in range(50):
        sa, sk = a.initial_state(), k.initial_state()
        for t in rng.integers(4, V, size=8):
            la, sa = a.score(sa, int(t))
            lk, sk = k.score(sk, int(t))
            assert la == pytest.approx(lk, abs=1e-4)
