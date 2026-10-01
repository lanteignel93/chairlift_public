"""verify/toy.py: the synthetic study recovers what was planted, and its folds respect the embargo."""

from __future__ import annotations

from pathlib import Path

from chairlift.verify.toy import FoldSpec, SourceSpec, folds, pipeline, sources


def test_planted_ic_is_recovered_out_of_sample(tmp_path: Path):
    p = pipeline(tmp_path / "r", ic=0.1)
    ev = p.load("evaluate", p.run())
    assert 0.06 < ev["ic_mean"] < 0.14 and ev["ic_t"] > 4


def test_no_signal_gives_no_ic(tmp_path: Path):
    p = pipeline(tmp_path / "r", ic=0.0)
    ev = p.load("evaluate", p.run())
    assert abs(ev["ic_mean"]) < 0.03 and abs(ev["ic_t"]) < 3


def test_sources_are_seeded_and_shaped():
    a, b = sources(SourceSpec(seed=3)), sources(SourceSpec(seed=3))
    assert a.equals(b) and a.shape == (50 * 300, 4)
    assert not a.equals(sources(SourceSpec(seed=4)))


def test_folds_leave_an_embargo_and_cover_the_test_period():
    src = sources(SourceSpec(days=300))
    fs = folds(FoldSpec(first_test=150, block=50, embargo=5), src)["folds"]
    assert [f["test_start"] for f in fs] == [150, 200, 250]
    assert all(f["test_start"] - f["train_end"] > 5 for f in fs)
    assert fs[-1]["test_end"] == 299
