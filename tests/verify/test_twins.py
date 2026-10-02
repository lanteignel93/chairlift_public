"""verify/twins.py: the planted answers are in the data, and the example studies recover them and nothing else."""

from __future__ import annotations

import runpy
from pathlib import Path

import polars as pl
import pytest

from chairlift.verify.twins import CrossSectionTwin, TimeSeriesTwin, cross_section_twin, time_series_twin

EXAMPLES = Path(__file__).parents[2] / "examples" / "studies"


def _ic(frame: pl.DataFrame, col: str, target: str = "y", by: str = "t0") -> float:
    return float(
        frame.drop_nulls(col).group_by(by).agg(ic=pl.corr(col, target, method="spearman"))["ic"].to_numpy().mean()
    )


def test_cross_section_twin_plants_what_it_says():
    t = cross_section_twin(CrossSectionTwin(names=300, dates=120))
    assert t.height == 300 * 120 and t["exit_date"].gt(t["t0"]).all()
    assert _ic(t, "f_sig0") == pytest.approx(0.06, abs=0.015)
    assert _ic(t, "f_sig1") == pytest.approx(0.04, abs=0.015)
    assert abs(_ic(t, "f_noise0")) < 0.015
    assert _ic(t, "f_fund_peek") == pytest.approx(0.05, abs=0.015)  # the leak: predictive
    assert abs(_ic(t, "f_fund")) < 0.015  # as published at t0: nothing


def test_twins_are_pure_functions_of_their_spec():
    assert cross_section_twin(CrossSectionTwin(dates=10)).equals(cross_section_twin(CrossSectionTwin(dates=10)))
    assert not cross_section_twin(CrossSectionTwin(dates=10)).equals(
        cross_section_twin(CrossSectionTwin(dates=10, seed=8))
    )


def test_time_series_twin_plants_drift_and_predictability():
    t = time_series_twin(TimeSeriesTwin(days=20000))
    c = t.select(pl.corr("x_sig0", "r_next")).item()
    assert c == pytest.approx(0.05, abs=0.02) and abs(t.select(pl.corr("x_noise0", "r_next")).item()) < 0.03
    assert t["r_next"].mean() == pytest.approx(0.0003, abs=0.0002)


def test_equity_ls_recovers_the_signal_flags_the_leak_and_finds_nothing_in_the_null(tmp_path: Path):
    pytest.importorskip("lightgbm")
    study = runpy.run_path(str(EXAMPLES / "equity_ls.py"))

    def ic(**kw: object) -> float:
        p = study["study"](**kw).pipeline(tmp_path / "r")
        return p.load("eval_ridge", p.run(["eval_ridge"]))["ic"]["oos_ic_mean"]

    honest, leak, null = ic(), ic(fundamentals="peek"), ic(signal_ic=0.0)
    assert 0.035 < honest < 0.065  # the planted signals, through a walk-forward
    # the look-ahead flatters the book: planted ICs add roughly in quadrature, so expect +0.010 to +0.016
    assert leak > honest + 0.007
    assert abs(null) < 0.02


def test_spy_timing_on_the_null_twin_does_not_beat_buy_and_hold(tmp_path: Path):
    study = runpy.run_path(str(EXAMPLES / "spy_timing.py"))
    p = study["study"](beta=0.0).pipeline(tmp_path / "r")
    r = p.run()
    d = p.load("report", r)["books"]["ridge"]["vs_buy_and_hold"]
    assert d["d_sharpe_ci"][0] < 0 < d["d_sharpe_ci"][1]
    assert abs(p.load("eval_ridge", r)["ic"]["oos_ic_all"]) < 0.03
