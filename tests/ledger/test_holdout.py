"""ledger/holdout.py: holdout rows are invisible until the ledger records the opening, and opening re-keys."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from chairlift.ledger.holdout import Holdout, guard, with_holdout
from chairlift.ledger.trials import Ledger
from chairlift.run.dag import Pipeline, Stage

H = Holdout(start=dt.date(2024, 1, 1), column="date")
FRAME = pl.DataFrame({"date": [dt.date(2023, 12, 29), dt.date(2024, 1, 2)], "x": [1, 2]})


def source(h: Holdout) -> pl.DataFrame:
    return guard(FRAME, h)


def test_sealed_until_opened_and_opening_changes_the_signature(tmp_path: Path):
    def pipe() -> Pipeline:
        return Pipeline([Stage("src", source, spec=H, fingerprint=with_holdout(lambda: "v1"))], tmp_path)

    sealed = pipe().run()
    assert pipe().load("src", sealed).height == 1
    Ledger(tmp_path / "ledger.jsonl").open_holdout(reason="frozen", by="me")
    opened = pipe().run()
    assert opened.signature != sealed.signature and opened.ran() == ["src"]
    assert pipe().load("src", opened).height == 2


def test_the_gate_refuses_to_answer_outside_a_run():
    with pytest.raises(RuntimeError, match="ledger"):
        guard(FRAME, H)


def test_runs_are_charged_to_the_ledger_with_their_headline(tmp_path: Path):
    p = Pipeline([Stage("ev", lambda: {"ls": {"sharpe": 1.5, "n_days": 300}}, fingerprint=lambda: "v")], tmp_path)
    p.run(headline={"stage": "ev", "sharpe": "ls.sharpe", "n_obs": "ls.n_days"})
    p.run(headline={"stage": "ev", "sharpe": "ls.sharpe", "n_obs": "ls.n_days"})
    trials = Ledger(tmp_path / "ledger.jsonl").trials()
    assert len(trials) == 1 and trials[0]["headline"] == {"sharpe": 1.5, "n_obs": 300, "periods": 252}
