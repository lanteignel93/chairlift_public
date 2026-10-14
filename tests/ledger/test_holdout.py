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


# ---- outcome-free windows: a horizon instead of an exit date ---------------------------------------------------------

SEAL = dt.date(2024, 1, 1)


def _trades(stop: float, seed: int = 7) -> pl.DataFrame:
    """A twin crossing the seal: one trade a day from mid-November 2023 into the next month-end expiry, each path marked
    daily. A trade whose path falls below −`stop` exits that day; the others are held to expiry. Same paths whatever the
    stop: only the exits move."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for i in range(70):
        t0 = dt.date(2023, 11, 15) + dt.timedelta(days=i)
        nxt = (t0.replace(day=1) + dt.timedelta(days=62)).replace(day=1) - dt.timedelta(days=1)  # next month's end
        days = [t0 + dt.timedelta(days=k) for k in range((nxt - t0).days + 1)]
        path = np.cumsum(rng.normal(0.0, 1.0, len(days)))
        hit = np.flatnonzero(path < -stop)
        end = int(hit[0]) if len(hit) else len(days) - 1
        rows += [
            {"t0": t0, "date": d, "expiry": nxt, "exit_date": days[end], "pnl": float(v)}
            for d, v in zip(days[: end + 1], path[: end + 1], strict=True)
        ]
    return pl.DataFrame(rows)


def _windows(tmp_path: Path, frame: pl.DataFrame, h: Holdout, opened: bool) -> set[dt.date]:
    def src(s: Holdout) -> pl.DataFrame:
        return guard(frame, s)

    content = str(frame.hash_rows().sum())
    root = tmp_path / f"{h.horizon}-{h.column}-{opened}-{content}"
    if opened:
        Ledger(root / "ledger.jsonl").open_holdout(reason="test", by="me")
    stage = Stage("src", src, spec=h, fingerprint=with_holdout(lambda: content), ignore=("frame", "content"))
    p = Pipeline([stage], root)
    return set(p.load("src", p.run())["t0"].to_list())


def test_the_windows_do_not_depend_on_how_the_rows_ended(tmp_path: Path):
    tight, loose = _trades(stop=0.5), _trades(stop=50.0)
    by_horizon = Holdout(start=SEAL, column="t0", horizon="expiry", early_exit=True)
    by_exit = Holdout(start=SEAL, column="exit_date")
    for opened in (False, True):
        assert _windows(tmp_path, tight, by_horizon, opened) == _windows(tmp_path, loose, by_horizon, opened)
    # sealing on the exit date keeps the stopped straddlers in dev: the dev set moves with the stop level
    assert _windows(tmp_path, tight, by_exit, False) != _windows(tmp_path, loose, by_exit, False)


def test_straddling_rows_are_in_no_window_and_are_counted(tmp_path: Path):
    from chairlift.run.events import read_events

    frame = _trades(stop=50.0)
    h = Holdout(start=SEAL, column="t0", horizon="expiry")
    dev = _windows(tmp_path, frame, h, opened=False)
    both = _windows(tmp_path, frame, h, opened=True)
    first = frame.group_by("t0").agg(pl.col("expiry").first())
    straddling = set(first.filter((pl.col("t0") < SEAL) & (pl.col("expiry") >= SEAL))["t0"].to_list())
    assert straddling and not straddling & both
    assert dev == set(first.filter(pl.col("expiry") < SEAL)["t0"].to_list())
    assert both - dev == set(first.filter(pl.col("t0") >= SEAL)["t0"].to_list())
    root = tmp_path / f"expiry-t0-False-{frame.hash_rows().sum()}"
    evs = [e for f in sorted((root / "runs").glob("*.events.jsonl")) for e in read_events(f)[0]]
    counted = [e["value"] for e in evs if e.get("kind") == "metric" and e.get("name") == "holdout_straddling_rows"]
    assert counted == [float(frame.filter(pl.col("t0").is_in(list(straddling))).height)]


def test_rows_that_end_early_need_a_horizon(tmp_path: Path):
    frame = _trades(stop=0.5)
    with pytest.raises(ValueError, match="horizon"):
        _windows(tmp_path, frame, Holdout(start=SEAL, column="exit_date", early_exit=True), opened=False)


def test_the_added_fields_keep_every_existing_seal_s_identity():
    from chairlift.core.spec import canonical

    assert set(canonical(Holdout(start=SEAL, column="exit_date"))) == {"__spec__", "__version__", "start", "column"}
