"""The holdout gate: rows from `start` on are invisible until the study's ledger records the one opening.

A source stage reads its data through `guard(frame, holdout)` and declares `fingerprint=with_holdout(fp)`, so the
gate's state is part of the stage's identity: opening the holdout changes the key, the stage re-runs, and every
result built on holdout rows is a different artifact from every result built without them.

Which rows are sealed must not depend on how a row turned out. Sealing on an exit date does when a row can end early
(a stop): of the rows entered before the seal into a later horizon, the stopped ones exit in time to stay in dev and
the held ones are sealed, so dev keeps the losers or the winners of the straddling rows depending on the trade. Such a
study declares the row's `horizon` (the date its outcome is final at the latest, an option's expiry):
- dev: rows whose horizon is before `start`
- holdout: rows entered (`column`) on or after `start`
- neither: rows entered before `start` into a horizon on or after it. They straddle the seal, are dropped in both
  states, and are counted in the run's events.

Both dates are known when the row is entered, so the windows are the same whatever happens to the rows. A study whose
rows end on a fixed horizon (exit = entry + h) may keep sealing on one column. `early_exit=True` declares that rows can
end early, and makes the gate refuse a seal without a horizon.

The Pipeline binds the study's ledger while it fingerprints and runs stages, the way it binds data roots.
"""

from __future__ import annotations

import contextvars
import datetime as dt
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from chairlift.core.spec import added, spec
from chairlift.ledger.trials import Ledger
from chairlift.run import events

_LEDGER: contextvars.ContextVar[Ledger | None] = contextvars.ContextVar("chairlift_ledger", default=None)


@spec(name="Holdout", version=1)
class Holdout:
    start: dt.date
    column: str = "t0"  # the entry date; without a horizon, the column the seal compares with `start`
    horizon: str | None = added(None)  # the date a row's outcome is final at the latest (outcome-free windows)
    early_exit: bool = added(False)  # rows can end before their horizon: a seal without a horizon is refused


@contextmanager
def bind_ledger(path: Path) -> Generator[None]:
    token = _LEDGER.set(Ledger(path))
    try:
        yield
    finally:
        _LEDGER.reset(token)


def _ledger() -> Ledger:
    led = _LEDGER.get()
    if led is None:
        raise RuntimeError("the holdout gate needs a study ledger: read data through a Pipeline run")
    return led


def state() -> str:
    opened = _ledger().holdout_opened()
    return f"open@{opened['at']}" if opened else "sealed"


def guard(frame: pl.DataFrame, h: Holdout) -> pl.DataFrame:
    """The frame without its holdout rows, unless the ledger records the opening; with a horizon, never the rows that
    straddle the seal."""
    if h.horizon is None:
        if h.early_exit:
            raise ValueError(
                f"rows that can end early cannot be sealed on {h.column!r} alone: which side of {h.start} a row lands"
                " on would depend on how it ended. Declare the row's horizon: Holdout(start, column=<entry date>,"
                " horizon=<the date its outcome is final at the latest>)"
            )
        if _ledger().holdout_opened() is not None:
            n = frame.filter(pl.col(h.column) >= h.start).height
            events.warning(f"holdout open: {n} rows from {h.start} are in this stage")
            return frame
        return frame.filter(pl.col(h.column) < h.start)
    dev = pl.col(h.horizon) < h.start
    held = pl.col(h.column) >= h.start
    n = frame.filter(~dev & ~held).height
    events.metric("holdout_straddling_rows", float(n))
    if n:
        events.warning(f"{n} rows straddle the seal at {h.start} (entered before it, final after it): in no window")
    if _ledger().holdout_opened() is not None:
        events.warning(f"holdout open: {frame.filter(held).height} rows entered from {h.start} are in this stage")
        return frame.filter(dev | held)
    return frame.filter(dev)


def with_holdout(fingerprint: Callable[[], str]) -> Callable[[], str]:
    """A source fingerprint that also changes when the holdout is opened."""

    def fp() -> str:
        return f"{fingerprint()}|holdout={state()}"

    return fp
