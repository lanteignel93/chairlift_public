"""The holdout gate: rows from `start` on are invisible until the study's ledger records the one opening.

A source stage reads its data through `guard(frame, holdout)` and declares `fingerprint=with_holdout(fp)`, so the
gate's state is part of the stage's identity: opening the holdout changes the key, the stage re-runs, and every
result built on holdout rows is a different artifact from every result built without them.

The Pipeline binds the study's ledger while it fingerprints and runs stages, the way it binds data roots.
"""

from __future__ import annotations

import contextvars
import datetime as dt
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from chairlift.core.spec import spec
from chairlift.ledger.trials import Ledger
from chairlift.run import events

_LEDGER: contextvars.ContextVar[Ledger | None] = contextvars.ContextVar("chairlift_ledger", default=None)


@spec(name="Holdout", version=1)
class Holdout:
    start: dt.date
    column: str = "t0"


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
    """The frame without its holdout rows, unless the ledger records the opening."""
    if _ledger().holdout_opened() is not None:
        n = frame.filter(pl.col(h.column) >= h.start).height
        events.warning(f"holdout open: {n} rows from {h.start} are in this stage")
        return frame
    return frame.filter(pl.col(h.column) < h.start)


def with_holdout(fingerprint: Callable[[], str]) -> Callable[[], str]:
    """A source fingerprint that also changes when the holdout is opened."""

    def fp() -> str:
        return f"{fingerprint()}|holdout={state()}"

    return fp
