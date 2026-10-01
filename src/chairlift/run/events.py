"""The event stream: what happened during a run, append-only, one JSON object per line.

The runner is the only writer of a run's event file (`<root>/runs/<run_id>.events.jsonl`). Readers — the live view,
`chairlift watch`, reports, alert rules — keep their own position and never slow the writer. In-process listeners
(the live view during `chairlift run`) are called synchronously but defensively: an exception in a listener is
swallowed, because observation must never fail a run.

Every line carries `kind` and `schema`; readers skip kinds they do not know, so new kinds never break old readers.

Stages report through module functions that write to the run currently bound by the runner (a context variable),
so a stage stays a pure function of its spec and inputs and the calls do nothing outside a run:

    from chairlift.run import events
    events.progress(3, 7, "fold 3 of 7")
    events.metric("oos_ic", 0.031, fold=3)
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
from collections.abc import Callable, Generator
from contextvars import ContextVar
from pathlib import Path
from typing import Any

SCHEMA = 1
KINDS = {"run_started", "stage_started", "progress", "metric", "warning", "stage_finished", "run_finished"}

Listener = Callable[[dict[str, Any]], None]


class EventLog:
    """The single writer of one run's event file."""

    def __init__(self, path: Path, run_id: str, listeners: list[Listener] | None = None) -> None:
        self.path = path
        self.run_id = run_id
        self.listeners = list(listeners or [])
        self.seq = 0
        self.listener_errors = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, **payload: Any) -> dict[str, Any]:
        self.seq += 1
        rec = {
            "kind": kind,
            "schema": SCHEMA,
            "run_id": self.run_id,
            "seq": self.seq,
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"),
            **payload,
        }
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        for listener in self.listeners:
            try:
                listener(rec)
            except Exception:  # an observer must never fail the run it observes
                self.listener_errors += 1
        return rec


_LOG: ContextVar[EventLog | None] = ContextVar("chairlift_event_log", default=None)
_STAGE: ContextVar[str | None] = ContextVar("chairlift_stage", default=None)


@contextlib.contextmanager
def bound(log: EventLog, stage: str | None = None) -> Generator[None]:
    """Make `log` (and `stage`) the target of progress / metric / warning calls inside the block."""
    t1, t2 = _LOG.set(log), _STAGE.set(stage)
    try:
        yield
    finally:
        _STAGE.reset(t2)
        _LOG.reset(t1)


def progress(done: int, total: int | None = None, message: str = "") -> None:
    """Report progress inside a stage, for example folds done out of total. A no-op outside a run."""
    log = _LOG.get()
    if log is not None:
        log.emit("progress", stage=_STAGE.get(), done=done, total=total, message=message)


def metric(name: str, value: float, **dims: Any) -> None:
    """Report a named value, with optional dimensions such as fold or year. A no-op outside a run."""
    log = _LOG.get()
    if log is not None:
        log.emit("metric", stage=_STAGE.get(), name=name, value=value, dims=dims)


def warning(message: str) -> None:
    """Report something a reader should notice without failing the stage. A no-op outside a run."""
    log = _LOG.get()
    if log is not None:
        log.emit("warning", stage=_STAGE.get(), message=message)


def read_events(path: Path, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Complete events from byte `offset` on, and the offset after the last complete line.

    A partial last line (the writer mid-write) is left for the next read; unknown kinds and later schemas are
    returned as they are, and it is the reader's choice to skip them.
    """
    if not path.exists():
        return [], offset
    with path.open("rb") as fh:
        fh.seek(offset)
        data = fh.read()
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    out: list[dict[str, Any]] = []
    for line in data[: end + 1].splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a corrupt line is skipped, never fatal to a reader
    return out, offset + end + 1
