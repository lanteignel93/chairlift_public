"""Compute settings bound for a run: how many worker processes stages may use. A site setting: it changes how fast a
result arrives, never the result, so it is never part of a key.

The CLI binds `compute.workers` from the site config; a library caller binds it with `bind_compute(n)` or leaves the
default of one (serial).
"""

from __future__ import annotations

import contextvars
from collections.abc import Generator
from contextlib import contextmanager

_WORKERS: contextvars.ContextVar[int] = contextvars.ContextVar("chairlift_workers", default=1)


def workers() -> int:
    return _WORKERS.get()


@contextmanager
def bind_compute(n: int) -> Generator[None]:
    token = _WORKERS.set(max(1, int(n)))
    try:
        yield
    finally:
        _WORKERS.reset(token)
