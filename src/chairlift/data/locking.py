"""Appending one line to a shared file from many processes: an exclusive advisory lock around each append.

Manifests, ledgers and stat caches are append-only JSON lines. A single short write is atomic on a local disk, but
not guaranteed on NFS, and parallel search candidates append from several processes at once. The lock (fcntl.lockf,
which NFS v4 honours) makes each append whole.
"""

from __future__ import annotations

import fcntl
from pathlib import Path


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fcntl.lockf(fh, fcntl.LOCK_EX)
        try:
            fh.write(line if line.endswith("\n") else line + "\n")
            fh.flush()
        finally:
            fcntl.lockf(fh, fcntl.LOCK_UN)
