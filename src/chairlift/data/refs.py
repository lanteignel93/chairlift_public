"""Data references: studies name data by alias, never by path, and identify it by content.

`DataRef(alias="vendor", relpath="option_close")` is part of a study's definition and of its specs; the site config's
`[data]` table maps the alias to a directory on this machine. Its fingerprint is a blake3 hash of the bytes (a file) or
a Merkle hash over sorted (relative path, file hash) pairs (a directory), so the same data copied to two machines, at
two different paths, gives one fingerprint and therefore one run signature. Dot-files are not data and are skipped.

Hashing every byte on every plan would be slow, so a stat cache maps (path, size, mtime_ns, inode) to the hash. The
cache is an optimization, never the source of truth: any change to the stat tuple re-hashes. It is one append-only
file per host under the cache directory, the same single-writer rule as the manifest.

Resolution is bound per run: the Pipeline binds its data roots and cache while it fingerprints and while stages run,
so a stage calls `ref.path()` and a source stage declares `fingerprint=fingerprint_of(ref)`.
"""

from __future__ import annotations

import contextvars
import json
import os
import socket
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import blake3

from chairlift.core.spec import spec
from chairlift.data.locking import append_line


class DataError(LookupError):
    """A data alias is unknown, unbound, or points at nothing."""


@spec(name="data_ref", version=1)
class DataRef:
    alias: str
    relpath: str = ""

    def path(self) -> Path:
        """Where this data is on this machine, through the data roots bound for the current run."""
        bound = _BOUND.get()
        if bound is None:
            raise DataError(f"DataRef(alias={self.alias!r}) resolved outside a run: no data roots are bound")
        roots, _ = bound
        if self.alias not in roots:
            known = ", ".join(sorted(roots)) or "none"
            raise DataError(f"no data alias {self.alias!r} on this machine (configured: {known}); add [data] to config")
        p = Path(roots[self.alias]) / self.relpath if self.relpath else Path(roots[self.alias])
        if not p.exists():
            raise DataError(f"DataRef(alias={self.alias!r}, relpath={self.relpath!r}) → {p}: does not exist")
        return p

    def fingerprint(self) -> str:
        bound = _BOUND.get()
        return content_hash(self.path(), bound[1] if bound else None)


_BOUND: contextvars.ContextVar[tuple[Mapping[str, Path], StatCache | None] | None] = contextvars.ContextVar(
    "chairlift_data", default=None
)


@contextmanager
def bind(roots: Mapping[str, Path], cache: StatCache | None = None) -> Generator[None]:
    token = _BOUND.set((roots, cache))
    try:
        yield
    finally:
        _BOUND.reset(token)


def fingerprint_of(*refs: DataRef) -> Callable[[], str]:
    """A source stage's fingerprint: the content of what it reads. Aliases are names, so they are left out."""
    if not refs:
        raise ValueError("fingerprint_of needs at least one DataRef")

    def fp() -> str:
        hashes = [r.fingerprint() for r in refs]
        return hashes[0] if len(hashes) == 1 else "b3m:" + _b3("\n".join(hashes).encode())

    return fp


# ---- hashing -----------------------------------------------------------------------------------------------------


def _b3(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def _hash_file(path: Path) -> str:
    h = blake3.blake3(max_threads=blake3.blake3.AUTO)
    if path.stat().st_size:
        h.update_mmap(path)
    return h.hexdigest()


def content_hash(path: Path, cache: StatCache | None = None) -> str:
    """`b3f:<hex>` for a file, `b3d:<hex>` for a directory; independent of where the data lives."""
    path = Path(path)
    if path.is_file():
        return "b3f:" + _file(path, cache)
    if path.is_dir():
        lines: list[str] = []
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                f = Path(dirpath) / name
                if f.is_file():
                    lines.append(f"{f.relative_to(path).as_posix()}\0{_file(f, cache)}")
        lines.sort()
        return "b3d:" + _b3("\n".join(lines).encode())
    raise DataError(f"{path}: not a file or a directory")


def _file(path: Path, cache: StatCache | None) -> str:
    if cache is None:
        return _hash_file(path)
    st = path.stat()
    stat = (st.st_size, st.st_mtime_ns, st.st_ino)
    hit = cache.get(path, stat)
    if hit is not None:
        return hit
    digest = _hash_file(path)
    cache.put(path, stat, digest)
    return digest


# ---- the stat cache ----------------------------------------------------------------------------------------------


@dataclass
class StatCache:
    """(absolute path, size, mtime_ns, inode) → blake3; reads every host's file, appends to this host's."""

    directory: Path
    host: str = ""

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.host = self.host or socket.gethostname()
        self._entries: dict[tuple[str, int, int, int], str] = {}
        self.hashed = 0  # files hashed by this process because the cache missed (tests and diagnostics)
        if self.directory.exists():
            for f in sorted(self.directory.glob("fingerprints*.jsonl")):
                for line in f.read_text().splitlines():
                    try:
                        e = json.loads(line)
                        self._entries[(e["path"], e["size"], e["mtime_ns"], e["inode"])] = e["b3"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue  # a torn last line from a killed writer costs one re-hash, nothing else

    @property
    def path(self) -> Path:
        return self.directory / f"fingerprints.{self.host}.jsonl"

    def get(self, file: Path, stat: tuple[int, int, int]) -> str | None:
        return self._entries.get((str(file.resolve()), *stat))

    def put(self, file: Path, stat: tuple[int, int, int], digest: str) -> None:
        key = (str(file.resolve()), *stat)
        self._entries[key] = digest
        self.hashed += 1
        self.directory.mkdir(parents=True, exist_ok=True)
        rec = {"path": key[0], "size": stat[0], "mtime_ns": stat[1], "inode": stat[2], "b3": digest}
        append_line(self.path, json.dumps(rec, sort_keys=True))
