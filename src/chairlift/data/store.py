"""Content-addressed artifact store: every stage output is saved once under the hash of its bytes.

Frames are written as parquet with fixed settings and hashed on those bytes. A different polars version may write
different bytes for the same data; that shows up as a cache miss downstream, which is the safe direction. Values
that must be compared across versions are compared on their contents (golden tests), never on their hashes.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

_KINDS = {"parquet", "json"}


class MissingArtifact(FileNotFoundError):
    """The manifest points at an object the store no longer holds."""


def _encode(obj: Any) -> tuple[bytes, str]:
    if isinstance(obj, pl.DataFrame):
        buf = io.BytesIO()
        obj.write_parquet(buf, compression="zstd", compression_level=3, statistics=False)
        return buf.getvalue(), "parquet"
    try:  # plain, sorted JSON: the value must read back as itself (tuples come back as lists)
        text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"stage output of type {type(obj).__qualname__} is neither a DataFrame nor plain JSON") from exc
    return text.encode(), "json"


class ArtifactStore:
    """objects/<first two hex>/<sha256>.<kind>, written atomically, never overwritten."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)  # directories are created on the first write, never by construction

    def _path(self, digest: str, kind: str) -> Path:
        return self.root / "objects" / digest[:2] / f"{digest}.{kind}"

    def put(self, obj: Any) -> str:
        data, kind = _encode(obj)
        digest = hashlib.sha256(data).hexdigest()
        path = self._path(digest, kind)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        return f"{kind}:{digest}"

    def has(self, ref: str) -> bool:
        kind, digest = self._split(ref)
        return self._path(digest, kind).exists()

    def get(self, ref: str) -> Any:
        kind, digest = self._split(ref)
        path = self._path(digest, kind)
        if not path.exists():
            raise MissingArtifact(f"{ref} is referenced but not in the store at {self.root}")
        if kind == "parquet":
            return pl.read_parquet(path)
        return json.loads(path.read_text())

    @staticmethod
    def _split(ref: str) -> tuple[str, str]:
        kind, _, digest = ref.partition(":")
        if kind not in _KINDS or len(digest) != 64:
            raise ValueError(f"not an artifact reference: {ref!r}")
        return kind, digest
