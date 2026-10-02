"""The manifest: an append-only log of (stage key → output) with the provenance of every build.

A stage's key is the hash of its name, its declared version, its spec and the references of its inputs. A lookup
succeeds only if a record for the key exists *and* the store still holds its output; anything else is a miss and the
stage runs again. Nothing is skipped because a file "looks done".

A shared store (several studies, possibly several machines over NFS) gives each host its own file,
`manifest.<host>.jsonl`, so every file has exactly one writer: appends from two machines never interleave. Lookups
read every host's file, so a stage built on one machine is reused on another. A self-contained root keeps one file.
"""

from __future__ import annotations

import datetime as dt
import json
import socket
from dataclasses import asdict, dataclass
from pathlib import Path

from chairlift.data.locking import append_line


@dataclass(frozen=True)
class Record:
    stage: str
    key: str
    version: int
    spec_hash: str
    inputs: dict[str, str]
    output: str
    code: str
    reason: str
    built_at: str
    host: str = ""


class Manifest:
    def __init__(self, path: Path | str, *, shared: bool = False, host: str | None = None) -> None:
        """`path` is `<dir>/manifest.jsonl`. With `shared`, this process writes `<dir>/manifest.<host>.jsonl` and
        reads every `manifest*.jsonl` in the directory."""
        self.base = Path(path)
        self.host = host or socket.gethostname()
        self.shared = shared
        self.path = self.base.with_name(f"{self.base.stem}.{self.host}{self.base.suffix}") if shared else self.base
        self._by_key: dict[str, Record] = {}
        self._offsets: dict[Path, int] = {}
        self._refresh()

    def _refresh(self) -> None:
        """Fold in every complete line appended since the last read, by this process or any other.

        Another pipeline over the same store (a sweep's next cell, a run on another host) may have built a key since
        this one was opened; a miss re-reads before it answers, so nothing is rebuilt that already exists.
        """
        new: list[Record] = []
        for f in self.files():
            with f.open("rb") as fh:
                fh.seek(self._offsets.get(f, 0))
                chunk = fh.read()
            end = chunk.rfind(b"\n") + 1  # a line still being written is left for the next read
            self._offsets[f] = self._offsets.get(f, 0) + end
            for line in chunk[:end].decode().splitlines():
                if line.strip():
                    new.append(Record(**json.loads(line)))
        for rec in sorted(new, key=lambda r: r.built_at):  # stable: file order breaks ties
            old = self._by_key.get(rec.key)
            if old is None or rec.built_at >= old.built_at:  # a later record for a key supersedes an earlier one
                self._by_key[rec.key] = rec

    def files(self) -> list[Path]:
        """Every manifest file this manifest reads, in a stable order."""
        if not self.shared:
            return [self.base] if self.base.exists() else []
        d = self.base.parent
        return sorted(d.glob(f"{self.base.stem}*{self.base.suffix}")) if d.exists() else []

    def lookup(self, key: str) -> Record | None:
        if key not in self._by_key:
            self._refresh()
        return self._by_key.get(key)

    def append(
        self,
        *,
        stage: str,
        key: str,
        version: int,
        spec_hash: str,
        inputs: dict[str, str],
        output: str,
        code: str,
        reason: str,
    ) -> Record:
        rec = Record(
            stage=stage,
            key=key,
            version=version,
            spec_hash=spec_hash,
            inputs=dict(sorted(inputs.items())),
            output=output,
            code=code,
            reason=reason,
            built_at=dt.datetime.now(dt.UTC).isoformat(timespec="microseconds"),
            host=self.host,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        append_line(self.path, json.dumps(asdict(rec), sort_keys=True))
        self._by_key[key] = rec
        return rec

    def records(self) -> list[Record]:
        self._refresh()
        return list(self._by_key.values())
