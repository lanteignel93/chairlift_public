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
        records: list[Record] = []
        for f in self.files():
            for line in f.read_text().splitlines():
                if line.strip():
                    records.append(Record(**json.loads(line)))
        for rec in sorted(records, key=lambda r: r.built_at):  # stable: file order breaks ties
            self._by_key[rec.key] = rec  # a later record for the same key supersedes an earlier one

    def files(self) -> list[Path]:
        """Every manifest file this manifest reads, in a stable order."""
        if not self.shared:
            return [self.base] if self.base.exists() else []
        d = self.base.parent
        return sorted(d.glob(f"{self.base.stem}*{self.base.suffix}")) if d.exists() else []

    def lookup(self, key: str) -> Record | None:
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
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
        self._by_key[key] = rec
        return rec

    def records(self) -> list[Record]:
        return list(self._by_key.values())
