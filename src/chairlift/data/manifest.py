"""The manifest: an append-only log of (stage key → output) with the provenance of every build.

A stage's key is the hash of its name, its declared version, its spec and the references of its inputs. A lookup
succeeds only if a record for the key exists *and* the store still holds its output; anything else is a miss and the
stage runs again. Nothing is skipped because a file "looks done".
"""

from __future__ import annotations

import datetime as dt
import json
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


class Manifest:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._by_key: dict[str, Record] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = Record(**json.loads(line))
                    self._by_key[rec.key] = rec  # a later record for the same key supersedes an earlier one

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
            built_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        )
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
        self._by_key[key] = rec
        return rec

    def records(self) -> list[Record]:
        return list(self._by_key.values())
