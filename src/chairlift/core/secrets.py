"""Secrets: credentials and webhook URLs, kept outside the configuration object entirely.

They come from `CHAIRLIFT_SECRET_<NAME>` environment variables or `<user config dir>/chairlift/secrets.toml`, which
must not be readable by group or others (mode 0600). A `Secrets` object never serializes its values: its repr is
masked and it refuses to be converted to JSON, so it cannot reach a run record or a hash by accident.
"""

from __future__ import annotations

import os
import stat
import tomllib
from collections.abc import Iterator, Mapping
from pathlib import Path

import platformdirs

PREFIX = "CHAIRLIFT_SECRET_"


class SecretsError(PermissionError):
    pass


class Secrets(Mapping[str, str]):
    def __init__(self, values: Mapping[str, str], sources: Mapping[str, str]) -> None:
        self._values = dict(values)
        self.sources = dict(sources)  # name → "env" or the file path; safe to record

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"Secrets({', '.join(f'{k}=***' for k in sorted(self._values))})"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("Secrets cannot be pickled or serialized")


def load_secrets(*, env: Mapping[str, str] | None = None, path: Path | None = None) -> Secrets:
    env = os.environ if env is None else env
    path = platformdirs.user_config_path("chairlift") / "secrets.toml" if path is None else path
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    if path.is_file():
        mode = path.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise SecretsError(
                f"{path} is readable by group or others (mode {oct(mode & 0o777)}); run chmod 600 {path}"
            )
        for k, v in tomllib.loads(path.read_text()).items():
            if not isinstance(v, str):
                raise SecretsError(f"{path}: secret {k!r} must be a string")
            values[k.lower()] = v
            sources[k.lower()] = str(path)
    for name, v in env.items():
        if name.startswith(PREFIX) and len(name) > len(PREFIX):
            values[name[len(PREFIX) :].lower()] = v
            sources[name[len(PREFIX) :].lower()] = "env"
    return Secrets(values, sources)
