"""Site configuration: where things live and how runs are executed on this machine.

Site settings never change a result, so they are recorded with every run and never hashed (the study and the
experiment carry everything that is hashed). Layers, lowest to highest precedence:

    default   built-in values; paths under the XDG data and cache directories, so zero configuration works
    system    <site config dir>/chairlift/config.toml        (/etc/xdg on Linux)
    user      <user config dir>/chairlift/config.toml        (~/.config on Linux)
    project   the nearest chairlift.toml in the working directory or a parent
    profile   [profile.<name>] from the merged files, chosen by the caller, else CHAIRLIFT_PROFILE, else the host map
    env       CHAIRLIFT_<SECTION>__<KEY>=value               (values parsed as TOML when they can be)
    cli       explicit flags

Every leaf records the layer that set it (`Resolved.explain()`). Paths expand `~` and `${VAR}` after merging; a
relative path is relative to the file that set it; an undefined variable is an error, never an empty string.
"""

from __future__ import annotations

import dataclasses
import os
import re
import socket
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import msgspec
import platformdirs

APP = "chairlift"
ENV_PREFIX = "CHAIRLIFT_"
RESERVED_ENV = {"CHAIRLIFT_PROFILE", "CHAIRLIFT_CONFIG"}


class ConfigError(ValueError):
    """A configuration problem, named by its dotted key and the layer that caused it."""


@dataclass(frozen=True, kw_only=True)
class Paths:
    """One home directory; everything else defaults to a fixed place under it and can be moved one by one.

    <home>/store/                  content store + manifest, shared by every study: studies reuse each other
    <home>/studies/<name>/         one directory per study: run records, events, reports
    <home>/reports/                the cross-study index
    <home>/cache/
    """

    home: Path
    store: Path | None = None
    studies: Path | None = None
    reports: Path | None = None
    cache: Path | None = None

    def store_dir(self) -> Path:
        return self.store or self.home / "store"

    def studies_dir(self) -> Path:
        return self.studies or self.home / "studies"

    def study_dir(self, name: str) -> Path:
        return self.studies_dir() / name

    def reports_dir(self) -> Path:
        return self.reports or self.home / "reports"

    def cache_dir(self) -> Path:
        return self.cache or self.home / "cache"


@dataclass(frozen=True, kw_only=True)
class Compute:
    workers: int = 1
    polars_threads: int | None = None
    memory_default: str | None = None


@dataclass(frozen=True, kw_only=True)
class Systemd:
    watchdog_default: str = "1h"
    on_failure: bool = True


@dataclass(frozen=True, kw_only=True)
class Alerts:
    sinks: tuple[str, ...] = ("page", "jsonl")


@dataclass(frozen=True, kw_only=True)
class Config:
    paths: Paths
    data: Mapping[str, Path] = field(default_factory=dict[str, Path])  # alias → directory on this machine
    compute: Compute = field(default_factory=Compute)
    systemd: Systemd = field(default_factory=Systemd)
    alerts: Alerts = field(default_factory=Alerts)


@dataclass(frozen=True)
class Resolved:
    config: Config
    provenance: dict[str, str]  # dotted key → layer label
    profile: str | None
    files: tuple[str, ...]  # config files read, in precedence order

    def explain(self) -> list[tuple[str, Any, str]]:
        flat = _flatten(_to_plain(self.config))
        return [(k, flat[k], self.provenance.get(k, "default")) for k in sorted(flat)]

    def record(self) -> dict[str, Any]:
        """What a run record stores: the resolved values, where each came from, the profile and the files."""
        return {
            "values": _to_plain(self.config),
            "provenance": dict(sorted(self.provenance.items())),
            "profile": self.profile,
            "files": list(self.files),
        }


# ---- layers ------------------------------------------------------------------------------------------------------


def _defaults() -> dict[str, Any]:
    return {
        "paths": {"home": str(platformdirs.user_data_path(APP))},
        "data": {},
        "compute": {},
        "systemd": {},
        "alerts": {},
    }


def _find_project_file(start: Path) -> Path | None:
    for d in (start, *start.parents):
        f = d / "chairlift.toml"
        if f.is_file():
            return f
    return None


def _read_toml(path: Path, label: str) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{label} ({path}): not valid TOML: {exc}") from exc


def _parse_env_value(raw: str) -> Any:
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def _env_layer(env: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, raw in env.items():
        if not name.startswith(ENV_PREFIX) or name in RESERVED_ENV or name.startswith("CHAIRLIFT_SECRET_"):
            continue
        parts = [p.lower() for p in name[len(ENV_PREFIX) :].split("__")]
        if len(parts) < 2 or not all(parts):
            raise ConfigError(f"env {name}: expected CHAIRLIFT_<SECTION>__<KEY>")
        _set_dotted(out, parts, _parse_env_value(raw))
    return out


def _set_dotted(target: dict[str, Any], parts: list[str], value: Any) -> None:
    node = target
    for p in parts[:-1]:
        nxt = node.setdefault(p, {})
        if not isinstance(nxt, dict):
            raise ConfigError(f"{'.'.join(parts)}: {p!r} is a value, not a table")
        node = cast(dict[str, Any], nxt)
    node[parts[-1]] = value


def _expand_dotted(table: Mapping[str, Any]) -> dict[str, Any]:
    """`paths.runs = "x"` inside a profile table means {"paths": {"runs": "x"}}."""
    out: dict[str, Any] = {}
    for k, v in table.items():
        value = _expand_dotted(cast(Mapping[str, Any], v)) if isinstance(v, Mapping) else v
        _set_dotted_merge(out, k.split("."), value)
    return out


def _set_dotted_merge(target: dict[str, Any], parts: list[str], value: Any) -> None:
    node = target
    for p in parts[:-1]:
        node = cast(dict[str, Any], node.setdefault(p, {}))
    if isinstance(value, dict) and isinstance(node.get(parts[-1]), dict):
        node[parts[-1]] = _merge(cast(dict[str, Any], node[parts[-1]]), cast(dict[str, Any], value), {}, "")[0]
    else:
        node[parts[-1]] = value


def _merge(
    base: dict[str, Any], top: Mapping[str, Any], prov: dict[str, str], label: str, prefix: str = ""
) -> tuple[dict[str, Any], dict[str, str]]:
    out = dict(base)
    for k, v in top.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k], prov = _merge(cast(dict[str, Any], out[k]), cast(Mapping[str, Any], v), prov, label, key + ".")
        elif isinstance(v, Mapping):
            out[k], prov = _merge({}, cast(Mapping[str, Any], v), prov, label, key + ".")
        else:
            out[k] = v
            if label:
                prov[key] = label
    return out, prov


# ---- schema checks, paths ----------------------------------------------------------------------------------------

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _expand_path(raw: str, key: str, layer: str, env: Mapping[str, str], base: Path) -> str:
    def repl(m: re.Match[str]) -> str:
        name = m.group(1) or m.group(2)
        if name not in env:
            raise ConfigError(f"{key}: ${{{name}}} is not defined (set by {layer})")
        return env[name]

    p = Path(_VAR.sub(repl, raw)).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def _check_keys(merged: Mapping[str, Any], schema: Any, prov: Mapping[str, str], prefix: str = "") -> None:
    fields = {f.name: f for f in dataclasses.fields(schema)}
    for k, v in merged.items():
        key = f"{prefix}{k}"
        if k not in fields:
            layer = next((prov[p] for p in prov if p == key or p.startswith(key + ".")), "unknown layer")
            raise ConfigError(f"{key}: unknown setting (set by {layer}); known: {', '.join(sorted(fields))}")
        ftype = fields[k].type
        sub = {"paths": Paths, "compute": Compute, "systemd": Systemd, "alerts": Alerts}.get(k) if not prefix else None
        if sub is not None and ftype in (sub.__name__, sub):
            if not isinstance(v, Mapping):
                raise ConfigError(f"{key}: expected a table")
            _check_keys(cast(Mapping[str, Any], v), sub, prov, key + ".")


def _dec_hook(tp: type, obj: Any) -> Any:
    if tp is Path:
        return Path(obj)
    raise NotImplementedError


# ---- entry point -------------------------------------------------------------------------------------------------


def load_config(
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    profile: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    system_file: Path | None = None,
    user_file: Path | None = None,
    hostname: str | None = None,
) -> Resolved:
    """Resolve the site configuration for this process. All arguments exist for tests; defaults read the machine."""
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else cwd
    system_file = platformdirs.site_config_path(APP) / "config.toml" if system_file is None else system_file
    user_file = platformdirs.user_config_path(APP) / "config.toml" if user_file is None else user_file
    explicit = env.get("CHAIRLIFT_CONFIG")

    merged, prov = _merge({}, _defaults(), {}, "")
    bases: dict[str, Path] = {}
    files: list[str] = []
    file_layers: list[tuple[str, Path]] = [("system", system_file), ("user", user_file)]
    project = _find_project_file(cwd)
    if project is not None:
        file_layers.append(("project", project))
    if explicit:
        file_layers.append(("CHAIRLIFT_CONFIG", Path(explicit)))
    profiles: dict[str, Any] = {}
    for label, path in file_layers:
        if not path.is_file():
            if label == "CHAIRLIFT_CONFIG":
                raise ConfigError(f"CHAIRLIFT_CONFIG points at {path}, which does not exist")
            continue
        table = _read_toml(path, label)
        prof = table.pop("profile", None)
        if prof is not None:
            if not isinstance(prof, dict):
                raise ConfigError(f"profile ({label}): expected a table of profiles")
            profiles, _ = _merge(profiles, cast(dict[str, Any], prof), {}, "")
        layer = f"{label} {path}"
        before = dict(prov)
        merged, prov = _merge(merged, table, prov, layer)
        for k in prov:
            if prov[k] == layer and before.get(k) != layer:
                bases[k] = path.parent
        files.append(str(path))

    hosts = cast(dict[str, Any], profiles.pop("hosts", {}))
    host = hostname if hostname is not None else socket.gethostname()
    chosen = profile or env.get("CHAIRLIFT_PROFILE") or hosts.get(host)
    if chosen is not None:
        if chosen not in profiles:
            raise ConfigError(f"profile {chosen!r} is not defined; defined: {', '.join(sorted(profiles)) or 'none'}")
        merged, prov = _merge(
            merged, _expand_dotted(cast(Mapping[str, Any], profiles[chosen])), prov, f"profile {chosen}"
        )
    merged, prov = _merge(merged, _env_layer(env), prov, "env")
    if overrides:
        cli: dict[str, Any] = {}
        for k, v in overrides.items():
            _set_dotted(cli, k.split("."), v)
        merged, prov = _merge(merged, cli, prov, "cli")

    _check_keys(merged, Config, prov)
    for section, keys in (("paths", merged["paths"]), ("data", merged["data"])):
        for k, v in cast(dict[str, Any], keys).items():
            dotted = f"{section}.{k}"
            if v is None:
                continue
            if not isinstance(v, str):
                raise ConfigError(f"{dotted}: expected a path string, got {type(v).__name__} ({prov.get(dotted)})")
            keys[k] = _expand_path(v, dotted, prov.get(dotted, "default"), env, bases.get(dotted, cwd))
    try:
        config = msgspec.convert(merged, Config, strict=False, dec_hook=_dec_hook)
    except msgspec.ValidationError as exc:
        raise ConfigError(f"invalid configuration: {exc}") from exc
    return Resolved(config, prov, chosen, tuple(files))


# ---- helpers -----------------------------------------------------------------------------------------------------


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v) for k, v in cast(Mapping[Any, Any], obj).items()}
    if isinstance(obj, tuple | list):
        return [_to_plain(v) for v in cast(tuple[Any, ...], obj)]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Mapping) and v:
            out |= _flatten(cast(Mapping[str, Any], v), f"{prefix}{k}.")
        else:
            out[f"{prefix}{k}"] = v
    return out


STARTER = """\
# chairlift site configuration — where things live on this machine. Nothing here changes a result.
# Layers (low → high): defaults, /etc/xdg/chairlift/config.toml, ~/.config/chairlift/config.toml, ./chairlift.toml,
# [profile.<name>], CHAIRLIFT_<SECTION>__<KEY> env vars, CLI flags. `chairlift config show --explain` shows the winner.

[paths]
# home = "~/chairlift"               # everything lives under here unless moved below
# store   = "/fast/chairlift-store"  # content store + manifest shared by every study (default <home>/store)
# studies = "..."                    # one directory per study  (default <home>/studies)
# reports = "..."                    # cross-study index        (default <home>/reports)
# cache   = "..."                    #                          (default <home>/cache)

[data]
# vendor = "/mnt/data/vendor"        # aliases: studies read DataRef("vendor", ...), never a path

[compute]
# workers = 8
# polars_threads = 16
# memory_default = "48G"

[systemd]
# watchdog_default = "1h"
# on_failure = true

[alerts]
# sinks = ["page", "jsonl"]          # "email" / "webhook" need a secret of the same name

# [profile.laptop]
# paths.home = "~/chairlift"
#
# [profile.hosts]
# "my-hostname" = "laptop"
"""
