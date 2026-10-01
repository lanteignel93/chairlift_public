"""Experiment files: the daily surface. One TOML file names a study, sets its parameters, and optionally sweeps them.

    study  = "chairlift.verify.toy:pipeline"     # module:factory, or path/to/file.py:factory (relative to this file)
    name   = "window-sweep"                      # optional; default: the file's stem
    reason = "does a longer feature window help?"
    study_name = "toy"                           # optional; default: the module's STUDY_NAME

    [params]                                     # keyword parameters of the factory
    window = 5

    [run]
    targets = ["evaluate"]                       # optional; default: every stage

    [sweep]                                      # optional; the cross product, one run per cell
    window = [3, 5, 10]

Parameters are checked against the factory's signature before anything is built: an unknown name, a missing
required parameter, or a value of the wrong type is an error naming the parameter. Values are converted strictly
(msgspec), so an int given for a float parameter becomes a float, and a string never becomes a number.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, get_type_hints

import msgspec

TOP = {"study", "name", "reason", "study_name", "params", "run", "sweep"}
RUN = {"targets"}


class ExperimentError(ValueError):
    """An experiment file or a parameter set that cannot be run, named by its key."""


@dataclass(frozen=True)
class Experiment:
    study: str
    name: str
    path: Path
    text: str
    reason: str = ""
    study_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict[str, Any])
    targets: tuple[str, ...] | None = None
    sweep: dict[str, tuple[Any, ...]] = field(default_factory=dict[str, tuple[Any, ...]])

    def digest(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    def cells(self) -> list[dict[str, Any]]:
        """The parameter set of every sweep cell, in file order with the last key varying fastest; one cell if no
        sweep."""
        if not self.sweep:
            return [dict(self.params)]
        keys = list(self.sweep)
        return [
            self.params | dict(zip(keys, values, strict=True)) for values in itertools.product(*self.sweep.values())
        ]

    def record(self) -> dict[str, Any]:
        """What a run record keeps: enough to rebuild the run from the record alone."""
        return {"name": self.name, "file": str(self.path), "sha256": self.digest(), "text": self.text}


def load_experiment(path: Path | str) -> Experiment:
    path = Path(path).resolve()
    text = path.read_text()
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ExperimentError(f"{path}: {exc}") from exc
    unknown = set(raw) - TOP
    if unknown:
        raise ExperimentError(
            f"{path}: unknown key(s) {', '.join(sorted(unknown))} (allowed: {', '.join(sorted(TOP))})"
        )
    study = raw.get("study")
    if not isinstance(study, str) or ":" not in study:
        raise ExperimentError(f"{path}: `study` must be 'module:factory' or 'file.py:factory'")
    target, _, attr = study.rpartition(":")
    if target.endswith(".py") and not Path(target).is_absolute():
        study = f"{(path.parent / target).resolve()}:{attr}"
    params = _table(raw, "params", path)
    run = _table(raw, "run", path)
    if set(run) - RUN:
        raise ExperimentError(f"{path}: unknown key(s) in [run]: {', '.join(sorted(set(run) - RUN))}")
    targets = run.get("targets")
    if targets is not None and not (
        isinstance(targets, list) and all(isinstance(t, str) for t in cast("list[Any]", targets))
    ):
        raise ExperimentError(f"{path}: [run] targets must be a list of stage names")
    sweep: dict[str, tuple[Any, ...]] = {}
    for k, v in _table(raw, "sweep", path).items():
        if not isinstance(v, list) or not v:
            raise ExperimentError(f"{path}: [sweep] {k} must be a non-empty list of values")
        if k in params:
            raise ExperimentError(f"{path}: {k} is both in [params] and [sweep]; keep it in one")
        sweep[k] = tuple(cast("list[Any]", v))
    for key in ("name", "reason", "study_name"):
        if key in raw and not isinstance(raw[key], str):
            raise ExperimentError(f"{path}: `{key}` must be a string")
    return Experiment(
        study=study,
        name=raw.get("name", path.stem),
        path=path,
        text=text,
        reason=raw.get("reason", ""),
        study_name=raw.get("study_name"),
        params=params,
        targets=tuple(cast("list[str]", targets)) if targets is not None else None,
        sweep=sweep,
    )


def _table(raw: Mapping[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ExperimentError(f"{path}: [{key}] must be a table")
    return dict(cast("dict[str, Any]", value))


def validate_params(factory: Callable[..., Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """`params` converted to the factory's annotated types; every problem is named before anything runs.

    A pipeline factory's first positional parameter is the run root, supplied by chairlift; a Study factory takes
    keyword parameters only. Either way the keyword parameters are the study's knobs.
    """
    sig = inspect.signature(factory)
    params_ = list(sig.parameters.values())
    takes_root = bool(params_) and params_[0].kind in (params_[0].POSITIONAL_ONLY, params_[0].POSITIONAL_OR_KEYWORD)
    knobs = params_[1:] if takes_root else params_  # a Study factory takes keyword parameters only
    try:
        hints = get_type_hints(factory)
    except (NameError, TypeError):
        hints = {}
    names = {p.name for p in knobs if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    takes_any = any(p.kind is p.VAR_KEYWORD for p in knobs)
    unknown = set(params) - names
    if unknown and not takes_any:
        raise ExperimentError(f"unknown parameter(s) {', '.join(sorted(unknown))}; the factory takes {_fmt(knobs)}")
    missing = [p.name for p in knobs if p.default is p.empty and p.kind is p.KEYWORD_ONLY and p.name not in params]
    if missing:
        raise ExperimentError(f"missing required parameter(s) {', '.join(missing)}")
    out: dict[str, Any] = {}
    for name, value in params.items():
        hint = hints.get(name, Any)
        try:
            out[name] = msgspec.convert(value, hint, strict=True)
        except msgspec.ValidationError as exc:
            raise ExperimentError(f"parameter {name}={value!r}: {exc}") from exc
    return out


def _fmt(knobs: list[inspect.Parameter]) -> str:
    return ", ".join(p.name for p in knobs) or "no parameters"
