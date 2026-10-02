"""Specs: frozen dataclasses whose canonical JSON is their identity.

A spec's hash is what the manifest keys a stage on and what the ledger records, so two specs that describe the same
thing must hash equal and any change must change the hash. Identity is the declared (name, version) plus the field
values; the Python module path is not part of it, so moving a class between modules does not invalidate caches.
Containers are frozen at construction (list → tuple, dict → read-only mapping, set → frozenset): a spec that could be
mutated after hashing would let the cache drift from the object.
"""

from __future__ import annotations

import copyreg
import dataclasses
import datetime as dt
import enum
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, TypeVar, cast, dataclass_transform

T = TypeVar("T")


class SpecError(TypeError):
    """A value that cannot be part of a spec's identity."""


def _freeze(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in cast("tuple[Any, ...]", value))
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(v) for v in cast("frozenset[Any]", value))
    if isinstance(value, Mapping):
        items = cast("Mapping[Any, Any]", value)
        return MappingProxyType({k: _freeze(v) for k, v in items.items()})
    return value


def _mappingproxy(d: dict[Any, Any]) -> MappingProxyType[Any, Any]:
    return MappingProxyType(d)


def _reduce_mappingproxy(m: MappingProxyType[Any, Any]) -> tuple[Any, tuple[Any, ...]]:
    return _mappingproxy, (dict(m),)


# a spec's mappings are frozen as MappingProxyType, which pickle refuses; specs cross process boundaries (parallel
# folds), so the proxy is pickled as the dict it wraps and rebuilt as a proxy
copyreg.pickle(MappingProxyType, _reduce_mappingproxy)


@dataclass_transform(kw_only_default=True, frozen_default=True)
def spec(*, name: str | None = None, version: int = 1) -> Callable[[type[T]], type[T]]:
    """Declare a frozen, keyword-only spec dataclass with an explicit identity.

    Bump `version` whenever the meaning of the spec changes without its fields changing (the code behind it
    computes something different): the hash changes, and every cached result built from the old meaning is rebuilt.
    """

    def wrap(cls: type[T]) -> type[T]:
        original_post_init = getattr(cls, "__post_init__", None)

        def __post_init__(self: Any) -> None:
            for f in dataclasses.fields(self):
                object.__setattr__(self, f.name, _freeze(getattr(self, f.name)))
            if original_post_init is not None:
                original_post_init(self)

        setattr(cls, "__post_init__", __post_init__)  # noqa: B010  # wraps any user __post_init__
        dc = dataclasses.dataclass(frozen=True, kw_only=True)(cls)
        dc.__spec_name__ = name or cls.__qualname__  # type: ignore[attr-defined]
        dc.__spec_version__ = version  # type: ignore[attr-defined]
        return dc

    return wrap


def added(default: Any) -> Any:
    """A field added to an existing spec without changing its meaning: while it holds `default` it is left out of the
    canonical JSON, so every spec written before the field existed keeps its hash (and its cached results)."""
    if isinstance(default, list | dict | set):
        raise SpecError("added() takes an immutable default (tuple, frozenset, a scalar or a spec)")
    return dataclasses.field(default=default, metadata={"added": True})


def is_spec(obj: Any) -> bool:
    return dataclasses.is_dataclass(obj) and hasattr(type(obj), "__spec_version__")


def canonical(obj: Any) -> Any:
    """Convert a value to a JSON structure that is identical for equal values and different otherwise."""
    if obj is None or isinstance(obj, bool | str):
        return obj
    if isinstance(obj, enum.Enum):  # before int: IntEnum is an int
        return {"__enum__": type(obj).__qualname__, "value": canonical(obj.value)}
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise SpecError(f"non-finite float {obj!r} cannot be part of a spec")
        return 0.0 if obj == 0.0 else obj  # -0.0 and 0.0 are the same parameter
    if is_spec(obj):
        cls = cast(Any, type(obj))
        body = {
            f.name: canonical(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if not (f.metadata.get("added") and getattr(obj, f.name) == f.default)
        }
        return {"__spec__": str(cls.__spec_name__), "__version__": int(cls.__spec_version__), **body}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        raise SpecError(f"{type(obj).__qualname__} is a dataclass but not a @spec; declare it with @spec")
    if isinstance(obj, dt.datetime):
        return {"__datetime__": obj.isoformat()}
    if isinstance(obj, dt.date):
        return {"__date__": obj.isoformat()}
    if isinstance(obj, dt.timedelta):
        return {"__timedelta__": obj.total_seconds()}
    if isinstance(obj, tuple | list):
        return [canonical(v) for v in cast("tuple[Any, ...]", obj)]
    if isinstance(obj, set | frozenset):
        items = [canonical(v) for v in cast("frozenset[Any]", obj)]
        return {"__set__": sorted(items, key=lambda v: json.dumps(v, sort_keys=True))}
    if isinstance(obj, Mapping):
        mapping = cast("Mapping[Any, Any]", obj)
        keys: list[str] = []
        for k in mapping:
            if not isinstance(k, str):
                raise SpecError("spec mappings need string keys")
            keys.append(k)
        return {"__map__": {k: canonical(mapping[k]) for k in sorted(keys)}}
    raise SpecError(f"{type(obj).__qualname__} cannot be part of a spec's identity")


def canonical_json(obj: Any) -> str:
    return json.dumps(canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def spec_hash(obj: Any) -> str:
    """sha256 of the canonical JSON: the value's identity in the manifest and the ledger."""
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()
