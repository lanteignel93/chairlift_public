"""Property tests for core/spec.py: hashing invariants for any value, not only the hand-picked cases."""

from __future__ import annotations

from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from chairlift.core.spec import canonical_json, spec, spec_hash

scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(10**12), 10**12),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    st.text(max_size=12),
)
values = st.recursive(
    scalars,
    lambda inner: st.one_of(st.lists(inner, max_size=4), st.dictionaries(st.text(max_size=8), inner, max_size=4)),
    max_leaves=12,
)
params = st.dictionaries(st.text(min_size=1, max_size=8), values, max_size=6)


@spec(name="PropSpec")
class PropSpec:
    params: dict[str, Any]


def _has_mutable(v: Any) -> bool:
    if isinstance(v, list | dict | set):
        return True
    if isinstance(v, tuple | frozenset):
        return any(_has_mutable(x) for x in v)  # pyright: ignore[reportUnknownVariableType]
    if hasattr(v, "items"):
        return any(_has_mutable(x) for _, x in v.items())
    return False


@given(params)
def test_hash_ignores_mapping_order(d: dict[str, Any]):
    reordered = dict(reversed(list(d.items())))
    assert spec_hash(PropSpec(params=d)) == spec_hash(PropSpec(params=reordered))


@given(params)
def test_canonical_json_is_deterministic_and_frozen(d: dict[str, Any]):
    s = PropSpec(params=d)
    assert canonical_json(s) == canonical_json(PropSpec(params=d))
    assert not _has_mutable(s.params)


@given(params, st.text(min_size=1, max_size=8), st.integers(0, 10**6))
def test_changing_one_value_changes_the_hash(d: dict[str, Any], key: str, n: int):
    a = dict(d) | {key: n}
    b = dict(d) | {key: n + 1}
    assert spec_hash(PropSpec(params=a)) != spec_hash(PropSpec(params=b))
