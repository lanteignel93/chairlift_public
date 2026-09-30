"""Property tests: the identity and rebuild rules hold for any input, not only the hand-picked cases."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from chairlift.core.spec import canonical_json, spec, spec_hash
from chairlift.run.dag import Pipeline, Stage

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


# ---- the rebuild rule on random DAGs -------------------------------------------------------------------------------


@st.composite
def dags(draw: st.DrawFn) -> tuple[list[tuple[int, ...]], set[int]]:
    """n stages; stage i may depend on any earlier stage; a random subset gets a version bump."""
    n = draw(st.integers(2, 8))
    deps = [tuple(sorted(draw(st.sets(st.integers(0, i - 1), max_size=3)))) if i else () for i in range(n)]
    bumped = draw(st.sets(st.integers(0, n - 1), min_size=1, max_size=n))
    return deps, bumped


def _pipeline(root: Path, deps: list[tuple[int, ...]], versions: dict[int, int]) -> Pipeline:
    stages: list[Stage] = []
    for i, d in enumerate(deps):
        v = versions.get(i, 1)

        def fn(_i: int = i, _v: int = v, **inputs: Any) -> dict[str, Any]:
            # carries its inputs' contents, so a change anywhere upstream changes these bytes too (a stage whose output
            # ignored its inputs would stop the propagation: content addressing, tested in test_dag)
            return {"stage": _i, "version": _v, "inputs": inputs}

        stages.append(
            Stage(f"s{i}", fn, tuple(f"s{j}" for j in d), version=v, fingerprint=(lambda: "fixed") if not d else None)
        )
    return Pipeline(stages, root)


def _descendants(deps: list[tuple[int, ...]], seeds: set[int]) -> set[int]:
    out = set(seeds)
    for i, d in enumerate(deps):  # deps only point backwards, so one forward pass closes the set
        if out.intersection(d):
            out.add(i)
    return out


@settings(max_examples=60, deadline=None)
@given(dags())
def test_version_bump_reruns_exactly_the_descendants(case: tuple[list[tuple[int, ...]], set[int]]):
    deps, bumped = case
    # fresh per example: hypothesis replays examples when shrinking
    root = Path(tempfile.mkdtemp(prefix="chairlift-prop-"))
    _pipeline(root, deps, {}).run()
    report = _pipeline(root, deps, {i: 2 for i in bumped}).run()
    assert {int(s[1:]) for s in report.ran()} == _descendants(deps, bumped)
    assert _pipeline(root, deps, {i: 2 for i in bumped}).run().ran() == []  # and a third build reuses everything
