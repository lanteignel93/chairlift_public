"""search/space.py: candidates drawn from a declared space, reproducibly and without duplicates."""

from __future__ import annotations

import pytest

from chairlift.search.space import SearchError, SearchSpace


def test_grid_is_the_cross_product_and_random_is_seeded():
    g = SearchSpace.parse({"sampler": "grid", "budget": 0, "params": {"a": [1, 2], "b": ["x", "y", "z"]}})
    assert len(g.candidates()) == 6
    r = {
        "sampler": "random",
        "budget": 8,
        "seed": 3,
        "params": {"a": {"low": 1e-4, "high": 1.0, "log": True}, "n": {"low": 2, "high": 6, "int": True}},
    }
    a, b = SearchSpace.parse(r).candidates(), SearchSpace.parse(r).candidates()
    assert a == b and len(a) == 8 and len({repr(sorted(c.items())) for c in a}) == 8
    assert all(1e-4 <= c["a"] <= 1.0 and isinstance(c["n"], int) and 2 <= c["n"] <= 6 for c in a)


def test_a_small_space_stops_at_its_distinct_points():
    s = SearchSpace.parse({"budget": 50, "params": {"a": [1, 2]}})
    assert sorted(c["a"] for c in s.candidates()) == [1, 2]


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"params": {"a": []}}, "empty"),
        ({"params": {"a": {"low": 1}}}, "range"),
        ({"params": {"a": 3}}, "list of values"),
        ({"sampler": "grid", "params": {"a": {"low": 0, "high": 1}}}, "grid needs lists"),
        ({"bogus": 1}, "unknown"),
    ],
)
def test_malformed_spaces_are_refused(raw: dict[str, object], match: str):
    with pytest.raises(SearchError, match=match):
        SearchSpace.parse(raw)
