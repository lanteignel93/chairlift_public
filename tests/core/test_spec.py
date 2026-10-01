import datetime as dt
import enum
from typing import Any

import pytest

from chairlift.core.spec import SpecError, canonical_json, spec, spec_hash


@spec(name="Window", version=1)
class Window:
    months: int
    expanding: bool = False


@spec(name="Model", version=1)
class Model:
    family: str
    params: dict
    window: Window
    seeds: tuple = (1,)


class Side(enum.Enum):
    LONG = "long"
    SHORT = "short"


def m(**kw: Any) -> Model:
    base: dict[str, Any] = dict(family="ridge", params={"alpha": 1e-3, "fit_intercept": True}, window=Window(months=48))
    return Model(**(base | kw))


def test_equal_specs_hash_equal_and_key_order_does_not_matter():
    a = m(params={"alpha": 1e-3, "fit_intercept": True})
    b = m(params={"fit_intercept": True, "alpha": 1e-3})
    assert spec_hash(a) == spec_hash(b)


def test_any_field_change_changes_the_hash():
    base = spec_hash(m())
    assert spec_hash(m(family="gbm")) != base
    assert spec_hash(m(params={"alpha": 1e-2, "fit_intercept": True})) != base
    assert spec_hash(m(window=Window(months=36))) != base
    assert spec_hash(m(seeds=(1, 2))) != base


def test_version_bump_changes_the_hash_without_a_field_change():
    @spec(name="Window", version=2)
    class WindowV2:
        months: int
        expanding: bool = False

    assert spec_hash(Window(months=48)) != spec_hash(WindowV2(months=48))


def test_identity_is_the_declared_name_not_the_class_location():
    @spec(name="Window", version=1)
    class Elsewhere:
        months: int
        expanding: bool = False

    assert spec_hash(Elsewhere(months=48)) == spec_hash(Window(months=48))


def test_containers_are_frozen_at_construction():
    s = m(params={"alpha": 1.0, "levels": [1, 3]})
    assert isinstance(s.params["levels"], tuple)
    with pytest.raises(TypeError):
        s.params["alpha"] = 2.0  # read-only mapping: the hash cannot drift from the object
    with pytest.raises(AttributeError):
        s.family = "gbm"  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[invalid-assignment]


def test_list_and_tuple_hash_the_same():
    assert spec_hash(m(seeds=[1, 2])) == spec_hash(m(seeds=(1, 2)))


def test_sets_are_order_free():
    assert canonical_json({"a": {3, 1, 2}}) == canonical_json({"a": frozenset([2, 3, 1])})


def test_floats_and_special_values():
    assert spec_hash(m(params={"alpha": 0.0})) == spec_hash(m(params={"alpha": -0.0}))
    with pytest.raises(SpecError):
        spec_hash(m(params={"alpha": float("nan")}))
    with pytest.raises(SpecError):
        spec_hash(m(params={"alpha": float("inf")}))


def test_int_and_float_are_different_parameters():
    assert spec_hash(m(params={"alpha": 1})) != spec_hash(m(params={"alpha": 1.0}))


def test_dates_enums_and_nested_specs_are_supported():
    s = m(params={"start": dt.date(2017, 1, 3), "side": Side.LONG, "embargo": dt.timedelta(days=21)})
    assert spec_hash(s) == spec_hash(
        m(params={"side": Side.LONG, "embargo": dt.timedelta(days=21), "start": dt.date(2017, 1, 3)})
    )


def test_unsupported_values_are_rejected():
    import dataclasses

    @dataclasses.dataclass
    class Plain:
        x: int

    with pytest.raises(SpecError):
        spec_hash(m(params={"obj": Plain(1)}))
    with pytest.raises(SpecError):
        spec_hash(m(params={"obj": object()}))
    with pytest.raises(SpecError):
        spec_hash({1: "non-string key"})
