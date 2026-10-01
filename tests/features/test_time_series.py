"""features/time_series.py: causal transforms; the forward return is a target, never a feature."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from chairlift.features.time_series import forward_return, lag, rolling_zscore


def test_rolling_zscore_uses_only_the_past():
    x = pl.DataFrame({"x": np.arange(20, dtype=float)})
    full = x.with_columns(z=rolling_zscore("x", 5, 3))["z"]
    cut = x.head(10).with_columns(z=rolling_zscore("x", 5, 3))["z"]
    assert full.head(10).equals(cut)  # appending later rows changes no earlier value
    assert full[:2].is_null().all() and full[4] == pytest.approx(2 / np.std([0, 1, 2, 3, 4], ddof=1))


def test_lag_and_forward_return_point_opposite_ways():
    f = pl.DataFrame({"g": ["a", "a", "a", "b", "b"], "p": [1.0, 2.0, 4.0, 10.0, 5.0]})
    out = f.with_columns(l=lag("p", over="g"), r=forward_return("p", 1, over="g"))
    assert out["l"].to_list() == [None, 1.0, 2.0, None, 10.0]
    assert out["r"].to_list()[:2] == pytest.approx([np.log(2), np.log(2)]) and out["r"][2] is None
    assert out["r"][3] == pytest.approx(np.log(0.5))
