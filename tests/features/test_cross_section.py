"""features/cross_section.py: within-date transforms never look across dates."""

from __future__ import annotations

import polars as pl
import pytest

from chairlift.features.cross_section import demean, percentile_rank, rank_of_rows, rank_panel, zscore

F = pl.DataFrame(
    {"t0": [1, 1, 1, 1, 2, 2, 2], "g": ["a", "a", "b", "b", "a", "b", "b"], "x": [3.0, 1.0, None, 2.0, 5.0, 5.0, 7.0]}
)


def test_percentile_rank_spans_zero_one_over_non_null_values_and_keeps_nulls():
    r = F.with_columns(r=percentile_rank("x"))["r"].to_list()
    assert r[:4] == [1.0, 0.0, None, 0.5]
    assert r[4:] == [0.25, 0.25, 1.0]  # ties share the average rank


def test_rank_of_rows_counts_every_row():
    r = F.filter(pl.col("t0") == 2).with_columns(r=rank_of_rows("x"))["r"].to_list()
    assert r == [0.25, 0.25, 1.0]


def test_zscore_and_demean_within_groups():
    z = F.filter(pl.col("t0") == 2).with_columns(z=zscore("x"))["z"]
    assert z.mean() == pytest.approx(0.0) and z.std() == pytest.approx(1.0)
    d = F.with_columns(d=demean("x", ["t0", "g"]))
    assert d.filter((pl.col("t0") == 1) & (pl.col("g") == "a"))["d"].to_list() == [1.0, -1.0]


def test_appending_a_later_date_changes_no_earlier_value():
    before = F.with_columns(r=percentile_rank("x"), z=zscore("x"))
    after = pl.concat([F, pl.DataFrame({"t0": [3], "g": ["a"], "x": [100.0]})]).with_columns(
        r=percentile_rank("x"), z=zscore("x")
    )
    assert after.head(F.height).equals(before)


def test_rank_panel_keeps_columns_and_ranks_features():
    out = rank_panel(F, ["x"], keep=["t0", "g"])
    assert out.columns == ["t0", "g", "x"] and out["x"].max() == 1.0
