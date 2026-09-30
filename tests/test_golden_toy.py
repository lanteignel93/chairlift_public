"""Golden reproduction: the toy study's outputs, compared on contents against a committed snapshot.

Parquet bytes and artifact refs may change with a polars upgrade; the values may not. If this test fails after a
dependency or code change, either the change was intended (update with `uv run pytest --snapshot-update` and say why
in the commit) or it silently changed what the pipeline computes.
"""

from pathlib import Path
from typing import Any

import polars as pl
from syrupy.assertion import SnapshotAssertion

from chairlift.verify.toy import pipeline


def _summary(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "shape": list(frame.shape),
        "columns": frame.columns,
        "head": frame.head(5).with_columns(pl.col(pl.Float64).round(10)).to_dicts(),
        "sums": {c: round(float(frame[c].sum()), 8) for c in frame.columns if frame[c].dtype == pl.Float64},
    }


def test_toy_outputs_match_the_golden_snapshot(tmp_path: Path, snapshot: SnapshotAssertion):
    p = pipeline(tmp_path / "toy")
    report = p.run()
    assert p.load("evaluate", report) == snapshot(name="evaluate")
    assert p.load("report", report) == snapshot(name="report")
    assert _summary(p.load("fit", report)) == snapshot(name="fit")
    assert p.load("folds", report) == snapshot(name="folds")
