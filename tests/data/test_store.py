"""data/store.py: content-addressed objects, atomic writes, read-back fidelity."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from chairlift.data.store import ArtifactStore, MissingArtifact


def frame() -> pl.DataFrame:
    return pl.DataFrame({"entity": ["a", "b", "c"], "t0": [1, 2, 3], "y": [0.5, -1.25, 2.0]})


# ---- identity ------------------------------------------------------------------------------------------------------


def test_frame_round_trip_and_deterministic_hash(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    ref1, ref2 = store.put(frame()), store.put(frame())
    assert ref1 == ref2 and ref1.startswith("parquet:")
    assert store.get(ref1).equals(frame())


def test_different_contents_get_different_refs(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    assert store.put(frame()) != store.put(frame().with_columns(pl.col("y") * 2))


def test_column_order_and_dtype_are_part_of_identity(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    base = store.put(frame())
    assert store.put(frame().select("y", "t0", "entity")) != base
    assert store.put(frame().with_columns(pl.col("t0").cast(pl.Int32))) != base


def test_json_values_are_order_free_and_read_back_as_themselves(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    ref = store.put({"b": 1, "a": [1, 2]})
    assert ref == store.put({"a": (1, 2), "b": 1})
    assert store.get(ref) == {"a": [1, 2], "b": 1}


# ---- fidelity ------------------------------------------------------------------------------------------------------


def test_dtypes_nulls_and_dates_survive_the_round_trip(tmp_path: Path):
    df = pl.DataFrame(
        {
            "d": [dt.date(2024, 1, 2), None],
            "ts": [dt.datetime(2024, 1, 2, 14, 55, tzinfo=dt.UTC), None],
            "i8": pl.Series([1, None], dtype=pl.Int8),
            "s": ["x", None],
            "f": [float("nan"), 1.0],
        }
    )
    back = ArtifactStore(tmp_path).get(ArtifactStore(tmp_path).put(df))
    assert back.schema == df.schema
    assert back.equals(df, null_equal=True)


def test_empty_frame_round_trip(tmp_path: Path):
    df = pl.DataFrame(schema={"a": pl.Int64, "b": pl.String})
    assert ArtifactStore(tmp_path).get(ArtifactStore(tmp_path).put(df)).schema == df.schema


# ---- refusal -------------------------------------------------------------------------------------------------------


def test_non_json_outputs_are_rejected(tmp_path: Path):
    with pytest.raises(TypeError):
        ArtifactStore(tmp_path).put({"x": object()})
    with pytest.raises(TypeError):
        ArtifactStore(tmp_path).put({"x": float("nan")})


@pytest.mark.parametrize("bad", ["", "json:", "json:abc", "csv:" + "0" * 64, "parquet" + "0" * 64])
def test_malformed_references_are_rejected(tmp_path: Path, bad: str):
    with pytest.raises(ValueError):
        ArtifactStore(tmp_path).has(bad)


def test_missing_object_raises(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    ref = store.put({"x": 1})
    next((tmp_path / "objects").rglob("*.json")).unlink()
    assert not store.has(ref)
    with pytest.raises(MissingArtifact):
        store.get(ref)


# ---- filesystem behaviour ------------------------------------------------------------------------------------------


def test_construction_creates_nothing_on_disk(tmp_path: Path):
    ArtifactStore(tmp_path / "store")
    assert not (tmp_path / "store").exists()


def test_writes_are_atomic_and_leave_no_temporaries(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    for i in range(5):
        store.put({"i": i})
    assert not list(tmp_path.rglob(".tmp-*"))
    assert len(list((tmp_path / "objects").rglob("*.json"))) == 5


def test_a_repeated_put_never_rewrites_the_object(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    ref = store.put({"x": 1})
    path = next((tmp_path / "objects").rglob("*.json"))
    mtime = path.stat().st_mtime_ns
    assert store.put({"x": 1}) == ref and path.stat().st_mtime_ns == mtime


def test_a_non_json_output_names_the_offending_path(tmp_path: Path):
    import pytest

    with pytest.raises(TypeError, match=r"\$\.ls\.corr = nan"):
        ArtifactStore(tmp_path).put({"ls": {"sharpe": 1.0, "corr": float("nan")}})


def test_equal_frames_with_different_chunking_have_one_address(tmp_path: Path):
    a = pl.DataFrame({"x": list(range(1000)), "s": [str(i) for i in range(1000)]})
    b = pl.concat([a.head(300), a.slice(300, 400), a.tail(300)], rechunk=False)
    assert b.n_chunks() > 1 and a.equals(b)
    store = ArtifactStore(tmp_path)
    assert store.put(a) == store.put(b)
