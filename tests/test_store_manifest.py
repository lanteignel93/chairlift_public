import polars as pl
import pytest

from chairlift.data.manifest import Manifest
from chairlift.data.store import ArtifactStore, MissingArtifact


def frame():
    return pl.DataFrame({"entity": ["a", "b", "c"], "t0": [1, 2, 3], "y": [0.5, -1.25, 2.0]})


def test_frame_round_trip_and_deterministic_hash(tmp_path):
    store = ArtifactStore(tmp_path)
    ref1 = store.put(frame())
    ref2 = store.put(frame())
    assert ref1 == ref2 and ref1.startswith("parquet:")
    assert store.get(ref1).equals(frame())


def test_different_contents_get_different_refs(tmp_path):
    store = ArtifactStore(tmp_path)
    assert store.put(frame()) != store.put(frame().with_columns(pl.col("y") * 2))


def test_json_values_are_order_free_and_read_back_as_themselves(tmp_path):
    store = ArtifactStore(tmp_path)
    ref = store.put({"b": 1, "a": [1, 2]})
    assert ref == store.put({"a": (1, 2), "b": 1})
    assert store.get(ref) == {"a": [1, 2], "b": 1}


def test_non_json_outputs_are_rejected(tmp_path):
    with pytest.raises(TypeError):
        ArtifactStore(tmp_path).put({"x": object()})
    with pytest.raises(TypeError):
        ArtifactStore(tmp_path).put({"x": float("nan")})


def test_missing_object_raises(tmp_path):
    store = ArtifactStore(tmp_path)
    ref = store.put({"x": 1})
    next((tmp_path / "objects").rglob("*.json")).unlink()
    assert not store.has(ref)
    with pytest.raises(MissingArtifact):
        store.get(ref)


def test_manifest_persists_and_later_records_supersede(tmp_path):
    path = tmp_path / "manifest.jsonl"
    m = Manifest(path)
    common = dict(stage="s", version=1, spec_hash="h", inputs={"b": "x", "a": "y"}, code="c")
    m.append(key="k1", output="json:" + "0" * 64, reason="first", **common)
    m.append(key="k1", output="json:" + "1" * 64, reason="rebuilt", **common)
    again = Manifest(path)
    rec = again.lookup("k1")
    assert rec is not None and rec.reason == "rebuilt" and list(rec.inputs) == ["a", "b"]
    assert again.lookup("nope") is None
    assert len(path.read_text().splitlines()) == 2  # append-only: both builds stay in the log
