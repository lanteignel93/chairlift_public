"""data/refs.py: data named by alias, identified by content, cached by stat."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import polars as pl
import pytest

from chairlift.core.spec import canonical, spec_hash
from chairlift.data.refs import DataError, DataRef, StatCache, bind, content_hash, fingerprint_of
from chairlift.run.dag import Pipeline, Stage


def _tree(root: Path) -> Path:
    (root / "sub").mkdir(parents=True)
    (root / "a.csv").write_text("x\n1\n2\n")
    (root / "sub" / "b.bin").write_bytes(bytes(range(256)) * 10)
    (root / "empty.txt").write_bytes(b"")
    return root


# ---- content hashing ---------------------------------------------------------------------------------------------


def test_same_bytes_at_two_paths_give_one_fingerprint(tmp_path: Path):
    a = _tree(tmp_path / "machine1" / "vendor")
    b = tmp_path / "elsewhere" / "copy"
    shutil.copytree(a, b)
    assert content_hash(a) == content_hash(b) and content_hash(a).startswith("b3d:")
    assert content_hash(a / "a.csv") == content_hash(b / "a.csv") and content_hash(a / "a.csv").startswith("b3f:")


def test_one_changed_byte_changes_the_fingerprint(tmp_path: Path):
    d = _tree(tmp_path / "d")
    before, before_file = content_hash(d), content_hash(d / "sub" / "b.bin")
    blob = bytearray((d / "sub" / "b.bin").read_bytes())
    blob[1000] ^= 1
    (d / "sub" / "b.bin").write_bytes(bytes(blob))
    assert content_hash(d) != before and content_hash(d / "sub" / "b.bin") != before_file


def test_a_rename_or_a_new_file_changes_a_directory_but_dot_files_do_not(tmp_path: Path):
    d = _tree(tmp_path / "d")
    base = content_hash(d)
    (d / ".DS_Store").write_text("finder noise")
    (d / ".git").mkdir()
    (d / ".git" / "HEAD").write_text("ref")
    assert content_hash(d) == base
    (d / "a.csv").rename(d / "a2.csv")
    renamed = content_hash(d)
    assert renamed != base
    (d / "new.txt").write_text("")
    assert content_hash(d) != renamed


def test_file_and_directory_hashes_never_collide(tmp_path: Path):
    (tmp_path / "f").write_text("")
    (tmp_path / "d").mkdir()
    assert content_hash(tmp_path / "f")[:4] == "b3f:" and content_hash(tmp_path / "d")[:4] == "b3d:"


def test_missing_path_is_an_error(tmp_path: Path):
    with pytest.raises(DataError, match="not a file or a directory"):
        content_hash(tmp_path / "nope")


# ---- the stat cache ----------------------------------------------------------------------------------------------


def test_cache_hit_skips_hashing_and_mtime_change_rehashes(tmp_path: Path):
    d = _tree(tmp_path / "d")
    cache = StatCache(tmp_path / "cache", host="h")
    first = content_hash(d, cache)
    assert cache.hashed == 3
    assert content_hash(d, cache) == first and cache.hashed == 3  # all three files answered from the cache
    st = (d / "a.csv").stat()
    os.utime(d / "a.csv", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert content_hash(d, cache) == first and cache.hashed == 4  # same bytes, new mtime: re-hashed once


def test_cache_persists_per_host_and_is_read_across_hosts(tmp_path: Path):
    d = _tree(tmp_path / "d")
    content_hash(d, StatCache(tmp_path / "cache", host="boxA"))
    assert [p.name for p in (tmp_path / "cache").iterdir()] == ["fingerprints.boxA.jsonl"]
    other = StatCache(tmp_path / "cache", host="boxB")
    content_hash(d, other)
    assert other.hashed == 0 and not other.path.exists()


def test_the_cache_is_never_the_truth_a_stale_entry_cannot_survive_a_content_change(tmp_path: Path):
    d = _tree(tmp_path / "d")
    cache = StatCache(tmp_path / "cache", host="h")
    before = content_hash(d, cache)
    (d / "a.csv").write_text("x\n1\n3\n")  # same size; mtime moves
    assert content_hash(d, cache) != before


def test_a_torn_cache_line_costs_a_rehash_not_an_error(tmp_path: Path):
    d = _tree(tmp_path / "d")
    cache = StatCache(tmp_path / "cache", host="h")
    content_hash(d, cache)
    with cache.path.open("a") as fh:
        fh.write('{"path": "/x", "si')
    again = StatCache(tmp_path / "cache", host="h")
    content_hash(d, again)
    assert again.hashed == 0


def test_cache_creates_nothing_until_it_hashes(tmp_path: Path):
    StatCache(tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


# ---- DataRef resolution ------------------------------------------------------------------------------------------


def test_dataref_is_a_spec_whose_identity_has_no_path(tmp_path: Path):
    ref = DataRef(alias="vendor", relpath="option_close")
    assert canonical(ref) == {"__spec__": "data_ref", "__version__": 1, "alias": "vendor", "relpath": "option_close"}
    with bind({"vendor": tmp_path}):
        assert spec_hash(ref) == spec_hash(DataRef(alias="vendor", relpath="option_close"))


def test_resolution_needs_bound_roots_a_known_alias_and_an_existing_path(tmp_path: Path):
    ref = DataRef(alias="vendor", relpath="x.csv")
    with pytest.raises(DataError, match="outside a run"):
        ref.path()
    with bind({"other": tmp_path}), pytest.raises(DataError, match=r"no data alias 'vendor'.*configured: other"):
        ref.path()
    with bind({"vendor": tmp_path}), pytest.raises(DataError, match="does not exist"):
        ref.path()
    (tmp_path / "x.csv").write_text("x\n")
    with bind({"vendor": tmp_path}):
        assert ref.path() == tmp_path / "x.csv"
        assert DataRef(alias="vendor").path() == tmp_path


def test_fingerprint_of_combines_refs_in_order_and_leaves_aliases_out(tmp_path: Path):
    (tmp_path / "a").write_text("a")
    (tmp_path / "b").write_text("b")
    with bind({"v": tmp_path, "w": tmp_path}):
        one = fingerprint_of(DataRef(alias="v", relpath="a"))()
        assert one == content_hash(tmp_path / "a")
        assert (
            fingerprint_of(DataRef(alias="v", relpath="a"), DataRef(alias="v", relpath="b"))()
            == fingerprint_of(DataRef(alias="w", relpath="a"), DataRef(alias="w", relpath="b"))()
        )
        assert (
            fingerprint_of(DataRef(alias="v", relpath="a"), DataRef(alias="v", relpath="b"))()
            != fingerprint_of(DataRef(alias="v", relpath="b"), DataRef(alias="v", relpath="a"))()
        )
    with pytest.raises(ValueError, match="at least one"):
        fingerprint_of()


# ---- through the pipeline ----------------------------------------------------------------------------------------

RAW = DataRef(alias="vendor", relpath="raw.csv")


def _pipeline(root: Path, data: Path, cache: Path | None = None) -> Pipeline:
    def raw() -> pl.DataFrame:
        return pl.read_csv(RAW.path())

    def total(raw: pl.DataFrame) -> dict[str, int]:
        return {"sum": int(raw["x"].sum())}

    stages = [Stage("raw", raw, fingerprint=fingerprint_of(RAW)), Stage("total", total, ("raw",))]
    return Pipeline(stages, root, data={"vendor": data}, cache=cache)


def test_two_machines_with_the_same_data_at_different_paths_have_one_signature(tmp_path: Path):
    m1, m2 = tmp_path / "m1" / "vendor", tmp_path / "m2" / "mnt" / "data"
    for d in (m1, m2):
        d.mkdir(parents=True)
        (d / "raw.csv").write_text("x\n1\n2\n3\n")
    r1 = _pipeline(tmp_path / "m1" / "home", m1, tmp_path / "m1" / "cache").run()
    r2 = _pipeline(tmp_path / "m2" / "home", m2, tmp_path / "m2" / "cache").run()
    assert r1.signature == r2.signature
    assert {n: r.output for n, r in r1.results.items()} == {n: r.output for n, r in r2.results.items()}


def test_a_changed_byte_reruns_the_source_and_the_signature_moves(tmp_path: Path):
    data = tmp_path / "vendor"
    data.mkdir()
    (data / "raw.csv").write_text("x\n1\n2\n3\n")
    first = _pipeline(tmp_path / "home", data).run()
    assert _pipeline(tmp_path / "home", data).run().ran() == []
    (data / "raw.csv").write_text("x\n1\n2\n4\n")
    second = _pipeline(tmp_path / "home", data).run()
    assert second.signature != first.signature and second.ran() == ["raw", "total"]


def test_data_roots_survive_rebase_and_are_recorded_but_not_hashed(tmp_path: Path):
    data = tmp_path / "vendor"
    data.mkdir()
    (data / "raw.csv").write_text("x\n1\n")
    p = _pipeline(tmp_path / "a", data).rebase(tmp_path / "b", tmp_path / "store")
    assert p.data == {"vendor": data}
    report = p.run()
    rec = json.loads((tmp_path / "b" / "runs" / f"{report.run_id}.json").read_text())
    assert rec["data"] == {"vendor": str(data)}
    assert rec["stages"]["raw"]["fingerprint"] == content_hash(data / "raw.csv")
