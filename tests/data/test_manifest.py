"""data/manifest.py: the append-only key → output log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chairlift.data.manifest import Manifest

COMMON: dict[str, Any] = dict(stage="s", version=1, spec_hash="h", inputs={"b": "x", "a": "y"}, code="c")


def ref(ch: str) -> str:
    return "json:" + ch * 64


def test_persists_and_later_records_supersede(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    m = Manifest(path)
    m.append(key="k1", output=ref("0"), reason="first", **COMMON)
    m.append(key="k1", output=ref("1"), reason="rebuilt", **COMMON)
    again = Manifest(path)
    rec = again.lookup("k1")
    assert rec is not None and rec.reason == "rebuilt" and list(rec.inputs) == ["a", "b"]
    assert again.lookup("nope") is None
    assert len(path.read_text().splitlines()) == 2  # append-only: both builds stay in the log


def test_missing_file_is_an_empty_manifest_and_creates_nothing(tmp_path: Path):
    m = Manifest(tmp_path / "deep" / "manifest.jsonl")
    assert m.records() == [] and m.lookup("k") is None
    assert not (tmp_path / "deep").exists()


def test_first_append_creates_the_directory(tmp_path: Path):
    m = Manifest(tmp_path / "deep" / "manifest.jsonl")
    m.append(key="k", output=ref("0"), reason="r", **COMMON)
    assert (tmp_path / "deep" / "manifest.jsonl").exists()


def test_blank_lines_are_ignored(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    Manifest(path).append(key="k", output=ref("0"), reason="r", **COMMON)
    path.write_text("\n" + path.read_text() + "\n\n")
    assert Manifest(path).lookup("k") is not None


def test_each_line_is_one_sorted_json_object_with_a_timestamp(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    Manifest(path).append(key="k", output=ref("0"), reason="r", **COMMON)
    line = path.read_text().splitlines()[0]
    obj = json.loads(line)
    assert list(obj) == sorted(obj)
    assert obj["built_at"].endswith("+00:00")


def test_records_returns_one_per_key(tmp_path: Path):
    m = Manifest(tmp_path / "m.jsonl")
    for k in ("a", "b", "a"):
        m.append(key=k, output=ref("0"), reason="r", **COMMON)
    assert sorted(r.key for r in m.records()) == ["a", "b"]


# ---- shared stores: one file per host --------------------------------------------------------------------------------


def test_shared_manifest_writes_one_file_per_host_and_reads_them_all(tmp_path: Path):
    a = Manifest(tmp_path / "manifest.jsonl", shared=True, host="boxA")
    b = Manifest(tmp_path / "manifest.jsonl", shared=True, host="boxB")
    a.append(key="ka", output=ref("0"), reason="built on A", **COMMON)
    b.append(key="kb", output=ref("1"), reason="built on B", **COMMON)
    assert sorted(p.name for p in tmp_path.glob("*.jsonl")) == ["manifest.boxA.jsonl", "manifest.boxB.jsonl"]
    reader = Manifest(tmp_path / "manifest.jsonl", shared=True, host="boxC")
    ka, kb = reader.lookup("ka"), reader.lookup("kb")
    assert ka is not None and kb is not None  # each host reuses the other's work
    assert (ka.host, kb.host) == ("boxA", "boxB")
    assert not (tmp_path / "manifest.boxC.jsonl").exists()  # reading never creates a file


def test_shared_manifest_resolves_one_key_built_on_two_hosts_by_time(tmp_path: Path):
    a = Manifest(tmp_path / "manifest.jsonl", shared=True, host="boxA")
    a.append(key="k", output=ref("0"), reason="first", **COMMON)
    b = Manifest(tmp_path / "manifest.jsonl", shared=True, host="boxB")
    b.append(key="k", output=ref("1"), reason="later", **COMMON)
    rec = Manifest(tmp_path / "manifest.jsonl", shared=True, host="x").lookup("k")
    assert rec is not None and rec.reason == "later"


def test_old_records_without_a_host_still_load(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    line = {
        "stage": "s",
        "key": "k",
        "version": 1,
        "spec_hash": "h",
        "inputs": {},
        "output": ref("0"),
        "code": "c",
        "reason": "r",
        "built_at": "2026-09-30T00:00:00+00:00",
    }
    path.write_text(json.dumps(line) + "\n")
    assert Manifest(path, shared=True, host="h").lookup("k") is not None
