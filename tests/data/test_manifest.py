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
