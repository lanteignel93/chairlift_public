"""run/events.py and the runner's events: what a run says about itself, in order, without ever hurting the run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from chairlift.run import events
from chairlift.run.dag import Pipeline, Stage
from chairlift.run.events import EventLog, bound, read_events

# ---- the log itself ------------------------------------------------------------------------------------------------


def test_every_event_carries_kind_schema_run_and_sequence(tmp_path: Path):
    log = EventLog(tmp_path / "e.jsonl", "r1")
    log.emit("run_started", study="s")
    log.emit("warning", message="m")
    evs, _ = read_events(tmp_path / "e.jsonl")
    assert [e["kind"] for e in evs] == ["run_started", "warning"]
    assert [e["seq"] for e in evs] == [1, 2]
    assert all(e["schema"] == 1 and e["run_id"] == "r1" and e["ts"].endswith("+00:00") for e in evs)


def test_a_failing_listener_never_fails_the_writer(tmp_path: Path):
    seen: list[str] = []

    def bad(_e: dict[str, Any]) -> None:
        raise RuntimeError("observer bug")

    log = EventLog(tmp_path / "e.jsonl", "r", [bad, lambda e: seen.append(e["kind"])])
    log.emit("run_started")
    assert seen == ["run_started"] and log.listener_errors == 1
    assert read_events(tmp_path / "e.jsonl")[0][0]["kind"] == "run_started"


def test_reader_leaves_a_partial_line_for_next_time(tmp_path: Path):
    path = tmp_path / "e.jsonl"
    EventLog(path, "r").emit("run_started")
    with path.open("a") as fh:
        fh.write('{"kind": "warn')  # the writer is mid-line
    evs, offset = read_events(path)
    assert len(evs) == 1
    with path.open("a") as fh:
        fh.write('ing", "message": "x"}\n')
    more, _ = read_events(path, offset)
    assert [e["kind"] for e in more] == ["warning"]


def test_reader_skips_corrupt_lines_and_handles_missing_files(tmp_path: Path):
    assert read_events(tmp_path / "absent.jsonl") == ([], 0)
    path = tmp_path / "e.jsonl"
    path.write_text('not json\n{"kind": "run_started"}\n')
    evs, _ = read_events(path)
    assert [e["kind"] for e in evs] == ["run_started"]


def test_stage_helpers_are_no_ops_outside_a_run():
    events.progress(1, 2, "x")
    events.metric("ic", 0.1)
    events.warning("w")  # nothing bound: nothing happens, nothing raises


def test_helpers_write_to_the_bound_run_and_name_the_stage(tmp_path: Path):
    log = EventLog(tmp_path / "e.jsonl", "r")
    with bound(log, "fit"):
        events.progress(3, 7, "fold 3")
        events.metric("ic", 0.05, fold=3)
    events.progress(1, 1)  # unbound again
    evs, _ = read_events(tmp_path / "e.jsonl")
    assert [(e["kind"], e["stage"]) for e in evs] == [("progress", "fit"), ("metric", "fit")]
    assert evs[1]["dims"] == {"fold": 3}


# ---- the runner's stream -------------------------------------------------------------------------------------------


def _events(root: Path) -> list[dict[str, Any]]:
    path = next((root / "runs").glob("*.events.jsonl"))
    return read_events(path)[0]


def _pipeline(root: Path, fail: bool = False) -> Pipeline:
    def src() -> pl.DataFrame:
        return pl.DataFrame({"x": [1, 2, 3]})

    def work(src: pl.DataFrame) -> dict[str, int]:
        events.progress(1, 2, "half")
        events.metric("rows", float(src.height))
        if fail:
            raise ValueError("bad fold")
        events.progress(2, 2, "done")
        return {"n": src.height}

    return Pipeline([Stage("src", src, fingerprint=lambda: "v"), Stage("work", work, ("src",))], root)


def test_a_run_writes_started_stages_and_finished_in_order(tmp_path: Path):
    _pipeline(tmp_path).run(meta={"name": "demo"})
    kinds = [(e["kind"], e.get("stage")) for e in _events(tmp_path)]
    assert kinds == [
        ("run_started", None),
        ("stage_started", "src"),
        ("stage_finished", "src"),
        ("stage_started", "work"),
        ("progress", "work"),
        ("metric", "work"),
        ("progress", "work"),
        ("stage_finished", "work"),
        ("run_finished", None),
    ]
    evs = _events(tmp_path)
    assert evs[0]["study"] == "demo" and evs[0]["stages"] == ["src", "work"]
    src_done = evs[2]
    assert src_done["rows"] == 3 and src_done["cols"] == 1 and src_done["peak_rss_mb"] > 0
    assert evs[-1]["status"] == "ok" and evs[-1]["ran"] == 2 and evs[-1]["reused"] == 0


def test_reused_stages_are_reported_as_hits(tmp_path: Path):
    _pipeline(tmp_path).run()
    report = _pipeline(tmp_path).run()
    path = tmp_path / "runs" / f"{report.run_id}.events.jsonl"
    finished = [e for e in read_events(path)[0] if e["kind"] == "stage_finished"]
    assert {e["status"] for e in finished} == {"hit"}


def test_a_failed_stage_is_reported_before_the_exception_propagates(tmp_path: Path):
    with pytest.raises(ValueError, match="bad fold"):
        _pipeline(tmp_path, fail=True).run()
    evs = _events(tmp_path)
    failed = [e for e in evs if e["kind"] == "stage_finished" and e["status"] == "failed"]
    assert failed and failed[0]["stage"] == "work" and "bad fold" in failed[0]["error"]
    assert evs[-1]["kind"] == "run_finished" and evs[-1]["status"] == "failed"


def test_events_never_enter_keys_or_outputs(tmp_path: Path):
    p = _pipeline(tmp_path / "a")
    r1 = p.run()
    for f in (tmp_path / "a" / "runs").glob("*.events.jsonl"):
        f.unlink()
    r2 = _pipeline(tmp_path / "a").run()
    assert r2.ran() == [] and r2.signature == r1.signature
    assert {n: r.key for n, r in r1.results.items()} == {n: r.key for n, r in r2.results.items()}


def test_listeners_receive_the_same_events_that_are_written(tmp_path: Path):
    got: list[dict[str, Any]] = []
    report = _pipeline(tmp_path).run(listeners=[got.append])
    written = read_events(tmp_path / "runs" / f"{report.run_id}.events.jsonl")[0]
    assert [json.dumps(g, sort_keys=True, default=str) for g in got] == [json.dumps(w, sort_keys=True) for w in written]
