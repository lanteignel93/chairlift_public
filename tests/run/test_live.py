"""run/live.py: the reducer folds events into a run's state; the renderer shows it."""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path
from typing import Any

from rich.console import Console

from chairlift.run.live import RunState, apply, render, replay


def ev(kind: str, seq: int, **kw: Any) -> dict[str, Any]:
    base = dt.datetime(2026, 9, 30, 12, 0, tzinfo=dt.UTC)
    return {
        "kind": kind,
        "schema": 1,
        "run_id": "r1",
        "seq": seq,
        "ts": (base + dt.timedelta(seconds=seq)).isoformat(),
        **kw,
    }


STREAM = [
    ev("run_started", 1, study="toy", signature="abc", stages=["sources", "fit"]),
    ev("stage_started", 2, stage="sources", will="reuse", reason="unchanged"),
    ev("stage_finished", 3, stage="sources", status="hit", seconds=0.0),
    ev("stage_started", 4, stage="fit", will="run", reason="new key"),
    ev("progress", 5, stage="fit", done=2, total=3, message="fold 2 of 3"),
    ev("metric", 6, stage="fit", name="ic", value=0.051, dims={"fold": 2}),
]


def text(renderable: Any) -> str:
    buf = io.StringIO()
    Console(file=buf, width=140, color_system=None).print(renderable)
    return buf.getvalue()


def test_reducer_tracks_stage_status_progress_and_metrics():
    s = replay(STREAM)
    assert s.status == "running" and s.study == "toy"
    assert s.stages["sources"].status == "hit"
    assert s.stages["fit"].status == "running" and s.stages["fit"].progress == (2, 3, "fold 2 of 3")
    assert s.metrics == [("fit", "ic", 0.051, {"fold": 2})]


def test_reducer_closes_the_run_and_records_failures():
    s = replay(
        [
            *STREAM,
            ev("stage_finished", 7, stage="fit", status="failed", error="ValueError: x"),
            ev("run_finished", 8, status="failed", error="ValueError: x"),
        ]
    )
    assert s.stages["fit"].status == "failed" and s.status == "failed" and s.error == "ValueError: x"
    assert s.finished is not None and s.started is not None and (s.finished - s.started).total_seconds() == 7


def test_unknown_kinds_are_counted_not_fatal():
    s = apply(RunState(), {"kind": "from_the_future", "schema": 9})
    assert s.unknown_kinds == 1


def test_stages_listed_at_start_render_as_pending_before_they_run():
    s = replay(STREAM[:1])
    out = text(render(s, now=dt.datetime(2026, 9, 30, 12, 0, 5, tzinfo=dt.UTC)))
    assert "sources" in out and "fit" in out and "pending" in out


def test_render_shows_header_progress_metrics_and_errors():
    s = replay(STREAM)
    out = text(render(s, now=dt.datetime(2026, 9, 30, 12, 0, 10, tzinfo=dt.UTC)))
    assert "toy" in out and "running" in out and "fold 2 of 3" in out
    assert "ic" in out and "+0.0510" in out and "fold=2" in out
    failed = replay([*STREAM, ev("run_finished", 9, status="failed", error="boom")])
    assert "error: boom" in text(render(failed))


def test_replay_of_a_real_run_matches_its_report(tmp_path: Path):
    from chairlift.run.events import read_events
    from chairlift.verify.toy import pipeline

    p = pipeline(tmp_path / "r")
    report = p.run(meta={"name": "toy"})
    s = replay(read_events(tmp_path / "r" / "runs" / f"{report.run_id}.events.jsonl")[0])
    assert s.status == "ok"
    assert {n: st.status for n, st in s.stages.items()} == report.status()
