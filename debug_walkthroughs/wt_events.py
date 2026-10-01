"""Walkthrough: from a stage's progress call to the live view's state.

    uv run python debug_walkthroughs/wt_events.py          # asserted run
    uv run python debug_walkthroughs/wt_events.py --pdb    # step through

Two stages on three rows. The second reports progress and a metric from inside its function. The walkthrough runs
the pipeline with an in-process listener (the same hook the live view uses), then replays the event file and checks
that both paths reach the same state.

Under --pdb, useful breakpoints:
  chairlift/run/events.py   EventLog.emit   — `rec` is exactly the line about to be written; `self.listeners` the
                                              observers it is about to call (their exceptions are swallowed)
  chairlift/run/events.py   progress        — `_LOG.get()` is the bound run; outside a run it is None: a no-op
  chairlift/run/live.py     apply           — watch `state.stages[...]` change as each event folds in
What to expect: nine events, in order run_started, stage_started/stage_finished for `src` (ran on the first run),
stage_started, progress, metric, progress, stage_finished for `work`, run_finished.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

from chairlift.run import events
from chairlift.run.dag import Pipeline, Stage
from chairlift.run.events import read_events
from chairlift.run.live import RunState, apply, replay


def src() -> pl.DataFrame:
    return pl.DataFrame({"x": [1, 2, 3]})


def work(src: pl.DataFrame) -> dict[str, int]:
    events.progress(1, 2, "first half")  # written to this run's event file, tagged with stage "work"
    events.metric("rows", float(src.height), part="all")
    events.progress(2, 2, "done")
    return {"n": src.height}


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="wt_events_"))
    live = RunState()
    seen: list[str] = []

    def on_event(e: dict[str, Any]) -> None:
        seen.append(e["kind"])
        apply(live, e)

    p = Pipeline([Stage("src", src, fingerprint=lambda: "v1"), Stage("work", work, ("src",))], root)
    report = p.run(meta={"name": "wt"}, listeners=[on_event])
    print("events seen in-process:", seen)
    assert seen == [
        "run_started",
        "stage_started",
        "stage_finished",
        "stage_started",
        "progress",
        "metric",
        "progress",
        "stage_finished",
        "run_finished",
    ]

    path = root / "runs" / f"{report.run_id}.events.jsonl"
    from_file, _ = read_events(path)
    print(f"{len(from_file)} events in {path.name}")
    replayed = replay(from_file)
    # both paths fold the same events, so they must agree on every stage
    assert {n: s.status for n, s in replayed.stages.items()} == {n: s.status for n, s in live.stages.items()}
    assert replayed.stages["work"].progress == (2, 2, "done")
    assert replayed.metrics == [("work", "rows", 3.0, {"part": "all"})]
    assert replayed.status == "ok" and replayed.stages["src"].rows == 3
    print("state:", {n: s.status for n, s in replayed.stages.items()}, "metrics:", replayed.metrics)
    print("walkthrough passed")


if __name__ == "__main__":
    if "--pdb" in sys.argv:
        import pdb

        pdb.set_trace()  # `s` into main(), then `b chairlift/run/events.py:<EventLog.emit line>`
    main()
