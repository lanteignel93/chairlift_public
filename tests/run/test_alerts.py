"""run/alerts.py: failed, died and stalled runs, each delivered once."""

from __future__ import annotations

import datetime as dt
import json
import os
import socket
from pathlib import Path

from chairlift.run.alerts import Delivery, judge


def _record(studies: Path, name: str, status: str, pid: int, last_event_minutes_ago: float | None = None) -> None:
    runs = studies / name / "runs"
    runs.mkdir(parents=True)
    rid = "20261001T000000.000000Z-abc"
    (runs / f"{rid}.json").write_text(
        json.dumps({"run_id": rid, "status": status, "host": socket.gethostname(), "pid": pid, "error": "E"})
    )
    if last_event_minutes_ago is not None:
        ts = (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=last_event_minutes_ago)).isoformat()
        (runs / f"{rid}.events.jsonl").write_text(json.dumps({"kind": "progress", "ts": ts}) + "\n")


def test_rules(tmp_path: Path):
    s = tmp_path / "studies"
    _record(s, "a_failed", "failed", os.getpid())
    _record(s, "b_died", "running", 2**22 + 12345)  # no such process
    _record(s, "c_stalled", "running", os.getpid(), last_event_minutes_ago=90)
    _record(s, "d_fine", "running", os.getpid(), last_event_minutes_ago=5)
    _record(s, "e_ok", "ok", os.getpid())
    found = {a["study"]: a["rule"] for a in judge(s, stall_minutes=60)}
    assert found == {"a_failed": "failed", "b_died": "died", "c_stalled": "stalled"}


def test_each_alert_is_delivered_once_to_the_jsonl_sink(tmp_path: Path):
    s = tmp_path / "studies"
    _record(s, "a", "failed", os.getpid())
    d = Delivery(tmp_path, ["jsonl"], {})
    assert len(d.deliver(judge(s, 60))) == 1
    assert d.deliver(judge(s, 60)) == []
    assert len((tmp_path / "alerts.jsonl").read_text().splitlines()) == 1


def test_the_webhook_sink_needs_its_secret(tmp_path: Path):
    import pytest

    with pytest.raises(RuntimeError, match="webhook"):
        Delivery(tmp_path, ["webhook"], {}).deliver([{"rule": "unit", "study": "x", "run_id": "x"}])
