"""Alerts: what needs a person, from the run records and event streams, delivered once.

Rules (each run is judged on its latest state):
- failed     the run raised (its record says so)
- died       the record says running, but its process is gone (same host) — killed, out of memory, rebooted
- stalled    running, but no event for longer than `[alerts] stall_minutes`
- unit       a systemd OnFailure hook reported a unit (`chairlift alerts unit NAME`)

Sinks (`[alerts] sinks`): "jsonl" appends to <home>/alerts.jsonl, "stdout" prints, "webhook" POSTs the alert as JSON
to the URL in the secret `webhook` (never stored in config or records). An alert is sent once: a state file
remembers (run, rule) pairs already delivered. `chairlift alerts check` exits 1 when it delivered anything (timers,
cron), 0 otherwise.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import socket
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _alive(rec: Mapping[str, Any]) -> bool | None:
    if rec.get("host") != socket.gethostname():
        return None
    try:
        os.kill(int(rec["pid"]), 0)
    except (ProcessLookupError, ValueError, KeyError):
        return False
    except PermissionError:
        return True
    return True


def _last_event_time(events_path: Path) -> dt.datetime | None:
    if not events_path.exists():
        return None
    last = None
    with events_path.open("rb") as fh:
        for line in fh:
            if line.strip():
                last = line
    if last is None:
        return None
    try:
        return dt.datetime.fromisoformat(json.loads(last)["ts"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def judge(studies_dir: Path, stall_minutes: float, now: dt.datetime | None = None) -> list[dict[str, Any]]:
    """Every alert the latest run of each study deserves right now."""
    now = now or dt.datetime.now(dt.UTC)
    out: list[dict[str, Any]] = []
    if not studies_dir.exists():
        return out
    for d in sorted(p for p in studies_dir.iterdir() if p.is_dir()):
        recs = sorted(
            (p for p in (d / "runs").glob("*Z-*.json") if not p.name.endswith(".events.jsonl")), key=lambda p: p.stem
        )
        if not recs:
            continue
        rec = json.loads(recs[-1].read_text())
        base = {"study": d.name, "run_id": rec["run_id"], "host": rec.get("host"), "reason": rec.get("reason", "")}
        status = rec.get("status")
        if status == "failed":
            out.append(base | {"rule": "failed", "detail": rec.get("error", "")})
        elif status == "running":
            if _alive(rec) is False:
                out.append(base | {"rule": "died", "detail": "marked running, process gone"})
            else:
                last = _last_event_time(recs[-1].parent / f"{rec['run_id']}.events.jsonl")
                if last is not None and (now - last).total_seconds() > stall_minutes * 60:
                    mins = (now - last).total_seconds() / 60
                    out.append(
                        base
                        | {"rule": "stalled", "detail": f"no event for {mins:.0f} min (threshold {stall_minutes:g})"}
                    )
    return out


class Delivery:
    def __init__(self, home: Path, sinks: Iterable[str], secrets: Mapping[str, str]) -> None:
        self.home, self.sinks, self.secrets = home, list(sinks), secrets
        self.state_path = home / "alerts.state.json"

    def _seen(self) -> set[str]:
        try:
            return set(json.loads(self.state_path.read_text()))
        except (OSError, json.JSONDecodeError):
            return set()

    def deliver(self, alerts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Send alerts not sent before; returns those sent."""
        seen = self._seen()
        sent: list[dict[str, Any]] = []
        for a in alerts:
            key = f"{a.get('study')}|{a.get('run_id')}|{a['rule']}"
            if key in seen:
                continue
            a = {"at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **a}
            for sink in self.sinks:
                self._send(sink, a)
            seen.add(key)
            sent.append(a)
        if sent:
            self.home.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(sorted(seen)))
            tmp.replace(self.state_path)
        return sent

    def _send(self, sink: str, a: dict[str, Any]) -> None:
        if sink == "jsonl":
            self.home.mkdir(parents=True, exist_ok=True)
            with (self.home / "alerts.jsonl").open("a") as fh:
                fh.write(json.dumps(a, sort_keys=True) + "\n")
        elif sink == "stdout":
            print(f"ALERT {a['rule']}: {a.get('study')} {a.get('run_id', '')} {a.get('detail', '')}")
        elif sink == "webhook":
            url = self.secrets.get("webhook")
            if not url:
                raise RuntimeError("the webhook sink needs a secret named 'webhook'")
            req = urllib.request.Request(url, data=json.dumps(a).encode(), headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        elif sink == "page":
            pass  # rendered by `chairlift report --index` from alerts.jsonl; nothing to push
        else:
            raise ValueError(f"unknown alert sink {sink!r}")
