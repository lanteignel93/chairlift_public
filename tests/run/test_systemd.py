"""run/systemd.py: durations, the submit command, sd_notify, and a heartbeat tied to progress."""

from __future__ import annotations

import socket
import time
from pathlib import Path

import pytest

from chairlift.run.systemd import Heartbeat, parse_duration, sd_notify, submit_command, unit_name, watchdog_interval


def test_durations_in_systemd_spelling():
    assert parse_duration("45min") == 2700 and parse_duration("1h") == 3600 and parse_duration("2h30min") == 9000
    for bad in ("", "5", "3 weeks", "-1h"):
        with pytest.raises(ValueError):
            parse_duration(bad)


def test_submit_command_sets_notify_watchdog_and_on_failure(tmp_path: Path):
    cmd = submit_command(
        "chairlift",
        ["s.py:study", "--set", "a=1"],
        "chairlift-x-1",
        "45min",
        True,
        tmp_path,
        {"CHAIRLIFT_PROFILE": "p"},
    )
    assert cmd[:2] == ["systemd-run", "--user"] and "Type=notify" in cmd and "WatchdogSec=45min" in cmd
    assert "OnFailure=chairlift-alert@%n.service" in cmd and "--setenv=CHAIRLIFT_PROFILE=p" in cmd
    assert cmd[-5:] == ["run", "s.py:study", "--set", "a=1", "--no-live"]
    assert unit_name("my study", time.gmtime(0)) == "chairlift-my-study-19700101T000000"


def _socket(tmp_path: Path) -> tuple[socket.socket, dict[str, str]]:
    path = tmp_path / "notify.sock"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.bind(str(path))
    s.settimeout(1)
    return s, {"NOTIFY_SOCKET": str(path), "WATCHDOG_USEC": "2000000"}


def test_sd_notify_sends_a_datagram_and_is_a_no_op_outside_systemd(tmp_path: Path):
    s, env = _socket(tmp_path)
    assert sd_notify("READY=1", env) and s.recv(64) == b"READY=1"
    assert sd_notify("READY=1", {}) is False
    assert watchdog_interval(env) == 2.0 and watchdog_interval({}) is None


def test_the_heartbeat_stops_when_progress_stops(tmp_path: Path):
    s, env = _socket(tmp_path)
    last = [time.monotonic()]
    hb = Heartbeat(2.0, lambda: last[0], env)
    hb.beat()
    assert s.recv(64) == b"WATCHDOG=1" and hb.sent == 1
    last[0] = time.monotonic() - 10  # nothing has happened for longer than the interval
    hb.beat()
    assert hb.sent == 1 and hb.withheld == 1
