"""Running under systemd: submit a study as a transient user unit with a watchdog, and keep the watchdog honest.

    chairlift submit STUDY|EXP.toml [--set ...] [--watchdog 45min]
        → systemd-run --user --unit chairlift-<study>-<time> -p Type=notify -p NotifyAccess=all
                       -p WatchdogSec=45min -p OnFailure=chairlift-alert@%n.service  chairlift run ... --no-live
    chairlift systemd install   → ~/.config/systemd/user/chairlift-alert@.service (runs `chairlift alerts unit %i`)

The heartbeat (sd_notify WATCHDOG=1) is tied to progress, not to the process being alive: it is sent only while the
run's event stream has advanced within the last watchdog interval. A stage that hangs, or loops without reporting,
stops the heartbeat; systemd kills the unit and OnFailure raises the alert (a timer that ticks on its own would have
kept a dead run looking alive).
"""

from __future__ import annotations

import os
import re
import shlex
import socket
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

ALERT_TEMPLATE = """[Unit]
Description=chairlift alert for %i

[Service]
Type=oneshot
ExecStart={exe} alerts unit %i
"""


def parse_duration(text: str) -> float:
    """'45min', '1h', '90s', '2h30min' → seconds (systemd's spelling)."""
    units = {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "hr": 3600, "d": 86400}
    total, pos = 0.0, 0
    for m in re.finditer(r"\s*(\d+(?:\.\d+)?)\s*([a-z]+)", text.strip().lower()):
        if m.start() != pos or m.group(2) not in units:
            raise ValueError(f"not a duration: {text!r}")
        total += float(m.group(1)) * units[m.group(2)]
        pos = m.end()
    if pos != len(text.strip().lower()) or total <= 0:
        raise ValueError(f"not a duration: {text!r}")
    return total


def sd_notify(message: str, env: dict[str, str] | None = None) -> bool:
    """Send one sd_notify datagram; False when not under systemd (no NOTIFY_SOCKET)."""
    addr = (env if env is not None else os.environ).get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):
        addr = "\0" + addr[1:]  # abstract namespace
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
        s.connect(addr)
        s.sendall(message.encode())
    return True


class Heartbeat:
    """WATCHDOG=1 every `interval / 2` seconds while `last_event()` is within `interval` of now."""

    def __init__(self, interval: float, last_event: Callable[[], float], env: dict[str, str] | None = None) -> None:
        self.interval, self.last_event, self.env = interval, last_event, env
        self._stop = threading.Event()
        self.sent = 0
        self.withheld = 0
        self._thread = threading.Thread(target=self._loop, daemon=True, name="chairlift-watchdog")

    def start(self) -> Heartbeat:
        sd_notify("READY=1", self.env)
        self._thread.start()
        return self

    def beat(self) -> None:
        if time.monotonic() - self.last_event() < self.interval:
            if sd_notify("WATCHDOG=1", self.env):
                self.sent += 1
        else:
            self.withheld += 1  # no progress for a whole interval: let systemd judge the run

    def _loop(self) -> None:
        while not self._stop.wait(self.interval / 2):
            self.beat()

    def stop(self) -> None:
        self._stop.set()
        sd_notify("STOPPING=1", self.env)


def watchdog_interval(env: dict[str, str] | None = None) -> float | None:
    """systemd passes WATCHDOG_USEC to a unit with WatchdogSec set."""
    usec = (env if env is not None else os.environ).get("WATCHDOG_USEC")
    return int(usec) / 1e6 if usec else None


def submit_command(
    exe: str, run_args: Sequence[str], unit: str, watchdog: str, on_failure: bool, workdir: Path, env: dict[str, str]
) -> list[str]:
    """The systemd-run invocation for one chairlift run (pure: tested without systemd)."""
    cmd = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        "--collect",
        f"--working-directory={workdir}",
        "-p",
        "Type=notify",
        "-p",
        "NotifyAccess=all",
        "-p",
        f"WatchdogSec={watchdog}",
    ]
    if on_failure:
        cmd += ["-p", "OnFailure=chairlift-alert@%n.service"]
    for k, v in sorted(env.items()):
        cmd += [f"--setenv={k}={v}"]
    return [*cmd, exe, "run", *run_args, "--no-live"]


def unit_name(study: str, now: time.struct_time | None = None) -> str:
    t = time.strftime("%Y%m%dT%H%M%S", now or time.gmtime())
    return "chairlift-" + re.sub(r"[^A-Za-z0-9_.-]", "-", study) + "-" + t


def quote(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)
