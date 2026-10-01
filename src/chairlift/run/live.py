"""The live view: a pure reducer from events to a run's state, and a rich renderer of that state.

`chairlift run` feeds it events in-process and animates it; `chairlift watch` feeds it the same events from the file,
so a run looks the same whether you started it here, in tmux, or under systemd. The reducer has no I/O and no
clock of its own (elapsed time is passed in), so it is tested on plain event lists.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from rich.console import Group, RenderableType
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

Metric = tuple[str, str, float, dict[str, Any]]  # stage, name, value, dimensions


@dataclass
class StageState:
    name: str
    status: str = "pending"  # pending · checking · running · hit · ran · failed
    seconds: float | None = None
    rows: int | None = None
    cols: int | None = None
    progress: tuple[int, int | None, str] | None = None
    reason: str = ""
    error: str | None = None
    peak_rss_mb: float | None = None


@dataclass
class RunState:
    run_id: str = ""
    study: str = ""
    signature: str = ""
    started: dt.datetime | None = None
    finished: dt.datetime | None = None
    status: str = "waiting"  # waiting · running · ok · failed
    stages: dict[str, StageState] = field(default_factory=dict[str, StageState])
    metrics: list[Metric] = field(default_factory=list[Metric])
    warnings: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    error: str | None = None
    unknown_kinds: int = 0


def _ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def apply(state: RunState, ev: dict[str, Any]) -> RunState:
    """Fold one event into the state (mutates and returns it)."""
    kind = ev.get("kind")
    if kind == "run_started":
        state.run_id = ev["run_id"]
        state.study = str(ev.get("study") or "")
        state.signature = ev.get("signature", "")
        state.started = _ts(ev["ts"])
        state.status = "running"
        for name in ev.get("stages", []):
            state.stages.setdefault(name, StageState(name))
    elif kind == "stage_started":
        st = state.stages.setdefault(ev["stage"], StageState(ev["stage"]))
        st.status = "running" if ev.get("will") == "run" else "checking"
        st.reason = ev.get("reason", "")
    elif kind == "progress" and ev.get("stage") in state.stages:
        state.stages[ev["stage"]].progress = (int(ev["done"]), ev.get("total"), ev.get("message", ""))
    elif kind == "metric":
        state.metrics.append((str(ev.get("stage")), ev["name"], float(ev["value"]), dict(ev.get("dims", {}))))
    elif kind == "warning":
        state.warnings.append((str(ev.get("stage")), ev["message"]))
    elif kind == "stage_finished":
        st = state.stages.setdefault(ev["stage"], StageState(ev["stage"]))
        st.status = ev["status"]
        st.seconds = ev.get("seconds")
        st.rows, st.cols = ev.get("rows"), ev.get("cols")
        st.error = ev.get("error")
        st.peak_rss_mb = ev.get("peak_rss_mb")
    elif kind == "run_finished":
        state.status = ev["status"]
        state.finished = _ts(ev["ts"])
        state.error = ev.get("error")
    else:
        state.unknown_kinds += 1
    return state


def replay(events: list[dict[str, Any]]) -> RunState:
    state = RunState()
    for ev in events:
        apply(state, ev)
    return state


_ICON = {"pending": ("○", "dim"), "hit": ("·", "dim"), "ran": ("✓", "green"), "failed": ("✗", "bold red")}


def render(state: RunState, now: dt.datetime | None = None, max_metrics: int = 6) -> RenderableType:
    """The whole view as one rich renderable: header, stage table, latest metrics, warnings."""
    now = now or dt.datetime.now(dt.UTC)
    end = state.finished or now
    elapsed = (end - state.started).total_seconds() if state.started else 0.0
    colour = {"running": "yellow", "ok": "green", "failed": "bold red"}.get(state.status, "dim")
    header = Text.assemble(
        ("chairlift ", "bold"),
        (state.study or "-", "bold cyan"),
        "  ",
        (state.status, colour),
        f"  {elapsed:6.1f}s  ",
        (f"run {state.run_id}", "dim"),
    )
    t = Table(box=None, pad_edge=False, show_header=True, header_style="bold dim")
    for col, wrap in (
        ("", False),
        ("stage", False),
        ("status", False),
        ("seconds", False),
        ("shape", False),
        ("progress", False),
        ("detail", True),
    ):
        t.add_column(col, no_wrap=not wrap, overflow="fold")
    for st in state.stages.values():
        icon: RenderableType
        if st.status in ("running", "checking"):
            icon = Spinner("dots", style="yellow")
        else:
            glyph, style = _ICON.get(st.status, ("?", "dim"))
            icon = Text(glyph, style=style)
        shape = f"{st.rows}×{st.cols}" if st.rows is not None else ""
        bar: RenderableType = Text("")
        detail = Text(st.error or "", style="red") if st.error else Text("")
        if st.progress is not None:
            done, total, msg = st.progress
            bar = ProgressBar(total=total, completed=done, width=20) if total else Text(f"{done}", style="dim")
            if not st.error:
                detail = Text(f"{done}/{total} {msg}" if total else msg, style="dim")
        secs = f"{st.seconds:.2f}" if st.seconds is not None else ""
        t.add_row(
            icon, st.name, Text(st.status, style=_ICON.get(st.status, ("", "yellow"))[1]), secs, shape, bar, detail
        )
    parts: list[RenderableType] = [header, t]
    if state.metrics:
        m = Table(box=None, pad_edge=False, show_header=True, header_style="bold dim", title_justify="left")
        for col in ("stage", "metric", "value", "where"):
            m.add_column(col, no_wrap=True)
        for stage, name, value, dims in state.metrics[-max_metrics:]:
            m.add_row(stage, name, f"{value:+.4f}", " ".join(f"{k}={v}" for k, v in dims.items()))
        parts += [Text("latest metrics", style="bold dim"), m]
    for stage, msg in state.warnings[-3:]:
        parts.append(Text(f"warning [{stage}] {msg}", style="yellow"))
    if state.error:
        parts.append(Text(f"error: {state.error}", style="bold red"))
    return Group(*parts)
