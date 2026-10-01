"""`chairlift` command line: run, plan, inspect, identify and configure a study's pipeline.

A study is referenced as `module:factory` or `path/to/file.py:factory`, where the factory builds the study's pipeline
for a run root: `factory(root, **params) -> Pipeline`. `--set name=value` passes keyword parameters to the factory;
values are parsed as JSON when they can be (5 → int, 0.1 → float, true → bool) and kept as strings otherwise.

Where things go comes from the site configuration (`chairlift config show --explain`): a study named N runs under
`<paths.home>/studies/N/` and shares `<paths.home>/store/` with every other study on the machine. The name is the
study module's `STUDY_NAME`, or `--name`. `--root DIR` instead makes one self-contained directory (its own store).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import polars as pl
from rich.console import Console
from rich.live import Live
from rich.table import Table

from chairlift.core.config import STARTER, ConfigError, Resolved, load_config
from chairlift.data.manifest import Manifest
from chairlift.run.dag import Pipeline, RunReport
from chairlift.run.events import read_events
from chairlift.run.live import RunState, apply, render, replay

# ---- helpers -----------------------------------------------------------------------------------------------------


@dataclass
class Ctx:
    profile: str | None
    _resolved: Resolved | None = None

    def config(self) -> Resolved:
        if self._resolved is None:
            try:
                self._resolved = load_config(profile=self.profile)
            except ConfigError as exc:
                raise click.UsageError(f"configuration: {exc}") from exc
        return self._resolved


def _load_factory(ref: str) -> tuple[Any, Any]:
    target, sep, attr = ref.rpartition(":")
    if not sep or not target or not attr:
        raise click.BadParameter(f"expected module:factory or file.py:factory, got {ref!r}", param_hint="STUDY")
    if target.endswith(".py"):
        path = Path(target).resolve()
        if not path.exists():
            raise click.BadParameter(f"no such file {path}", param_hint="STUDY")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(target)
        except ModuleNotFoundError as exc:
            raise click.BadParameter(f"cannot import {target!r}: {exc}", param_hint="STUDY") from exc
    try:
        return module, getattr(module, attr)
    except AttributeError as exc:
        raise click.BadParameter(f"{target!r} has no attribute {attr!r}", param_hint="STUDY") from exc


def _parse_sets(pairs: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        name, sep, raw = pair.partition("=")
        if not sep or not name:
            raise click.BadParameter(f"expected name=value, got {pair!r}", param_hint="--set")
        try:
            out[name] = json.loads(raw)
        except json.JSONDecodeError:
            out[name] = raw
    return out


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _study_name(ref: str, module: Any, explicit: str | None) -> str:
    name = explicit or getattr(module, "STUDY_NAME", None)
    if name is None:
        target, _, attr = ref.rpartition(":")
        base = Path(target).stem if target.endswith(".py") else target.rsplit(".", 1)[-1]
        name = base if attr == "pipeline" else f"{base}-{attr}"
    if not isinstance(name, str) or not _NAME.match(name):
        raise click.BadParameter(f"study name {name!r}: letters, digits, '.', '_', '-' only", param_hint="--name")
    return name


@dataclass
class Built:
    pipe: Pipeline
    name: str
    root: Path
    shared_store: bool


def _build(ctx: Ctx, study: str, root: Path | None, sets: tuple[str, ...], name: str | None) -> Built:
    module, factory = _load_factory(study)
    sname = _study_name(study, module, name)
    store: Path | None = None
    cfg = ctx.config().config
    cache = cfg.paths.cache_dir()
    if root is None:
        root, store = cfg.paths.study_dir(sname), cfg.paths.store_dir()
    else:
        cache = root / "cache"
    try:
        pipe = factory(root, **_parse_sets(sets))
    except TypeError as exc:
        raise click.UsageError(f"the factory rejected the parameters: {exc}") from exc
    if not isinstance(pipe, Pipeline):
        raise click.UsageError(f"{study} returned {type(pipe).__qualname__}, not a Pipeline")
    pipe = pipe.rebase(root, store, data=cfg.data, cache=cache)  # data roots come from this machine's config
    return Built(pipe, sname, root, store is not None)


_RICH = {"hit": "dim", "ran": "green", "run": "yellow", "upstream": "cyan"}


def _console() -> Console:
    # sys.stdout read at call time, so CliRunner captures it; rich colours only when it is a terminal;
    # piped or captured output gets a wide fixed width so columns are never truncated mid-value
    width = None if sys.stdout.isatty() else 240
    return Console(file=sys.stdout, highlight=False, soft_wrap=False, width=width)


def _table(*columns: str) -> Table:
    t = Table(box=None, pad_edge=False, header_style="bold")
    for c in columns:
        t.add_column(c, no_wrap=c not in ("reason", "value"), overflow="fold")
    return t


study_arg = click.argument("study")
root_opt = click.option(
    "--root",
    type=click.Path(path_type=Path),
    default=None,
    help="A self-contained run directory (own store). Default: <paths.home>/studies/<name>, shared store.",
)
name_opt = click.option("--name", default=None, help="Study name (default: the module's STUDY_NAME).")
set_opt = click.option("--set", "sets", multiple=True, metavar="NAME=VALUE", help="Factory parameter (repeatable).")
target_opt = click.option("--target", "targets", multiple=True, help="Run only these stages and their ancestors.")
json_opt = click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")


# ---- commands ----------------------------------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="chairlift")
@click.option(
    "--profile", default=None, help="Site configuration profile (overrides CHAIRLIFT_PROFILE and the host map)."
)
@click.pass_context
def main(ctx: click.Context, profile: str | None) -> None:
    """chairlift: a research pipeline in which the protocol is code."""
    ctx.obj = Ctx(profile)


@main.command()
@study_arg
@root_opt
@name_opt
@set_opt
@target_opt
@click.option("--reason", default="", help="Why this build; recorded on every stage that runs.")
@click.option("--live/--no-live", default=None, help="Animate the run (default: on a terminal).")
@json_opt
@click.pass_obj
def run(
    ctx: Ctx,
    study: str,
    root: Path | None,
    name: str | None,
    sets: tuple[str, ...],
    targets: tuple[str, ...],
    reason: str,
    live: bool | None,
    as_json: bool,
) -> None:
    """Run a study's pipeline; reuse every stage whose inputs, spec and version are unchanged."""
    b = _build(ctx, study, root, sets, name)
    meta: dict[str, Any] = {"study": study, "name": b.name, "params": _parse_sets(sets)}
    if b.shared_store:
        meta["config"] = ctx.config().record()
    animate = (sys.stdout.isatty() if live is None else live) and not as_json
    if animate:
        report = _run_live(b, list(targets) or None, reason, meta)
        ran = report.ran()
        _console().print(f"{len(ran)} ran, {len(report.results) - len(ran)} reused · {b.name} · run {report.run_id}")
        return
    report = b.pipe.run(list(targets) or None, reason=reason, meta=meta)
    if as_json:
        click.echo(json.dumps([r.__dict__ for r in report.results.values()], indent=1))
        return
    t = _table("stage", "status", "key", "seconds", "reason")
    for r in report.results.values():
        took = f"{r.seconds:.2f}" if r.status == "ran" else ""
        t.add_row(r.stage, f"[{_RICH[r.status]}]{r.status}[/]", r.key[:12], took, r.reason)
    con = _console()
    con.print(t)
    ran = report.ran()
    con.print(f"\n{len(ran)} ran, {len(report.results) - len(ran)} reused · {b.name} · run {report.run_id} · {b.root}")


def _run_live(b: Built, targets: list[str] | None, reason: str, meta: dict[str, Any]) -> RunReport:
    """Run with the live view: events fold into a state the display re-renders ten times a second."""
    state = RunState()

    def on_event(e: dict[str, Any]) -> None:
        apply(state, e)

    with Live(console=_console(), refresh_per_second=10, get_renderable=lambda: render(state), transient=False):
        return b.pipe.run(targets, reason=reason, meta=meta, listeners=[on_event])


@main.command()
@study_arg
@root_opt
@name_opt
@set_opt
@target_opt
@json_opt
@click.pass_obj
def plan(
    ctx: Ctx,
    study: str,
    root: Path | None,
    name: str | None,
    sets: tuple[str, ...],
    targets: tuple[str, ...],
    as_json: bool,
) -> None:
    """Show what `run` would do, without running anything."""
    items = _build(ctx, study, root, sets, name).pipe.plan(list(targets) or None)
    if as_json:
        click.echo(json.dumps([i.__dict__ for i in items], indent=1))
        return
    t = _table("stage", "status", "key", "reason")
    for i in items:
        t.add_row(i.stage, f"[{_RICH[i.status]}]{i.status}[/]", (i.key or "")[:12], i.reason)
    _console().print(t)


@main.command()
@study_arg
@click.argument("stage")
@root_opt
@name_opt
@set_opt
@click.option("--rows", default=10, show_default=True, help="Rows to print for a frame.")
@click.pass_obj
def show(
    ctx: Ctx, study: str, stage: str, root: Path | None, name: str | None, sets: tuple[str, ...], rows: int
) -> None:
    """Print a stage's stored output (building it and its ancestors first if needed)."""
    pipe = _build(ctx, study, root, sets, name).pipe
    if stage not in pipe.stages:
        raise click.BadParameter(f"unknown stage {stage!r}; stages: {', '.join(pipe.order)}", param_hint="STAGE")
    report = pipe.run([stage])
    out = pipe.load(stage, report)
    if isinstance(out, pl.DataFrame):
        with pl.Config(tbl_rows=rows, tbl_cols=-1):
            click.echo(out.head(rows))
        click.echo(f"{out.height} rows × {out.width} columns · {report.results[stage].output}")
    else:
        click.echo(json.dumps(out, indent=1))


@main.command()
@study_arg
@root_opt
@name_opt
@set_opt
@target_opt
@json_opt
@click.pass_obj
def signature(
    ctx: Ctx,
    study: str,
    root: Path | None,
    name: str | None,
    sets: tuple[str, ...],
    targets: tuple[str, ...],
    as_json: bool,
) -> None:
    """Print the run signature (inputs-only identity) and each stage's plan key, without running."""
    sig = _build(ctx, study, root, sets, name).pipe.signature(list(targets) or None)
    if as_json:
        click.echo(json.dumps({"signature": sig.signature, "plan_keys": sig.plan_keys}, indent=1))
        return
    t = _table("stage", "plan key")
    for stage_name, k in sig.plan_keys.items():
        t.add_row(stage_name, k[:16])
    con = _console()
    con.print(t)
    con.print(f"\nsignature {sig.signature}")


@main.command()
@root_opt
@click.option("--stage", help="Only this stage's records.")
@click.option("--last", default=20, show_default=True, help="Most recent records to show.")
@json_opt
@click.pass_obj
def log(ctx: Ctx, root: Path | None, stage: str | None, last: int, as_json: bool) -> None:
    """Read the manifest: what was built, when, from which inputs and code, and why (default: the shared store's)."""
    path = (root if root is not None else ctx.config().config.paths.store_dir()) / "manifest.jsonl"
    manifest = Manifest(path, shared=root is None)
    files = manifest.files()
    if not files:
        raise click.UsageError(f"no manifest at {path.parent}")
    lines = sorted(
        (json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()),
        key=lambda r: r["built_at"],
    )
    if stage:
        lines = [r for r in lines if r["stage"] == stage]
    lines = lines[-last:]
    if as_json:
        click.echo(json.dumps(lines, indent=1))
        return
    t = _table("built_at", "host", "stage", "v", "output", "code", "reason")
    for r in lines:
        t.add_row(
            r["built_at"],
            r.get("host", ""),
            r["stage"],
            str(r["version"]),
            r["output"][:20] + "…",
            r["code"],
            r["reason"],
        )
    con = _console()
    con.print(t)
    con.print(f"\n{len(manifest.records())} keys across {len(files)} manifest file(s)")


def _records(ctx: Ctx, root: Path | None, study: str | None) -> list[dict[str, Any]]:
    if root is not None:
        files = sorted((root / "runs").glob("*.json"))
        where = root / "runs"
    else:
        studies = ctx.config().config.paths.studies_dir()
        files = sorted(studies.glob(f"{study or '*'}/runs/*.json"))
        where = studies
    if not files:
        raise click.UsageError(f"no runs recorded under {where}")
    recs = [json.loads(p.read_text()) for p in files]
    return sorted(recs, key=lambda r: r["run_id"])


@main.group(invoke_without_command=True)
@root_opt
@click.option("--study", "study_name", default=None, help="Only this study's runs.")
@click.pass_context
def runs(ctx: click.Context, root: Path | None, study_name: str | None) -> None:
    """List recorded runs across studies (newest last); `runs show RUN` prints one record."""
    c: Ctx = ctx.obj
    ctx.meta["runs_root"], ctx.meta["runs_study"] = root, study_name
    if ctx.invoked_subcommand is not None:
        return
    t = _table("run_id", "study", "status", "signature", "params", "reason")
    for r in _records(c, root, study_name):
        meta = r.get("meta", {})
        t.add_row(
            r["run_id"],
            str(meta.get("name", meta.get("study", ""))),
            r["status"],
            r["signature"][:12],
            json.dumps(meta.get("params", {})),
            r.get("reason", ""),
        )
    _console().print(t)


@runs.command("show")
@click.argument("run")
@click.pass_context
def runs_show(ctx: click.Context, run: str) -> None:
    """Print a run record by run id or signature prefix (the newest match)."""
    recs = _records(ctx.obj, ctx.meta["runs_root"], ctx.meta["runs_study"])
    matches = [r for r in recs if r["run_id"].startswith(run) or r["signature"].startswith(run)]
    if not matches:
        raise click.BadParameter(f"no run or signature starting with {run!r}", param_hint="RUN")
    click.echo(json.dumps(matches[-1], indent=1, sort_keys=True))


@main.group()
def config() -> None:
    """Site configuration: show, check, or start a file."""


@config.command("show")
@click.option("--explain", is_flag=True, help="Show which layer set each value.")
@json_opt
@click.pass_obj
def config_show(ctx: Ctx, explain: bool, as_json: bool) -> None:
    """Print the resolved site configuration for this machine."""
    r = ctx.config()
    if as_json:
        click.echo(json.dumps(r.record(), indent=1))
        return
    t = _table("key", "value", "layer") if explain else _table("key", "value")
    for key, value, layer in r.explain():
        t.add_row(key, str(value), layer) if explain else t.add_row(key, str(value))
    con = _console()
    con.print(t)
    p = r.config.paths
    con.print(f"\nprofile {r.profile or '-'} · store {p.store_dir()} · studies {p.studies_dir()}")
    con.print("files: " + (", ".join(r.files) if r.files else "none (defaults)"))


@config.command("check")
@click.pass_obj
def config_check(ctx: Ctx) -> None:
    """Validate every layer; exit non-zero on the first problem."""
    r = ctx.config()
    click.echo(f"ok · {len(r.files)} file(s) · profile {r.profile or '-'}")


@config.command("init")
@click.option("--project", is_flag=True, help="Write ./chairlift.toml instead of the user file.")
@click.option("--force", is_flag=True, help="Overwrite an existing file.")
def config_init(project: bool, force: bool) -> None:
    """Write a commented starter configuration file."""
    import platformdirs

    path = Path("chairlift.toml") if project else platformdirs.user_config_path("chairlift") / "config.toml"
    if path.exists() and not force:
        raise click.UsageError(f"{path} exists; use --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STARTER)
    click.echo(f"wrote {path}")


def _latest(ctx: Ctx, root: Path | None, study_name: str | None) -> tuple[Path, dict[str, Any]]:
    """The newest run record (and its path) for a study, or for a self-contained root."""
    if root is not None:
        runs_dir = root / "runs"
    elif study_name is not None:
        runs_dir = ctx.config().config.paths.study_dir(study_name) / "runs"
    else:
        candidates = list(ctx.config().config.paths.studies_dir().glob("*/runs/*.json"))
        if not candidates:
            raise click.UsageError("no runs recorded yet")
        newest = max(candidates, key=lambda p: p.stem)  # run ids start with their UTC time
        runs_dir = newest.parent
    records = sorted(runs_dir.glob("*.json"))
    if not records:
        raise click.UsageError(f"no runs recorded under {runs_dir}")
    rec_path = max(records, key=lambda p: p.stem)
    return rec_path, json.loads(rec_path.read_text())


def _alive(rec: dict[str, Any]) -> bool | None:
    """Whether a run's process still exists (None when it ran on another host)."""
    if rec.get("host") != socket.gethostname():
        return None
    try:
        os.kill(int(rec["pid"]), 0)
    except (ProcessLookupError, ValueError, KeyError):
        return False
    except PermissionError:
        return True
    return True


@main.command()
@root_opt
@click.option("--study", "study_name", default=None, help="Study name (default: the most recent run of any study).")
@click.option("--interval", default=0.25, show_default=True, help="Seconds between reads of the event file.")
@click.option("--once", is_flag=True, help="Print the current state and exit, even if the run is still going.")
@click.pass_obj
def watch(ctx: Ctx, root: Path | None, study_name: str | None, interval: float, once: bool) -> None:
    """Follow a run from its event file: the same live view as `run`, from any terminal."""
    rec_path, rec = _latest(ctx, root, study_name)
    events_path = rec_path.parent / f"{rec['run_id']}.events.jsonl"
    evs, offset = read_events(events_path)
    state = replay(evs)
    con = _console()
    if once or state.status in ("ok", "failed") or not sys.stdout.isatty():
        con.print(render(state))
        return
    with Live(console=con, refresh_per_second=10, get_renderable=lambda: render(state)):
        while state.status not in ("ok", "failed"):
            time.sleep(interval)
            new, offset = read_events(events_path, offset)
            for ev in new:
                apply(state, ev)
            if not new and _alive(rec) is False:
                state.status, state.error = "failed", "the run's process is gone (killed, or the machine restarted)"


@main.command()
@root_opt
@click.option("--study", "study_name", default=None, help="Study name (default: the most recent run of any study).")
@json_opt
@click.pass_obj
def status(ctx: Ctx, root: Path | None, study_name: str | None, as_json: bool) -> None:
    """The latest run's state. Exit 0 ok, 1 failed or died, 2 still running: for scripts and timers."""
    _, rec = _latest(ctx, root, study_name)
    state = rec["status"]
    if state == "running" and _alive(rec) is False:
        state = "died"
    out = {"run_id": rec["run_id"], "status": state, "study": rec.get("meta", {}).get("name"), "host": rec.get("host")}
    click.echo(json.dumps(out) if as_json else f"{out['study'] or '-'}  {out['run_id']}  {state}")
    raise SystemExit({"ok": 0, "failed": 1, "died": 1, "running": 2}.get(state, 1))
