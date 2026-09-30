"""`chairlift` command line: run, plan, inspect and read the log of a study's pipeline.

A study is referenced as `module:factory` or `path/to/file.py:factory`, where the factory builds the study's pipeline
for a run root: `factory(root, **params) -> Pipeline`. `--set name=value` passes keyword parameters to the factory;
values are parsed as JSON when they can be (5 → int, 0.1 → float, true → bool) and kept as strings otherwise.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import click
import polars as pl
from rich.console import Console
from rich.table import Table

from chairlift.data.manifest import Manifest
from chairlift.run.dag import Pipeline


def _load_factory(ref: str) -> Any:
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
        return getattr(module, attr)
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


def _build(study: str, root: Path, sets: tuple[str, ...]) -> Pipeline:
    factory = _load_factory(study)
    try:
        pipe = factory(root, **_parse_sets(sets))
    except TypeError as exc:
        raise click.UsageError(f"the factory rejected the parameters: {exc}") from exc
    if not isinstance(pipe, Pipeline):
        raise click.UsageError(f"{study} returned {type(pipe).__qualname__}, not a Pipeline")
    return pipe


_RICH = {"hit": "dim", "ran": "green", "run": "yellow", "upstream": "cyan"}


def _console() -> Console:
    # sys.stdout read at call time, so CliRunner captures it; rich colours only when it is a terminal
    return Console(file=sys.stdout, highlight=False, soft_wrap=False)


def _table(*columns: str) -> Table:
    t = Table(box=None, pad_edge=False, header_style="bold")
    for c in columns:
        t.add_column(c, no_wrap=c != "reason", overflow="fold")
    return t


study_arg = click.argument("study")
root_opt = click.option(
    "--root", type=click.Path(path_type=Path), default=Path("runs/default"), show_default=True, help="Run root."
)
set_opt = click.option("--set", "sets", multiple=True, metavar="NAME=VALUE", help="Factory parameter (repeatable).")
target_opt = click.option("--target", "targets", multiple=True, help="Run only these stages and their ancestors.")
json_opt = click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")


@click.group()
@click.version_option(package_name="chairlift")
def main() -> None:
    """chairlift: a research pipeline in which the protocol is code."""


@main.command()
@study_arg
@root_opt
@set_opt
@target_opt
@click.option("--reason", default="", help="Why this build; recorded on every stage that runs.")
@json_opt
def run(study: str, root: Path, sets: tuple[str, ...], targets: tuple[str, ...], reason: str, as_json: bool) -> None:
    """Run a study's pipeline; reuse every stage whose inputs, spec and version are unchanged."""
    pipe = _build(study, root, sets)
    report = pipe.run(list(targets) or None, reason=reason)
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
    con.print(f"\n{len(ran)} ran, {len(report.results) - len(ran)} reused · root {root}")


@main.command()
@study_arg
@root_opt
@set_opt
@target_opt
@json_opt
def plan(study: str, root: Path, sets: tuple[str, ...], targets: tuple[str, ...], as_json: bool) -> None:
    """Show what `run` would do, without running anything."""
    items = _build(study, root, sets).plan(list(targets) or None)
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
@set_opt
@click.option("--rows", default=10, show_default=True, help="Rows to print for a frame.")
def show(study: str, stage: str, root: Path, sets: tuple[str, ...], rows: int) -> None:
    """Print a stage's stored output (building it and its ancestors first if needed)."""
    pipe = _build(study, root, sets)
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
@root_opt
@click.option("--stage", help="Only this stage's records.")
@click.option("--last", default=20, show_default=True, help="Most recent records to show.")
@json_opt
def log(root: Path, stage: str | None, last: int, as_json: bool) -> None:
    """Read the manifest: what was built, when, from which inputs and code, and why."""
    path = root / "manifest.jsonl"
    if not path.exists():
        raise click.UsageError(f"no manifest at {path}")
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if stage:
        lines = [r for r in lines if r["stage"] == stage]
    lines = lines[-last:]
    if as_json:
        click.echo(json.dumps(lines, indent=1))
        return
    t = _table("built_at", "stage", "v", "output", "code", "reason")
    for r in lines:
        t.add_row(r["built_at"], r["stage"], str(r["version"]), r["output"][:20] + "…", r["code"], r["reason"])
    con = _console()
    con.print(t)
    con.print(f"\n{len(Manifest(path).records())} keys in the manifest")
