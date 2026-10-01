"""Generate docs/guide/08-cli-reference.md from the click definitions; tests/test_docs.py fails when it is stale.

uv run python scripts/gen_cli_reference.py          # rewrite the page
"""

from __future__ import annotations

import inspect
from pathlib import Path

import click

from chairlift.run.cli import main

OUT = Path(__file__).parents[1] / "docs" / "guide" / "08-cli-reference.md"
HEADER = """# 8. CLI reference

Generated from the click definitions by `scripts/gen_cli_reference.py`; do not edit by hand (`tests/test_docs.py`
fails when this page and the code disagree). Every command also answers `--help`.

Conventions shared by most commands:
- `STUDY` is `module:factory`, `path/to/file.py:factory`, or an experiment file `*.toml` ([7](07-experiments.md)).
- Without `--root`, a study runs under `<paths.home>/studies/<name>/` and shares `<paths.home>/store/`
  ([2](02-configure.md)); `--root DIR` makes one self-contained directory instead.
- `--json` prints machine-readable output; everything else is for people.
"""


def _md(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _commands(group: click.Group, prefix: str, ctx: click.Context) -> list[tuple[str, click.Command, click.Context]]:
    out: list[tuple[str, click.Command, click.Context]] = []
    for name in sorted(group.list_commands(ctx)):
        cmd = group.get_command(ctx, name)
        if cmd is None or cmd.hidden:
            continue
        sub = click.Context(cmd, info_name=name, parent=ctx)
        out.append((f"{prefix} {name}", cmd, sub))
        if isinstance(cmd, click.Group):
            out.extend(_commands(cmd, f"{prefix} {name}", sub))
    return out


def _option_rows(cmd: click.Command, ctx: click.Context) -> list[str]:
    rows: list[str] = []
    for p in cmd.get_params(ctx):
        if isinstance(p, click.Option):
            record = p.get_help_record(ctx)
            if record is None or "--help" in p.opts:
                continue
            flags, text = record
            rows.append(f"| `{flags}` | {_md(text).replace('|', '&#124;')} |")
    return rows


def render() -> str:
    root_ctx = click.Context(main, info_name="chairlift")
    parts = [HEADER]
    entries = [("chairlift", main, root_ctx), *_commands(main, "chairlift", root_ctx)]
    parts.append("## Commands\n")
    parts.extend(f"- [`{path}`](#{path.replace(' ', '-')})" for path, _, _ in entries[1:])
    parts.append("")
    for path, cmd, ctx in entries:
        usage = " ".join(cmd.collect_usage_pieces(ctx))
        parts.append(f'<a id="{path.replace(" ", "-")}"></a>\n\n## `{path}`\n')
        parts.append(f"```\n{path} {usage}\n```\n")
        if cmd.help:
            parts.append(_md(inspect.cleandoc(click.unstyle(cmd.help))) + "\n")  # 3.13+ dedents at compile time
        rows = _option_rows(cmd, ctx)
        if rows:
            parts.append("| option | meaning |\n|---|---|")
            parts.extend(rows)
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")
