"""Generate docs/guide/10-api.md from `chairlift.__all__`; tests/docs/test_docs.py fails when it is stale.

uv run python scripts/gen_api_reference.py          # rewrite the page
"""

from __future__ import annotations

import inspect
from pathlib import Path

import chairlift

OUT = Path(__file__).parents[1] / "docs" / "guide" / "10-api.md"
HEADER = """# 10. API reference

Generated from `chairlift.__all__` by `scripts/gen_api_reference.py`; do not edit by hand. Every name below imports
from the top level (`from chairlift import Study`) and is part of the public API: it changes only with a version
bump and a CHANGELOG entry. Anything else is internal, even when importable.
"""


def _sig(obj: object) -> str:
    try:
        sig = str(inspect.signature(obj))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    return sig if len(sig) <= 100 else "(…)"


def render() -> str:
    parts = [HEADER]
    group = ""
    for name, module in chairlift._EXPORTS.items():  # pyright: ignore[reportPrivateUsage]
        section = module.rsplit(".", 1)[0].removeprefix("chairlift.")
        if section != group:
            group = section
            parts.append(f"\n## `chairlift.{section}`\n")
        obj = getattr(chairlift, name)
        kind = "class" if inspect.isclass(obj) else "function"
        doc = (inspect.getdoc(obj) or "").split("\n\n")[0].replace("\n", " ")
        parts.append(f"- **`{name}`** ({kind}, `{module}`) `{_sig(obj)}`  \n  {doc}")
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")
